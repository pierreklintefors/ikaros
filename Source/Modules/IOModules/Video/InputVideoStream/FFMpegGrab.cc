//
//    FFMpegGrab.cc        This file is a part of the IKAROS project
//
//    Copyright (C) 2018 Birger Johansson
//
//    This program is free software; you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation; either version 2 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program; if not, write to the Free Software
//    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//    See http://www.ikaros-project.org/ for more information.
//

#include "FFMpegGrab.h"
#include "ikaros.h"

#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

using namespace ikaros;

namespace
{
    int
    decode_packet(AVCodecContext *avctx, AVFrame *frame, int *got_frame, AVPacket *pkt)
    {
        int ret;
        *got_frame = 0;
        if (pkt)
        {
            ret = avcodec_send_packet(avctx, pkt);
            if (ret != 0)
                return ret == AVERROR_EOF ? 0 : ret;
        }

        ret = avcodec_receive_frame(avctx, frame);
        if (ret != 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF)
            return ret;

        if (ret == 0)
            *got_frame = 1;

        return 0;
    }

}

struct FFMpegGrab::Impl
{
    AVFormatContext *input_format_context = nullptr;
    int videoStreamId = -1;
    const AVCodec *input_codec = nullptr;
    AVCodecContext *avctx = nullptr;
    AVFrame *inputFrame = nullptr;
    AVFrame *rgbFrame = nullptr;
    AVFrame *outputFrame = nullptr;
    int rgbWidth = 0;
    int rgbHeight = 0;
    AVPacket packet{};

    int outputSizeX = 0;
    int outputSizeY = 0;

    // Crop rectangle in source-image coordinates. A width/height <= 0 means
    // use the full source dimension for that axis.
    int cropX = 0;
    int cropY = 0;
    int cropWidth = 0;
    int cropHeight = 0;

    // Two-stage conversion is deliberate:
    //   decoded pixel format -> full-resolution RGB24 -> cropped/resized RGB24
    // This avoids error-prone pointer arithmetic on subsampled YUV planes.
    SwsContext *decode_to_rgb_ctx = nullptr;
    SwsContext *crop_resize_ctx = nullptr;

    std::vector<uint8_t> frame;

    bool uv4l = false;
    std::string url;
    std::string lastError;
    bool printInfo = false;
    bool synchronized = false;
    bool freshData = false;
    bool shutdown = false;
    float FPS = 0.0f;
    Timer timer;

    std::mutex state_mutex;
    std::condition_variable state_cv;
    std::thread worker;
};

FFMpegGrab::FFMpegGrab() : impl_(std::make_unique<Impl>())
{
    avformat_network_init();
    av_log_set_level(AV_LOG_INFO);
}

const std::string &
FFMpegGrab::LastError() const
{
    return impl_->lastError;
}

FFMpegGrab::~FFMpegGrab()
{
    {
        std::lock_guard<std::mutex> lock(impl_->state_mutex);
        impl_->shutdown = true;
    }
    impl_->state_cv.notify_all();

    if (impl_->worker.joinable())
        impl_->worker.join();

    resetDecoderState();
}

void FFMpegGrab::resetDecoderState()
{
    {
        std::lock_guard<std::mutex> lock(impl_->state_mutex);
        impl_->shutdown = true;
    }
    impl_->state_cv.notify_all();

    if (impl_->worker.joinable())
        impl_->worker.join();

    if (impl_->decode_to_rgb_ctx)
    {
        sws_freeContext(impl_->decode_to_rgb_ctx);
        impl_->decode_to_rgb_ctx = nullptr;
    }

    if (impl_->crop_resize_ctx)
    {
        sws_freeContext(impl_->crop_resize_ctx);
        impl_->crop_resize_ctx = nullptr;
    }

    if (impl_->rgbFrame)
    {
        av_freep(&impl_->rgbFrame->data[0]);
        av_frame_free(&impl_->rgbFrame);
    }

    if (impl_->outputFrame)
    {
        av_freep(&impl_->outputFrame->data[0]);
        av_frame_free(&impl_->outputFrame);
    }

    if (impl_->inputFrame)
        av_frame_free(&impl_->inputFrame);

    if (impl_->avctx)
        avcodec_free_context(&impl_->avctx);

    if (impl_->input_format_context)
        avformat_close_input(&impl_->input_format_context);

    impl_->videoStreamId = -1;
    impl_->input_codec = nullptr;
    impl_->rgbWidth = 0;
    impl_->rgbHeight = 0;
    impl_->frame.clear();
    impl_->freshData = false;
    impl_->shutdown = false;
}

void FFMpegGrab::SetOutputSize(int width, int height)
{
    impl_->outputSizeX = width;
    impl_->outputSizeY = height;
}

void FFMpegGrab::SetCrop(int x, int y, int width, int height)
{
    impl_->cropX = x;
    impl_->cropY = y;
    impl_->cropWidth = width;
    impl_->cropHeight = height;
}

void FFMpegGrab::SetUrl(const char *stream_url)
{
    impl_->url = stream_url ? stream_url : "";
}

void FFMpegGrab::SetPrintInfo(bool enabled)
{
    impl_->printInfo = enabled;
}

void FFMpegGrab::SetUv4l(bool enabled)
{
    impl_->uv4l = enabled;
}

void FFMpegGrab::SetSynchronized(bool enabled)
{
    impl_->synchronized = enabled;
}

bool FFMpegGrab::ReadFrame(uint8_t *destination, std::size_t bytes, bool wait_for_new_frame)
{
    std::unique_lock<std::mutex> lock(impl_->state_mutex);

    if (wait_for_new_frame)
    {
        impl_->state_cv.wait(lock, [this]
                             { return impl_->shutdown || impl_->freshData; });
    }

    if (!impl_->freshData || impl_->frame.empty())
        return false;

    if (bytes != impl_->frame.size())
        throw std::runtime_error("FFMpegGrab::ReadFrame received a destination buffer of unexpected size.");

    std::memcpy(destination, impl_->frame.data(), bytes);
    impl_->freshData = false;
    lock.unlock();
    impl_->state_cv.notify_all();
    return true;
}

bool FFMpegGrab::Init()
{
    impl_->lastError.clear();
    if (impl_->printInfo)
        std::cout << "FFMpegGrab Init()" << std::endl;

    if (impl_->outputSizeX <= 0 || impl_->outputSizeY <= 0)
    {
        std::cout << "FFMpegGrab: Output size must be positive" << std::endl;
        return false;
    }

    if (impl_->url.empty())
    {
        std::cout << "FFMpegGrab: Stream url is empty" << std::endl;
        return false;
    }

    resetDecoderState();

    const AVInputFormat *file_iformat = nullptr;
    if (impl_->uv4l)
        file_iformat = av_find_input_format("h264");

    AVDictionary *options = nullptr;
    if (impl_->uv4l)
        av_dict_set(&options, "probesize", "192", 0);

    if (avformat_open_input(&impl_->input_format_context, impl_->url.c_str(), file_iformat, &options) != 0)
    {
        impl_->lastError = "Could not open \"" + impl_->url + "\".";
        av_dict_free(&options);
        return false;
    }
    av_dict_free(&options);

    if (av_find_best_stream(impl_->input_format_context, AVMEDIA_TYPE_VIDEO, 0, 0, nullptr, 0) < 0)
    {
        impl_->lastError = "Could not find stream information.";
        resetDecoderState();
        return false;
    }

    if (impl_->printInfo)
        av_dump_format(impl_->input_format_context, 0, impl_->url.c_str(), 0);

    av_log_set_level(AV_LOG_ERROR);

    impl_->videoStreamId = -1;
    for (unsigned int i = 0; i < impl_->input_format_context->nb_streams; ++i)
    {
        if (impl_->input_format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            impl_->videoStreamId = static_cast<int>(i);
            break;
        }
    }
    if (impl_->videoStreamId == -1)
    {
        impl_->lastError = "Could not find a video stream.";
        resetDecoderState();
        return false;
    }

    if (impl_->uv4l)
    {
        impl_->input_codec = avcodec_find_decoder_by_name("h264_mmal");
        if (impl_->input_codec)
            std::cout << "FFMpegGrab: Using HW decoding (mmal)" << std::endl;
        else
            impl_->input_codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    }
    else
    {
        impl_->input_codec = avcodec_find_decoder(
            impl_->input_format_context->streams[impl_->videoStreamId]->codecpar->codec_id);
    }

    if (impl_->input_codec == nullptr)
    {
        impl_->lastError = "The video codec is unsupported.";
        resetDecoderState();
        return false;
    }

    impl_->avctx = avcodec_alloc_context3(impl_->input_codec);
    if (!impl_->avctx)
    {
        impl_->lastError = "Could not allocate a decoding context.";
        return false;
    }

    int error = avcodec_parameters_to_context(
        impl_->avctx,
        impl_->input_format_context->streams[impl_->videoStreamId]->codecpar);
    if (error < 0)
    {
        impl_->lastError = "Could not copy codec parameters into the decoding context.";
        resetDecoderState();
        return false;
    }

    if (avcodec_open2(impl_->avctx, impl_->input_codec, nullptr) < 0)
    {
        impl_->lastError = "Could not open the video codec.";
        resetDecoderState();
        return false;
    }

    impl_->inputFrame = av_frame_alloc();
    if (impl_->inputFrame == nullptr)
    {
        impl_->lastError = "Could not allocate the input video frame.";
        resetDecoderState();
        return false;
    }

    // Allocate only the AVFrame structure here. For some network streams
    // (including MJPEG), the codec context does not know width/height until
    // the first frame has actually been decoded. The RGB image buffer is
    // therefore allocated lazily in loop() from inputFrame->width/height.
    impl_->rgbFrame = av_frame_alloc();
    if (impl_->rgbFrame == nullptr)
    {
        impl_->lastError = "Could not allocate the intermediate RGB video frame.";
        resetDecoderState();
        return false;
    }

    impl_->outputFrame = av_frame_alloc();
    if (impl_->outputFrame == nullptr)
    {
        impl_->lastError = "Could not allocate the output video frame.";
        resetDecoderState();
        return false;
    }

    impl_->outputFrame->format = AV_PIX_FMT_RGB24;
    impl_->outputFrame->width = impl_->outputSizeX;
    impl_->outputFrame->height = impl_->outputSizeY;

    if (av_image_alloc(impl_->outputFrame->data, impl_->outputFrame->linesize,
                       impl_->outputFrame->width, impl_->outputFrame->height,
                       AV_PIX_FMT_RGB24, 32) < 0)
    {
        impl_->lastError = "Could not allocate the output image buffer.";
        resetDecoderState();
        return false;
    }

    impl_->frame.resize(static_cast<std::size_t>(impl_->outputSizeX) *
                        static_cast<std::size_t>(impl_->outputSizeY) * 3);

    impl_->worker = std::thread(&FFMpegGrab::loop, this);
    return true;
}

void FFMpegGrab::loop()
{
#ifdef FFMPEG_TIMER
    Timer timerSub;
#endif

    int gotFrame = 0;
    while (true)
    {
        {
            std::lock_guard<std::mutex> lock(impl_->state_mutex);
            if (impl_->shutdown)
                break;
        }

        if (av_read_frame(impl_->input_format_context, &impl_->packet) >= 0)
        {
            if (impl_->packet.stream_index == impl_->videoStreamId)
            {
#ifdef FFMPEG_TIMER
                timerSub.Restart();
#endif
                decode_packet(impl_->avctx, impl_->inputFrame, &gotFrame, &impl_->packet);
                if (gotFrame)
                {
#ifdef FFMPEG_TIMER
                    std::cout << "FFMpegGrab (timing): Decoded Time\t"
                              << round(timerSub.GetTime() / 1000) << " ms" << std::endl;
                    timerSub.Restart();
#endif

                    const int source_width = impl_->inputFrame->width;
                    const int source_height = impl_->inputFrame->height;
                    const AVPixelFormat source_format =
                        static_cast<AVPixelFormat>(impl_->inputFrame->format);

                    if (source_width <= 0 || source_height <= 0 || source_format == AV_PIX_FMT_NONE)
                    {
                        impl_->lastError = "Decoded video frame has invalid dimensions or pixel format.";
                        av_packet_unref(&impl_->packet);
                        break;
                    }

                    // Allocate (or reallocate) the full-resolution RGB buffer only
                    // after FFmpeg has decoded a frame and therefore knows its size.
                    if (impl_->rgbWidth != source_width || impl_->rgbHeight != source_height ||
                        impl_->rgbFrame->data[0] == nullptr)
                    {
                        if (impl_->rgbFrame->data[0])
                            av_freep(&impl_->rgbFrame->data[0]);

                        impl_->rgbFrame->format = AV_PIX_FMT_RGB24;
                        impl_->rgbFrame->width = source_width;
                        impl_->rgbFrame->height = source_height;

                        if (av_image_alloc(impl_->rgbFrame->data, impl_->rgbFrame->linesize,
                                           source_width, source_height,
                                           AV_PIX_FMT_RGB24, 32) < 0)
                        {
                            impl_->lastError = "Could not allocate the intermediate RGB image buffer.";
                            av_packet_unref(&impl_->packet);
                            break;
                        }

                        impl_->rgbWidth = source_width;
                        impl_->rgbHeight = source_height;
                    }

                    // Stage 1: convert the complete decoded frame to RGB24 at
                    // native resolution. This makes the subsequent crop independent
                    // of YUV chroma subsampling.
                    impl_->decode_to_rgb_ctx = sws_getCachedContext(
                        impl_->decode_to_rgb_ctx,
                        source_width, source_height, source_format,
                        source_width, source_height, AV_PIX_FMT_RGB24,
                        SWS_BILINEAR, nullptr, nullptr, nullptr);

                    if (impl_->decode_to_rgb_ctx == nullptr)
                    {
                        impl_->lastError = "Could not create RGB conversion context.";
                        av_packet_unref(&impl_->packet);
                        break;
                    }

                    sws_scale(impl_->decode_to_rgb_ctx,
                              (const uint8_t *const *)impl_->inputFrame->data,
                              impl_->inputFrame->linesize, 0, source_height,
                              impl_->rgbFrame->data, impl_->rgbFrame->linesize);

                    // Resolve the configured crop rectangle. Width/height <= 0
                    // means "use the remaining/full source dimension".
                    const int crop_x = impl_->cropX;
                    const int crop_y = impl_->cropY;
                    const int crop_w = impl_->cropWidth > 0 ? impl_->cropWidth : source_width - crop_x;
                    const int crop_h = impl_->cropHeight > 0 ? impl_->cropHeight : source_height - crop_y;

                    if (crop_x < 0 || crop_y < 0 || crop_w <= 0 || crop_h <= 0 ||
                        crop_x + crop_w > source_width ||
                        crop_y + crop_h > source_height)
                    {
                        impl_->lastError = "Crop region is outside the decoded video frame.";
                        av_packet_unref(&impl_->packet);
                        break;
                    }

                    // Point at the upper-left RGB pixel of the crop. RGB24 has
                    // three bytes per pixel; the original line stride is retained.
                    const uint8_t *cropped_data[4] = {
                        impl_->rgbFrame->data[0] + static_cast<std::ptrdiff_t>(crop_y) * impl_->rgbFrame->linesize[0] + static_cast<std::ptrdiff_t>(crop_x) * 3,
                        nullptr, nullptr, nullptr};
                    int cropped_linesize[4] = {
                        impl_->rgbFrame->linesize[0], 0, 0, 0};

                    // Stage 2: resize only the cropped RGB rectangle to the output
                    // dimensions requested by InputVideoStream.
                    impl_->crop_resize_ctx = sws_getCachedContext(
                        impl_->crop_resize_ctx,
                        crop_w, crop_h, AV_PIX_FMT_RGB24,
                        impl_->outputSizeX, impl_->outputSizeY, AV_PIX_FMT_RGB24,
                        SWS_BICUBIC, nullptr, nullptr, nullptr);

                    if (impl_->crop_resize_ctx == nullptr)
                    {
                        av_packet_unref(&impl_->packet);
                        break;
                    }

                    sws_scale(impl_->crop_resize_ctx,
                              cropped_data, cropped_linesize,
                              0, crop_h,
                              impl_->outputFrame->data, impl_->outputFrame->linesize);

                    {
                        std::unique_lock<std::mutex> lock(impl_->state_mutex);
                        for (int y = 0; y < impl_->outputSizeY; ++y)
                        {
                            std::memcpy(impl_->frame.data() + static_cast<std::size_t>(y) * impl_->outputSizeX * 3,
                                        impl_->outputFrame->data[0] + y * impl_->outputFrame->linesize[0],
                                        static_cast<std::size_t>(impl_->outputSizeX) * 3);
                        }
                        impl_->freshData = true;
                        lock.unlock();
                        impl_->state_cv.notify_all();

                        if (impl_->synchronized)
                        {
                            lock.lock();
                            impl_->state_cv.wait(lock, [this]
                                                 { return impl_->shutdown || !impl_->freshData; });
                            if (impl_->shutdown)
                                break;
                        }
                    }

                    gotFrame = 0;
#ifdef FFMPEG_FPS
                    impl_->FPS = impl_->FPS + (1.0 / impl_->timer.GetTime() - impl_->FPS) * 0.02f;
                    impl_->timer.Restart();
                    std::cout << "FFMpegGrab " << impl_->url << ": FPS " << impl_->FPS << std::endl;
#endif
                }
            }
            av_packet_unref(&impl_->packet);
        }
    }
    impl_->state_cv.notify_all();
}

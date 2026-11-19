#include "ikaros.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <Vision/Vision.h>

using namespace ikaros;

namespace
{
    unsigned char
    ToByte(float v)
    {
        if(v <= 1.0f)
            v *= 255.0f;
        return static_cast<unsigned char>(std::clamp(std::lround(v), 0l, 255l));
    }


    CGImageRef
    CreateImage(const matrix & input, std::vector<unsigned char> & pixels)
    {
        const int width = input.size_x();
        const int height = input.size_y();
        if(width <= 0 || height <= 0)
            return nullptr;

        pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4);
        const bool color = input.rank() == 3 && input.shape(0) >= 3;
        for(int y = 0; y < height; ++y)
            for(int x = 0; x < width; ++x)
            {
                const std::size_t offset = (static_cast<std::size_t>(y) * width + x) * 4;
                pixels[offset + 0] = ToByte(color ? input(0, y, x) : input(y, x));
                pixels[offset + 1] = ToByte(color ? input(1, y, x) : input(y, x));
                pixels[offset + 2] = ToByte(color ? input(2, y, x) : input(y, x));
                pixels[offset + 3] = 255;
            }

        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        CGDataProviderRef provider = CGDataProviderCreateWithData(nullptr, pixels.data(), pixels.size(), nullptr);
        CGImageRef image = CGImageCreate(width, height, 8, 32, static_cast<std::size_t>(width) * 4, colorSpace,
                                         static_cast<CGBitmapInfo>(static_cast<uint32_t>(kCGImageAlphaLast) |
                                                                   static_cast<uint32_t>(kCGBitmapByteOrderDefault)),
                                         provider, nullptr, false, kCGRenderingIntentDefault);
        CGDataProviderRelease(provider);
        CGColorSpaceRelease(colorSpace);
        return image;
    }


    struct Joint
    {
        bool found = false;
        float x = 0;    // pixels from the left
        float y = 0;    // pixels from the bottom, as Vision gives them
    };


    float
    Distance(const Joint & a, const Joint & b)
    {
        return std::hypot(a.x - b.x, a.y - b.y);
    }
}


class AppleVisionHandDetector : public Module
{
    parameter point;
    parameter maxHands;
    parameter minConfidence;
    parameter extensionRatio;
    parameter holdTime;
    parameter downsample;
    parameter interval;
    parameter minWidth;

    matrix input;
    matrix enable;
    matrix target;
    matrix valid;
    matrix open;
    matrix pointing;
    matrix count;

    matrix downsampledA;
    matrix downsampledB;
    std::vector<unsigned char> pixels;
    float sinceFound = 1e9f;
    int sinceDetection = 1 << 20;

    VNDetectHumanHandPoseRequest * request = nil;
    VNSequenceRequestHandler * handler = nil;
    NSArray<VNRequest *> * requests = nil;

    Joint
    Find(VNHumanHandPoseObservation * hand, VNHumanHandPoseObservationJointName name, float width, float height) const
    {
        Joint joint;
        NSError * error = nil;
        VNRecognizedPoint * p = [hand recognizedPointForJointName:name error:&error];
        if(p == nil || p.confidence < minConfidence.as_float())
            return joint;
        joint.found = true;
        joint.x = static_cast<float>(p.location.x) * width;
        joint.y = static_cast<float>(p.location.y) * height;
        return joint;
    }


    const matrix &
    DetectionImage()
    {
        const matrix * image = &input;
        for(int i = 0; i < downsample.as_int(); ++i)
        {
            if(image->size_x() < 2 || image->size_y() < 2 || image->size_x() % 2 != 0 || image->size_y() % 2 != 0)
                break;
            matrix & output = i % 2 == 0 ? downsampledA : downsampledB;
            if(image->rank() == 2)
            {
                if(output.rank() != 2 || output.size_y() != image->size_y() / 2 || output.size_x() != image->size_x() / 2)
                    output = matrix(image->size_y() / 2, image->size_x() / 2);
                output.downsample(*image);
            }
            else
            {
                if(output.rank() != 3 || output.size_z() != image->size_z() || output.size_y() != image->size_y() / 2 ||
                   output.size_x() != image->size_x() / 2)
                    output = matrix(image->size_z(), image->size_y() / 2, image->size_x() / 2);
                for(int c = 0; c < image->size_z(); ++c)
                    output[c].downsample((*image)[c]);
            }
            image = &output;
        }
        return *image;
    }

    void
    Clear()
    {
        valid(0) = 0;
        target.set(0);
        open(0) = 0;
        pointing(0) = 0;
    }

public:
    ~AppleVisionHandDetector() override
    {
        @autoreleasepool
        {
            [requests release];
            [handler release];
            [request release];
        }
    }


    void
    Init() override
    {
        Bind(point, "point");
        Bind(maxHands, "max_hands");
        Bind(minConfidence, "min_confidence");
        Bind(extensionRatio, "extension_ratio");
        Bind(holdTime, "hold_time");
        Bind(downsample, "downsample");
        Bind(interval, "interval");
        Bind(minWidth, "min_width");
        Bind(input, "INPUT");
        Bind(enable, "ENABLE");
        Bind(target, "TARGET");
        Bind(valid, "VALID");
        Bind(open, "OPEN");
        Bind(pointing, "POINTING");
        Bind(count, "COUNT");

        @autoreleasepool
        {
            request = [[VNDetectHumanHandPoseRequest alloc] init];
            request.maximumHandCount = static_cast<NSUInteger>(std::max(1, maxHands.as_int()));
            handler = [[VNSequenceRequestHandler alloc] init];
            requests = [[NSArray alloc] initWithObjects:request, nil];
        }
    }


    void
    Tick() override
    {
        sinceFound += static_cast<float>(GetTickDuration());

        // Detection is the expensive part, so it can be switched off or run on every interval-th tick;
        // in between, the last result is kept.
        if(enable.connected() && enable(0) <= 0.5f)
        {
            sinceFound = 1e9f;
            sinceDetection = 1 << 20;
            count(0) = 0;
            Clear();
            return;
        }
        if(++sinceDetection < std::max(1, interval.as_int()))
            return;
        sinceDetection = 0;
        count(0) = 0;

        @autoreleasepool
        {
            const matrix & image = DetectionImage();
            const float width = static_cast<float>(image.size_x());
            const float height = static_cast<float>(image.size_y());
            CGImageRef cgImage = CreateImage(image, pixels);
            if(cgImage == nullptr)
                return;
            NSError * error = nil;
            const BOOL ok = [handler performRequests:requests onCGImage:cgImage error:&error];
            CGImageRelease(cgImage);
            if(!ok)
            {
                NSString * message = error ? [error localizedDescription] : @"unknown Vision error";
                Notify(msg_warning, std::string("AppleVisionHandDetector failed: ") + [message UTF8String]);
                return;
            }

            NSArray<VNHumanHandPoseObservation *> * hands = request.results;
            count(0) = static_cast<float>([hands count]);

            // The largest hand is usually the nearest, the one that is shown to the robot.
            float bestWidth = 0;
            for(VNHumanHandPoseObservation * hand in hands)
            {
                const std::array<VNHumanHandPoseObservationJointName, 4> tips = {
                    VNHumanHandPoseObservationJointNameIndexTip, VNHumanHandPoseObservationJointNameMiddleTip,
                    VNHumanHandPoseObservationJointNameRingTip, VNHumanHandPoseObservationJointNameLittleTip};
                const std::array<VNHumanHandPoseObservationJointName, 4> middles = {
                    VNHumanHandPoseObservationJointNameIndexPIP, VNHumanHandPoseObservationJointNameMiddlePIP,
                    VNHumanHandPoseObservationJointNameRingPIP, VNHumanHandPoseObservationJointNameLittlePIP};
                const std::array<VNHumanHandPoseObservationJointName, 4> knuckles = {
                    VNHumanHandPoseObservationJointNameIndexMCP, VNHumanHandPoseObservationJointNameMiddleMCP,
                    VNHumanHandPoseObservationJointNameRingMCP, VNHumanHandPoseObservationJointNameLittleMCP};

                const Joint wrist = Find(hand, VNHumanHandPoseObservationJointNameWrist, width, height);

                // Hand width: the horizontal extent of all joints that were found.
                float left = width;
                float right = 0;
                NSDictionary<VNHumanHandPoseObservationJointName, VNRecognizedPoint *> * all =
                    [hand recognizedPointsForJointsGroupName:VNHumanHandPoseObservationJointsGroupNameAll error:nil];
                for(VNHumanHandPoseObservationJointName name in all)
                {
                    VNRecognizedPoint * p = all[name];
                    if(p.confidence < minConfidence.as_float())
                        continue;
                    left = std::min(left, static_cast<float>(p.location.x) * width);
                    right = std::max(right, static_cast<float>(p.location.x) * width);
                }
                const float handWidth = right > left ? 2.0f * (right - left) / width : 0.0f;
                if(handWidth <= bestWidth || handWidth < minWidth.as_float())
                    continue;

                std::array<bool, 4> extended = {false, false, false, false};
                Joint palm;
                int palmJoints = 0;
                if(wrist.found)
                {
                    palm.x = wrist.x;
                    palm.y = wrist.y;
                    palmJoints = 1;
                }
                for(int f = 0; f < 4; ++f)
                {
                    const Joint tip = Find(hand, tips[f], width, height);
                    const Joint middle = Find(hand, middles[f], width, height);
                    const Joint knuckle = Find(hand, knuckles[f], width, height);
                    if(wrist.found && tip.found && middle.found)
                        extended[f] = Distance(wrist, tip) > extensionRatio.as_float() * Distance(wrist, middle);
                    if(knuckle.found)
                    {
                        palm.x += knuckle.x;
                        palm.y += knuckle.y;
                        ++palmJoints;
                    }
                }

                Joint chosen;
                if(point.as_int() == 0)
                    chosen = Find(hand, VNHumanHandPoseObservationJointNameIndexTip, width, height);
                else if(palmJoints > 0)
                {
                    chosen.found = true;
                    chosen.x = palm.x / palmJoints;
                    chosen.y = palm.y / palmJoints;
                }
                if(!chosen.found)
                    continue;

                bestWidth = handWidth;
                target(0) = 2.0f * chosen.x / width - 1.0f;
                target(1) = 1.0f - 2.0f * chosen.y / height;
                target(2) = handWidth;
                open(0) = static_cast<float>(extended[0] + extended[1] + extended[2] + extended[3]) / 4.0f;
                pointing(0) = extended[0] && !extended[1] && !extended[2] && !extended[3] ? 1.0f : 0.0f;
            }

            if(bestWidth > 0)
                sinceFound = 0;
        }

        valid(0) = sinceFound <= holdTime.as_float() ? 1.0f : 0.0f;
        if(valid(0) == 0.0f)
            Clear();
    }
};

INSTALL_CLASS(AppleVisionHandDetector)

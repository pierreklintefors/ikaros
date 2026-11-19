#include "ikaros.h"

#include <chrono>
#include <regex>
#include <string>
#include <vector>

using namespace ikaros;

// The control ids of the raspicam driver on the UV4L settings page.
static constexpr const char * qualityControl = "134217739";
static constexpr const char * frameRateControl = "134217741";

class UV4LSettings : public Module
{
    parameter url;
    parameter width;
    parameter height;
    parameter frameRate;
    parameter quality;
    parameter extra;
    parameter timeout;
    parameter required;

    matrix applied;

    bool skipped = false;
    bool ok = false;
    std::string problem;

    // Attribute value as given in the model, with @ references resolved; the parameters are not yet
    // bound while the constructor runs.
    std::string
    Attribute(const std::string & name, const std::string & fallback)
    {
        if(!info_.contains_non_null(name))
            return fallback;
        const std::string raw = info_.at(name).as_string();
        return raw.find('@') != std::string::npos ? ComputeValue(raw) : raw;
    }


    static std::string
    PanelValue(const std::string & page, const std::string & name)
    {
        const std::regex pattern("name=\"" + name + "\"[^>]*value=\"([^\"]*)\"|value=\"([^\"]*)\"[^>]*name=\"" + name + "\"");
        std::smatch match;
        if(!std::regex_search(page, match, pattern))
            return "";
        return match[1].matched ? match[1].str() : match[2].str();
    }


    bool
    Request(const std::string & host, int port, const std::string & path, double seconds, std::string & body)
    {
        Socket socket(std::chrono::milliseconds(static_cast<int>(seconds * 1000)));
        const std::string request = "GET " + path + " HTTP/1.1\r\nHost: " + host + "\r\nConnection: close\r\n\r\n";
        std::vector<char> buffer(512 * 1024);
        const int bytes = socket.Get(host.c_str(), port, request.c_str(), buffer.data(), static_cast<int>(buffer.size()));
        socket.Close();
        if(bytes <= 0)
            return false;
        body.assign(buffer.data(), bytes);
        return body.rfind("HTTP/1.", 0) == 0 && body.find(" 200 ") != std::string::npos;
    }


    // Sends the settings and checks them, while the model is built and before any stream is opened.
    void
    Apply()
    {
        const std::string address = Attribute("url", "");
        const std::smatch hostPort = [&]
        {
            std::smatch m;
            std::regex_search(address, m, std::regex("^http://([^/:]+)(?::(\\d+))?"));
            return m;
        }();
        if(hostPort.empty())
        {
            skipped = true;
            return;
        }
        const std::string host = hostPort[1].str();
        const int port = hostPort[2].matched ? std::stoi(hostPort[2].str()) : 80;
        const std::string w = Attribute("width", "640");
        const std::string h = Attribute("height", "480");
        const std::string fps = Attribute("frame_rate", "30");
        const std::string q = Attribute("quality", "70");
        const std::string more = Attribute("extra", "");
        const double seconds = std::stod(Attribute("timeout", "3"));

        std::string path = "/panel?width=" + w + "&height=" + h + "&" + qualityControl + "=" + q + "&" +
                           frameRateControl + "=" + fps + "&apply_changed=on";
        if(!more.empty())
            path += "&" + more;

        std::string page;
        if(!Request(host, port, path, seconds, page) || !Request(host, port, "/panel", seconds, page))
        {
            problem = "could not reach the camera server at " + host + ":" + std::to_string(port);
            return;
        }
        const std::string gotWidth = PanelValue(page, "width");
        const std::string gotHeight = PanelValue(page, "height");
        const std::string gotQuality = PanelValue(page, qualityControl);
        const std::string gotFrameRate = PanelValue(page, frameRateControl);
        ok = gotWidth == w && gotHeight == h && gotQuality == q && gotFrameRate == fps;
        if(!ok)
            problem = "the camera server at " + host + " reports " + gotWidth + " x " + gotHeight + ", " + gotFrameRate +
                      " frames/s, quality " + gotQuality + " instead of " + w + " x " + h + ", " + fps + ", " + q +
                      "; it cannot change the resolution while another client streams";
    }

public:
    UV4LSettings() : Module()
    {
        Apply();
    }


    void
    Init() override
    {
        Bind(url, "url");
        Bind(width, "width");
        Bind(height, "height");
        Bind(frameRate, "frame_rate");
        Bind(quality, "quality");
        Bind(extra, "extra");
        Bind(timeout, "timeout");
        Bind(required, "required");
        Bind(applied, "APPLIED");

        applied(0) = ok ? 1.0f : 0.0f;
        if(skipped || ok)
            return;
        if(required.as_bool())
            throw exception("UV4LSettings: " + problem + ".", path_);
        Notify(msg_warning, "UV4LSettings: " + problem + ".");
    }


    void
    Tick() override
    {
    }
};

INSTALL_CLASS(UV4LSettings)

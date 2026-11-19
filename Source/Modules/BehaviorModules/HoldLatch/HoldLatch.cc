#include <algorithm>

#include "ikaros.h"

using namespace ikaros;

class HoldLatch : public Module
{
    static constexpr float epsilon = 1e-5f;

    parameter holdTime;
    parameter releaseTime;
    parameter threshold;

    matrix input;
    matrix present;
    matrix active;
    matrix start;
    matrix end;
    matrix progress;

    float held = 0;
    float absent = 0;

public:
    void
    Init() override
    {
        Bind(holdTime, "hold_time");
        Bind(releaseTime, "release_time");
        Bind(threshold, "threshold");
        Bind(input, "INPUT");
        Bind(present, "PRESENT");
        Bind(active, "ACTIVE");
        Bind(start, "START");
        Bind(end, "END");
        Bind(progress, "PROGRESS");
    }


    void
    Tick() override
    {
        const float dt = static_cast<float>(GetTickDuration());
        const float level = threshold.as_float();
        start(0) = 0;
        end(0) = 0;

        if(active(0) < 0.5f)
        {
            held = input(0) > level ? held + dt : 0.0f;
            const float needed = holdTime.as_float();
            progress(0) = needed > 0 ? std::min(held / needed, 1.0f) : (input(0) > level ? 1.0f : 0.0f);
            if(input(0) > level && held >= needed - epsilon)
            {
                active(0) = 1;
                start(0) = 1;
                absent = 0;
            }
            return;
        }

        const float presence = present.connected() ? present(0) : input(0);
        absent = presence > level ? 0.0f : absent + dt;
        if(absent >= releaseTime.as_float() - epsilon)
        {
            active(0) = 0;
            end(0) = 1;
            held = 0;
            progress(0) = 0;
        }
    }
};

INSTALL_CLASS(HoldLatch)

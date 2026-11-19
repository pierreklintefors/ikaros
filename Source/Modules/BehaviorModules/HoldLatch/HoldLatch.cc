#include <algorithm>

#include "ikaros.h"

using namespace ikaros;

class HoldLatch : public Module
{
    static constexpr float epsilon = 1e-5f;

    parameter holdTime;
    parameter releaseTime;
    matrix threshold;

    matrix input;
    matrix present;
    matrix active;
    matrix start;
    matrix end;
    matrix progress;

    float held = 0;
    float absent = 0;

    // INPUT counts as held when every element is above its threshold; a shorter threshold list
    // repeats its last value.
    bool
    Held() const
    {
        if(input.size() == 0)
            return false;
        for(int i = 0; i < input.size(); ++i)
        {
            const int t = std::min(i, threshold.size() - 1);
            if(input.data()[i] <= threshold.data()[t])
                return false;
        }
        return true;
    }

public:
    void
    Init() override
    {
        Bind(holdTime, "hold_time");
        Bind(releaseTime, "release_time");
        Bind(threshold, "threshold");
        Bind(input, "INPUT");
        Bind(present, "PRESENT");
        if(threshold.size() == 0)
            throw exception("HoldLatch: threshold needs at least one value.", path_);
        Bind(active, "ACTIVE");
        Bind(start, "START");
        Bind(end, "END");
        Bind(progress, "PROGRESS");
    }


    void
    Tick() override
    {
        const float dt = static_cast<float>(GetTickDuration());
        const bool holding = Held();
        start(0) = 0;
        end(0) = 0;

        if(active(0) < 0.5f)
        {
            held = holding ? held + dt : 0.0f;
            const float needed = holdTime.as_float();
            progress(0) = needed > 0 ? std::min(held / needed, 1.0f) : (holding ? 1.0f : 0.0f);
            if(holding && held >= needed - epsilon)
            {
                active(0) = 1;
                start(0) = 1;
                absent = 0;
            }
            return;
        }

        const bool isPresent = present.connected() ? present(0) > 0.5f : holding;
        absent = isPresent ? 0.0f : absent + dt;
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

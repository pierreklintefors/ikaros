#include "ikaros.h"

#include <cmath>
#include <deque>

using namespace ikaros;

class OscillationDetector : public Module
{
    parameter amplitude;
    parameter swings;
    parameter window;
    parameter refractory;
    parameter smoothing;

    matrix input;
    matrix valid;

    matrix detected;
    matrix swingCount;

    bool started = false;
    float value = 0;
    float low = 0;        // lowest value since the last turning point, before the first swing
    float high = 0;       // highest value since the last turning point, before the first swing
    float extreme = 0;    // turning point of the present swing
    int direction = 0;    // 1 while rising, -1 while falling, 0 before the first swing
    float time = 0;
    float sinceDetection = 0;
    std::deque<float> swingTimes;

    void
    Reset(float v)
    {
        low = high = extreme = v;
        direction = 0;
        swingTimes.clear();
    }


    void
    Swing(int newDirection)
    {
        direction = newDirection;
        extreme = value;
        swingTimes.push_back(time);
    }

public:
    void
    Init() override
    {
        Bind(amplitude, "amplitude");
        Bind(swings, "swings");
        Bind(window, "window");
        Bind(refractory, "refractory");
        Bind(smoothing, "smoothing");

        Bind(input, "INPUT");
        Bind(valid, "VALID");

        Bind(detected, "DETECTED");
        Bind(swingCount, "SWINGS");

        sinceDetection = refractory.as_float();
    }


    void
    Tick() override
    {
        const float dt = static_cast<float>(GetTickDuration());
        time += dt;
        sinceDetection += dt;
        detected(0) = 0;

        if(valid.connected() && valid(0) <= 0.5f)
        {
            started = false;
            swingTimes.clear();
            swingCount(0) = 0;
            return;
        }

        // Rate parameters are delivered per tick by the kernel.
        const float blend = smoothing.as_float() <= 0.0f ? 1.0f : 1.0f - std::exp(-smoothing.as_float());
        value = started ? value + blend * (input(0) - value) : input(0);
        if(!started)
        {
            started = true;
            Reset(value);
        }

        const float a = amplitude.as_float();
        if(direction == 0)
        {
            low = std::min(low, value);
            high = std::max(high, value);
            if(value - low >= a)
                Swing(1);
            else if(high - value >= a)
                Swing(-1);
        }
        else if(direction > 0)
        {
            if(value > extreme)
                extreme = value;
            else if(extreme - value >= a)
                Swing(-1);
        }
        else
        {
            if(value < extreme)
                extreme = value;
            else if(value - extreme >= a)
                Swing(1);
        }

        while(!swingTimes.empty() && time - swingTimes.front() > window.as_float() + 1e-5f)
            swingTimes.pop_front();
        swingCount(0) = static_cast<float>(swingTimes.size());

        if(static_cast<int>(swingTimes.size()) >= swings.as_int() && sinceDetection >= refractory.as_float())
        {
            detected(0) = 1;
            sinceDetection = 0;
            swingTimes.clear();
        }
    }
};

INSTALL_CLASS(OscillationDetector)

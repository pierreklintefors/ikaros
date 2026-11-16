#include "ikaros.h"

#include <algorithm>
#include <random>

using namespace ikaros;

class Blinker : public Module
{
    static constexpr float epsilon = 1e-5f;

    parameter minInterval;
    parameter maxInterval;
    parameter doubleBlinkProbability;
    parameter doubleBlinkGap;
    parameter closeTime;
    parameter closedTime;
    parameter openTime;
    parameter seed;

    matrix trigger;
    matrix inhibit;

    matrix lid;
    matrix blink;

    std::mt19937 randomGenerator;
    bool blinking = false;
    bool secondBlink = false;
    float blinkTime = 0; // seconds since the current blink started, including the present tick
    float untilNext = 0; // seconds until the next spontaneous blink
    float lastTrigger = 0;

    float
    Uniform(float low, float high)
    {
        if(high <= low)
            return low;
        return std::uniform_real_distribution<float>(low, high)(randomGenerator);
    }


    static float
    Smooth(float x)
    {
        x = std::clamp(x, 0.0f, 1.0f);
        return x * x * (3.0f - 2.0f * x);
    }


    void
    StartBlink(float dt)
    {
        blinking = true;
        blinkTime = dt;
        blink(0) = 1;
    }


    // Lid closure at time t into a blink: a fast close, a short hold, and a slower open.
    float
    Closure(float t) const
    {
        const float close = closeTime.as_float();
        const float closed = closedTime.as_float();
        const float open = openTime.as_float();
        if(t <= close + epsilon)
            return close > 0.0f ? Smooth(t / close) : 1.0f;
        if(t <= close + closed + epsilon)
            return 1.0f;
        return open > 0.0f ? 1.0f - Smooth((t - close - closed) / open) : 0.0f;
    }

public:
    void
    Init() override
    {
        Bind(minInterval, "min_interval");
        Bind(maxInterval, "max_interval");
        Bind(doubleBlinkProbability, "double_blink_probability");
        Bind(doubleBlinkGap, "double_blink_gap");
        Bind(closeTime, "close_time");
        Bind(closedTime, "closed_time");
        Bind(openTime, "open_time");
        Bind(seed, "seed");

        Bind(trigger, "TRIGGER");
        Bind(inhibit, "INHIBIT");

        Bind(lid, "LID");
        Bind(blink, "BLINK");

        if(seed.as_int() < 0)
            randomGenerator.seed(std::random_device{}());
        else
            randomGenerator.seed(static_cast<std::mt19937::result_type>(seed.as_int()));

        untilNext = Uniform(minInterval.as_float(), maxInterval.as_float());
    }


    void
    Tick() override
    {
        const float dt = static_cast<float>(GetTickDuration());
        blink(0) = 0;

        const float triggerValue = trigger.connected() ? trigger(0) : 0.0f;
        const bool triggered = triggerValue > 0.5f && lastTrigger <= 0.5f;
        lastTrigger = triggerValue;
        const bool inhibited = inhibit.connected() && inhibit(0) > 0.5f;

        if(blinking)
        {
            blinkTime += dt;
            const float total = closeTime.as_float() + closedTime.as_float() + openTime.as_float();
            if(blinkTime >= total - epsilon)
            {
                blinking = false;
                // A double blink follows the first one after a short gap; the second one is never doubled.
                if(!secondBlink && Uniform(0.0f, 1.0f) < doubleBlinkProbability.as_float())
                {
                    secondBlink = true;
                    untilNext = doubleBlinkGap.as_float();
                }
                else
                {
                    secondBlink = false;
                    untilNext = Uniform(minInterval.as_float(), maxInterval.as_float());
                }
            }
        }
        else
            untilNext -= dt;

        if(triggered && !blinking)
        {
            secondBlink = false;
            StartBlink(dt);
        }
        else if(!blinking && !inhibited && untilNext <= epsilon)
            StartBlink(dt);

        lid(0) = blinking ? Closure(blinkTime) : 0.0f;
    }
};

INSTALL_CLASS(Blinker)

#include "ikaros.h"

#include <algorithm>
#include <cmath>

using namespace ikaros;

class Arousal : public Module
{
    parameter idleLevel;
    parameter engagedLevel;
    matrix boosts;
    parameter driveGain;
    parameter decayRate;
    parameter responseRate;
    matrix low;
    matrix high;

    matrix engaged;
    matrix stimulus;
    matrix drive;

    matrix arousal;
    matrix mapped;

    matrix lastStimulus;
    float excitation = 0;
    float level = 0;

    // Rate parameters are delivered per tick by the kernel.
    static float
    Blend(const parameter & rate)
    {
        const float ratePerTick = rate.as_float();
        return ratePerTick <= 0.0f ? 1.0f : 1.0f - std::exp(-ratePerTick);
    }

public:
    void
    Init() override
    {
        Bind(idleLevel, "idle_level");
        Bind(engagedLevel, "engaged_level");
        Bind(boosts, "boosts");
        Bind(driveGain, "drive_gain");
        Bind(decayRate, "decay_rate");
        Bind(responseRate, "response_rate");
        Bind(low, "low");
        Bind(high, "high");

        Bind(engaged, "ENGAGED");
        Bind(stimulus, "STIMULUS");
        Bind(drive, "DRIVE");

        Bind(arousal, "AROUSAL");
        Bind(mapped, "MAPPED");

        if(high.size() != low.size())
            throw exception("Arousal: high must have as many values as low.", path_);
        if(boosts.size() < 1)
            throw exception("Arousal: boosts must contain at least one value.", path_);

        if(stimulus.connected())
        {
            lastStimulus.realloc(stimulus.size());
            lastStimulus.set(0.0f);
        }
        level = idleLevel.as_float();
    }


    void
    Tick() override
    {
        excitation *= 1.0f - Blend(decayRate);
        if(stimulus.connected())
            for(int i = 0; i < stimulus.size(); ++i)
            {
                const float value = stimulus.data()[i];
                if(value > 0.5f && lastStimulus(i) <= 0.5f)
                    excitation += boosts.data()[std::min(i, boosts.size() - 1)];
                lastStimulus(i) = value;
            }
        excitation = std::clamp(excitation, 0.0f, 1.0f);

        const bool isEngaged = engaged.connected() && engaged(0) > 0.5f;
        float target = isEngaged ? engagedLevel.as_float() : idleLevel.as_float();
        if(drive.connected())
            target += driveGain.as_float() * drive(0);
        target = std::clamp(target + excitation, 0.0f, 1.0f);

        level += Blend(responseRate) * (target - level);
        arousal(0) = level;
        for(int i = 0; i < mapped.size(); ++i)
            mapped(i) = low.data()[i] + level * (high.data()[i] - low.data()[i]);
    }
};

INSTALL_CLASS(Arousal)

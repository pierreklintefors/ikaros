#include "ikaros.h"

#include <algorithm>
#include <cmath>
#include <random>

using namespace ikaros;

class GlanceGenerator : public Module
{
    static constexpr float epsilon = 1e-5f;

    matrix amplitude;
    parameter minFraction;
    parameter minInterval;
    parameter maxInterval;
    parameter centerProbability;
    parameter returnTime;
    parameter rate;
    parameter seed;

    matrix inhibit;

    matrix output;
    matrix shift;

    matrix target;
    std::mt19937 randomGenerator;
    float untilNext = 0;
    float untilReturn = -1; // negative while no return is pending

    float
    Uniform(float low, float high)
    {
        if(high <= low)
            return low;
        return std::uniform_real_distribution<float>(low, high)(randomGenerator);
    }


    void
    Center()
    {
        target.set(0.0f);
        untilReturn = -1;
        shift(0) = 1;
    }


    void
    Glance()
    {
        untilNext = Uniform(minInterval.as_float(), maxInterval.as_float());
        if(Uniform(0.0f, 1.0f) < centerProbability.as_float())
        {
            Center();
            return;
        }
        const float lowest = std::clamp(minFraction.as_float(), 0.0f, 1.0f);
        for(int i = 0; i < target.size(); ++i)
        {
            const float size = Uniform(lowest, 1.0f) * amplitude.data()[i];
            target(i) = Uniform(0.0f, 1.0f) < 0.5f ? -size : size;
        }
        untilReturn = returnTime.as_float() > 0.0f ? returnTime.as_float() : -1.0f;
        shift(0) = 1;
    }

public:
    void
    Init() override
    {
        Bind(amplitude, "amplitude");
        Bind(minFraction, "min_fraction");
        Bind(minInterval, "min_interval");
        Bind(maxInterval, "max_interval");
        Bind(centerProbability, "center_probability");
        Bind(returnTime, "return_time");
        Bind(rate, "rate");
        Bind(seed, "seed");

        Bind(inhibit, "INHIBIT");

        Bind(output, "OUTPUT");
        Bind(shift, "SHIFT");

        if(output.size() != amplitude.size())
            throw exception("GlanceGenerator: OUTPUT must have one element per amplitude value.", path_);

        if(seed.as_int() < 0)
            randomGenerator.seed(std::random_device{}());
        else
            randomGenerator.seed(static_cast<std::mt19937::result_type>(seed.as_int()));

        target.realloc(amplitude.size());
        target.set(0.0f);
        untilNext = Uniform(minInterval.as_float(), maxInterval.as_float());
    }


    void
    Tick() override
    {
        const float dt = static_cast<float>(GetTickDuration());
        shift(0) = 0;

        if(inhibit.connected() && inhibit(0) > 0.5f)
        {
            target.set(0.0f);
            untilReturn = -1;
            untilNext = Uniform(minInterval.as_float(), maxInterval.as_float());
        }
        else
        {
            untilNext -= dt;
            if(untilReturn >= 0.0f)
                untilReturn -= dt;

            if(untilNext <= epsilon)
                Glance();
            else if(untilReturn >= 0.0f && untilReturn <= epsilon)
                Center();
        }

        // Rate parameters are delivered per tick by the kernel.
        const float ratePerTick = rate.as_float();
        const float blend = ratePerTick <= 0.0f ? 1.0f : 1.0f - std::exp(-ratePerTick);
        for(int i = 0; i < output.size(); ++i)
            output(i) += blend * (target(i) - output(i));
    }
};

INSTALL_CLASS(GlanceGenerator)

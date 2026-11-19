#include "ikaros.h"

#include <random>

using namespace ikaros;

class Boredom : public Module
{
    static constexpr float epsilon = 1e-5f;

    parameter minWait;
    parameter maxWait;
    parameter minDuration;
    parameter maxDuration;
    parameter arousalThreshold;
    parameter seed;

    matrix activity;
    matrix arousal;

    matrix bored;
    matrix quiet;

    std::mt19937 randomGenerator;
    bool isBored = false;
    float quietTime = 0;
    float wait = 0;
    float episode = 0;
    float episodeTime = 0;

    float
    Uniform(float low, float high)
    {
        if(high <= low)
            return low;
        return std::uniform_real_distribution<float>(low, high)(randomGenerator);
    }


    void
    Restart()
    {
        isBored = false;
        quietTime = 0;
        wait = Uniform(minWait.as_float(), maxWait.as_float());
    }

public:
    void
    Init() override
    {
        Bind(minWait, "min_wait");
        Bind(maxWait, "max_wait");
        Bind(minDuration, "min_duration");
        Bind(maxDuration, "max_duration");
        Bind(arousalThreshold, "arousal_threshold");
        Bind(seed, "seed");

        Bind(activity, "ACTIVITY");
        Bind(arousal, "AROUSAL");

        Bind(bored, "BORED");
        Bind(quiet, "QUIET");

        if(seed.as_int() < 0)
            randomGenerator.seed(std::random_device{}());
        else
            randomGenerator.seed(static_cast<std::mt19937::result_type>(seed.as_int()));
        Restart();
    }


    void
    Tick() override
    {
        const float dt = static_cast<float>(GetTickDuration());
        bool active = false;
        for(int i = 0; i < activity.size(); ++i)
            active = active || activity.data()[i] > 0.5f;
        if(arousal.connected() && arousal(0) >= arousalThreshold.as_float())
            active = true;

        if(active)
            Restart();
        else if(isBored)
        {
            episodeTime += dt;
            if(episodeTime >= episode - epsilon)
                Restart();
        }
        else
        {
            quietTime += dt;
            if(quietTime >= wait - epsilon)
            {
                isBored = true;
                episodeTime = 0;
                episode = Uniform(minDuration.as_float(), maxDuration.as_float());
            }
        }

        bored(0) = isBored ? 1.0f : 0.0f;
        quiet(0) = quietTime;
    }
};

INSTALL_CLASS(Boredom)

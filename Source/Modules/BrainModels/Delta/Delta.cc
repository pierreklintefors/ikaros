#include "ikaros.h"

#include <algorithm>

using namespace ikaros;

class Delta: public Module
{
    parameter alpha;
    parameter extinction;
    parameter inverse;
    parameter trials;
    parameter gap;

    matrix cs;
    matrix us;
    matrix cr;
    matrix weights;

    matrix trialCs;         // largest CS of the present presentation, so that short gaps do not end it
    bool inTrial = false;
    bool trialHadUs = false;
    float csAbsentTime = 0;
    float usAbsentTime = 0;

    void Init()
    {
        Bind(alpha, "alpha");
        Bind(extinction, "extinction");
        Bind(inverse, "inverse");
        Bind(trials, "trials");
        Bind(gap, "gap");

        Bind(cs, "CS");
        Bind(us, "US");
        Bind(cr, "CR");
        Bind(weights, "WEIGHTS");

        if (cs.connected())
        {
            trialCs.realloc(cs.size());
            trialCs.set(0.0f);
        }
        usAbsentTime = gap.as_float();
    }


    // Moves the weights of the active stimulus elements toward target, with alpha upward and
    // extinction downward.
    void Learn(const matrix & stimulus, float target)
    {
        const float error = target - dot(stimulus, weights);
        const float delta = (error > 0 ? alpha.as_float() : extinction.as_float()) * error;
        if (delta != 0)
            weights.apply(stimulus, [delta](float w, float s) { return w + s * delta; });
    }


    // One learning step per US onset during a CS presentation, and one extinction step when a
    // presentation ends without a US. A CS or US that is absent for less than gap seconds continues.
    float TrialTick(float usSum)
    {
        const float dt = static_cast<float>(GetTickDuration());
        const float gapTime = gap.as_float();

        if (cs.sum() > 0)
        {
            if (!inTrial)
            {
                trialCs.set(0.0f);
                trialHadUs = false;
                inTrial = true;
            }
            trialCs.apply(cs, [](float held, float present) { return std::max(held, present); });
            csAbsentTime = 0;
        }
        else if (inTrial)
        {
            csAbsentTime += dt;
            if (csAbsentTime >= gapTime)
            {
                if (!trialHadUs)
                    Learn(trialCs, 0.0f);
                inTrial = false;
            }
        }

        if (usSum > 0)
        {
            if (inTrial && usAbsentTime >= gapTime)
                Learn(trialCs, usSum);
            if (inTrial)
                trialHadUs = true;
            usAbsentTime = 0;
        }
        else
            usAbsentTime += dt;

        return inTrial ? dot(trialCs, weights) : 0.0f;
    }


    void Tick()
    {
        if (!cs.connected() || !us.connected())
            return;

        float us_sum = us.sum();
        float response;
        if (trials.as_bool())
            response = TrialTick(us_sum);
        else
        {
            response = dot(cs, weights);
            Learn(cs, us_sum);
        }

        if (inverse.as_bool())
            cr = std::max(0.0f, response - us_sum);
        else
            cr = response;
    }
};

INSTALL_CLASS(Delta)

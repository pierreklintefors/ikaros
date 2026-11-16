#include "ikaros.h"

#include <algorithm>
#include <random>

using namespace ikaros;

class BehaviorSelector : public Module
{
    static constexpr float epsilon = 1e-5f;

    parameter count;
    matrix idleWeights;
    matrix engagedWeights;
    parameter minInterval;
    parameter maxInterval;
    parameter cooldown;
    parameter greeting;
    parameter greetingDelay;
    parameter farewell;
    parameter reactionCooldown;
    parameter seed;

    matrix engaged;
    matrix newTarget;
    matrix busy;
    matrix enable;

    matrix trigger;
    matrix selected;
    matrix wait;

    matrix sinceChosen; // seconds since each behaviour was last started
    matrix weights;     // work buffer for the weights of the present state
    std::mt19937 randomGenerator;
    float untilNext = 0;
    float untilGreeting = -1; // negative while no greeting is pending
    float sinceReaction = 0;
    bool wasEngaged = false;
    float lastNewTarget = 0;

    float
    Uniform(float low, float high)
    {
        if(high <= low)
            return low;
        return std::uniform_real_distribution<float>(low, high)(randomGenerator);
    }


    void
    Fire(int index)
    {
        if(index < 0 || index >= trigger.size())
            return;
        trigger(index) = 1;
        selected(0) = static_cast<float>(index);
        sinceChosen(index) = 0;
        untilNext = Uniform(minInterval.as_float(), maxInterval.as_float());
    }


    // Chooses a behaviour by weight among those outside their cooldown; -1 when none is available.
    int
    Choose(const matrix & stateWeights)
    {
        float total = 0;
        for(int i = 0; i < weights.size(); ++i)
        {
            const float w = i < stateWeights.size() ? std::max(stateWeights.data()[i], 0.0f) : 0.0f;
            weights(i) = sinceChosen(i) >= cooldown.as_float() - epsilon ? w : 0.0f;
            total += weights(i);
        }
        if(total <= 0.0f)
            return -1;
        float pick = Uniform(0.0f, total);
        for(int i = 0; i < weights.size(); ++i)
        {
            if(weights(i) > 0.0f && pick < weights(i))
                return i;
            pick -= weights(i);
        }
        for(int i = weights.size() - 1; i >= 0; --i)
            if(weights(i) > 0.0f)
                return i;
        return -1;
    }

public:
    void
    Init() override
    {
        Bind(count, "count");
        Bind(idleWeights, "idle_weights");
        Bind(engagedWeights, "engaged_weights");
        Bind(minInterval, "min_interval");
        Bind(maxInterval, "max_interval");
        Bind(cooldown, "cooldown");
        Bind(greeting, "greeting");
        Bind(greetingDelay, "greeting_delay");
        Bind(farewell, "farewell");
        Bind(reactionCooldown, "reaction_cooldown");
        Bind(seed, "seed");

        Bind(engaged, "ENGAGED");
        Bind(newTarget, "NEW_TARGET");
        Bind(busy, "BUSY");
        Bind(enable, "ENABLE");

        Bind(trigger, "TRIGGER");
        Bind(selected, "SELECTED");
        Bind(wait, "WAIT");

        if(greeting.as_int() >= trigger.size() || farewell.as_int() >= trigger.size())
            throw exception("BehaviorSelector: greeting and farewell must be below count.", path_);

        if(seed.as_int() < 0)
            randomGenerator.seed(std::random_device{}());
        else
            randomGenerator.seed(static_cast<std::mt19937::result_type>(seed.as_int()));

        sinceChosen.realloc(trigger.size());
        sinceChosen.set(cooldown.as_float());
        weights.realloc(trigger.size());
        selected(0) = -1;
        sinceReaction = reactionCooldown.as_float();
        untilNext = Uniform(minInterval.as_float(), maxInterval.as_float());
    }


    void
    Tick() override
    {
        const float dt = static_cast<float>(GetTickDuration());
        trigger.set(0.0f);
        for(int i = 0; i < sinceChosen.size(); ++i)
            sinceChosen(i) += dt;
        sinceReaction += dt;

        const bool isEngaged = engaged(0) > 0.5f;
        const float newTargetValue = newTarget.connected() ? newTarget(0) : 0.0f;
        const bool arrived = (isEngaged && !wasEngaged) || (newTargetValue > 0.5f && lastNewTarget <= 0.5f);
        const bool left = !isEngaged && wasEngaged;
        wasEngaged = isEngaged;
        lastNewTarget = newTargetValue;

        if(enable.connected() && enable(0) <= 0.5f)
        {
            untilGreeting = -1;
            wait(0) = untilNext;
            return;
        }

        const bool reactionAllowed = sinceReaction >= reactionCooldown.as_float() - epsilon;
        if(arrived && greeting.as_int() >= 0 && reactionAllowed)
        {
            untilGreeting = greetingDelay.as_float();
            sinceReaction = 0;
        }
        if(left)
        {
            untilGreeting = -1;
            if(farewell.as_int() >= 0 && reactionAllowed)
            {
                Fire(farewell.as_int());
                sinceReaction = 0;
            }
        }

        // Reactions do not wait for BUSY, so the robot responds to people at once.
        if(untilGreeting >= 0.0f)
        {
            if(untilGreeting <= epsilon)
            {
                untilGreeting = -1;
                Fire(greeting.as_int());
            }
            else
                untilGreeting -= dt;
        }

        // The idle interval runs only while nothing plays, is pending, or has just started.
        const bool isBusy = busy.connected() && busy(0) > 0.5f;
        const bool fired = trigger.max() > 0.0f;
        if(!isBusy && !fired && untilGreeting < 0.0f)
        {
            untilNext -= dt;
            if(untilNext <= epsilon)
            {
                const int choice = Choose(isEngaged ? engagedWeights : idleWeights);
                if(choice >= 0)
                    Fire(choice);
                else
                    untilNext = minInterval.as_float();
            }
        }
        wait(0) = std::max(untilNext, 0.0f);
    }
};

INSTALL_CLASS(BehaviorSelector)

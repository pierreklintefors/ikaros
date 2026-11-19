#include "ikaros.h"

#include <algorithm>
#include <cmath>
#include <deque>
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
    matrix arousalLevels;
    parameter arousalWidth;
    parameter memory;
    parameter excitedIntervalScale;
    parameter seed;

    matrix engaged;
    matrix newTarget;
    matrix busy;
    matrix enable;
    matrix pause;
    matrix arousal;
    matrix request;

    matrix trigger;
    matrix selected;
    matrix wait;

    matrix sinceChosen; // seconds since each behaviour was last started
    matrix weights;     // work buffer for the weights of the present state
    std::mt19937 randomGenerator;
    float untilNext = 0;
    bool greetingPending = false;
    float untilGreeting = 0; // seconds until a pending greeting starts
    float sinceReaction = 0;
    bool wasEngaged = false;
    float lastNewTarget = 0;
    std::deque<int> recent; // most recent spontaneous behaviours, newest first
    matrix lastRequest;

    float
    Uniform(float low, float high)
    {
        if(high <= low)
            return low;
        return std::uniform_real_distribution<float>(low, high)(randomGenerator);
    }


    float
    ArousalLevel() const
    {
        return arousal.connected() ? std::clamp(arousal(0), 0.0f, 1.0f) : 0.0f;
    }


    // The idle interval shrinks toward excited_interval_scale times its length as arousal rises.
    float
    NextInterval()
    {
        const float scale = 1.0f + (excitedIntervalScale.as_float() - 1.0f) * ArousalLevel();
        return std::max(scale, 0.0f) * Uniform(minInterval.as_float(), maxInterval.as_float());
    }


    void
    Fire(int index)
    {
        if(index < 0 || index >= trigger.size())
            return;
        trigger(index) = 1;
        selected(0) = static_cast<float>(index);
        sinceChosen(index) = 0;
        untilNext = NextInterval();
    }


    bool
    Recent(int index) const
    {
        return std::find(recent.begin(), recent.end(), index) != recent.end();
    }


    // Weight of behaviour i: its state weight, times how well its arousal level matches the present
    // arousal; 0 during its cooldown, and 0 when it is among the most recent ones and avoidRecent is set.
    float
    Weight(const matrix & stateWeights, int i, bool avoidRecent) const
    {
        float w = i < stateWeights.size() ? std::max(stateWeights.data()[i], 0.0f) : 0.0f;
        if(arousal.connected() && i < arousalLevels.size())
        {
            const float width = std::max(arousalWidth.as_float(), 0.01f);
            const float difference = arousalLevels.data()[i] - ArousalLevel();
            w *= std::exp(-0.5f * difference * difference / (width * width));
        }
        if(sinceChosen(i) < cooldown.as_float() - epsilon || (avoidRecent && Recent(i)))
            return 0.0f;
        return w;
    }


    // Chooses a behaviour by weight. Recent behaviours are avoided unless nothing else is available;
    // returns -1 when every weighted behaviour is cooling down.
    int
    Choose(const matrix & stateWeights)
    {
        float total = 0;
        for(int i = 0; i < weights.size(); ++i)
        {
            weights(i) = Weight(stateWeights, i, true);
            total += weights(i);
        }
        if(total <= 0.0f)
            for(int i = 0; i < weights.size(); ++i)
            {
                weights(i) = Weight(stateWeights, i, false);
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
        Bind(arousalLevels, "arousal_levels");
        Bind(arousalWidth, "arousal_width");
        Bind(memory, "memory");
        Bind(excitedIntervalScale, "excited_interval_scale");
        Bind(seed, "seed");

        Bind(engaged, "ENGAGED");
        Bind(newTarget, "NEW_TARGET");
        Bind(busy, "BUSY");
        Bind(enable, "ENABLE");
        Bind(pause, "PAUSE");
        Bind(arousal, "AROUSAL");
        Bind(request, "REQUEST");

        Bind(trigger, "TRIGGER");
        Bind(selected, "SELECTED");
        Bind(wait, "WAIT");

        if(greeting.as_int() >= trigger.size() || farewell.as_int() >= trigger.size())
            throw exception("BehaviorSelector: greeting and farewell must be below count.", path_);

        if(seed.as_int() < 0)
            randomGenerator.seed(std::random_device{}());
        else
            randomGenerator.seed(static_cast<std::mt19937::result_type>(seed.as_int()));

        if(request.connected())
        {
            lastRequest.realloc(request.size());
            lastRequest.set(0.0f);
        }
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
            greetingPending = false;
            wait(0) = untilNext;
            return;
        }

        // While paused, only requests start behaviours; nothing is saved up for later.
        const bool paused = pause.connected() && pause(0) > 0.5f;
        if(paused)
            greetingPending = false;

        const bool reactionAllowed = sinceReaction >= reactionCooldown.as_float() - epsilon;
        if(!paused && arrived && greeting.as_int() >= 0 && reactionAllowed)
        {
            greetingPending = true;
            untilGreeting = greetingDelay.as_float();
            sinceReaction = 0;
        }
        if(!paused && left)
        {
            greetingPending = false;
            if(farewell.as_int() >= 0 && reactionAllowed)
            {
                Fire(farewell.as_int());
                sinceReaction = 0;
            }
        }

        // A request starts its behaviour at once, unless that behaviour is still cooling down.
        if(request.connected())
            for(int i = 0; i < request.size(); ++i)
            {
                const float value = request.data()[i];
                if(value > 0.5f && lastRequest(i) <= 0.5f && i < trigger.size() && sinceChosen(i) >= cooldown.as_float() - epsilon)
                {
                    greetingPending = false;
                    Fire(i);
                }
                lastRequest(i) = value;
            }

        // Reactions do not wait for BUSY, so the robot responds to people at once.
        // The countdown can pass zero between ticks, so the pending flag, not its sign, marks a greeting.
        if(greetingPending)
        {
            if(untilGreeting <= epsilon)
            {
                greetingPending = false;
                Fire(greeting.as_int());
            }
            else
                untilGreeting -= dt;
        }

        // The idle interval runs only while nothing plays, is pending, or has just started.
        const bool isBusy = busy.connected() && busy(0) > 0.5f;
        const bool fired = trigger.max() > 0.0f;
        if(!paused && !isBusy && !fired && !greetingPending)
        {
            untilNext -= dt;
            if(untilNext <= epsilon)
            {
                const int choice = Choose(isEngaged ? engagedWeights : idleWeights);
                if(choice >= 0)
                {
                    Fire(choice);
                    recent.push_front(choice);
                    while(static_cast<int>(recent.size()) > std::max(memory.as_int(), 0))
                        recent.pop_back();
                }
                else
                    untilNext = minInterval.as_float();
            }
        }
        wait(0) = std::max(untilNext, 0.0f);
    }
};

INSTALL_CLASS(BehaviorSelector)

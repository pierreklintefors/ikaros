#include "ikaros.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using namespace ikaros;

class KeyframeAnimator : public Module
{
    static constexpr float epsilon = 1e-5f;

    enum class Ease
    {
        smooth,
        linear,
        step,
        back
    };

    struct Key
    {
        float time;
        float value;
        Ease ease;
    };

    struct Track
    {
        int channel;
        std::vector<Key> keys;
    };

    struct Event
    {
        float time;
        std::vector<int> indices; // alternatives; one is chosen at random each time
        float probability;
    };

    struct Animation
    {
        std::string name;
        float duration = 0;
        std::vector<Track> tracks;
        std::vector<Event> events;
    };

    parameter file;
    parameter channels;
    matrix rest;
    parameter events;
    parameter eventCount;
    parameter releaseRate;
    parameter interrupt;
    parameter intensityChannels;
    parameter mirrorChannels;
    parameter mirrorProbability;
    parameter speedVariation;
    parameter intensityVariation;
    parameter seed;

    matrix trigger;
    matrix stop;
    matrix speedInput;
    matrix intensityInput;

    matrix output;
    matrix eventOutput;
    matrix playing;
    matrix animationIndex;
    matrix progress;
    matrix done;
    matrix variation;

    std::vector<std::string> channelNames;
    std::vector<std::string> eventNames;
    std::vector<Animation> animations;

    matrix startValue; // channel values when the present animation started
    matrix lastTrigger;
    std::vector<bool> driven; // channels with a track in the present animation
    std::vector<bool> scaled; // channels whose deviation from rest follows INTENSITY
    std::vector<bool> mirrored; // channels whose deviation from rest may change sign per play
    std::mt19937 randomGenerator;
    float playSpeed = 1; // random tempo factor of the present play
    float playIntensity = 1; // random size factor of the present play
    float playMirror = 1; // -1 when the mirrored channels are flipped in the present play
    float lastStop = 0;

    int current = -1;
    float time = 0;
    size_t nextEvent = 0;

    std::atomic<int> playRequest{-1};
    std::atomic<bool> stopRequest{false};

    static std::vector<std::string>
    Names(const std::string & list)
    {
        std::vector<std::string> names;
        for(const std::string & name : split(list, ","))
            if(!trim(name).empty())
                names.push_back(trim(name));
        return names;
    }


    static int
    Find(const std::vector<std::string> & names, const std::string & name)
    {
        for(size_t i = 0; i < names.size(); ++i)
            if(names[i] == name)
                return static_cast<int>(i);
        return -1;
    }


    Ease
    ParseEase(const std::string & name, const std::string & where) const
    {
        if(name == "smooth")
            return Ease::smooth;
        if(name == "linear")
            return Ease::linear;
        if(name == "step")
            return Ease::step;
        if(name == "back")
            return Ease::back;
        throw exception("KeyframeAnimator: unknown easing \"" + name + "\" in " + where + "; use smooth, linear, step, or back.", path_);
    }


    // A track is either a list of [time, value, ease] keys or a dictionary with "keys", an optional
    // "start" time, and "repeat", which repeats the keys back to back.
    std::vector<Key>
    ParseKeys(const value & track, const std::string & where) const
    {
        const value * keyList = &track;
        float start = 0;
        int repeat = 1;
        if(track.is_dictionary())
        {
            const dictionary & d = track.as_dictionary();
            if(!d.contains("keys"))
                throw exception("KeyframeAnimator: track " + where + " has no keys.", path_);
            keyList = &d.at("keys");
            if(d.contains("start"))
                start = d.at("start").as_float();
            if(d.contains("repeat"))
                repeat = std::max(1, d.at("repeat").as_int());
        }
        if(!keyList->is_list() || keyList->size() == 0)
            throw exception("KeyframeAnimator: track " + where + " must be a non-empty list of [time, value] keys.", path_);

        std::vector<Key> base;
        for(const value & k : keyList->as_list())
        {
            if(!k.is_list() || k.size() < 2)
                throw exception("KeyframeAnimator: every key in " + where + " must be [time, value] or [time, value, ease].", path_);
            const list & l = k.as_list();
            Key key {l.at(0).as_float(), l.at(1).as_float(), Ease::smooth};
            if(l.size() > 2)
                key.ease = ParseEase(l.at(2).as_string(), where);
            if(!base.empty() && key.time < base.back().time)
                throw exception("KeyframeAnimator: key times in " + where + " must not decrease.", path_);
            base.push_back(key);
        }

        std::vector<Key> keys;
        const float period = base.back().time - base.front().time;
        for(int r = 0; r < repeat; ++r)
            for(const Key & key : base)
            {
                Key shifted = key;
                shifted.time += start + r * period;
                // The first key of a repetition coincides with the last key of the previous one.
                if(!keys.empty() && std::fabs(shifted.time - keys.back().time) < epsilon && r > 0 && &key == &base.front())
                    continue;
                keys.push_back(shifted);
            }
        return keys;
    }


    void
    LoadLibrary()
    {
        const std::string filename = file.as_string();
        if(filename.empty())
            throw exception("KeyframeAnimator: file must name a JSON animation library.", path_);
        std::filesystem::path resolved;
        if(!kernel().SanitizeReadPath(filename, resolved))
            throw exception("KeyframeAnimator: \"" + filename + "\" is not inside the project directory or UserData.", path_);

        dictionary library;
        try
        {
            library.load_json(resolved.string());
        }
        catch(const std::exception & error)
        {
            throw exception("KeyframeAnimator: could not read \"" + resolved.string() + "\": " + error.what(), path_);
        }

        if(!library.contains("animations") || !library.at("animations").is_list())
            throw exception("KeyframeAnimator: \"" + filename + "\" must contain a list named animations.", path_);

        for(const value & entry : library.at("animations").as_list())
        {
            if(!entry.is_dictionary())
                throw exception("KeyframeAnimator: every animation must be a dictionary.", path_);
            const dictionary & d = entry.as_dictionary();
            Animation animation;
            animation.name = d.contains("name") ? d.at("name").as_string() : "animation_" + std::to_string(animations.size());
            for(const Animation & other : animations)
                if(other.name == animation.name)
                    throw exception("KeyframeAnimator: animation name \"" + animation.name + "\" is used twice.", path_);

            float end = 0;
            if(d.contains("tracks"))
            {
                if(!d.at("tracks").is_dictionary())
                    throw exception("KeyframeAnimator: tracks of " + animation.name + " must be a dictionary from channel name to keys.", path_);
                for(const auto & [channelName, track] : d.at("tracks").as_dictionary())
                {
                    const int channel = Find(channelNames, channelName);
                    if(channel < 0)
                        throw exception("KeyframeAnimator: animation " + animation.name + " uses unknown channel \"" + channelName + "\".", path_);
                    Track t {channel, ParseKeys(track, animation.name + "." + channelName)};
                    end = std::max(end, t.keys.back().time);
                    animation.tracks.push_back(std::move(t));
                }
            }

            if(d.contains("events"))
            {
                if(!d.at("events").is_list())
                    throw exception("KeyframeAnimator: events of " + animation.name + " must be a list of [time, name].", path_);
                for(const value & e : d.at("events").as_list())
                {
                    if(!e.is_list() || e.size() < 2)
                        throw exception("KeyframeAnimator: every event in " + animation.name + " must be [time, name] or [time, name, probability].", path_);
                    const list & l = e.as_list();
                    Event event {l.at(0).as_float(), {}, l.size() > 2 ? std::clamp(l.at(2).as_float(), 0.0f, 1.0f) : 1.0f};
                    // The name can also be a list of alternatives.
                    std::vector<value> names;
                    if(l.at(1).is_list())
                        for(const value & alternative : l.at(1).as_list())
                            names.push_back(alternative);
                    else
                        names.push_back(l.at(1));
                    for(const value & name : names)
                    {
                        const int index = name.is_number() ? name.as_int() : Find(eventNames, name.as_string());
                        if(index < 0 || index >= eventOutput.size())
                            throw exception("KeyframeAnimator: animation " + animation.name + " uses unknown event \"" + name.as_string() + "\".", path_);
                        event.indices.push_back(index);
                    }
                    if(event.indices.empty())
                        throw exception("KeyframeAnimator: an event in " + animation.name + " has an empty list of alternatives.", path_);
                    end = std::max(end, event.time);
                    animation.events.push_back(std::move(event));
                }
                std::stable_sort(animation.events.begin(), animation.events.end(),
                                 [](const Event & a, const Event & b) { return a.time < b.time; });
            }

            animation.duration = d.contains("duration") ? std::max(d.at("duration").as_float(), end) : end;
            animations.push_back(std::move(animation));
        }
    }


    static float
    Shape(Ease ease, float x)
    {
        x = std::clamp(x, 0.0f, 1.0f);
        switch(ease)
        {
            case Ease::linear:
                return x;
            case Ease::step:
                return x >= 1.0f ? 1.0f : 0.0f;
            case Ease::back:
            {
                // Overshoots by about 10 percent before settling, which gives a springy pop.
                const float c = 1.70158f;
                const float y = x - 1.0f;
                return 1.0f + (c + 1.0f) * y * y * y + c * y * y;
            }
            case Ease::smooth:
            default:
                return x * x * (3.0f - 2.0f * x);
        }
    }


    // Value of a track at time t. Key values are taken as rest + gain * (value - rest), which scales
    // and mirrors the movement around rest. Before the first key the channel moves from its start value.
    static float
    Evaluate(const Track & track, float t, float start, float rest, float gain)
    {
        const std::vector<Key> & keys = track.keys;
        auto value = [&](const Key & key) { return rest + gain * (key.value - rest); };
        if(t < keys.front().time)
            return start + Shape(keys.front().ease, t / keys.front().time) * (value(keys.front()) - start);
        for(size_t k = 1; k < keys.size(); ++k)
            if(t < keys[k].time)
            {
                const Key & a = keys[k - 1];
                const Key & b = keys[k];
                return value(a) + Shape(b.ease, (t - a.time) / (b.time - a.time)) * (value(b) - value(a));
            }
        return value(keys.back());
    }


    float
    Uniform(float low, float high)
    {
        if(high <= low)
            return low;
        return std::uniform_real_distribution<float>(low, high)(randomGenerator);
    }


    // Marks the channels named in list; an empty list marks all channels when allIfEmpty is true.
    std::vector<bool>
    ChannelSet(const std::string & list, bool allIfEmpty, const std::string & parameterName) const
    {
        const std::vector<std::string> names = Names(list);
        std::vector<bool> set(channelNames.size(), allIfEmpty && names.empty());
        for(const std::string & name : names)
        {
            const int channel = Find(channelNames, name);
            if(channel < 0)
                throw exception("KeyframeAnimator: " + parameterName + " names unknown channel \"" + name + "\".", path_);
            set[channel] = true;
        }
        return set;
    }


    bool
    Start(int index)
    {
        if(index < 0 || index >= static_cast<int>(animations.size()))
        {
            Notify(msg_warning, "KeyframeAnimator: there is no animation " + std::to_string(index) + ".");
            return false;
        }
        if(current >= 0 && !interrupt.as_bool())
            return false;
        current = index;
        time = 0;
        nextEvent = 0;
        playSpeed = 1.0f + Uniform(-speedVariation.as_float(), speedVariation.as_float());
        playIntensity = 1.0f + Uniform(-intensityVariation.as_float(), intensityVariation.as_float());
        playMirror = Uniform(0.0f, 1.0f) < mirrorProbability.as_float() ? -1.0f : 1.0f;
        startValue.copy(output);
        std::fill(driven.begin(), driven.end(), false);
        for(const Track & track : animations[index].tracks)
            driven[track.channel] = true;
        return true;
    }


    void
    Finish()
    {
        current = -1;
        std::fill(driven.begin(), driven.end(), false);
    }

public:
    void
    Init() override
    {
        Bind(file, "file");
        Bind(channels, "channels");
        Bind(rest, "rest");
        Bind(events, "events");
        Bind(eventCount, "event_count");
        Bind(releaseRate, "release_rate");
        Bind(interrupt, "interrupt");
        Bind(intensityChannels, "intensity_channels");
        Bind(mirrorChannels, "mirror_channels");
        Bind(mirrorProbability, "mirror_probability");
        Bind(speedVariation, "speed_variation");
        Bind(intensityVariation, "intensity_variation");
        Bind(seed, "seed");

        Bind(trigger, "TRIGGER");
        Bind(stop, "STOP");
        Bind(speedInput, "SPEED");
        Bind(intensityInput, "INTENSITY");

        Bind(output, "OUTPUT");
        Bind(eventOutput, "EVENTS");
        Bind(playing, "PLAYING");
        Bind(animationIndex, "ANIMATION");
        Bind(progress, "PROGRESS");
        Bind(done, "DONE");
        Bind(variation, "VARIATION");

        channelNames = Names(channels.as_string());
        eventNames = Names(events.as_string());
        if(static_cast<int>(channelNames.size()) != rest.size())
            throw exception("KeyframeAnimator: channels lists " + std::to_string(channelNames.size()) +
                            " channels, but rest has " + std::to_string(rest.size()) + " values.", path_);
        if(output.size() != rest.size())
            throw exception("KeyframeAnimator: OUTPUT must have one element per rest value.", path_);
        if(static_cast<int>(eventNames.size()) > eventOutput.size())
            throw exception("KeyframeAnimator: event_count must be at least the number of event names.", path_);

        LoadLibrary();
        scaled = ChannelSet(intensityChannels.as_string(), true, "intensity_channels");
        mirrored = ChannelSet(mirrorChannels.as_string(), false, "mirror_channels");

        if(seed.as_int() < 0)
            randomGenerator.seed(std::random_device{}());
        else
            randomGenerator.seed(static_cast<std::mt19937::result_type>(seed.as_int()));

        startValue.realloc(rest.size());
        driven.assign(rest.size(), false);
        output.copy(rest);
        animationIndex(0) = -1;
        if(trigger.connected())
        {
            lastTrigger.realloc(trigger.size());
            lastTrigger.set(0.0f);
            if(trigger.size() > static_cast<int>(animations.size()))
                Notify(msg_warning, "KeyframeAnimator: TRIGGER has " + std::to_string(trigger.size()) + " elements but the library has only " +
                                    std::to_string(animations.size()) + " animations.");
        }
    }


    void
    Command(std::string command_name, dictionary & parameters) override
    {
        if(command_name == "play")
        {
            int index = -1;
            if(parameters.contains("name"))
                for(size_t i = 0; i < animations.size(); ++i)
                    if(animations[i].name == parameters["name"].as_string())
                        index = static_cast<int>(i);
            if(index < 0 && parameters.contains("x"))
                index = parameters["x"].as_int();
            playRequest = index;
        }
        else if(command_name == "stop")
            stopRequest = true;
        else
            Module::Command(command_name, parameters);
    }


    void
    Tick() override
    {
        eventOutput.set(0.0f);
        done(0) = 0;

        bool stopNow = stopRequest.exchange(false);
        if(stop.connected())
        {
            stopNow = stopNow || (stop(0) > 0.5f && lastStop <= 0.5f);
            lastStop = stop(0);
        }
        if(stopNow)
            Finish();

        int startIndex = playRequest.exchange(-1);
        if(trigger.connected())
            for(int i = trigger.size() - 1; i >= 0; --i)
            {
                if(trigger(i) > 0.5f && lastTrigger(i) <= 0.5f)
                    startIndex = i;
                lastTrigger(i) = trigger(i);
            }

        const bool started = startIndex >= 0 && Start(startIndex);

        const float speed = playSpeed * (speedInput.connected() ? std::max(speedInput(0), 0.05f) : 1.0f);
        const float intensity = playIntensity * (intensityInput.connected() ? std::max(intensityInput(0), 0.0f) : 1.0f);
        if(current >= 0 && !started)
            time += speed * static_cast<float>(GetTickDuration());

        // Rate parameters are delivered per tick by the kernel.
        const float release = releaseRate.as_float() <= 0.0f ? 1.0f : 1.0f - std::exp(-releaseRate.as_float());
        bool finished = false;
        if(current >= 0)
        {
            const Animation & animation = animations[current];
            const float t = std::min(time, animation.duration);
            for(const Track & track : animation.tracks)
            {
                const int c = track.channel;
                const float gain = (scaled[c] ? intensity : 1.0f) * (mirrored[c] ? playMirror : 1.0f);
                output(c) = Evaluate(track, t, startValue(c), rest(c), gain);
            }
            while(nextEvent < animation.events.size() && animation.events[nextEvent].time <= time + epsilon)
            {
                const Event & event = animation.events[nextEvent++];
                if(event.probability < 1.0f && Uniform(0.0f, 1.0f) >= event.probability)
                    continue;
                const int choice = std::min(static_cast<int>(Uniform(0.0f, static_cast<float>(event.indices.size()))),
                                            static_cast<int>(event.indices.size()) - 1);
                eventOutput(event.indices[choice]) = 1;
            }
            progress(0) = animation.duration > 0.0f ? std::clamp(time / animation.duration, 0.0f, 1.0f) : 1.0f;
            finished = time >= animation.duration - epsilon;
        }
        else
            progress(0) = 0;

        for(int i = 0; i < output.size(); ++i)
            if(!driven[i])
                output(i) += release * (rest(i) - output(i));

        // The last values are output on the final tick; the channels return to rest from the next tick.
        if(finished)
        {
            done(0) = 1;
            Finish();
        }

        playing(0) = current >= 0 ? 1.0f : 0.0f;
        animationIndex(0) = static_cast<float>(current);
        variation(0) = speed;
        variation(1) = intensity;
        variation(2) = playMirror;
    }
};

INSTALL_CLASS(KeyframeAnimator)

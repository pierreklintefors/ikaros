#include "ikaros.h"

#include <algorithm>
#include <cmath>

using namespace ikaros;

class ProximityVergence : public Module
{
    static constexpr float degreesPerRadian = 57.29577951308232f;

    parameter objectWidth;
    parameter eyeBaseline;
    parameter horizontalFov;
    parameter startDistance;
    parameter fullDistance;
    parameter gain;
    parameter maxVergence;
    parameter rate;
    parameter holdTime;

    matrix target;
    matrix valid;
    matrix measuredDistance;
    matrix measuredValid;

    matrix vergence;
    matrix distance;
    matrix near;
    matrix measured;

    float goal = 0;
    float nearGoal = 0;
    float sinceSeen = 0;
    float current = 0;
    float currentNear = 0;

    static float
    Smooth(float x)
    {
        x = std::clamp(x, 0.0f, 1.0f);
        return x * x * (3.0f - 2.0f * x);
    }

public:
    void
    Init() override
    {
        Bind(objectWidth, "object_width");
        Bind(eyeBaseline, "eye_baseline");
        Bind(horizontalFov, "horizontal_fov");
        Bind(startDistance, "start_distance");
        Bind(fullDistance, "full_distance");
        Bind(gain, "gain");
        Bind(maxVergence, "max_vergence");
        Bind(rate, "rate");
        Bind(holdTime, "hold_time");

        Bind(target, "TARGET");
        Bind(valid, "VALID");
        Bind(measuredDistance, "DISTANCE");
        Bind(measuredValid, "DISTANCE_VALID");

        Bind(vergence, "VERGENCE");
        Bind(distance, "DISTANCE");
        Bind(near, "NEAR");
        Bind(measured, "MEASURED");

        if(target.size() < 3)
            throw exception("ProximityVergence: TARGET must contain x, y, and width.", path_);
    }


    void
    Tick() override
    {
        const float width = target(2);
        const bool seen = valid.connected() ? valid(0) > 0.5f && width > 0.0f : width > 0.0f;
        // A measured distance, for example from StereoDistance, replaces the estimate from the size.
        const bool hasMeasurement = measuredDistance.connected() && measuredDistance(0) > 0.0f &&
                                    (!measuredValid.connected() || measuredValid(0) > 0.5f);
        measured(0) = hasMeasurement ? 1.0f : 0.0f;

        if(seen || hasMeasurement)
        {
            sinceSeen = 0;
            float d = measuredDistance.connected() ? measuredDistance(0) : 0.0f;
            if(!hasMeasurement)
            {
                // Half the target's angular width follows from its half width in the image, where the
                // image half width corresponds to half the field of view.
                const float halfField = 0.5f * horizontalFov.as_float() / degreesPerRadian;
                const float halfAngle = std::atan(0.5f * width * std::tan(halfField));
                d = 0.5f * objectWidth.as_float() / std::max(std::tan(halfAngle), 1e-6f);
            }
            distance(0) = d;

            // The eyes meet at distance d; the effect fades in between start_distance and full_distance.
            const float geometric = 2.0f * std::atan(0.5f * eyeBaseline.as_float() / d) * degreesPerRadian;
            const float span = startDistance.as_float() - fullDistance.as_float();
            nearGoal = span > 0.0f ? Smooth((startDistance.as_float() - d) / span) : (d <= fullDistance.as_float() ? 1.0f : 0.0f);
            goal = std::min(gain.as_float() * geometric * nearGoal, maxVergence.as_float());
        }
        else
        {
            sinceSeen += static_cast<float>(GetTickDuration());
            if(sinceSeen > holdTime.as_float())
            {
                goal = 0;
                nearGoal = 0;
                distance(0) = 0;
            }
        }

        // Rate parameters are delivered per tick by the kernel.
        const float blend = rate.as_float() <= 0.0f ? 1.0f : 1.0f - std::exp(-rate.as_float());
        current += blend * (goal - current);
        currentNear += blend * (nearGoal - currentNear);
        vergence(0) = current;
        near(0) = currentNear;
    }
};

INSTALL_CLASS(ProximityVergence)

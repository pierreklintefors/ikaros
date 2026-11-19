#include "ikaros.h"

#include <algorithm>
#include <cmath>

using namespace ikaros;

class StereoDistance : public Module
{
    static constexpr float degreesPerRadian = 57.29577951308232f;

    parameter baseline;
    parameter horizontalFov;
    parameter xSign;
    parameter leftSign;
    parameter positionOffset;
    parameter minDistance;
    parameter maxDistance;
    parameter maxHeightDifference;
    parameter smoothing;
    parameter holdTime;

    matrix leftTarget;
    matrix leftValid;
    matrix rightTarget;
    matrix rightValid;
    matrix eyePan;

    matrix distance;
    matrix valid;

    float current = 0;
    float sinceMeasured = 1e9f;

    // Direction to a target at image x seen by an eye panned by eyeDegrees, in radians.
    float
    Direction(float x, float eyeDegrees) const
    {
        const float halfField = 0.5f * horizontalFov.as_float() / degreesPerRadian;
        return eyeDegrees / degreesPerRadian + xSign.as_float() * std::atan(x * std::tan(halfField));
    }

public:
    void
    Init() override
    {
        Bind(baseline, "baseline");
        Bind(horizontalFov, "horizontal_fov");
        Bind(xSign, "x_sign");
        Bind(leftSign, "left_sign");
        Bind(positionOffset, "position_offset");
        Bind(minDistance, "min_distance");
        Bind(maxDistance, "max_distance");
        Bind(maxHeightDifference, "max_height_difference");
        Bind(smoothing, "smoothing");
        Bind(holdTime, "hold_time");

        Bind(leftTarget, "LEFT_TARGET");
        Bind(leftValid, "LEFT_VALID");
        Bind(rightTarget, "RIGHT_TARGET");
        Bind(rightValid, "RIGHT_VALID");
        Bind(eyePan, "EYE_PAN");

        Bind(distance, "DISTANCE");
        Bind(valid, "VALID");

        if(leftTarget.size() < 2 || rightTarget.size() < 2)
            throw exception("StereoDistance: LEFT_TARGET and RIGHT_TARGET must contain at least x and y.", path_);
    }


    void
    Tick() override
    {
        sinceMeasured += static_cast<float>(GetTickDuration());

        const float leftEye = eyePan(0) - positionOffset.as_float();
        const float rightEye = eyePan(1) - positionOffset.as_float();
        const bool seen = leftValid(0) > 0.5f && rightValid(0) > 0.5f &&
                          std::fabs(leftTarget(1) - rightTarget(1)) <= maxHeightDifference.as_float() &&
                          std::isfinite(leftEye) && std::isfinite(rightEye) &&
                          std::fabs(leftEye) < 80.0f && std::fabs(rightEye) < 80.0f;
        if(seen)
        {
            const float spread = leftSign.as_float() *
                (std::tan(Direction(rightTarget(0), rightEye)) - std::tan(Direction(leftTarget(0), leftEye)));
            if(spread > 0.0f)
            {
                const float measured = baseline.as_float() / spread;
                if(measured >= minDistance.as_float() && measured <= maxDistance.as_float())
                {
                    // Rate parameters are delivered per tick by the kernel.
                    const float blend = smoothing.as_float() <= 0.0f || sinceMeasured > holdTime.as_float()
                                            ? 1.0f : 1.0f - std::exp(-smoothing.as_float());
                    current += blend * (measured - current);
                    sinceMeasured = 0;
                }
            }
        }

        const bool isValid = sinceMeasured <= holdTime.as_float();
        valid(0) = isValid ? 1.0f : 0.0f;
        distance(0) = isValid ? current : 0.0f;
    }
};

INSTALL_CLASS(StereoDistance)

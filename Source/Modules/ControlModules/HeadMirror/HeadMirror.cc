#include "ikaros.h"

#include <algorithm>
#include <cmath>

using namespace ikaros;

class HeadMirror : public Module
{
    static constexpr float degreesPerRadian = 57.29577951308232f;
    static constexpr float followThreshold = 0.02f; // weight below which GazeLock takes over again

    enum State
    {
        noPerson = 0,
        tracked = 1,
        mirroring = 2
    };

    parameter yawGain;
    parameter pitchGain;
    parameter yawSign;
    parameter pitchSign;
    parameter maxPan;
    parameter maxTilt;
    parameter poseSmoothing;
    parameter tiltDeadZone;
    parameter panDeadZone;
    parameter settleTime;

    parameter engageTime;
    parameter minWidth;
    parameter facingLimit;
    parameter lostTime;
    parameter fadeIn;
    parameter fadeOut;

    parameter viewCompensation;
    parameter horizontalFov;
    parameter verticalFov;

    matrix pose;
    matrix valid;
    matrix direction;
    matrix newTarget;
    matrix target;
    matrix locked;
    matrix enable;

    matrix headGoal;
    matrix follow;
    matrix mirroringFlag;
    matrix weightOutput;
    matrix offset;
    matrix state;

    bool seen = false;          // a valid detection of the current person has arrived
    bool poseFresh = false;     // the pose filter has been set for the current person
    bool isMirroring = false;
    float timeSinceValid = 0;
    float engageTimer = 0;
    float filteredYaw = 0;
    float filteredPitch = 0;
    float mirrorTilt = 0;       // offset before weighting; held while not mirroring
    float mirrorPan = 0;
    float weight = 0;
    // Dead-zone state per axis: 0 tilt, 1 pan.
    struct Axis
    {
        float held = 0;         // head goal sent to GazeLock
        bool tracking = false;  // following the goal after it left the dead zone
        float anchor = 0;       // goal value the settle time is measured from
        float settled = 0;      // seconds the goal has stayed near the anchor
    };
    Axis axes[2];

    // Rate parameters are delivered per tick by the kernel.
    static float
    Blend(const parameter & rate)
    {
        const float ratePerTick = rate.as_float();
        return ratePerTick <= 0.0f ? 1.0f : 1.0f - std::exp(-ratePerTick);
    }


    static float
    ImageAngle(float coordinate, float fieldOfView)
    {
        const float halfField = 0.5f * fieldOfView / degreesPerRadian;
        return std::atan(coordinate * std::tan(halfField)) * degreesPerRadian;
    }


    static bool
    Active(const matrix & input, bool unconnected)
    {
        return input.connected() ? input(0) > 0.5f : unconnected;
    }


    void
    StartNewPerson()
    {
        seen = false;
        poseFresh = false;
        isMirroring = false;
        engageTimer = 0;
    }


    void
    UpdatePose()
    {
        float yaw = pose(0);
        float pitch = pose(1);
        const float compensation = viewCompensation.as_float();
        if(compensation != 0.0f && target.connected() && target.size() >= 2)
        {
            yaw -= compensation * ImageAngle(target(0), horizontalFov.as_float());
            pitch -= compensation * ImageAngle(target(1), verticalFov.as_float());
        }
        if(!std::isfinite(yaw) || !std::isfinite(pitch))
            return;

        const float blend = poseFresh ? Blend(poseSmoothing) : 1.0f;
        filteredYaw += blend * (yaw - filteredYaw);
        filteredPitch += blend * (pitch - filteredPitch);
        poseFresh = true;
    }


    // The goal is held while the desired value stays within the dead zone. Once it leaves the dead zone
    // the goal follows it every tick until it has stayed within a quarter of the dead zone for
    // settle_time, so a movement is followed to its end in one go and then held still.
    float
    StepAxis(Axis & axis, float desired, float deadZone, bool driving, float dt)
    {
        if(!driving)
        {
            axis.tracking = false;
            axis.held = desired;
            return axis.held;
        }

        if(!axis.tracking && std::fabs(desired - axis.held) > deadZone)
        {
            axis.tracking = true;
            axis.anchor = desired;
            axis.settled = 0;
        }

        if(axis.tracking)
        {
            axis.held = desired;
            if(std::fabs(desired - axis.anchor) > 0.25f * deadZone)
            {
                axis.anchor = desired;
                axis.settled = 0;
            }
            else
                axis.settled += dt;
            if(deadZone > 0.0f && axis.settled >= settleTime.as_float())
                axis.tracking = false;
        }
        return axis.held;
    }


    bool
    CloseEnough() const
    {
        const float width = minWidth.as_float();
        if(width <= 0.0f || !target.connected() || target.size() < 3)
            return true;
        return target(2) >= width;
    }

public:
    void
    Init() override
    {
        Bind(yawGain, "yaw_gain");
        Bind(pitchGain, "pitch_gain");
        Bind(yawSign, "yaw_sign");
        Bind(pitchSign, "pitch_sign");
        Bind(maxPan, "max_pan");
        Bind(maxTilt, "max_tilt");
        Bind(poseSmoothing, "pose_smoothing");
        Bind(tiltDeadZone, "tilt_dead_zone");
        Bind(panDeadZone, "pan_dead_zone");
        Bind(settleTime, "settle_time");

        Bind(engageTime, "engage_time");
        Bind(minWidth, "min_width");
        Bind(facingLimit, "facing_limit");
        Bind(lostTime, "lost_time");
        Bind(fadeIn, "fade_in");
        Bind(fadeOut, "fade_out");

        Bind(viewCompensation, "view_compensation");
        Bind(horizontalFov, "horizontal_fov");
        Bind(verticalFov, "vertical_fov");

        Bind(pose, "POSE");
        Bind(valid, "VALID");
        Bind(direction, "DIRECTION");
        Bind(newTarget, "NEW_TARGET");
        Bind(target, "TARGET");
        Bind(locked, "LOCKED");
        Bind(enable, "ENABLE");

        Bind(headGoal, "HEAD_GOAL");
        Bind(follow, "FOLLOW");
        Bind(mirroringFlag, "MIRRORING");
        Bind(weightOutput, "WEIGHT");
        Bind(offset, "OFFSET");
        Bind(state, "STATE");

        if(pose.size() < 2)
            throw exception("HeadMirror: POSE must contain at least yaw and pitch.", path_);
    }


    void
    Tick() override
    {
        const float dt = static_cast<float>(GetTickDuration());

        if(Active(newTarget, false))
            StartNewPerson();

        if(valid(0) > 0.5f)
        {
            seen = true;
            timeSinceValid = 0;
            UpdatePose();
        }
        else
            timeSinceValid += dt;

        const bool present = seen && timeSinceValid <= lostTime.as_float() && Active(locked, true);
        if(!present)
            StartNewPerson();

        const bool enabled = Active(enable, true);
        const bool facing = std::fabs(filteredYaw) <= facingLimit.as_float();
        if(present && enabled && poseFresh && CloseEnough() && facing)
            engageTimer += dt;
        else
            engageTimer = 0;

        if(!enabled)
            isMirroring = false;
        else if(!isMirroring && present && engageTimer >= engageTime.as_float() && weight <= followThreshold)
            isMirroring = true; // a new person is mirrored only after the previous offset has faded out

        // The offset is frozen when mirroring ends so that it fades out without changing shape.
        if(isMirroring)
        {
            mirrorPan = std::clamp(yawSign.as_float() * yawGain.as_float() * filteredYaw,
                                   -maxPan.as_float(), maxPan.as_float());
            mirrorTilt = std::clamp(pitchSign.as_float() * pitchGain.as_float() * filteredPitch,
                                    -maxTilt.as_float(), maxTilt.as_float());
        }

        const float goal = isMirroring ? 1.0f : 0.0f;
        weight += (goal > weight ? Blend(fadeIn) : Blend(fadeOut)) * (goal - weight);
        if(!isMirroring && weight < 1e-4f)
            weight = 0;

        offset(0) = weight * mirrorTilt + 0.0f; // + 0 turns -0 into 0 for display
        offset(1) = weight * mirrorPan + 0.0f;
        // While HEAD_GOAL drives the head, it moves only when the goal is more than a dead zone away,
        // which turns many small corrections into fewer, larger movements.
        const bool driving = isMirroring || weight > followThreshold;
        const float desiredTilt = direction(0) + offset(0);
        const float desiredPan = direction(1) + offset(1);
        headGoal(0) = StepAxis(axes[0], desiredTilt, tiltDeadZone.as_float(), driving, dt);
        headGoal(1) = StepAxis(axes[1], desiredPan, panDeadZone.as_float(), driving, dt);

        follow(0) = driving ? 0.0f : 1.0f;
        mirroringFlag(0) = isMirroring ? 1.0f : 0.0f;
        weightOutput(0) = weight;
        state(0) = static_cast<float>(isMirroring ? mirroring : present ? tracked : noPerson);
    }
};

INSTALL_CLASS(HeadMirror)

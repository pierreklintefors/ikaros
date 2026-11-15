#include "ikaros.h"

#include <algorithm>
#include <atomic>
#include <cmath>

using namespace ikaros;

class GazeLock : public Module
{
    static constexpr int historyLength = 64;
    static constexpr float degreesPerRadian = 57.29577951308232f;

    parameter positionOffset;
    parameter pupilSize;

    parameter horizontalFov;
    parameter verticalFov;
    parameter xSign;
    parameter ySign;
    parameter aimX;
    parameter aimY;

    parameter updateRate;
    parameter latency;
    parameter holdTime;
    parameter releaseRate;

    parameter headGain;
    parameter headDeadZone;
    parameter headMaxSpeed;
    parameter headMaxAcceleration;
    parameter eyeSmoothing;

    parameter tiltMin;
    parameter tiltMax;
    parameter panMin;
    parameter panMax;
    parameter eyeMin;
    parameter eyeMax;
    parameter vergence;

    matrix presentPosition;
    matrix target;
    matrix valid;
    matrix follow;
    matrix manualHead;
    matrix pupilInput;

    matrix goalPosition;
    matrix headGoal;
    matrix eyeGoal;
    matrix targetDirection;
    matrix locked;

    matrix gazeHistory; // rows of head tilt, head pan, eye pan in centered degrees
    int historyIndex = 0;

    float targetElevation = 0;
    float targetAzimuth = 0;
    float eyeCenter = 0;
    float timeSinceSeen = 0;
    bool isLocked = false;
    bool manualLock = false;
    bool headMoving[2] = {false, false};
    float headVelocity[2] = {0, 0}; // head goal change per tick

    std::atomic<bool> lockRequested{false};
    std::atomic<bool> releaseRequested{false};

    // Rate parameters are delivered per tick by the kernel.
    static float
    Blend(const parameter & rate)
    {
        const float ratePerTick = rate.as_float();
        return ratePerTick <= 0.0f ? 1.0f : 1.0f - std::exp(-ratePerTick);
    }


    bool
    PresentAvailable() const
    {
        const float offset = positionOffset.as_float();
        for(int i = 0; i < 4; ++i)
        {
            const float value = presentPosition(i);
            if(!std::isfinite(value) || std::fabs(value - offset) > 90.0f)
                return false;
        }
        return true;
    }


    float
    ImageAngle(float coordinate, float fieldOfView, float sign) const
    {
        const float halfField = 0.5f * fieldOfView / degreesPerRadian;
        return sign * std::atan(coordinate * std::tan(halfField)) * degreesPerRadian;
    }


    void
    RecordGaze(float tilt, float pan, float eyePan)
    {
        historyIndex = (historyIndex + 1) % historyLength;
        gazeHistory(historyIndex, 0) = tilt;
        gazeHistory(historyIndex, 1) = pan;
        gazeHistory(historyIndex, 2) = eyePan;
    }


    int
    DelayedHistoryIndex(float dt) const
    {
        int delay = dt > 0.0f ? static_cast<int>(std::lround(latency.as_float() / dt)) : 0;
        delay = std::clamp(delay, 0, historyLength - 1);
        return (historyIndex - delay + historyLength) % historyLength;
    }


    void
    UpdateTarget(float dt)
    {
        const int row = DelayedHistoryIndex(dt);
        // The stored direction is where the gaze must point for the target to appear at the aim point.
        const float measuredElevation = gazeHistory(row, 0) +
            ImageAngle(target(1), verticalFov.as_float(), ySign.as_float()) -
            ImageAngle(aimY.as_float(), verticalFov.as_float(), ySign.as_float());
        const float measuredAzimuth = gazeHistory(row, 1) + gazeHistory(row, 2) +
            ImageAngle(target(0), horizontalFov.as_float(), xSign.as_float()) -
            ImageAngle(aimX.as_float(), horizontalFov.as_float(), xSign.as_float());

        const float blend = isLocked ? Blend(updateRate) : 1.0f;
        targetElevation += blend * (measuredElevation - targetElevation);
        targetAzimuth += blend * (measuredAzimuth - targetAzimuth);
        isLocked = true;
        timeSinceSeen = 0;
    }


    // Moves one head axis toward desired. The speed follows the error, is limited to head_max_speed,
    // and changes by at most head_max_acceleration so that starts and stops are smooth. Near the goal
    // the speed is also limited to what can still be braked within the remaining distance.
    float
    StepHead(int axis, float current, float desired, bool following, float minimum, float maximum, float dt)
    {
        // Braking toward a joint limit is as smooth as braking toward the goal.
        desired = std::clamp(desired, minimum, maximum);
        const float error = desired - current;
        const float maxStep = headMaxSpeed.as_float();
        float targetVelocity = error;

        if(following)
        {
            const float deadZone = headDeadZone.as_float();
            if(std::fabs(error) > deadZone)
                headMoving[axis] = true;
            else if(std::fabs(error) < 0.25f * deadZone)
                headMoving[axis] = false;
            targetVelocity = headMoving[axis] ? headGain.as_float() * error : 0.0f;
        }
        targetVelocity = std::clamp(targetVelocity, -maxStep, maxStep);

        float & velocity = headVelocity[axis];
        const float maxChange = headMaxAcceleration.as_float() * dt * dt;
        if(maxChange > 0.0f)
        {
            // Largest speed from which the head can still stop within the error in whole ticks.
            const float braking = std::sqrt(2.0f * maxChange * std::fabs(error) + 0.25f * maxChange * maxChange) - 0.5f * maxChange;
            targetVelocity = std::clamp(targetVelocity, -braking, braking);
            velocity += std::clamp(targetVelocity - velocity, -maxChange, maxChange);
        }
        else
            velocity = targetVelocity;

        // Arriving on the goal ends the movement when the remaining speed can be removed within the
        // acceleration limit; carrying the speed over would overshoot.
        const bool canStop = maxChange <= 0.0f || std::fabs(velocity) <= 1.5f * maxChange;
        if(canStop && velocity * error > 0.0f && std::fabs(velocity) >= std::fabs(error))
        {
            velocity = 0;
            return std::clamp(desired, minimum, maximum);
        }

        const float next = current + velocity;
        if(next < minimum || next > maximum)
        {
            velocity = 0;
            return std::clamp(next, minimum, maximum);
        }
        return next;
    }

public:
    void
    Init() override
    {
        Bind(positionOffset, "position_offset");
        Bind(pupilSize, "pupil_size");

        Bind(horizontalFov, "horizontal_fov");
        Bind(verticalFov, "vertical_fov");
        Bind(xSign, "x_sign");
        Bind(ySign, "y_sign");
        Bind(aimX, "aim_x");
        Bind(aimY, "aim_y");

        Bind(updateRate, "update_rate");
        Bind(latency, "latency");
        Bind(holdTime, "hold_time");
        Bind(releaseRate, "release_rate");

        Bind(headGain, "head_gain");
        Bind(headDeadZone, "head_dead_zone");
        Bind(headMaxSpeed, "head_max_speed");
        Bind(headMaxAcceleration, "head_max_acceleration");
        Bind(eyeSmoothing, "eye_smoothing");

        Bind(tiltMin, "tilt_min");
        Bind(tiltMax, "tilt_max");
        Bind(panMin, "pan_min");
        Bind(panMax, "pan_max");
        Bind(eyeMin, "eye_min");
        Bind(eyeMax, "eye_max");
        Bind(vergence, "vergence");

        Bind(presentPosition, "PRESENT_POSITION");
        Bind(target, "TARGET");
        Bind(valid, "VALID");
        Bind(follow, "FOLLOW");
        Bind(manualHead, "MANUAL_HEAD");
        Bind(pupilInput, "PUPIL_SIZE");

        Bind(goalPosition, "GOAL_POSITION");
        Bind(headGoal, "HEAD_GOAL");
        Bind(eyeGoal, "EYE_GOAL");
        Bind(targetDirection, "TARGET_DIRECTION");
        Bind(locked, "LOCKED");

        if(presentPosition.size() < 4)
            throw exception("GazeLock: PRESENT_POSITION must contain at least tilt, pan, left eye, and right eye.", path_);
        if(goalPosition.size() < 6)
            throw exception("GazeLock: GOAL_POSITION must contain at least six servos.", path_);
        if(target.connected() && target.size() < 2)
            throw exception("GazeLock: TARGET must contain at least x and y.", path_);

        gazeHistory.realloc(historyLength, 3);
        gazeHistory.set(0.0f);
    }


    void
    Command(std::string command_name, dictionary & parameters) override
    {
        if(command_name == "lock")
            lockRequested = true;
        else if(command_name == "release")
            releaseRequested = true;
        else
            Module::Command(command_name, parameters);
    }


    void
    Tick() override
    {
        const float dt = static_cast<float>(GetTickDuration());
        const float offset = positionOffset.as_float();
        const bool following = !follow.connected() || follow(0) > 0.5f;

        const bool presentAvailable = PresentAvailable();
        const float headTilt = presentAvailable ? presentPosition(0) - offset : headGoal(0);
        const float headPan = presentAvailable ? presentPosition(1) - offset : headGoal(1);
        const float eyePan = presentAvailable ?
            0.5f * (presentPosition(2) + presentPosition(3)) - offset :
            0.5f * (eyeGoal(0) + eyeGoal(1));
        RecordGaze(headTilt, headPan, eyePan);

        if(releaseRequested.exchange(false))
        {
            isLocked = false;
            manualLock = false;
        }

        if(lockRequested.exchange(false))
        {
            targetElevation = headTilt;
            targetAzimuth = headPan + eyePan;
            isLocked = true;
            manualLock = true;
            timeSinceSeen = 0;
        }

        if(target.connected() && valid.connected() && valid(0) > 0.5f)
            UpdateTarget(dt);
        else
        {
            timeSinceSeen += dt;
            if(isLocked && !manualLock && timeSinceSeen > holdTime.as_float())
                isLocked = false;
        }

        const float release = Blend(releaseRate);
        if(!isLocked)
        {
            targetElevation -= release * targetElevation;
            targetAzimuth -= release * targetAzimuth;
        }

        float desiredTilt = headGoal(0);
        float desiredPan = headGoal(1);
        if(following)
        {
            desiredTilt = targetElevation;
            desiredPan = targetAzimuth;
        }
        else if(manualHead.connected())
        {
            desiredTilt = manualHead(0);
            desiredPan = manualHead(1);
        }

        headGoal(0) = StepHead(0, headGoal(0), desiredTilt, following, tiltMin.as_float(), tiltMax.as_float(), dt);
        headGoal(1) = StepHead(1, headGoal(1), desiredPan, following, panMin.as_float(), panMax.as_float(), dt);

        if(isLocked)
            eyeCenter += Blend(eyeSmoothing) * (targetAzimuth - headPan - eyeCenter);
        else
            eyeCenter -= release * eyeCenter;
        eyeCenter = std::clamp(eyeCenter, eyeMin.as_float(), eyeMax.as_float());

        const float halfVergence = 0.5f * vergence.as_float();
        eyeGoal(0) = std::clamp(eyeCenter + halfVergence, eyeMin.as_float(), eyeMax.as_float());
        eyeGoal(1) = std::clamp(eyeCenter - halfVergence, eyeMin.as_float(), eyeMax.as_float());

        targetDirection(0) = targetElevation;
        targetDirection(1) = targetAzimuth;
        locked(0) = isLocked ? 1.0f : 0.0f;

        goalPosition.set(offset);
        goalPosition(0) = headGoal(0) + offset;
        goalPosition(1) = headGoal(1) + offset;
        goalPosition(2) = eyeGoal(0) + offset;
        goalPosition(3) = eyeGoal(1) + offset;
        goalPosition(4) = pupilInput.connected() ? pupilInput(0) : pupilSize.as_float();
        goalPosition(5) = pupilInput.connected() ? pupilInput(1) : pupilSize.as_float();
    }
};

INSTALL_CLASS(GazeLock)

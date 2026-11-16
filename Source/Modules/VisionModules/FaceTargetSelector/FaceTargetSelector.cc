#include "ikaros.h"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace ikaros;

class FaceTargetSelector : public Module
{
    parameter selection;
    parameter minWidth;
    parameter minConfidence;
    parameter matchDistance;
    parameter switchRatio;
    parameter switchDelay;
    parameter holdTime;
    parameter smoothing;
    parameter ignoreRepeated;

    matrix boxes;
    matrix poses;

    matrix target;
    matrix valid;
    matrix targetBox;
    matrix count;
    matrix targetPose;
    matrix newTarget;

    matrix boxRow;

    bool tracking = false;
    float trackedX = 0;
    float trackedY = 0;
    float missingTime = 0;
    bool challenged = false;
    float challengeTime = 0; // seconds another face has been clearly better than the tracked one

    std::vector<float> previousBoxes; // detector output of the previous tick, to recognize repeated frames

    int
    BoxCount() const
    {
        if(boxes.empty())
            return 0;
        if(boxes.rank() == 2)
            return boxes.cols() >= 5 ? boxes.rows() : 0;
        if(boxes.rank() == 1)
            return boxes.size() >= 5 ? 1 : 0;
        return 0;
    }


    float
    Box(int index, int field) const
    {
        if(boxes.rank() == 2)
            return boxes(index, field);
        return boxes(field);
    }


    // Pose rows are aligned with the box rows; missing rows give zero.
    float
    Pose(int index, int field) const
    {
        if(!poses.connected() || poses.empty())
            return 0.0f;
        if(poses.rank() == 2)
            return index < poses.rows() && poses.cols() >= 3 ? poses(index, field) : 0.0f;
        if(poses.rank() == 1)
            return index == 0 && poses.size() >= 3 ? poses(field) : 0.0f;
        return 0.0f;
    }


    float
    CenterX(int index) const
    {
        return Box(index, 0) + 0.5f * Box(index, 2);
    }


    float
    CenterY(int index) const
    {
        return Box(index, 1) + 0.5f * Box(index, 3);
    }


    bool
    Accepted(int index) const
    {
        const float width = Box(index, 2);
        const float confidence = Box(index, 4);
        return std::isfinite(width) && std::isfinite(confidence) &&
               width >= minWidth.as_float() && width > 0.0f &&
               confidence >= minConfidence.as_float();
    }


    // Larger is better for both selection modes.
    float
    Score(int index) const
    {
        if(selection.as_int() == 1)
        {
            const float x = CenterX(index);
            const float y = CenterY(index);
            return 1.0f / (0.05f + std::sqrt(x * x + y * y));
        }
        return Box(index, 2);
    }


    // A detector run on the same camera frame produces exactly the same boxes. Identical non-empty
    // output therefore means that no new frame has arrived since the previous tick.
    bool
    RepeatedDetection()
    {
        const int n = BoxCount();
        const std::size_t valueCount = static_cast<std::size_t>(n) * 5;
        bool repeated = n > 0 && previousBoxes.size() == valueCount;

        if(previousBoxes.size() != valueCount)
            previousBoxes.resize(valueCount);

        for(int i = 0; i < n; ++i)
            for(int field = 0; field < 5; ++field)
            {
                float & previous = previousBoxes[static_cast<std::size_t>(i) * 5 + field];
                const float current = Box(i, field);
                if(previous != current)
                    repeated = false;
                previous = current;
            }
        return repeated;
    }


    // Keeps the target during a missing or repeated frame; drops it after hold_time.
    bool
    HoldTarget()
    {
        missingTime += static_cast<float>(GetTickDuration());
        if(missingTime <= holdTime.as_float())
            return true;
        tracking = false;
        challenged = false;
        target.reset();
        targetPose.reset();
        return false;
    }


    void
    Publish(int index, bool newIdentity)
    {
        const float x = CenterX(index);
        const float y = CenterY(index);
        const float width = Box(index, 2);

        // Rate parameters are delivered per tick by the kernel.
        const float ratePerTick = smoothing.as_float();
        const float blend = newIdentity || ratePerTick <= 0.0f ? 1.0f : 1.0f - std::exp(-ratePerTick);

        target(0) += blend * (x - target(0));
        target(1) += blend * (y - target(1));
        target(2) += blend * (width - target(2));
        for(int i = 0; i < 3; ++i)
            targetPose(i) += blend * (Pose(index, i) - targetPose(i));
        valid(0) = 1;
        if(newIdentity)
        {
            newTarget(0) = 1;
            challenged = false;
        }

        for(int i = 0; i < 5; ++i)
            boxRow(i) = Box(index, i);
        targetBox.append(boxRow);

        tracking = true;
        trackedX = x;
        trackedY = y;
        missingTime = 0;
    }

public:
    void
    Init() override
    {
        Bind(selection, "selection");
        Bind(minWidth, "min_width");
        Bind(minConfidence, "min_confidence");
        Bind(matchDistance, "match_distance");
        Bind(switchRatio, "switch_ratio");
        Bind(switchDelay, "switch_delay");
        Bind(holdTime, "hold_time");
        Bind(smoothing, "smoothing");
        Bind(ignoreRepeated, "ignore_repeated");

        Bind(boxes, "BOXES");
        Bind(poses, "POSES");
        Bind(target, "TARGET");
        Bind(valid, "VALID");
        Bind(targetBox, "TARGET_BOX");
        Bind(count, "COUNT");
        Bind(targetPose, "TARGET_POSE");
        Bind(newTarget, "NEW_TARGET");

        boxRow.realloc(5);
        previousBoxes.reserve(16 * 5);
    }


    void
    Tick() override
    {
        valid.reset();
        newTarget.reset();
        targetBox.clear();

        if(RepeatedDetection() && ignoreRepeated.as_bool())
        {
            if(challenged)
                challengeTime += static_cast<float>(GetTickDuration());
            if(tracking && HoldTarget())
                targetBox.append(boxRow);
            return;
        }

        count.reset();
        const int n = BoxCount();
        int best = -1;
        int matched = -1;
        float matchedDistance = matchDistance.as_float();
        int accepted = 0;

        for(int i = 0; i < n; ++i)
        {
            if(!Accepted(i))
                continue;
            ++accepted;

            if(best < 0 || Score(i) > Score(best))
                best = i;

            if(tracking)
            {
                const float dx = CenterX(i) - trackedX;
                const float dy = CenterY(i) - trackedY;
                const float distance = std::sqrt(dx * dx + dy * dy);
                if(distance <= matchedDistance)
                {
                    matchedDistance = distance;
                    matched = i;
                }
            }
        }

        count(0) = static_cast<float>(accepted);

        if(matched >= 0)
        {
            // A clearly better face takes over once it has stayed better for switch_delay.
            if(best != matched && Score(best) > switchRatio.as_float() * Score(matched))
            {
                challengeTime = challenged ? challengeTime + static_cast<float>(GetTickDuration()) : 0.0f;
                challenged = true;
                if(challengeTime >= switchDelay.as_float())
                {
                    Publish(best, true);
                    return;
                }
            }
            else
                challenged = false;
            Publish(matched, false);
            return;
        }

        if(tracking && HoldTarget())
            return;

        if(best >= 0)
            Publish(best, true);
    }
};

INSTALL_CLASS(FaceTargetSelector)

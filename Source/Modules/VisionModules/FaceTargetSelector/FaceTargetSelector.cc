#include "ikaros.h"

#include <algorithm>
#include <cmath>

using namespace ikaros;

class FaceTargetSelector : public Module
{
    parameter selection;
    parameter minWidth;
    parameter minConfidence;
    parameter matchDistance;
    parameter switchRatio;
    parameter holdTime;
    parameter smoothing;

    matrix boxes;

    matrix target;
    matrix valid;
    matrix targetBox;
    matrix count;

    matrix boxRow;

    bool tracking = false;
    float trackedX = 0;
    float trackedY = 0;
    float missingTime = 0;

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
        valid(0) = 1;

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
        Bind(holdTime, "hold_time");
        Bind(smoothing, "smoothing");

        Bind(boxes, "BOXES");
        Bind(target, "TARGET");
        Bind(valid, "VALID");
        Bind(targetBox, "TARGET_BOX");
        Bind(count, "COUNT");

        boxRow.realloc(5);
    }


    void
    Tick() override
    {
        valid.reset();
        count.reset();
        targetBox.clear();

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
            if(best != matched && Score(best) > switchRatio.as_float() * Score(matched))
                Publish(best, true);
            else
                Publish(matched, false);
            return;
        }

        if(tracking)
        {
            missingTime += static_cast<float>(GetTickDuration());
            if(missingTime <= holdTime.as_float())
                return;
            tracking = false;
            target.reset();
        }

        if(best >= 0)
            Publish(best, true);
    }
};

INSTALL_CLASS(FaceTargetSelector)

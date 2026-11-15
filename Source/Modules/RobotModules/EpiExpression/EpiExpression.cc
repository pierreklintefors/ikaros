#include "ikaros.h"

#include <algorithm>
#include <cmath>

using namespace ikaros;

class EpiExpression : public Module
{
    matrix idleColor;
    matrix engagedColor;
    matrix mouthColor;
    parameter intensity;
    parameter idlePupil;
    parameter engagedPupil;
    parameter attackRate;
    parameter releaseRate;

    matrix engaged;

    matrix leftEye;
    matrix rightEye;
    matrix mouthHigh;
    matrix mouthLow;
    matrix pupilSize;
    matrix level;

    float currentLevel = 0;

    // Rate parameters are delivered per tick by the kernel.
    static float
    Blend(const parameter & rate)
    {
        const float ratePerTick = rate.as_float();
        return ratePerTick <= 0.0f ? 1.0f : 1.0f - std::exp(-ratePerTick);
    }


    static void
    Fill(matrix & leds, const float rgb[3])
    {
        for(int c = 0; c < 3; ++c)
            for(int i = 0; i < leds.size_x(); ++i)
                leds(c, i) = rgb[c];
    }


    void
    CheckColor(const matrix & color, const std::string & name)
    {
        if(color.size() < 3)
            throw exception("EpiExpression: " + name + " must contain red, green, and blue.", path_);
    }

public:
    void
    Init() override
    {
        Bind(idleColor, "idle_color");
        Bind(engagedColor, "engaged_color");
        Bind(mouthColor, "mouth_color");
        Bind(intensity, "intensity");
        Bind(idlePupil, "idle_pupil");
        Bind(engagedPupil, "engaged_pupil");
        Bind(attackRate, "attack_rate");
        Bind(releaseRate, "release_rate");

        Bind(engaged, "ENGAGED");

        Bind(leftEye, "LEFT_EYE");
        Bind(rightEye, "RIGHT_EYE");
        Bind(mouthHigh, "MOUTH_HIGH");
        Bind(mouthLow, "MOUTH_LOW");
        Bind(pupilSize, "PUPIL_SIZE");
        Bind(level, "LEVEL");

        CheckColor(idleColor, "idle_color");
        CheckColor(engagedColor, "engaged_color");
        CheckColor(mouthColor, "mouth_color");
    }


    void
    Tick() override
    {
        const float goal = engaged(0) > 0.5f ? 1.0f : 0.0f;
        const float blend = goal > currentLevel ? Blend(attackRate) : Blend(releaseRate);
        currentLevel = std::clamp(currentLevel + blend * (goal - currentLevel), 0.0f, 1.0f);
        level(0) = currentLevel;

        const float brightness = std::clamp(intensity.as_float(), 0.0f, 1.0f);
        float eye[3];
        float mouth[3];
        for(int c = 0; c < 3; ++c)
        {
            const float idle = idleColor.data()[c];
            const float active = engagedColor.data()[c];
            eye[c] = std::clamp(brightness * (idle + currentLevel * (active - idle)), 0.0f, 1.0f);
            mouth[c] = std::clamp(brightness * mouthColor.data()[c], 0.0f, 1.0f);
        }

        Fill(leftEye, eye);
        Fill(rightEye, eye);
        Fill(mouthHigh, mouth);
        Fill(mouthLow, mouth);

        const float pupil = idlePupil.as_float() + currentLevel * (engagedPupil.as_float() - idlePupil.as_float());
        pupilSize(0) = pupil;
        pupilSize(1) = pupil;
    }
};

INSTALL_CLASS(EpiExpression)

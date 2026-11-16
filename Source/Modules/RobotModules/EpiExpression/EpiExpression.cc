#include "ikaros.h"

#include <algorithm>
#include <cmath>

using namespace ikaros;

class EpiExpression : public Module
{
    matrix idleColor;
    matrix engagedColor;
    matrix mouthColor;
    matrix mirrorColor;
    matrix mouthIdleColor;
    matrix mouthMirrorColor;
    parameter intensity;
    parameter idlePupil;
    parameter engagedPupil;
    parameter attackRate;
    parameter releaseRate;

    matrix engaged;
    matrix mirroring;

    matrix leftEye;
    matrix rightEye;
    matrix mouthHigh;
    matrix mouthLow;
    matrix pupilSize;
    matrix level;
    matrix mirrorLevel;

    float currentLevel = 0;
    float currentMirrorLevel = 0;

    // Rate parameters are delivered per tick by the kernel.
    static float
    Blend(const parameter & rate)
    {
        const float ratePerTick = rate.as_float();
        return ratePerTick <= 0.0f ? 1.0f : 1.0f - std::exp(-ratePerTick);
    }


    static float
    Mix(float a, float b, float t)
    {
        return a + t * (b - a);
    }


    float
    Step(float current, const matrix & input) const
    {
        const float goal = input(0) > 0.5f ? 1.0f : 0.0f;
        const float blend = goal > current ? Blend(attackRate) : Blend(releaseRate);
        return std::clamp(current + blend * (goal - current), 0.0f, 1.0f);
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
        Bind(mirrorColor, "mirror_color");
        Bind(mouthIdleColor, "mouth_idle_color");
        Bind(mouthMirrorColor, "mouth_mirror_color");
        Bind(intensity, "intensity");
        Bind(idlePupil, "idle_pupil");
        Bind(engagedPupil, "engaged_pupil");
        Bind(attackRate, "attack_rate");
        Bind(releaseRate, "release_rate");

        Bind(engaged, "ENGAGED");
        Bind(mirroring, "MIRRORING");

        Bind(leftEye, "LEFT_EYE");
        Bind(rightEye, "RIGHT_EYE");
        Bind(mouthHigh, "MOUTH_HIGH");
        Bind(mouthLow, "MOUTH_LOW");
        Bind(pupilSize, "PUPIL_SIZE");
        Bind(level, "LEVEL");
        Bind(mirrorLevel, "MIRROR_LEVEL");

        CheckColor(idleColor, "idle_color");
        CheckColor(engagedColor, "engaged_color");
        CheckColor(mouthColor, "mouth_color");
        CheckColor(mirrorColor, "mirror_color");
        CheckColor(mouthIdleColor, "mouth_idle_color");
        CheckColor(mouthMirrorColor, "mouth_mirror_color");
    }


    void
    Tick() override
    {
        currentLevel = Step(currentLevel, engaged);
        level(0) = currentLevel;

        // Without MIRRORING the mouth keeps mouth_color, as before the mirroring state existed.
        const bool showMirroring = mirroring.connected();
        currentMirrorLevel = showMirroring ? Step(currentMirrorLevel, mirroring) : 0.0f;
        mirrorLevel(0) = currentMirrorLevel;

        const float brightness = std::clamp(intensity.as_float(), 0.0f, 1.0f);
        float eye[3];
        float mouth[3];
        for(int c = 0; c < 3; ++c)
        {
            const float eyeColor = Mix(Mix(idleColor.data()[c], engagedColor.data()[c], currentLevel),
                                       mirrorColor.data()[c], currentMirrorLevel);
            float mouthValue = mouthColor.data()[c];
            if(showMirroring)
                mouthValue = Mix(Mix(mouthIdleColor.data()[c], mouthColor.data()[c], currentLevel),
                                 mouthMirrorColor.data()[c], currentMirrorLevel);
            eye[c] = std::clamp(brightness * eyeColor, 0.0f, 1.0f);
            mouth[c] = std::clamp(brightness * mouthValue, 0.0f, 1.0f);
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

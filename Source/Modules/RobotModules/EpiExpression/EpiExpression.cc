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
    parameter pupilMin;
    parameter pupilMax;
    parameter lidSoftness;
    parameter ledFirstAngle;
    parameter ledDirection;

    matrix engaged;
    matrix mirroring;
    matrix blink;
    matrix eyeOverlay;
    matrix mouthOverlay;
    matrix brightnessInput;
    matrix pupilOffset;

    matrix leftEye;
    matrix rightEye;
    matrix mouthHigh;
    matrix mouthLow;
    matrix pupilSize;
    matrix level;
    matrix mirrorLevel;

    float currentLevel = 0;
    float currentMirrorLevel = 0;

    matrix ledHeight; // height of each eye LED on the ring, from -1 (bottom) to 1 (top)

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


    // Blends color toward the r, g, b of an overlay input by its fourth element.
    static void
    Overlay(float rgb[3], const matrix & overlay)
    {
        if(!overlay.connected())
            return;
        const float weight = std::clamp(overlay(3), 0.0f, 1.0f);
        for(int c = 0; c < 3; ++c)
            rgb[c] = Mix(rgb[c], overlay(c), weight);
    }


    // The lid line sweeps from above the top LED to below the bottom LED as closure goes from 0 to 1.
    // LEDs above the line are dark, with a soft edge of lid_softness in ring heights.
    void
    ApplyLid(matrix & leds, float closure) const
    {
        if(closure <= 0.0f)
            return;
        const float softness = std::max(lidSoftness.as_float(), 0.001f);
        const float line = 1.0f + softness - std::clamp(closure, 0.0f, 1.0f) * (2.0f + 2.0f * softness);
        for(int i = 0; i < leds.size_x(); ++i)
        {
            const float visible = std::clamp((line - ledHeight(i)) / softness, 0.0f, 1.0f);
            for(int c = 0; c < 3; ++c)
                leds(c, i) *= visible;
        }
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
        Bind(pupilMin, "pupil_min");
        Bind(pupilMax, "pupil_max");
        Bind(lidSoftness, "lid_softness");
        Bind(ledFirstAngle, "led_first_angle");
        Bind(ledDirection, "led_direction");

        Bind(engaged, "ENGAGED");
        Bind(mirroring, "MIRRORING");
        Bind(blink, "BLINK");
        Bind(eyeOverlay, "EYE_OVERLAY");
        Bind(mouthOverlay, "MOUTH_OVERLAY");
        Bind(brightnessInput, "BRIGHTNESS");
        Bind(pupilOffset, "PUPIL_OFFSET");

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
        if(eyeOverlay.connected() && eyeOverlay.size() < 4)
            throw exception("EpiExpression: EYE_OVERLAY must contain red, green, blue, and weight.", path_);
        if(mouthOverlay.connected() && mouthOverlay.size() < 4)
            throw exception("EpiExpression: MOUTH_OVERLAY must contain red, green, blue, and weight.", path_);

        // Angles are clockwise from the 3 o'clock position as seen from the front, as in the epi-head widget.
        const int ledCount = leftEye.size_x();
        ledHeight.realloc(ledCount);
        for(int i = 0; i < ledCount; ++i)
        {
            const float degrees = ledFirstAngle.as_float() + ledDirection.as_float() * 360.0f * i / ledCount;
            ledHeight(i) = -std::sin(degrees * 3.14159265358979f / 180.0f);
        }
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

        float brightness = std::clamp(intensity.as_float(), 0.0f, 1.0f);
        if(brightnessInput.connected())
            brightness *= std::max(brightnessInput(0), 0.0f);
        float eye[3];
        float mouth[3];
        for(int c = 0; c < 3; ++c)
        {
            eye[c] = Mix(Mix(idleColor.data()[c], engagedColor.data()[c], currentLevel),
                         mirrorColor.data()[c], currentMirrorLevel);
            mouth[c] = mouthColor.data()[c];
            if(showMirroring)
                mouth[c] = Mix(Mix(mouthIdleColor.data()[c], mouthColor.data()[c], currentLevel),
                               mouthMirrorColor.data()[c], currentMirrorLevel);
        }
        Overlay(eye, eyeOverlay);
        Overlay(mouth, mouthOverlay);
        for(int c = 0; c < 3; ++c)
        {
            eye[c] = std::clamp(brightness * eye[c], 0.0f, 1.0f);
            mouth[c] = std::clamp(brightness * mouth[c], 0.0f, 1.0f);
        }

        Fill(leftEye, eye);
        Fill(rightEye, eye);
        Fill(mouthHigh, mouth);
        Fill(mouthLow, mouth);

        if(blink.connected())
        {
            float closure = 0.0f;
            for(int i = 0; i < blink.size(); ++i)
                closure = std::max(closure, blink.data()[i]);
            ApplyLid(leftEye, closure);
            ApplyLid(rightEye, closure);
        }

        float pupil = idlePupil.as_float() + currentLevel * (engagedPupil.as_float() - idlePupil.as_float());
        if(pupilOffset.connected())
            pupil += pupilOffset(0);
        pupil = std::clamp(pupil, pupilMin.as_float(), pupilMax.as_float());
        pupilSize(0) = pupil;
        pupilSize(1) = pupil;
    }
};

INSTALL_CLASS(EpiExpression)

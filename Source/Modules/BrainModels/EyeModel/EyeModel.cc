#include <algorithm>
#include <cmath>

#include "ikaros.h"

using namespace ikaros;


class EyeModel: public Module
{
    parameter pupilMin;
    parameter pupilMax;
    parameter m3;
    parameter alpha1a;
    parameter amplifier;
    parameter timeConstant;

    matrix sphincter;
    matrix dilator;
    matrix pupilDiameter;

    bool started = false;

    static float
    Mean(const matrix & m)
    {
        if(m.size() == 0)
            return 0;
        return m.sum() / static_cast<float>(m.size());
    }


    void
    Init() override
    {
        Bind(pupilMin, "pupil_min");
        Bind(pupilMax, "pupil_max");
        Bind(m3, "m3");
        Bind(alpha1a, "alpha1a");
        Bind(amplifier, "amplifier");
        Bind(timeConstant, "time_constant");

        Bind(sphincter, "PUPIL_SPHINCTER");
        Bind(dilator, "PUPIL_DILATOR");
        Bind(pupilDiameter, "PUPIL_DIAMETER");

        if(pupilMax.as_float() < pupilMin.as_float())
            throw exception("EyeModel pupil_max must be at least pupil_min.");
    }


    void
    Reset() override
    {
        Component::Reset();
        started = false;
    }


    void
    Tick() override
    {
        const float drive = amplifier.as_float() *
                            (alpha1a.as_float() * Mean(dilator) - m3.as_float() * Mean(sphincter));
        const float goal = pupilMin.as_float() +
                           (pupilMax.as_float() - pupilMin.as_float()) * std::clamp(drive, 0.0f, 1.0f);

        const float tc = timeConstant.as_float();
        if(!started || tc <= 0)
            pupilDiameter(0) = goal;
        else
            pupilDiameter(0) += -std::expm1(-GetTickDuration() / tc) * (goal - pupilDiameter(0));
        started = true;
    }
};

INSTALL_CLASS(EyeModel)

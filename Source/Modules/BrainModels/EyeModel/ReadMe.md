# EyeModel

## Description

`EyeModel` is the last stage of a pupil model: the two muscles of the iris. The sphincter, driven by
the parasympathetic pathway through the ciliary ganglion, makes the pupil smaller, and the dilator,
driven by the sympathetic pathway through the superior cervical ganglion, makes it larger. The goal
diameter is

    pupil_min + (pupil_max - pupil_min) * clip(amplifier * (alpha1a * DILATOR - m3 * SPHINCTER), 0, 1)

where `m3` and `alpha1a` are the sensitivities of the muscarinic and alpha-1a receptors of the two
muscles. Raising `m3` mimics a drug that strengthens constriction, and raising `alpha1a` one that
strengthens dilation. The pupil follows its goal with the time constant `time_constant` and starts
at the goal of the first tick, so a model does not begin with a sweep from zero. Inputs with several
elements are averaged.

The module comes from the Ikaros 2 pupil model, where it also passed a gaze direction through for
visualization; that part is not ported.

## Parameters

| Name | Description | Type | Default |
| --- | --- | --- | --- |
| pupil_min | Smallest pupil diameter in mm | number | 2 |
| pupil_max | Largest pupil diameter in mm | number | 8 |
| m3 | Sensitivity of the sphincter (muscarinic M3 receptors) | number | 1 |
| alpha1a | Sensitivity of the dilator (alpha-1a receptors) | number | 1 |
| amplifier | Gain on the difference between dilator and sphincter drive | number | 1 |
| time_constant | Seconds for the pupil to cover about 63 percent of the way to its goal; 0 follows at once | number | 0.3 |

## Inputs

| Name | Description |
| --- | --- |
| PUPIL_SPHINCTER | Constriction drive, for example from the ciliary ganglion |
| PUPIL_DILATOR | Dilation drive, for example from the superior cervical ganglion |

## Outputs

| Name | Description |
| --- | --- |
| PUPIL_DIAMETER | Pupil diameter in mm |

## Example

```xml
<module class="Nucleus" name="CG" />
<module class="Nucleus" name="SCG" />
<module class="EyeModel" name="Eyes" pupil_min="5" pupil_max="16" amplifier="2" />

<connection source="CG.OUTPUT" target="Eyes.PUPIL_SPHINCTER" />
<connection source="SCG.OUTPUT" target="Eyes.PUPIL_DILATOR" />
```

`tests/test_EyeModel.ikg` checks the goal for several drives, the clipping at both ends, and the
smooth approach with a time constant.

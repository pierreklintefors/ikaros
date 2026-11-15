# EpiExpression

## Description

`EpiExpression` gives the Epi robot a simple attentive expression. While `ENGAGED` is 1 the eye
LEDs fade toward `engaged_color` and the pupils open toward `engaged_pupil`. When `ENGAGED` returns
to 0, both fade back to the idle values.

The fade is controlled by one engagement level between 0 and 1, available as `LEVEL`. It rises at
`attack_rate` and falls at `release_rate`. A fast attack and a slow release make the robot react
quickly when it finds someone, while short detection dropouts do not make the eyes flicker.

The LED outputs have the shapes that `FadeCandy` expects: three rows of red, green, and blue, with
12 LEDs per eye and 8 per mouth half. All LEDs of a group get the same color, scaled by `intensity`.
The mouth uses `mouth_color` and does not change with engagement.

`PUPIL_SIZE` is meant for `GazeLock.PUPIL_SIZE`, which places it in the servo goal vector.
`EpiServos` clips pupil goals to 5..16 mm.

`attack_rate` and `release_rate` are Ikaros `rate` parameters given per second. A rate of 0 changes
the expression immediately.

## Parameters

| Name | Description | Type | Default |
| --- | --- | --- | --- |
| idle_color | Eye LED red, green, blue in 0..1 when idle | matrix | 0.8, 0.7, 0.1 |
| engaged_color | Eye LED red, green, blue in 0..1 when engaged | matrix | 0.1, 0.6, 1.0 |
| mouth_color | Mouth LED red, green, blue in 0..1 | matrix | 0.8, 0.7, 0.1 |
| intensity | Brightness factor applied to all LEDs | number | 0.6 |
| idle_pupil | Pupil size in mm when idle | number | 9 |
| engaged_pupil | Pupil size in mm when engaged | number | 15 |
| attack_rate | Rate in 1/s toward engaged; 0 changes immediately | rate | 4 |
| release_rate | Rate in 1/s back to idle; 0 changes immediately | rate | 1 |

## Inputs

| Name | Description | Optional |
| --- | --- | --- |
| ENGAGED | 1 while the robot attends to someone, otherwise 0 | |

## Outputs

| Name | Description |
| --- | --- |
| LEFT_EYE | Left eye LED colors as rows of red, green, blue |
| RIGHT_EYE | Right eye LED colors as rows of red, green, blue |
| MOUTH_HIGH | Upper mouth LED colors as rows of red, green, blue |
| MOUTH_LOW | Lower mouth LED colors as rows of red, green, blue |
| PUPIL_SIZE | Left and right pupil goals in mm |
| LEVEL | Smoothed engagement level from 0 (idle) to 1 (engaged) |

## Example

```xml
<module class="EpiExpression" name="Expression" />
<module class="FadeCandy" name="LEDs" />

<connection source="Gaze.LOCKED" target="Expression.ENGAGED" />
<connection source="Expression.PUPIL_SIZE" target="Gaze.PUPIL_SIZE" />
<connection source="Expression.LEFT_EYE" target="LEDs.LEFT_EYE" />
<connection source="Expression.RIGHT_EYE" target="LEDs.RIGHT_EYE" />
<connection source="Expression.MOUTH_HIGH" target="LEDs.MOUTH_HIGH" />
<connection source="Expression.MOUTH_LOW" target="LEDs.MOUTH_LOW" />
```

The test in `tests/` steps `ENGAGED` on and off and checks the level, pupil sizes, and that a
simulated `FadeCandy` accepts the LED outputs.

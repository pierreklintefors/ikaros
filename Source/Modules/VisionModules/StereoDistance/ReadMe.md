# StereoDistance

## Description

`StereoDistance` measures how far away a target is when a robot sees it with two cameras, one in each
eye, for example the eye cameras of an Epi robot. It needs no knowledge of the target's size, so it
works for small and large faces alike.

The direction from each eye to the target is the eye's pan angle, from `EYE_PAN`, plus the target's
angle in that eye's camera image:

```text
angle = eye pan + x_sign * atan(x * tan(horizontal_fov / 2))
```

With the cameras `baseline` meters apart and the angles counted positive toward the left eye, the two
directions meet at the target, whose distance in front of the eyes is

```text
distance = baseline / (tan(right angle) - tan(left angle))
```

Because the eye angles are part of the calculation, the distance stays correct while the eyes
converge, dart, or follow the target. `left_sign` turns the sign around for robots whose positive pan
turns the eyes toward the right eye, and `position_offset` is subtracted from `EYE_PAN`, for example
180 for Epi servo positions.

A measurement is made only while both cameras see a target (`LEFT_VALID` and `RIGHT_VALID`) at similar
heights, within `max_height_difference`, so that two different faces are not combined. Results
outside `min_distance` to `max_distance` are ignored, because small angle differences far away are
unreliable. `DISTANCE` follows new measurements at `smoothing` and stays valid for `hold_time`
seconds after the last one.

The two cameras usually deliver their frames at slightly different times; the smoothing evens out
the small errors this causes while the target moves.

## Parameters

| Name | Description | Type | Default |
| --- | --- | --- | --- |
| baseline | Distance between the two cameras in meters; 0.058 matches Epi | number | 0.058 |
| horizontal_fov | Horizontal field of view of both cameras in degrees | number | 62.2 |
| x_sign | Sign that maps image x to an angle in the same direction as the eye pan, as in GazeLock | number | 1 |
| left_sign | 1 when positive angles turn toward the left eye, -1 when toward the right eye | number | 1 |
| position_offset | Value subtracted from EYE_PAN | number | 0 |
| min_distance | Shorter measurements are ignored, in meters | number | 0.05 |
| max_distance | Longer measurements are ignored, in meters | number | 3 |
| max_height_difference | Largest difference of the target heights in the two images | number | 0.3 |
| smoothing | Rate in 1/s at which DISTANCE follows new measurements; 0 follows immediately | rate | 6 |
| hold_time | Seconds DISTANCE stays valid after the last measurement | number | 0.3 |

## Inputs

| Name | Description | Optional |
| --- | --- | --- |
| LEFT_TARGET | Target x, y, and optionally width in centered image coordinates of the left camera | |
| LEFT_VALID | 1 when LEFT_TARGET is a current detection | |
| RIGHT_TARGET | Target x, y, and optionally width in centered image coordinates of the right camera | |
| RIGHT_VALID | 1 when RIGHT_TARGET is a current detection | |
| EYE_PAN | Pan of the left and the right eye in degrees, minus position_offset | |

## Outputs

| Name | Description |
| --- | --- |
| DISTANCE | Distance of the target in front of the eyes in meters; 0 when not valid |
| VALID | 1 while a measurement is less than hold_time old |

## Example

```xml
<module class="StereoDistance" name="Stereo" position_offset="180" x_sign="-1" />

<connection source="LeftSelector.TARGET" target="Stereo.LEFT_TARGET" />
<connection source="LeftSelector.VALID" target="Stereo.LEFT_VALID" />
<connection source="RightSelector.TARGET" target="Stereo.RIGHT_TARGET" />
<connection source="RightSelector.VALID" target="Stereo.RIGHT_VALID" />
<connection source="Servos.PRESENT_POSITION[2:4]" target="Stereo.EYE_PAN" delay="1" />
```

The test in `tests/` triangulates a target straight ahead with parallel and converged eyes and one
off to the side, checks the hold after a camera loses the target, and rejects detections at
different heights.

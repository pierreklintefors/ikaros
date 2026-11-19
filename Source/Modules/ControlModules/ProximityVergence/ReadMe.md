# ProximityVergence

## Description

`ProximityVergence` makes a robot with two moving eyes converge on things that come close, the way
people's eyes cross a little when someone leans in to their face. It needs no depth sensor: it
estimates the distance of a target of known size, such as a face, from the width of its box in the
image.

With the target's image width `w` in centered coordinates, where the whole image is 2 wide, and the
horizontal field of view `fov`, the target's angular half width and distance are

```text
half angle = atan(w / 2 * tan(fov / 2))
distance   = object_width / 2 / tan(half angle)
```

The eyes of a robot with eye spacing `eye_baseline` meet at that distance when they are turned
inward by

```text
geometric vergence = 2 * atan(eye_baseline / 2 / distance)
```

When `MEASURED_DISTANCE` is connected, for example to `StereoDistance.DISTANCE`, a valid measured distance is
used instead of the estimate from the target's size, which depends on the assumed `object_width`
and is wrong for unusually small or large faces. The measurement also keeps working when the target
box of this camera is lost for a moment. Without a valid measurement the size estimate is used, and
`MEASURED` shows which one is in use.

For people further away this angle is small, and real eyes would hardly move. To keep the eyes
parallel for people at a normal distance and make the convergence clearly visible for someone who
comes really close, the vergence fades in smoothly between `start_distance` and `full_distance`.
`gain` can exaggerate it, and `max_vergence` limits it. `NEAR` rises from 0 to 1 over the same
range; in people, convergence comes with a constriction of the pupils, and `NEAR` can drive that.

When `VALID` is 0, the last vergence is kept for `hold_time`, so a missed detection does not make
the eyes jump, and then returns to 0. The output follows its goal at `rate`.

`VERGENCE` is the inward eye rotation that `GazeLock.VERGENCE` expects. The camera's field of view
must match the camera that produced `TARGET`.

## Parameters

| Name | Description | Type | Default |
| --- | --- | --- | --- |
| object_width | Real width of the target in meters; 0.15 is a typical face box | number | 0.15 |
| eye_baseline | Distance between the two eyes in meters; 0.058 matches Epi | number | 0.058 |
| horizontal_fov | Horizontal camera field of view in degrees | number | 62.2 |
| start_distance | Distance in meters at which convergence starts to fade in | number | 0.8 |
| full_distance | Distance in meters at which the full convergence is reached | number | 0.35 |
| gain | Factor on the geometric vergence | number | 1 |
| max_vergence | Largest vergence in degrees | number | 20 |
| rate | Rate in 1/s at which the vergence follows its goal; 0 follows immediately | rate | 4 |
| hold_time | Seconds the last vergence is kept after VALID has become 0 | number | 0.5 |

## Inputs

| Name | Description | Optional |
| --- | --- | --- |
| TARGET | Target center x, y, and width in centered image coordinates, such as FaceTargetSelector.TARGET | |
| VALID | 1 when TARGET is a current detection; when not connected, a width above 0 counts as valid | yes |
| MEASURED_DISTANCE | Measured distance of the target in meters, used instead of the size estimate while it is valid | yes |
| MEASURED_VALID | 1 while MEASURED_DISTANCE is a current measurement; when not connected, a distance above 0 counts as valid | yes |

## Outputs

| Name | Description |
| --- | --- |
| VERGENCE | Inward eye rotation in degrees, for example for GazeLock.VERGENCE |
| DISTANCE | Estimated target distance in meters; 0 when no target is seen |
| NEAR | How close the target is, from 0 at start_distance to 1 at full_distance |
| MEASURED | 1 when the distance comes from MEASURED_DISTANCE, 0 when it is estimated from the target's size |

## Example

```xml
<module class="ProximityVergence" name="Near" horizontal_fov="62.2" gain="1.5" />

<connection source="Selector.TARGET" target="Near.TARGET" />
<connection source="Selector.VALID" target="Near.VALID" />
<connection source="Near.VERGENCE" target="Gaze.VERGENCE" />
```

The tests in `tests/` check the distance, the fade-in, the limit, and the hold and release, and that a
measured distance replaces the size estimate while it is valid.

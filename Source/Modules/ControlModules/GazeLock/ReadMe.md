# GazeLock

## Description

`GazeLock` keeps the gaze of a robot with a pan-tilt neck and panning eyes locked on a target. It
was written for the Epi torso robots and produces the complete `GOAL_POSITION` vector for
`EpiServos`.

The module stores where the target is in the body frame, as an elevation and an azimuth in degrees.
Every tick the eyes are pointed at that direction from the present head position:

```text
eye pan = target azimuth - present neck pan
```

The eyes therefore stay on the target however the head moves. The head may be moved by GazeLock
itself, by `MANUAL_HEAD`, or by anything else that changes the servo positions, because the
compensation uses `PRESENT_POSITION`.

### Target updates

When `VALID` is 1, `TARGET` holds the target's image position in the centered coordinates used by
`AppleVisionFaceDetector` and `FaceTargetSelector`. The image offset is converted to angles with
the camera field of view:

```text
angle = sign * atan(coordinate * tan(field_of_view / 2))
```

The angles are added to the gaze direction at the time the image was captured. That direction is
taken from a short history of the head and eye positions, `latency` seconds back, so stream and
detector delays do not make the robot overshoot. New measurements are blended into the stored
direction at `update_rate`.

`aim_x` and `aim_y` set where in the image the target should end up. With the default 0 the gaze
centers the target. A positive `aim_y` keeps the target below the image center, so the robot looks
higher than the center of the face box, for example at a person's eyes instead of at the nose.
The image angle of the aim point is subtracted from each measurement, so a target that is already
at the aim point leaves the stored direction unchanged.

`x_sign` and `y_sign` must match the robot's mounting and servo directions. If the robot turns away
from a face instead of toward it, invert the corresponding sign.

### Losing the target

After `VALID` has been 0 for `hold_time` seconds, `LOCKED` becomes 0. The eyes then return to
straight ahead at `release_rate`, and in follow mode the stored direction and thus the head return
to straight ahead as well. A rate of 0 makes these returns immediate.

### Head control

With `FOLLOW` at 1, or unconnected, the head turns toward the stored target direction. It starts
moving when the target is more than `head_dead_zone` degrees away and stops when it is within a
quarter of that. The eyes cover the remaining error, so small target movements are followed by the
eyes alone and the head moves in calm, larger steps. The head speed is `head_gain` times the error,
limited to `head_max_speed`.

With `FOLLOW` at 0 the head moves toward `MANUAL_HEAD` at `head_max_speed`. If `MANUAL_HEAD` is not
connected, the head stays where it is. Switching between the modes never makes the head jump.

All goals are clipped to the configured limits. The defaults lie inside the `EpiServos` default
position limits, and `eye_max` is 13 degrees because that is the right eye's default limit.

### Startup

While any of the first four `PRESENT_POSITION` values is more than 90 degrees from
`position_offset`, the module treats the feedback as unavailable and uses its own goals instead. This
covers the first ticks after startup, when servo feedback is still zero.

### Servo order and units

`GOAL_POSITION` uses the Epi order: neck tilt, neck pan, left eye, right eye, left pupil, right
pupil, followed by the remaining joints, which are set to straight ahead. Angles are in servo
degrees with `position_offset` meaning straight ahead. Pupils are in mm, from `PUPIL_SIZE` or
`pupil_size`. `HEAD_GOAL`, `EYE_GOAL`, and `TARGET_DIRECTION` are centered on zero.

`update_rate`, `release_rate`, `head_gain`, and `head_max_speed` are Ikaros `rate` parameters given
per second. The kernel converts them to per-tick values, so behavior does not depend on
`tick_duration`.

## Commands

| Name | Description |
| --- | --- |
| lock | Store the present gaze direction as the target and keep it until `release`, without a timeout. Useful for demonstrating eye compensation without vision. |
| release | Drop the lock. |

## Parameters

| Name | Description | Type | Default |
| --- | --- | --- | --- |
| servo_count | Size of GOAL_POSITION; 19 matches EpiServos | number | 19 |
| position_offset | Servo position that corresponds to straight ahead, in degrees | number | 180 |
| pupil_size | Pupil goal in mm when PUPIL_SIZE is not connected | number | 12 |
| horizontal_fov | Horizontal camera field of view in degrees | number | 62.2 |
| vertical_fov | Vertical camera field of view in degrees | number | 48.8 |
| x_sign | Sign that maps image x (right positive) to pan angle | number | 1 |
| y_sign | Sign that maps image y (down positive) to tilt angle | number | -1 |
| aim_x | Image x where the target should appear; 0 is the image center | number | 0 |
| aim_y | Image y where the target should appear; positive keeps the target below the center | number | 0 |
| update_rate | Rate in 1/s at which new measurements update the stored target direction; 0 replaces it directly | rate | 8 |
| latency | Seconds between image capture and detection | number | 0.1 |
| hold_time | Seconds the lock is kept after the target was last seen | number | 1.5 |
| release_rate | Rate in 1/s at which the eyes and, in follow mode, the target return to straight ahead when unlocked | rate | 0.5 |
| head_gain | Head speed per degree of error in 1/s when following the target | rate | 2 |
| head_dead_zone | Head starts moving when the target is this many degrees away and stops at a quarter of it | number | 6 |
| head_max_speed | Maximum head goal speed in degrees per second | rate | 60 |
| tilt_min | Minimum neck tilt goal, centered degrees | number | -30 |
| tilt_max | Maximum neck tilt goal, centered degrees | number | 30 |
| pan_min | Minimum neck pan goal, centered degrees | number | -45 |
| pan_max | Maximum neck pan goal, centered degrees | number | 45 |
| eye_min | Minimum eye pan goal, centered degrees | number | -18 |
| eye_max | Maximum eye pan goal, centered degrees | number | 13 |
| vergence | Inward eye rotation in degrees, split between the eyes | number | 0 |

The default field of view matches the Raspberry Pi Camera Module v2.

## Inputs

| Name | Description | Optional |
| --- | --- | --- |
| PRESENT_POSITION | Present servo positions in the same order and units as GOAL_POSITION | |
| TARGET | Target center x, y in centered image coordinates | yes |
| VALID | 1 when TARGET is a new detection | yes |
| FOLLOW | 1 turns the head toward the target, 0 uses MANUAL_HEAD; follows when not connected | yes |
| MANUAL_HEAD | Manual head goal tilt, pan in centered degrees | yes |
| PUPIL_SIZE | Pupil goals in mm for the left and right pupil | yes |

## Outputs

| Name | Description |
| --- | --- |
| GOAL_POSITION | Servo goal vector for EpiServos |
| HEAD_GOAL | Head goal tilt, pan in centered degrees |
| EYE_GOAL | Left and right eye pan goals in centered degrees |
| TARGET_DIRECTION | Stored target elevation and azimuth in the body frame, degrees |
| LOCKED | 1 while the gaze is locked on a target |

## Example

```xml
<module class="FaceTargetSelector" name="Selector" />
<module class="GazeLock" name="Gaze" />
<module class="EpiServos" name="Servos" robot="EpiBlack" />

<connection source="Selector.TARGET" target="Gaze.TARGET" />
<connection source="Selector.VALID" target="Gaze.VALID" />
<connection source="Gaze.GOAL_POSITION" target="Servos.GOAL_POSITION" />
<connection source="Servos.PRESENT_POSITION" target="Gaze.PRESENT_POSITION" />
```

The tests in `tests/` feed `GOAL_POSITION` back as `PRESENT_POSITION` to model ideal servos. They
check eye compensation during a manual head move, release after `hold_time`, the head dead zone,
eye clipping, the pupil input, and the aim point.

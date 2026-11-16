# HeadMirror

## Description

`HeadMirror` makes a robot with a pan-tilt head copy the head movements of the person it is looking
at. It is built to work with `FaceTargetSelector`, which provides the pose of the followed face, and
`GazeLock`, which keeps the eyes on the person and moves the head.

The head goal is the direction of the person plus a mirrored offset:

```text
pan  = azimuth   + weight * clamp(yaw_sign   * yaw_gain   * yaw,   ±max_pan)
tilt = elevation + weight * clamp(pitch_sign * pitch_gain * pitch, ±max_tilt)
```

`DIRECTION` is the stored direction of the person, usually `GazeLock.TARGET_DIRECTION`, and the yaw
and pitch come from `POSE`. Because `GazeLock` points the eyes at the person from the present head
position, the robot keeps looking at the person while its head turns and nods with them. The
default `yaw_sign` of -1 gives a mirror image: when the person turns to their left, the robot turns
to its right. Set `yaw_sign="1"` for an anatomical copy. Confirm both signs on each robot, since they
depend on how the camera and the servos are mounted.

With an upright camera image, the correct `pitch_sign` equals `GazeLock`'s `y_sign`: both depend only
on whether a larger tilt angle looks up or down. A model can therefore set `pitch_sign="@y_sign"`. A
wrong `pitch_sign` is easy to notice: the measured pitch changes when the camera tilts, and with the
wrong sign that change pushes the head further, so the tilt runs to `max_tilt` and stays there.

Keep `max_pan` within the range the eyes can compensate. A camera in a turning eye otherwise loses
the face when the head turns away. `max_tilt` should keep the face inside the vertical field of view.

## States

| `STATE` | Meaning | `FOLLOW` |
| --- | --- | --- |
| 0 | No person: nothing valid for `lost_time`, or `LOCKED` is 0 | 1 |
| 1 | A person is tracked but not mirrored yet | 1 |
| 2 | Mirroring | 0 |

Mirroring starts when the same person has been tracked for `engage_time` seconds with a face at
least `min_width` wide and a yaw within `facing_limit`. These checks skip people far away and people
who only run past. `WEIGHT` then fades in at the `fade_in` rate.

When `NEW_TARGET` reports another person, when the person is lost, or when `ENABLE` turns 0,
mirroring stops and `WEIGHT` fades out at the `fade_out` rate. The last offset is kept during the fade,
so the head glides from the mirrored pose back to the person's direction. `FOLLOW` stays 0 until the
fade has almost ended, so `GazeLock` keeps taking its head goal from `HEAD_GOAL` until it can follow
the new person by itself. The new person must then pass the start checks again, and mirroring
starts only after the fade-out has ended, so the old and new offsets are never mixed.

The pose is low-pass filtered with `pose_smoothing` and kept through ticks without a new detection.

`tilt_dead_zone` and `pan_dead_zone` turn many small corrections into fewer, larger movements. While
`HEAD_GOAL` drives the head, each axis keeps its goal until the wanted position is more than the
dead zone away and then moves there in one step, which `GazeLock` turns into one smooth movement.
This helps a joint with play in its gears, where small back-and-forth corrections cause jitter. With
the default of 0 the goal changes on every tick.

`view_compensation` subtracts the angle of the face from the image center, computed from `TARGET`
and the fields of view, from the measured yaw and pitch. Use it if the measured pose changes when
only the robot's head moves. Its sign depends on the camera and must be measured.

## Parameters

| Name | Description | Type | Default |
| --- | --- | --- | --- |
| yaw_gain | Head pan per degree of the person's yaw | number | 0.5 |
| pitch_gain | Head tilt per degree of the person's pitch | number | 0.5 |
| yaw_sign | Sign from yaw to pan; -1 mirror image, 1 anatomical copy | number | -1 |
| pitch_sign | Sign from pitch (positive nodding down) to tilt (positive looking up) | number | -1 |
| max_pan | Largest pan offset in degrees | number | 10 |
| max_tilt | Largest tilt offset in degrees | number | 10 |
| pose_smoothing | Pose filter rate in 1/s; 0 follows immediately | rate | 6 |
| tilt_dead_zone | While mirroring, smallest tilt goal change in degrees; 0 updates every tick | number | 0 |
| pan_dead_zone | While mirroring, smallest pan goal change in degrees; 0 updates every tick | number | 0 |
| engage_time | Seconds before mirroring starts | number | 1 |
| min_width | Smallest face width from `TARGET` for mirroring to start; 0 disables | number | 0 |
| facing_limit | Largest yaw in degrees for mirroring to start | number | 35 |
| lost_time | Seconds without a valid detection before the person is lost | number | 0.5 |
| fade_in | Fade-in rate in 1/s; 0 is immediate | rate | 2 |
| fade_out | Fade-out rate in 1/s; 0 is immediate | rate | 4 |
| view_compensation | Factor and sign for subtracting the face's image angle from the pose | number | 0 |
| horizontal_fov | Horizontal camera field of view in degrees | number | 62.2 |
| vertical_fov | Vertical camera field of view in degrees | number | 48.8 |

## Inputs

| Name | Description | Optional |
| --- | --- | --- |
| POSE | Yaw, pitch, and optionally roll of the person in degrees | |
| VALID | 1 when `POSE` comes from a new detection | |
| DIRECTION | Direction of the person as elevation, azimuth in degrees | |
| NEW_TARGET | 1 when another person has been selected | yes |
| TARGET | Face center x, y and width in centered image coordinates | yes |
| LOCKED | 1 while the gaze is locked on the person | yes |
| ENABLE | 1 allows mirroring; enabled when not connected | yes |

## Outputs

| Name | Description |
| --- | --- |
| HEAD_GOAL | Head goal tilt, pan in degrees for `GazeLock.MANUAL_HEAD` |
| FOLLOW | 1 when `GazeLock` should follow the person itself, for `GazeLock.FOLLOW` |
| MIRRORING | 1 while mirroring, for example for `EpiExpression.MIRRORING` |
| WEIGHT | Mirroring fade level from 0 to 1 |
| OFFSET | Applied head offset tilt, pan in degrees |
| STATE | 0 no person, 1 tracked, 2 mirroring |

## Example

```xml
<module class="FaceTargetSelector" name="Selector" />
<module class="GazeLock" name="Gaze" />
<module class="HeadMirror" name="Mirror" min_width="0.12" />

<connection source="Selector.TARGET_POSE" target="Mirror.POSE" />
<connection source="Selector.VALID" target="Mirror.VALID" />
<connection source="Selector.NEW_TARGET" target="Mirror.NEW_TARGET" />
<connection source="Selector.TARGET" target="Mirror.TARGET" />
<connection source="Gaze.TARGET_DIRECTION" target="Mirror.DIRECTION" delay="1" />
<connection source="Gaze.LOCKED" target="Mirror.LOCKED" delay="1" />
<connection source="Mirror.HEAD_GOAL" target="Gaze.MANUAL_HEAD" delay="0" />
<connection source="Mirror.FOLLOW" target="Gaze.FOLLOW" delay="0" />
```

`DIRECTION` and `LOCKED` need a delay of 1 because `GazeLock` depends on `HeadMirror` in the same tick.
`Projects/Epi/EpiHeadMirroring.ikg` shows the complete model. The tests in `tests/` check the start
conditions, gains and limits, the handover to a new person, loss of the person, the fades, and that
a new person is mirrored only after the previous fade-out, and that the dead zone holds the tilt goal
until it would move by more than the dead zone.

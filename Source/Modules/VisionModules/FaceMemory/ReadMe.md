# FaceMemory

## Description

`FaceMemory` remembers where the faces around a robot are and chooses whom it looks at next. It works
together with `FaceTargetSelector`, which follows one face in the image, and `GazeLock`, which turns
the eyes and head toward it.

Each face is stored by its direction in the body frame, elevation and azimuth in degrees: the camera's
direction when the image was taken (`CAMERA`, from `GazeLock.CAMERA_DIRECTION`) plus the face's angle
in the image. A face therefore keeps its place in memory while the head and eyes turn, and a person
who leaves the view and comes back is recognised by where they are. Faces are matched to memory
closest pairs first, faces seen within the last second before older ones, so a person in view keeps
the same memory. A remembered face moves halfway toward each new measurement. A face seen in only one
frame is dropped when it is missing from the next, and a face not seen for `forget_time` is forgotten.

Measurements are taken only on new detector frames and only once the camera has been slower than
`max_camera_speed` for `still_time`. The camera direction and the image are not taken at exactly the
same moment, so a frame taken while the camera moves would place the faces in a wrong direction.

From the selector's `TARGET`, `FaceMemory` knows which face is followed and for how long (`DWELL`).
Once the robot has looked at one face for `min_dwell` seconds, `PREFERRED` asks the selector to turn to
a face in view that it has never looked at, the largest first. Otherwise, after a random time between
`min_dwell` and `max_dwell`, it turns to another face in view, drawn at random with a weight that grows
with the time since that face was last looked at. So the robot looks around a group rather than going
back and forth. Faces seen for less than `min_seen` are not chosen, and nothing changes while `BUSY` is
1, for example while an animation plays; that time does not count toward the dwell.

Only `FaceMemory` decides when the robot changes face. A face detector often misses a face for a
second, and the selector then takes the largest face in view, which with people at the same distance
is a matter of chance. When the selector follows another face than the one `FaceMemory` chose, and the
chosen face is seen again within `return_time`, `PREFERRED` turns the robot back to it, once a second
at most, and the other face does not count as looked at. A switch is kept when the dwell on the chosen
face was over, when the other face is twice as wide or more (someone much closer), or when the chosen
face stays out of sight for `return_time`.

`CALL` lets someone ask for the robot's attention, for example by waving: when its flag rises, its
image position is turned into a direction, and on the next measured frame, within a second, the robot
turns to the face in view nearest that direction within `call_radius`, also while `BUSY`. If that is
the face looked at already, the robot stays with it for a new dwell. The face counts as looked at,
so `NEW_FACE` does not mark it; the reaction to the call is left to the module that made it.

`NEW_FACE` is 1 for one tick when the robot looks at a face it has never looked at, or has not looked
at for `regreet_time` seconds. It is judged once the look has lasted `min_seen` with the camera
settled. A face within half `match_radius` of a face looked at recently counts as known, because it is
most likely the same person stored twice. A look cut short before the judgement does not count.

`FACES` lists the remembered faces, one row each: elevation, azimuth, width, seconds since seen, and
seconds since looked at (-1 if never). Unused rows are zero, and `COUNT` gives the number of faces.

Faces are told apart only by where they are. Two people who cross paths can swap places in memory,
and someone who steps into the spot where another person stood recently is taken for that person.

## Parameters

| Name | Description | Type | Default |
| --- | --- | --- | --- |
| horizontal_fov | Horizontal camera field of view in degrees, as in `GazeLock` | number | 62.2 |
| vertical_fov | Vertical camera field of view in degrees, as in `GazeLock` | number | 48.8 |
| x_sign | Sign that maps image x to azimuth, as in `GazeLock` | number | 1 |
| y_sign | Sign that maps image y to elevation, as in `GazeLock` | number | -1 |
| min_width | Ignore faces narrower than this; use the selector's value | number | 0.08 |
| min_confidence | Ignore faces with a lower detector confidence; use the selector's value | number | 0.5 |
| capacity | Largest number of faces remembered | number | 16 |
| match_radius | Degrees a face may move between measured frames and still count as the same face | number | 10 |
| max_camera_speed | Degrees per second above which the camera counts as moving | number | 40 |
| still_time | Seconds the camera must have been still before faces are measured | number | 0.25 |
| forget_time | Seconds after which a face that has not been seen is forgotten | number | 120 |
| min_seen | Seconds a face must be known before the robot turns to it, and a look must last before it is judged new | number | 0.5 |
| min_dwell | Shortest time in seconds on one face before turning to another | number | 3 |
| max_dwell | Longest time in seconds on one face while others are in view | number | 8 |
| regreet_time | Seconds without a look after which a face counts as new again | number | 60 |
| call_radius | Degrees from a `CALL` within which the calling person's face is looked for | number | 20 |
| return_time | Seconds after a switch the selector made by itself during which the robot turns back to the face it chose; 0 keeps every switch | number | 2 |
| seed | Random seed; negative selects nondeterministic seeding | number | -1 |

## Inputs

| Name | Description | Optional |
| --- | --- | --- |
| BOXES | Face box rows: x, y, width, height, confidence in centered image coordinates | |
| CAMERA | Elevation and azimuth of the camera's image center when the image was taken, for example `GazeLock.CAMERA_DIRECTION` | |
| TARGET | The followed face: x, y, width, with width 0 when none, for example `FaceTargetSelector.TARGET` | |
| BUSY | 1 while the robot does something else; it does not turn to another face | yes |
| CALL | x, y in centered image coordinates of someone asking for attention, such as a waving hand, and a flag; a rising flag turns the robot to the face nearest that place | yes |

## Outputs

| Name | Description |
| --- | --- |
| PREFERRED | x, y of the face to turn to and 1 on the tick to turn, for `FaceTargetSelector.PREFERRED` |
| NEW_FACE | 1 for one tick when a face looked at is judged new |
| DWELL | Seconds on the present face, not counting `BUSY` time |
| COUNT | Number of faces remembered |
| FACES | One row per remembered face: elevation, azimuth, width, seconds since seen, seconds since looked at |

## Example

```xml
<module class="AppleVisionFaceDetector" name="Faces" />
<module class="FaceMemory" name="Memory" />
<module class="FaceTargetSelector" name="Target" />
<module class="GazeLock" name="Gaze" />

<connection source="Faces.BOXES" target="Memory.BOXES" />
<connection source="Gaze.CAMERA_DIRECTION" target="Memory.CAMERA" delay="1" />
<connection source="Target.TARGET" target="Memory.TARGET" delay="1" />
<connection source="Faces.BOXES" target="Target.BOXES" />
<connection source="Memory.PREFERRED" target="Target.PREFERRED" />
<connection source="Target.TARGET" target="Gaze.TARGET" />
<connection source="Target.VALID" target="Gaze.VALID" />
```

`GazeLock` and `FaceMemory` must use the same field of view and signs; in a group, set them once as
group attributes. The tests in `tests/` check turning to new and known faces, `BUSY`, memory while the
camera turns, a face seen during a fast camera move, `regreet_time`, `forget_time`, turning back
after the selector has switched by itself, and `CALL`.

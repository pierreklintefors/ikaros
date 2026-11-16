# FaceTargetSelector

## Description

`FaceTargetSelector` turns the per-frame face boxes from `AppleVisionFaceDetector` or
`DlibFaceDetector` into one stable target. It is meant for robots that should follow a single person
when several people are in view.

Each tick the module:

1. Drops faces narrower than `min_width` or with a confidence below `min_confidence`. Small faces are
   usually people far away in the background.
2. Looks for the currently tracked face by position. The face whose center moved least since the
   previous tick, and at most `match_distance`, is treated as the same face. The detector's row order
   therefore does not matter.
3. Keeps that face unless another face scores more than `switch_ratio` times higher. With
   `selection="largest"` the score is the face width, so a person who comes clearly closer takes over.
   With `selection="center"` the score grows as the face approaches the image center.
   With `switch_delay` above 0, the better face must stay better for that many seconds first, so
   someone who only passes close by, such as a running child, does not take over.
4. If the tracked face is missing, keeps the last `TARGET` for `hold_time` seconds with `VALID` set to
   0. Short detection dropouts therefore do not make the robot jump to someone else. After the hold
   time, the best remaining face is selected, or `TARGET` becomes zero if there is none.

A detector that runs on the same camera frame twice produces exactly the same boxes. With
`ignore_repeated` set, identical non-empty output from one tick to the next is therefore treated as a
repeated frame: `VALID` stays 0, the target and its box are kept, and the time counts toward
`hold_time`. This lets the model tick at a steady rate without waiting for each camera frame, and a
stalled stream releases the target after `hold_time` instead of reporting the same face forever.

`TARGET` can be smoothed with the `smoothing` rate. The smoothing is reset whenever a different face
is selected, so the target jumps directly to the new person instead of sliding across the image.

`NEW_TARGET` is 1 on the tick a different face is selected. Modules that react to the person, such
as `HeadMirror`, use it to fade out and start again with the new person.

When `POSES` is connected, for example to `AppleVisionFaceDetector.POSES`, `TARGET_POSE` contains
the pose row of the selected face. It is smoothed, held, and reset together with `TARGET`.

All coordinates use the detector's centered image scale: `-1,-1` is the upper-left corner, `0,0` the
image center, and a full image width is `2`.

## Parameters

| Name | Description | Type | Default |
| --- | --- | --- | --- |
| selection | How a new target is chosen: `largest` (closest person) or `center` | number (options) | largest |
| min_width | Ignore faces narrower than this | number | 0.08 |
| min_confidence | Ignore faces with a lower detector confidence | number | 0.5 |
| match_distance | Maximum change in face center between ticks for the same face | number | 0.25 |
| switch_ratio | Score ratio another face must exceed to take over | number | 1.3 |
| switch_delay | Seconds another face must stay better by `switch_ratio` before it takes over; 0 switches at once | number | 0 |
| hold_time | Seconds the tracked face may be missing before another face can be selected | number | 0.5 |
| smoothing | Exponential smoothing rate for `TARGET` in 1/s; 0 disables smoothing | rate | 0 |
| ignore_repeated | Treat identical detector output on consecutive ticks as a repeated camera frame | bool | true |

## Inputs

| Name | Description | Optional |
| --- | --- | --- |
| BOXES | Face box rows: x, y, width, height, confidence in centered image coordinates | |
| POSES | Head pose rows aligned with `BOXES`: yaw, pitch, roll in degrees | yes |

## Outputs

| Name | Description |
| --- | --- |
| TARGET | Selected face center_x, center_y, width; held during hold_time and zero when no face is tracked |
| VALID | 1 on ticks where the tracked face was detected, otherwise 0 |
| TARGET_BOX | Selected face box row for a `target-boxes` widget; empty when no face is tracked |
| TARGET_POSE | Pose row of the selected face from `POSES`; held during hold_time and zero when no face is tracked |
| NEW_TARGET | 1 on the tick a different face is selected, otherwise 0 |
| COUNT | Number of faces that passed the width and confidence filters |

## Example

```xml
<module class="AppleVisionFaceDetector" name="Faces" />
<module class="FaceTargetSelector" name="Selector" selection="largest" hold_time="0.5" />
<connection source="Faces.BOXES" target="Selector.BOXES" />
```

The tests in `tests/` read recorded box sequences with `InputFile` and check selection, switching,
filtering, the hold behavior, smoothing, repeated frames, the pose output, and the switch delay.

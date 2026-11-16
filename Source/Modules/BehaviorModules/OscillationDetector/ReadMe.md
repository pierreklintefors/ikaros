# OscillationDetector

## Description

`OscillationDetector` notices when a value swings back and forth, for example when a person pans
their head from side to side to play with a robot, waves, or shakes their head. It watches a single
value, such as the yaw from `FaceTargetSelector.TARGET_POSE`.

A swing is a movement of at least `amplitude` in one direction, measured from the last turning
point. The next swing must go the other way. When `swings` alternating swings, for example left,
right, left, happen within `window` seconds, `DETECTED` is 1 for one tick. After a detection the
count starts over, and no new detection is made for `refractory` seconds.

Because each swing is measured from the last turning point, small wobbles below `amplitude`, such
as detection noise, never count, and a single slow turn counts as only one swing. `smoothing`
can filter the input further. While `VALID` is 0, for example when the face is lost, the count
starts over.

`SWINGS` shows how many swings have been counted within the window, which helps to tune
`amplitude` and `window`.

## Parameters

| Name | Description | Type | Default |
| --- | --- | --- | --- |
| amplitude | Smallest movement that counts as a swing, in the units of INPUT | number | 12 |
| swings | Number of alternating swings needed, for example 3 for left, right, left | number | 3 |
| window | Seconds within which the swings must happen | number | 3 |
| refractory | Seconds after a detection during which no new detection is made | number | 2 |
| smoothing | Rate in 1/s at which an internal copy of INPUT follows it; 0 uses INPUT directly | rate | 0 |

## Inputs

| Name | Description | Optional |
| --- | --- | --- |
| INPUT | The value to watch, for example head yaw | |
| VALID | 1 while INPUT is meaningful; while 0 the swing count starts over | yes |

## Outputs

| Name | Description |
| --- | --- |
| DETECTED | 1 on the tick the back-and-forth movement is recognized |
| SWINGS | Number of alternating swings within the window so far |

## Example

Start a mirroring game when a person pans their head left, right, and left again:

```xml
<module class="OscillationDetector" name="HeadPan" amplitude="12" swings="3" window="3" />

<connection source="Selector.TARGET_POSE[0:1]" target="HeadPan.INPUT" />
<connection source="Selector.VALID" target="HeadPan.VALID" />
<connection source="HeadPan.DETECTED" target="Behaviors.REQUEST[10:11]" />
```

The test in `tests/` checks that three alternating swings of 15 are detected once on the third
swing, and that a wobble of 8 and a single turn are not detected.

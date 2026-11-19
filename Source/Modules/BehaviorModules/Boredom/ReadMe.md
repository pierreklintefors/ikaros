# Boredom

## Description

`Boredom` decides when a robot gets bored, so that it can start something to pass the time, such as
a light show, when nothing has happened for a while, and stop as soon as something does.

While nothing happens, that is while every element of `ACTIVITY` is at most 0.5 and `AROUSAL`, if
connected, is below `arousal_threshold`, the quiet time in `QUIET` grows. After a random wait between
`min_wait` and `max_wait` seconds of quiet, `BORED` becomes 1 for a random duration between
`min_duration` and `max_duration` seconds. Then the quiet time starts over, so bored episodes come
back from time to time while the robot stays alone.

Any activity, for example a face that appears, ends a bored episode at once and starts the wait
again. With `seed` 0 or above the waits and durations are reproducible.

## Parameters

| Name | Description | Type | Default |
| --- | --- | --- | --- |
| min_wait | Shortest quiet time in seconds before a bored episode | number | 20 |
| max_wait | Longest quiet time in seconds before a bored episode | number | 45 |
| min_duration | Shortest bored episode in seconds | number | 6 |
| max_duration | Longest bored episode in seconds | number | 15 |
| arousal_threshold | AROUSAL at or above this counts as activity | number | 0.3 |
| seed | Random seed; negative selects nondeterministic seeding | number | -1 |

## Inputs

| Name | Description | Optional |
| --- | --- | --- |
| ACTIVITY | Above 0.5 while something happens, for example a face is seen | |
| AROUSAL | Present excitement; at or above arousal_threshold counts as activity | yes |

## Outputs

| Name | Description |
| --- | --- |
| BORED | 1 during a bored episode, otherwise 0 |
| QUIET | Seconds of quiet since the last activity or the last bored episode |

## Example

```xml
<module class="Boredom" name="Boredom" min_wait="20" max_wait="45" min_duration="6" max_duration="15" />

<connection source="Selector.VALID" target="Boredom.ACTIVITY" />
<connection source="Arousal.AROUSAL" target="Boredom.AROUSAL" />
<connection source="Boredom.BORED" target="LightShow.RUN" />
```

The test in `tests/` checks the start of an episode after the wait, its end on activity, the end of
a full episode after its duration, and that high arousal prevents boredom.

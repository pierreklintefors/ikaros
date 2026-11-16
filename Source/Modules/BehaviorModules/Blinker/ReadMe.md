# Blinker

## Description

`Blinker` makes a robot blink like a living creature. It outputs an eyelid closure, `LID`, from 0
(open) to 1 (closed), which can drive physical eyelids or LED eyes such as `EpiExpression.BLINK`.

Blinks start at random intervals, drawn uniformly between `min_interval` and `max_interval`
seconds after the previous blink has ended. Each blink has the shape of a human blink:

| Phase | Duration | Lid |
| --- | --- | --- |
| Close | `close_time` | rises smoothly from 0 to 1 |
| Closed | `closed_time` | 1 |
| Open | `open_time` | falls smoothly from 1 to 0 |

With probability `double_blink_probability` a blink is followed by a second one after
`double_blink_gap` seconds. The second blink is never doubled again.

`TRIGGER` starts a blink at once when it changes from 0 to 1, for example from an animation event.
While `INHIBIT` is above 0.5 no spontaneous blink starts, for example while the robot stares in
surprise. A blink that falls due during the inhibition follows as soon as it ends, which looks
natural after a wide-eyed moment. `TRIGGER` also works during the inhibition.

`BLINK` is 1 on the tick a blink starts, so other modules can react to it.

With `seed` 0 or above the intervals are reproducible, which is useful for tests.

## Parameters

| Name | Description | Type | Default |
| --- | --- | --- | --- |
| min_interval | Shortest time in seconds from the end of one blink to the start of the next | number | 1.5 |
| max_interval | Longest time in seconds from the end of one blink to the start of the next | number | 5 |
| double_blink_probability | Probability that a blink is followed at once by a second blink | number | 0.2 |
| double_blink_gap | Seconds between the two blinks of a double blink | number | 0.1 |
| close_time | Seconds to close the lid | number | 0.08 |
| closed_time | Seconds the lid stays closed | number | 0.05 |
| open_time | Seconds to open the lid | number | 0.16 |
| seed | Random seed; negative selects nondeterministic seeding | number | -1 |

## Inputs

| Name | Description | Optional |
| --- | --- | --- |
| TRIGGER | A change from 0 to 1 starts a blink at once | yes |
| INHIBIT | While above 0.5 no spontaneous blink starts | yes |

## Outputs

| Name | Description |
| --- | --- |
| LID | Eyelid closure from 0 (open) to 1 (closed) |
| BLINK | 1 on the tick a blink starts, otherwise 0 |

## Example

```xml
<module class="Blinker" name="Blinker" min_interval="1.5" max_interval="5" />
<module class="EpiExpression" name="Expression" />

<connection source="Blinker.LID" target="Expression.BLINK" />
```

The test in `tests/` checks the lid profile and interval of spontaneous blinks, a blink forced by
`TRIGGER`, and a blink delayed by `INHIBIT`.

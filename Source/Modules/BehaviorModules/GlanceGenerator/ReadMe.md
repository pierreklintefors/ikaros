# GlanceGenerator

## Description

`GlanceGenerator` makes a robot look around by itself, so it never sits completely still. It
produces a random offset in any number of dimensions, for example one eye pan offset for quick
glances, or a head tilt and pan offset for slowly looking around a room.

At random intervals, drawn uniformly between `min_interval` and `max_interval` seconds after the
previous glance, it chooses a new offset:

- With probability `center_probability` the new offset is the center, 0 in every dimension.
- Otherwise every dimension gets a random sign and a size between `min_fraction` and 1 times its
  `amplitude`. A `min_fraction` above 0 keeps glances large enough to be noticed.

With `return_time` above 0 a glance returns to the center after that many seconds, which gives a
quick look away and back. With 0 the offset is kept until the next glance.

The output jumps to the chosen offset when `rate` is 0. Otherwise it moves there exponentially at
`rate` per second, which suits slow head movements. Fast eye glances should jump, since the servos
and `GazeLock` already limit the speed.

While `INHIBIT` is above 0.5 the offset returns to the center and no glance starts; the interval
starts again when the inhibition ends. Connect `GazeLock.LOCKED`, for example, so that the head only
looks around while nobody is being followed.

`SHIFT` is 1 on every tick a new offset is chosen, including returns to the center.

## Parameters

| Name | Description | Type | Default |
| --- | --- | --- | --- |
| amplitude | Largest offset in each dimension; the number of values sets the size of OUTPUT | matrix | 10 |
| min_fraction | Smallest glance as a fraction of amplitude | number | 0.3 |
| min_interval | Shortest time in seconds between two glances | number | 1 |
| max_interval | Longest time in seconds between two glances | number | 3 |
| center_probability | Probability that a new offset is the center | number | 0.3 |
| return_time | Seconds after which a glance returns to the center; 0 keeps it until the next glance | number | 0 |
| rate | Rate in 1/s at which the output moves toward the chosen offset; 0 jumps | rate | 0 |
| seed | Random seed; negative selects nondeterministic seeding | number | -1 |

## Inputs

| Name | Description | Optional |
| --- | --- | --- |
| INHIBIT | While above 0.5 the output returns to the center and no new glances start | yes |

## Outputs

| Name | Description |
| --- | --- |
| OUTPUT | Present offset in each dimension |
| SHIFT | 1 on the tick a new offset is chosen, including returns to the center |

## Example

Quick eye glances that come back, and slow head look-around while the gaze is not locked:

```xml
<module class="GlanceGenerator" name="EyeGlances" amplitude="8" return_time="0.5" min_interval="1.5" max_interval="4" />
<module class="GlanceGenerator" name="HeadGlances" amplitude="8, 25" rate="1.5" min_interval="2" max_interval="5" />

<connection source="EyeGlances.OUTPUT" target="Gaze.EYE_OFFSET" />
<connection source="HeadGlances.OUTPUT" target="Gaze.HEAD_OFFSET" />
<connection source="Gaze.LOCKED" target="HeadGlances.INHIBIT" delay="1" />
```

The test in `tests/` checks the glance interval, the glance size with `min_fraction` 1, the return
after `return_time`, the exponential approach at `rate`, and the wait while `INHIBIT` is 1.

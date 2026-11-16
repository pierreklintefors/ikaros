# Arousal

## Description

`Arousal` models how excited a robot is, from 0 (calm) to 1 (very excited). A single level like this
lets every part of a robot show the same mood: when something exciting happens, animations play
faster and larger, sounds get louder, and the pupils open; when nothing happens, the robot calms
down again.

The level has three parts:

- A resting level: `idle_level`, or `engaged_level` while `ENGAGED` is 1. A robot with company is a
  little livelier.
- An excitation from events. A change from 0 to 1 in element i of `STIMULUS` adds `boosts[i]`, for
  example a large boost when someone new appears and a small one for each reaction the robot plays.
  The excitation fades at `decay_rate`, so the robot calms down after a few seconds.
- An optional continuous drive: `drive_gain` times `DRIVE`, for example the size of the nearest face,
  so a person who comes close excites the robot more.

`AROUSAL` follows the sum, limited to 0..1, at `response_rate`, which keeps changes smooth.

`MAPPED` turns the level into any number of values: element i is `low[i] + AROUSAL * (high[i] -
low[i])`. One module can then set, for example, the tempo and size of animations, the sound volume,
and a pupil offset.

## Parameters

| Name | Description | Type | Default |
| --- | --- | --- | --- |
| idle_level | Resting level while ENGAGED is 0 or not connected | number | 0.2 |
| engaged_level | Resting level while ENGAGED is 1 | number | 0.45 |
| boosts | Excitation added by a rising edge of each STIMULUS element; elements beyond the list use the last value | matrix | 0.3 |
| drive_gain | Excitation per unit of DRIVE | number | 0 |
| decay_rate | Rate in 1/s at which the excitation fades | rate | 0.15 |
| response_rate | Rate in 1/s at which AROUSAL follows its target; 0 follows immediately | rate | 3 |
| low | MAPPED values at arousal 0; sets the size of MAPPED | matrix | 0 |
| high | MAPPED values at arousal 1; as many values as low | matrix | 1 |

## Inputs

| Name | Description | Optional |
| --- | --- | --- |
| ENGAGED | 1 while the robot has company | yes |
| STIMULUS | Event pulses; a change from 0 to 1 in an element adds its boost | yes |
| DRIVE | Continuous excitation, for example the size of the nearest face | yes |

## Outputs

| Name | Description |
| --- | --- |
| AROUSAL | Excitement from 0 (calm) to 1 (very excited) |
| MAPPED | low + AROUSAL * (high - low), element by element |

## Example

Arousal rises when someone new is found and when a reaction plays, and sets the animation tempo
(0.85 to 1.25), movement size (0.8 to 1.2), and sound volume (0.45 to 1):

```xml
<module class="Arousal" name="Arousal" boosts="0.35, 0.1" low="0.85, 0.8, 0.45" high="1.25, 1.2, 1" />

<connection source="Target.NEW_TARGET" target="Arousal.STIMULUS[0:1]" />
<connection source="Animator.PLAYING" target="Arousal.STIMULUS[1:2]" />
<connection source="Arousal.MAPPED[0:1]" target="Animator.SPEED" />
<connection source="Arousal.MAPPED[1:2]" target="Animator.INTENSITY" />
<connection source="Arousal.MAPPED[2:3]" target="Sounds.VOLUME" />
```

The test in `tests/` checks the boost, the decay of the excitation, the engaged resting level,
the smoothing, and `MAPPED`.

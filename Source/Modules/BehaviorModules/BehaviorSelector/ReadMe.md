# BehaviorSelector

## Description

`BehaviorSelector` decides when a robot does something by itself and what it does. It produces
start pulses for a set of numbered behaviours, for example the animations of a `KeyframeAnimator`,
so a robot shows varied, spontaneous behaviour and reacts when people arrive and leave.

### Spontaneous behaviours

After `min_interval` to `max_interval` seconds without a behaviour, a spontaneous behaviour starts.
It is chosen at random by weight:

- While `ENGAGED` is 0, for example when nobody is being followed, the weights come from
  `idle_weights`.
- While `ENGAGED` is 1 they come from `engaged_weights`.

A missing weight counts as 0, so a behaviour can be kept for reactions only. A behaviour that started
less than `cooldown` seconds ago is not chosen again, which keeps the robot from repeating itself.
When every weighted behaviour is cooling down, the selector tries again after `min_interval`.

While `BUSY` is 1, for example from `KeyframeAnimator.PLAYING`, the interval does not run, so it
measures the quiet time between behaviours rather than the time between starts.

### Mood and memory

Picking behaviours purely at random makes a robot look erratic. Three settings make the choice
follow a mood and avoid repetition:

- With `AROUSAL` connected, for example to `Arousal.AROUSAL`, each behaviour's weight is multiplied
  by how close its energy in `arousal_levels` lies to the present arousal: a Gaussian with width
  `arousal_width`. A calm robot then prefers calm behaviours and an excited one lively behaviours,
  and the mood changes gradually with the arousal instead of jumping between extremes.
- `memory` keeps the most recent spontaneous behaviours from being chosen again while others are
  available, so the robot does not repeat itself even when cooldowns are short.
- `excited_interval_scale` shortens the idle interval as arousal rises; at arousal 1 it is that
  factor times its length. A calm robot waits longer between behaviours.

### Reactions

When someone arrives, that is when `ENGAGED` changes to 1 or `NEW_TARGET` is 1, the `greeting`
behaviour starts after `greeting_delay` seconds. When `ENGAGED` changes to 0, the `farewell`
behaviour starts at once. Reactions do not wait for `BUSY`, so the robot responds to people even in
the middle of a spontaneous behaviour; a `KeyframeAnimator` then blends into the new animation. After
a greeting or farewell, no other reaction starts for `reaction_cooldown` seconds, so a person who is
lost for a moment is not greeted again. Set `greeting` or `farewell` to -1 to turn them off.

`REQUEST` lets other modules ask for a particular behaviour. A change from 0 to 1 in element i
starts behaviour i at once, interrupting like a reaction, unless it started less than `cooldown`
seconds ago. Connect a detector to one element with a target range, for example
`OscillationDetector.DETECTED` to `REQUEST[10:11]`, so that a person who pans their head starts a
mirroring game.

`ENABLE` at 0 stops all triggers, for example from a switch on a dashboard.

`TRIGGER` has `count` elements. The element of a starting behaviour is 1 for one tick. `SELECTED`
holds the index of the last started behaviour, and `WAIT` the idle seconds left before the next
spontaneous one.

## Parameters

| Name | Description | Type | Default |
| --- | --- | --- | --- |
| count | Number of behaviours; sets the size of TRIGGER | number | 1 |
| idle_weights | Relative probability of each behaviour while ENGAGED is 0 | matrix | 1 |
| engaged_weights | Relative probability of each behaviour while ENGAGED is 1 | matrix | 1 |
| min_interval | Shortest idle time in seconds before a spontaneous behaviour | number | 4 |
| max_interval | Longest idle time in seconds before a spontaneous behaviour | number | 10 |
| cooldown | Seconds before a spontaneous behaviour can be chosen again | number | 15 |
| greeting | Behaviour started when someone arrives; -1 for none | number | -1 |
| greeting_delay | Seconds from the arrival to the greeting | number | 0.5 |
| farewell | Behaviour started when ENGAGED becomes 0; -1 for none | number | -1 |
| reaction_cooldown | Seconds before another greeting or farewell | number | 10 |
| arousal_levels | Energy of each behaviour from 0 (calm) to 1 (excited); used with AROUSAL | matrix | 0.5 |
| arousal_width | How far from the present arousal a behaviour may lie and still be likely | number | 0.25 |
| memory | Number of most recent spontaneous behaviours not chosen again while others are available | number | 0 |
| excited_interval_scale | Factor on the idle interval at arousal 1 | number | 1 |
| seed | Random seed; negative selects nondeterministic seeding | number | -1 |

## Inputs

| Name | Description | Optional |
| --- | --- | --- |
| ENGAGED | 1 while the robot attends to someone, for example GazeLock.LOCKED | |
| NEW_TARGET | 1 when a different person has been selected | yes |
| BUSY | 1 while a behaviour plays; spontaneous behaviours wait | yes |
| ENABLE | 1 allows behaviours, 0 stops all triggers; enabled when not connected | yes |
| AROUSAL | Present excitement from 0 to 1; selects by arousal_levels and scales the interval | yes |
| REQUEST | A change from 0 to 1 in element i starts behaviour i at once, unless it is cooling down | yes |

## Outputs

| Name | Description |
| --- | --- |
| TRIGGER | 1 for one tick in the element of the behaviour that starts |
| SELECTED | Index of the last started behaviour, or -1 |
| WAIT | Idle seconds left before the next spontaneous behaviour |

## Example

```xml
<module class="BehaviorSelector" name="Selector" count="4" idle_weights="0, 3, 1, 0" engaged_weights="0, 1, 2, 2"
        greeting="0" min_interval="3" max_interval="8" cooldown="10" />
<module class="KeyframeAnimator" name="Animator" file="../Projects/Epi/EpiAnimations.json" channels="a" rest="0" />

<connection source="Gaze.LOCKED" target="Selector.ENGAGED" delay="1" />
<connection source="Animator.PLAYING" target="Selector.BUSY" delay="1" />
<connection source="Selector.TRIGGER" target="Animator.TRIGGER" />
```

The tests in `tests/` check the weighted choice in each state, the greeting delay, the cooldown,
the wait while `BUSY` is 1, the farewell, and the reaction cooldown, and the mood test checks
the choice by arousal, `memory`, and the shorter interval when excited. The request test checks
that a request starts its behaviour while `BUSY` is 1 and is ignored during the cooldown.

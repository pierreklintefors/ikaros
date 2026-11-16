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

### Reactions

When someone arrives, that is when `ENGAGED` changes to 1 or `NEW_TARGET` is 1, the `greeting`
behaviour starts after `greeting_delay` seconds. When `ENGAGED` changes to 0, the `farewell`
behaviour starts at once. Reactions do not wait for `BUSY`, so the robot responds to people even in
the middle of a spontaneous behaviour; a `KeyframeAnimator` then blends into the new animation. After
a greeting or farewell, no other reaction starts for `reaction_cooldown` seconds, so a person who is
lost for a moment is not greeted again. Set `greeting` or `farewell` to -1 to turn them off.

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
| seed | Random seed; negative selects nondeterministic seeding | number | -1 |

## Inputs

| Name | Description | Optional |
| --- | --- | --- |
| ENGAGED | 1 while the robot attends to someone, for example GazeLock.LOCKED | |
| NEW_TARGET | 1 when a different person has been selected | yes |
| BUSY | 1 while a behaviour plays; spontaneous behaviours wait | yes |
| ENABLE | 1 allows behaviours, 0 stops all triggers; enabled when not connected | yes |

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

The test in `tests/` checks the weighted choice in each state, the greeting delay, the cooldown,
the wait while `BUSY` is 1, the farewell, and the reaction cooldown.

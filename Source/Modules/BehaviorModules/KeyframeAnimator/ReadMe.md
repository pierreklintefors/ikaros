# KeyframeAnimator

## Description

`KeyframeAnimator` plays short, hand-made animations, such as a giggle, a surprised look, or a
nod, on any set of control signals. It is not tied to a robot: the channels can be eye offsets,
head offsets, pupil sizes, LED colors, or anything else that a downstream module reads. Timed events
can start sounds, blinks, or other modules.

The channels are named in `channels`, and `rest` gives their rest values. `OUTPUT` has one element per
channel, in the same order. Use connection ranges to send parts of `OUTPUT` to different modules, for
example `Animator.OUTPUT[2:4]` for a two-element head offset.

The animations are read from a JSON library, `file`. A relative path is resolved in `UserData`, so a
library in a project folder is named for example `../Projects/Epi/EpiAnimations.json`.

## Animation library

```json
{
    "animations": [
        {
            "name": "nod",
            "duration": 1.2,
            "tracks": {
                "head_tilt": [[0.2, -6], [0.5, 2, "back"], [0.9, 0]],
                "pupil": {"start": 0.1, "repeat": 3, "keys": [[0, 0], [0.1, 3], [0.2, 0]]}
            },
            "events": [[0, "blink"], [0.3, ["giggle_1", "giggle_2"], 0.8]]
        }
    ]
}
```

- `name` identifies the animation for the `play` command. Animations are numbered from 0 in the
  order of the list, which is the element of `TRIGGER` that starts them.
- `tracks` maps channel names to keys. A key is `[time, value]` or `[time, value, ease]`, with time in
  seconds from the start of the animation. Key times must not decrease.
- A track can also be a dictionary with `keys`, an optional `start` time that shifts all keys, and
  `repeat`, which repeats the keys back to back. The repeat period is the time from the first to the
  last key, so `[[0, 0], [0.1, 3], [0.2, 0]]` repeated three times pulses three times in 0.6 s.
- `events` lists `[time, name]` or `[time, name, probability]` entries. The name is one of
  `events`, or the index of an element of `EVENTS`. On the tick the time is reached, that element of
  `EVENTS` is 1. With a probability below 1 the event fires only in that fraction of the plays, for
  example so that a robot does not make a sound every time. The name can also be a list of
  alternatives, such as `["giggle_1", "giggle_2", "giggle_3"]`; one of them is chosen at random
  each time, so a sound or other event varies between plays.
- `duration` is optional. The animation lasts until its last key or event, or until `duration` if
  that is later.

The ease of a key sets how the channel moves from the previous key to it:

| Ease | Movement |
| --- | --- |
| `smooth` | Starts and stops gently; the default |
| `linear` | Constant speed |
| `step` | Holds the previous value and jumps when the key time is reached |
| `back` | Overshoots by about 10 percent and settles, which gives a springy pop |

## Playing

A rising edge on element i of `TRIGGER`, or the `play` command, starts animation i. Every animation
starts from the present channel values and moves to its first key with that key's ease, so a new
animation can interrupt another one without jumps. Set `interrupt` to false to ignore starts while an
animation plays.

Channels without a track in the playing animation, and all channels after the animation has ended,
return to their rest values at `release_rate`. A channel keeps the value of its last key until the
end of the animation, so an animation should end its tracks where it wants the release to begin,
often at the rest value.

`PLAYING` is 1 while an animation plays, `ANIMATION` gives its index or -1, and `PROGRESS` goes from 0
to 1. On the last tick of an animation the final values are output and `DONE` is 1; the release
starts on the next tick. A rising edge on `STOP`, or the `stop` command, stops the animation and
releases all channels.

## Variation

Hand-made animations soon look repetitive when they play the same way every time. Five settings
make every play a little different, and two inputs let another module, such as `Arousal`, change
how animations play:

- `SPEED` is a tempo factor. At 2 an animation, including its events, plays twice as fast.
- `INTENSITY` scales the movement of the `intensity_channels` around their rest values. At 0.5 a
  head nod is half as deep. Leave color channels out, so that colors are not dimmed.
- `speed_variation` and `intensity_variation` give each play a random tempo and size, up to that
  fraction faster or slower, and larger or smaller.
- `mirror_channels` change the sign of their movement around rest together, with probability
  `mirror_probability` per play. Name for example the eye and head pan channels, so that the
  robot glances to the left one time and to the right the next.

`VARIATION` shows the present tempo, size, and mirror sign. `seed` makes the variation
reproducible.

## Commands

| Name | Description |
| --- | --- |
| play | Plays the animation given by `name`, or by the index `x` when no name is given. A WebUI button can use `command="Animator.play"` with `select_x` set to the index. |
| stop | Stops the playing animation. |

## Parameters

| Name | Description | Type | Default |
| --- | --- | --- | --- |
| file | JSON animation library; a relative path is resolved in UserData | string |  |
| channels | Comma-separated channel names in OUTPUT order | string |  |
| rest | Rest value of each channel; sets the size of OUTPUT and must match channels | matrix | 0 |
| events | Comma-separated event names in EVENTS order | string |  |
| event_count | Size of EVENTS; at least the number of event names | number | 1 |
| release_rate | Rate in 1/s at which unused channels return to rest; 0 returns immediately | rate | 3 |
| interrupt | A new animation replaces a playing one; when false, starts are ignored while one plays | bool | true |
| intensity_channels | Channels whose movement around rest is scaled by INTENSITY and intensity_variation; empty scales all | string |  |
| mirror_channels | Channels whose movement around rest changes sign together in some plays | string |  |
| mirror_probability | Probability that a play mirrors the mirror_channels | number | 0.5 |
| speed_variation | Each play runs at a random tempo up to this fraction faster or slower | number | 0 |
| intensity_variation | Each play moves a random fraction up to this much more or less around rest | number | 0 |
| seed | Random seed for the variations; negative selects nondeterministic seeding | number | -1 |

## Inputs

| Name | Description | Optional |
| --- | --- | --- |
| TRIGGER | A change from 0 to 1 in element i plays animation i | yes |
| STOP | A change from 0 to 1 stops the playing animation | yes |
| SPEED | Tempo factor; 2 plays animations and their events twice as fast | yes |
| INTENSITY | Size factor for the movement of intensity_channels around rest | yes |

## Outputs

| Name | Description |
| --- | --- |
| OUTPUT | Present value of each channel |
| EVENTS | 1 on the tick an event of the playing animation fires, otherwise 0 |
| PLAYING | 1 while an animation plays |
| ANIMATION | Index of the playing animation, or -1 |
| PROGRESS | Fraction of the playing animation that has passed, from 0 to 1 |
| DONE | 1 on the tick an animation plays to its end |
| VARIATION | Present tempo factor, size factor, and mirror sign |

## Example

```xml
<module class="KeyframeAnimator" name="Animator" file="../Projects/Epi/EpiAnimations.json"
        channels="eye_offset, vergence, head_tilt, head_pan" rest="0, 0, 0, 0"
        events="blink, giggle_1, giggle_2" event_count="3" />

<connection source="Animator.OUTPUT[0:1]" target="Gaze.EYE_OFFSET" />
<connection source="Animator.OUTPUT[1:2]" target="Gaze.VERGENCE" />
<connection source="Animator.OUTPUT[2:4]" target="Gaze.HEAD_OFFSET" />
<connection source="Animator.EVENTS[0:1]" target="Blinker.TRIGGER" />
```

The tests in `tests/` check linear keys, events on the right ticks, a repeated step track, an
interruption that starts from the present value, the release to rest, and that an unknown channel
stops the model at startup. The variation test checks `SPEED` with its events, and `INTENSITY` with
mirroring. The events test checks that one of several alternatives fires, and event probabilities
0 and 1.

# EpiLightShow

## Description

`EpiLightShow` plays light patterns on the eye and mouth LEDs of an Epi robot, over the expression
from `EpiExpression`, and maps all LED colors to the wiring of each robot.

It sits between `EpiExpression` and `FadeCandy`: the expression LEDs go into `EYE_LEFT`, `EYE_RIGHT`,
`MOUTH_TOP`, and `MOUTH_BOTTOM`, and `LEFT_EYE`, `RIGHT_EYE`, `MOUTH_HIGH`, and `MOUTH_LOW` go to
`FadeCandy`. While no pattern plays, the expression passes through unchanged apart from the wiring
map. The `VIEW_` outputs show the same colors in the layout of the `epi-head` widget for the dashboard.

## LED layout

All patterns are made in the layout of the `epi-head` widget, seen from the front:

- Eye LED *i* sits at *i* x 30 degrees clockwise from 3 o'clock: 9 is the top, 0 the right, 3 the
  bottom, and 6 the left side as seen from the front. The robot's left eye appears on the viewer's
  right.
- Mouth LED *i* of each row counts from the viewer's left; `MOUTH_HIGH` is the upper row. LEDs 0 and
  7 are hidden behind the face on Epi.

## Patterns

| Number | Pattern | Description |
| --- | --- | --- |
| 1 | Wave | Lights one row at a time from the top of the eyes down: LED 9, then 8 and 10, and so on to LED 3, then the upper and lower mouth rows, and starts over. The row just left glows dimly for a moment. `wave_rate` rows per second. |
| 2 | Circle | A light with a fading tail runs around each eye from the top, clockwise in the eye on the viewer's right and counterclockwise in the other, so the eyes mirror each other. Each half of the mouth, its upper and lower row, forms a small loop that follows the eye on its side. `circle_rate` laps per second, `tail` long. |

While `RUN` is 1, for example from `Boredom.BORED`, a show plays the patterns in `show_patterns`
in turn, `pattern_time` seconds each. `INTERRUPT` stops a show at once, for example when a face
is seen. The `play` command with `x` 1 or 2 plays that pattern for `button_duration` seconds without
interruption. Patterns fade in and out at `fade_rate`.

The color is a random hue for each pattern, or with `color_mode` arousal a hue from calm blue to
excited orange that follows `AROUSAL`. `COLOR` overrides both. `AROUSAL` also sets the speed, from
0.7 times when calm to 1.5 times when very excited.

## LED orientation check

The LEDs may be wired in a different order on each robot, so each robot's orientation is stored in
`calibration_file`, a JSON file in `UserData`, under the name in `robot`. A robot that is not in the
file shows the orientation check at start-up, and `check` or the `check` command show it again:

| LEDs | Color |
| --- | --- |
| Top of each eye | Red |
| Right side of each eye, seen from the front | Green |
| Bottom of each eye | Blue |
| Left side of each eye, seen from the front | Yellow |
| Upper mouth row | Cyan, with its leftmost visible LED white |
| Lower mouth row | Magenta, with its leftmost visible LED white |

Look at the robot from the front and correct the orientation with the commands until every LED is
where the table says, then save:

| Command | Effect |
| --- | --- |
| rotate | Turns the colors of one eye by one LED; `x` 0 for the left eye, 1 for the right eye |
| flip | Reverses the direction of one eye; `x` as for rotate |
| swap_mouth_rows | Exchanges the upper and lower mouth rows |
| reverse_mouth | Reverses the order of the mouth LEDs |
| save | Stores the orientation in `calibration_file` and ends the check |
| check | Shows the check again |
| play | Plays pattern `x` (1 wave, 2 circle) for `button_duration` seconds |
| stop | Stops a pattern from `play` |

`ORIENTATION` shows the present orientation: left eye first angle and direction, right eye first
angle and direction, mouth rows swapped, and mouth reversed. `CALIBRATED` is 1 once the robot is in
the file. In the file, `left_eye_first_angle` is the angle of the eye's LED 0, clockwise from 3 o'clock
as seen from the front, and `left_eye_direction` is 1 when the LEDs are numbered clockwise.

## Parameters

| Name | Description | Type | Default |
| --- | --- | --- | --- |
| robot | Robot name under which the LED orientation is stored | string |  |
| calibration_file | JSON file in UserData with the LED orientation of each robot | string | EpiLEDCalibration.json |
| check | Show the orientation check at start-up even when the robot is in the file | bool | false |
| show_patterns | Patterns that a show cycles through: 1 wave, 2 circle | matrix | 1, 2 |
| pattern_time | Seconds each pattern plays during a show | number | 5 |
| button_duration | Seconds a pattern from the play command plays without interruption | number | 5 |
| color_mode | Pattern color: random hue, or following AROUSAL | number | random |
| brightness | Brightness of the patterns | number | 0.8 |
| wave_rate | Rows per second of the wave at speed 1 | number | 8 |
| circle_rate | Laps per second of the circle at speed 1 | number | 0.8 |
| tail | Length of the circle's fading tail as a fraction of a lap | number | 0.3 |
| fade_rate | Rate in 1/s at which patterns fade in and out; 0 switches at once | rate | 6 |
| seed | Random seed for the colors; negative selects nondeterministic seeding | number | -1 |

## Inputs

| Name | Description | Optional |
| --- | --- | --- |
| EYE_LEFT | Left eye colors from EpiExpression, 3 x 12 | |
| EYE_RIGHT | Right eye colors from EpiExpression, 3 x 12 | |
| MOUTH_TOP | Upper mouth colors from EpiExpression, 3 x 8 | |
| MOUTH_BOTTOM | Lower mouth colors from EpiExpression, 3 x 8 | |
| RUN | While above 0.5 a show plays | yes |
| INTERRUPT | Above 0.5 stops a show at once; a pattern from the play command continues | yes |
| AROUSAL | Present excitement; sets the speed and, with color_mode arousal, the color | yes |
| COLOR | Red, green, blue of the patterns, overriding color_mode | yes |

## Outputs

| Name | Description |
| --- | --- |
| LEFT_EYE | Left eye colors in the robot's own LED order, for FadeCandy |
| RIGHT_EYE | Right eye colors in the robot's own LED order, for FadeCandy |
| MOUTH_HIGH | Upper mouth colors in the robot's own LED order, for FadeCandy |
| MOUTH_LOW | Lower mouth colors in the robot's own LED order, for FadeCandy |
| VIEW_LEFT_EYE | Left eye colors in the layout of the epi-head widget |
| VIEW_RIGHT_EYE | Right eye colors in the layout of the epi-head widget |
| VIEW_MOUTH_HIGH | Upper mouth colors in the layout of the epi-head widget |
| VIEW_MOUTH_LOW | Lower mouth colors in the layout of the epi-head widget |
| PATTERN | Playing pattern: 0 none, 1 wave, 2 circle, 3 orientation check |
| CALIBRATED | 1 when the robot's LED orientation is stored |
| ORIENTATION | Left eye first angle and direction, right eye first angle and direction, mouth rows swapped, mouth reversed |

## Example

```xml
<module class="EpiLightShow" name="LightShow" robot="EpiBlack" color_mode="arousal" />

<connection source="Expression.LEFT_EYE" target="LightShow.EYE_LEFT" />
<connection source="Expression.RIGHT_EYE" target="LightShow.EYE_RIGHT" />
<connection source="Expression.MOUTH_HIGH" target="LightShow.MOUTH_TOP" />
<connection source="Expression.MOUTH_LOW" target="LightShow.MOUTH_BOTTOM" />
<connection source="Boredom.BORED" target="LightShow.RUN" />
<connection source="Selector.VALID" target="LightShow.INTERRUPT" />
<connection source="LightShow.LEFT_EYE" target="LEDs.LEFT_EYE" />
<connection source="LightShow.RIGHT_EYE" target="LEDs.RIGHT_EYE" />
<connection source="LightShow.MOUTH_HIGH" target="LEDs.MOUTH_HIGH" />
<connection source="LightShow.MOUTH_LOW" target="LEDs.MOUTH_LOW" />
```

The tests in `tests/` check the wave rows, the mirrored circle and its mouth loops, the check
for a robot that is not in the file, and the mapping of a stored orientation to rotated, reversed
eye wiring and swapped, reversed mouth rows.

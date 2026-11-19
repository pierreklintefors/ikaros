# Epi animatronics

`EpiAnimatronics.ikg` makes an Epi robot look alive, playful, and curious. Most of the time it is
subtle: it blinks, glances, and slowly looks around the room. When it finds someone new it perks up
with a happy chirp, when someone comes really close its eyes cross, when someone pans their head
back and forth it starts a game of copycat, and now and then it plays a little reaction: a sly look to the side with a "he-he-heh", a
cross-eyed moment with blue eyes, a giggle, being "mind blown" with eyes that move apart, a rising
whistle of surprise, and a "pooof", or a game of copycat in which it mirrors the person's head
movements. It speaks only in robot chirps and whistles, not every time and never quite the same,
and an excitement level, its arousal, makes it livelier and louder when something happens and
calmer when nothing does.

The model works with every Epi robot that `EpiServos` knows. Choose the robot with `robot`.

## How it works

The behaviour is built in layers, the way Disney's robot characters combine continuous background
motion, triggered animations on top, and "show functions" such as eyes, lights, and sound that are
synchronized with the motion:

1. **Seeing.** `InputVideoStream`, `AppleVisionFaceDetector`, and `FaceTargetSelector` find the
   closest face in the right eye camera, as in `EpiHeadMirroring.ikg`, and a second chain finds it
   in the left eye camera. `StereoDistance` combines the two views with the eyes' angles to measure
   how far away the face is, without assuming its size.
2. **Looking.** `GazeLock` keeps the eyes on the face and turns the head after it. `camera_eye`
   tells it that the camera is in the right eye, so the face direction stays correct while the eyes
   cross or dart. When someone comes really close, `ProximityVergence` makes the eyes converge on
   them, as people's eyes do, but three times as strongly: not at all beyond 1 m, about 15 degrees at
   0.55 m, and about 27 degrees at 0.3 m and nearer. From about 0.45 m the right eye is at its limit
   and the left eye does the rest of the turn. It uses the distance measured by both eyes, and the
   size of the face box only when the left camera does not see the face. The pupils narrow a little
   at the same time, as in the human near response. While a face is closer than about 0.8 m,
   the head holds still (`HOLD_HEAD` on `Gaze`) and only the eyes follow the face. Without this, the
   head would turn as the face comes near, because the camera sits in the right eye and sees a close
   face at a different angle than the middle of the head does. When the person steps back, the head
   follows again.
3. **Background life.** `Blinker` blinks every 1.5 to 4.5 seconds, sometimes twice, as an eyelid
   that closes from the top of the LED ring down. `EyeGlances` makes quick glances away and back, and
   `HeadGlances` slowly looks around while nobody is followed, which also lets the camera find new
   faces. The glances pause while an animation plays. Blinks continue through most animations and
   pause only where wide-open eyes carry the moment (found you, cross-eyed, and mind blown); a blink
   that falls due then follows right after it, as people blink after a surprise.
4. **Mood.** `Arousal` keeps an excitement level from 0 to 1. It rests low when Epi is alone and a
   little higher with company. Finding someone new raises it a lot, a face that comes close raises
   it, and each reaction raises it a little. It then falls back over about half a minute. Arousal
   sets the tempo of the animations (0.8 to 1.15 times), their size (0.8 to 1.2 times), the sound
   volume, and the pupils, which open with excitement as people's do.
5. **Choosing.** `BehaviorSelector` decides when Epi does what. When someone new appears it plays
   *found you*, and when the person leaves, *curious*. Alone, it only now and then looks around or
   peeks curiously. With company it starts an animation after 8 to 16 quiet seconds, sooner when
   excited, so the subtle background behaviour fills most of the time. When the person pans their
   head left, right, and left again within 3 seconds, `OscillationDetector` asks for *copycat* at
   once. The choice follows the mood: each animation has an energy
   level, and the ones that match the present arousal are preferred, so Epi is lively right after
   meeting someone and calmer later. The last three animations are avoided, and none comes back
   within 25 seconds.
6. **Performing.** `KeyframeAnimator` plays the animations from `EpiAnimations.json` on 17
   channels: eye offset, vergence, head tilt and pan, pupil, eye and mouth colors, brightness,
   eyelid, mirroring, and a blink hold. Each play varies a little in tempo and size, and the sideways
   movements are mirrored at random, so the same animation glances left one time and right the next.
7. **Playing copycat.** Copycat starts when the person pans their head back and forth, and now
   and then by itself. During the copycat animation its `mirror` channel turns on `HeadMirror`,
   which copies the person's head turns and nods as a mirror image, as in `EpiHeadMirroring.ikg`,
   while the eyes stay on the person and glow green.
8. **Voice.** Most sounds play only in some of the plays, and each time one of three variants is
   chosen, so Epi is not chatty and never repeats itself exactly. The signature sounds of mind
   blown always play.
9. **Body.** `EpiExpression` shows the state and the animation on the LEDs and pupils. `SoundOutput`
   plays the sounds, and `EpiServos` and `FadeCandy` drive the hardware.

All modules apart from `EpiServos`, `EpiExpression`, and `FadeCandy` are general and can be used
with other robots.

## Animations

The animations follow classic animation principles: a small anticipation before a big move, an
overshoot that settles, the eyes leading and the head following, holds that let a pose read, and a
blink with large gaze shifts. Sounds land on the motion accents: the mouth flashes and the head
bobs on each chirp of the giggle and the "he-he-heh", the eyes reach the nose with the hum of
cross-eyed, and the flash comes with the pooof.

| # | Name | What Epi does | Sound | Energy |
| ---: | --- | --- | --- | ---: |
| 0 | Found you | A noticing blink and a still moment, then it perks up (pupils open, a lift of the head, a warm glow), gives a small happy nod, and smiles | a happy, quirky one-syllable chirp | 0.8 |
| 1 | Greet | A small lift, a nod on the chirp, a springy return, warm eyes and a smile | bouncy chirps that end high | 0.5 |
| 2 | Cross-eyed | A tiny outward flick, then the eyes meet at the nose and turn blue, pupils tighten, head dips | a hum, then a rising, questioning chirp | 0.6 |
| 3 | Eye dart | A small counter-move, then four swings from side to side with flickering cyan eyes | | 0.5 |
| 4 | Giggle | Mouth flashes and pupils pulse on every chirp, the head bobs, eyes squint happily in pink | quick chirps hopping between high and low | 0.75 |
| 5 | Mind blown | It gathers (squint, small pupils, chin down), then the eyes move apart and the head lifts with the rising whistle; a flash and a blackout on the pooof, a dazed wobble, and a purple recovery | rising whistle, then a sputtering power-down and a puff | 0.9 |
| 6 | Curious | The eyes lead, the head follows to peek to the side and up, the pupils slowly open, teal eyes | a hum, then a rising, questioning chirp | 0.3 |
| 7 | Shy | Looks down and away with pink eyes and mouth, peeks back once, half-closed eyes | a soft whistle that lifts and sinks away | 0.35 |
| 8 | Mischief | The eyes slide slowly to the side under half-closed lids, purple glow, a sly laugh with a flash and a bob on each syllable, then a quick, innocent look back with wide eyes | a sly "he-he-heh" | 0.55 |
| 9 | Look around | The eyes lead each turn and the head follows slowly, about 12 degrees, to one side and the other | | 0.15 |
| 10 | Copycat | Started by panning the head back and forth: a bright, inviting look and a little lift of the head, then about 17 seconds of mirroring the person's head movements with green eyes, ending with a happy squint | sometimes a hello at the start and a giggle at the end | 0.6 |

The sound column lists what can play. Each sound event has a probability, from 35 percent for the
hum of cross-eyed to 85 percent for the giggles; the chirp of found you and the wow and pooof of mind
blown always play.
Each time, one of three takes of the sound is chosen.

Alone, Epi only looks around and sometimes peeks curiously. With company it mixes mischief,
cross-eyed, giggle, greet, shy, eye dart, mind blown, and, more rarely by itself, copycat, weighted
by the mood.

Every animation can also be played with the buttons on the dashboard.

## Light shows

When Epi has been alone and calm for 20 to 45 seconds, `Boredom` makes it bored for 6 to 15
seconds, and `EpiLightShow` plays light patterns over the eyes and mouth, changing pattern every 5
seconds:

- **Wave.** One row of LEDs lights at a time, from the top of the eyes down to the bottom, then the
  upper and lower mouth rows, and it starts over at the top. The row it has just left glows dimly.
- **Circle.** A light with a fading tail runs around each eye from the top, clockwise in one eye and
  counterclockwise in the other, so the eyes mirror each other. Each half of the mouth, its upper and
  lower row, forms a small loop that follows the eye on its side.

A face that appears stops a show at once. The **Wave (5 s)** and **Circle (5 s)** buttons play a
pattern for 5 seconds that a face does not stop. The colors are random, one per pattern; start the
model with `light_colors=arousal` to make them follow the mood, from calm blue to excited orange.
Excitement also speeds the patterns up.

### LED orientation check

The LEDs may be wired in a different order on each robot. The first time a robot runs the model,
its eyes and mouth show a check pattern instead of the expression, and **LEDs saved** shows 0.
Look at the robot from the front:

| Where | Color |
| --- | --- |
| Top of each eye | Red |
| Right side of each eye, seen from the front | Green |
| Bottom of each eye | Blue |
| Left side of each eye, seen from the front | Yellow |
| Upper mouth row | Cyan, with its leftmost visible LED white |
| Lower mouth row | Magenta, with its leftmost visible LED white |

If an eye is wrong, press **Left eye: turn** or **Right eye: turn** until red is at the top (the
left eye is the robot's left, on your right), and **flip** if green and yellow are then on the wrong
sides. If the mouth rows are exchanged, press **Mouth: swap rows**; if the white LEDs are on the
right, press **Mouth: reverse**. Then press **Save LEDs**. The orientation is stored in
`UserData/EpiLEDCalibration.json` under the robot's name, so each robot is checked only once;
**LED check** shows the pattern again. The robot view on the dashboard always shows the intended
layout.

## LEDs

| State | Eyes | Mouth |
| --- | --- | --- |
| Nobody followed | Dim amber | Dim |
| Looking at someone | Bright warm white, pupils open | Dim |
| Copycat while mirroring | Green | Bright warm white |
| Animation | The animation's colors on top | The animation's colors on top |

## Running

Without the robot, simulate the motors and LEDs and use the Mac's own camera. Epi then follows
your face on the dashboard and plays its animations and sounds:

```sh
./Bin/ikaros simulate=true camera=InputVideo left_camera=InputImage Projects/Epi/EpiAnimatronics.ikg
```

`left_camera=InputImage` replaces the left eye camera with a still picture, `NoCamera.jpg`, in which
no face is found, so the distance comes from the size of the face box. Use it also on a robot whose
left eye camera does not work.

`camera_id` selects the camera when there are several, numbered as macOS lists them; the numbers
change when cameras are plugged in. To list them:

```sh
ffmpeg -f avfoundation -list_devices true -i ""
```

For example, `camera_id=1` is the built-in FaceTime camera when an external webcam is number 0.

`simulate=true` on its own still uses the robot's eye camera, which needs the robot's USB hub
connected. Without it, Ikaros stops with an error saying that `InputVideoStream` could not open the
stream. To test with a recorded video instead, give the file as `eye_url` with
`synchronized_framegrabber=true`.

Run on a robot, for example EpiBlack, which is the default:

```sh
./Bin/ikaros Projects/Epi/EpiAnimatronics.ikg
./Bin/ikaros robot=EpiWhite Projects/Epi/EpiAnimatronics.ikg
```

The model does not start by itself. Press play in the WebUI when the robot is clear of people. The
**Autonomous behaviours** switch stops the spontaneous animations and the reactions to people. Face
following, blinks, and glances continue, and the buttons still work.

The sounds play on the Mac running Ikaros. Connect a speaker near the robot for the demo.

## Settings

These model-level settings can be changed on the command line:

| Name | Default | Meaning |
| --- | --- | --- |
| `robot` | `EpiBlack` | Robot entry used by `EpiServos`, for example `EpiWhite`, `EpiRed`, `EpiGreen`, or `EpiOrange` |
| `simulate` | `false` | Simulate the servos and LEDs |
| `camera` | `InputVideoStream` | Camera module: `InputVideoStream` reads `eye_url`; `InputVideo` uses a camera on the Mac |
| `camera_id` | `0` | Mac camera used with `camera=InputVideo`, numbered as macOS lists them |
| `eye_url` | `http://RightEye.local:8080/stream/video.mjpeg` | Eye camera stream, or a video file, used with `camera=InputVideoStream` |
| `left_camera` | `InputVideoStream` | Left camera module: `InputVideoStream` reads `left_eye_url`; `InputImage` uses a still picture without faces when there is no left camera |
| `left_eye_url` | `http://LeftEye.local:8080/stream/video.mjpeg` | Left eye camera stream, or a video file |
| `camera_eye` | `right` | Eye that carries the camera; must match `eye_url` |
| `mirror_sign` | `-1` | `-1` mirrors the person in copycat like a mirror image; `1` copies the movement with the same side |
| `mirror_gain` | `0.5` | Robot head movement per degree of the person's head movement in copycat |
| `pupil_limit_steps` | `12, 244, 45, 320` | Open and closed servo positions of the left and then the right pupil, passed to `EpiServos.pupil_limits`; measured by hand on EpiBlack, see the pupil check below |
| `light_colors` | `random` | Color of the light shows: `random`, one hue per pattern, or `arousal`, from calm blue to excited orange |
| `vergence_sign` | `-1` | Direction of vergence; `-1` makes positive vergence turn the eyes inward on Epi, where a positive eye pan turns the eye to the robot's left |
| `x_sign` | `-1` | Direction from image x to head and eye pan; `-1` was found on EpiBlack |
| `y_sign` | `1` | Direction from image y to head tilt; `1` was found on EpiBlack |
| `aim_y` | `0.3` | Where the face is kept vertically; positive values look at the eyes rather than the middle of the face |
| `horizontal_fov` | `62.2` | Horizontal camera field of view in degrees |
| `vertical_fov` | `48.8` | Vertical camera field of view in degrees |
| `synchronized_framegrabber` | `false` | Grab frames in step with the ticks; set to `true` when testing with a video file |
| `synchronized_tick` | `false` | Wait for each camera frame; `false` keeps the motion steady |
| `tick_duration` | `0.04` | Seconds per tick |
| `real_time` | `true` | Run in real time |

The character is tuned on the modules:

| Module | Settings | Effect |
| --- | --- | --- |
| `Behaviors` | `min_interval`, `max_interval`, `excited_interval_scale` | How often Epi does something, and how much sooner when excited |
| `Behaviors` | `idle_weights`, `engaged_weights`, `arousal_levels`, `arousal_width` | Which animations it prefers alone, with company, and in each mood |
| `Behaviors` | `memory`, `cooldown` | How long before an animation can come back |
| `Arousal` | `idle_level`, `engaged_level`, `boosts`, `drive_gain`, `decay_rate` | How excitable Epi is and how fast it calms down |
| `Arousal` | `low`, `high` | Tempo, size, volume, and pupil offset at arousal 0 and 1 |
| `Animator` | `speed_variation`, `intensity_variation`, `mirror_channels` | How much each play varies |
| `Blinker`, `EyeGlances`, `HeadGlances` | intervals and sizes | The background life |
| `Mirror` | `max_pan`, `max_tilt`, dead zones, `engage_time` | The copycat game; see `EpiHeadMirroring.md` |
| `Boredom` | `min_wait`, `max_wait`, `min_duration`, `max_duration` | How long Epi must be alone before a light show, and how long the show lasts |
| `LightShow` | `show_patterns`, `pattern_time`, `wave_rate`, `circle_rate`, `brightness` | The light patterns |
| `Near` | `start_distance`, `full_distance`, `gain`, `max_vergence`, `object_width` | When and how strongly the eyes converge on a close face |
| `NearPupil` | `factor` | How much the pupils narrow when a face is close, in mm (negative) |
| `HeadPan` | `amplitude`, `swings`, `window` | How large and quick a back-and-forth head pan must be to start copycat |

The camera settings in `EpiFaceFollowing.md` (640 x 480, 30 frames per second, JPEG quality about
70) apply here too.

## Dashboard

- **Autonomous behaviours** turns the spontaneous animations and the reactions to people on and off.
- **Faces**, **Locked**, **Animation**, **Next in**, and **Arousal** show how many faces passed the
  filters, whether the gaze is locked, the playing animation (-1 for none), the quiet seconds left
  before the next spontaneous animation, and the excitement level.
- The eleven numbered buttons play each animation at once.
- **Wave (5 s)** and **Circle (5 s)** play a light pattern, and the other buttons in those rows run the
  LED orientation check. **Lights** shows the playing pattern (0 none, 1 wave, 2 circle, 3 check),
  and **LEDs saved** whether the robot's LED orientation is stored.
- The camera view shows all faces in green and the followed face in magenta.
- The robot view shows Epi from the front, with its left eye on your right: the eye directions,
  including vergence, the pupils, the head, and every LED, so the eyelid sweep is visible.
- The plots show the eye and head goals, the pupil goals, and the arousal.
- At the bottom, **Face distance** shows how far away the followed face is, **Distance measured by
  both eyes** whether it comes from the two cameras (1) or from the face size (0), and **Near** how
  close the face is; above 0.3 the head holds still. The left eye camera is shown beside them.

## Changing the animations

The animations are in `EpiAnimations.json`; the format is described in the `KeyframeAnimator`
documentation. The channels are, in this order: `eye_offset` and `vergence` in degrees,
`head_tilt` and `head_pan` in degrees, `pupil` in mm added to the pupil size, `eye_r`, `eye_g`,
`eye_b`, and `eye_mix` for the eye color overlay, `mouth_r`, `mouth_g`, `mouth_b`, and `mouth_mix`
for the mouth, `brightness` as a factor, `lid` from 0 (open) to 1 (closed), `mirror`, which turns
head mirroring on while it is 1, and `hold`, which stops blinks while it is 1. Arousal scales the
movement channels (eye offset, vergence, head, pupil, and lid) but not the colors, and mirrors
`eye_offset` and `head_pan` at random.

To add an animation, append it to the list, raise `count` on `Behaviors`, add a weight for it to
`idle_weights` and `engaged_weights` and an energy to `arousal_levels`, and add a button. End the
`eye_mix` and `mouth_mix` tracks at 0 within the animation, so the colors fade out before the
channels return to rest. For the mouth to flash in time with a sound, put keys on the sound's
syllable onsets; `render_sounds.py` lists the onsets of the giggle and the "he-he-heh".

The sounds are in `Sounds/Animatronics`, with a description of how they were made and how to make
new ones. `events` on `Animator` lists `blink` and the 24 sound takes, `hi_1` to `hehe_3`; event 0
triggers a blink, and the others play the sounds in the order of `sounds` on `Sounds`. In
`EpiAnimations.json` a sound event names its three takes as alternatives, with a probability, for
example `[0.5, ["hi_1", "hi_2", "hi_3"], 0.7]`.

## Before the demo

Check these on each robot, first with the room empty, then with one person in front of it. Moving
the robot needs someone next to it.

1. **LED orientation.** On a robot's first run, do the LED orientation check described above.
2. **Looking directions.** As in `EpiFaceFollowing.md`, check `x_sign` and `y_sign` first.
3. **Vergence direction.** Press **2 Cross-eyed**. Both eyes must turn inward, toward the nose.
   Then press **5 Mind blown**, where the eyes must turn outward. EpiBlack needed `vergence_sign=-1`;
   if a robot does the opposite, start it with `vergence_sign=1`.
4. **Eyelid.** Watch a blink closely, or press **4 Giggle**, which half closes the eyes. The upper
   LEDs of each eye must go dark first. The eyelid is drawn in the dashboard's layout and mapped to
   the robot by the LED orientation, so if the wrong LEDs go dark, repeat the LED check (step 1).
5. **Eye darts.** Press **3 Eye dart** and **8 Mischief**. The eyes should swing visibly. If they
   barely move, the eye servos are slower than the animation; make the steps in
   `EpiAnimations.json` longer.
6. **Head gestures.** Press **1 Greet**, **4 Giggle**, and **9 Look around**. The nod, the bobs, and
   the turns must be visible but not jerky. If the tilt shakes, lower `tilt_acceleration_scale` on
   `Gaze`, or make the head tracks smaller in `EpiAnimations.json`.
7. **Nod direction.** In the greeting, the head should first lift a little and then dip down. If it
   does the opposite, the tilt direction of the robot is the opposite of what the animations
   assume; change the sign of the `head_tilt` values in `EpiAnimations.json`.
8. **Pupils.** The pupil servos need 6.0 to 8.4 V; EpiBlack had 5.3 V before its regulator was
   repaired and now has 7.5 V. The model runs without pupils when they fail to start
   (`require_pupils="false"`), but found you, giggle, and mind blown look much better with working
   pupils. `EpiServos` ignores limits set in Dynamixel Wizard. The demo gives each pupil fixed open
   and closed positions in `pupil_limit_steps`, so no pupil is driven to a stop at start-up and the
   servos refuse to go beyond those ends. To measure them, move each pupil by hand, with torque off,
   between its two stops and note the servo positions (1 step is 0.29 degrees); use about 12 steps
   inside each stop, and the smaller closed reading if measurements differ. On EpiBlack the left
   pupil opened at 0 and closed at 256 to 286, and the right opened at 33 and closed at 332 to 379.
   The gentle automatic calibration (`EpiServos.pupil_ranges` with `pupil_calibration_torque`)
   stopped 23 steps short of the left pupil's open end, so the fixed limits are more accurate here.
9. **Sound.** Play every animation and set the speaker volume so that the calm sounds are still
   heard; arousal plays them at 35 to 100 percent.
10. **Close up.** Lean slowly toward Epi's face. Beyond about 1 m the eyes should stay parallel,
   and from there they should cross more and more, to about 27 degrees at 0.3 m, while the head
   stays still. **Distance measured by both eyes** should show 1 once both cameras see you; compare
   **Face distance** with a tape measure. If the measured distance is off, check `baseline` and the
   field of view on `Stereo`; if it is 0 although both cameras see you, the two cameras chose
   different faces or one camera's view is mirrored. Without the left camera, the distance comes from
   the face size; adjust `object_width` or the distances on `Near` if convergence then starts too
   early or too late. The **Eye goal** plot shows the two eyes moving apart in opposite directions.
11. **Copycat.** Face Epi and pan your head left, right, and left again within about 3 seconds; the
   game should start. Then turn and nod your head slowly. Epi should mirror you as described in
   `EpiHeadMirroring.md`, whose checks for `mirror_sign` and the nodding direction apply here too.
   If panning does not start the game, lower `amplitude` on `HeadPan`; if it starts by accident,
   raise it or `swings`. The game cannot restart within 25 seconds of its start.
12. **Pacing and mood.** Watch for a few minutes alone and with company. Raise `min_interval` and
   `max_interval` on `Behaviors` if Epi is too busy, and lower them if it is too calm. Lower
   `decay_rate` on `Arousal` if the excitement fades too fast.

## Known limitations

- **Untested on the robot.** The model has been checked in simulation with a scripted person, not
  on a robot. The head tilt sign, the LED ring layout, the motion speeds, and the pacing must be
  checked as listed above.
- **Tempo and sound.** Arousal changes the tempo of the motion by up to 20 percent but not the
  sounds, so at the extremes the mouth flashes drift a little from the chirps.
- **Reactions interrupt.** When the person leaves, curious starts at once, even in the middle of
  another animation, for example between the wow and the pooof of mind blown.
- **One sound at a time.** `SoundOutput` plays one sound at a time. A sound that is started while
  another plays waits until the first has ended, and only the most recent waiting sound is kept.
- **Combined head limits.** The animation and glance head offsets are limited separately from the
  face-following movement, so the head goal can change up to twice as fast while both move.
- **Mac camera.** With `camera=InputVideo` the camera is not in Epi's eye, so the face moves in
  the image only when you move, not when the simulated head turns. Unless your face is near the
  middle of the image, the simulated head and eyes keep turning toward it until they reach their
  limits. This is fine for watching the behaviours, but not for tuning the gaze.
- **Camera.** The right eye camera steers the gaze; the left one only measures distance. During
  cross-eyed, eye darts, mischief, and glances, the right camera points away from the face for a
  moment, so the face can be lost briefly.
- **Close faces.** While a face is close the head holds still, so a close person who moves far to
  the side can leave the eyes' range; the head follows again once they step back. The eyes cannot
  look up or down, so a close face that moves up or down can also leave the view.
- **Two faces.** Each camera follows the largest face it sees. With several people close by, the two
  cameras can pick different faces; detections at clearly different heights are ignored, but two
  people at the same height could give a wrong distance.
- **Simulation.** Simulated servos start at 0 degrees and move at 45 degrees per second, so the
  first seconds and fast eye movements look different from the robot.

## Sources

- [Design and Control of a Bipedal Robotic Character (Disney Research)](https://la.disneyresearch.com/publication/design-and-control-of-a-bipedal-robotic-character/):
  an animation engine that blends background idle motion with triggered animations and
  synchronizes show functions with the motion.
- [Disney Research's interactive gaze robot](https://blooloop.com/news/disney-interactive-robot-gaze/):
  layering simple motor behaviours into lifelike attention.
- The robot sound sources are listed in `Sounds/Animatronics/ReadMe.md`.

# Epi animatronics

`EpiAnimatronics.ikg` makes an Epi robot look alive, playful, and curious. It never sits still: it
blinks, glances, and looks around the room. When it finds someone new it perks up and giggles,
and while it has company it plays little reactions: a sly look to the side with a "he-he-heh", a
cross-eyed moment with blue eyes, a giggle, or being "mind blown" with eyes that move apart, a
rising whistle of surprise, and a "pooof". It speaks only in robot chirps and whistles, and an
excitement level, its arousal, makes it livelier and louder when something happens and calmer when
nothing does.

The model works with every Epi robot that `EpiServos` knows. Choose the robot with `robot`.

## How it works

The behaviour is built in layers, the way Disney's robot characters combine continuous background
motion, triggered animations on top, and "show functions" such as eyes, lights, and sound that are
synchronized with the motion:

1. **Seeing.** `InputVideoStream`, `AppleVisionFaceDetector`, and `FaceTargetSelector` find the
   closest face in the right eye camera, as in `EpiHeadMirroring.ikg`.
2. **Looking.** `GazeLock` keeps the eyes on the face and turns the head after it. `camera_eye`
   tells it that the camera is in the right eye, so the face direction stays correct while the eyes
   cross or dart.
3. **Background life.** `Blinker` blinks every 1.5 to 4.5 seconds, sometimes twice, as an eyelid
   that closes from the top of the LED ring down. `EyeGlances` makes quick glances away and back, and
   `HeadGlances` slowly looks around while nobody is followed, which also lets the camera find new
   faces. Both pause while an animation plays; a blink that falls due during an animation follows
   right after it, as people blink after a surprise.
4. **Mood.** `Arousal` keeps an excitement level from 0 to 1. It rests low when Epi is alone and a
   little higher with company. Finding someone new raises it a lot, a face that comes close raises
   it, and each reaction raises it a little. It then falls back over about half a minute. Arousal
   sets the tempo of the animations (0.8 to 1.15 times), their size (0.8 to 1.2 times), the sound
   volume, and the pupils, which open with excitement as people's do.
5. **Choosing.** `BehaviorSelector` decides when Epi does what. When someone new appears it plays
   *found you*, and when the person leaves, *curious*. In between it starts an animation after 5 to
   10 quiet seconds, sooner when excited. The choice follows the mood: each animation has an energy
   level, and the ones that match the present arousal are preferred, so Epi is lively right after
   meeting someone and calmer later. The last three animations are avoided, and none comes back
   within 15 seconds.
6. **Performing.** `KeyframeAnimator` plays the animations from `EpiAnimations.json` on 15
   channels: eye offset, vergence, head tilt and pan, pupil, eye and mouth colors, brightness, and
   eyelid. Each play varies a little in tempo and size, and the sideways movements are mirrored at
   random, so the same animation glances left one time and right the next.
7. **Body.** `EpiExpression` shows the state and the animation on the LEDs and pupils. `SoundOutput`
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
| 0 | Found you | A noticing blink and a still moment, then it perks up (pupils open, a lift of the head, a warm glow) and giggles with flashing mouth and bobbing head | sparkle, then giggle | 0.8 |
| 1 | Greet | A small lift, a nod on the chirp, a springy return, warm eyes and a smile | bouncy chirps that end high | 0.5 |
| 2 | Cross-eyed | A tiny outward flick, then the eyes meet at the nose and turn blue, pupils tighten, head dips | a hum, then a rising, questioning chirp | 0.6 |
| 3 | Eye dart | A small counter-move, then four swings from side to side with flickering cyan eyes | | 0.5 |
| 4 | Giggle | Mouth flashes and pupils pulse on every chirp, the head bobs, eyes squint happily in pink | quick chirps hopping between high and low | 0.75 |
| 5 | Mind blown | It gathers (squint, small pupils, chin down), then the eyes move apart and the head lifts with the rising whistle; a flash and a blackout on the pooof, a dazed wobble, and a purple recovery | rising whistle, then a sputtering power-down and a puff | 0.9 |
| 6 | Curious | The eyes lead, the head follows to peek to the side and up, the pupils slowly open, teal eyes | a hum, then a rising, questioning chirp | 0.3 |
| 7 | Shy | Looks down and away with pink eyes and mouth, peeks back once, half-closed eyes | a soft whistle that lifts and sinks away | 0.35 |
| 8 | Mischief | The eyes slide slowly to the side under half-closed lids, purple glow, a sly laugh with a flash and a bob on each syllable, then a quick, innocent look back with wide eyes | a sly "he-he-heh" | 0.55 |
| 9 | Look around | The eyes lead each turn and the head follows slowly to one side and the other | | 0.15 |

Alone, Epi mostly looks around and peeks curiously, with an occasional eye dart, cross-eyed moment,
or bit of mischief. With company it mixes mischief, cross-eyed, giggle, greet, shy, eye dart, and
mind blown, weighted by the mood.

Every animation can also be played with the buttons on the dashboard.

## LEDs

| State | Eyes | Mouth |
| --- | --- | --- |
| Nobody followed | Dim amber | Dim |
| Looking at someone | Bright warm white, pupils open | Dim |
| Animation | The animation's colors on top | The animation's colors on top |

## Running

Without the robot, simulate the motors and LEDs and use the Mac's own camera. Epi then follows
your face on the dashboard and plays its animations and sounds:

```sh
./Bin/ikaros simulate=true camera=InputVideo Projects/Epi/EpiAnimatronics.ikg
```

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
| `camera_eye` | `right` | Eye that carries the camera; must match `eye_url` |
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

The camera settings in `EpiFaceFollowing.md` (640 x 480, 30 frames per second, JPEG quality about
70) apply here too.

## Dashboard

- **Autonomous behaviours** turns the spontaneous animations and the reactions to people on and off.
- **Faces**, **Locked**, **Animation**, **Next in**, and **Arousal** show how many faces passed the
  filters, whether the gaze is locked, the playing animation (-1 for none), the quiet seconds left
  before the next spontaneous animation, and the excitement level.
- The ten numbered buttons play each animation at once.
- The camera view shows all faces in green and the followed face in magenta.
- The robot view shows Epi from the front, with its left eye on your right: the eye directions,
  including vergence, the pupils, the head, and every LED, so the eyelid sweep is visible.
- The plots show the eye and head goals, the pupil goals, and the arousal.

## Changing the animations

The animations are in `EpiAnimations.json`; the format is described in the `KeyframeAnimator`
documentation. The channels are, in this order: `eye_offset` and `vergence` in degrees,
`head_tilt` and `head_pan` in degrees, `pupil` in mm added to the pupil size, `eye_r`, `eye_g`,
`eye_b`, and `eye_mix` for the eye color overlay, `mouth_r`, `mouth_g`, `mouth_b`, and `mouth_mix`
for the mouth, `brightness` as a factor, and `lid` from 0 (open) to 1 (closed). Arousal scales the
movement channels (eye offset, vergence, head, pupil, and lid) but not the colors, and mirrors
`eye_offset` and `head_pan` at random.

To add an animation, append it to the list, raise `count` on `Behaviors`, add a weight for it to
`idle_weights` and `engaged_weights` and an energy to `arousal_levels`, and add a button. End the
`eye_mix` and `mouth_mix` tracks at 0 within the animation, so the colors fade out before the
channels return to rest. For the mouth to flash in time with a sound, put keys on the sound's
syllable onsets; `render_sounds.py` lists the onsets of the giggle and the "he-he-heh".

The sounds are in `Sounds/Animatronics`, with a description of how they were made and how to make
new ones. The events `blink`, `hi`, `wow`, `poof`, `giggle`, `hmm`, `aww`, `sparkle`, and `hehe` are
listed in `events` on `Animator`. Event 0 triggers a blink, and the others play the sounds in the
order of `sounds` on `Sounds`.

## Before the demo

Check these on each robot, first with the room empty, then with one person in front of it. Moving
the robot needs someone next to it.

1. **Looking directions.** As in `EpiFaceFollowing.md`, check `x_sign` and `y_sign` first.
2. **Vergence direction.** Press **2 Cross-eyed**. Both eyes must turn inward, toward the nose.
   Then press **5 Mind blown**, where the eyes must turn outward. EpiBlack needed `vergence_sign=-1`;
   if a robot does the opposite, start it with `vergence_sign=1`.
3. **Eyelid.** Watch a blink closely, or press **4 Giggle**, which half closes the eyes. The upper
   LEDs of each eye must go dark first. If the wrong LEDs go dark, set `led_first_angle` and
   `led_direction` on `Expression` as described in the `EpiExpression` documentation.
4. **Eye darts.** Press **3 Eye dart** and **8 Mischief**. The eyes should swing visibly. If they
   barely move, the eye servos are slower than the animation; make the steps in
   `EpiAnimations.json` longer.
5. **Head gestures.** Press **1 Greet**, **4 Giggle**, and **9 Look around**. The nod, the bobs, and
   the turns must be visible but not jerky. If the tilt shakes, lower `tilt_acceleration_scale` on
   `Gaze`, or make the head tracks smaller in `EpiAnimations.json`.
6. **Nod direction.** In the greeting, the head should first lift a little and then dip down. If it
   does the opposite, the tilt direction of the robot is the opposite of what the animations
   assume; change the sign of the `head_tilt` values in `EpiAnimations.json`.
7. **Pupils.** EpiBlack's pupil servos have measured 5.3 V, below their 6.0 V minimum. The model
   runs without pupils when they fail to start (`require_pupils="false"`), but found you, giggle,
   and mind blown look much better with working pupils. Check the pupil supply.
8. **Sound.** Play every animation and set the speaker volume so that the calm sounds are still
   heard; arousal plays them at 35 to 100 percent.
9. **Pacing and mood.** Watch for a few minutes alone and with company. Raise `min_interval` and
   `max_interval` on `Behaviors` if Epi is too busy, and lower them if it is too calm. Lower
   `decay_rate` on `Arousal` if the excitement fades too fast.

## Known limitations

- **Untested on the robot.** The model has been checked in simulation with a scripted person, not
  on a robot. The head tilt sign, the LED ring layout, the motion speeds, and the pacing must be
  checked as listed above.
- **Tempo and sound.** Arousal changes the tempo of the motion by up to 20 percent but not the
  sounds, so at the extremes the mouth flashes drift a little from the chirps.
- **One sound at a time.** `SoundOutput` plays one sound at a time. A sound that is started while
  another plays waits until the first has ended, and only the most recent waiting sound is kept.
- **Combined head limits.** The animation and glance head offsets are limited separately from the
  face-following movement, so the head goal can change up to twice as fast while both move.
- **Mac camera.** With `camera=InputVideo` the camera is not in Epi's eye, so the face moves in
  the image only when you move, not when the simulated head turns. Unless your face is near the
  middle of the image, the simulated head and eyes keep turning toward it until they reach their
  limits. This is fine for watching the behaviours, but not for tuning the gaze.
- **Camera.** Only the right eye camera is used. During cross-eyed, eye darts, mischief, and
  glances, the camera points away from the face for a moment, so the face can be lost briefly.
- **Simulation.** Simulated servos start at 0 degrees and move at 45 degrees per second, so the
  first seconds and fast eye movements look different from the robot.

## Sources

- [Design and Control of a Bipedal Robotic Character (Disney Research)](https://la.disneyresearch.com/publication/design-and-control-of-a-bipedal-robotic-character/):
  an animation engine that blends background idle motion with triggered animations and
  synchronizes show functions with the motion.
- [Disney Research's interactive gaze robot](https://blooloop.com/news/disney-interactive-robot-gaze/):
  layering simple motor behaviours into lifelike attention.
- The robot sound sources are listed in `Sounds/Animatronics/ReadMe.md`.

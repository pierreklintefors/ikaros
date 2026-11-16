# Epi animatronics

`EpiAnimatronics.ikg` makes an Epi robot look alive and playful. It never sits still: it blinks
often, glances around, and looks curiously around the room when nobody is there. When someone
comes close it follows their face, greets them, and now and then plays a little reaction: it goes
cross-eyed with blue eyes, giggles with pulsing pupils, or is "mind blown" with diverging eyes, a
rising whistle of surprise, and a "pooof". It speaks only in robot chirps and whistles.

The model works with every Epi robot that `EpiServos` knows. Choose the robot with `robot`.

## How it works

1. `InputVideoStream`, `AppleVisionFaceDetector`, and `FaceTargetSelector` find the closest face
   in the right eye camera, as in `EpiHeadMirroring.ikg`.
2. `GazeLock` keeps the eyes on the face and turns the head after it. `camera_eye="right"` tells it
   that the camera is in the right eye, so the face direction stays correct while the eyes cross or
   dart.
3. `BehaviorSelector` decides when Epi does what. It greets every new person, says goodbye with a
   curious look when the person leaves, and starts a spontaneous animation after 2.5 to 6 quiet
   seconds. Animations are chosen at random with different weights for being alone and for having
   company, and the same one does not come back within 8 seconds.
4. `KeyframeAnimator` plays the animations from `EpiAnimations.json`. Each animation moves a set of
   channels: eye offset, vergence, head tilt and pan, pupil size, eye and mouth colors, brightness,
   and eyelid. It also fires events that start sounds and blinks.
5. `Blinker` blinks every 1.5 to 4.5 seconds, sometimes twice. The blink is shown on the eye LEDs
   like an eyelid that closes from the top down. Blinks pause during an animation and come right
   after it, as people blink after a surprise.
6. Two `GlanceGenerator` modules keep Epi moving. `EyeGlances` makes quick glances away and back.
   `HeadGlances` slowly looks around the room while nobody is followed, which also lets the camera
   find new faces.
7. `EpiExpression` shows the state and the animation on the LEDs and pupils. `SoundOutput` plays
   the sounds, and `EpiServos` and `FadeCandy` drive the hardware.

All modules apart from `EpiServos`, `EpiExpression`, and `FadeCandy` are general and can be used
with other robots.

## Animations

| # | Name | What Epi does | Sound |
| ---: | --- | --- | --- |
| 0 | Greet | Double blink, warm sparkling eyes, pupils open, a small nod and a smile | climbing blips, then bouncy chirps that end high |
| 1 | Cross-eyed | Eyes converge 22 degrees and turn blue, pupils contract, head dips a little | a hum, then a rising, questioning chirp |
| 2 | Eye dart | Eyes swing quickly from side to side three times with flickering cyan eyes | |
| 3 | Giggle | Pupils pulse, eyes squint happily in pink, the mouth flashes, the head bobs | quick chirps hopping between high and low |
| 4 | Mind blown | Eyes diverge and glow white, pupils open fully, head leans back; then a flash, a blackout, small pupils, a dazed wobble, and a purple recovery | a whistle climbing from low to high, then a sputtering power-down and a puff |
| 5 | Curious | Eyes lead, head peeks to the side and up, pupils slowly open, teal eyes | a hum, then a rising, questioning chirp |
| 6 | Shy | Looks down and away with pink eyes and mouth, peeks back once, half-closed eyes | a soft whistle that lifts and sinks away |

When nobody is followed, Epi mostly plays curious and eye dart, and sometimes cross-eyed. With
company it plays cross-eyed, giggle, shy, mind blown, and eye dart. The greeting plays for each new
person, and curious plays when the person leaves.

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
**Autonomous behaviours** switch stops the spontaneous animations and the greetings. Face following,
blinks, and glances continue, and the buttons still work.

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

How often things happen is set on the modules: `min_interval`, `max_interval`, the weights, and
`cooldown` on `Behaviors`; the blink intervals on `Blinker`; and the glance intervals and sizes on
`EyeGlances` and `HeadGlances`.

The camera settings in `EpiFaceFollowing.md` (640 x 480, 30 frames per second, JPEG quality about
70) apply here too.

## Dashboard

- **Autonomous behaviours** turns the spontaneous animations, greetings, and farewells on and off.
- **Faces**, **Locked**, **Animation**, **Next in**, and **Eyelid** show how many faces passed the
  filters, whether the gaze is locked, the playing animation (-1 for none), the quiet seconds left
  before the next spontaneous animation, and the eyelid closure.
- The numbered buttons play each animation at once.
- The camera view shows all faces in green and the followed face in magenta.
- The robot view shows Epi from the front, with its left eye on your right: the eye directions,
  including vergence, the pupils, the head, and every LED, so the eyelid sweep is visible.
- The plots show the eye and head goals, the pupil goals, and the eyelid.

## Changing the animations

The animations are in `EpiAnimations.json`; the format is described in the `KeyframeAnimator`
documentation. The channels are, in this order: `eye_offset` and `vergence` in degrees,
`head_tilt` and `head_pan` in degrees, `pupil` in mm added to the pupil size, `eye_r`, `eye_g`,
`eye_b`, and `eye_mix` for the eye color overlay, `mouth_r`, `mouth_g`, `mouth_b`, and `mouth_mix`
for the mouth, `brightness` as a factor, and `lid` from 0 (open) to 1 (closed).

To add an animation, append it to the list, raise `count` on `Behaviors`, add a weight for it to
`idle_weights` and `engaged_weights`, and add a button. End the `eye_mix` and `mouth_mix` tracks at
0 within the animation, so the colors fade out before the channels return to rest.

The sounds are in `Sounds/Animatronics`, with a description of how they were made and how to make
new ones. The events `blink`, `hi`, `wow`, `poof`, `giggle`, `hmm`, `aww`, and `sparkle` are listed
in `events` on `Animator`. Event 0 triggers a blink, and the others play the sounds in the order of
`sounds` on `Sounds`.

## Before the demo

Check these on each robot, first with the room empty, then with one person in front of it. Moving
the robot needs someone next to it.

1. **Looking directions.** As in `EpiFaceFollowing.md`, check `x_sign` and `y_sign` first.
2. **Vergence direction.** Press **1 Cross-eyed**. Both eyes must turn inward, toward the nose.
   Then press **4 Mind blown**, where the eyes must turn outward. EpiBlack needed `vergence_sign=-1`;
   if a robot does the opposite, start it with `vergence_sign=1`.
3. **Eyelid.** Watch a blink closely, or press **3 Giggle**, which half closes the eyes. The upper
   LEDs of each eye must go dark first. If the wrong LEDs go dark, set `led_first_angle` and
   `led_direction` on `Expression` as described in the `EpiExpression` documentation.
4. **Eye darts.** Press **2 Eye dart**. The eyes should swing visibly from side to side. If they
   barely move, the eye servos are slower than the animation; make the steps in
   `EpiAnimations.json` longer.
5. **Head gestures.** Press **0 Greet** and **3 Giggle**. The nod and the bobs must be visible but
   not jerky. If the tilt shakes, lower `tilt_acceleration_scale` on `Gaze`, or make the head tracks
   smaller in `EpiAnimations.json`.
6. **Nod direction.** In the greeting, the head should first dip down. If it first looks up, the
   tilt direction of the robot is the opposite of what the animations assume; change the sign of
   the `head_tilt` values in `EpiAnimations.json`.
7. **Pupils.** EpiBlack's pupil servos have measured 5.3 V, below their 6.0 V minimum. The model
   runs without pupils when they fail to start (`require_pupils="false"`), but the giggle and mind
   blown animations look much better with working pupils. Check the pupil supply.
8. **Sound.** Play every animation and set the speaker volume.
9. **Pacing.** Watch for a few minutes alone and with company. Raise `min_interval` and
   `max_interval` on `Behaviors` if Epi is too busy, and lower them if it is too calm.

## Known limitations

- **Untested on the robot.** The model has been checked in simulation with a scripted person, not
  on a robot. The signs of vergence and head tilt, the LED ring layout, and the motion speeds must
  be checked as listed above.
- **One sound at a time.** `SoundOutput` plays one sound at a time. A sound that is started while
  another plays waits until the first has ended, and only the most recent waiting sound is kept.
- **Combined head limits.** The animation and glance head offsets are limited separately from the
  face-following movement, so the head goal can change up to twice as fast while both move.
- **Mac camera.** With `camera=InputVideo` the camera is not in Epi's eye, so the face moves in
  the image only when you move, not when the simulated head turns. Unless your face is near the
  middle of the image, the simulated head and eyes keep turning toward it until they reach their
  limits. This is fine for watching the behaviours, but not for tuning the gaze.
- **Camera.** Only the right eye camera is used. During cross-eyed, eye darts, and glances, the
  camera points away from the face for a moment, so the face can be lost briefly.
- **Simulation.** Simulated servos start at 0 degrees and move at 45 degrees per second, so the
  first seconds and fast eye movements look different from the robot.

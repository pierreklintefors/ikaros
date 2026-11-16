# Epi head mirroring

`EpiHeadMirroring.ikg` makes an Epi robot look at the closest person and copy that person's head
movements as a mirror image. When the person turns their head to their left, the robot turns to its
right, and when they nod, it nods. The robot keeps looking at the person while it moves its head. It
is made for a room with many people, including children who come close and run around, and it
hands over smoothly from one person to another.

The model works with every Epi robot that `EpiServos` knows. Choose the robot with `robot`.

## How it works

1. `InputVideoStream` reads one eye camera, the right eye by default.
2. `AppleVisionFaceDetector` finds all faces and estimates the head pose of each: yaw (turning),
   pitch (nodding), and roll. It uses Apple's Vision framework, so no extra libraries are needed.
3. `FaceTargetSelector` chooses the largest face, which is usually the closest person. Another face
   takes over only when it is at least 1.5 times as wide for 0.7 seconds, so a child who runs past
   does not take the robot's attention. It passes on the pose of the chosen face.
4. `HeadMirror` waits until the person has been followed for one second, is close enough, and
   roughly faces the robot. It then fades in a mirrored head offset: half of the person's yaw and
   pitch, limited to 10 degrees. When another person takes over or the person leaves, the offset
   fades out completely before the robot starts mirroring anyone else.
5. `GazeLock` keeps the eyes on the person and moves the head smoothly, limiting its speed and
   acceleration. While the robot mirrors, the head follows the mirrored goal from `HeadMirror`, and
   the eyes turn the other way so that the robot still looks at the person.
6. `EpiExpression` shows the state on the LEDs, and `EpiServos` and `FadeCandy` drive the hardware.

## When the face is lost

The camera sits in the right eye, so the robot must not turn so far that the face leaves the image.
The mirrored offset is therefore limited to 10 degrees of pan, which the eyes can compensate, and
10 degrees of tilt, which keeps the face inside the image. If the face is lost anyway, for example
because the person moves quickly or turns their head too far for the detector:

1. For about half a second nothing changes. `FaceTargetSelector` holds the target and `HeadMirror`
   keeps the last pose, so single missed frames do not matter.
2. Then the person counts as lost. The mirrored offset fades out, and the head returns to the
   direction where the person was last seen, not to the mirrored pose. The camera then points at the
   person again, the face usually reappears, and mirroring restarts after `engage_time`.
3. If no face is seen for 1.5 seconds (`hold_time` on `Gaze`), the robot stops looking at that
   place. Its head and eyes return slowly to straight ahead and the LEDs fade back to idle.

When the robot looks up, a very close face that is kept low in the image comes nearest to the
bottom edge. Lower `max_tilt` on `Mirror` if close children are lost when they look up.

## LEDs

| State | Eyes | Mouth |
| --- | --- | --- |
| Nobody followed | Dim amber | Off |
| Looking at someone | Blue, pupils open | Dim |
| Mirroring | Green | Bright warm white |

All changes fade, so the LEDs do not flicker when a face is missed for a moment.

## Running

Start with simulated motors and LEDs to check the camera and the dashboard:

```sh
./Bin/ikaros simulate=true Projects/Epi/EpiHeadMirroring.ikg
```

Run on a robot, for example EpiWhite:

```sh
./Bin/ikaros robot=EpiWhite Projects/Epi/EpiHeadMirroring.ikg
```

The default robot is EpiBlack. The model does not start by itself. Press play in the WebUI when the
robot is clear of people. The **Mirror head movements** switch on the dashboard turns mirroring off.
The robot then only looks at people, like `EpiFaceFollowing.ikg`.

To use the left eye camera, for example when it works again:

```sh
./Bin/ikaros eye_url=http://LeftEye.local:8080/stream/video.mjpeg Projects/Epi/EpiHeadMirroring.ikg
```

With the left eye camera, check `x_sign` again.

## Settings

These model-level settings can be changed on the command line:

| Name | Default | Meaning |
| --- | --- | --- |
| `robot` | `EpiBlack` | Robot entry used by `EpiServos`, for example `EpiWhite`, `EpiRed`, `EpiGreen`, or `EpiOrange` |
| `simulate` | `false` | Simulate the servos and LEDs |
| `eye_url` | `http://RightEye.local:8080/stream/video.mjpeg` | Eye camera stream; every Epi uses the same host names over USB |
| `x_sign` | `-1` | Direction from image x to head and eye pan; `-1` was found on EpiBlack with the right eye camera |
| `y_sign` | `-1` | Direction from image y to head tilt |
| `aim_y` | `0.3` | Where the face is kept vertically, from -1 (top) to 1 (bottom); positive values make the robot look at the eyes rather than the middle of the face |
| `horizontal_fov` | `62.2` | Horizontal camera field of view in degrees |
| `vertical_fov` | `48.8` | Vertical camera field of view in degrees |
| `mirror_sign` | `-1` | `-1` mirrors the person like a mirror image; `1` copies the movement with the same side of the body |
| `mirror_gain` | `0.5` | Robot head movement per degree of the person's head movement |
| `synchronized_framegrabber` | `false` | Grab frames in step with the ticks; set to `true` when testing with a video file |
| `synchronized_tick` | `false` | Wait for each camera frame; `false` keeps the motion steady |
| `tick_duration` | `0.04` | Seconds per tick |
| `real_time` | `true` | Run in real time |

The camera settings in `EpiFaceFollowing.md` (640 x 480, 30 frames per second, JPEG quality about
70) apply here too.

## Dashboard

- **Mirror head movements** turns mirroring on and off.
- **Faces**, **Locked**, **State**, and **Mirror weight** show how many faces passed the filters,
  whether the gaze is locked, the mirroring state (0 nobody, 1 looking, 2 mirroring), and how far
  mirroring has faded in.
- The camera view shows all faces in green and the followed face in magenta.
- The robot view shows eye direction, pupils, head, and LED colors. Its body color is only a
  drawing color; widget settings cannot refer to `robot`.
- The plots show the followed person's head pose, the mirrored offset, the head goal, and the eye
  goals.

## Before the demo

Check these on each robot with the room empty, then with one person in front of it:

1. **Looking directions.** Stand to the robot's side. If it turns away from you, invert `x_sign`, for
   example with `x_sign=1`. If it looks up when you crouch, invert `y_sign`.
2. **Mirror direction.** Face the robot and turn your head slowly to your left. The **Person head
   pose** plot shows the yaw change. Mirrored, the robot turns to its own right, toward the same side
   of the room. If it turns the other way, invert `mirror_sign`. `mirror_sign` also changes between
   a mirror image and a copy, so check it again after changing the camera.
3. **Nodding.** Nod down. The robot should nod down too. If it looks up, change `pitch_sign` on
   `Mirror` to `1`.
4. **Pose that follows the robot.** Stand still, turn mirroring off, and move the robot's gaze by
   stepping a little to the side. If the yaw or pitch in the pose plot changes although your head
   did not turn, set `view_compensation` on `Mirror` to `1` or `-1`, whichever keeps the pose
   steadier.
5. **Range.** Turn your head as far as is comfortable. The robot should stay within about 10 degrees
   and keep looking at you. If it loses you at the extremes, lower `max_pan` on `Mirror`. Raise
   `mirror_gain` for larger movements.
6. **Handover.** Let a second person step closer than the first. The robot should fade out
   mirroring, turn to the new person, and start mirroring them after about a second. Adjust
   `switch_ratio` and `switch_delay` on `Selector` and `engage_time` on `Mirror` for the room.
7. **Distance.** Raise `min_width` on `Mirror` if the robot mirrors people who are too far away to
   notice it, and `min_width` on `Selector` if it reacts to the background.
8. **Limits.** Confirm that the `GazeLock` limits keep the robot clear of its mechanical stops.

## Known limitations

- **One camera.** Only the right eye camera is used. On EpiBlack the left eye camera was not
  available because its Raspberry Pi did not appear as a USB network device on the Mac, so the cause
  is the cable, the hub port, the Pi's power, or the Pi's startup, not Ikaros.
- **Head pose accuracy.** Vision's pose is noisy for small or strongly turned faces. The pose is
  smoothed and mirroring only starts for people who roughly face the robot, but large head turns
  (beyond about 45 degrees) are not reliable.
- **No roll.** Epi's neck has tilt and pan but no roll, so a sideways head tilt is not mirrored.
- **Stalled camera stream.** As in `EpiFaceFollowing.ikg`, a stalled stream releases the lock after
  the hold times and the head returns slowly to straight ahead. With `synchronized_tick=true` a stalled
  stream stops the whole model.
- **Simulation.** Simulated servos start at 0 degrees and take about four seconds to reach the
  straight-ahead position; gaze directions measured in that period are wrong.

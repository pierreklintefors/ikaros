# EpiBlack face following

`EpiFaceFollowing.ikg` makes the EpiBlack torso robot follow a face with its head and eyes. When the
robot has found someone, its eye LEDs change color and its pupils open.

## How it works

1. `InputVideoStream` reads one eye camera, the right eye by default. Both eyes turn together, so
   one camera is enough to control the gaze.
2. `AppleVisionFaceDetector` finds all faces in the image.
3. `FaceTargetSelector` chooses one face to follow. It prefers the largest face, which is usually
   the closest child, and keeps following that face until another one is clearly larger or the
   followed face has been gone for half a second.
4. `GazeLock` stores the direction of that face relative to the body. The eyes point at it at once
   and the head follows in calm steps. The eyes compensate for every head movement, including manual
   ones.
5. `EpiExpression` fades the eye LEDs and pupils toward the engaged expression while the gaze is
   locked, and back when the face is lost.
6. `EpiServos` drives the motors and `FadeCandy` drives the LEDs.

## Running

Start with simulated motors and LEDs to check the camera and the dashboard:

```sh
./Bin/ikaros simulate=true Projects/Epi/EpiFaceFollowing.ikg
```

Run on the robot:

```sh
./Bin/ikaros Projects/Epi/EpiFaceFollowing.ikg
```

The model does not start by itself. Press play in the WebUI when the robot is clear of people.

To use the left eye camera instead, for example when its stream works again:

```sh
./Bin/ikaros eye_url=http://LeftEye.local:8080/stream/video.mjpeg Projects/Epi/EpiFaceFollowing.ikg
```

The model-level settings below can be changed on the command line:

| Name | Default | Meaning |
| --- | --- | --- |
| `robot` | `EpiBlack` | Robot entry used by `EpiServos` |
| `simulate` | `false` | Simulate the servos and LEDs |
| `eye_url` | `http://RightEye.local:8080/stream/video.mjpeg` | Eye camera stream that drives the gaze |
| `x_sign` | `-1` | Direction from image x to neck and eye pan; `-1` is correct for EpiBlack's right eye camera |
| `y_sign` | `-1` | Direction from image y to neck tilt |
| `aim_y` | `0.4` | Where the face is kept vertically, from -1 (top) to 1 (bottom); 0.4 keeps it in the lower part of the image so the robot looks at the eyes rather than the middle of the face |
| `synchronized_framegrabber` | `false` | Grab frames in step with the ticks; set to `true` when testing with a video file |
| `synchronized_tick` | `true` | Wait for a new camera frame on every tick |
| `tick_duration` | `0.04` | Seconds per tick |
| `real_time` | `true` | Run in real time |

## Dashboard

- **Head follows face** switches between following and manual head control. In manual mode the head
  follows the tilt and pan sliders and the eyes stay on the locked target.
- **Lock gaze** stores the current gaze direction as a target without needing a face. Turn following
  off, press it, and move the pan slider to show the eye compensation. **Release** drops the lock.
- **Faces**, **Seen**, **Locked**, and **Expression** show how many faces passed the filters,
  whether the followed face was seen this tick, whether the gaze is locked, and the expression level.
- The camera view shows all faces in green and the followed face in magenta.
- The robot view shows eye direction, pupils, and LED colors. The plots show the stored target
  direction and the head and eye goals.

## Before the demo

Check these on the robot with the room empty, then with one person:

1. **Directions.** Stand to the robot's side. If it turns away from you, invert `x_sign`, for
   example with `x_sign=1` on the command line. If it looks up when you crouch, invert `y_sign`.
2. **Eye contact.** If the robot looks below a person's eyes, raise `aim_y`, for example `aim_y=0.5`.
   If it looks above them, lower it.
3. **Overshoot.** If the head swings past the face, increase `latency` on `Gaze` in steps of 0.05 s.
   If it lags behind, decrease it.
4. **Calmness.** Raise `head_dead_zone` for fewer, larger head movements, or lower `head_gain` and
   `head_max_speed` for slower ones.
5. **Distance.** Raise `min_width` on `Selector` if the robot reacts to people far in the background.
6. **Limits.** Confirm that `tilt_min`/`tilt_max`, `pan_min`/`pan_max`, and `eye_min`/`eye_max` on
   `Gaze` keep EpiBlack clear of its mechanical stops.

## Known limitations

- **Stalled camera stream.** With `synchronized_tick="true"`, the model waits for each new frame. If
  a camera stream stops, the whole model stops, the robot holds its last position, and the program
  can only be ended by force. Check that the stream is stable before the demo. Setting
  `synchronized_tick=false` keeps the robot running but re-detects old frames.
- **Video files.** A video file ends, and the model then waits forever for the next frame. Use a
  long file and `synchronized_framegrabber=true` when testing without cameras.
- **Simulation.** Simulated servos start at 0 degrees and take about four seconds to reach the
  straight-ahead position. Gaze directions measured in that period are wrong, which only matters in
  simulation.
- **Mimicking head movements** needs a face-pose detector and is not part of this model.

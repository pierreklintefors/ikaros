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
| `horizontal_fov` | `62.2` | Horizontal camera field of view in degrees; see Camera settings |
| `vertical_fov` | `48.8` | Vertical camera field of view in degrees; see Camera settings |
| `synchronized_framegrabber` | `false` | Grab frames in step with the ticks; set to `true` when testing with a video file |
| `synchronized_tick` | `false` | Wait for a new camera frame on every tick; `false` keeps the ticks, and the motion, steady |
| `tick_duration` | `0.04` | Seconds per tick |
| `real_time` | `true` | Run in real time |

## Camera settings

The eye cameras run a UV4L streaming server. With its original settings, 1280 x 720 MJPEG at
JPEG quality 85, the right eye sent about 420 kB per frame, roughly 100 Mbit/s. That is more than
the network handles smoothly and causes the lag and stalls. Use these settings instead:

| Setting | Value |
| --- | --- |
| Resolution | 640 x 480 |
| Frame rate | 30 |
| JPEG quality | 70 |
| Format | MJPEG |

At 640 x 480 and 30 frames per second the Pi Camera Module v2 uses its full field of view, which
matches the default `horizontal_fov` and `vertical_fov` of 62.2 and 48.8 degrees.

To change the settings until the Pi restarts, close every client of the stream, including Ikaros and
browser tabs, open `http://RightEye.local:8080/panel`, enter the values, and press Apply. UV4L
cannot change the resolution while any stream is open.

To keep the settings, set these options in the UV4L raspicam configuration file on the Pi, usually
`/etc/uv4l/uv4l-raspicam.conf`, and restart the UV4L service:

```text
encoding = mjpeg
width = 640
height = 480
framerate = 30
quality = 70
```

If the camera stays at 1280 x 720, its image is 16:9 and the vertical field of view is smaller,
roughly 37 degrees at 30 frames per second. Start the model with `vertical_fov=37` in that case.

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
4. **Smoothness and speed.** `Gaze` uses `head_gain` 3, `head_max_speed` 90, and
   `head_max_acceleration` 300. Raise `head_gain` and `head_max_speed` for quicker head turns, lower
   `head_max_acceleration` for softer starts and stops, raise `head_dead_zone` for fewer, larger
   head movements, and raise `eye_smoothing` if the eyes still jitter.
5. **Distance.** Raise `min_width` on `Selector` if the robot reacts to people far in the background.
6. **Limits.** Confirm that `tilt_min`/`tilt_max`, `pan_min`/`pan_max`, and `eye_min`/`eye_max` on
   `Gaze` keep EpiBlack clear of its mechanical stops.

## Known limitations

- **Stalled camera stream.** The model ticks at a steady rate and does not wait for camera frames.
  If the stream stalls, `FaceTargetSelector` recognizes the repeated frames, the lock is released
  after the hold times, and the head returns slowly to straight ahead. With `synchronized_tick=true`
  the model instead waits for each frame, and a stalled stream stops the whole model until it is
  ended by force.
- **Video files.** A video file ends, and the model then waits forever for the next frame. Use a
  long file and `synchronized_framegrabber=true` when testing without cameras.
- **Simulation.** Simulated servos start at 0 degrees and take about four seconds to reach the
  straight-ahead position. Gaze directions measured in that period are wrong, which only matters in
  simulation.
- **Mimicking head movements** needs a face-pose detector and is not part of this model.

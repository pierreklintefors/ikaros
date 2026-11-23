# CLAUDE.md

Read and follow the project instructions in AGENTS.md before doing anything else:

@AGENTS.md

The notes below are practical facts that are easy to miss and cost time in earlier sessions.

## Ikaros framework

- `rate` parameters are delivered per tick: the kernel has already multiplied the per-second value by
  the tick duration. Use `rate.as_float()` directly, for example `1 - exp(-rate)`, and do not multiply
  by `GetTickDuration()` again.
- Release hardware in `Stop()`, not only in the destructor. The kernel is a function-local static, so
  module destructors can run during process exit, where `Notify()` and `Warning()` are unsafe.
- Module unit tests: `python3 Source/Kernel/UnitTesting/KernelTests/kernel_test.py <tests dir>
  <name> --skip-webui-js` runs every `test*.ikg` in that directory. Use `cli_args="-u${TEST_DIR}"`
  for fixtures next to the test, `InputFile` with a `.jsonl` fixture for time-varying inputs, and a
  connection `label` to prefix `Print` output. `msg_print` notifications only appear with
  `log_level="7"`. Every line of a `.jsonl` fixture needs the same row count; an empty array fails,
  so pad absent rows with zeros.
- Running the full kernel suite deletes the tracked fixture
  `Source/Kernel/UnitTesting/KernelTests/test_373_relative_state_path.state`; restore it with
  `git checkout --` afterwards. Test 239 (WebUI fatal-step) is a known pre-existing failure.
- `Print` shows dynamic outputs, such as detector `BOXES`, as empty even when they contain rows.
- WebUI widget sources take whole buffers; there is no element selection such as `X.OUTPUT[1]`.
  `switch` and slider widgets write into matrix parameters such as `Constant.data`.
- Module attributes can take group values with `@name`, for example `pitch_sign="@y_sign"`; widget
  attributes do not resolve `@` references.
- Headless Firefox screenshots only capture the WebUI loading screen; no other browser automation is
  installed. Ask the user to check WebUI layouts in a normal browser. To read outputs of a model run
  with `-w <port>`, use `curl http://localhost:<port>/data/Group.Module.OUTPUT` (CSV).
- The `ikaros-bot` Keychain token is not installed on this Mac. An empty `GH_TOKEN` makes `gh` use
  the user's own account, so check that the Keychain lookup succeeds before any `gh` command, and
  otherwise give the issue in the chat.
- `Projects/` is ignored by `Projects/.gitignore`; project files are tracked with `git add -f`.

## Epi robots and EpiBlack

- Use `EpiServos` for the Epi robots; it is the hardware-tested servo module. `DynamixelController`,
  `PIDController`, `ServoConnector`, `Arbiter`, and `HeadServos` have not been tested on the robots.
- `EpiServos` uses a 19-element vector in degrees with 180 meaning straight ahead: 0 neck tilt,
  1 neck pan, 2 left eye, 3 right eye, 4-5 pupils in mm (clipped to 5-16). `simulate="true"` starts
  every joint at 0 and moves toward the goal at 45 degrees per second.
- The robot connects to the Mac through one USB hub: the head and pupil FTDI serial adapters and both
  eye Raspberry Pis, which appear as USB Ethernet interfaces. If `/dev/cu.usbserial-*` ports and the
  cameras disappear together, the hub has been disconnected.
- Eye cameras are UV4L servers on the Pis: `http://RightEye.local:8080/stream/video.mjpeg` and
  `http://LeftEye.local:8080/stream/video.mjpeg`; settings at `/panel` on the same host. UV4L cannot
  change the resolution while any client is streaming. Use 640 x 480, 30 frames per second, and
  JPEG quality about 70; 1280 x 720 at quality 85 overloaded the link.
- `InputVideoStream` with `synchronized_tick="true"` waits forever when a stream stalls or a video
  file ends, and Ikaros then cannot be stopped normally.
- Read-only diagnostics of servo registers are safe without the user present; anything that writes
  to the servos or moves the robot needs the user present.

## Face-following and head-mirroring demos

- `Projects/Epi/EpiFaceFollowing.ikg` with its guide `EpiFaceFollowing.md`, built from
  `FaceTargetSelector`, `GazeLock`, and `EpiExpression`. Open items are in `status.md` under the
  EpiBlack sections.
- `Projects/Epi/EpiHeadMirroring.ikg` with its guide `EpiHeadMirroring.md` adds `HeadMirror`, which
  drives `GazeLock.MANUAL_HEAD` from the head pose given by `AppleVisionFaceDetector.POSES`.

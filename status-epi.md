# Epi status

Task records for the Epi robot demos on the `ikaros-epi` branch, moved here from `status.md`.


## EpiBlack face-following demo

Face following for EpiBlack with eye/head gaze compensation, LED and pupil expression, and a WebUI
dashboard. Built from `InputVideoStream`, `AppleVisionFaceDetector`, `EpiServos`, `FadeCandy`, and
`Constant` plus three new C++ modules.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Add `FaceTargetSelector`, which turns detector boxes into one stable target (largest face, identity matching, hysteresis, hold). | Completed | Release build; `tests/` switch and hold tests pass through `kernel_test.py` (largest-face selection, no switch below `switch_ratio`, switch above it, width/confidence filtering, hold then reacquire); detector-to-selector smoke run on `UserData/pierre.jpg` selects the face; Markdown checker reports no issue for the new ReadMe; `git diff --check`. | `FaceTargetSelector now selects one stable face target` |
| 2 | Add `GazeLock`, which stores the target direction in the body frame, compensates head motion with the eyes, moves the head, and outputs the `EpiServos` goal vector. | Completed | Release build; `tests/` compensation and follow tests pass through `kernel_test.py` (eyes rotate by the head displacement while locked, release after `hold_time`, head dead-zone start and stop, eye clipping, pupil input, full 19-element goal vector); 150-tick integration run with simulated `EpiServos robot="EpiBlack"` settles on the manual head goal without warnings; Markdown checker reports no issue for the new ReadMe; `git diff --check`. | `GazeLock now keeps the eyes on a target while the head moves` |
| 3 | Add `EpiExpression`, which fades eye LEDs and pupil size between idle and engaged states for `FadeCandy` and `GazeLock`. | Completed | Release build; `tests/` fade test passes through `kernel_test.py` (instant attack, per-second release independent of tick duration, eye color and pupil interpolation, simulated `FadeCandy` accepts all four LED outputs without size warnings); Markdown checker reports no issue for the new ReadMe; `git diff --check`. | `EpiExpression now animates the Epi eye LEDs and pupils` |
| 4 | Add the `Projects/Epi/EpiFaceFollowing.ikg` demo model with documentation and a WebUI dashboard. | Completed | Release build; XML parses; 300-tick simulated run (`simulate=true`, synthetic moving-face video for both eyes, `synchronized_framegrabber=true`) ends cleanly; probed run shows detections, `VALID`, lock, rising expression level, pupils opening from 5 to 15 mm, and head and eye goals moving; static layout check finds no widget overlaps, consistent `x/_x` and `y/_y`, and all widgets right of the component graph; WebUI server answers. Visual WebUI inspection was not possible: headless Firefox only captured the loading screen and no other browser automation is installed. No hardware run. | `Added the EpiBlack face-following demo` |

### Constraints

- C++ modules only; no kernel changes, Python, or external libraries.
- Keep `PIDController`, `ServoConnector`, `Arbiter`, and `HeadServos` out of the servo loop; they are untested on the robot.
- Do not include the uncommitted `InputVideoStream` changes or `.Rhistory` in any commit.
- Do not move the physical robot without the user present.

### Additional fixes found during the tasks

| Fix | Verification | Commit |
|---|---|---|
| `FaceTargetSelector` multiplied the per-tick `smoothing` rate by the tick duration a second time; it now uses the kernel's per-tick value. Found while implementing task 2. | New smoothing test passes with the other selector tests through `kernel_test.py`. | `Fixed FaceTargetSelector smoothing rate scaling` |
| `EpiServos` was not compiled because commit `ad8057dd` removed it from `Source/Modules/RobotModules/CMakeLists.txt`; the demo requires it, so it is built again when the Dynamixel SDK is found. | Release build compiles `EpiServos.cc`; 150-tick simulated `EpiBlack` run with `GazeLock`; full kernel suite passes except the pre-existing WebUI fatal-step test 239. | `EpiServos is now included in the build again` |
| The demo now uses one eye camera, the right eye by default, because the left eye stream does not connect; `eye_url` selects the camera. | 300-tick simulated run ends cleanly; probe shows detection, lock, and expression on every tick; static layout check finds no overlaps. | `EpiBlack face following now uses a single eye camera` |
| On EpiBlack the head and eyes turned away from the face; the demo now sets `x_sign=-1` and exposes `x_sign` and `y_sign` as model-level settings. | Simulated probe: a face right of center now gives a negative stored azimuth, and `x_sign=1` on the command line restores the previous direction. To be confirmed on the robot. | `EpiBlack face following now turns toward the face` |

### Outstanding issues and questions

- Partly tested on EpiBlack: face following runs and `x_sign` was inverted after the first robot run; `y_sign`, the camera field of view, `latency`, head gains, and the
  `GazeLock` limits must be checked on the robot as listed in `Projects/Epi/EpiFaceFollowing.md`.
- The dashboard has not been inspected in a browser; check alignment, clipping, and console errors.
- `InputVideoStream` with `synchronized_tick="true"` waits forever when a stream stalls or a file
  ends, and the program then cannot be stopped normally. This is a possible Ikaros issue; a GitHub
  issue has been proposed to the user but not created.
- Decide whether the demo should use `synchronized_tick=false` to survive stream dropouts at the cost
  of re-detecting old frames.
- `EpiServos` simulation starts at 0 degrees, so simulated gaze measurements are wrong for the first
  seconds.
- `Print` showed detector `BOXES` and `FaceTargetSelector.TARGET_BOX` as empty although downstream
  modules received rows; the dynamic-output display in `Print` may be incomplete.
- The `pan-tilt-hud` widget reads only the first element of a source, so it could not show pan from
  `GazeLock.HEAD_GOAL`; plots are used instead.
- The `InputVideoStream` ReadMe does not yet list the uncommitted crop parameters, which the Markdown
  checker reports; those changes were left untouched.
- The full kernel suite still has the pre-existing failure in WebUI fatal-step test 239.

## EpiBlack face-following robot feedback

Issues reported after the first runs on EpiBlack.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Pupil communication fails repeatedly after about five minutes (`Could not set pupil torque for servo 2`); make `EpiServos` report the exact cause, keep driving the other pupil, and stop repeating the same warning every tick. | Completed | Release build; simulated `EpiBlack` run unchanged. The pupil path is skipped in simulation, so the new messages must be confirmed on the robot. The repeated failure after five minutes suggests a latched XL-320 alarm such as overload; the next occurrence will name the cause. | `EpiServos now reports the cause of pupil communication failures` |
| 2 | Aim the gaze so the face sits below the image center, making the robot look at the eyes instead of the middle of the face box. | Completed | Release build; new `GazeLock` aim test passes with the other GazeLock tests (a target at the aim point leaves the direction at zero, a centered target gives +10.28 degrees elevation and -8.58 degrees azimuth); demo sets `aim_y=0.4` and runs 60 simulated ticks cleanly; Markdown checker reports no new issue. | `GazeLock can now keep the target at an aim point below the image center` |
| 3 | Reduce camera lag and stalls: lighter UV4L stream settings, matching field of view, and documentation. | Completed | Measured the right eye stream at 1280 x 720, JPEG quality 85: 34 frames/s, about 420 kB per frame, about 114 Mbit/s. Applied frame rate 30 and quality 70 through the UV4L panel at runtime (old values 45 and 85); the stream then ran at 30 frames/s, still about 418 kB per frame. UV4L refused the 640 x 480 resolution because another client was streaming. Demo now exposes `horizontal_fov` and `vertical_fov`; the guide documents the recommended settings, how to apply and persist them, and the field of view per mode; 60 simulated ticks run cleanly. | `EpiBlack face following now documents lighter camera settings` |
| 4 | Ctrl-C ends Ikaros with a segmentation fault and the next `EpiServos` start fails; release the servos while the kernel is still running. | Completed | Cause: the `EpiServos` destructor ran the power-off ramp, including `Warning()` calls from several threads, during process exit after the kernel had finished. `EpiServos` now powers off in `Stop()`, which the kernel calls before exit; the destructor keeps the ramp only as a fallback. Release build; a Ctrl-C probe in the simulated demo shows module `Stop()` running before `Ikaros 3.0 Ended` with exit status 0. The hardware path cannot run here and must be confirmed on EpiBlack. | `EpiServos now releases the servos when the model stops` |
| 5 | Pupil servos still do not respond after a power cycle; they rest at maximum dilation. | Completed | Read-only register dump of both XL-320 pupils: every reply carries the alert flag 0x80, hardware error status 4 (input voltage), supply 5.3 V (XL-320 needs 6.0 to 8.4 V), torque off, servo 2 at position 0. Head MX servos read 11.7 to 11.9 V without errors. `EpiServos` aborted every pupil transfer on the alert flag, so no pupil goal was ever sent; it now fails only on bus or instruction errors, still drives the pupils during an alert, and reports the alert and voltage once. Also marked `Init` and `Tick` `override`, which the earlier `Stop()` override required. Release build without warnings; simulated run unchanged. Hardware run not done here. | `EpiServos now drives the pupils despite a servo hardware alert` |
| 6 | Make head and eye motion smoother and quicker. | Completed | `GazeLock` head motion is now acceleration-limited with discrete braking toward the goal and joint limits (`head_max_acceleration`), and eye goals can be smoothed (`eye_smoothing`). `FaceTargetSelector` treats identical consecutive detections as repeated frames (`ignore_repeated`), so the demo now ticks steadily with `synchronized_tick=false`; demo head settings are gain 3/s, 90 deg/s, 300 deg/s^2, dead zone 5, eye smoothing 20/s. Release build without warnings; GazeLock tests (4) and FaceTargetSelector tests (4) pass, including new acceleration and repeated-frame tests; 400-tick probed simulation keeps every head speed change within 0.48 deg/tick; a stalled video no longer freezes the model, which releases the lock and runs to completion. Not yet run on EpiBlack. | `Face following now moves smoothly at a steady tick rate` |
| 7 | `EpiServos` startup fails in `SetPupilParameters`, so the robot cannot start; retry pupil setup writes, report the failing write, and allow the demo to run without pupils. | Completed | Read-only register dump after the failed start: both pupils at 5.3 V with an input-voltage alert, servo 3 limits rewritten, servo 2 torque still on, so the failure came from a missed reply late in the setup. Pupil setup writes now retry three times and name the failing servo, register, and cause; new `require_pupils` (default true) lets the demo continue without pupils. Release build; simulated run unchanged; Markdown checker clean. Not run on the robot. | `EpiServos can now start without pupils and retries pupil setup` |

### Constraints

- `EpiServos` changes can only be build-checked here; behavior on the robot must be confirmed by the user.
- Do not disconnect other clients of the camera server.

### Outstanding issues and questions

- The camera resolution is still 1280 x 720 because another client held the stream. Close all viewers
  and apply 640 x 480 in the UV4L panel, and persist it in the Pi's UV4L configuration.
- The JPEG quality change did not reduce the frame size of the open stream; check the frame size again
  after the resolution change.
- The UV4L runtime changes (30 frames/s, quality 70) are lost when the Pi restarts.
- The pupil failure cause is still unknown; the next occurrence will name it. If it reports overload,
  consider a smaller pupil range in `EpiExpression` than 9 to 15 mm.
- A pupil bus that stops answering now costs two timeouts per tick instead of one, because the second
  pupil is still tried.
- `aim_y=0.4` is a first guess for eye contact and should be tuned on the robot.
- Stopping the model from the WebUI now also powers off the servos; restarting runs the full `EpiServos` initialization again.
- The second-start failure in `SetPupilParameters` should disappear with a clean shutdown; if it persists, it points to the pupil bus problem from task 1.
- The pupil servos run at 5.3 V, below the XL-320 minimum of 6.0 V; check the pupil power supply. Under-voltage makes them weaker and may still cause unreliable motion.
- Tune `head_gain`, `head_max_speed`, `head_max_acceleration`, and `eye_smoothing` on EpiBlack; the servo-side profile velocity and acceleration in `ServoParametersEpiTorso_Position.json` also shape the motion.

## EpiBlack head-mirroring demo

EpiBlack copies the head movements (yaw and pitch) of the closest person as a mirror image, hands
over smoothly between people, and shows idle, watching, and mirroring with the eye and mouth LEDs.
Head pose comes from Apple Vision, which the face detector already runs; no Python or external
libraries.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | `AppleVisionFaceDetector` outputs the head pose (yaw, pitch, roll in degrees) of every face in `POSES`, aligned with `BOXES`. | Completed | Release build without new warnings; Vision face request set to revision 3. Still-image probe through the WebUI `/data` route: `UserData/pierre.jpg` gives one row, yaw -5.7, pitch 4.4, roll -0.5 degrees (continuous, not quantized); the horizontally flipped image gives yaw +4.2 and pitch 3.0, so yaw changes sign as expected. Roll is noisy on the 120 x 168 test image. `BOXES` unchanged. Live camera check of the yaw and pitch signs needs a person in front of EpiBlack and is left for task 5. | `AppleVisionFaceDetector now outputs the head pose of each face` |
| 2 | `FaceTargetSelector` passes on the pose of the selected face (`TARGET_POSE`), signals identity switches (`NEW_TARGET`), and can delay switches (`switch_delay`). | Completed | Release build without new warnings; all 6 `FaceTargetSelector` tests pass through `kernel_test.py`: the new pose test (pose of the selected row passed on, held during `hold_time`, zero after, `NEW_TARGET` once) and the new delay test (a two-tick challenge is ignored; a lasting one takes over on the fourth tick with `switch_delay=0.12`, checked with distinct positions). The delay test fails as intended when `switch_delay` is 0. The four existing tests are unchanged and pass. | `FaceTargetSelector now passes on the face pose and can delay switches` |
| 3 | Add `HeadMirror`, which turns the selected face pose into a mirrored head goal for `GazeLock` with smooth fade-in, fade-out, and handover between people. | Completed | Release build without new warnings; both `HeadMirror` tests pass through `kernel_test.py`. Engage test: tracked state for `engage_time`, then mirroring with yaw 10 and pitch 4 giving head {3, 15} from direction {5, 20}; yaw 40 and pitch -30 clamped to {15, 10}; `NEW_TARGET` stops mirroring at once; a turned-away (yaw 50) and then a too-small face do not start mirroring; mirroring restarts for a close, facing person; loss after `lost_time`. Fade test: exponential fade-in, offset frozen during fade-out, `FOLLOW` back to 1 below weight 0.02. Closed-loop smoke run with `GazeLock` (`DIRECTION` and `LOCKED` with delay 1): no loop or shape errors; head pans to the mirrored offset under the acceleration limit while the eyes counter-rotate. The module is not specific to any robot. | `Added HeadMirror, which mirrors a person's head pose with a pan-tilt head` |
| 4 | `EpiExpression` shows a third, mirroring state on the eye and mouth LEDs through an optional `MIRRORING` input. | Completed | Release build without new warnings; both `EpiExpression` tests pass through `kernel_test.py`. The new mirroring test steps through idle, engaged, mirroring, engaged, idle and checks the eye colors, mouth off, `mouth_color`, `mouth_mirror_color`, `MIRROR_LEVEL`, and that simulated `FadeCandy` accepts all outputs. The existing fade test, without `MIRRORING`, is unchanged and passes, so `EpiFaceFollowing.ikg` keeps its expression. `GazeLock` tests (4) still pass. | `EpiExpression can now show a mirroring state on the eyes and mouth` |
| 5 | Add the `Projects/Epi/EpiHeadMirroring.ikg` demo with guide and WebUI dashboard; the robot is a model-level setting so the demo works on every Epi robot, not only EpiBlack (requested during the work). | Completed | Release build. The model loads and runs 150 simulated ticks on `UserData/video.mp4` without warnings, with all probed buffers present. That clip shows a face only briefly, so behavior was checked in a closed-loop run of the model with the camera and detector replaced by a scripted `InputFile` (two people, simulated `EpiServos` and `FadeCandy`, field of view set to 1 degree so that the scripted face stays straight ahead). Mirroring starts 1 s after person A appears; A's +/-25 degree yaw gives a +/-10 degree clamped head pan with the eyes counter-rotating within their range; closer person B takes over 0.8 s after appearing (`switch_delay` 0.7); A's offset fades out before B is mirrored; after B leaves, mirroring fades out, the lock is released, and the head returns; LEDs go dim amber, blue with dim mouth, green with bright mouth. The head goal never exceeds 90 deg/s or the 300 deg/s^2 acceleration limit. The stored direction drifts slowly in that run, which comes from the 1-degree field of view shortcut, not from the model. Static layout check: no widget overlaps, matching `x/_x` and `y/_y`, graph left of the dashboard, all connections and widget sources resolve. Markdown checker reports no issue for the new and changed docs. Robot name, camera, signs, and gains are model-level settings; the `epi-head` widget uses the neutral `EpiGray` because widget attributes cannot refer to `@robot`. Not run on a robot and not inspected in a browser. | `Added the Epi head-mirroring demo` |

### Additional fixes found during the tasks

| Fix | Verification | Commit |
|---|---|---|
| `HeadMirror` restarted mirroring for a new person while the previous offset was still fading, so the offset jumped by the remaining weight times the difference between the two people's offsets; a new person is now mirrored only after the fade-out has ended. Found in the closed-loop simulation of task 5. | New handover test passes with the other two `HeadMirror` tests; it fails with the previous code because state 1 never appears with `engage_time=0`. | `HeadMirror now waits for the fade-out before mirroring a new person` |
| First robot run: pan mirroring worked, but the tilt went the wrong way. The user found that EpiBlack follows faces vertically with `y_sign=1`; `Mirror` now uses `pitch_sign="@y_sign"`, since with an upright camera image the correct nodding sign equals `y_sign`. | Simulated run resolves `Mirror.pitch_sign` to 1 without warnings; Markdown checker clean. Nodding direction still to be confirmed on the robot. | `Epi head mirroring now nods with the same tilt sign as face following` |
| Robot feedback: mirroring works, but the tilt jitters because of play in the tilt servo under the head's weight. While mirroring, `HEAD_GOAL` changed on every tick with Vision's pitch noise and the measured face elevation. `HeadMirror` now has `tilt_dead_zone` and `pan_dead_zone` (default 0, unchanged behavior): the goal is held until it would move by more than the dead zone, then moves there in one step, which `GazeLock` smooths. The demo uses `tilt_dead_zone=3`. | New dead-zone test passes with the other three `HeadMirror` tests. Closed-loop simulation with a slow nod plus 2-degree pitch noise over 21.6 s of mirroring: without the dead zone the tilt goal moved on 519 ticks with 93 direction reversals; with 3 degrees on 110 ticks in 22 movements with 11 reversals, with similar amplitude; no warnings. Markdown checker clean. To be confirmed on EpiBlack. | `HeadMirror can now move the head in fewer, larger steps` |
| Robot feedback: the tilt is better with the dead zone but still jittery. `GazeLock` moved tilt with the same 90 deg/s and 300 deg/s^2 limits as pan, so each tilt step started and stopped sharply on a loaded joint with gear play. `GazeLock` now has `tilt_speed_scale` and `tilt_acceleration_scale` (default 1, unchanged behavior); the demo uses 0.5 and 0.3 (45 deg/s, 90 deg/s^2). | New tilt-scale test passes with the other four `GazeLock` tests (pan keeps the exact profile of the existing acceleration test; tilt accelerates at half the rate and stops on the goal without overshoot); four `HeadMirror` tests pass. Closed-loop simulation of a noisy nod: peak tilt acceleration falls from 335 to 101 deg/s^2 and peak speed from 29 to 17 deg/s with a similar nod range; no warnings. Markdown checker clean. To be confirmed on EpiBlack. If the head still shakes while the tilt goal is still, the cause is servo hunting (tilt servo P gain 850, D gain 0 in `ServoParametersEpiTorso_Position.json`), which this change does not address. | `GazeLock can now move the tilt more gently than the pan` |
| Robot feedback: tilt should be faster, neither axis should follow very small movements, and a movement beyond the threshold should be reached quickly. The previous dead zone jumped to the goal at the moment it crossed the threshold and then waited for the next threshold, so a nod became a staircase of steps. It now follows the goal continuously once it leaves the dead zone until the goal has stayed within a quarter of the dead zone for the new `settle_time` (default 0.2 s). The demo now uses a 3-degree dead zone on both axes, `pose_smoothing` 8, `tilt_speed_scale` 1, and `tilt_acceleration_scale` 0.6. | Rewritten dead-zone test passes with the other three `HeadMirror` tests: +/-3 degree pitch noise does not move the head, a step to 8 and then 10 degrees is followed to -5 (the previous code stopped at -4), and small changes after settling are ignored. Closed-loop simulation with 2-degree pose noise and a frozen target direction, old against new demo settings: pan corrections while the person holds still 14 to 0 and while a turn is held 13 to 0; 90% of a nod reached after 0.44 s instead of 0.52 s, of a larger nod 0.48 s instead of 0.68 s, of a turn 0.40 s instead of 0.48 s; tilt stays still in both. To be confirmed on EpiBlack. | `HeadMirror now follows a movement beyond the dead zone to its end` |

### Constraints

- C++ and Objective-C++ modules only; no kernel changes, Python, or external libraries.
- `EpiFaceFollowing.ikg` and existing module behavior stay unchanged when the new inputs are not connected.
- Do not include the uncommitted `InputVideoStream` changes, `.Rhistory`, or `CLAUDE.md` in any commit.
- Do not move the physical robot without the user present.

### Outstanding issues and questions

- Partly run on EpiBlack: pan mirroring works and `y_sign=1` keeps the face in view vertically.
  Confirm that nodding now goes the right way (`pitch_sign` follows `y_sign`), whether `view_compensation` is needed, `max_pan` and
  `max_tilt`, and the crowd settings, following the checklist in `Projects/Epi/EpiHeadMirroring.md`.
- The default yaw and pitch signs come from Vision's documented conventions and `GazeLock`'s angle
  conventions; the test portraits were too frontal to confirm the yaw sign from images.
- `x_sign=-1` and `y_sign=1` were found on EpiBlack; whether they hold for the other Epi robots is
  unverified. `EpiFaceFollowing.ikg` now also uses `y_sign=1`.
- The dashboard has not been inspected in a browser; check alignment, clipping, and console errors.
- The left eye camera of EpiBlack does not appear as a USB network device on the Mac (only one
  RNDIS/Ethernet Gadget interface; `LeftEye.local` does not resolve). Check its cable, hub port, Pi
  power, and SD card.
- WebUI widget attributes do not resolve `@` references, so the `epi-head` widget cannot follow the
  `robot` setting; a possible Ikaros improvement, not proposed as an issue yet.
- Vision's pose is noisy for small or strongly turned faces; the effect on the demo is untested with
  real crowds and children.
- `max_tilt=10` upward with `aim_y=0.3` brings a very close face near the bottom image edge; lower
  `max_tilt` if close children are lost when they look up.
- The `InputVideoStream` ReadMe still lacks the uncommitted crop parameters (pre-existing, untouched).

## EpiBlack animatronics

Playful, lifelike behaviour for EpiBlack: frequent LED eyelid blinks, quick eye darts, curious
look-around when alone, and reactions to people (cross-eyed blue eyes, giggle with pulsing pupils,
mind-blown wow with diverging eyes and a pooof). Built from general behaviour modules that drive
`GazeLock` and `EpiExpression` through new optional inputs.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | `GazeLock` takes optional `EYE_OFFSET`, `VERGENCE`, and `HEAD_OFFSET` inputs and a `camera_eye` parameter. | Completed | Release build without new warnings; all 6 GazeLock tests pass through `kernel_test.py`. The new offsets test checks that the head offset moves 2 degrees per tick under `head_max_speed` to {4, -6}, that eye offset 3 with vergence 4 gives eyes {5, 1}, and that one centered detection is stored at azimuth -5 with `camera_eye=right` and -3 with `both`. The five existing tests are unchanged and pass. Markdown checker reports only the known `InputVideoStream` issue. | `GazeLock can now add eye, vergence, and head offsets from behaviour modules` |
| 2 | `EpiExpression` takes optional `BLINK` (eyelid sweep), `EYE_OVERLAY`, `MOUTH_OVERLAY`, `BRIGHTNESS`, and `PUPIL_OFFSET` inputs. | Completed | Release build without new warnings; all 3 `EpiExpression` tests pass through `kernel_test.py`. The new overlay test checks that a half-weight blue overlay with brightness 0.5 gives eyes {0.25, 0.25, 0.5}, that closure 0.5 leaves only the lower LEDs 1-5 lit and closure 1 turns every eye LED off, that a full red mouth overlay gives {0.5, 0, 0}, and that pupil offsets 3, 20, and -20 give 12, 16, and 5 mm; simulated `FadeCandy` accepts all outputs. The two existing tests are unchanged and pass. Markdown checker reports only the known `InputVideoStream` issue. The real LED ring orientation must be checked on the robot. | `EpiExpression can now blink, overlay colors, flash, and dilate the pupils for behaviours` |
| 3 | Add `Blinker`, a random natural blink generator, in a new `BehaviorModules` category. | Completed | Release build without warnings; `Blinker` test passes through `kernel_test.py`: with a fixed 1 s interval, blinks start on ticks 9 and 23 with lid 0.5, 1, 1, 0.5, 0 (close 0.2 s, closed 0.1 s, open 0.2 s); a `TRIGGER` edge starts a blink at once, and a blink that falls due while `INHIBIT` is 1 starts on the tick the inhibition ends. A manual run with `double_blink_probability=1` starts blinks on ticks 4, 9, 18, and 23 (pairs 0.1 s apart, never tripled). Markdown checker clean. | `Added Blinker, which generates natural eyelid blinks` |
| 4 | Add `GlanceGenerator`, random gaze shifts in any number of dimensions. | Completed | Release build without warnings; `GlanceGenerator` test passes through `kernel_test.py` with single-threaded, ordered output: with a fixed 1 s interval a two-dimensional glance of size {10, 4} (`min_fraction` 1) appears on ticks 9, 19, and 29 and returns to the center after `return_time` 0.3 s; a `rate` that halves the distance per tick gives 4, 6, 7, 7.5; with `INHIBIT` 1 until tick 15 the first glance comes on tick 24 and returns on tick 27. Changing one expected value makes the test fail. Markdown checker clean. | `Added GlanceGenerator, which makes a robot look around by itself` |
| 5 | Add `KeyframeAnimator`, a keyframe player for named channels and events from a JSON library. | Completed | Release build without warnings; both `KeyframeAnimator` tests pass through `kernel_test.py`. The behaviour test, with ordered single-threaded output, checks a linear ramp 0, 2, 4, 6, 8 that is held and then eased to 4, the `start` and `middle` events on ticks 1 and 4, a repeated step track that pulses twice with an event on its tick, a second animation that interrupts at the present value 4 and glides 4, 7, 10, the halving release to rest of channels that are not used, and the switch of `ANIMATION` from 0 to -1 on the final tick. The bad-channel test checks that an unknown channel stops the model at startup with a clear message. The `play` command, sent through the WebUI command URL, starts animations by index (`x`) and by name. Markdown checker clean. | `Added KeyframeAnimator, which plays named keyframe animations and events` |
| 6 | Add `BehaviorSelector`, which chooses spontaneous and greeting behaviours by state, weight, and cooldown. | Completed | Release build without warnings; `BehaviorSelector` test passes through `kernel_test.py` with ordered single-threaded output and fixed 1 s intervals: the idle-weighted behaviour 0 starts on tick 9, the greeting 0.2 s after `ENGAGED` rises (tick 17), and the engaged-weighted behaviour 1 on tick 27; a 1.5 s cooldown alternates two behaviours on ticks 9, 19, and 29; `BUSY` from tick 5 to 14 holds the interval so the first start comes on tick 19; a greeting with zero delay fires on arrival (tick 2), the farewell on departure (tick 8), and a return 0.2 s later (tick 10) is not greeted again within the 0.5 s reaction cooldown. Markdown checker clean. | `Added BehaviorSelector, which decides when a robot does what by itself` |
| 7 | Render voice clips with Kokoro TTS (offline, venv) and effects with the Ikaros synthesizer. | Completed | Kokoro (`kokoro-onnx` 0.6.1, `onnxruntime` 1.32.0) installed only in `UserData/venvs/kokoro`; model files in `UserData/models/kokoro`, both ignored by git. `render_voice.py` renders hi, wow, giggle, hmm, and aww with voice `af_heart` raised four semitones (0.37-0.83 s, 24 kHz mono, peak 0.89); phonemes were checked, and "Wooow" was replaced by a slow "Wow!" because Kokoro read it as two syllables. `RenderPoof.ikg` and `RenderSparkle.ikg` render the effects with `Noise`, `AudioOscillator`, `AudioFilter`, `ADSREnvelope`, and `OutputAudioFile` without warnings: the poof (1.1 s, peak 0.88) starts with a low thump and its noise falls from about 2.5 kHz to 280 Hz as it fades; the sparkle (0.42 s, peak 0.85) has three rising notes. All files open with `afinfo`. Not listened to here; the user judges the sound. | `Added synthesized voice and effect sounds for Epi animatronics` |
| 8 | Add the `Projects/Epi/EpiAnimatronics.ikg` demo with animation library, guide, and dashboard. | Completed | Release build. The model loads and runs 300 simulated ticks on the local test video without warnings; with the WebUI on, every dashboard source answers through `/data` and the page is served. Scripted closed-loop simulation over 60 s, with the camera and detector replaced by a person whose image position follows the simulated eye and head positions and with sound replaced by a log: the gaze locks when person A appears at 10 s and the stored direction stays on A (azimuth 8.6 to 11.8 for A at 10); the greeting (sparkle, "hi") follows 0.4 s after the lock, curious ("hmm") follows the release at 36.5 s, and person B is greeted at 45 s; spontaneous animations come every few seconds (cross-eyed, mind blown with wow and poof 1.3 s apart, giggle, eye dart); 23 blinks (mean gap 2.6 s), 22 eye glances, and an idle head look-around of up to 15 degrees of pan. A second scripted run plays all seven animations: convergence 22.9 and divergence -19.8 degrees between the eyes, pupils 7 to 16 mm, eyelid squint that keeps only the lower LEDs lit, LED flash to 1 and blackout to 0. Eye goals stay within -14.7..13 degrees. Head speeds stay within 90 deg/s; the pan acceleration stays within 300 deg/s^2 for a single movement and reaches 354 deg/s^2 when a glance overlaps following (see the fix below). Static layout check: no widget overlaps, matching `x/_x` and `y/_y`, components left of the dashboard, every connection, widget source, parameter, and command resolves, and no input has more than one connection. Full kernel suite: failures only in 239 (known), 374 (known state-path pair), and 207 and 210, which pass when run alone. Not run on a robot, not inspected in a browser, and the sounds were not listened to here. | `Added the Epi animatronics demo` |

### Additional fixes found during the tasks

| Fix | Verification | Commit |
|---|---|---|
| The `GazeLock` documentation said that `HEAD_OFFSET` obeys the head limits; the offset and the following movement are limited separately, so their sum can change up to twice as fast. The scripted demo simulation reached 429 deg/s^2 of pan acceleration while an idle head glance overlapped the return to center after a lock was released. The ReadMe and `.ikc` now say so. Found in task 8. | Markdown checker clean; GazeLock tests (6) pass. | `GazeLock documentation now explains the combined head limits` |

### Constraints

- C++ modules only; no kernel changes and no external libraries in the build. Kokoro and its Python
  packages are used only offline, in `UserData/venvs/kokoro`, to render WAV files.
- Existing behaviour of `GazeLock` and `EpiExpression` stays unchanged when the new inputs are not
  connected.
- Do not include the uncommitted `InputVideoStream` changes, `.Rhistory`, or `CLAUDE.md` in any commit.
- Do not move the physical robot without the user present.

### Outstanding issues and questions

- Nothing has been run on a robot. Check the vergence sign (cross-eyed must turn the eyes inward),
  the LED ring orientation for the eyelid sweep (`led_first_angle`, `led_direction`), the head tilt
  sign of the nod, the eye-dart speed, and the pacing, following the checklist in
  `Projects/Epi/EpiAnimatronics.md`. Moving the robot requires the user to be present.
- The dashboard has not been inspected in a browser; check alignment, clipping, and console errors.
- The sounds were not listened to here; judge the Kokoro voice (`af_heart`, +4 semitones) and the
  synthesized poof and sparkle, and re-render with other settings if needed.
- EpiBlack's pupil servos still run at 5.3 V; the giggle and mind-blown animations depend on the
  pupils for much of their effect.
- Possible Ikaros bug: when a module fails during setup, for example `InputVideoStream` with a bad
  URL, a model that contains `EpiServos` aborts with `uncaught exception ... Uninitialized
  parameter` instead of reporting the real error. `EpiHeadMirroring.ikg` shows the same behavior.
  `EpiServos` probably reads an unbound parameter in `Stop()` or its destructor; not fixed here.
- Possible Ikaros bug or undocumented behavior: several connections without target ranges into one
  input do not concatenate; only one source arrives, and the connection labels are lost in `Print`.
  The demo avoids fan-in by combining signals with `Add` and `Maximum`. A GitHub issue could be
  proposed; the `ikaros-bot` token is not installed here.
- The head offsets and following are limited separately, so the head goal can accelerate up to
  twice the limit while both move; the servo profile acceleration still applies on the robot.
- `SoundOutput` plays one sound at a time and keeps only the latest waiting sound, so a sound
  started during a long clip can be delayed or dropped.
- Kernel tests 207 and 210 failed in the parallel full-suite run (210 after an HTTP 500 from remote
  session logging) and pass when run alone; possibly flaky under load.
- The `InputVideoStream` ReadMe still lacks the uncommitted crop parameters (pre-existing, untouched).

## EpiAnimatronics simulation without the robot camera

`./Bin/ikaros simulate=True Projects/Epi/EpiAnimatronics.ikg` without the robot connected aborted with
`Uninitialized parameter` after the eye stream `RightEye.local` could not be resolved.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | `EpiServos` no longer aborts when another module fails during setup, so the real error is reported. | Completed | Cause: `Stop()` and the destructor read the `simulate` parameter, which is unbound when another module fails before `EpiServos::Init()` runs; the exception escaped the destructor and aborted the process. Both now rely only on `hardwarePowered`, which is set only on the hardware path, and the hardware pointers, which start as `nullptr`. Release build without warnings. `EpiAnimatronics.ikg` with `simulate=True` and an unresolvable eye URL now ends with `SetUp Failed ... InputVideoStream could not create its frame grabber` instead of the abort; a 50-tick simulated `EpiHeadMirroring.ikg` run on the test video ends normally. The hardware shutdown path is unchanged apart from the removed parameter checks and was not run on a robot. The stale note in `EpiAnimatronics.md` is removed. | `EpiServos no longer aborts when another module fails during setup` |
| 2 | `EpiAnimatronics.ikg` can use the Mac's own camera through a `camera` setting, so it can be simulated without the robot. | Completed | The eye module's class is now `@camera` (default `InputVideoStream`, unchanged behaviour); `camera=InputVideo` uses an AVFoundation camera selected by `camera_id`. The kernel cannot pick the class from `simulate`, because indirect references such as `@camera_@simulate` do not resolve, so the camera is a separate setting. Checked in simulation: the default with an unreachable URL reports the stream error; the default with the test video runs 50 ticks and ends normally; `camera=InputVideo camera_id=97` resolves the class and fails cleanly with `Couldn't open input stream`. The real Mac camera was not opened here; `ffmpeg` lists device 0 as an external webcam and 1 as the FaceTime HD camera. Guide updated with the simulation command, device selection, settings, and the fixed-camera limitation. | `Epi animatronics can now be simulated with the Mac's camera` |

### Constraints

- Do not change the uncommitted `InputVideoStream` changes.
- Do not include `.Rhistory` or `CLAUDE.md` in any commit.

### Outstanding issues and questions

- The Mac camera path was not run with a real device here, to avoid turning on the camera; run
  `./Bin/ikaros simulate=true camera=InputVideo Projects/Epi/EpiAnimatronics.ikg` and check that
  your face is found. If the camera refuses 640 x 480, `InputVideo` stops with `Couldn't open input
  stream`.
- `simulate=true` alone still uses the robot's eye stream and stops with a clear error when the
  robot is not connected; the camera cannot follow `simulate` automatically without a kernel
  change, for example support for indirect references such as `@camera_@simulate`.
- With a fixed Mac camera, the simulated head and eyes run to their limits unless the face is near
  the image center.
- `EpiFaceFollowing.ikg` and `EpiHeadMirroring.ikg` could get the same `camera` setting.
- The `EpiServos` hardware shutdown path was only build-checked; confirm a normal stop on the robot.

## Camera setting for the other Epi demos

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | `EpiFaceFollowing.ikg` and `EpiHeadMirroring.ikg` get the `camera` and `camera_id` settings of `EpiAnimatronics.ikg`, so they can use the Mac's camera. | Completed | Both eye modules now use `class="@camera"` with `id="@camera_id"`; the default `InputVideoStream` keeps the previous behaviour. Checked in simulation for both models: 50 ticks on the test video end normally; an unreachable stream reports the `InputVideoStream` setup error; `camera=InputVideo camera_id=97` fails cleanly with `Couldn't open input stream`. The real Mac camera was not opened here. Both guides now give the simulation command, how to list and choose cameras, the new settings, and the fixed-camera limitation. Markdown checker clean. | `Epi face following and head mirroring can now use the Mac's camera` |

### Outstanding issues and questions

- The Mac camera was not opened here; check `camera=InputVideo` with a real camera in both demos.
- With a fixed Mac camera the simulated head and eyes run to their limits unless the face is near
  the image center.

## Natural animatronics sounds

The user found the animatronics sounds abrupt and unnatural. Measured causes: every voice clip and
the sparkle end about 45 ms after their last loud sample, which cuts the natural decay; all sounds
are completely dry; loudness differs by about 10 dB (sparkle -9 dB RMS, poof -19 dB RMS); and the
voices are raised four semitones by resampling, which sounds like a sped-up recording.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Render the voice clips naturally: no chipmunk pitch shift, full decay with gentle fades, a small room, matched loudness, a giggle built from natural syllables, and an audition set of voices. | Completed | `render_voice.py` now asks Kokoro not to trim (its own trim also cut the decay), keeps a 60 ms lead and the decay down to -55 dB of the peak, applies raised-cosine fades, adds a soft 0.35 s room at 12 percent, ends each clip when that tail falls below -50 dB, and matches the voiced loudness to -17 dB RMS with peaks below -3 dBFS. Default pitch is 0 instead of +4 semitones. The giggle is five "hee" syllables, accelerating from 150 to 100 ms apart and falling 2 semitones and to 60 percent in level. Measured: hi 0.76 s, wow 0.90 s, hmm 0.75 s, aww 0.80 s, giggle 1.10 s; all start and end at exactly 0; voiced RMS -17.0 dB for all; peaks 0.47-0.63. The voice now ends at about 0.5 s, with the tail fading out over the following 0.25 s, instead of being cut 45 ms after the last loud sample. `--audition` renders all clips with six voices (af_heart, af_bella, af_nicole, af_sky, af_jessica, bf_lily) into `UserData/output/animatronics_voices`. Not listened to here. | `Epi animatronics voice clips now sound natural` |
| 2 | Render the poof and sparkle with smooth decays, overlapping bell notes, a reverb tail, and loudness matched to the voices. | Completed | Both patches render with the Ikaros synthesizer without warnings. Amplitude envelopes are squared with `Multiply`; the sparkle has three bells with their own delayed triggers instead of one retriggered envelope; `AudioReverb` adds a room; a gated master `ADSREnvelope` fades the end. Measured: poof 1.40 s, -17.3 dB RMS, peak 0.58, falling smoothly by about 4 dB per 0.1 s while its brightness falls from 1.8 kHz to 280 Hz; sparkle 0.92 s, -17.3 dB RMS, peak 0.51, with notes 1 and 2 sounding together at 0.2 s and notes 2 and 3 from 0.3 s on; both start and end at exactly 0. All seven sounds are now within -17.0 to -17.3 dB RMS (previously -9 to -19 dB). Because `SoundOutput` plays one sound at a time, the greeting's "hi" moved from 0.5 s to 0.95 s in `EpiAnimations.json`, so it follows the sparkle instead of waiting; the mind-blown wow ends at 1.0 s, before the poof at 1.4 s. Not listened to here. | `Epi animatronics effects now decay smoothly in a small room` |

### Outstanding issues and questions

- None of the sounds were listened to here; judge them on the speaker used for demos. To try other
  voices, run `render_voice.py --audition` and compare the clips in
  `UserData/output/animatronics_voices`.
- If Epi should sound smaller, `--pitch` 1 to 2 semitones keeps the voice natural; more sounds sped up.
- The giggle is assembled from single syllables; if it still sounds mechanical, the syllable timing
  and pitch steps are at the top of `render_voice.py`.

## Non-verbal robot sounds

The user found the voice clips weird and asked for robotic, non-verbal sounds: chirps and whistles
whose pitch and prosody alternate, with varied attack and decay, for example a surprise that starts
low and rises, following the principles of film robot sound design.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Replace the voice clips and synthesized effects with non-verbal robot chirp phrases rendered by one script, keeping the file names used by the demo. | Completed | New `render_sounds.py` (numpy only) builds every sound from syllables with their own smooth log-pitch contours, attack, decay, vibrato, sample-and-hold warble, and overtones, with continuous phase, a small room, and matched loudness; the design rules and sources are in the sound ReadMe. Kokoro, `render_voice.py`, `RenderPoof.ikg`, and `RenderSparkle.ikg` are removed; the Ikaros audio modules update pitch once per tick, which would step fast glides. Measured pitch every 40 ms matches each design: hi 940-1480-1180-1860 Hz ending high; wow climbs 171 to 1290 Hz, then 1.7-2 kHz; poof falls 1580 to 160 Hz with a warble; giggle hops between about 1.8 and 1.1 kHz while drifting down; hmm holds 620 Hz, then rises to 1260 Hz; aww lifts 1070 to 1190 Hz, then sinks to 730 Hz; sparkle climbs 1330 to 2050 Hz into a 2.5-2.7 kHz trill. Lengths 0.50-0.97 s; all start and end at 0; RMS -17 dB (aww -20, sparkle -19). The greeting's chirp moved back to 0.55 s because the sparkle now ends at 0.5 s; the mind-blown surprise ends at 1.07 s, before the poof at 1.4 s. 20 simulated ticks of the demo run without warnings. Not listened to here. | `Epi animatronics now speaks in non-verbal robot chirps` |

### Outstanding issues and questions

- The sounds were not listened to here; tune them by ear, either the syllables in
  `render_sounds.py` or other warble takes with `--variants`.
- The Kokoro venv and model (about 350 MB) in `UserData/venvs/kokoro` and `UserData/models/kokoro`
  are no longer used and can be deleted by the user.

## Livelier, less repetitive animatronics

Requested after the robot-sound change: delete the Kokoro files; slow some animations; add a
mischievous look to the side with a "hehehe" laugh; swap the eye directions of cross-eyed and mind
blown; make behaviour changes less frequent and less random; look around and giggle on finding
someone new; avoid repetition by combining behaviour parameters; and add an arousal level that sets
the sound volume and the robot's excitement, following how Disney layers background motion,
triggered animations, and synchronized show functions.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Delete the Kokoro venv and model from `UserData`. | Completed | Deleted `UserData/venvs/kokoro` (166 MB) and `UserData/models/kokoro` (337 MB), both ignored by git, and the empty parent folders. Created `UserData/venvs/sounds` with numpy only, as the sound ReadMe describes, for `render_sounds.py`. | No repository change |
| 2 | Swap the vergence direction so that cross-eyed converges and mind blown diverges on the robot. | Completed | The user saw cross-eyed diverge on EpiBlack, so a positive eye pan turns Epi's eyes to the robot's left. `GazeLock` has a new `vergence_sign` (default 1, unchanged behaviour); the animatronics demo sets the model-level `vergence_sign=-1`. The `epi-head` widget turned the eyes as seen from the front but drew the robot's left eye on the viewer's left; it now draws it on the viewer's right, so the dashboard matches the robot. Release build without warnings; all 6 GazeLock tests pass, and the offsets test now also checks that `vergence_sign=-1` gives eyes {1, 5} instead of {5, 1}; `node --check` passes for the widget. Scripted demo run: left minus right eye goal reaches -24.2 degrees in cross-eyed (inward on Epi) and +19.8 in mind blown (outward). Markdown checker clean. To be confirmed on the robot and in a browser. | `Cross-eyed now converges and mind blown diverges on Epi` |
| 3 | `KeyframeAnimator` varies each play: `SPEED` and `INTENSITY` inputs, random speed and intensity variation, and random mirroring of chosen channels. | Completed | New optional `SPEED` and `INTENSITY` inputs, parameters `intensity_channels`, `mirror_channels`, `mirror_probability`, `speed_variation`, `intensity_variation`, and `seed`, and a `VARIATION` output; key values are taken as rest + gain * (value - rest), so scaling and mirroring work around rest and interruptions still start from the present value. Release build without warnings; all 3 `KeyframeAnimator` tests pass. The new variation test checks that `SPEED` 2 reaches the ramp's 8 in two ticks with its event two ticks early, and that `INTENSITY` 0.5 with mirroring gives -1, -2, -3, -4 on the scaled channel and leaves the other channel alone. Unconnected inputs and default parameters leave the existing tests unchanged. Markdown checker clean. | `KeyframeAnimator can now vary the tempo, size, and direction of each play` |
| 4 | Add `Arousal`, a general excitement level that rises with events and people and settles back over time. | Completed | Release build without warnings; the `Arousal` test passes through `kernel_test.py` with ordered output: from the idle level 0.2 a stimulus with boost 0.4 gives 0.6, the excitation halves each tick to 0.4, 0.3, and 0.25, `ENGAGED` raises the resting level to 0.5 (0.525 with the remaining excitation), a response rate that halves the distance per tick gives 0.4, 0.4, 0.35, 0.3, and `MAPPED` with low {0.8, 0} and high {1.3, 1} gives {1.1, 0.6} at 0.6. Markdown checker clean; the ReadMe example uses `SoundOutput.GAIN` from task 5. | `Added Arousal, which keeps a robot's excitement level` |
| 5 | `SoundOutput` takes an optional `GAIN` input (named so because `VOLUME` is already an output). | Completed | Security review: playback still uses `posix_spawn` with an explicit argument list and the fixed `/usr/bin/afplay`; the change only adds afplay's `-v` option with a number formatted from a value clamped to 0..1, so no shell, path, or executable handling changes and security is not reduced. Below 1 the gain is passed as `-v`; at 1, or when `GAIN` is not connected, the argument list is unchanged. Release build without warnings. A silent run with `GAIN` 0 started `/usr/bin/afplay -v 0.000 .../poof.wav`, seen with `ps`. RMS and VOLUME still describe the file. Markdown checker clean. | `SoundOutput can now play sounds more quietly through a GAIN input` |
| 6 | `BehaviorSelector` chooses behaviours that match the arousal level, avoids recent ones, and spaces them by arousal. | Completed | New optional `AROUSAL` input and parameters `arousal_levels`, `arousal_width` (Gaussian match of each behaviour's energy to the present arousal), `memory` (recent spontaneous behaviours avoided while others are available), and `excited_interval_scale`. Release build without warnings; both `BehaviorSelector` tests pass: the existing test unchanged, and the new mood test with ordered output checks that at arousal 0.9 only the behaviour with energy 0.9 is chosen (ticks 9, 19, 29), that `memory` 1 alternates 0, 1, 0 although behaviour 0 is a million times more likely, and that `excited_interval_scale` 0.5 at arousal 1 halves the interval (ticks 9, 14, 19, 24, 29). Markdown checker clean. | `BehaviorSelector can now follow the robot's mood and avoid repeating itself` |
| 7 | Update the demo: slower animations with anticipation and follow-through, a mischievous animation with a "hehehe" sound, a giggling greeting when someone new is found, arousal-driven volume, speed, intensity, and pupils, and less frequent behaviour changes. | Completed | Ten animations (new: found you, mischief, look around; the others slower, with anticipation, overshoot, eyes leading the head, and mouth flashes and head bobs on the chirp onsets); new `hehe.wav` (three chirps sliding down, 737 to 546 Hz, then a rise to 840 Hz, -17 dB RMS); `Arousal` drives animation tempo 0.8-1.15, size 0.8-1.2, sound gain 0.35-1, and a pupil offset of -1 to +3 mm; `Behaviors` uses energies, `memory` 3, `cooldown` 15 s, intervals 5-10 s (0.6 times when excited), found you as greeting and curious as farewell; ten dashboard buttons in two rows and an arousal table and plot. Release build; layout check: no widget overlaps, matching `x/_x` and `y/_y`, components left of the dashboard, all sources resolve, no fan-in; modules below `Animator` moved down, as it now has 11 ports. Scripted runs: each of the ten animations plays with its own tempo (0.94-1.14), size (0.93-1.17), and mirroring, with events on time (hehe 0.96 s into mischief, poof 1.4 s into mind blown); cross-eyed gives -25.8 and mind blown +17.9 degrees between the eyes. Two-minute scenario: alone, curious and look around at arousal 0.15-0.2 and volume 0.45; found you at 20.4 s (sparkle, then giggle 0.6 s later) with arousal rising to 0.96; then mischief, greet, eye dart, giggle, mind blown without repetition, 6-11 s apart; curious when the person leaves; found you again for the next person; 38 blinks (mean gap 3.1 s). The scenario found the `BehaviorSelector` greeting bug fixed below. WebUI serves arousal and variation, and the mischief button command plays animation 8. Markdown checker clean. Not run on the robot, not inspected in a browser, sounds not listened to here. | `Epi animatronics is now mood-driven, more varied, and less repetitive` |

### Additional fixes found during the tasks

| Fix | Verification | Commit |
|---|---|---|
| `BehaviorSelector` never started the greeting when `greeting_delay` was not a whole number of ticks: the countdown passed zero between ticks and a negative value meant "no greeting pending", so the 0.3 s delay at 0.04 s ticks in the demo dropped every greeting. A pending flag now marks the greeting. Found in the scripted demo run of task 7. | New greeting-delay test (0.3 s at 0.04 s ticks fires on tick 13, once) fails with the previous code and passes now; all 3 `BehaviorSelector` tests pass. | `BehaviorSelector now starts greetings whose delay is not a whole number of ticks` |

### Outstanding issues and questions

- Nothing from this round has run on the robot. Check the new animations, the head tilt direction
  of the nod, the eye-dart and mischief speeds, and the pacing with real people, following the
  checklist in `Projects/Epi/EpiAnimatronics.md`.
- The dashboard has not been inspected in a browser, in particular the two button rows and the
  robot view with the swapped eye placement.
- The `epi-head` widget moves the head drawing sideways with the first element of its head source
  and vertically with the second, but `GazeLock.HEAD_GOAL` is tilt, pan; the head in the robot view
  may therefore move in swapped directions. Not changed here.
- The sounds were not listened to; the hehe and the louder, excited playback need a listening check.
- Arousal changes the motion tempo by up to 20 percent but not the sounds, so chirp-synchronized
  mouth flashes can drift slightly at the extremes.
- The head goal reached 467 deg/s^2 in the scenario while animation offsets and following overlap,
  above the 300 deg/s^2 limit of each part; the servo profile still limits the motion on the robot.
- With a person standing still in front of the robot, arousal settles around 0.62, which keeps the
  engaged mood lively; lower `drive_gain` or `engaged_level` on `Arousal` if Epi should calm down more.

## Playful mirroring and varied vocalizations

Requested next: combine the animatronics with playful behaviour such as head mirroring, and make
Epi less vocal, with more varied and dynamic vocalizations.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | `KeyframeAnimator` events can have a probability and a list of alternatives, one of which is chosen at random. | Completed | Events are `[time, name]`, `[time, name, probability]`, or have a list of names as alternatives; existing libraries are unchanged. Release build without warnings; all 4 `KeyframeAnimator` tests pass. The new events test plays four times: on each start tick exactly one of two alternatives fires (a, b, a, b with seed 2; other seeds give other mixes), the probability-0 event never fires, and the probability-1 event always fires on its tick. Markdown checker clean. | `KeyframeAnimator events can now be optional and choose among alternatives` |
| 2 | Render several procedural variants of every sound, varying syllables, contours, pitch, and timbre while keeping the rhythm of sounds that the motion follows. | Completed | `render_sounds.py` now draws each sound's shape from its own seeded generator and writes three variants, `name_1.wav` to `name_3.wav` (24 files). Measured: every variant keeps its meaning (hi ends at 1650-1750 Hz, wow climbs from about 185 Hz to 1.4-2.3 kHz, hmm rises at the end, aww sinks to 540-775 Hz, sparkle climbs past 3 kHz, poof falls to about 200 Hz); syllable counts and lengths differ (hi 0.47-0.67 s, aww 0.63-1.07 s with an occasional second sigh); loudness varies from -15 to -21 dB RMS; all start and end at 0. Giggle and hehe keep their fixed onsets for the motion. The unnumbered files stay until the demo uses the variants in task 3. Markdown checker clean; not listened to here. | `Every Epi sound now comes in three varied takes` |
| 3 | Add a "copycat" mirroring game to the demo with `HeadMirror`, let blinks continue during most animations, and make the vocalizations sparser and varied. | Completed | New copycat animation (11 s) whose `mirror` channel drives `HeadMirror.ENABLE`; `HeadMirror` gets the face pose through `FaceTargetSelector.POSES` and drives `GazeLock.FOLLOW` and `MANUAL_HEAD`, and `EpiExpression.MIRRORING` shows the green mirroring state (idle mouth kept at the previous color). A new `hold` channel inhibits blinks only in found you, cross-eyed, and mind blown. Sound events now name three takes as alternatives with probabilities of 0.35-0.85 (wow and pooof always); 25 events and 24 sound files; the old single files are removed. Eleven buttons in two rows of six; new model settings `mirror_sign` and `mirror_gain`. Release build; layout check: no widget overlaps, matching `x/_x` and `y/_y`, components left of the dashboard, all sources resolve, no fan-in; 100 simulated ticks on the test video without warnings. Scripted runs with a person whose head yaw swings +-20 degrees: copycat mirrors from 1.2 s for 7.9 s with the pan offset clamped at +-9.9 degrees, and Epi blinks 8 times during it; blinks now also occur in giggle, shy, mischief, and look around; two-minute scenario: 12 animations but only 6 sounds (12 before) in 5 different takes, copycat twice with 15.5 s of mirroring, 40 blinks. Markdown checker clean. Not run on the robot, not inspected in a browser, sounds not listened to here. | `Epi animatronics now plays copycat and speaks less, in varied takes` |

### Outstanding issues and questions

- Not run on the robot: check copycat with a person (mirror direction and nodding, as in
  `EpiHeadMirroring.md`), and listen to the sound takes and how often Epi now vocalizes.
- Reactions interrupt animations: when the person leaves, curious starts at once, for example
  between the wow and the pooof of mind blown. Decide whether farewells should wait for the playing
  animation to end.
- Copycat only mirrors when the person faces the robot and is close (`min_width`, `facing_limit`);
  otherwise the game shows only its start and end.
- The dashboard has not been inspected in a browser, in particular the two rows of six buttons.

## Convergence on close faces

Requested: Epi's eyes converge when someone comes really close, that is when the face box gets big.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Add `ProximityVergence`, which estimates the distance of a target of known size from its image width and outputs the vergence for that distance, faded in below a start distance, with a near level for the pupils. | Completed | Distance from the face box width and the field of view, geometric vergence for the eye baseline (default 0.058 m from the Epi forward model in `Epi.ikg`), smooth fade-in between `start_distance` and `full_distance`, `gain`, `max_vergence`, `rate`, and `hold_time`. Release build without warnings; the test passes through `kernel_test.py` with ordered output and a 90-degree field of view: widths 0.15, 0.25, 0.375, and 0.75 give distances 1, 0.6, 0.4, and 0.2 m and vergence 0, 3.81 (half of 7.63 halfway through the fade), 11.42, and 20 (limited from 22.6) degrees with `NEAR` 0, 0.5, 1, 1; after `VALID` drops the vergence is held for 0.2 s and then returns to 0. Markdown checker clean. | `Added ProximityVergence, which converges the eyes on close targets` |
| 2 | Use it in the animatronics demo: add the proximity vergence to the animation vergence and constrict the pupils a little when someone is close. | Completed | `Near` (`ProximityVergence`, fade-in from 0.7 to 0.3 m, gain 1.5, at most 18 degrees, using the model's `horizontal_fov`) reads `FaceTargetSelector.TARGET` and `VALID`; its vergence is added to the animation vergence before `GazeLock.VERGENCE`, and `NEAR` times -2 mm is added to the pupil offset. Release build; 100 simulated ticks on the test video without warnings. Scripted approach with behaviours off, the face box widening from 0.2 to 0.95 and back: distance 1.24 m and 0.77 m give no convergence, 0.55 m 2.2, 0.43 m 7.9, 0.35 m 12.8, 0.30 m 16.2, and 0.26 m 18 degrees, with the eyes turning inward on Epi (left minus right -18 degrees) and relaxing back as the person steps away; the pupils narrow by about 1.5 mm against the arousal that a close face raises. Guide updated with the near response, its settings, and a close-up check. Markdown checker clean. Not run on the robot. | `Epi's eyes now converge on faces that come really close` |

### Outstanding issues and questions

- Not run on the robot: check at which real distances the eyes start to cross. The distance comes
  from an assumed face box width of 0.15 m; Apple Vision's boxes and children's faces may differ,
  so `object_width` or the distances on `Near` may need tuning.
- Close faces also raise arousal through its drive, which opens the pupils, so the near
  constriction partly cancels out; raise the `NearPupil` factor if the narrowing should show more.
- During cross-eyed the animation and the near vergence add up and can reach the eye limits.

## Subtle baseline, found-you chirp, and head panning starts copycat

Requested: let the eyes converge more; make the normal behaviour subtle (gaze, blinks, looking
around); a happy one-syllable chirp when a face is found; and start the mirroring game when a
person pans their head back and forth.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Add `OscillationDetector`, which signals when a value swings back and forth, such as a head panning from side to side. | Completed | Swings are movements of at least `amplitude` from the last turning point in alternating directions; `swings` of them within `window` give a one-tick `DETECTED`, followed by `refractory`. Release build without warnings; the test passes through `kernel_test.py` with ordered output: a 0-15 back-and-forth signal counts swings on ticks 3, 6, and 9 and is detected once on tick 9, while a 0-8 wobble and a single ramp to 40 are never detected. Markdown checker clean. | `Added OscillationDetector, which notices back-and-forth movements` |
| 2 | `BehaviorSelector` takes a `REQUEST` input that starts a given behaviour at once, like a reaction. | Completed | Optional `REQUEST` input; a rising edge in element i fires behaviour i regardless of `BUSY`, respecting `ENABLE` and the behaviour's cooldown. Release build without warnings; all 4 `BehaviorSelector` tests pass. The new request test connects one signal to `REQUEST[2:3]`: requests on ticks 3, 12, and 25 with a 2 s cooldown start behaviour 2 on ticks 3 and 25 only, while `BUSY` is 1 throughout. Markdown checker clean. | `BehaviorSelector can now start a behaviour on request` |
| 3 | Render a happy one-syllable chirp in three takes for finding a face. | Completed | New `chirp` sound: one syllable that scoops up and falls back slightly at the end, in three takes (0.34-0.41 s; 1230 to 2310 Hz, 880 to 1470 then 1350 Hz, 1190 to 2075 then 2056 Hz; -17 to -18 dB RMS; start and end at 0). Found while adding it: the variant seeds depended on the sorted position of each sound's name, so a new sound changed the takes of later ones; fixed seed numbers in `SEED_IDS` keep the previous seeds, and a full re-render reproduces all 24 committed files byte for byte. Markdown checker clean; not listened to here. | `Epi now has a happy one-syllable chirp for finding someone` |
| 4 | Update the demo: stronger convergence, a subtle baseline, the chirp when a face is found, and head panning starting copycat. | Completed | `Near` gain 2 and up to 26 degrees, cross-eyed 26 degrees; alone only look around (now +-12 degrees) and the odd curious peek; behaviours with company every 8-16 s; smaller glances (eyes 5, head 5 and 15 degrees) and calm movement size 0.6; found you is a perk-up with one chirp, a nod, and a smile; `HeadPan` (`OscillationDetector` on the face yaw, 12 degrees, 3 swings in 3 s) requests copycat through `Behaviors.REQUEST[10:11]`. Release build; 100 simulated ticks on the test video without warnings; no fan-in. Scripted scenario: alone, Epi only looks around (head pan within about -12 to 16 degrees, eyes within +-6 degrees); found you plays one chirp for each new person; a quick head pan at 40-44 s is detected at 41.3 s and starts copycat at once (mirroring from 42.5 to 50.4 s), a second detection during the game is ignored, and the slow sway at other times never triggers it; an animation plays during 43 of 120 s, with 6 sounds and 52 blinks. Scripted approach: no convergence at 1.24 and 0.77 m, then 2.9, 10.6, 17.1, and 21.6 degrees at 0.55, 0.43, 0.35, and 0.30 m, and about 25 degrees at 0.26 m, where the right eye reaches its 13-degree limit. Guide updated. Markdown checker clean. Not run on the robot, not inspected in a browser, sounds not listened to here. | `Epi animatronics is now subtle by default, chirps when it finds someone, and plays copycat when a person pans their head` |

### Outstanding issues and questions

- Nothing from this round has run on the robot. Check that a real head pan starts copycat reliably
  without false starts (Vision's yaw is noisy; tune `amplitude` on `HeadPan`), that the near
  convergence starts at the right distance, and how the subtle baseline feels over several minutes.
- At the strongest convergence the right eye reaches its 13-degree limit, so the eyes no longer
  turn symmetrically; raising the limit needs a check of the `EpiServos` eye limits on the robot.
- The chirp and the other sounds have not been listened to.
- The open questions from the earlier sections remain: whether farewells should wait for the
  playing animation, the `epi-head` widget's head axes, and the browser check of the dashboard.

## Longer copycat and stronger convergence

Requested after the chirp was approved: make the mirroring game last longer, and let the eyes converge
more as a face approaches.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Lengthen the copycat game. | Completed | Copycat now lasts 19.6 s at tempo 1 with mirroring from 0.6 to 18 s (was 11 s with 9 s of mirroring); its end squint and giggle moved with it. The behaviour cooldown rose from 15 to 25 s, so panning late in a game cannot restart it. Scripted run: the game lasted 18.3 s at Epi's arousal tempo, with 15.4 s of mirroring (7.9 s before), the pan offset following the person to +-9.9 degrees, and 6 blinks. Markdown checker clean. | `Epi's copycat game now lasts about twice as long` |
| 2 | Converge more, and from further away, as a face approaches, within Epi's eye limits. | Completed | `Near` now fades in from 1 m to 0.45 m with gain 3 and up to 31 degrees (was 0.7-0.3 m, gain 2, 26 degrees). Scripted approach, inward eye angle at each distance (before in parentheses): 0.77 m 3.8 (0), 0.55 m 15.3 (2.9), 0.43 m 19.4 with a 22.4-degree goal (10.6), 0.35 m 25.3 (17.1), 0.30 m 27.1 (21.6), peaking at 30.7. The right eye stays at its 13-degree limit from about 0.43 m and the left eye turns further inward, to about -14 degrees, within its -18-degree limit. Guide updated. Markdown checker clean. Not run on the robot. | `Epi's eyes now converge earlier and more strongly as a face approaches` |

### Outstanding issues and questions

- Not run on the robot: check the longer game with a person and how the stronger convergence looks.
- At close range the eyes turn asymmetrically, because the right eye stops at 13 degrees while the
  left goes on; a larger right-eye limit would need the `EpiServos` limits checked on the robot.
- The camera is in the right eye, which now turns up to 13 degrees inward when someone is close;
  a very close face can therefore move toward the edge of the image. Watch for lost faces at close range.

## Pupil range from EpiServos

After the pupil voltage was fixed, the user found that the pupils move too little. `EpiServos` ignores limits set in Dynamixel Wizard: `AutoCalibratePupil` resets them to 0..1023, finds the stop, and uses a fixed range of 80 steps from it.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | `EpiServos` gets a `pupil_range` setting (steps, default 80 as before) for the usable pupil servo range from the calibrated stop. | Completed | After the regulator repair a read-only reader built against the Dynamixel SDK showed the pupil servos at 7.5 and 7.4 V with no alerts. With the user moving the pupils by hand between the stops, servo 2 travelled 0 to 286 and servo 3 15 to 332 steps (about 290-320 steps, 84-93 degrees), against the 80 steps `EpiServos` used. New `pupil_range` setting (default 80, unchanged behaviour; clamped to 10..1023 and to the servo's limit); the demo sets 260 as a model-level `pupil_range`. Release build without warnings; a sweep of 10, 16, 5, 16, 5, 10 mm on EpiBlack calibrates to limits 5-265 and 11-271, inside the measured stops, without warnings or alerts. The amount the pupils open was not measured; the user judges it. | `EpiServos can now use the full mechanical range of the pupils` |

### Outstanding issues and questions

- `EpiServos` reports the commanded pupil size as the present size instead of reading it from the
  servos, so the pupil position cannot be seen in the Ikaros log; reading it back (address 37) would
  allow a check without a separate tool.
- The 260-step range was measured on EpiBlack only. Other robots keep the default of 80 unless they
  are started with `pupil_range`; measure them the same way before raising it.
- Calibration finds only the closed stop; if the top end is reached by a worn or loosened pupil
  mechanism, the servo can push against it. Watch for hardware alerts or warm servos at full open.

## Per-servo pupil calibration

The left pupil is slightly smaller than the right one. The pupil servos have different mechanical travel
(about 286 and 332 steps), but `EpiServos` calibrates only the stop at goal position 0 and adds the same
`pupil_range` to both, so the other end of the range differs mechanically between the pupils.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | `AutoCalibratePupil` finds both mechanical stops of each pupil servo by driving toward the second stop with the lowered torque limit until it stalls, and maps the pupil size onto each servo's own range. `pupil_range` 0 selects this; other values keep the fixed range. | Abandoned | The stall detection did not stop the servos: with the torque limit at 500 the auto calibration drove both pupils to calibrated limits of 718 and 990 steps, far beyond the stops of about 286 and 332 steps measured by hand, and the pupil servos then stopped answering ("no status packet"). The user reported that the sweep almost broke the pupils. The change was reverted and never committed; the committed fixed `pupil_range` of 260 is unchanged. Do not retry with motion tests on the robot before the pupil mechanism and servos are inspected. | None |

### Outstanding issues and questions

- The pupils and their servos need inspection after the failed calibration: look for skipped gear
  teeth or a loose linkage, power-cycle the robot, and read the servos' hardware error status
  (read-only) before any further motion.
- Matching the pupils mechanically still needs a safer method, for example measuring both stops by
  hand with torque off, as before, and giving each pupil its own range as parameters instead of
  moving toward the stop with the servo.

## Pupil limits set by Ikaros

After the failed auto calibration, the user asked for Ikaros to set the pupil limits to a safe range instead of Dynamixel Wizard.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | `EpiServos` gets `pupil_limits`: the servo positions of the open and closed end of each pupil. When set, they are written to the servos at start-up and the pupils are not driven to any stop. The demo sets them to the hand-measured stops minus a margin. | Completed | New `pupil_limits` matrix (left open, left closed, right open, right closed; all zeros keep the previous calibration with `pupil_range`). When set, `AutoCalibratePupil` returns before any pupil motion with these limits, which `SetPupilParameters` writes as the servos' CW and CCW limits; invalid values (outside 0..1023 or less than 20 steps apart) stop the model at startup. The demo's model-level `pupil_limit_steps` is 12, 274, 27, 320: the hand-measured stops (about 0-286 and 15-332 steps) minus about 12 steps. Release build without warnings; the demo starts in simulation with the new setting and rejects invalid limits; Markdown checker clean. Not run on the robot: the user is to power it on and run it with the pupils unobstructed. | `EpiServos can now take safe pupil limits instead of finding the stops` |

### Outstanding issues and questions

- Not run on the robot. The first run should be short, with the user present, watching that neither
  pupil strains at either end. The limits written to the servos can be read back with the read-only
  reader (addresses 6 and 8 of IDs 2 and 3).
- The values come from a hand measurement made before the pupils were strained; they should be
  measured again if the mechanism changed.

## Gentle pupil calibration and LED light shows

Requested: restore the pupil calibration so that it drives the pupils in the opening (dilation)
direction to the stop without pushing hard; and add LED animations to the animatronics demo: a
top-to-bottom row wave through the eyes and mouth, a circular animation mirrored between the eyes
with the mouth split into left and right halves, colors random or from arousal, started when the
robot is bored and interrupted by a face, and played for 5 s without interruption from a button. A
per-robot LED orientation check (top, left, right, bottom of the eyes; top and bottom row of the
mouth) is saved in a JSON file so each robot needs it once. The robot is not available; everything is
verified in simulation.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Restore the pupil calibration gently: open direction, slow speed, low torque, stop as soon as the pupil stops moving, and a range per pupil. | Completed | Direction confirmed from the code: `PupilMMToDynamixel` maps larger pupils to lower positions, so the calibration's move toward position 0 opens the pupils; the user saw them close when the abandoned calibration drove toward higher positions, which agrees. The calibration now sets moving speed 150 and the new `pupil_calibration_torque` (default 300; was a fixed 500), polls the positions every 50 ms, switches the torque off as soon as both pupils have moved no more than 2 steps for 0.25 s (at most 3 s, with a warning if still moving), and uses the new per-pupil `pupil_ranges` from the open stop (zeros keep `pupil_range`). `pupil_limits` still bypasses the motion. The demo uses `pupil_travel` 255, 290: the hand-measured travel of about 286 and 317 steps minus a margin in case the gentle calibration stops early. Release build without warnings; the demo starts in simulation; Markdown checker clean. Not run on the robot (busy with another experiment); to be tested slowly with the user. | `EpiServos now opens the pupils gently to calibrate them` |
| 2 | Add `Boredom`, which signals bored episodes after quiet, calm time and ends them when something happens. | Completed | Random quiet wait (`min_wait`..`max_wait`), random episode length (`min_duration`..`max_duration`), activity or arousal at or above `arousal_threshold` restarts the wait. Release build without warnings; the test passes through `kernel_test.py` with ordered output: with a 1 s wait and a 0.5 s episode, `BORED` is 1 on ticks 9-13, ends at the activity on tick 14, is 1 again on ticks 24-28, and ends on tick 29; an instance with arousal 0.5 is never bored. Markdown checker clean. | `Added Boredom, which decides when a robot gets bored` |
| 3 | Add `EpiLightShow`: wave and mirrored circle patterns over the expression LEDs, colors random or from arousal, and a saved per-robot LED orientation check. | Completed | LED layout taken from the `epi-head` widget (eye LED i at i x 30 degrees clockwise from 3 o'clock seen from the front, 9 at the top; mouth rows counted from the viewer's left, `MOUTH_HIGH` on top, LEDs 0 and 7 hidden). The module mixes patterns over the expression, maps the widget layout to each robot's wiring from a JSON orientation, and offers check, rotate, flip, swap_mouth_rows, reverse_mouth, save, play, and stop commands. Release build without warnings; all 3 tests pass with ordered output: the wave lights LED 9, then 8 and 10, down to LED 3 with a 0.4 trail, then the upper and lower mouth rows, and restarts on tick 10; the circle goes 9, 0, 3, 6 in the robot's left eye and 9, 6, 3, 0 in the right eye, with the mouth halves going 4, 6, lower 7, lower 5 and 3, 1, lower 0, lower 2; an unknown robot starts in the check with `CALIBRATED` 0, and a stored orientation (left eye LED 0 at the bottom, counterclockwise; mouth rows swapped and reversed) puts the check colors on the hand-computed physical LEDs. A live WebUI run checked rotate, flip, swap_mouth_rows, save (the JSON file is written and the check ends), and a 5 s play that runs without `RUN`. Markdown checker clean. | `Added EpiLightShow, which plays LED patterns and maps them to each robot's wiring` |
| 4 | Use them in the animatronics demo with dashboard buttons and the check, and update the guide. | Completed | `LightShow` sits between `Expression` and `FadeCandy`; `Boredom` (wait 20-45 s, episodes 6-15 s, arousal threshold 0.3) drives `RUN`, the face's `VALID` drives `ACTIVITY` and `INTERRUPT`, arousal drives speed and, with the new model setting `light_colors=arousal`, the color; the robot view shows `LightShow`'s `VIEW_` outputs. Dashboard: Wave and Circle buttons (5 s, `play`), LED check, rotate and flip for each eye, swap and reverse for the mouth, Save LEDs, and Lights and LEDs saved tables; the camera and plots moved down 104 px. Layout check: no widget overlaps, matching `x/_x` and `y/_y`, all sources and commands resolve, no fan-in. Simulated runs: EpiBlack is not yet in `UserData/EpiLEDCalibration.json`, so the demo starts with the check and a warning; with an EpiBlack orientation in a scratch user directory, a scripted 60 s alone gives a bored episode at 43.6 s with wave, circle, and wave 5 s apart and a fade-out at 55.9 s; with a 10 s wait and a long episode, the show alternates wave and circle from 10 s and stops on the tick after a face appears at 60 s, returning to the expression colors. The guide describes the light shows, the LED check, the settings, and the dashboard, and its eyelid check now refers to the LED check. Markdown checker clean. Not run on a robot, and the dashboard not inspected in a browser. | `Epi animatronics now plays light shows when bored and checks each robot's LED orientation` |

### Outstanding issues and questions

- The pupil calibration is untested on the robot; test it slowly with the user before using the demo.
  Watch that the pupils open (not close) at start-up and that the torque switches off at the stop.
- The LED check has not been done on any real robot; EpiBlack and every other robot show the check
  pattern on their first run until its orientation is saved.
- `EpiExpression`'s own `led_first_angle` and `led_direction` are now redundant in this demo, because
  `EpiLightShow` maps all LEDs; they still matter in models without `EpiLightShow`.
- A face stops a show at once, while the expression returns without a fade; a short fade might look
  softer.
- The dashboard has not been inspected in a browser, in particular the new button rows.

## Hand-measured pupil limits for EpiBlack

Testing on EpiBlack with the user: the gentle calibration ran without straining (open stops found at 23 and 29, limits 28-283 and 34-324), and a slow, guarded move to 9 mm showed both pupils closing smoothly, the left slightly more than the right. Hand measurements with torque off gave the left pupil 0 (open, the end of the servo's sensor range) to 256-261 (closed; 286 in an earlier measurement) and the right 33 to 373-379 (332 earlier). The calibrated left limit of 283 is beyond the left's closed end, and the demo can command 5 mm, which is position 270 on the left, at least on the first tick.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | The demo uses fixed, hand-measured pupil limits (left 12-244, right 45-320: the smaller closed readings, 12 steps inside each end) instead of the calibration. | Completed | The demo starts in simulation without warnings; Markdown checker clean. With these limits 5 mm is position 232 on the left and 306 on the right, and 16 mm 67 and 111, all inside the measured ends. Not yet run in the demo on the robot. | `Epi animatronics now uses hand-measured pupil limits on EpiBlack` |

### Outstanding issues and questions

- Whether both pupils look the same size at the same setting was not yet checked by eye with these limits; the slow, guarded tool in the scratchpad can move them to 9, 16, and 5 mm for comparison.
- The left pupil's open end is at the end of the servo's sensor range (position 0); if it opens further than the sensor can read, its open limit is slightly conservative.
- The closed ends differed between the two hand measurements by 30-45 steps, so the right pupil may close further than its limit of 320 allows.

## Stereo distance and a still head for close faces

Requested: when a face comes close and the eyes converge, the neck tilt and pan should stay put; use
the newly connected left eye camera to decide when only the eyes should converge. A finger-following
mode may come later.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | `GazeLock` takes a `HOLD_HEAD` input: while it is above `hold_threshold`, the followed head goal stays where it is and only the eyes follow the target. | Completed | Release build without warnings; all 7 GazeLock tests pass. The new hold test reuses the follow scenario: while `HOLD_HEAD` is 1 the head goal stays at 0 and the eyes follow the target 20 degrees away to their 13-degree limit; after it drops, the head moves 10 and then 15 degrees toward the target. Markdown checker clean. | `GazeLock can now hold the head still while the eyes keep following` |
| 2 | Add `StereoDistance`, which triangulates the distance of a target seen by two cameras in panning eyes. | Completed | Distance = baseline / (tan right direction - tan left direction), with each direction the eye pan plus the image angle; measurements only when both cameras see the target at similar heights and within `min_distance`..`max_distance`; smoothing and hold time. Release build without warnings; the test passes with synthetic geometry (baseline 0.06 m, 90-degree field of view): 0.5 m straight ahead with parallel eyes and 0.499999 m with the eyes converged +-5 degrees, 1 m for a target 0.2 m off to the side, the distance held for 0.2 s after one camera loses the target and then invalid, and detections at heights -0.3 and 0.3 rejected. Markdown checker clean. | `Added StereoDistance, which measures distance with two eye cameras` |
| 3 | `ProximityVergence` takes an optional measured `DISTANCE` in place of its size-based estimate. | Completed | New optional `MEASURED_DISTANCE` and `MEASURED_VALID` inputs and a `MEASURED` output (first committed as `DISTANCE` and `DISTANCE_VALID`; see the fix below). Release build without warnings; both tests pass: the existing test is unchanged, and the new one shows a valid measurement of 0.4 m used instead of the size estimate of 1 m (11.42 degrees), the size estimate used again when the measurement is not valid, a valid measurement of 0.6 m (3.81 degrees) used while the target box is lost, and release when neither is available. Markdown checker clean. | `ProximityVergence can now use a measured distance` |
| 4 | Use the left camera in the animatronics demo: a second face pipeline, stereo distance for the convergence, and a still head while a face is close; update the guide. | Completed | New model settings `left_camera` (default `InputVideoStream`; `InputImage` shows `NoCamera.jpg`, in which no face is found) and `left_eye_url`; a left chain `LeftEye`, `LeftFaces`, `LeftTarget`; `Stereo` (`StereoDistance`, eye pans from the servo positions) feeds `Near.MEASURED_DISTANCE`; `Near.NEAR` drives `Gaze.HOLD_HEAD` with `hold_threshold` 0.3 (about 0.8 m). Dashboard: a bottom row with the face distance plot, measured and near tables, and the left camera image. Layout check: no widget overlaps, matching `x/_x` and `y/_y`, all sources resolve, no fan-in. The demo starts in simulation with both cameras on a video and with `left_camera=InputImage`. Scripted approach with a 10-degree sideways drift while close (face-size distance, since the probe replaces the cameras): the head stays at -1.25 degrees pan and -1.48 degrees tilt while the face is close, the eyes converge and follow the drift, and the head follows again once the person steps back. This run found the `GazeLock` hold bug fixed above. The stereo distance itself is tested only with synthetic geometry; with real cameras it is not yet checked. Guide updated. Markdown checker clean. | `Epi animatronics now measures face distance with both eyes and keeps the head still for close faces` |

### Additional fixes found during the tasks

| Fix | Verification | Commit |
|---|---|---|
| The first commit of task 3 named the new input `DISTANCE`, the same as an existing output. The input then bound to the module's own output, so the module kept using its previous distance; the original test failed while the new one passed by coincidence, and the failure was committed because the commit command did not stop on the test result. The inputs are now `MEASURED_DISTANCE` and `MEASURED_VALID`. | Both `ProximityVergence` tests pass. | `ProximityVergence now names its measured-distance inputs MEASURED_DISTANCE and MEASURED_VALID` |
| `GazeLock` with `HOLD_HEAD` fell through to `MANUAL_HEAD` while holding, so in the demo, where `HeadMirror` drives `MANUAL_HEAD` with the person's direction, the held head still followed the person (the scripted approach moved it from 3.6 to 8.3 degrees while the face was close). Holding now keeps the head goal; `MANUAL_HEAD` applies only while `FOLLOW` is 0. The hold test missed it because its manual goal equalled the held position; it now uses a different manual goal. Found in task 4. | The strengthened hold test fails with the previous code and passes now; all 7 GazeLock tests pass; the scripted approach keeps the head at -1.25 degrees pan and -1.48 degrees tilt while the face is close. | `GazeLock now keeps the head still while holding, also with MANUAL_HEAD connected` |

### Outstanding issues and questions

- The stereo distance has not been checked with the real cameras: compare **Face distance** with a tape
  measure, and check that **Distance measured by both eyes** shows 1 when both cameras see the face.
  If the left camera's image or orientation differs from the right's, `x_sign` may differ for it.
- The two cameras deliver frames at different times; while a face moves quickly, the measured
  distance can jump briefly; the smoothing reduces this.
- While the head holds, a close person who moves far to the side or up or down can leave the eyes'
  range; a mild head follow with a large dead zone could be added if that happens often.
- Possible Ikaros issue: the kernel accepts a module with an input and an output of the same name
  without a warning, and the input then reads the output. A GitHub issue could be proposed; the
  `ikaros-bot` token is not installed here.
- A finger-following mode is planned next; Apple Vision's hand pose detection can provide the
  index fingertip.

## Camera settings at start-up

Both eye cameras streamed 1280 x 720 at up to 45 frames/s and JPEG quality 85 (about 590 kB per frame), which overloaded the USB link (realtime lag, corrupt frames) and, in the cropped 1280 x 720 sensor mode, narrowed the field of view, so the stereo distance came out too short and the eyes stayed converged. The settings were set by hand to 640 x 480, 30 frames/s, quality 70 on both Pis (identical otherwise), but the camera server forgets them when a Pi restarts. Requested: Ikaros sets them automatically at start-up.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Add `UV4LSettings`, which sets the resolution, frame rate, and JPEG quality of a UV4L camera server when the model is built, before the camera stream opens, and checks that they took effect; use it for both eyes in the animatronics demo. | Done | Build without warnings. `UV4LSettings` tests: 3 of 3 pass (non-http url skipped, unreachable server warns, `required` stops the model). On EpiBlack: quality 71 and back to 70 on both eyes; the demo (`simulate=true`) restored the right eye from quality 85 and from 1280 x 720 to 640 x 480 / 30 / 70 with no warnings, and the stream then delivered 640 x 480 frames; requesting 1280 x 720 while another client streamed gave the expected warning and left the camera unchanged. | This commit |

### Outstanding issues and questions

- Without the robot, each eye adds about 2.5 s to start-up while the `.local` host name fails to resolve, followed by a warning.
- Frame rate and quality use the raspicam driver's control ids; another UV4L driver may need different ids.

## Finger following

Requested: a finger-following mode for the animatronics demo. An open palm held up for 2 s starts it; Epi then follows the index fingertip with eyes and head and converges on it as it comes close, using both cameras, with a sound and light cue on entry and exit; the mode ends when no hand has been seen for 3 s. The tuning saved from the WebUI in `UserData/EpiAnimatronics.ikg` also goes into the committed demo. A different Epi is connected and in use for experiments, so all verification runs without the robot or its cameras.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Copy the WebUI tuning into `Projects/Epi/EpiAnimatronics.ikg`: `aim_y` 0, `vertical_fov` 70, `HeadPan.smoothing` 20. | Done | Compared both files module by module; only these three settings differed apart from WebUI defaults (the WebUI copy also replaced `@camera` with a fixed class, which is not copied). The demo runs 25 ticks with simulated servos and still images without errors. | This commit |
| 2 | Add `AppleVisionHandDetector`: the most prominent hand's index fingertip (or palm) as a target row, VALID, an open-palm score, and the hand count. | Done | Build without warnings. Committed test: an image without a hand gives VALID 0 and zero outputs (passes; the very first run failed once, probably while Vision loaded its hand model, and passed in four runs after). Checked locally with Wikimedia Commons photos that are not committed because their licences need attribution: two open palms gave OPEN 1 with TARGET on the index fingertip, and with `point=palm` on the middle of the palm; two fists gave OPEN 0. Not verified: POINTING (no detected pointing photo) and a live camera. | This commit |
| 3 | Add `HoldLatch`: becomes active after an input has been held for `hold_time` and ends after a presence input has been absent for `release_time`, with start and end pulses. | Done | Build without warnings. `test_HoldLatch.ikg` passes: a 0.3 s hold does not start the mode and resets PROGRESS, a 0.5 s hold starts it on the right tick with one START pulse, a 0.2 s absence of PRESENT is forgiven, a 0.3 s absence ends it with one END pulse, and without PRESENT the end of INPUT ends it. | This commit |
| 4 | `BehaviorSelector`: optional `PAUSE` input that stops spontaneous behaviours, greetings, and farewells while still letting `REQUEST` start behaviours. | Done | Build without warnings. New `test_BehaviorSelector_pause.ikg`: during the pause a REQUEST edge starts its behaviour on the same tick, the greeting for an arrival is dropped (never fires), and no spontaneous behaviour starts; the first one comes 0.5 s after the pause ends. All five BehaviorSelector tests pass. | This commit |
| 5 | Finger mode in the demo: hand detection on both cameras, `HoldLatch` on the open palm, gaze, stereo, and convergence switched to the fingertip, behaviours paused, entry and exit animations, guide and dashboard. | Done | Demo runs without the robot (simulated servos and LEDs, still images) without errors. With a copy of the demo whose right camera reads 100 frames of an open-palm photo and then 150 frames without a hand: the mode starts 2.0 s (50 ticks) into the palm with one START, `finger_on` (behaviour 11) is requested while behaviours are paused, the gaze target switches to the index fingertip (0.65, -0.83) and the gaze locks, and 3 s after the hand is gone the mode ends with `finger_off` (behaviour 12). Static layout check: no widget overlaps, matching x/_x and y/_y, components left of the dashboard. Not verified: a live camera, convergence on a finger with both cameras, and the robot. | This commit |

### Outstanding issues and questions

- Finger following has not been tried with a live camera or on a robot; the connected Epi was in use for experiments. Check it with the new step 12 in `EpiAnimatronics.md`, with the user present.
- `AppleVisionHandDetector` has a committed test only for an image without a hand; the photos used to check open palms and fists are not committed because their licences require attribution. `POINTING` is unverified.
- Hand detection now runs on both cameras every tick in addition to face detection; its cost per frame was not measured. If the demo lags, set `downsample="1"` on `Hands` and `LeftHands`.
- Finger on and finger off share the 25 s behaviour cooldown, so a quick second start or end of the mode plays no cue.
- While testing, `InputVideoStream` with a video file and `synchronized_framegrabber=true` blocked the tick for more than 5 s, and with `false` it ran through the whole file at once and showed only its last frame. This was with the uncommitted `InputVideoStream` changes in the working tree, so it may come from them; the guide's advice to test with a recorded video may not work at present.
- A model saved from the WebUI (`UserData/EpiAnimatronics.ikg`) replaces `class="@camera"` with the resolved class, so the saved copy can no longer switch cameras from the command line. This may be worth an Ikaros issue.

## Finger following lag

After finger following was added, the WebUI stopped responding and Ikaros reported "Realtime lag exceeded 1 seconds" every few seconds.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Make hand detection cheaper: `AppleVisionHandDetector` gets `interval` (detect on every n-th tick, keep the result in between) and an optional `ENABLE` input; the demo detects hands at half resolution on every second tick and in the left camera only during finger following. | Done | 250 ticks without real time, still images, three runs each on an Apple M2: before finger following 5.0-5.6 s (about 21 ms per tick), with finger following 11.4-12.5 s (about 45 ms per tick, over the 40 ms tick), after this change 5.7-7.1 s (about 23 ms per tick). Open-palm photo: `ENABLE` 0 gives VALID 0 on every tick, and `interval="3"` keeps the fingertip on every tick. The palm-then-nothing sequence still starts the mode at tick 48 and ends it at tick 177, with the fingertip within 0.006 of the full-resolution result. Hand detector test passes. | This commit |

### Outstanding issues and questions

- The timings are from this Mac with still images; live camera streams and the WebUI add work, and the other computer may be slower. If the lag returns, raise `interval` on `Hands` to 3.
- `AppleVisionFaceDetector` measured slower with `downsample="1"` (about 26 ms per tick) than without (about 15 ms), apparently because the Ikaros downsampling before Vision costs more than it saves. The demo still uses `downsample="1"` on both face detectors; switching to 0 may save more time but was not checked for its effect on face detection.

## Finger following only on a deliberate palm

Finger following started too easily: Epi was distracted by hands and followed them instead of faces.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Start finger following only on a deliberate gesture: `HoldLatch` accepts several INPUT elements with one threshold each, all of which must hold at once; `AppleVisionHandDetector` gets `min_width`; the demo requires all four fingers stretched out (OPEN above 0.9), a hand wider than 0.15 (within about 0.8 m), and the gaze locked on a face, for 2 s. | Done | Build without warnings. HoldLatch tests 2 of 2 pass; the new one shows that the hold does not build up while any one condition is missing and starts on the right tick when all hold. Hand detector test passes; `min_width` 0.5 keeps and 1.0 ignores a hand 0.73 wide. Demo with still images: an open palm without a face does not start the mode; a face with an open palm beside it starts it 2.0 s after the gaze locks; the same images with the face condition forced off do not start it. (`FaceTargetSelector.ignore_repeated` was switched off in the test copy only, since still images repeat the same face box.) | This commit |

### Outstanding issues and questions

- `min_width` on the detector is not used in the demo, because the same detector tracks the pointing finger, whose hand is narrower than an open palm; the width condition is applied in `HoldLatch` instead.
- If a raised palm covers the face for more than 1.5 s, the gaze lock ends and the hold starts again; hold the palm beside the face.
- The WebUI rendering problem reported on the other computer could not be reproduced here: the WebUI server answers the page, network, and update requests in milliseconds with the demo running.

## Eyes cross on a close finger instead of finger following

The head should stay with faces: finger following is removed, and a finger brought close to the cameras only makes the eyes cross.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Remove finger following from the demo (palm latch, left-camera hand detection, target switches, behaviour pause, finger on and finger off animations) and restore the face-only connections; add `FingerNear`, a second `ProximityVergence` on the right camera's hand, whose vergence is combined with the face's by `Maximum`. Update the guide and dashboard. | Done | Demo runs without the robot (simulated servos and LEDs, still images) without warnings. Hand photos in the right camera: a hand 0.28 wide gives an estimated 0.26 m and 0.8 degrees of vergence, hands 0.5 and wider give the full 31 degrees, and the eye goals move apart (164.5 and 193); the head goal comes only from the idle look-around, since no face is in view. `HoldLatch`, `AppleVisionHandDetector`, and the `BehaviorSelector` `PAUSE` input stay in the library. | This commit |

### Outstanding issues and questions

- Untested with a live camera and the robot; see step 12 in `EpiAnimatronics.md`.
- The distance comes from the width of the hand, taken as 0.1 m, so an open hand looks closer than it is and crosses the eyes from about 0.5 m. Requiring `Hands.POINTING` would limit it to a pointing finger, but POINTING has not been verified.
- A finger very close to the camera may show too little of the hand for Vision to detect it.
- The 5 s task timeout at start-up reported by the user is not explained yet; a hardware module's first tick is the likely cause.

## Pupil model and brain activity in the animatronics demo

The pupil model of `PupilLedDemo.ikg` (Ikaros 2) comes to the animatronics demo: a light reflex and an emotional (locus coeruleus) pathway made of `Nucleus` modules drive the pupils, a new way to elicit emotions replaces the QR markers, and the brain activity view is redone as a livelier animation. Nothing may add noticeable lag to the demo.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Port `EyeModel` (pupil sphincter and dilator to pupil diameter) to Ikaros 3 as a C++ module with a test. | Done | Build without warnings. `test_EyeModel.ikg` passes: dilator 0.5 gives 6 mm in a 2-10 mm range, a sphincter with `m3` 2 cancels it (2 mm), drives above 1 and below 0 clip to 10 and 2 mm, and with `time_constant` 0.1 s at 0.1 s ticks the pupil starts at its first goal (5 mm) and moves 63 percent of the way to each new goal per tick. | This commit |
| 2 | Pupil circuit in the demo: light level from the right camera (mean brightness) through pretectal area, Edinger-Westphal, and ciliary ganglion to the sphincter; locus coeruleus through IML and SCG to the dilator; amygdala (`Delta`) and arousal into the locus coeruleus; combined with the animation and near-response pupil offsets. Measure the cost per tick. | Done | Brain group alone with a mid-grey image: pupil 8.1 mm at arousal 0.15, 10.2 at 0.4, 13.0 at 0.8, and 14.1 at 1.0. Demo without the robot (simulated servos, still images): uniform images of brightness 0.05, 0.43, and 0.94 give 12.4, 8.1, and 5.5 mm at rest, and `Expression.PUPIL_SIZE` follows. Cost: 1000 ticks with a 640 x 480 image take the same time with and without the group (under 0.1 ms per tick); 250 ticks of the full demo take 12.2-12.4 s with it and 11.0-13.3 s without (the Mac was busy with other programs). The arousal offset on the pupils is replaced by arousal through the locus coeruleus, and `EpiExpression` now only adds offsets (`idle_pupil` and `engaged_pupil` 0). | This commit |
| 3 | Learned fear of a finger poke in place of the QR markers: a finger close up (`FingerNear.NEAR`) is the unconditioned stimulus and a hand in view (`Hands.VALID`) the conditioned one; the amygdala (`Delta`, with a new optional `extinction` rate) learns to predict the poke, and its response and the poke itself excite the locus coeruleus. | Done | Build without warnings. New `Delta` test: with `extinction` 0.5 the learned response halves on each tick without the US (0.875, 0.4375, 0.21875) while the default keeps it (0.875); kernel tests 167 and 169 (Delta state) pass. Demo copy reading a 580-frame image sequence (a small far hand that counts as seen but not close, an open palm filling the view as the poke, and mid-grey frames), without the robot: before learning the far hand leaves the pupil at 8.1 mm and the amygdala response at 0; three pokes of 1 s raise the response to 0.28, 0.50, and 0.63; after 4 s of rest (pupil 8.4 mm) the far hand alone gives a response of 0.67 and widens the pupil to 12.1 mm. Arousal stayed at 0.15-0.2 throughout, so the widening comes from the amygdala. | This commit |
| 4 | New WebUI widget that animates the brain activity: regions that glow with their activity and signals that travel along the pathways; a brain page in the demo. | Done | New `brain-activity` widget (`WebUIWidgetBrainActivity.js`, registered in `index.html` and the widget menu): a side view of the brain with the eye, eleven nuclei that glow and pulse with their activity, and comet-like spikes along thirteen pathways, more of them the more active the sending nucleus; the pupil in the drawn eye follows `pupil_source`. Its drawing code (`BrainActivityRenderer`) needs only a 2D context. New `Source/WebUI/UnitTesting/brain_activity_test.js` (run with node; not yet in the kernel test runner list) passes: silent nuclei send no spikes, the light reflex sends spikes only along its own pathways, more activity sends more spikes, values and pupil follow their targets, and rendering makes only standard canvas calls. Rendered in headless Firefox with test values for calm and fear (checked by eye: no overlapping labels, the pupil small when calm and wide in fear). The Brain group page has the widget (1000 x 620) above four plots, with its components to the right. Not checked: the widget inside the live WebUI, since headless screenshots of the WebUI only show its loading screen. | This commit |
| 5 | Guide, dashboard, and checklist for the pupil model. | Done | `EpiAnimatronics.md`: a "Pupils" section on the two pathways, learned fear, and the pupil sizes to expect; settings rows for `Brain.PTA`, `Brain.LC`, `Brain.Eyes`, and `Brain.Amygdala` (the 14 s extinction half-life follows from 0.002 per tick at 25 ticks per second); the Brain page in the dashboard section; checklist step 13 (covering the camera, a phone light, and three pokes); known limitations for exposure, forgetting at exit, and the unchecked WebUI page. No stale references to the removed arousal pupil offset remain. | This commit |

### Outstanding issues and questions

- The pupil model, the learned fear, and the Brain page have not been tried with a live camera, on the robot, or in a browser; the animated brain was checked only in a test page in headless Firefox. Check the Brain page in a normal browser and follow step 13 in `EpiAnimatronics.md` with the robot.
- The camera's automatic exposure partly cancels slow changes in room light; the light reflex is clearest for sudden changes.
- The learned fear starts from zero on every run unless the state is saved with `-W` and loaded with `-L`.
- `Source/WebUI/UnitTesting/brain_activity_test.js` passes with node but is not in the list of WebUI tests in `kernel_test.py`; adding it there is a one-line change to the test runner.
- The "Task execution exceeded 5 seconds" stop that the user reported happened twice here in simulation (once with the committed demo plus the new Brain group, once with a test copy), but not in 100 further runs of the demo, and not more often with the Brain group than without. Its cause is still unknown. Naming the module whose task is still running in the timeout message would identify it; that is a kernel change and needs the user's approval.

## Pupil light reflex and brain page fixes

The user reported that the brain widget did not show and that the pupils did not respond to light shone into the cameras.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Make the light reflex respond to a light shone into the camera: the light level is now the mean brightness plus 2.5 times the fraction of pixels above 0.9 (glare), and the Edinger-Westphal nucleus is damped less by the locus coeruleus (`psi` 0.6), so light wins more clearly when Epi is excited. Spread out the Brain group's components, which overlapped. Explain in the guide that a browser tab opened before the widget existed must be reloaded. | Done | In the live run the light level stayed at 0.38 and the pupils near 12 mm with the locus coeruleus at about 0.5, so the mean brightness barely changed and arousal dominated. Demo copy with synthetic images, without the robot: mid-grey gives light 0.43 and 7.5 mm, a darkened scene with a white disc over 15 percent of the view gives glare 0.15, light 0.75, and 5.7 mm within about a second, and a dark image gives 12.3 mm. Calculated for higher arousal: bright light narrows the pupils from about 12 to 9 mm. The glare costs about 0.15 ms per tick (2000 ticks with a 640 x 480 image). The brain widget renders and animates on the Brain page of the live WebUI (screenshot in headless Firefox through a delayed wrapper page), so the widget itself works; the user's tab most likely predates it. | This commit |

### Outstanding issues and questions

- Not checked with the real eye camera: reading a snapshot from the camera was not permitted, so the glare threshold and gain are chosen for synthetic images and may need tuning (step 13 in the guide).
- The 5 s task timeout appeared once more here, with the WebUI open, in the first demo run after several minutes without runs; 40 back-to-back real-time runs and 4 runs with a browser attached did not show it. All three occurrences were first runs after a pause, which points to something loaded lazily after idle time, but this is not confirmed.

## Stronger pupil light reflex

The user found the pupils' response to light and to a covered camera too weak: light should close them more, and covering the eyes should open them fully.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Refit the pupil brain: stronger sphincter (`m3` 1.5), higher gain (`amplifier` 1.3), less damping of the Edinger-Westphal nucleus by the locus coeruleus (`psi` 0.2), and a slightly higher resting locus coeruleus drive (`alpha` 0.3), so that normal room light gives about the same sizes as before. | Done | Calculated steady states: covered camera 16 mm at every arousal level; normal room (light 0.45) 7.7, 10.3, and 13.5 mm at arousal 0.15, 0.4, and 0.8; light level 1.0 gives 5 mm at rest and with company and 7.3 mm when very excited; a learned fear of a hand still widens the pupils to 13.7 mm at rest. Demo copy with synthetic images, without the robot: mid-grey 7.8 mm, a light shone in 5.0 mm after 1 s, a dark image 16.0 mm (15.3 mm after 1 s). | This commit |

### Outstanding issues and questions

- Not yet checked with the real camera; the light level in a normal room depends on the camera's exposure (0.38 in the user's last run), and the pupils now react strongly between light levels 0.3 and 0.6.
- The 5 s task timeout happened once more here in a single-threaded batch run of a demo copy, so it is not tied to the WebUI, real-time mode, or threads. A watcher samples the user's next run when it happens.

## WebUI disconnects on the robot

The WebUI disconnected while the demo ran on EpiBlack. A sample of the running process showed the HTTP thread waiting 95 percent of the time for the kernel lock, which a tick holds from start to end, and the main loop ticking back to back without idle time. The biggest share of each tick was `EpiServos` head communication, mostly separate torque-enable and PWM writes, each waiting for a status reply, to the two eye servos on every tick.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | `EpiServos`: write torque enable and goal PWM to servos without goal-current support only when they change, after a failed write, and once a second; the goal position is still written every tick. | Done | Build without warnings; the simulated demo runs. In the sample of the user's run, `write1ByteTxRx` (torque) and `write2ByteTxRx` (PWM) took 560 of the 695 samples spent in separate writes, so this removes four of the six waiting writes per tick on the head bus. Not verified on the robot yet: the eyes must still move and hold, the WebUI must stay connected, and no "Direct position write failed" warnings may appear. | This commit |
| 2 | Run the hardware and the detectors in parallel: one-tick delays on the connections into `Servos` and `LEDs`, from `LeftTarget` into `Stereo`, and into and out of `Hands`, so that each becomes its own task; the kernel runs separate tasks in parallel but the modules of one task one after another. | Done | The user measured 50-60 ms per tick on the robot. Simulated demo without real-time pacing (still image with a face): 21.2 ms per tick before (23.0 ms with one thread, so almost nothing ran in parallel), 12.9-15.2 ms after; the servo and LED communication, absent in simulation, now also runs in parallel with it. Fear learning with the image sequence still gives the same result (learned response 0.26, 0.51, 0.65 over three pokes; a far hand afterwards widens the pupils from 8.4 to 12.7 mm). Costs: motor and LED commands one tick (40 ms) later, the stereo distance one tick older, and finger and fear reactions two ticks later. Not verified on the robot yet. | This commit |

## Visible fear response and its conditioning

A poke makes Epi visibly scared with LEDs and sound; after a few pokes the hand alone triggers the same response; showing the hand without pokes extinguishes it.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | `BehaviorSelector`: `cooldown` takes one value per behaviour (a shorter list repeats its last value), so a reaction such as fear can repeat quickly while other behaviours keep their cooldown. | Done | Build without warnings. The five existing BehaviorSelector tests pass unchanged (a single cooldown value works as before). New `test_BehaviorSelector_cooldowns.ikg`: with `cooldown="10, 0.3"` and requests for both behaviours 0.4 s apart, the first starts only on the first request and the second on all three. | This commit |
| 2 | New `eek` sound in `render_sounds.py`: a sudden shriek and a trembling whimper, three variants. | Done | Rendered with the existing sounds venv; only `eek` was rendered, with the next free seed number (9), so the other sounds are unchanged. The variants are 0.82, 0.89, and 0.96 s long at 24 kHz, -16.1 to -17.4 dB RMS over the sounding part (the others are about -17 dB), peak 0.32 to 0.37, and start and end at zero. Not listened to; the user judges the sound. | This commit |
| 3 | `scared` animation (red-white flashes, trembling pale eyes, red mouth, flinch, tremble, the `eek`) as behaviour 11, requested when the brain's fear (`Brain.Fear`, the larger of the learned response and the poke) rises above a threshold; guide and dashboard. Test acquisition, response to the hand alone, and extinction with an image sequence. | Done | Simulated demo copy reading a 1330-frame sequence without the robot (a far hand that counts as seen but not close, an open palm filling the view as the poke, mid-grey rest frames): no scare at rest or for the hand before learning; the first two pokes scare Epi (fear from the poke; learned response 0.26, then 0.51); before the third poke the hand alone scares it (learned response rising past 0.4), and the poke that follows does not scare it a second time; six later appearances of the hand alone, 3 s each with 2 s breaks, scare it on the first four (learned response at their end 0.61, 0.52, 0.45, 0.38) and not on the last two or a final appearance (0.33, 0.28, 0.25). Every scare started one `eek`. The demo loads and runs without warnings. Not checked on the robot or by ear. | This commit |

### Outstanding issues and questions

- The fear response, the `eek` sound, and the conditioning have not been tried on the robot; the hand detection of a real pointing hand and the timing of the extinction need the user's check (step 13 in the guide).
- A poke that follows the hand within the scared animation does not start a second scare, because the fear stays above the threshold; this keeps the response from stuttering.

## A red card as the feared cue, and a closer poke

On the robot Epi was scared all the time: hands are always in view, so a learned fear of the hand fired constantly. The user chose a coloured card as the cue and asked for the poke to count only closer to the cameras.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Detect a red card in the right camera (half-resolution image, `ColorMatch` on red, fraction of strongly red pixels, seen above 2 percent of the view) as the amygdala's conditioned stimulus instead of a hand; a poke counts only when the finger is very close (`FingerNear.NEAR` above 0.85, about 0.15 m) while the eyes still start to cross at 0.3 m. Guide, brain view label, and a test of acquisition, response to the card alone, extinction, and that a hand or skin does not count as the card. | Done | Card detection with photos and synthetic red cards (160 x 110 pixels, 5.7 percent of the view): a card on grey 5.7 percent, a card next to an open palm 5.4, a card with a far hand 5.7; without a card an open palm 0.02 percent, the reddest palm photo 1.2, two faces and the placeholder image 0, all below the 2 percent mark. (A stricter setting cut the reddest palm to 0.6 percent but would likely miss a real, less saturated card.) Full demo copy, 1565 frames, without the robot: no scare for the card before learning or for a hand at about 0.26 m (closeness 0.14, no poke); pokes with the card scare Epi and raise the learned response to 0.18, 0.37, and 0.51; the card coming into view before the third poke scares Epi; a hand without the card does not; a poke without the card scares Epi without learning; the card alone scares Epi three times (0.48, 0.41, 0.35) and then no longer (0.30 to 0.21). Brain widget test passes. Not checked with the real camera, a real card, or by ear. | This commit |

### Outstanding issues and questions

- The card's colour through the real eye camera decides the detection; the settings and the dashboard value **Red in view** are described in the guide for tuning on the robot. Red objects in the room count as the card.
- Learning is a little slower than with the hand cue, because the poke now lasts only while the finger is very close; three pokes bring the learned response to about 0.5.


## A toy corn as the feared cue, and a poke closer to the face

The red card was replaced by something the audience is unlikely to show: the user's plastic toy corn, a saturated yellow cob with a green base. The left eye camera shows its yellow clearly, the right camera's washed-out colours mix it with skin and a paper bag. The user also found that a poke counted while the finger was still too far away.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Detect the corn in the left camera as the amygdala's conditioned stimulus: a yellow fraction and a green fraction must both be above their thresholds in the same frame. Tune the thresholds with the real corn; update the guide, the brain view label, and the dashboard. | Done | Colours measured through the live left eye camera: yellow 0.54, 0.42, 0.04 and green 0.10, 0.72, 0.18 as red, green, and blue fractions (the right camera sees the yellow as 0.50, 0.36, 0.14, close to skin and a paper bag). A camera-only probe sampled the left camera at 5 Hz for about two minutes while the user held the corn at arm's length, at about 30 cm, turned, and out of view: without the corn, yellow stayed at 0 (largest 0.0002) while the green of the user's checked shirt reached 0.0024, so green alone would not do. With yellow above 0.001 and green above 0.0002 the corn was seen in nearly every sample while it was held up, with short gaps while it was turned, and in none of the 214 samples without it. The extracted cue chain on saved frames: corn in the left eye 0.0046 yellow, 0.0010 green, seen; left eye without the corn 0, 0, not seen; right eye with the corn 0, 0.0008, not seen. A simulated demo copy on still images, without LEDs and sound, loads and runs without warnings and gives the corn to `Brain.Amygdala.CS`. The brain view's label (V1, sees the cue) was already generic. Not yet checked in the running demo, with pokes, or with other people in view. | This commit |
| 2 | A poke counts only when the finger is closer to the face than now. | Done | Measured in the running demo with a ruler. A pointing finger counted as a poke only at about 5 cm already, but an open hand counted from 20 cm, the case the user had seen, because an open hand looks about twice as wide. The hand's distance estimate hardly changes this close (0.19 m at 10 cm, 0.16 m at 5 cm), and `Hands.POINTING` was 0 for most pointing fingers, so neither can decide. Widths before the change (width, `OPEN`, poke share): pointing 10 cm 0.38-0.41, 0.75, 0 %; pointing 5 cm 0.43-0.47, 0.15, 60 %; open 20 cm 0.51, 1.0, 93 %; open 10 cm 0.75-0.93, 1.0, 87 %; open 5 cm 0.95-1.73, 0.01, 73 %. Now a hand that is not open counts from a width of 0.42 and any hand from 1.0, replacing `FingerNear.NEAR` above 0.85; the eyes still cross as before. The rule on these values with constant inputs: pointing at 10 cm and every open hand up to 0.93 no poke; pointing at 0.45 and hands from 1.05 or with cut-off fingers a poke. After the user restarted the demo: pointing 10 cm 0.33-0.34, 0 %; pointing 5 cm 0.47-0.53, 83 %; open 20 cm 0.41-0.43, 0 %; open 10 cm 0.82-0.94, 0 %; and the corn at arm's length seen in 30 of 30 samples (task 1). Not tried: a full conditioning run with the corn and pokes, other people's hands, and the sound and lights of the scare. | This commit |

### Outstanding issues and questions

- The open hand at 10 cm reached a width of 0.94, close to the 1.0 at which any hand counts; a larger hand may count from about 10 cm. Raise `VeryClose.threshold` if that happens.
- The widths were measured with one person's hand. A half-open hand counts as a finger, and a pointing finger at 10 cm was found in only part of the frames before the restart.
- The thresholds of the corn were set in one room; other light can change its colours. Something yellow and green together counts as the corn, and the corn turned so that its green base is hidden does not.
- The full conditioning sequence (corn with pokes, the corn alone, extinction) has not been run on the robot with the new cue and poke.

## Face tracking with several people, and servo read errors

With several people at about the same distance, Epi switches back and forth between their faces. On another laptop, the head servos twice reported a `GroupSyncRead` error; it has not happened on this Mac.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | Stop the gaze from switching between faces of about the same size. | Done | A 165 s log of the running demo at about 7 Hz with two people: the followed face was about 0.11 wide and the other 0.16 to 0.22, at about the same distance, and every switch to the other face came after it had been 1.5 to 2 times as wide for more than 0.7 s, the selector's `switch_ratio` 1.5 and `switch_delay` 0.7 s. Replaying the logged face boxes through `FaceTargetSelector` with the demo's settings gave 21 target changes; with `switch_ratio` 2 to 3 and `switch_delay` 1.5 to 2 s it gave 9, all after the followed face was lost or at the start, and partly artefacts of the 7 Hz sampling. Both selectors (right and left eye) now use 2.5 and 1.5 s; the guide explains the setting. Not yet checked live with two people. | This commit |
| 2 | Find the cause of the `GroupSyncRead` errors on the other laptop and make the servo communication report and recover from them. | Done | The other laptop could not be examined; the user saw the warning twice, together with pupil problems, and the robot kept running. A failed synchronous read already only skipped that tick's head commands, but the warning said only "GroupSyncRead failed: " without the bus result, followed by "Can not communicate with head" on every failing tick. Each chain (head, arms, body) now records why its tick failed, with the SDK's result text (for example a timeout or a garbled reply), and `Tick` reports it once when failures start, about once a second while they last with the number of failed ticks in a row, and once when communication is restored. The per-tick writes to the eye servos on the head bus (torque, goal position, PWM) used the SDK helpers with the 11-byte stack buffer that crashed Ikaros on the pupil bus; they now use the same full-size buffers, one set per chain because the chains run in parallel threads. Build without warnings. Against the installed SDK through a pseudo-terminal, the new write sends the same bytes as `write1ByteTxRx`, `write2ByteTxRx`, and `write4ByteTxRx`, including a value that needs byte stuffing, and reads the reply. On EpiBlack after the restart, 15 s of normal running: the eyes stayed within 0.5 to 0.7 degrees and the neck within 0.3 to 0.8 degrees of their goals. The failure messages were not seen, because no failure occurred; the cause on the other laptop is still unknown, and the next warning will name it. | This commit |
| 3 | Epi is scared at random: find what triggers the scared animation when nobody pokes and the corn is not in view. | Done | Scared has weight 0 among the spontaneous behaviours; a 40 s log of the running demo showed no pokes, but `CornSeen` was 1 while the learned fear stood at 0.45. A watcher saved a left-eye snapshot at each corn detection: in the room without the corn, a plant with a yellow stand and greenish leaves gave the yellow, and the green came from other places 60 or more pixels away; these false detections came when Epi looked around the room. The corn now counts only when yellow and green lie together in one patch: a new module `AveragePool` gives each colour's share in overlapping windows (24 x 24 pixels of the half-size image, 8 apart), and a patch needs more than 2 percent yellow and 1 percent green. Five chained `Downsample` modules could not be used, because 240 rows halve to 15 and `Downsample` needs even sizes, and non-overlapping cells split the corn across cell boundaries (21 of 42 corn frames). The demo's cue chain run in Ikaros on 42 snapshots of the corn held at different distances and angles: seen in 38 (missed when the green base was hidden or the corn was far away); on 38 snapshots the old rule had taken for the corn: seen in none. `AveragePool` tests: overlapping 2 x 2 windows with stride 1 and a two-channel image with 3 x 3 windows and stride 2 pass. A summed-area table, allocated in `Init`, keeps the pooling of a 240 x 320 map at no measurable cost in a 2000-tick Release benchmark (0.29 to 0.31 s with and without it; summing each window directly cost about 1 ms per tick). A simulated demo copy loads and runs without warnings. Not yet checked in the running demo. | This commit |

### Outstanding issues and questions

- Face switching with the new settings has not been checked live with two people. If Epi still switches, the remaining switches come from the followed face being lost for 0.5 s, mostly when the eyes move fast; `FaceTargetSelector` could then compensate for the camera's own movement.
- The corn patch rule was checked on snapshots and in a simulated demo, not yet for a whole session; a learned fear from earlier false detections does not carry over, because the amygdala starts at zero.
- The cause of the `GroupSyncRead` errors on the other laptop is unknown; the new warning names the bus result. Run the demo there and note the message.
- The start-up settings of `EpiServos` still use the SDK's write helpers with the 11-byte buffer; a garbled reply during start-up could still crash Ikaros.
- Task 3 was done before task 2 because the random scares disturbed the demo.

## Faster fear learning, and attention in a crowd

The user wants the corn conditioned by two pokes and extinguished after two to three appearances without a poke. With many people in the room, Epi should remember where the faces are, look from person to person at a natural pace instead of staying on the closest face, check new faces, and play *found you* only for a new face or one it has not looked at for a while.

| # | Task | Status | Verification | Commit |
|---:|---|---|---|---|
| 1 | `Delta`: optional trial-based learning (one step per US onset, one extinction step per CS presentation without a US, with a gap that bridges detection drop-outs); the demo's amygdala learns from two pokes and loses the fear after two to three appearances of the corn alone. Test and guide. | Done | Build without warnings. New `test_Delta_trials.ikg` (alpha 0.35, extinction 0.2, gap 0.25 s at 0.1 s ticks): a poke with a one-tick dropout counts once (0.35), a second poke after 0.3 s without one counts again (0.5775), a one-tick CS dropout neither splits the presentation nor drops `CR`, a presentation with pokes ends without extinction, three presentations alone give 0.5775, 0.462, 0.3696 and leave 0.29568, and a US without the CS changes nothing; the existing extinction test and kernel tests 167 and 169 (Delta state) pass, so per-tick learning is unchanged. The demo's fear chain (`Amygdala` with the demo settings, `Fear`, `FearCue`, and the demo's `BehaviorSelector` cooldowns) at 25 Hz with a scripted corn and poke: two 0.6 s pokes with a 0.08 s dropout each, while the corn is in view with a 0.3 s dropout, scared Epi twice and gave 0.35 and 0.58; the corn alone for 3 s, four times with 3 s breaks, scared Epi on the first two (0.58, 0.46) and not on the last two (0.37, 0.30). The simulated demo loads and runs without new warnings. Not tried on the robot. | This commit |
| 2 | Face memory and attention: a new module `FaceMemory` remembers the faces around Epi in body directions and chooses whom to look at next; `FaceTargetSelector` takes a preferred face, which also keeps the left eye's target on the right eye's face; `GazeLock` reports the camera direction at image capture; *found you* only for a new face or one not looked at for a minute. Tests, demo, and guide. | Done | Build without warnings. New tests: `test_GazeLock_camera.ikg` (`CAMERA_DIRECTION` is the tilt and the head pan plus the camera eye's pan from `latency` earlier), `test_FaceTargetSelector_preferred.ikg` (a request on a repeated frame waits for the next new frame and turns to the smaller face; a request with no face nearby or a flag of 0.4 changes nothing), `test_FaceMemory_attention.ikg` (with the real selector in the loop: the first face is new; after `min_dwell` Epi turns to a face never looked at, which is new; after the dwell back to the known face, which is not; nothing while `BUSY`; a face seen for less than `min_seen` is not chosen), and `test_FaceMemory_memory.ikg` (a face that leaves the view while the camera turns 20 degrees and comes back at another image position keeps its memory and is not new; it is new again after `regreet_time`; a face seen only during a 100 degrees per second jump is not remembered; an unseen face is forgotten after `forget_time`). All tests of FaceMemory, FaceTargetSelector, GazeLock, Delta, HeadMirror, ProximityVergence, Arousal, Blinker, BehaviorSelector, and StereoDistance pass (34). A scripted crowd of seven people (six standing and swaying, one walking across the group) over 120 s, with a camera that looks from person to person every 3 to 8 s at up to 300 degrees per second, a reported camera direction one tick behind the image with 0.5 degree noise, 15 percent missed detections, and a detector at half the tick rate: in eight runs every standing person got `NEW_FACE` when first looked at, the walker was missed in four runs, one person was greeted a second time in four runs (the two traced came from the walker passing them, which swaps or leaves entries at that place), and 7 to 8 faces were remembered at the end. On the way there, matching by detection order, updating a face from a frame at the start of a turn, and merging nearby faces each gave 4 to 30 extra greetings in these runs; the final matching is closest-pairs-first with recently seen faces first, half-step updates, one-frame faces dropped, and the judgement made once a look has lasted `min_seen` with the camera settled. The 165 s log of two people from the last session replayed through `FaceMemory` and the selector with the demo settings: a turn to the other person every 3 to 15 s while both were in view (6 to 15 turns depending on the version, against 21 changes with the old settings); its greeting count cannot be judged because the log has no camera directions. The simulated demo, with the face detector replaced by that log, loads and runs without new warnings: each `PREFERRED` turns the selector on the same tick, `NEW_FACE` starts *found you* (behaviour 0, at most every 10 s) and raises arousal, and each turn triggers a blink. `EpiFaceFollowing.ikg` and `EpiHeadMirroring.ikg` still load and run in simulation. Not tried on the robot, with real people, or in the WebUI. | This commit |

### Outstanding issues and questions

- Neither task has been tried on the robot. The fear chain needs a real session with the corn and pokes (guide step 13), and the attention a group of people (guide step 14).
- `FaceMemory` relies on `GazeLock.CAMERA_DIRECTION` lining up with the image; if the real camera latency differs much from `GazeLock.latency` (0.1 s), faces are placed in wrong directions and people are greeted again. The guide says what to watch (**Faces remembered** much higher than the number of people) and what to change.
- Faces are known only by where they are: people who cross paths can swap places in memory, and a newcomer standing where someone else stood within a minute is not greeted. `DlibFaceRecognizer` exists in the repository and could tell people apart by their faces, but it was not used, to keep the change simple.
- Epi remembers people out of view but does not turn toward them; with the 109 degree camera most of a group is in view at once. Turning to remembered faces now and then would need a way to give `GazeLock` a body direction to look at.
- The new dashboard tables (**Faces remembered**, **On this face for**) and the component positions of `Memory` and `BlinkTrigger` were placed by the layout rules but not checked in a browser; please look at the WebUI page.
- The *scared* animation's flinch can take the corn out of view for more than 1 s; its return then counts as a new appearance of the corn and uses up one of the two or three extinction steps.
- A switch to another person changes `Target.TARGET_POSE` at once, which counts as one swing in `HeadPan`; copycat needs three swings within 3 s and switches are at least 3 s apart, so it should not start by itself, but this has not been tried with people.

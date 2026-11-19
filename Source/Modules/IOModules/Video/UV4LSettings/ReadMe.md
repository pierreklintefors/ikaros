# UV4LSettings

## Description

`UV4LSettings` sets the resolution, frame rate, and JPEG quality of a UV4L camera server when a model
starts. UV4L serves the Raspberry Pi camera in each Epi eye. The server forgets these settings when
the Pi restarts, and it cannot change the resolution while a client is streaming, so they must be
set before the stream is opened.

The module sends the settings while the model is built, before any module's `Init()` runs, so it
does not matter where it appears in the model or whether its name sorts before the camera module.
`url` can be the same stream address that `InputVideoStream` reads; only the host and port are used.
The settings go to the server's `/panel` page, which changes only the values that are given. The
module then reads the page back and sets `APPLIED` to 1 when the server reports the requested width,
height, frame rate, and quality.

When the server cannot be reached or reports other values, the module warns and the model continues,
or with `required="true"` the model stops at start-up. A `url` that is not an `http://` address, such
as a video file, is skipped without a warning, so a model can switch between a camera and a recording
by changing one setting.

`extra` passes further settings to the panel as `name=value` pairs separated by `&`, using the names
on the server's `/panel` page.

## Example

```xml
<module class="UV4LSettings" name="CameraSettings" url="@eye_url" width="640" height="480" frame_rate="30" quality="70" />
<module class="InputVideoStream" name="Eye" url="@eye_url" size_x="640" size_y="480" />
```

## Notes

- The frame rate and quality are sent as the raspicam driver's control ids, `134217741` and
  `134217739`. Another UV4L driver may use other ids; check its `/panel` page.
- Start-up waits up to `timeout` seconds per request. An unknown host name can take a few seconds
  more to fail.

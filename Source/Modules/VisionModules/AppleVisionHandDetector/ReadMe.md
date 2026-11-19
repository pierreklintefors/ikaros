# AppleVisionHandDetector

## Description

`AppleVisionHandDetector` finds hands and their finger joints with the hand pose detection in Apple's
Vision framework (macOS only). When several hands are seen, it chooses the largest, which is usually
the one nearest the camera.

`TARGET` has the same form as `FaceTargetSelector.TARGET`: the x and y of one point of the hand and the
hand's width, in centered image coordinates from -1 to 1 with y downward. A model can therefore follow
a hand, or measure its distance, with the modules it uses for faces, such as `GazeLock`,
`StereoDistance`, and `ProximityVergence`. `point` chooses the point: the index fingertip (default) or
the middle of the palm, which is the average of the wrist and the four knuckles.

`OPEN` is the fraction of the four fingers, not counting the thumb, that are stretched out. A finger
counts as stretched out when its tip is more than `extension_ratio` times as far from the wrist as its
middle joint. An open palm gives 1 and a fist 0. `POINTING` is 1 when only the index finger is
stretched out.

`min_width` ignores hands that look narrower than the given width, such as the hands of people
farther away, so that only a hand held up close to the camera counts.

Vision sometimes misses a hand for a frame. The last hand is then kept for `hold_time` seconds before
`VALID` becomes 0 and the outputs are cleared.

Hand detection costs about as much as face detection, so it can be made cheaper. `downsample`
halves the image before detection, `interval` detects only on every second or third tick and keeps
the last result in between, and `ENABLE` at 0 stops detection altogether, for example in a second
camera that is only needed during a game.

## Example

An open palm held up for two seconds, measured with `HoldLatch`:

```xml
<module class="AppleVisionHandDetector" name="Hands" />
<module class="HoldLatch" name="Palm" hold_time="2" release_time="3" />
<connection source="Camera.OUTPUT" target="Hands.INPUT" />
<connection source="Hands.OPEN" target="Palm.INPUT" />
<connection source="Hands.VALID" target="Palm.PRESENT" />
```

## Notes

- The hand width is the horizontal extent of the joints that were found, so it changes when the
  fingers spread or close.
- Joints with a confidence below `min_confidence` are ignored.

# HoldLatch

## Description

`HoldLatch` turns a signal that is held for a while into a mode that lasts until the person or thing
behind it has gone. A typical use is a gesture: an open palm held up for two seconds starts a game,
and the game ends when the hand has been out of view for three seconds.

- `ACTIVE` becomes 1 when `INPUT` has been above `threshold` for `hold_time` seconds without a break.
  A shorter hold starts nothing, and the time starts again from zero.
- While active, the mode continues as long as `PRESENT` is above `threshold`. When `PRESENT` has been
  at or below it for `release_time` seconds, `ACTIVE` returns to 0. A shorter absence is forgiven.
  Without `PRESENT`, `INPUT` itself keeps the mode going.
- `START` and `END` are 1 for one tick when the mode begins and ends, for example to request an
  animation or a sound.
- `PROGRESS` rises from 0 to 1 during the hold and stays 1 while active, so a dashboard bar or a light
  can show the person that the gesture is being noticed.

## Example

```xml
<module class="AppleVisionHandDetector" name="Hands" />
<module class="HoldLatch" name="Palm" hold_time="2" release_time="3" />
<connection source="Hands.OPEN" target="Palm.INPUT" />
<connection source="Hands.VALID" target="Palm.PRESENT" />
```

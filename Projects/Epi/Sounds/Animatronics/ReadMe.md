# Animatronics sounds

Non-verbal robot sounds for `Projects/Epi/EpiAnimatronics.ikg`, played by `SoundOutput`. Epi does not
speak. It chirps and whistles, and its feelings come across through intonation: how the pitch
moves, how fast, and how each sound starts and ends.

| File | Used for | Sound |
| --- | --- | --- |
| `hi.wav` | greet | three bouncy chirps that end high and open |
| `wow.wav` | surprise in mind blown | a long whistle that starts low and climbs while its vibrato widens, then a bright "eep" |
| `poof.wav` | mind blown | a sputtering whistle that powers down inside a soft puff of air and a thump |
| `giggle.wav` | found you, giggle | eight quick chirps that hop between high and low, speed up, and drift down |
| `hehe.wav` | mischief | a sly, low "he-he-heh" of three clipped chirps that slide down, and a sneaky rise at the end |
| `hmm.wav` | cross-eyed and curious | a thoughtful hum, then a chirp that rises like a question |
| `aww.wav` | shy | a soft, slow whistle that lifts a little and sinks away |
| `sparkle.wav` | found you | bright blips that climb into a happy trill |

## Design

The sounds follow how film robots are made to sound alive without words:

- Ben Burtt found that synthesizer sounds alone were too electronic to express emotion. R2-D2's
  voice came alive when the chirps followed the inflections of a human voice, including baby-talk
  wails. The sounds here are original, but they follow the same idea: each syllable has the
  intonation of a spoken word, without the word.
- An analysis of WALL-E's sounds found that intonation, pitch, and timbre carry most of the
  emotion and intention. Pitch movement is therefore the main tool here.
- Listeners recognize basic emotions such as happy, sad, surprised, and tired from such sounds, but
  rarely subtle ones, so each sound is one clear gesture.

The rules used:

| Feeling | Pitch | Attack and decay |
| --- | --- | --- |
| Greeting, joy | bounces and ends high | quick attacks, short decays, a trill at the end |
| Surprise | starts low and climbs, vibrato widening | slow swell, then a sharp "eep" |
| Question, curiosity | level, then rising at the end | soft hum, then a clear chirp |
| Laughter | hops quickly between high and low, speeding up | very short chirps |
| Mischief | low, each syllable sliding down, a sneaky rise at the end | clipped, nasal chirps |
| Shyness | lifts slightly, then sinks | slow, soft attack and long fade |
| Mind blown | falls from high to low with a stuttering warble | sudden start, long power-down |

Each syllable is a whistle with a few overtones, like a resonant filter, with a smooth pitch
contour, vibrato, and optional sample-and-hold warble. Its phase is continuous, so pitch glides
are smooth. A small, soft room is added, each sound ends once its tail has died away, and all are
matched to -17 dB RMS over the sounding part, with the shy and sparkle sounds a little softer.
They start and end at zero. `SoundOutput` plays one sound at a time, so short endings keep the next
sound on time. In the demo, `Arousal` sets the playback volume through `SoundOutput.GAIN`, so the
same sound is softer when Epi is calm and louder when it is excited.

The giggle and "he-he-heh" onsets are listed in `render_sounds.py`; `EpiAnimations.json` flashes the
mouth and bobs the head on those beats, so motion and sound land together.

## Rendering again

`render_sounds.py` needs only numpy:

```sh
python3 -m venv UserData/venvs/sounds
UserData/venvs/sounds/bin/pip install numpy
UserData/venvs/sounds/bin/python Projects/Epi/Sounds/Animatronics/render_sounds.py
```

Each sound is a short function at the end of the script that lists its syllables with their
timing, pitch points, attack, decay, vibrato, warble, and brightness. `--only` renders some sounds,
and `--seed` changes the warbles. To compare takes, render variants into
`UserData/output/animatronics_sounds` and listen with `afplay`:

```sh
UserData/venvs/sounds/bin/python Projects/Epi/Sounds/Animatronics/render_sounds.py --variants 4
afplay UserData/output/animatronics_sounds/poof_seed2.wav
```

## Sources

- [Ben Burtt on R2-D2's voice](https://designingsound.org/2009/09/10/ben-burtt-special-star-wars-the-sounds-part-2)
- [The ARP 2600 behind R2-D2](https://www.musicradar.com/how-to/the-40-greatest-synth-sounds-of-all-time-no-22-star-wars-r2-d2)
- [Ben Burtt and the voice of WALL-E](https://news.symbolicsound.com/?p=4035)
- [Analysis of WALL-E's sounds](https://www.smc2019.uma.es/articles/P2/P2_07_SMC2019_paper.pdf)
- [Teaching robots to communicate like R2-D2](https://spectrum.ieee.org/beep-boop-teaching-robots-to-communicate-like-r2d2)

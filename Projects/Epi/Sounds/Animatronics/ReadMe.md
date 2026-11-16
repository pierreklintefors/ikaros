# Animatronics sounds

Short sounds for `Projects/Epi/EpiAnimatronics.ikg`, played by `SoundOutput`.

| File | Sound | Made with |
| --- | --- | --- |
| `hi.wav` | "Hi!" | Kokoro TTS |
| `wow.wav` | a slow "Wow!" | Kokoro TTS |
| `giggle.wav` | a giggle of five "hee" syllables that speed up and fall | Kokoro TTS |
| `hmm.wav` | a curious "Hmm?" | Kokoro TTS |
| `aww.wav` | a shy "Aww." | Kokoro TTS |
| `poof.wav` | a pooof: a low thump under a breath of noise that darkens as it fades | Ikaros synthesizer, `RenderPoof.ikg` |
| `sparkle.wav` | three quick rising bell notes | Ikaros synthesizer, `RenderSparkle.ikg` |

The voice clips use the Kokoro voice `af_heart` at its natural pitch. Raising the pitch by
resampling makes a voice sound sped up, so `render_voice.py` allows only small steps and uses none
by default. Each clip keeps the natural decay of the voice, gets gentle fades and a small, soft
room, ends once that tail has died away, and is matched to the same loudness (-17 dB RMS over the
voiced part, peaks below -3 dBFS). The giggle is built from single "hee" syllables that come
faster, lower, and softer, which sounds more like laughing than a text read aloud.

The files are committed, so the demo needs neither Kokoro nor Python.

## Rendering again

The effects need only Ikaros. `OutputAudioFile` writes only inside the user directory, so set it to
this folder:

```sh
./Bin/ikaros -b -u Projects/Epi/Sounds/Animatronics Projects/Epi/Sounds/Animatronics/RenderPoof.ikg
./Bin/ikaros -b -u Projects/Epi/Sounds/Animatronics Projects/Epi/Sounds/Animatronics/RenderSparkle.ikg
```

The voice clips need Kokoro in its own virtual environment. The environment and the model files are
kept in `UserData`, which is not committed:

```sh
python3 -m venv UserData/venvs/kokoro
UserData/venvs/kokoro/bin/pip install kokoro-onnx soundfile
mkdir -p UserData/models/kokoro
curl -L -o UserData/models/kokoro/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o UserData/models/kokoro/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
UserData/venvs/kokoro/bin/python Projects/Epi/Sounds/Animatronics/render_voice.py
```

`render_voice.py` takes `--voice` (for example `af_bella`, `af_sky`, or `af_nicole`), `--pitch` in
semitones, and `--only` to render some clips. To compare voices, render every clip with six voices
into `UserData/output/animatronics_voices/<voice>/` and listen with `afplay`:

```sh
UserData/venvs/kokoro/bin/python Projects/Epi/Sounds/Animatronics/render_voice.py --audition
afplay UserData/output/animatronics_voices/af_bella/giggle.wav
```

Then render the chosen voice into this folder with `--voice`. The texts and speeds are listed at the top of the
script. Spellings with stretched vowels, such as "Wooow", are read as separate syllables, so a long
vowel is made with a lower speed instead.

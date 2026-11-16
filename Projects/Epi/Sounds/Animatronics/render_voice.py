"""Render the short voice clips for EpiAnimatronics with Kokoro TTS.

Kokoro runs only here, offline; Ikaros plays the resulting WAV files with SoundOutput. Set up the
environment once from the repository root:

    python3 -m venv UserData/venvs/kokoro
    UserData/venvs/kokoro/bin/pip install kokoro-onnx soundfile
    mkdir -p UserData/models/kokoro
    (download kokoro-v1.0.onnx and voices-v1.0.bin from the kokoro-onnx model-files-v1.0 release
     into UserData/models/kokoro)

Then render:

    UserData/venvs/kokoro/bin/python Projects/Epi/Sounds/Animatronics/render_voice.py
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile
from kokoro_onnx import Kokoro

# File name, text, speed. Short, simple words come out cleanest; stretched spellings such as
# "Wooow" are read as separate syllables, so a long vowel is made with a low speed instead.
CLIPS = [
    ("hi", "Hi!", 1.0),
    ("wow", "Wow!", 0.55),
    ("giggle", "Hee hee hee hee!", 1.0),
    ("hmm", "Hmm?", 0.9),
    ("aww", "Aww.", 0.9),
]


def trim(samples, threshold=0.01, pad=0.03, rate=24000):
    """Removes leading and trailing silence, keeping a short pad."""
    loud = np.flatnonzero(np.abs(samples) > threshold)
    if loud.size == 0:
        return samples
    margin = int(pad * rate)
    return samples[max(loud[0] - margin, 0):loud[-1] + margin]


def pitch_up(samples, semitones):
    """Raises pitch and tempo together by resampling, which gives a small, cute voice."""
    if semitones == 0:
        return samples
    factor = 2.0 ** (semitones / 12.0)
    positions = np.arange(0, len(samples) - 1, factor)
    return np.interp(positions, np.arange(len(samples)), samples)


def finish(samples, rate, peak=0.89):
    """Fades the ends to avoid clicks and normalizes the peak to about -1 dBFS."""
    fade = min(int(0.01 * rate), len(samples) // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        samples[:fade] *= ramp
        samples[-fade:] *= ramp[::-1]
    largest = np.max(np.abs(samples))
    return samples * (peak / largest) if largest > 0 else samples


def main():
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=root / "UserData" / "models" / "kokoro")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice, for example af_heart, af_bella, af_sky, or af_nicole")
    parser.add_argument("--pitch", type=float, default=4.0, help="Semitones to raise the voice")
    parser.add_argument("--only", nargs="*", help="Render only these clip names")
    args = parser.parse_args()

    kokoro = Kokoro(str(args.model_dir / "kokoro-v1.0.onnx"), str(args.model_dir / "voices-v1.0.bin"))
    for name, text, speed in CLIPS:
        if args.only and name not in args.only:
            continue
        samples, rate = kokoro.create(text, voice=args.voice, speed=speed, lang="en-us")
        samples = finish(pitch_up(trim(np.asarray(samples, dtype=np.float64), rate=rate), args.pitch), rate)
        path = args.out_dir / f"{name}.wav"
        soundfile.write(path, samples.astype(np.float32), rate, subtype="PCM_16")
        print(f"{path.name}: {len(samples) / rate:.2f} s")


if __name__ == "__main__":
    main()

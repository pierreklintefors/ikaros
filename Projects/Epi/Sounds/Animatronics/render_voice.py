"""Render the short voice clips for EpiAnimatronics with Kokoro TTS.

Kokoro runs only here, offline; Ikaros plays the resulting WAV files with SoundOutput. Set up the
environment once from the repository root:

    python3 -m venv UserData/venvs/kokoro
    UserData/venvs/kokoro/bin/pip install kokoro-onnx soundfile
    mkdir -p UserData/models/kokoro
    (download kokoro-v1.0.onnx and voices-v1.0.bin from the kokoro-onnx model-files-v1.0 release
     into UserData/models/kokoro)

Then render the clips used by the demo:

    UserData/venvs/kokoro/bin/python Projects/Epi/Sounds/Animatronics/render_voice.py

or render every clip with several voices into UserData/output/animatronics_voices to compare them:

    UserData/venvs/kokoro/bin/python Projects/Epi/Sounds/Animatronics/render_voice.py --audition
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile
from kokoro_onnx import Kokoro

# File name, text, speed. Short, simple words come out cleanest; stretched spellings such as
# "Wooow" are read as separate syllables, so a long vowel is made with a lower speed instead.
CLIPS = [
    ("hi", "Hi!", 1.0),
    ("wow", "Wow!", 0.8),
    ("hmm", "Hmm?", 0.9),
    ("aww", "Aww.", 0.9),
]

# The giggle is built from single "hee" syllables that speed up and fall in pitch and loudness,
# which sounds more like laughing than a text read aloud: onset in seconds, semitones, gain.
GIGGLE_TEXT = "Hee!"
GIGGLE = [(0.00, 2.0, 1.00), (0.15, 1.5, 0.90), (0.28, 1.0, 0.82), (0.39, 0.5, 0.72), (0.49, 0.0, 0.60)]

AUDITION_VOICES = ["af_heart", "af_bella", "af_nicole", "af_sky", "af_jessica", "bf_lily"]

TARGET_RMS_DB = -17.0  # loudness of the voiced part, so all clips sound equally loud
PEAK_LIMIT = 0.7       # about -3 dBFS, leaving headroom for the speaker


def db_to_gain(db):
    return 10.0 ** (db / 20.0)


def raised_cosine(n):
    return 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n)) if n > 0 else np.zeros(0)


def trim(samples, rate, lead=0.06, tail=0.15):
    """Removes silence but keeps a short lead and the full natural decay of the voice."""
    level = np.abs(samples)
    peak = level.max()
    if peak <= 0:
        return samples
    start = np.flatnonzero(level > peak * db_to_gain(-40))[0]
    end = np.flatnonzero(level > peak * db_to_gain(-55))[-1]
    return samples[max(start - int(lead * rate), 0):min(end + int(tail * rate), len(samples))]


def fade(samples, rate, fade_in=0.015, fade_out=0.12):
    """Raised-cosine fades, so no clip starts or stops with a click."""
    samples = samples.copy()
    n_in = min(int(fade_in * rate), len(samples) // 2)
    n_out = min(int(fade_out * rate), len(samples) // 2)
    samples[:n_in] *= raised_cosine(n_in)
    samples[len(samples) - n_out:] *= raised_cosine(n_out)[::-1]
    return samples


def pitch_shift(samples, semitones):
    """Raises the pitch by resampling; only small steps are used, which still sound natural."""
    if semitones == 0:
        return samples
    factor = 2.0 ** (semitones / 12.0)
    positions = np.arange(0, len(samples) - 1, factor)
    return np.interp(positions, np.arange(len(samples)), samples)


def room(samples, rate, decay=0.35, wet=0.12, pre_delay=0.012, seed=3):
    """Adds a small, soft room: noise with an exponential decay and dull high frequencies."""
    rng = np.random.default_rng(seed)
    length = int(decay * rate)
    t = np.arange(length) / rate
    response = rng.standard_normal(length) * np.exp(-6.9 * t / decay)
    response = np.convolve(response, np.ones(8) / 8.0, mode="same")
    response /= np.sqrt(np.sum(response ** 2))
    tail = np.convolve(samples, response)
    delay = int(pre_delay * rate)
    out = np.zeros(len(tail) + delay)
    out[:len(samples)] += samples
    out[delay:] += wet * tail
    return out


def end_after_decay(samples, rate, floor_db=-50.0, fade_out=0.08):
    """Ends the clip once the room tail has died away, so the next sound is not held up."""
    envelope = np.convolve(np.abs(samples), np.ones(240) / 240, mode="same")
    end = np.flatnonzero(envelope > envelope.max() * db_to_gain(floor_db))[-1]
    return fade(samples[:min(end + int(fade_out * rate), len(samples))], rate, fade_in=0.0, fade_out=fade_out)


def level(samples):
    """Matches the loudness of the voiced part and keeps the peak below the limit."""
    voiced = samples[np.abs(samples) > np.abs(samples).max() * db_to_gain(-30)]
    rms = np.sqrt(np.mean(voiced ** 2))
    samples = samples * (db_to_gain(TARGET_RMS_DB) / rms)
    peak = np.abs(samples).max()
    return samples * (PEAK_LIMIT / peak) if peak > PEAK_LIMIT else samples


def speak(kokoro, text, voice, speed):
    samples, rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us", trim=False)
    return np.asarray(samples, dtype=np.float64), rate


def finish(samples, rate, pitch):
    samples = fade(trim(pitch_shift(samples, pitch), rate), rate)
    return level(end_after_decay(room(samples, rate), rate))


def giggle(kokoro, voice, pitch):
    syllable, rate = speak(kokoro, GIGGLE_TEXT, voice, 1.1)
    syllable = fade(trim(syllable, rate, lead=0.01, tail=0.06), rate, fade_in=0.008, fade_out=0.05)
    parts = [(onset, pitch_shift(syllable, pitch + semitones) * gain) for onset, semitones, gain in GIGGLE]
    out = np.zeros(int(max(onset * rate + len(part) for onset, part in parts)) + 1)
    for onset, part in parts:
        start = int(onset * rate)
        out[start:start + len(part)] += part
    return level(end_after_decay(room(out, rate), rate)), rate


def render(kokoro, voice, pitch, out_dir, only=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text, speed in CLIPS:
        if only and name not in only:
            continue
        samples, rate = speak(kokoro, text, voice, speed)
        write(out_dir / f"{name}.wav", finish(samples, rate, pitch), rate)
    if not only or "giggle" in only:
        samples, rate = giggle(kokoro, voice, pitch)
        write(out_dir / "giggle.wav", samples, rate)


def write(path, samples, rate):
    soundfile.write(path, samples.astype(np.float32), rate, subtype="PCM_16")
    print(f"{path}: {len(samples) / rate:.2f} s")


def main():
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=root / "UserData" / "models" / "kokoro")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice, for example af_heart, af_bella, af_sky, or af_nicole")
    parser.add_argument("--pitch", type=float, default=0.0, help="Semitones to raise the voice; more than about 2 sounds sped up")
    parser.add_argument("--only", nargs="*", help="Render only these clip names")
    parser.add_argument("--audition", action="store_true", help="Render every clip with several voices into UserData/output/animatronics_voices")
    args = parser.parse_args()

    kokoro = Kokoro(str(args.model_dir / "kokoro-v1.0.onnx"), str(args.model_dir / "voices-v1.0.bin"))
    if args.audition:
        for voice in AUDITION_VOICES:
            render(kokoro, voice, args.pitch, root / "UserData" / "output" / "animatronics_voices" / voice, args.only)
    else:
        render(kokoro, args.voice, args.pitch, args.out_dir, args.only)


if __name__ == "__main__":
    main()

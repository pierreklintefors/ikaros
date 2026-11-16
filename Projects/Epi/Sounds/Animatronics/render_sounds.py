"""Render the non-verbal robot sounds for EpiAnimatronics.

Every sound is a short phrase of chirps and whistles. Each syllable has its own pitch contour,
attack, decay, vibrato, and warble, so the phrase carries emotion through intonation alone: a
greeting bounces up, a question ends rising, surprise starts low and climbs, shyness sinks softly.

Needs only numpy. From the repository root, for example with the venv in UserData:

    python3 -m venv UserData/venvs/sounds
    UserData/venvs/sounds/bin/pip install numpy
    UserData/venvs/sounds/bin/python Projects/Epi/Sounds/Animatronics/render_sounds.py

The warbles are random but seeded, so the files are reproducible. To hear other takes, render
variants with other seeds into UserData/output/animatronics_sounds:

    UserData/venvs/sounds/bin/python Projects/Epi/Sounds/Animatronics/render_sounds.py --variants 4
"""

import argparse
import wave
from pathlib import Path

import numpy as np

RATE = 24000
TARGET_RMS_DB = -17.0  # loudness of the sounding part, so all sounds are equally loud
PEAK_LIMIT = 0.7       # about -3 dBFS, leaving headroom for the speaker


def db_to_gain(db):
    return 10.0 ** (db / 20.0)


def raised_cosine(n):
    return 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n)) if n > 0 else np.zeros(0)


def smooth_step(x):
    return x * x * (3.0 - 2.0 * x)


def pitch_curve(points, n):
    """Pitch in Hz from (time 0..1, Hz) points, eased smoothly between points on a log scale."""
    times = np.array([p[0] for p in points])
    logs = np.log([p[1] for p in points])
    t = np.linspace(0.0, 1.0, n)
    index = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)
    span = np.maximum(times[index + 1] - times[index], 1e-9)
    x = np.clip((t - times[index]) / span, 0.0, 1.0)
    return np.exp(logs[index] + smooth_step(x) * (logs[index + 1] - logs[index]))


def warble(n, rate, depth, rng, glide=0.008):
    """Sample-and-hold pitch steps in semitones with a short glide between them."""
    if rate <= 0 or depth <= 0:
        return np.zeros(n)
    hold = max(int(RATE / rate), 1)
    steps = rng.uniform(-depth, depth, n // hold + 2)
    stepped = np.repeat(steps, hold)[:n]
    alpha = 1.0 - np.exp(-1.0 / (glide * RATE))
    out = np.empty(n)
    value = stepped[0]
    for i in range(n):
        value += alpha * (stepped[i] - value)
        out[i] = value
    return out


def syllable(duration, pitch, attack=0.01, release=0.05, vibrato=(0.0, 0.0), vibrato_growth=0.0,
             warble_rate=0.0, warble_depth=0.0, brightness=0.25, gain=1.0, rng=None):
    """One chirp or whistle.

    pitch: (time 0..1, Hz) points. vibrato: (rate Hz, depth semitones); vibrato_growth > 0 lets the
    vibrato deepen toward the end. brightness 0 is a pure whistle; higher adds overtones, like a
    resonant filter that opens.
    """
    n = int(duration * RATE)
    t = np.arange(n) / RATE
    semitones = np.zeros(n)
    rate, depth = vibrato
    if rate > 0 and depth > 0:
        growth = 1.0 + vibrato_growth * (t / duration)
        semitones += depth * growth * np.sin(2 * np.pi * rate * t)
    semitones += warble(n, warble_rate, warble_depth, rng or np.random.default_rng(0))
    frequency = pitch_curve(pitch, n) * 2.0 ** (semitones / 12.0)
    phase = 2 * np.pi * np.cumsum(frequency) / RATE

    # A whistle with overtones that fade out before the Nyquist frequency.
    tone = np.sin(phase)
    for k, weight in ((2, 0.5), (3, 0.25), (4, 0.12)):
        audible = np.clip((0.45 * RATE - k * frequency) / (0.05 * RATE), 0.0, 1.0)
        tone += brightness * weight * audible * np.sin(k * phase)

    envelope = np.ones(n)
    a = min(int(attack * RATE), n // 2)
    r = min(int(release * RATE), n - a)
    envelope[:a] = raised_cosine(a)
    envelope[n - r:] = raised_cosine(r)[::-1] ** 2  # a decay that tails off softly
    return gain * tone * envelope


def puff(duration, rng, attack=0.02, cutoff_start=3000.0, cutoff_end=200.0, gain=1.0):
    """Breathy noise whose brightness and loudness fall together."""
    n = int(duration * RATE)
    noise = rng.standard_normal(n)
    cutoff = np.geomspace(cutoff_start, cutoff_end, n)
    out = np.empty(n)
    state = 0.0
    for i in range(n):
        state += (1.0 - np.exp(-2 * np.pi * cutoff[i] / RATE)) * (noise[i] - state)
        out[i] = state
    envelope = np.exp(-4.0 * np.arange(n) / n)
    a = int(attack * RATE)
    envelope[:a] *= raised_cosine(a)
    return gain * out * envelope


def phrase(parts):
    """Mixes (onset seconds, samples) parts."""
    length = max(int(onset * RATE) + len(samples) for onset, samples in parts)
    out = np.zeros(length + 1)
    for onset, samples in parts:
        start = int(onset * RATE)
        out[start:start + len(samples)] += samples
    return out


def room(samples, decay=0.3, wet=0.1, pre_delay=0.012, seed=3):
    """A small, soft room: noise with an exponential decay and dull high frequencies."""
    rng = np.random.default_rng(seed)
    length = int(decay * RATE)
    t = np.arange(length) / RATE
    response = rng.standard_normal(length) * np.exp(-6.9 * t / decay)
    response = np.convolve(response, np.ones(8) / 8.0, mode="same")
    response /= np.sqrt(np.sum(response ** 2))
    tail = np.convolve(samples, response)
    delay = int(pre_delay * RATE)
    out = np.zeros(len(tail) + delay)
    out[:len(samples)] += samples
    out[delay:] += wet * tail
    return out


def finish(samples, gain_db=0.0, floor_db=-50.0, fade_out=0.06):
    """Adds the room, ends the sound when the tail has died away, and matches the loudness."""
    samples = room(samples)
    envelope = np.convolve(np.abs(samples), np.ones(240) / 240, mode="same")
    end = np.flatnonzero(envelope > envelope.max() * db_to_gain(floor_db))[-1]
    samples = samples[:min(end + int(fade_out * RATE), len(samples))].copy()
    n = min(int(fade_out * RATE), len(samples) // 2)
    samples[len(samples) - n:] *= raised_cosine(n)[::-1]
    sounding = samples[np.abs(samples) > np.abs(samples).max() * db_to_gain(-30)]
    samples *= db_to_gain(TARGET_RMS_DB + gain_db) / np.sqrt(np.mean(sounding ** 2))
    peak = np.abs(samples).max()
    return samples * (PEAK_LIMIT / peak) if peak > PEAK_LIMIT else samples


# The sounds. File names match the events in EpiAnimations.json.

def hi(rng):
    """Greeting: three bouncy chirps that end high and open, like a cheerful hello."""
    return finish(phrase([
        (0.00, syllable(0.08, [(0, 900), (1, 1350)], attack=0.004, release=0.03, brightness=0.3, rng=rng)),
        (0.10, syllable(0.07, [(0, 1500), (1, 1150)], attack=0.004, release=0.03, brightness=0.3, rng=rng)),
        (0.19, syllable(0.20, [(0, 1050), (0.5, 1700), (1, 2050)], attack=0.006, release=0.08,
                        vibrato=(10, 0.35), brightness=0.25, rng=rng)),
    ]))


def wow(rng):
    """Surprise: a long whistle that starts low and climbs, its vibrato widening, then a bright 'eep'."""
    return finish(phrase([
        (0.00, syllable(0.62, [(0, 170), (0.2, 190), (0.8, 1000), (1, 1450)], attack=0.05, release=0.06,
                        vibrato=(7, 0.15), vibrato_growth=3.0, brightness=0.45, rng=rng)),
        (0.66, syllable(0.13, [(0, 1700), (1, 2200)], attack=0.004, release=0.06, brightness=0.3, rng=rng)),
    ]))


def poof(rng):
    """Mind blown: a sputtering whistle that powers down, inside a soft puff of air and a thump."""
    return finish(phrase([
        (0.00, syllable(0.85, [(0, 1600), (0.3, 750), (1, 110)], attack=0.005, release=0.3,
                        warble_rate=16, warble_depth=1.8, brightness=0.35, gain=0.8, rng=rng)),
        (0.00, puff(0.7, rng, gain=0.9)),
        (0.00, syllable(0.25, [(0, 90), (1, 60)], attack=0.004, release=0.2, brightness=0.0, gain=0.6, rng=rng)),
    ]))


def giggle(rng):
    """Laughter: quick chirps that hop between high and low, speed up, and drift down."""
    onsets = [0.0, 0.10, 0.185, 0.26, 0.325, 0.385, 0.44, 0.49]
    parts = []
    for i, onset in enumerate(onsets):
        base = 1500 * 2 ** (-i * 0.6 / 12) * (1.18 if i % 2 == 0 else 0.9)
        parts.append((onset, syllable(0.06, [(0, base), (1, base * 1.2)], attack=0.003, release=0.03,
                                      brightness=0.3, gain=1.0 - 0.05 * i, rng=rng)))
    return finish(phrase(parts))


def hmm(rng):
    """Curious question: a thoughtful hum, then a chirp that rises at the end."""
    return finish(phrase([
        (0.00, syllable(0.22, [(0, 620), (1, 600)], attack=0.04, release=0.06, vibrato=(5, 0.2),
                        brightness=0.15, gain=0.8, rng=rng)),
        (0.26, syllable(0.24, [(0, 640), (0.4, 700), (1, 1450)], attack=0.01, release=0.06,
                        brightness=0.25, rng=rng)),
    ]))


def aww(rng):
    """Shy: a soft, slow whistle that lifts a little and then sinks away."""
    return finish(phrase([
        (0.00, syllable(0.6, [(0, 1050), (0.3, 1200), (1, 600)], attack=0.09, release=0.25,
                        vibrato=(5, 0.25), brightness=0.05, rng=rng)),
    ]), gain_db=-3.0)


def sparkle(rng):
    """Joy: four bright blips that climb, the last one ringing with a happy trill."""
    notes = [1300, 1650, 2000]
    parts = [(0.06 * i, syllable(0.05, [(0, f), (1, f * 1.08)], attack=0.003, release=0.03,
                                 brightness=0.2, rng=rng)) for i, f in enumerate(notes)]
    parts.append((0.18, syllable(0.22, [(0, 2500), (1, 2700)], attack=0.003, release=0.12,
                                 vibrato=(14, 0.4), brightness=0.15, rng=rng)))
    return finish(phrase(parts), gain_db=-2.0)


SOUNDS = {"hi": hi, "wow": wow, "poof": poof, "giggle": giggle, "hmm": hmm, "aww": aww, "sparkle": sparkle}


def write(path, samples):
    data = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(data.tobytes())
    print(f"{path}: {len(samples) / RATE:.2f} s")


def main():
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--only", nargs="*", help="Render only these sounds")
    parser.add_argument("--seed", type=int, default=1, help="Seed for the warbles")
    parser.add_argument("--variants", type=int, default=0,
                        help="Render this many takes with other seeds into UserData/output/animatronics_sounds")
    args = parser.parse_args()

    names = args.only or list(SOUNDS)
    if args.variants > 0:
        out_dir = root / "UserData" / "output" / "animatronics_sounds"
        out_dir.mkdir(parents=True, exist_ok=True)
        for seed in range(args.seed, args.seed + args.variants):
            for name in names:
                write(out_dir / f"{name}_seed{seed}.wav", SOUNDS[name](np.random.default_rng(seed)))
    else:
        for name in names:
            write(args.out_dir / f"{name}.wav", SOUNDS[name](np.random.default_rng(args.seed)))


if __name__ == "__main__":
    main()

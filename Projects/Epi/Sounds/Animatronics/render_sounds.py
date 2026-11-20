"""Render the non-verbal robot sounds for EpiAnimatronics.

Every sound is a short phrase of chirps and whistles. Each syllable has its own pitch contour,
attack, decay, vibrato, and warble, so the phrase carries emotion through intonation alone: a
greeting bounces up, a question ends rising, surprise starts low and climbs, shyness sinks softly.

Needs only numpy. From the repository root, for example with the venv in UserData:

    python3 -m venv UserData/venvs/sounds
    UserData/venvs/sounds/bin/pip install numpy
    UserData/venvs/sounds/bin/python Projects/Epi/Sounds/Animatronics/render_sounds.py

Every sound is rendered in three variants, name_1.wav to name_3.wav, and the demo picks one at
random each time. The variants are seeded, so the files are reproducible; --seed gives other ones.
To audition other takes without replacing the demo files:

    UserData/venvs/sounds/bin/python Projects/Epi/Sounds/Animatronics/render_sounds.py --seed 10 --out-dir UserData/output
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


# The sounds. Every function draws its own shape from rng, so each variant is a different
# utterance with the same meaning: the number of syllables, their contours, the transposition,
# vibrato, brightness, and loudness vary. Sounds that the motion follows (giggle, hehe, poof) keep
# their rhythm and vary only in pitch and timbre.

def transpose(rng, semitones=2.0):
    """A random pitch factor within +-semitones, so variants sit higher or lower."""
    return 2.0 ** (rng.uniform(-semitones, semitones) / 12.0)


def level(rng, spread=2.0):
    """A random loudness offset in dB, so variants are not all equally loud."""
    return rng.uniform(-spread, spread)


def hi(rng):
    """Greeting: two to four bouncy chirps that end high and open, like a cheerful hello."""
    k = transpose(rng)
    count = int(rng.integers(2, 5))
    parts, t = [], 0.0
    for i in range(count - 1):
        start = rng.uniform(850, 1500) * k
        end = start * (rng.uniform(1.2, 1.5) if rng.random() < 0.6 else rng.uniform(0.7, 0.85))
        duration = rng.uniform(0.06, 0.09)
        parts.append((t, syllable(duration, [(0, start), (1, end)], attack=0.004, release=0.03,
                                  brightness=rng.uniform(0.2, 0.4), rng=rng)))
        t += duration + rng.uniform(0.02, 0.04)
    low = rng.uniform(950, 1150) * k
    parts.append((t, syllable(rng.uniform(0.16, 0.24), [(0, low), (0.5, low * 1.6), (1, low * rng.uniform(1.8, 2.1))],
                              attack=0.006, release=0.08, vibrato=(rng.uniform(8, 12), rng.uniform(0.2, 0.45)),
                              brightness=0.25, rng=rng)))
    return finish(phrase(parts), gain_db=level(rng))


def wow(rng):
    """Surprise: a long whistle that starts low and climbs, its vibrato widening, often ending in an 'eep'."""
    k = transpose(rng, 1.5)
    start, top = rng.uniform(150, 210) * k, rng.uniform(1250, 1650) * k
    parts = [(0.00, syllable(0.62, [(0, start), (rng.uniform(0.15, 0.3), start * 1.1), (0.8, top * 0.7), (1, top)],
                             attack=0.05, release=0.06, vibrato=(rng.uniform(6, 8), 0.15),
                             vibrato_growth=rng.uniform(2.0, 4.0), brightness=rng.uniform(0.35, 0.55), rng=rng))]
    if rng.random() < 0.7:  # the 'eep' stays on the beat that the head lift in mind blown peaks on
        eep = top * rng.uniform(1.1, 1.3)
        parts.append((0.66, syllable(0.13, [(0, eep), (1, eep * 1.3)], attack=0.004, release=0.06, brightness=0.3, rng=rng)))
    return finish(phrase(parts), gain_db=level(rng, 1.0))


def poof(rng):
    """Mind blown: a sputtering whistle that powers down, inside a soft puff of air and a thump."""
    k = transpose(rng, 2.5)
    return finish(phrase([
        (0.00, syllable(0.85, [(0, 1600 * k), (rng.uniform(0.2, 0.4), 750 * k), (1, rng.uniform(90, 140))],
                        attack=0.005, release=0.3, warble_rate=rng.uniform(12, 20), warble_depth=rng.uniform(1.2, 2.4),
                        brightness=rng.uniform(0.25, 0.45), gain=0.8, rng=rng)),
        (0.00, puff(rng.uniform(0.55, 0.8), rng, cutoff_start=rng.uniform(2500, 3500), gain=0.9)),
        (0.00, syllable(0.25, [(0, 90), (1, 60)], attack=0.004, release=0.2, brightness=0.0, gain=0.6, rng=rng)),
    ]), gain_db=level(rng, 1.0))


# Onsets of the giggle chirps in seconds; EpiAnimations.json flashes the mouth on these beats.
GIGGLE_ONSETS = [0.0, 0.10, 0.185, 0.26, 0.325, 0.385, 0.44, 0.49]


def giggle(rng):
    """Laughter: quick chirps that hop between high and low, speed up, and drift down."""
    k = transpose(rng, 2.5)
    hop = rng.uniform(1.1, 1.3)
    glide = rng.uniform(1.1, 1.3) if rng.random() < 0.6 else rng.uniform(0.8, 0.9)
    drift = rng.uniform(0.3, 1.0)
    brightness = rng.uniform(0.2, 0.45)
    parts = []
    for i, onset in enumerate(GIGGLE_ONSETS):
        base = 1500 * k * 2 ** (-i * drift / 12) * (hop if i % 2 == 0 else 1.0 / hop)
        parts.append((onset, syllable(rng.uniform(0.05, 0.07), [(0, base), (1, base * glide)], attack=0.003,
                                      release=0.03, brightness=brightness, gain=1.0 - 0.05 * i, rng=rng)))
    return finish(phrase(parts), gain_db=level(rng))


def hmm(rng):
    """Curious question: a thoughtful hum, then a chirp that rises at the end, sometimes in two steps."""
    k = transpose(rng)
    hum = rng.uniform(560, 680) * k
    hum_length = rng.uniform(0.16, 0.26)
    parts = [(0.00, syllable(hum_length, [(0, hum), (1, hum * rng.uniform(0.95, 1.02))], attack=0.04, release=0.06,
                             vibrato=(rng.uniform(4, 6), 0.2), brightness=0.15, gain=0.8, rng=rng))]
    t = hum_length + rng.uniform(0.03, 0.06)
    if rng.random() < 0.35:
        parts.append((t, syllable(0.09, [(0, hum * 1.05), (1, hum * 1.25)], attack=0.008, release=0.03, brightness=0.25, rng=rng)))
        t += 0.12
    top = hum * rng.uniform(2.0, 2.6)
    parts.append((t, syllable(rng.uniform(0.2, 0.28), [(0, hum * 1.05), (0.4, hum * 1.15), (1, top)],
                              attack=0.01, release=0.06, brightness=0.25, rng=rng)))
    return finish(phrase(parts), gain_db=level(rng))


def aww(rng):
    """Shy: a soft, slow whistle that lifts a little and then sinks away, sometimes with a second sigh."""
    k = transpose(rng)
    start = rng.uniform(950, 1150) * k
    parts = [(0.00, syllable(rng.uniform(0.5, 0.7), [(0, start), (0.3, start * rng.uniform(1.08, 1.2)), (1, start * rng.uniform(0.52, 0.62))],
                             attack=0.09, release=0.25, vibrato=(rng.uniform(4, 6), rng.uniform(0.15, 0.3)), brightness=0.05, rng=rng))]
    if rng.random() < 0.3:
        parts.append((0.62, syllable(0.3, [(0, start * 0.7), (1, start * 0.55)], attack=0.06, release=0.15,
                                     brightness=0.03, gain=0.5, rng=rng)))
    return finish(phrase(parts), gain_db=-3.0 + level(rng, 1.5))


def sparkle(rng):
    """Joy: three or four bright blips that climb, the last one ringing with a happy trill."""
    k = transpose(rng)
    count = int(rng.integers(3, 5))
    steps = [0, 4, 7, 12, 16]  # a rising major chord
    root = rng.uniform(1150, 1400) * k
    spacing = rng.uniform(0.05, 0.07)
    notes = [root * 2 ** (steps[i] / 12) for i in range(count)]
    parts = [(spacing * i, syllable(0.05, [(0, f), (1, f * 1.08)], attack=0.003, release=0.03,
                                    brightness=0.2, rng=rng)) for i, f in enumerate(notes)]
    last = notes[-1] * rng.uniform(1.15, 1.3)
    parts.append((spacing * count, syllable(rng.uniform(0.18, 0.26), [(0, last), (1, last * 1.08)], attack=0.003,
                                            release=0.12, vibrato=(rng.uniform(12, 16), 0.4), brightness=0.15, rng=rng)))
    return finish(phrase(parts), gain_db=-2.0 + level(rng, 1.5))


# Onsets of the "he-he-heh" syllables in seconds; EpiAnimations.json bobs the head on these beats.
HEHE_ONSETS = [0.0, 0.16, 0.32, 0.50]


def hehe(rng):
    """Mischief: a sly, low "he-he-heh" of three clipped chirps that slide down, then a sneaky rise or a low chuckle."""
    k = transpose(rng, 2.5)
    slide = rng.uniform(0.75, 0.88)
    parts = []
    for i, onset in enumerate(HEHE_ONSETS[:3]):
        f = 760 * k * (0.92 ** i) * rng.uniform(0.97, 1.03)
        parts.append((onset, syllable(rng.uniform(0.08, 0.1), [(0, f), (1, f * slide)], attack=0.004, release=0.04,
                                      brightness=rng.uniform(0.45, 0.65), gain=1.0 - 0.08 * i, rng=rng)))
    if rng.random() < 0.7:
        parts.append((HEHE_ONSETS[3], syllable(0.2, [(0, 620 * k), (0.5, 640 * k), (1, rng.uniform(900, 1100) * k)],
                                               attack=0.02, release=0.07, vibrato=(7, 0.2), brightness=0.35, gain=0.8, rng=rng)))
    else:
        parts.append((HEHE_ONSETS[3], syllable(0.16, [(0, 560 * k), (1, 430 * k)], attack=0.01, release=0.08,
                                               brightness=0.5, gain=0.75, rng=rng)))
    return finish(phrase(parts), gain_db=level(rng))


def chirp(rng):
    """Found someone: one happy, quirky syllable that scoops up quickly and drops back a little at the end."""
    k = transpose(rng)
    start = rng.uniform(900, 1200) * k
    top = start * rng.uniform(1.6, 2.1)
    return finish(phrase([
        (0.00, syllable(rng.uniform(0.12, 0.2), [(0, start), (rng.uniform(0.55, 0.7), top), (1, top * rng.uniform(0.86, 0.94))],
                        attack=0.004, release=0.05, vibrato=(rng.uniform(14, 20), rng.uniform(0.1, 0.25)),
                        brightness=rng.uniform(0.25, 0.4), rng=rng)),
    ]), gain_db=level(rng, 1.5))


def eek(rng):
    """Fear: a sudden shriek that leaps up, then a whimper that sinks with a fast, shaky tremble."""
    k = transpose(rng, 2.0)
    start = rng.uniform(1300, 1600) * k
    peak = start * rng.uniform(1.7, 2.0)
    shriek = rng.uniform(0.16, 0.22)
    parts = [(0.00, syllable(shriek, [(0, start), (0.25, peak), (1, peak * rng.uniform(0.85, 0.95))],
                             attack=0.002, release=0.04, warble_rate=rng.uniform(25, 35), warble_depth=0.3,
                             brightness=rng.uniform(0.5, 0.7), rng=rng))]
    whimper = start * rng.uniform(0.9, 1.05)
    parts.append((shriek + rng.uniform(0.03, 0.06),
                  syllable(rng.uniform(0.45, 0.6), [(0, whimper), (0.3, whimper * 1.05), (1, whimper * rng.uniform(0.5, 0.6))],
                           attack=0.02, release=0.2, vibrato=(rng.uniform(11, 14), rng.uniform(0.5, 0.8)),
                           brightness=rng.uniform(0.15, 0.3), gain=0.7, rng=rng)))
    return finish(phrase(parts), gain_db=level(rng, 1.0))


SOUNDS = {"chirp": chirp, "hi": hi, "hehe": hehe, "wow": wow, "poof": poof, "giggle": giggle, "hmm": hmm, "aww": aww, "sparkle": sparkle, "eek": eek}


# Fixed seed number of each sound, so adding a sound never changes the takes of the others.
SEED_IDS = {"aww": 0, "giggle": 1, "hehe": 2, "hi": 3, "hmm": 4, "poof": 5, "sparkle": 6, "wow": 7, "chirp": 8, "eek": 9}


def write(path, samples):
    data = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(data.tobytes())
    print(f"{path}: {len(samples) / RATE:.2f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--only", nargs="*", help="Render only these sounds")
    parser.add_argument("--variants", type=int, default=3, help="Number of variants of each sound, written as name_1.wav, name_2.wav, ...")
    parser.add_argument("--seed", type=int, default=1, help="Seed of the first variant; other seeds give other variants")
    args = parser.parse_args()

    for name in args.only or list(SOUNDS):
        for i in range(args.variants):
            # Each sound and variant has its own seed, so changing one sound leaves the others alone.
            rng = np.random.default_rng([args.seed + i, SEED_IDS[name]])
            write(args.out_dir / f"{name}_{i + 1}.wav", SOUNDS[name](rng))


if __name__ == "__main__":
    main()

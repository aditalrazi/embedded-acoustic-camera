"""Fake microphone signals: horn + room echoes + noise.

This lets you test the entire signal chain before a single wire is
soldered, and it gives you the 'expected performance' numbers to compare
your real measurements against. When your real array reports 9 degrees of
error and the simulator says 2, you know the problem is in the hardware,
and you know to go looking there instead of rewriting the maths.
"""

import numpy as np

from array_geometry import speed_of_sound


def horn_signal(n, fs, f0=420.0, n_harm=6, rng=None):
    """A synthetic car horn.

    Real car horns are dual-tone: two notes roughly a minor third apart
    (frequency ratio about 1.19), each with a stack of harmonics.
    Fundamentals are usually 400-500 Hz.
    """
    rng = rng or np.random.default_rng()
    t = np.arange(n) / fs
    x = np.zeros(n)
    for f in (f0, f0 * 1.19):
        for k in range(1, n_harm + 1):
            x += (1.0 / k) * np.sin(2 * np.pi * f * k * t
                                    + rng.random() * 2 * np.pi)
    return x / np.max(np.abs(x))


def render(u, mics, fs=44100, dur=0.1, snr_db=15.0, n_reflections=12,
           reflection_gain=0.5, f0=420.0, temp_c=20.0, gain_spread_db=0.0,
           source=None, rng=None):
    """Render what each microphone hears for a source in direction u.

    Echoes are modelled crudely, as extra copies of the same signal
    arriving from random directions 2-30 ms later. Crude, but it captures
    the thing that actually breaks TDOA: correlated, delayed copies.
    """
    rng = rng or np.random.default_rng()
    mics = np.asarray(mics, dtype=float)
    u = np.asarray(u, dtype=float)
    c = speed_of_sound(temp_c)

    n, pad = int(dur * fs), 4096
    src = source if source is not None else horn_signal(n + 2 * pad, fs, f0, rng=rng)
    src = np.asarray(src, dtype=float)[:n + 2 * pad]
    N = len(src)
    S = np.fft.rfft(src)
    k = np.fft.rfftfreq(N, d=1.0)

    paths = [(u, 1.0, 0.0)]
    for _ in range(n_reflections):
        v = rng.standard_normal(3)
        v[2] = abs(v[2])  # keep echoes in front of the plate
        v /= np.linalg.norm(v)
        paths.append((v,
                      reflection_gain * rng.uniform(0.3, 1.0) / np.sqrt(n_reflections),
                      rng.uniform(0.002, 0.030)))

    out = []
    for m in mics:
        acc = np.zeros(n)
        for v, g, extra in paths:
            delay_samples = (-(m @ v) / c + extra) * fs
            shifted = np.fft.irfft(S * np.exp(-2j * np.pi * k * delay_samples), N)
            acc += g * shifted[pad:pad + n]
        if gain_spread_db:
            acc *= 10 ** (rng.uniform(-gain_spread_db, gain_spread_db) / 20.0)
        noise = rng.standard_normal(n)
        noise *= (np.sqrt(np.mean(acc ** 2))
                  / (10 ** (snr_db / 20.0) * np.sqrt(np.mean(noise ** 2)) + 1e-12))
        out.append(acc + noise)
    return np.array(out)

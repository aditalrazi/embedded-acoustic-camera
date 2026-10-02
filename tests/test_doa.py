"""Run with: pytest tests/ -v

Every one of these must pass before you plug in a single wire. When the
real hardware misbehaves you want to be certain the maths is innocent.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from array_geometry import (square_array, unit_from_angles, angles_from_unit,
                            angular_error_deg, speed_of_sound,
                            alias_free_limit_hz)
from doa import gcc_phat, estimate_doa, is_horn, horn_score, srp_phat_map
from simulate import render, horn_signal

MICS = square_array(0.12)
FS = 44100


# ---------------------------------------------------------------- basics
def test_speed_of_sound():
    assert abs(speed_of_sound(20.0) - 343.4) < 0.5
    assert speed_of_sound(32.0) > speed_of_sound(20.0)


def test_angle_roundtrip():
    for az, el in [(0, 0), (30, -15), (-45, 20), (12.5, 7.5)]:
        u = unit_from_angles(az, el)
        a2, e2 = angles_from_unit(u)
        assert abs(a2 - az) < 1e-6 and abs(e2 - el) < 1e-6


def test_alias_limit():
    assert abs(alias_free_limit_hz(0.12) - 1429.0) < 2.0


# ------------------------------------------------------------- gcc-phat
def test_gcc_phat_recovers_known_delay():
    """Inject an exact 7.3-sample delay and demand it back."""
    rng = np.random.default_rng(0)
    n = 4096
    x = rng.standard_normal(n)
    X = np.fft.rfft(x, 2 * n)
    k = np.fft.rfftfreq(2 * n, d=1.0)
    y = np.fft.irfft(X * np.exp(-2j * np.pi * k * 7.3), 2 * n)[:n]
    tau, _ = gcc_phat(y, x, FS, max_tau=0.01, band=None, beta=1.0)
    assert abs(tau * FS - 7.3) < 0.15


def test_gcc_phat_sign_convention():
    """A LATE signal must give a POSITIVE delay. Get this backwards and
    every angle in the project is mirrored."""
    rng = np.random.default_rng(3)
    n = 4096
    x = rng.standard_normal(n)
    X = np.fft.rfft(x, 2 * n)
    k = np.fft.rfftfreq(2 * n, d=1.0)
    late = np.fft.irfft(X * np.exp(-2j * np.pi * k * 5.0), 2 * n)[:n]
    tau, _ = gcc_phat(late, x, FS, max_tau=0.01, band=None, beta=1.0)
    assert tau > 0


# ------------------------------------------------------------------ doa
@pytest.mark.parametrize("az,el", [(0, 0), (20, -10), (-30, 15), (35, 20)])
def test_doa_anechoic(az, el):
    u_true = unit_from_angles(az, el)
    sig = render(u_true, MICS, fs=FS, snr_db=30, n_reflections=0,
                 rng=np.random.default_rng(1))
    assert angular_error_deg(u_true, estimate_doa(sig, MICS, FS)) < 3.0


def test_doa_reverberant_median():
    """Any single trial can be unlucky, so test the median of thirty."""
    u_true = unit_from_angles(20, -10)
    errs = [angular_error_deg(
        u_true,
        estimate_doa(render(u_true, MICS, fs=FS, snr_db=10, n_reflections=20,
                            reflection_gain=0.7,
                            rng=np.random.default_rng(s)), MICS, FS))
        for s in range(30)]
    assert np.median(errs) < 6.0


def test_gain_mismatch_does_not_matter():
    """PHAT divides out magnitude, so it is blind to gain. You must match
    your microphones in TIMING, not in loudness. If this fails you have a
    bug in the weighting."""
    u_true = unit_from_angles(25, 0)
    a = estimate_doa(render(u_true, MICS, fs=FS, snr_db=20, gain_spread_db=0,
                            rng=np.random.default_rng(5)), MICS, FS)
    b = estimate_doa(render(u_true, MICS, fs=FS, snr_db=20, gain_spread_db=10,
                            rng=np.random.default_rng(5)), MICS, FS)
    assert angular_error_deg(a, b) < 2.0


def test_channel_swap_is_detected():
    """Swapping two channels MUST produce a wrong answer. This proves the
    test suite would catch the most common wiring mistake in the project."""
    u_true = unit_from_angles(30, 0)
    sig = render(u_true, MICS, fs=FS, snr_db=30, n_reflections=0,
                 rng=np.random.default_rng(2))
    swapped = sig[[1, 0, 2, 3]]
    assert angular_error_deg(u_true, estimate_doa(swapped, MICS, FS)) > 10.0


def test_band_limiting_helps():
    """Assert the band mask is actually being applied. It is roughly a 4x
    effect, so a silently broken mask would be an expensive thing to miss."""
    u_true = unit_from_angles(25, 10)
    seeds = range(12)
    full = [angular_error_deg(u_true, estimate_doa(
        render(u_true, MICS, fs=FS, snr_db=10, n_reflections=15,
               rng=np.random.default_rng(s)), MICS, FS, band=None))
        for s in seeds]
    lim = [angular_error_deg(u_true, estimate_doa(
        render(u_true, MICS, fs=FS, snr_db=10, n_reflections=15,
               rng=np.random.default_rng(s)), MICS, FS, band=(300, 2500)))
        for s in seeds]
    assert np.median(lim) < np.median(full)


def test_error_grows_off_axis():
    """arcsin is steep near the edges, so a fixed timing error becomes a
    bigger angle error off-axis. If this fails, your geometry or your sign
    conventions are wrong."""
    def med(az):
        u = unit_from_angles(az, 0)
        return np.median([angular_error_deg(u, estimate_doa(
            render(u, MICS, fs=FS, snr_db=6, n_reflections=10,
                   rng=np.random.default_rng(s)), MICS, FS))
            for s in range(20)])
    assert med(55) > med(0)


def test_residual_flags_two_sources():
    """The live overlay's outlier gate depends on this being true."""
    rng = np.random.default_rng(11)
    ua = unit_from_angles(-25, 0)
    ub = unit_from_angles(25, 0)
    one = render(ua, MICS, fs=FS, snr_db=20, n_reflections=4, rng=rng)
    two = one + render(ub, MICS, fs=FS, snr_db=20, n_reflections=4,
                       f0=505.0, rng=rng)
    _, i1 = estimate_doa(one, MICS, FS, return_detail=True)
    _, i2 = estimate_doa(two, MICS, FS, return_detail=True)
    assert i2["residual_rms_m"] > i1["residual_rms_m"]


# ----------------------------------------------------------- horn gate
def test_horn_detector_accepts_horns():
    rng = np.random.default_rng(7)
    x = horn_signal(8192, FS, rng=rng)
    x = x + 0.5 * rng.standard_normal(len(x)) * np.sqrt(np.mean(x ** 2))
    assert is_horn(x, FS)


@pytest.mark.parametrize("kind", ["noise", "impulse", "rumble", "tone"])
def test_horn_detector_rejects_non_horns(kind):
    rng = np.random.default_rng(9)
    n = 8192
    if kind == "noise":
        x = rng.standard_normal(n)
    elif kind == "impulse":
        x = np.zeros(n)
        x[n // 2:n // 2 + 20] = rng.standard_normal(20)
    elif kind == "rumble":
        x = rng.standard_normal(n)
        f = np.fft.rfftfreq(n, 1 / FS)
        X = np.fft.rfft(x) / (1.0 + (f / 60.0) ** 2)
        x = np.fft.irfft(X, n)
    else:                      # a pure sine: huge fundamental,
        t = np.arange(n) / FS  # but no harmonics at all
        x = np.sin(2 * np.pi * 440 * t)
    assert not is_horn(x, FS)


def test_horn_score_finds_the_fundamental():
    x = horn_signal(8192, FS, f0=420.0, rng=np.random.default_rng(4))
    f0, p0, _ = horn_score(x, FS)
    assert abs(f0 - 420.0) < 25.0 and p0 > 20.0


# --------------------------------------------------------------- srp
def test_srp_peak_near_truth():
    u_true = unit_from_angles(25, -15)
    sig = render(u_true, MICS, fs=FS, snr_db=15, n_reflections=12,
                 rng=np.random.default_rng(6))
    axis, P = srp_phat_map(sig, MICS, FS, grid=21, half_fov_deg=40)
    r, c = np.unravel_index(np.argmax(P), P.shape)
    step = axis[1] - axis[0]
    assert abs(axis[c] - 25) <= 1.5 * step
    assert abs(axis[r] + 15) <= 1.5 * step

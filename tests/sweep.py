"""Generate the baseline performance table.

    python tests/sweep.py > docs/simulation_baseline.txt

Run this in week 0, before your hardware arrives. It tells you what
performance to EXPECT. In week 3, when your real array reports 9 degrees
of error and this table says 2, you know immediately that the problem is
in the hardware and not in the maths -- and that is worth days.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from array_geometry import square_array, unit_from_angles, angular_error_deg
from doa import estimate_doa
from simulate import render

FS = 44100
N_TRIALS = 24
ANGLES = [(0, 0), (15, 8), (-20, -10), (30, 15), (-35, 5)]


def trial_errors(mics, snr_db, refl, gain, band, beta, seed0=0):
    errs = []
    for k, (az, el) in enumerate(ANGLES):
        u = unit_from_angles(az, el)
        for s in range(N_TRIALS // len(ANGLES) + 1):
            rng = np.random.default_rng(seed0 + 1000 * k + s)
            sig = render(u, mics, fs=FS, snr_db=snr_db, n_reflections=refl,
                         reflection_gain=gain, rng=rng)
            errs.append(angular_error_deg(u, estimate_doa(
                sig, mics, FS, band=band, beta=beta)))
    return np.array(errs)


def row(label, errs):
    print(f" {label:<34s} {np.mean(errs):6.2f} {np.median(errs):8.2f} "
          f"{np.percentile(errs, 90):8.2f}")


def head(title):
    print(f"\n{title}")
    print(f" {'condition':<34s} {'mean':>6s} {'median':>8s} {'p90':>8s}")
    print(" " + "-" * 60)


def main():
    m12 = square_array(0.12)
    print("Baseline simulation, 4-mic square array, fs = 44100 Hz")
    print(f"{N_TRIALS + 5} trials per condition, spread over 5 source angles")
    print("All errors in degrees (great-circle angle between true and "
          "estimated direction)")

    head("1. Effect of signal-to-noise ratio (12 cm array, medium reverb)")
    for snr in [30, 20, 10, 5, 0, -5]:
        row(f"SNR = {snr:+3d} dB", trial_errors(m12, snr, 12, 0.5, (300, 2500), 0.7))

    head("2. Effect of array size (10 dB SNR, medium reverb)")
    for side in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]:
        row(f"side = {side * 100:4.0f} cm",
            trial_errors(square_array(side), 10, 12, 0.5, (300, 2500), 0.7))

    head("3. Effect of band-limiting (10 dB SNR, medium reverb)")
    for band, name in [(None, "no band limit (full band)"),
                       ((300, 2500), "300 - 2500 Hz"),
                       ((300, 4000), "300 - 4000 Hz"),
                       ((200, 1500), "200 - 1500 Hz")]:
        row(name, trial_errors(m12, 10, 12, 0.5, band, 0.7))

    head("4. Effect of the PHAT exponent beta, by room")
    for refl, gain, name in [(0, 0.0, "anechoic"),
                             (8, 0.3, "light reverb"),
                             (16, 0.6, "medium reverb"),
                             (24, 0.9, "heavy reverb")]:
        for beta in [0.0, 0.5, 0.7, 1.0]:
            row(f"{name:<16s} beta = {beta:.1f}",
                trial_errors(m12, 10, refl, gain, (300, 2500), beta))

    head("5. Effect of microphone gain mismatch (PHAT should ignore it)")
    for spread in [0.0, 6.0, 12.0]:
        errs = []
        for k, (az, el) in enumerate(ANGLES):
            u = unit_from_angles(az, el)
            for s in range(6):
                rng = np.random.default_rng(500 + 1000 * k + s)
                sig = render(u, m12, fs=FS, snr_db=10, n_reflections=12,
                             gain_spread_db=spread, rng=rng)
                errs.append(angular_error_deg(u, estimate_doa(sig, m12, FS)))
        row(f"gain spread = +/-{spread:4.1f} dB", np.array(errs))

    head("6. Effect of off-axis angle (10 dB SNR, medium reverb)")
    for az in [0, 10, 20, 30, 40, 50, 60]:
        u = unit_from_angles(az, 0)
        errs = [angular_error_deg(u, estimate_doa(
            render(u, m12, fs=FS, snr_db=10, n_reflections=12,
                   rng=np.random.default_rng(700 + s)), m12, FS))
            for s in range(20)]
        row(f"azimuth = {az:3d} deg", np.array(errs))

    head("7. Effect of analysis window length (10 dB SNR, medium reverb)")
    for dur in [0.02, 0.046, 0.093, 0.186]:
        errs = []
        for k, (az, el) in enumerate(ANGLES):
            u = unit_from_angles(az, el)
            for s in range(6):
                sig = render(u, m12, fs=FS, dur=dur, snr_db=10,
                             n_reflections=12,
                             rng=np.random.default_rng(900 + 1000 * k + s))
                errs.append(angular_error_deg(u, estimate_doa(sig, m12, FS)))
        row(f"window = {dur * 1000:5.1f} ms "
            f"({int(dur * FS):5d} samples)", np.array(errs))


if __name__ == "__main__":
    main()

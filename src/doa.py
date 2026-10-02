"""Direction-of-arrival estimation.

Everything in this file is a PURE FUNCTION: numbers go in, numbers come
out. No serial ports, no camera, no global state. That is deliberate --
it means the identical code runs against the simulator, against a
recorded WAV file, and against the live Teensy stream, so when the live
system misbehaves you already know the maths is not the problem.
"""

import numpy as np

from array_geometry import all_pairs, speed_of_sound, max_delay_s
import config


# ----------------------------------------------------------------------
# 1. Time delay between two microphones
# ----------------------------------------------------------------------
def gcc_phat(sig_a, sig_b, fs, max_tau=None, interp=config.INTERP,
             band=config.BAND, beta=config.BETA):
    """Arrival-time delay of sig_a relative to sig_b, in SECONDS.

    Sign convention: a POSITIVE result means sig_a arrived LATE, which
    means the source is closer to microphone b. Get this backwards and
    every angle you produce is mirrored.

    max_tau  Clamp the search to +/- this many seconds. Always pass
             |m_i - m_j| / c. Any delay bigger than that is physically
             impossible, so allowing it can only ever let a reflection win.
    band     (low, high) in Hz. Frequency bins outside are zeroed before
             correlating. Two lines of code, roughly a 4x accuracy win.
    beta     Phase-transform exponent. 0 = ordinary cross-correlation,
             1 = full PHAT, 0.7 = the best compromise in a real, echoey room.
    interp   Up-sampling factor for the correlation function, so the peak
             can land between samples.

    Returns (tau_seconds, peak_quality).
    """
    sig_a = np.asarray(sig_a, dtype=float)
    sig_b = np.asarray(sig_b, dtype=float)

    n = len(sig_a) + len(sig_b)  # zero-pad -> linear, not circular
    A = np.fft.rfft(sig_a, n=n)
    B = np.fft.rfft(sig_b, n=n)

    R = A * np.conj(B)  # cross-power spectrum

    if beta > 0:  # (partial) phase transform
        R = R / (np.abs(R) ** beta + 1e-12)

    if band is not None:  # band-limit
        f = np.fft.rfftfreq(n, d=1.0 / fs)
        R[(f < band[0]) | (f > band[1])] = 0.0

    cc = np.fft.irfft(R, n=n * interp)  # up-sampled correlation function

    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = min(int(interp * fs * max_tau) + interp, max_shift)

    # rearrange so that index 0 is the most negative delay
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    mag = np.abs(cc)
    peak = int(np.argmax(mag))

    # Parabolic sub-sample interpolation: fit a parabola through the peak
    # and its two neighbours and take the vertex. Three lines, and it is
    # why a 44.1 kHz sample rate is plenty.
    offset = 0.0
    if 0 < peak < len(cc) - 1:
        y0, y1, y2 = mag[peak - 1], mag[peak], mag[peak + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            offset = 0.5 * (y0 - y2) / denom
            offset = float(np.clip(offset, -0.5, 0.5))

    tau_samples = (peak + offset) - max_shift
    quality = float(mag.max() / (mag.mean() + 1e-12))
    return tau_samples / (interp * fs), quality


# ----------------------------------------------------------------------
# 2. Six delays -> one direction
# ----------------------------------------------------------------------
def estimate_doa(blocks, mics, fs, temp_c=config.TEMP_C, band=config.BAND,
                 beta=config.BETA, return_detail=False):
    """blocks: (n_mics, n_samples) float array. Returns unit vector u.

    Because every microphone lies in the z = 0 plane, the vector between
    any two of them has no z component, so the delay equation

        c * tau_ij = -( dx_ij * ux + dy_ij * uy )

    is LINEAR in ux and uy and does not involve uz at all. Four
    microphones give six pairs -- six equations, two unknowns -- which is
    an ordinary over-determined least-squares problem. No iteration, no
    starting guess, no local minima. uz then comes from the fact that u
    has length 1.
    """
    blocks = np.asarray(blocks, dtype=float)
    mics = np.asarray(mics, dtype=float)
    c = speed_of_sound(temp_c)

    A_rows, b_vals, quals, taus = [], [], [], []
    for i, j in all_pairs(len(mics)):
        d = mics[i] - mics[j]
        tau, q = gcc_phat(blocks[i], blocks[j], fs,
                          max_tau=max_delay_s(mics[i], mics[j], c),
                          band=band, beta=beta)
        A_rows.append([-d[0], -d[1]])
        b_vals.append(c * tau)
        quals.append(q)
        taus.append(tau)

    A = np.asarray(A_rows)
    b = np.asarray(b_vals)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)

    ux, uy = float(sol[0]), float(sol[1])
    r = float(np.hypot(ux, uy))
    clipped = r > 1.0
    if clipped:                  # outside the physically possible disc
        ux, uy = ux / r, uy / r  # -> source near the horizon, or bad data
    uz = float(np.sqrt(max(0.0, 1.0 - ux * ux - uy * uy)))
    u = np.array([ux, uy, uz])

    if not return_detail:
        return u

    resid = A @ sol - b
    return u, {
        "taus_us": [t * 1e6 for t in taus],
        "peak_quality": float(np.mean(quals)),
        "residual_rms_m": float(np.sqrt(np.mean(resid ** 2))),
        "clipped_to_horizon": bool(clipped),
        "radius": r,
    }


# ----------------------------------------------------------------------
# 3. Is that actually a horn?
# ----------------------------------------------------------------------
def _local_noise_floor(mag, f, width_hz=250.0, lo=100.0, hi=6000.0):
    """Median magnitude in ~250 Hz neighbourhoods, interpolated smoothly.

    A single median over the whole spectrum does not work: real room noise
    (fans, traffic, footsteps) is heavily tilted towards low frequencies,
    so against a global median that tilt alone makes any low rumble look
    hugely 'prominent' and the detector fires on nothing.
    """
    edges = np.arange(lo, hi + width_hz, width_hz)
    centres, meds = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (f >= a) & (f < b)
        if m.any():
            centres.append(0.5 * (a + b))
            meds.append(np.median(mag[m]))
    if len(centres) < 2:
        return np.full_like(mag, np.median(mag) + 1e-12)
    return np.interp(f, centres, meds)


def horn_score(x, fs, f0_range=(300.0, 800.0)):
    """How horn-like is this signal?

    Returns (f0_hz, fundamental_prominence, best_harmonic_prominence).
    Prominence is a plain ratio above the local noise floor, so 20 means
    'twenty times louder than its neighbours', i.e. about 26 dB.

    Why prominence and not energy: an energy test cannot tell a horn from
    a door slam. Both put lots of power in the 300-800 Hz region. What
    makes a horn a horn is that it is a HARMONIC STACK -- a few very sharp
    spikes at f0, 2*f0, 3*f0 standing far above everything around them.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 512:
        return 0.0, 0.0, 0.0

    w = np.hanning(n)
    mag = np.abs(np.fft.rfft(x * w))
    f = np.fft.rfftfreq(n, 1.0 / fs)

    floor = _local_noise_floor(mag, f)
    prom = mag / (floor + 1e-12)

    sel = (f >= f0_range[0]) & (f <= f0_range[1])
    if not sel.any():
        return 0.0, 0.0, 0.0
    k = int(np.argmax(prom[sel]))
    f0 = float(f[sel][k])
    p0 = float(prom[sel][k])

    best_h = 0.0
    for mult in (2, 3):
        target = mult * f0
        if target > f[-1]:
            continue
        near = np.abs(f - target) < max(15.0, 0.03 * target)
        if near.any():
            best_h = max(best_h, float(prom[near].max()))

    return f0, p0, best_h


def is_horn(x, fs, p0_thresh=20.0, harm_thresh=6.3):
    """Two criteria, both must pass:
      * the fundamental stands >= 26 dB above its local neighbourhood
      * at least one of 2*f0 or 3*f0 stands >= 16 dB above its own

    The second criterion is what rejects a pure sine tone, which has an
    astronomically prominent fundamental but no harmonics at all.
    """
    _, p0, ph = horn_score(x, fs)
    return bool(p0 >= p0_thresh and ph >= harm_thresh)


# ----------------------------------------------------------------------
# 4. SRP-PHAT heat map (optional, for the 'or heatmap' in the brief)
# ----------------------------------------------------------------------
def srp_phat_map(blocks, mics, fs, grid=21, half_fov_deg=40.0,
                 band=config.BAND, beta=config.BETA, temp_c=config.TEMP_C):
    """Steered Response Power with PHAT weighting.

    Instead of solving for one direction, score EVERY direction on a grid.
    For each candidate direction, work out what delay each microphone pair
    would have if the source were there, then add up the cross-correlations
    evaluated at exactly those delays. The true direction scores highest.

    Returns (axis_degrees, power_grid) where power_grid[row, col] is
    indexed [elevation, azimuth].
    """
    blocks = np.asarray(blocks, dtype=float)
    mics = np.asarray(mics, dtype=float)
    c = speed_of_sound(temp_c)

    n = blocks.shape[1] * 2
    F = np.fft.rfft(blocks, n=n, axis=1)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    keep = (f >= band[0]) & (f <= band[1])
    f, F = f[keep], F[:, keep]

    ax = np.radians(np.linspace(-half_fov_deg, half_fov_deg, grid))
    AZ, EL = np.meshgrid(ax, ax, indexing="xy")
    U = np.stack([np.sin(AZ) * np.cos(EL),
                  np.sin(EL),
                  np.cos(AZ) * np.cos(EL)], axis=-1)

    P = np.zeros((grid, grid))
    for i, j in all_pairs(len(mics)):
        R = F[i] * np.conj(F[j])
        R = R / (np.abs(R) ** beta + 1e-12)
        tau = -(U @ (mics[i] - mics[j])) / c  # (grid, grid)
        phase = np.exp(2j * np.pi * np.multiply.outer(tau, f))
        P += np.real(np.tensordot(phase, R, axes=([2], [0])))

    return np.degrees(ax), P

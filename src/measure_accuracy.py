"""Guided accuracy measurement. This is what turns a gadget into an
engineering project, and it is the part markers actually reward.

    python src/measure_accuracy.py

For each position it asks you for the TRUE angle, takes ten independent
estimates, and appends everything to docs/accuracy.csv. At the end it
prints a summary table and writes two plots.

THE PROTOCOL (do not improvise this -- write it down and follow it):

  * Tape marks on the floor at a fixed radius of 2 m from the array
    centre, at -40, -20, 0, +20, +40 degrees. Use a protractor and a
    tape measure, not your judgement.
  * Three heights for the speaker: on the floor, on a table, on a box.
    Measure the height difference h and the horizontal distance d, then
    the true elevation is degrees(arctan(h / d)).
  * 10 trials per position, 15 positions, so 150 measurements.
  * Record the room temperature and put it in config.TEMP_C first. A
    10 C error is a 1.8 % error in every angle.
  * Do the whole set in one session. If you move the array, start again.
"""

import sys
import os
import csv
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from array_geometry import (mic_positions, unit_from_angles,
                            angles_from_unit, angular_error_deg)
from doa import estimate_doa
from teensy_source import TeensySource

CSV_PATH = "docs/accuracy.csv"
FIELDS = ["true_az", "true_el", "meas_az", "meas_el", "err_deg",
          "resid_mm", "quality", "rms", "temp_c"]


def measure_point(src, mics, true_az, true_el, n_trials=10):
    u_true = unit_from_angles(true_az, true_el)
    rows = []
    for k in range(n_trials):
        block = src.wait_for_data(config.BLOCK, timeout=5.0)
        if block is None:
            print(" no audio")
            break
        x = block.astype(float)
        rms = float(np.sqrt(np.mean(x ** 2)))
        u, info = estimate_doa(x, mics, config.FS, temp_c=config.TEMP_C,
                               return_detail=True)
        az, el = angles_from_unit(u)
        err = angular_error_deg(u_true, u)
        rows.append({
            "true_az": true_az, "true_el": true_el,
            "meas_az": round(az, 2), "meas_el": round(el, 2),
            "err_deg": round(err, 2),
            "resid_mm": round(info["residual_rms_m"] * 1000, 2),
            "quality": round(info["peak_quality"], 1),
            "rms": round(rms, 5), "temp_c": config.TEMP_C,
        })
        print(f" trial {k + 1:2d}: az {az:+7.2f} el {el:+7.2f} "
              f"err {err:5.2f} deg resid {info['residual_rms_m'] * 1000:5.1f} mm")
        time.sleep(0.25)
    return rows


def summarise():
    if not os.path.exists(CSV_PATH):
        print("no data yet")
        return
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    key = lambda r: (float(r["true_az"]), float(r["true_el"]))
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)

    print("\n" + "=" * 74)
    print(f"{'true az':>8s} {'true el':>8s} {'n':>4s} "
          f"{'mean err':>9s} {'sd':>7s} {'max':>7s} {'bias az':>9s} {'bias el':>9s}")
    print("-" * 74)
    all_err = []
    for (az, el), g in sorted(groups.items()):
        e = np.array([float(r["err_deg"]) for r in g])
        ba = np.mean([float(r["meas_az"]) for r in g]) - az
        be = np.mean([float(r["meas_el"]) for r in g]) - el
        all_err.append(e)
        print(f"{az:8.1f} {el:8.1f} {len(g):4d} {e.mean():9.2f} "
              f"{e.std():7.2f} {e.max():7.2f} {ba:+9.2f} {be:+9.2f}")
    e = np.concatenate(all_err)
    print("-" * 74)
    print(f"OVERALL n = {len(e)} mean {e.mean():.2f} deg "
          f"median {np.median(e):.2f} p90 {np.percentile(e, 90):.2f} "
          f"max {e.max():.2f}")
    print("=" * 74)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(install matplotlib to get the plots: pip install matplotlib)")
        return

    ta = np.array([float(r["true_az"]) for r in rows])
    ma = np.array([float(r["meas_az"]) for r in rows])
    er = np.array([float(r["err_deg"]) for r in rows])

    fig, ax = plt.subplots(figsize=(5.2, 5))
    lim = [min(ta.min(), ma.min()) - 5, max(ta.max(), ma.max()) + 5]
    ax.plot(lim, lim, "k--", lw=1, label="ideal")
    ax.scatter(ta, ma, s=18, alpha=0.7)
    ax.set_xlabel("true azimuth (deg)")
    ax.set_ylabel("measured azimuth (deg)")
    ax.set_title("Measured vs true azimuth")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("docs/azimuth_scatter.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5.6, 4))
    off = np.abs(ta)
    for a in np.unique(off):
        m = off == a
        ax.errorbar(a, er[m].mean(), yerr=er[m].std(), fmt="o", capsize=4)
    ax.set_xlabel("off-axis angle |azimuth| (deg)")
    ax.set_ylabel("angular error (deg)")
    ax.set_title("Error grows off-axis, as arcsin predicts")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("docs/error_vs_offaxis.png", dpi=150)
    print("\nplots written to docs/")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarise()
        return

    os.makedirs("docs", exist_ok=True)
    new = not os.path.exists(CSV_PATH)
    mics = mic_positions()
    src = TeensySource().start()

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        try:
            while True:
                s = input("\ntrue azimuth,elevation in degrees "
                          "(e.g. -20,0) or 'done': ").strip()
                if s.lower() in ("done", "q", ""):
                    break
                try:
                    az, el = [float(v) for v in s.split(",")]
                except ValueError:
                    print("format is: -20,0")
                    continue
                input("start the horn playing, then press Enter ...")
                for r in measure_point(src, mics, az, el):
                    writer.writerow(r)
                f.flush()
        except KeyboardInterrupt:
            pass
        finally:
            src.stop()

    summarise()


if __name__ == "__main__":
    main()

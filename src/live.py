"""THE DEMO. Live video with a marker on the sound source.

    python src/live.py

Keys:
    q    quit
    h    toggle the horn-only gate (track any loud sound vs horns only)
    m    toggle the SRP-PHAT heat map
    s    save a screenshot to docs/
    r    reset the smoothing history

Three design decisions in here are worth defending in your report and in
the viva, because each of them is the difference between a demo that looks
professional and one that looks broken:

1. THE LEVEL GATE. In silence, GCC-PHAT will happily correlate noise
   against noise and hand you a confident, meaningless direction. Without
   a gate your marker twitches around the screen whenever nobody is
   honking. Gating on input level makes it hold still and grey out, which
   is both better looking and more honest.

2. THE RESIDUAL GATE. The six pairwise delays are over-determined: only
   two numbers are unknown. How badly they disagree tells you whether the
   sound really came from ONE direction. Two overlapping sources, or a
   wall reflection beating the direct path, make the residual spike.
   Throwing those frames away is a principled outlier filter that costs
   nothing.

3. MEDIAN, NOT MEAN, SMOOTHING. A median over the last five estimates
   discards a single bad frame completely. A mean would let one wild
   outlier drag the marker halfway across the screen. The price is about
   0.15 s of lag, which nobody can see.
"""

import sys
import os
import json
import time
import threading
import collections

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from array_geometry import mic_positions, angles_from_unit
from doa import estimate_doa, horn_score, srp_phat_map
from teensy_source import TeensySource
from align_array_camera import direction_to_pixel

GREEN = (80, 220, 80)
RED = (60, 60, 235)
GREY = (140, 140, 140)
WHITE = (245, 245, 245)


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.u = None
        self.hist = collections.deque(maxlen=config.SMOOTH_N)
        self.rms = 0.0
        self.resid = 0.0
        self.f0 = 0.0
        self.horn = False
        self.live = False
        self.heat = None
        self.heat_axis = None
        self.dsp_ms = 0.0
        self.stop = False


def audio_worker(shared, src, mics, want_heat):
    counter = 0
    while not shared.stop:
        block = src.get_latest(config.BLOCK)
        if block is None:
            time.sleep(0.02)
            continue
        x = block.astype(float)
        t0 = time.perf_counter()

        rms = float(np.sqrt(np.mean(x ** 2)))
        if rms < config.RMS_GATE:
            with shared.lock:
                shared.rms, shared.live, shared.horn = rms, False, False
            time.sleep(0.02)
            continue

        u, info = estimate_doa(x, mics, config.FS,
                               temp_c=config.TEMP_C, return_detail=True)
        f0, p0, ph = horn_score(x[0], config.FS)
        is_horn = bool(p0 >= 20.0 and ph >= 6.3)

        heat = None
        counter += 1
        if want_heat[0] and counter % 4 == 0:
            axis, P = srp_phat_map(x, mics, config.FS, grid=21)
            heat = (axis, P)

        with shared.lock:
            shared.rms = rms
            shared.resid = info["residual_rms_m"]
            shared.f0 = f0
            shared.horn = is_horn
            shared.dsp_ms = (time.perf_counter() - t0) * 1000.0
            if info["residual_rms_m"] < config.RESID_GATE:
                shared.hist.append(u)
                m = np.median(np.array(shared.hist), axis=0)
                shared.u = m / np.linalg.norm(m)
                shared.live = True
            else:
                shared.live = False
            if heat is not None:
                shared.heat_axis, shared.heat = heat


def draw_marker(frame, p, colour, label):
    h, w = frame.shape[:2]
    x, y = int(round(p[0])), int(round(p[1]))
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(frame, (x, y), 36, colour, 3)
        cv2.circle(frame, (x, y), 4, colour, -1)
        cv2.line(frame, (x - 52, y), (x - 22, y), colour, 2)
        cv2.line(frame, (x + 22, y), (x + 52, y), colour, 2)
        cv2.line(frame, (x, y - 52), (x, y - 22), colour, 2)
        cv2.line(frame, (x, y + 22), (x, y + 52), colour, 2)
        cv2.putText(frame, label, (x - 70, y - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    else:
        # Source is real but outside the lens's field of view. Draw an
        # arrow at the frame edge instead of silently dropping it -- the
        # C270 sees far less than the array hears, and pretending
        # otherwise looks like a bug.
        cx, cy = w // 2, h // 2
        vx, vy = x - cx, y - cy
        n = max(1e-6, np.hypot(vx, vy))
        s = min((w / 2 - 60) / abs(vx) if vx else 1e9,
                (h / 2 - 60) / abs(vy) if vy else 1e9)
        ex, ey = int(cx + vx * s), int(cy + vy * s)
        cv2.arrowedLine(frame, (int(cx + vx / n * 60), int(cy + vy / n * 60)),
                        (ex, ey), colour, 3, tipLength=0.25)
        cv2.putText(frame, "OFF-SCREEN " + label, (ex - 150, ey - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


def overlay_heat(frame, axis, P, R, K, dist):
    lo, hi = float(P.min()), float(P.max())
    if hi - lo < 1e-9:
        return frame
    norm = (P - lo) / (hi - lo)
    norm[norm < 0.70] = 0.0  # show a hot spot, not a wash
    small = (norm * 255).astype(np.uint8)

    h, w = frame.shape[:2]
    canvas = np.zeros((h, w), np.uint8)
    g = len(axis)
    for r in range(g):
        for c in range(g):
            if small[r, c] == 0:
                continue
            az, el = np.radians(axis[c]), np.radians(axis[r])
            u = np.array([np.sin(az) * np.cos(el), np.sin(el),
                          np.cos(az) * np.cos(el)])
            p = direction_to_pixel(R @ u, K, dist)
            if p is None:
                continue
            x, y = int(p[0]), int(p[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(canvas, (x, y), 28, int(small[r, c]), -1)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 15)
    heat = cv2.applyColorMap(canvas, cv2.COLORMAP_JET)
    heat[canvas < 20] = 0
    return cv2.addWeighted(frame, 1.0, heat, 0.45, 0)


def main():
    cal = json.load(open(config.INTRINSICS_FILE))
    K = np.array(cal["K"])
    dist = np.array(cal["dist"])
    try:
        R = np.array(json.load(open(config.EXTRINSICS_FILE))["R"])
    except FileNotFoundError:
        print("No array-to-camera calibration found; assuming the camera is "
              "perfectly aligned with the array (R = identity).")
        R = np.eye(3)

    mics = mic_positions()
    shared = Shared()
    want_heat = [False]
    horn_only = [False]

    src = TeensySource().start()
    threading.Thread(target=audio_worker,
                     args=(shared, src, mics, want_heat), daemon=True).start()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_H)

    os.makedirs("docs", exist_ok=True)
    t_prev, fps = time.time(), 0.0
    shot = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.03)
            continue

        with shared.lock:
            u, live, horn = shared.u, shared.live, shared.horn
            rms, resid, f0 = shared.rms, shared.resid, shared.f0
            dsp_ms = shared.dsp_ms
            heat = shared.heat
            axis = shared.heat_axis

        show = live and (horn or not horn_only[0])

        if want_heat[0] and heat is not None and show:
            frame = overlay_heat(frame, axis, heat, R, K, dist)

        if u is not None:
            p = direction_to_pixel(R @ u, K, dist)
            if p is not None:
                az, el = angles_from_unit(u)
                draw_marker(frame, p, RED if show else GREY,
                            f"az {az:+.0f} el {el:+.0f}")

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(1e-6, now - t_prev)
        t_prev = now

        status = ("TRACKING" if show else
                  ("waiting for horn" if live else "idle"))
        hud = [
            f"{status}  rms {rms:.4f}  residual {resid * 1000:5.1f} mm",
            f"f0 {f0:4.0f} Hz  horn {'yes' if horn else 'no '}  "
            f"dsp {dsp_ms:4.0f} ms  video {fps:4.1f} fps",
            f"[h] horn-gate {'ON ' if horn_only[0] else 'OFF'}  "
            f"[m] heatmap {'ON ' if want_heat[0] else 'OFF'}  "
            f"[s] shot  [q] quit",
        ]
        for i, line in enumerate(hud):
            cv2.putText(frame, line, (12, 28 + 26 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        GREEN if i == 0 and show else WHITE, 2)

        cv2.imshow("Acoustic Camera", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q") or k == 27:
            break
        elif k == ord("h"):
            horn_only[0] = not horn_only[0]
        elif k == ord("m"):
            want_heat[0] = not want_heat[0]
        elif k == ord("r"):
            with shared.lock:
                shared.hist.clear()
        elif k == ord("s"):
            cv2.imwrite(f"docs/screenshot_{shot:03d}.png", frame)
            print(f"saved docs/screenshot_{shot:03d}.png")
            shot += 1

    shared.stop = True
    src.stop()
    cap.release()
    cv2.destroyAllWindows()
    print(src.health())


if __name__ == "__main__":
    main()

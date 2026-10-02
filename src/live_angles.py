"""Live azimuth and elevation, printed as text. No camera involved.

    python src/live_angles.py

This is the milestone that proves the acoustics work. Get this right
BEFORE you touch the camera, because once video is in the loop you have
two things that can be wrong instead of one.

Reading the output:

  az / el      the estimated direction, in degrees
  resid        how well the six pairwise delays agree with a single plane
               wave, in millimetres. A clean single source in a quiet room
               gives a few mm. Large values mean two sources, a strong
               echo, or a broken channel. Watch this number -- it is your
               best single health indicator.
  q            correlation peak sharpness. Low means there is no real signal.
  horn         whether the harmonic detector thinks this is a horn.
"""

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from array_geometry import mic_positions, angles_from_unit
from doa import estimate_doa, horn_score
from teensy_source import TeensySource


def main():
    mics = mic_positions()
    print("microphone positions in channel order (metres):")
    for i, m in enumerate(mics):
        print(f" ch{i} -> {config.CHANNEL_LABELS[i]} "
              f"x={m[0]:+.4f} y={m[1]:+.4f}")
    print()

    src = TeensySource().start()
    try:
        while True:
            block = src.get_latest(config.BLOCK)
            if block is None:
                time.sleep(0.05)
                continue

            x = block.astype(float)
            rms = float(np.sqrt(np.mean(x ** 2)))
            if rms < config.RMS_GATE:
                print(f"\rsilence  rms {rms:.5f}                                                                  ",
                      end="", flush=True)
                time.sleep(0.05)
                continue

            u, info = estimate_doa(x, mics, config.FS,
                                   temp_c=config.TEMP_C, return_detail=True)
            az, el = angles_from_unit(u)
            f0, p0, ph = horn_score(x[0], config.FS)
            horn = "HORN" if (p0 >= 20 and ph >= 6.3) else "    "

            flag = "CLIP" if info["clipped_to_horizon"] else "    "
            print(f"\raz {az:+7.2f}  el {el:+7.2f}  "
                  f"resid {info['residual_rms_m'] * 1000:6.2f} mm  "
                  f"q {info['peak_quality']:6.1f}  rms {rms:.4f}  "
                  f"{horn}  f0 {f0:5.0f} Hz  {flag}",
                  end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()
        print("\n" + src.health())


if __name__ == "__main__":
    main()

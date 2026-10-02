"""Live level meter for all four channels. Run this constantly while
building. It is the single most useful diagnostic in the project.

    python src/check_channels.py

What to look for:

  1. All four bars move when you make a noise.
     A permanently empty bar = that microphone is dead, mis-wired, or its
     L/R pin is on the wrong rail.

  2. Bars are roughly the same length for a sound in the middle.
     One channel 20 dB down means a bad solder joint.

  3. THE TAP TEST. Scratch a fingernail on top of ONE microphone. Exactly
     one bar should jump. That tells you which Teensy channel that physical
     microphone is. Write the channel number on the board with a marker,
     then repeat for the other three.

  4. 'lost' and 'resync' counters should stay at 0. If they climb, your
     USB cable or your computer is dropping data.
"""

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from teensy_source import TeensySource


def bar(value, width=40, full=0.25):
    n = int(np.clip(value / full, 0, 1) * width)
    return "#" * n + "." * (width - n)


def main():
    src = TeensySource()
    try:
        src.start()
    except Exception as exc:
        print(f"Could not open {config.SERIAL_PORT}: {exc}")
        print("Run python src/teensy_source.py to list available ports,")
        print("then put the right one in src/config.py.")
        return

    print(f"Reading {config.SERIAL_PORT} at {config.FS} Hz. Ctrl-C to stop.\n")
    peaks = np.zeros(config.N_CHANNELS)
    try:
        while True:
            block = src.get_latest(2048)
            if block is None:
                time.sleep(0.05)
                continue
            rms = np.sqrt(np.mean(block.astype(float) ** 2, axis=1))
            peaks = np.maximum(peaks * 0.90, rms)

            lines = []
            loudest = int(np.argmax(rms))
            for i in range(config.N_CHANNELS):
                mark = " <== LOUDEST" if i == loudest and rms[i] > 0.004 else ""
                dead = " *** SILENT ***" if rms[i] < 1e-5 else ""
                lines.append(f"ch{i} [{bar(peaks[i])}] rms {rms[i]:.5f}{dead}{mark}")
            lines.append(src.health())

            sys.stdout.write("\033[H\033[J" + "\n".join(lines) + "\n")
            sys.stdout.flush()
            time.sleep(0.08)
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()
        print("\nstopped.")


if __name__ == "__main__":
    main()

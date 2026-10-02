"""Record a 4-channel WAV file from the Teensy.

    python src/record_wav.py recordings/test.wav 5

Recording to a file matters more than it sounds. Once you have a WAV you
can re-run your algorithm on the same data a hundred times while changing
parameters, which is how you produce a results table. Live-only debugging
gives you no repeatability and therefore no report.
"""

import sys
import os
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from teensy_source import TeensySource


def record(path, seconds):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    src = TeensySource(capacity_s=seconds + 2.0)
    src.start()
    n = int(seconds * config.FS)
    print(f"Recording {seconds:.1f} s ...")
    time.sleep(seconds + 0.5)
    block = src.get_latest(n)
    src.stop()

    if block is None:
        print("No data captured. Check the Teensy is running four_mic_stream.")
        return

    # WAV wants interleaved samples: ch0 ch1 ch2 ch3 ch0 ch1 ...
    inter = (np.clip(block.T, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(config.N_CHANNELS)
        w.setsampwidth(2)
        w.setframerate(config.FS)
        w.writeframes(inter.tobytes())

    rms = np.sqrt(np.mean(block.astype(float) ** 2, axis=1))
    print(f"Saved {path}")
    print("per-channel RMS:", " ".join(f"{r:.5f}" for r in rms))
    print(src.health())


def load(path):
    """Read a 4-channel WAV back as a (4, n) float array."""
    with wave.open(path, "rb") as w:
        nch, fs, n = w.getnchannels(), w.getframerate(), w.getnframes()
        raw = np.frombuffer(w.readframes(n), dtype="<i2")
    return raw.reshape(-1, nch).T.astype(float) / 32768.0, fs


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "recordings/test.wav"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    record(out, secs)

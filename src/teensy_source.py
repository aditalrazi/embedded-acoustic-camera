"""Reads the 4-channel audio stream coming out of the Teensy over USB.

The Teensy sends a continuous run of fixed-size frames:

    byte 0     0xAA  |
    byte 1     0x55  |  4-byte header
    byte 2-3   sequence number, unsigned 16-bit, little-endian
    byte 4..   128 samples of channel 0, int16, little-endian (256 bytes)
               128 samples of channel 1 (256 bytes)
               128 samples of channel 2 (256 bytes)
               128 samples of channel 3 (256 bytes)

    total      1028 bytes per frame, 344.5 frames per second

Two details in here are not optional, and both exist because of the same
failure: USB is a byte stream with no message boundaries. If you lose a
single byte, everything after it shifts, and your four channels silently
stop lining up. The symptom is "it worked yesterday, today the angles are
nonsense", which is a horrible thing to debug at 2 a.m. in week 5.

    * The 0xAA 0x55 marker lets the reader find the start of a frame again
      after any glitch, automatically.
    * The sequence number lets you SEE dropped frames instead of guessing.
"""

import threading
import time

import numpy as np

try:
    import serial    # pip install pyserial
    import serial.tools.list_ports
except ImportError:  # pragma: no cover
    serial = None

import config

SYNC = b"\xAA\x55"
HEADER = 4
BLOCK = config.TEENSY_BLOCK
NCH = config.N_CHANNELS
FRAME_BYTES = HEADER + BLOCK * 2 * NCH  # 4 + 1024 = 1028


def list_ports():
    """Print every serial port on this computer. Run this if you do not
    know what to put in config.SERIAL_PORT."""
    if serial is None:
        print("pyserial is not installed: pip install pyserial")
        return []
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found. Is the Teensy plugged in with a DATA "
              "cable (not a charge-only cable)?")
    for p in ports:
        print(f" {p.device:20s} {p.description}")
    return ports


class TeensySource:
    """Background reader that keeps the most recent few seconds of audio.

    Usage:
        src = TeensySource()
        src.start()
        block = src.get_latest(4096) # -> (4, 4096) float32, or None
        src.stop()
    """

    def __init__(self, port=None, capacity_s=4.0, fs=config.FS):
        if serial is None:
            raise RuntimeError("pyserial is not installed: pip install pyserial")
        self.port_name = port or config.SERIAL_PORT
        self.fs = fs
        self.cap = int(capacity_s * fs)
        self._buf = np.zeros((NCH, self.cap), dtype=np.float32)
        self._w = 0       # write cursor
        self._filled = 0  # how many samples ever written
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.frames_ok = 0
        self.frames_lost = 0
        self.resyncs = 0
        self._last_seq = None
        self.ser = None

    # -- lifecycle ------------------------------------------------------
    def start(self):
        self.ser = serial.Serial(self.port_name, 2000000, timeout=0.2)
        # The Teensy ignores the baud rate -- USB always runs at full speed.
        # It is required by pyserial's API, that is all.
        time.sleep(0.3)
        self.ser.reset_input_buffer()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- reader thread --------------------------------------------------
    def _run(self):
        raw = b""
        while not self._stop.is_set():
            chunk = self.ser.read(max(1, min(self.ser.in_waiting, 65536)))
            if not chunk:
                continue
            raw += chunk

            while True:
                i = raw.find(SYNC)
                if i < 0:  # no marker anywhere yet
                    raw = raw[-1:]
                    break
                if len(raw) - i < FRAME_BYTES:  # marker found, frame incomplete
                    raw = raw[i:]
                    break
                if i != 0:
                    self.resyncs += 1
                frame = raw[i:i + FRAME_BYTES]
                raw = raw[i + FRAME_BYTES:]
                self._consume(frame)

    def _consume(self, frame):
        seq = int.from_bytes(frame[2:4], "little")
        if self._last_seq is not None:
            gap = (seq - self._last_seq) & 0xFFFF
            if gap != 1:
                self.frames_lost += max(0, gap - 1)
        self._last_seq = seq

        data = np.frombuffer(frame[HEADER:], dtype="<i2").reshape(NCH, BLOCK)
        data = data.astype(np.float32) / 32768.0

        with self._lock:
            w = self._w
            end = w + BLOCK
            if end <= self.cap:
                self._buf[:, w:end] = data
            else:  # wrap around
                k = self.cap - w
                self._buf[:, w:] = data[:, :k]
                self._buf[:, :end - self.cap] = data[:, k:]
            self._w = end % self.cap
            self._filled += BLOCK
        self.frames_ok += 1

    # -- consumer API ---------------------------------------------------
    def get_latest(self, n):
        """Most recent n samples as a (4, n) float32 array, or None if the
        buffer has not filled up yet."""
        n = int(n)
        with self._lock:
            if self._filled < n or n > self.cap:
                return None
            start = (self._w - n) % self.cap
            if start + n <= self.cap:
                return self._buf[:, start:start + n].copy()
            k = self.cap - start
            out = np.empty((NCH, n), dtype=np.float32)
            out[:, :k] = self._buf[:, start:]
            out[:, k:] = self._buf[:, :n - k]
            return out

    def wait_for_data(self, n, timeout=5.0):
        """Block until n samples are available. Returns them, or None."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            b = self.get_latest(n)
            if b is not None:
                return b
            time.sleep(0.02)
        return None

    def health(self):
        return (f"frames ok {self.frames_ok} lost {self.frames_lost} "
                f"resyncs {self.resyncs}")


if __name__ == "__main__":
    list_ports()

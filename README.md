# Embedded Acoustic Camera

Our EEE 416 final project at BUET (Section B1, Group 3). Four MEMS microphones on a Teensy 4.0 listen for a car horn, a laptop works out which direction it came from, and a marker appears on the live webcam video at that spot.

Demo video: https://youtu.be/lWSrFJojGBg

## How it works

The Teensy drives all four INMP441 microphones from a single I²S clock, so every channel is sampled at the same instant. It streams the four channels to the laptop over USB (44.1 kHz, 16-bit). On the laptop, Python finds the time delay for each of the six mic pairs with band-limited GCC-PHAT. It then solves for the direction by least squares and draws it on the C270 image, using a one-time camera calibration. A harmonic check makes the marker follow horns rather than any loud noise, and there is an SRP-PHAT heat map you can switch on.

In simulation the 12 cm array gives about 2° mean error. On our laptop the processing took about 12 to 15 ms for every 93 ms of audio, so it keeps up in real time.

## Hardware

- Teensy 4.0
- 4 × INMP441 microphone modules at the corners of a 12 cm square on a rigid plate
- Logitech C270 webcam in the middle of the plate
- a micro-USB **data** cable (a charge-only cable won't show up as a port)

| Mic pin | Goes to |
|---|---|
| VDD | 3.3V only. The Teensy 4.0 pins are not 5 V tolerant |
| GND | GND |
| SCK | pin 21 (all four mics) |
| WS | pin 20 (all four mics) |
| SD | mics A and B to pin 8, mics C and D to pin 6 |
| L/R | A and C to GND, B and D to 3.3V |

## Running it

1. **Firmware.** Install the Arduino IDE with Teensyduino. Upload `firmware/test_mics/test_mics.ino` first and check in the Serial Monitor (115200 baud) that all four bars move. Then upload `firmware/four_mic_stream/four_mic_stream.ino`, which streams the audio.
2. **Python** (3.10 or newer):
   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # macOS / Linux
   pip install -r requirements.txt
   pytest tests/ -q
   ```
   All 22 tests should pass. They use simulated signals, so you don't need the hardware for this step.
3. **Settings.** Everything is in `src/config.py`. Set `SERIAL_PORT` (`python src/teensy_source.py` lists the ports), `CAMERA_INDEX` and `TEMP_C`. Put your measured mic spacing in `ARRAY_SIDE_M`.
4. **Check the mics.** Run `python src/check_channels.py`. Scratch one mic at a time (the tap test) and write the order you find into `CHANNEL_LABELS`. Then `python src/live_angles.py` should print sensible angles while a horn plays.
5. **Calibrate.** Run `python src/capture_calib_images.py` and `python src/calibrate_camera.py` with a printed chessboard (9 × 6 inner corners). Then run `python src/align_array_camera.py` with at least 8 speaker positions spread over the picture.
6. **Demo.** Run `python src/live.py`. Keys: `q` quit, `h` horns only, `m` heat map, `s` screenshot, `r` reset the smoothing.

Run the scripts from the repo root, because they read and write `data/` and `docs/` relative to it. `python tests/sweep.py` rebuilds the simulation tables (takes a few minutes), and `python src/measure_accuracy.py` walks you through an accuracy test.

## A few things we learned

- If the marker moves the wrong way, the channel map is wrong. Redo the tap test and fix `CHANNEL_LABELS` rather than re-soldering.
- The array hears about ±40°, but the C270 only sees about ±25°. A source outside the picture gets an OFF-SCREEN arrow at the edge.
- With four mics in one plane, it tracks one source at a time and can't tell front from back.

## Credits

Most of this code comes from the *Embedded Acoustic Camera: Teensy 4.0 Build Manual*, which we followed for the build. It uses the PJRC Teensy Audio Library, NumPy and OpenCV. The test horns were Freesound recordings.

## Team

Md Ihasan Chowdhury Leyon (2106072), Tahmid Adib (2106089), Adit Al Razi (2106090), Rafin Al Nahian (2106091)

Course teachers: Dr. Zabir Ahmed and Sadman Sakib Ahbab, Department of EEE, BUET

"""ONE place for every number you are allowed to change.

Every other file imports from here. If you find yourself editing a number
in a different file, stop -- move it here instead. When something breaks in
week 5, you want exactly one file to check.
"""

# ----------------------------------------------------------------------
# 1. SERIAL PORT -- where the Teensy appears on your computer
# ----------------------------------------------------------------------
# Windows : "COM3", "COM4", ...  (Device Manager -> Ports)
# Linux   : "/dev/ttyACM0"
# macOS   : "/dev/cu.usbmodem12345601"
SERIAL_PORT = "COM3"

# ----------------------------------------------------------------------
# 2. AUDIO -- fixed by the Teensy Audio Library. Do not change.
# ----------------------------------------------------------------------
FS = 44100          # samples per second, per channel
N_CHANNELS = 4
TEENSY_BLOCK = 128  # samples per channel in one Teensy audio block

# Analysis window. 4096 samples = 92.9 ms of audio.
# Bigger = more accurate, more lag. Smaller = snappier, noisier.
BLOCK = 4096

# ----------------------------------------------------------------------
# 3. ARRAY GEOMETRY
# ----------------------------------------------------------------------
# Nominal square side in metres. AFTER you build the array, MEASURE the
# real corner-to-corner spacing with calipers and put the true number here.
ARRAY_SIDE_M = 0.120

# Which physical microphone does each Teensy channel correspond to?
#
# Physical labels, looking at the array FROM BEHIND (i.e. standing where
# the camera stands, facing the scene):
#
#     D ----------- C        A = bottom-left
#     |             |        B = bottom-right
#     |      +      |        C = top-right
#     |             |        D = top-left
#     A ----------- B
#
# You will discover the true mapping with the tap test in Part 6 of the
# manual. Until you have done that, this is a guess and your angles will
# be wrong. CHANNEL_LABELS[i] = the physical mic that Teensy channel i sees.
CHANNEL_LABELS = ["A", "B", "C", "D"]

# Corner coordinates in metres, relative to the array centre.
# +x = to the RIGHT as seen from behind the array
# +y = UP
# +z = out of the plate, towards the scene (where the camera looks)
def corner_positions(side_m=None):
    s = (ARRAY_SIDE_M if side_m is None else side_m) / 2.0
    return {
        "A": (-s, -s, 0.0),
        "B": (+s, -s, 0.0),
        "C": (+s, +s, 0.0),
        "D": (-s, +s, 0.0),
    }

# ----------------------------------------------------------------------
# 4. ENVIRONMENT
# ----------------------------------------------------------------------
TEMP_C = 30.0  # your room temperature in Celsius. Dhaka lab: 25-32.

# ----------------------------------------------------------------------
# 5. SIGNAL PROCESSING
# ----------------------------------------------------------------------
BAND = (300.0, 2500.0)  # Hz. Car-horn band. Biggest single accuracy win.
BETA = 0.7              # 0 = plain correlation, 1 = full PHAT, 0.7 = best
INTERP = 16             # correlation up-sampling factor

RMS_GATE = 0.004    # below this input level, assume silence
RESID_GATE = 0.012  # metres. Above this, the four mics disagree -> reject
SMOOTH_N = 5        # median filter length over recent estimates

# ----------------------------------------------------------------------
# 6. CAMERA
# ----------------------------------------------------------------------
CAMERA_INDEX = 0     # 0 is usually the built-in webcam, 1 the C270.
CAM_W, CAM_H = 1280, 720
CHESSBOARD = (9, 6)  # INNER corners, not squares
SQUARE_MM = 25.0     # MEASURE your printed squares with a ruler

INTRINSICS_FILE = "data/camera_intrinsics.json"
EXTRINSICS_FILE = "data/array_to_camera.json"

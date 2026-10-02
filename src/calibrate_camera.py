"""Work out the camera's focal length, centre point and lens distortion.

    python src/calibrate_camera.py

This is a completely standard, solved problem. Do not invent anything
here -- just run it and check the reprojection error.

  reprojection RMS < 0.5 px good
                   < 1.0 px acceptable
                   > 1.0 px recapture: flatter board, more varied angles

Sanity check on the focal length: for a 1280-pixel-wide image,
fx is approximately 640 / tan(horizontal_FOV / 2). A Logitech C270 has
roughly a 50 degree horizontal field of view at 4:3, so expect fx to land
somewhere around 1200-1450 px. If you get 300 or 6000, something is wrong.
"""

import sys
import os
import glob
import json

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def main():
    pattern = config.CHESSBOARD
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp *= config.SQUARE_MM  # printers rescale -- MEASURE your squares

    obj_points, img_points = [], []
    size = None

    files = sorted(glob.glob("data/calib/*.jpg"))
    if not files:
        print("No images in data/calib/. Run capture_calib_images.py first.")
        return

    for path in files:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
        if not found:
            print(f" no board found in {os.path.basename(path)}")
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        obj_points.append(objp)
        img_points.append(corners)

    print(f"using {len(obj_points)} of {len(files)} images")
    if len(obj_points) < 8:
        print("Fewer than 8 usable images. Capture more before trusting this.")
        if not obj_points:
            return

    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, size, None, None)

    print(f"\nreprojection RMS : {rms:.3f} px")
    print(f"fx, fy           : {K[0, 0]:.1f}, {K[1, 1]:.1f}")
    print(f"cx, cy           : {K[0, 2]:.1f}, {K[1, 2]:.1f}  "
          f"(image centre is {size[0] / 2:.0f}, {size[1] / 2:.0f})")
    hfov = 2 * np.degrees(np.arctan(size[0] / (2 * K[0, 0])))
    vfov = 2 * np.degrees(np.arctan(size[1] / (2 * K[1, 1])))
    print(f"implied field of view: {hfov:.1f} deg horizontal, "
          f"{vfov:.1f} deg vertical")
    print(f" -> your marker is only meaningful within +/-{hfov / 2:.0f} deg "
          f"azimuth and +/-{vfov / 2:.0f} deg elevation")

    os.makedirs("data", exist_ok=True)
    json.dump({"K": K.tolist(), "dist": dist.tolist(), "rms": float(rms),
               "image_size": list(size)},
              open(config.INTRINSICS_FILE, "w"), indent=2)
    print(f"\nwritten to {config.INTRINSICS_FILE}")


if __name__ == "__main__":
    main()

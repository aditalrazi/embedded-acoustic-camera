"""Capture chessboard photos for camera calibration.

    python src/capture_calib_images.py

Press SPACE to save a frame, Q to quit. The window shows a green overlay
when it can see the whole board -- only save frames where it does.

Get 20-30 images with the board:
  * near and far
  * tilted left, right, up, down
  * in the CORNERS of the frame, not just the middle

Corner coverage is the part everyone skips, and it is what determines
whether the lens distortion coefficients are meaningful.
"""

import sys
import os

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

OUT = "data/calib"


def main():
    os.makedirs(OUT, exist_ok=True)
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_H)

    if not cap.isOpened():
        print(f"Cannot open camera index {config.CAMERA_INDEX}. "
              f"Try 0, 1, 2 in src/config.py.")
        return

    n = len([f for f in os.listdir(OUT) if f.endswith(".jpg")])
    print("SPACE = save    Q = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, config.CHESSBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)

        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, config.CHESSBOARD, corners, found)
        cv2.putText(view, f"saved {n}   board {'FOUND' if found else 'not found'}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("calibration capture", view)

        k = cv2.waitKey(1) & 0xFF
        if k == ord(" ") and found:
            cv2.imwrite(f"{OUT}/calib_{n:03d}.jpg", frame)
            n += 1
            print(f"saved {n}")
        elif k in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

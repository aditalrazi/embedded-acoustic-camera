"""Find the rotation between the microphone array and the camera.

    python src/align_array_camera.py

You now have two ways of describing a direction: the array says
u_array = (ux, uy, uz), the camera says "that pixel". These agree only if
the camera is mounted perfectly square to the plate, which it never is.
The fix is to MEASURE the 3x3 rotation R that maps one into the other:

    u_camera = R @ u_array

How this script works. For each of at least eight speaker positions:
  1. put the speaker somewhere in view and play a horn loop
  2. press SPACE -- it grabs a frame and records a second of audio
  3. click on the speaker in the frozen frame
  4. repeat, spreading the positions out: left, right, up, down, corners

Then it solves for the R that best maps all the acoustic directions onto
all the clicked directions. This is the orthogonal Procrustes problem and
it has an exact closed-form answer via the SVD -- no optimiser, no
starting guess. It is a genuinely nice bit of linear algebra and it is
worth a paragraph in your report, because this step is precisely the
"camera-to-acoustic-array calibration" the brief asks for and it is where
most student projects wave their hands.

IMPORTANT: spread the positions out. Eight points clustered near the
middle of the frame give a badly conditioned fit and a rotation that
looks fine on your calibration data and fails everywhere else.
"""

import sys
import os
import json
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from array_geometry import mic_positions, angles_from_unit, angular_error_deg
from doa import estimate_doa
from teensy_source import TeensySource


def pixel_to_direction(px, py, K, dist):
    """A pixel -> a unit direction in the camera's own frame."""
    pts = np.array([[[float(px), float(py)]]], dtype=np.float32)
    und = cv2.undistortPoints(pts, K, dist)  # normalised image coords
    v = np.array([float(und[0, 0, 0]), float(und[0, 0, 1]), 1.0])
    return v / np.linalg.norm(v)


def direction_to_pixel(u_cam, K, dist):
    """A unit direction in the camera frame -> a pixel. Returns None if
    the direction points behind the camera."""
    if u_cam[2] <= 1e-6:
        return None
    pts = np.array([[u_cam[0], u_cam[1], u_cam[2]]], dtype=np.float32)
    img, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, dist)
    return float(img[0, 0, 0]), float(img[0, 0, 1])


def solve_rotation(u_array_list, u_cam_list):
    """Closed-form least-squares rotation (orthogonal Procrustes / Kabsch).

    Find R minimising sum_i || R a_i - b_i ||^2 subject to R being a
    proper rotation. The answer is R = V diag(1,1,d) U^T where U S V^T is
    the SVD of A B^T, and d = sign(det(V U^T)) exists purely to stop the
    solver from returning a mirror image instead of a rotation.
    """
    A = np.asarray(u_array_list, dtype=float).T  # 3 x N
    B = np.asarray(u_cam_list, dtype=float).T    # 3 x N
    H = A @ B.T
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


class Collector:
    def __init__(self):
        cal = json.load(open(config.INTRINSICS_FILE))
        self.K = np.array(cal["K"])
        self.dist = np.array(cal["dist"])
        self.mics = mic_positions()
        self.pairs = []
        self.click = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)

    def run(self):
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_H)

        src = TeensySource().start()
        cv2.namedWindow("align")
        cv2.setMouseCallback("align", self.on_mouse)
        print("SPACE = capture a point, U = undo last, S = solve and save, Q = quit")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                view = frame.copy()
                cv2.putText(view, f"points: {len(self.pairs)} (need >= 8)",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
                cv2.imshow("align", view)
                k = cv2.waitKey(1) & 0xFF

                if k == ord(" "):
                    print("recording 1.0 s of audio ...")
                    time.sleep(1.0)
                    block = src.get_latest(config.FS)
                    if block is None:
                        print(" no audio -- is the Teensy streaming?")
                        continue
                    u_arr = estimate_doa(block.astype(float), self.mics,
                                         config.FS, temp_c=config.TEMP_C)
                    az, el = angles_from_unit(u_arr)
                    print(f" acoustic direction: az {az:+.1f} el {el:+.1f}")
                    print(" now CLICK the speaker in the window")

                    self.click = None
                    frozen = frame.copy()
                    while self.click is None:
                        cv2.imshow("align", frozen)
                        if (cv2.waitKey(20) & 0xFF) == ord("q"):
                            break
                    if self.click is None:
                        continue
                    px, py = self.click
                    u_cam = pixel_to_direction(px, py, self.K, self.dist)
                    self.pairs.append((u_arr.tolist(), u_cam.tolist(),
                                       [px, py]))
                    print(f" clicked pixel ({px}, {py}) -> "
                          f"point {len(self.pairs)} stored\n")

                elif k == ord("u") and self.pairs:
                    self.pairs.pop()
                    print(f"undone, {len(self.pairs)} left")

                elif k == ord("s"):
                    self.solve()

                elif k in (ord("q"), 27):
                    break
        finally:
            src.stop()
            cap.release()
            cv2.destroyAllWindows()

    def solve(self):
        if len(self.pairs) < 4:
            print("need at least 4 points, 8 or more strongly preferred")
            return
        A = [p[0] for p in self.pairs]
        B = [p[1] for p in self.pairs]
        R = solve_rotation(A, B)

        errs = [angular_error_deg(R @ np.array(a), np.array(b))
                for a, b in zip(A, B)]
        errs = np.array(errs)
        print("\n--- alignment result ---")
        print("R =\n", np.round(R, 4))
        print(f"residual angular error: mean {errs.mean():.2f} deg, "
              f"max {errs.max():.2f} deg, n = {len(errs)}")
        if errs.mean() < 3.0:
            print("Good. Under 3 degrees is what you want.")
        elif errs.mean() < 8.0:
            print("Usable, but check for sloppy clicking or a noisy room.")
        else:
            print("Too large. Likely causes, in order: wrong CHANNEL_LABELS,")
            print("a sign convention flipped, or the acoustic estimates are")
            print("themselves bad. Fix live_angles.py before coming back here.")

        os.makedirs("data", exist_ok=True)
        json.dump({"R": R.tolist(),
                   "residual_mean_deg": float(errs.mean()),
                   "residual_max_deg": float(errs.max()),
                   "n_points": len(errs),
                   "points": self.pairs},
                  open(config.EXTRINSICS_FILE, "w"), indent=2)
        print(f"written to {config.EXTRINSICS_FILE}\n")


if __name__ == "__main__":
    Collector().run()

"""Geometry, coordinates and angle conventions.

Fix this convention in your head now and never change it:

    +x points RIGHT, as seen from behind the array looking at the scene
    +y points UP
    +z points OUT of the microphone plate, towards the scene
        (the camera also looks along +z)

A direction towards a sound source is a unit vector u = (ux, uy, uz)
with uz > 0, i.e. the source is somewhere in front of the plate.

    azimuth   = left/right angle off straight-ahead (+ = to the right)
    elevation = up/down angle off straight-ahead    (+ = upwards)
"""

import numpy as np

import config


def speed_of_sound(temp_c=20.0):
    """Speed of sound in dry air, metres per second.

    343 m/s is the 20 C textbook value. In a 32 C Dhaka lab it is 351 m/s,
    a 2.3 % difference, which becomes a 2.3 % error in every angle you
    report. Measure the room and pass the real number.
    """
    return 331.3 + 0.606 * temp_c


def mic_positions(channel_labels=None, side_m=None):
    """Microphone coordinates in Teensy CHANNEL order.

    Returns a (4, 3) array where row i is the position of whichever
    physical microphone Teensy channel i is listening to. This indirection
    is the whole point: you discover the mapping experimentally with the
    tap test, write it into config.CHANNEL_LABELS, and never have to
    re-solder anything.
    """
    labels = channel_labels or config.CHANNEL_LABELS
    corners = config.corner_positions(side_m)
    return np.array([corners[l] for l in labels], dtype=float)


def all_pairs(n_mics):
    """Every unordered pair of microphones. 4 mics -> 6 pairs."""
    return [(i, j) for i in range(n_mics) for j in range(i + 1, n_mics)]


def unit_from_angles(az_deg, el_deg):
    """(azimuth, elevation) in degrees -> unit direction vector."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([
        np.sin(az) * np.cos(el),
        np.sin(el),
        np.cos(az) * np.cos(el),
    ])


def angles_from_unit(u):
    """Unit direction vector -> (azimuth, elevation) in degrees."""
    az = np.degrees(np.arctan2(u[0], u[2]))
    el = np.degrees(np.arcsin(np.clip(u[1], -1.0, 1.0)))
    return az, el


def angular_error_deg(u_a, u_b):
    """The angle between two directions, in degrees.

    THIS is the error metric to report. Do not report azimuth error and
    elevation error as two separate numbers -- they interact, and one
    combined angle is what a reader actually wants to know.
    """
    d = float(np.dot(u_a, u_b) / (np.linalg.norm(u_a) * np.linalg.norm(u_b)))
    return np.degrees(np.arccos(np.clip(d, -1.0, 1.0)))


def max_delay_s(mic_a, mic_b, c):
    """Largest physically possible arrival-time difference for a mic pair."""
    return float(np.linalg.norm(np.asarray(mic_a) - np.asarray(mic_b))) / c


def alias_free_limit_hz(spacing_m, c=343.0):
    """Highest frequency this spacing can handle without spatial aliasing."""
    return c / (2.0 * spacing_m)


def square_array(side_m):
    """A plain 4-corner square in A, B, C, D order. Used by the simulator."""
    s = side_m / 2.0
    return np.array([
        [-s, -s, 0.0],
        [+s, -s, 0.0],
        [+s, +s, 0.0],
        [-s, +s, 0.0],
    ])

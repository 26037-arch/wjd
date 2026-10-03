"""Calibrated REP5X CAD transforms for RTCP consistency checks.

The caller supplies measured pivots and tip coordinates in a single machine
home coordinate frame. No hardware offset is inferred from the firmware's LB
and LC scalars alone.
"""

from __future__ import annotations

import math

import numpy as np


def _rz(degrees):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _ry(degrees):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def tip_from_carriage(carriage_xyz, b_deg, c_deg, c_pivot, b_pivot,
                      tip_local):
    """Apply C yaw about c_pivot, then B tilt about b_pivot in C's frame."""
    carriage = np.asarray(carriage_xyz, dtype=float)
    cp = np.asarray(c_pivot, dtype=float)
    bp = np.asarray(b_pivot, dtype=float)
    tip = np.asarray(tip_local, dtype=float)
    return carriage + cp + _rz(c_deg) @ (bp - cp + _ry(b_deg) @ (tip - bp))


def carriage_for_rtcp(tip_xyz, b_deg, c_deg, c_pivot, b_pivot, tip_local):
    """Translate the carriage so the CAD tip matches G-code RTCP XYZ."""
    return np.asarray(tip_xyz, dtype=float) - tip_from_carriage(
        (0, 0, 0), b_deg, c_deg, c_pivot, b_pivot, tip_local)


def rtcp_error(tip_xyz, b_deg, c_deg, c_pivot, b_pivot, tip_local):
    carriage = carriage_for_rtcp(tip_xyz, b_deg, c_deg, c_pivot,
                                 b_pivot, tip_local)
    rendered = tip_from_carriage(carriage, b_deg, c_deg, c_pivot,
                                 b_pivot, tip_local)
    return float(np.linalg.norm(rendered - np.asarray(tip_xyz, dtype=float)))


def tip_from_transforms(carriage_xyz, b_deg, c_deg, h_t_c, c_t_b, b_t_n):
    """Evaluate H_T_C Rz(C) C_T_B Ry(B) B_T_N in homogeneous coordinates."""
    matrices = [np.asarray(m, dtype=float) for m in (h_t_c, c_t_b, b_t_n)]
    if any(m.shape != (4, 4) or not np.allclose(m[3], [0, 0, 0, 1])
           for m in matrices):
        raise ValueError("Calibration requires three homogeneous 4x4 transforms")
    yaw = np.eye(4)
    tilt = np.eye(4)
    yaw[:3, :3] = _rz(c_deg)
    tilt[:3, :3] = _ry(b_deg)
    tip = matrices[0] @ yaw @ matrices[1] @ tilt @ matrices[2] @ [0, 0, 0, 1]
    return np.asarray(carriage_xyz, dtype=float) + tip[:3]


def carriage_for_rtcp_transforms(tip_xyz, b_deg, c_deg, h_t_c, c_t_b, b_t_n):
    return np.asarray(tip_xyz, dtype=float) - tip_from_transforms(
        (0, 0, 0), b_deg, c_deg, h_t_c, c_t_b, b_t_n)

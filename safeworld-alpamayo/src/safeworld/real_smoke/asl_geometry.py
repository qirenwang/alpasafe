"""Coordinate transforms + alignment metrics for ASL-derived targets (T25A, v2.1).

Frames (AlpaSim ``CONTRIBUTING.md`` "Coordinate Systems"):
  * ``local`` — inertial ENU frame, right-handed, fixed per scenario (the global
    frame). For this scene it is anchored at the ego ``rig`` pose at scene start.
  * ``rig``   — body-fixed, x-forward, y-left, z-up, origin at rear axle on ground.
  * ``aabb``  — same orientation as ``rig``; origin at bounding-box centre.
    ``ActorPoses`` log entries store every actor as ``local→aabb`` (verified).
  * ``ego_t0_rig`` — the ego ``rig`` frame at the candidate decision time ``t0``.

Quaternion order is **wxyz**; rotations are right-handed; yaw is about +Z and uses
the same ``atan2`` convention as the AlpaSim driver/runtime so transforms round-trip
against the real log.

Nothing here fabricates data: callers pass raw poses parsed from the ``.asl`` log.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "quat_wxyz_to_yaw",
    "yaw_to_quat_wxyz",
    "rot2d",
    "aabb_xy_to_rig_xy",
    "rig_xy_to_aabb_xy",
    "local_xy_to_ego_t0",
    "ego_t0_to_local_xy",
    "ade_fde",
    "finite_difference_velocity",
]


def quat_wxyz_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """Yaw about +Z from a wxyz quaternion (matches AlpaSim ``_quat_to_yaw``)."""
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def yaw_to_quat_wxyz(yaw: float) -> list[float]:
    """wxyz quaternion for a pure +Z (yaw) rotation."""
    return [float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0))]


def rot2d(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=float)


def aabb_xy_to_rig_xy(aabb_xy: np.ndarray, yaw: np.ndarray, rig_to_aabb_x: float) -> np.ndarray:
    """``rig`` xy from logged ``aabb`` xy: subtract the yaw-rotated forward offset.

    The rig→aabb transform is a pure forward+up translation in the body frame with
    identity rotation, so ``aabb_xy = rig_xy + R(yaw)·[rig_to_aabb_x, 0]``.
    ``aabb_xy`` is ``(N,2)``; ``yaw`` is ``(N,)``.
    """
    aabb_xy = np.asarray(aabb_xy, dtype=float).reshape(-1, 2)
    yaw = np.asarray(yaw, dtype=float).reshape(-1)
    fwd = np.stack([np.cos(yaw), np.sin(yaw)], axis=1) * float(rig_to_aabb_x)
    return aabb_xy - fwd


def rig_xy_to_aabb_xy(rig_xy: np.ndarray, yaw: np.ndarray, rig_to_aabb_x: float) -> np.ndarray:
    """Inverse of :func:`aabb_xy_to_rig_xy`."""
    rig_xy = np.asarray(rig_xy, dtype=float).reshape(-1, 2)
    yaw = np.asarray(yaw, dtype=float).reshape(-1)
    fwd = np.stack([np.cos(yaw), np.sin(yaw)], axis=1) * float(rig_to_aabb_x)
    return rig_xy + fwd


def local_xy_to_ego_t0(local_xy: np.ndarray, x0: float, y0: float, yaw0: float) -> np.ndarray:
    """Express ``local``-frame xy in the ``ego_t0_rig`` frame (passive transform).

    ``p_ego_t0 = R(yaw0)^T · (p_local - [x0, y0])``.
    """
    local_xy = np.asarray(local_xy, dtype=float).reshape(-1, 2)
    return (local_xy - np.array([x0, y0], dtype=float)) @ rot2d(yaw0)


def ego_t0_to_local_xy(ego_xy: np.ndarray, x0: float, y0: float, yaw0: float) -> np.ndarray:
    """Inverse of :func:`local_xy_to_ego_t0` (the driver's anchoring transform)."""
    ego_xy = np.asarray(ego_xy, dtype=float).reshape(-1, 2)
    return ego_xy @ rot2d(yaw0).T + np.array([x0, y0], dtype=float)


def ade_fde(
    planned_xy: np.ndarray, executed_xy: np.ndarray, valid: np.ndarray | None = None
) -> dict[str, float | int]:
    """Average/Final/Max displacement error between two same-length xy paths.

    ``valid`` (optional) is a boolean mask of aligned steps; errors are computed on
    valid steps only. FDE is the error at the LAST valid step.
    """
    planned_xy = np.asarray(planned_xy, dtype=float).reshape(-1, 2)
    executed_xy = np.asarray(executed_xy, dtype=float).reshape(-1, 2)
    if planned_xy.shape != executed_xy.shape:
        raise ValueError(f"shape mismatch {planned_xy.shape} vs {executed_xy.shape}")
    if valid is None:
        valid = np.ones(planned_xy.shape[0], dtype=bool)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    err = np.linalg.norm(planned_xy - executed_xy, axis=1)
    err_v = err[valid]
    if err_v.size == 0:
        return {"ade_m": float("nan"), "fde_m": float("nan"), "max_error_m": float("nan"),
                "n_aligned_steps": 0}
    return {
        "ade_m": float(err_v.mean()),
        "fde_m": float(err_v[-1]),
        "max_error_m": float(err_v.max()),
        "n_aligned_steps": int(valid.sum()),
    }


def finite_difference_velocity(positions: np.ndarray, timestamps_us: np.ndarray) -> np.ndarray:
    """Central-difference velocity (m/s) for an ``(N,D)`` path at ``(N,)`` µs stamps.

    Formula: forward difference at the first sample, backward at the last, central
    ``(p[i+1]-p[i-1]) / (t[i+1]-t[i-1])`` elsewhere; ``Δt`` taken from the real
    timestamps (µs→s), so irregular intervals are handled exactly. Units: m/s.
    """
    positions = np.asarray(positions, dtype=float)
    t_s = np.asarray(timestamps_us, dtype=float) / 1e6
    n = positions.shape[0]
    if n < 2:
        return np.zeros_like(positions)
    vel = np.zeros_like(positions)
    vel[0] = (positions[1] - positions[0]) / (t_s[1] - t_s[0])
    vel[-1] = (positions[-1] - positions[-2]) / (t_s[-1] - t_s[-2])
    if n > 2:
        dt = (t_s[2:] - t_s[:-2])[:, None]
        vel[1:-1] = (positions[2:] - positions[:-2]) / dt
    return vel

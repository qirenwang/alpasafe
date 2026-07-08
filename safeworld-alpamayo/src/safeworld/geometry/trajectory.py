from __future__ import annotations

import numpy as np


def as_xy_array(trajectory: list[list[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(trajectory, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("trajectory must have shape [T, >=2]")
    return arr[:, :2]


def step_distances(trajectory: np.ndarray) -> np.ndarray:
    if len(trajectory) < 2:
        return np.zeros(0)
    return np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)


def speeds(trajectory: np.ndarray, dt: float = 0.1) -> np.ndarray:
    return step_distances(trajectory) / dt


def accelerations(trajectory: np.ndarray, dt: float = 0.1) -> np.ndarray:
    speed = speeds(trajectory, dt=dt)
    if len(speed) < 2:
        return np.zeros(0)
    return np.diff(speed) / dt


def jerks(trajectory: np.ndarray, dt: float = 0.1) -> np.ndarray:
    accel = accelerations(trajectory, dt=dt)
    if len(accel) < 2:
        return np.zeros(0)
    return np.diff(accel) / dt


def curvature(trajectory: np.ndarray) -> np.ndarray:
    if len(trajectory) < 3:
        return np.zeros(0)
    p = trajectory[:, :2]
    dx = np.gradient(p[:, 0])
    dy = np.gradient(p[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = np.power(dx * dx + dy * dy, 1.5)
    denom = np.where(denom < 1e-6, np.inf, denom)
    return np.abs(dx * ddy - dy * ddx) / denom


def terminal_displacement(trajectory: np.ndarray) -> float:
    if len(trajectory) == 0:
        return 0.0
    return float(np.linalg.norm(trajectory[-1, :2] - trajectory[0, :2]))


def progress(trajectory: np.ndarray) -> float:
    if len(trajectory) == 0:
        return 0.0
    return float(trajectory[-1, 0] - trajectory[0, 0])


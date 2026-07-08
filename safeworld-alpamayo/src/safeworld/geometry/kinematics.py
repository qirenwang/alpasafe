from __future__ import annotations

import numpy as np

from safeworld.geometry.trajectory import accelerations, as_xy_array, jerks, speeds


def max_abs_acceleration(trajectory: list[list[float]] | np.ndarray, dt: float = 0.1) -> float:
    acc = accelerations(as_xy_array(trajectory), dt=dt)
    return float(np.max(np.abs(acc))) if acc.size else 0.0


def max_abs_jerk(trajectory: list[list[float]] | np.ndarray, dt: float = 0.1) -> float:
    jerk = jerks(as_xy_array(trajectory), dt=dt)
    return float(np.max(np.abs(jerk))) if jerk.size else 0.0


def mean_speed(trajectory: list[list[float]] | np.ndarray, dt: float = 0.1) -> float:
    speed = speeds(as_xy_array(trajectory), dt=dt)
    return float(np.mean(speed)) if speed.size else 0.0


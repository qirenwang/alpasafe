from __future__ import annotations

import numpy as np

from safeworld.geometry.trajectory import as_xy_array


def slow_down(trajectory: list[list[float]], factor: float = 0.72) -> list[list[float]]:
    arr = as_xy_array(trajectory).copy()
    origin = arr[0].copy()
    return (origin + (arr - origin) * factor).round(4).tolist()


def stop(
    trajectory: list[list[float]],
    distance_fraction: float = 0.08,
    stop_time_fraction: float = 1.0,
) -> list[list[float]]:
    arr = as_xy_array(trajectory).copy()
    if len(arr) < 2:
        return arr.round(4).tolist()
    origin = arr[0].copy()
    final = arr[-1].copy()
    stop_idx = max(2, int(len(arr) * stop_time_fraction))
    stopped = np.zeros_like(arr)
    for idx in range(len(arr)):
        u = min(float(idx) / float(stop_idx), 1.0)
        smooth = 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3
        alpha = distance_fraction * smooth
        stopped[idx] = origin + (final - origin) * alpha
    return stopped.round(4).tolist()


def lateral_shift(trajectory: list[list[float]], offset_m: float) -> list[list[float]]:
    arr = as_xy_array(trajectory).copy()
    arr[:, 1] += offset_m
    return arr.round(4).tolist()


def speed_up(trajectory: list[list[float]], factor: float = 1.12) -> list[list[float]]:
    arr = as_xy_array(trajectory).copy()
    origin = arr[0].copy()
    return (origin + (arr - origin) * factor).round(4).tolist()


def smooth_lane_keep(trajectory: list[list[float]]) -> list[list[float]]:
    arr = as_xy_array(trajectory).copy()
    arr[:, 1] *= 0.25
    return arr.round(4).tolist()

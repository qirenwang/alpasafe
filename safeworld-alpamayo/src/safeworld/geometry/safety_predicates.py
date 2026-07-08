from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from safeworld.geometry.trajectory import (
    accelerations,
    as_xy_array,
    curvature,
    jerks,
    progress,
    speeds,
)


@dataclass(slots=True)
class SafetyPredicateResult:
    name: str
    is_safe: bool
    margin: float
    time_index: int | None
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _margin_result(name: str, values: np.ndarray, limit: float, less_equal: bool = True) -> SafetyPredicateResult:
    if values.size == 0:
        return SafetyPredicateResult(name, True, float(limit), None, {"limit": limit, "value": 0.0})
    if less_equal:
        margins = limit - np.abs(values)
    else:
        margins = values - limit
    idx = int(np.argmin(margins))
    margin = float(margins[idx])
    return SafetyPredicateResult(
        name=name,
        is_safe=margin >= 0.0,
        margin=margin,
        time_index=idx,
        details={"limit": limit, "value": float(values[idx])},
    )


def kinematic_acceleration_margin(trajectory: list[list[float]], max_accel: float = 4.0) -> SafetyPredicateResult:
    return _margin_result("kinematic_acceleration_margin", accelerations(as_xy_array(trajectory)), max_accel)


def kinematic_jerk_margin(trajectory: list[list[float]], max_jerk: float = 8.0) -> SafetyPredicateResult:
    return _margin_result("kinematic_jerk_margin", jerks(as_xy_array(trajectory)), max_jerk)


def curvature_margin(trajectory: list[list[float]], max_curvature: float = 0.35) -> SafetyPredicateResult:
    return _margin_result("curvature_margin", curvature(as_xy_array(trajectory)), max_curvature)


def offroad_margin(
    trajectory: list[list[float]],
    map_context: dict[str, Any] | None = None,
    lateral_buffer_m: float = 0.2,
) -> SafetyPredicateResult:
    arr = as_xy_array(trajectory)
    lane_width = float((map_context or {}).get("lane_width_m", 3.6))
    max_abs_y = np.abs(arr[:, 1]) if len(arr) else np.zeros(0)
    limit = lane_width / 2.0 - lateral_buffer_m
    return _margin_result("offroad_margin", max_abs_y, limit)


def _track_positions(track: dict[str, Any], horizon: int) -> np.ndarray:
    if "positions" in track and track["positions"]:
        arr = np.asarray(track["positions"], dtype=float)[:, :2]
        if len(arr) >= horizon:
            return arr[:horizon]
        tail = np.repeat(arr[-1][None, :], horizon - len(arr), axis=0)
        return np.vstack([arr, tail])
    start = np.asarray(track.get("position", [0.0, 0.0]), dtype=float)[:2]
    velocity = np.asarray(track.get("velocity", [0.0, 0.0]), dtype=float)[:2]
    return np.asarray([start + velocity * 0.1 * i for i in range(horizon)], dtype=float)


def distance_curve(trajectory: list[list[float]], object_tracks: list[dict[str, Any]] | None) -> np.ndarray:
    arr = as_xy_array(trajectory)
    if not object_tracks or len(arr) == 0:
        return np.full(len(arr), np.inf)
    curves = []
    for track in object_tracks:
        positions = _track_positions(track, len(arr))
        curves.append(np.linalg.norm(arr - positions, axis=1))
    return np.min(np.vstack(curves), axis=0)


def minimum_distance_margin(
    trajectory: list[list[float]],
    object_tracks: list[dict[str, Any]] | None,
    min_distance_m: float = 2.0,
) -> SafetyPredicateResult:
    distances = distance_curve(trajectory, object_tracks)
    if distances.size == 0 or np.all(np.isinf(distances)):
        return SafetyPredicateResult(
            "minimum_distance_margin", True, float("inf"), None, {"min_distance_m": min_distance_m}
        )
    margins = distances - min_distance_m
    idx = int(np.argmin(margins))
    return SafetyPredicateResult(
        "minimum_distance_margin",
        bool(margins[idx] >= 0.0),
        float(margins[idx]),
        idx,
        {"min_distance_m": min_distance_m, "distance": float(distances[idx])},
    )


def ttc_curve(
    trajectory: list[list[float]],
    object_tracks: list[dict[str, Any]] | None,
    collision_distance_m: float = 2.0,
) -> np.ndarray:
    distances = distance_curve(trajectory, object_tracks)
    if distances.size == 0 or np.all(np.isinf(distances)):
        return np.full(len(as_xy_array(trajectory)), np.inf)
    ttc = np.full_like(distances, np.inf, dtype=float)
    for idx, dist in enumerate(distances):
        if dist <= collision_distance_m:
            ttc[: idx + 1] = np.minimum(ttc[: idx + 1], idx * 0.1)
            break
    return ttc


def ttc_margin(
    trajectory: list[list[float]],
    object_tracks: list[dict[str, Any]] | None,
    min_ttc_s: float = 2.0,
) -> SafetyPredicateResult:
    ttc = ttc_curve(trajectory, object_tracks)
    finite = ttc[np.isfinite(ttc)]
    if finite.size == 0:
        return SafetyPredicateResult("ttc_margin", True, float("inf"), None, {"min_ttc_s": min_ttc_s})
    idx = int(np.argmin(ttc))
    margin = float(ttc[idx] - min_ttc_s)
    return SafetyPredicateResult(
        "ttc_margin",
        margin >= 0.0,
        margin,
        idx,
        {"min_ttc_s": min_ttc_s, "ttc": float(ttc[idx])},
    )


def rss_like_longitudinal_margin(
    trajectory: list[list[float]],
    object_tracks: list[dict[str, Any]] | None,
    reaction_time_s: float = 1.0,
    min_brake_mps2: float = 4.0,
) -> SafetyPredicateResult:
    arr = as_xy_array(trajectory)
    speed = speeds(arr)
    if not object_tracks or speed.size == 0:
        return SafetyPredicateResult("rss_like_longitudinal_margin", True, float("inf"), None, {})
    distances = distance_curve(trajectory, object_tracks)[: len(speed)]
    required = speed * reaction_time_s + (speed * speed) / (2.0 * min_brake_mps2)
    margins = distances - required
    idx = int(np.argmin(margins))
    return SafetyPredicateResult(
        "rss_like_longitudinal_margin",
        bool(margins[idx] >= 0.0),
        float(margins[idx]),
        idx,
        {"distance": float(distances[idx]), "required": float(required[idx])},
    )


def progress_score(trajectory: list[list[float]], min_progress_m: float = 5.0) -> SafetyPredicateResult:
    prog = progress(as_xy_array(trajectory))
    margin = prog - min_progress_m
    return SafetyPredicateResult(
        "progress_score",
        margin >= 0.0,
        float(margin),
        None,
        {"progress_m": float(prog), "min_progress_m": min_progress_m},
    )


def comfort_score(trajectory: list[list[float]], max_accel: float = 3.0, max_jerk: float = 6.0) -> SafetyPredicateResult:
    acc = kinematic_acceleration_margin(trajectory, max_accel=max_accel)
    jerk = kinematic_jerk_margin(trajectory, max_jerk=max_jerk)
    margin = min(acc.margin, jerk.margin)
    return SafetyPredicateResult(
        "comfort_score",
        margin >= 0.0,
        float(margin),
        acc.time_index if acc.margin <= jerk.margin else jerk.time_index,
        {"accel_margin": acc.margin, "jerk_margin": jerk.margin},
    )


def evaluate_all(
    trajectory: list[list[float]],
    object_tracks: list[dict[str, Any]] | None = None,
    map_context: dict[str, Any] | None = None,
) -> list[SafetyPredicateResult]:
    return [
        kinematic_acceleration_margin(trajectory),
        kinematic_jerk_margin(trajectory),
        curvature_margin(trajectory),
        offroad_margin(trajectory, map_context),
        minimum_distance_margin(trajectory, object_tracks),
        ttc_margin(trajectory, object_tracks),
        rss_like_longitudinal_margin(trajectory, object_tracks),
        progress_score(trajectory),
        comfort_score(trajectory),
    ]


def unsafe_label(results: list[SafetyPredicateResult]) -> int:
    safety_names = {
        "kinematic_acceleration_margin",
        "kinematic_jerk_margin",
        "curvature_margin",
        "offroad_margin",
        "minimum_distance_margin",
        "ttc_margin",
        "rss_like_longitudinal_margin",
        "comfort_score",
    }
    return int(any((not result.is_safe) for result in results if result.name in safety_names))


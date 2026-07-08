from safeworld.geometry.safety_predicates import (
    evaluate_all,
    minimum_distance_margin,
    offroad_margin,
    unsafe_label,
)


def test_minimum_distance_detects_close_object() -> None:
    trajectory = [[float(i), 0.0] for i in range(10)]
    tracks = [{"label": "pedestrian", "positions": [[4.0, 0.5] for _ in range(10)]}]
    result = minimum_distance_margin(trajectory, tracks, min_distance_m=2.0)
    assert not result.is_safe
    assert result.margin < 0.0


def test_offroad_detects_lane_departure() -> None:
    trajectory = [[float(i), 2.1] for i in range(10)]
    result = offroad_margin(trajectory, {"lane_width_m": 3.6})
    assert not result.is_safe


def test_evaluate_all_produces_unsafe_label() -> None:
    trajectory = [[float(i), 2.1] for i in range(10)]
    results = evaluate_all(trajectory, None, {"lane_width_m": 3.6})
    assert unsafe_label(results) == 1


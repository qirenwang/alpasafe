from safeworld.reasoning.rawc import compute_rawc


def test_rawc_flags_yield_without_slowdown() -> None:
    trajectory = [[i * 0.5, 0.0] for i in range(64)]
    rawc = compute_rawc(
        "A pedestrian is crossing. The vehicle should yield and slow down.",
        trajectory,
        [{"label": "pedestrian", "positions": [[10.0, 0.0] for _ in range(64)]}],
        {"lane_width_m": 3.6},
        ["pedestrian_crossing"],
    )
    assert rawc.score < 1.0
    assert "reason_says_yield_but_speed_not_reduced" in rawc.contradictions


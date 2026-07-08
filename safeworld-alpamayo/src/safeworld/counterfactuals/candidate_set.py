from __future__ import annotations

from safeworld.counterfactuals.trajectory_perturb import (
    lateral_shift,
    slow_down,
    smooth_lane_keep,
    speed_up,
    stop,
)


def generate_candidates(trajectory: list[list[float]]) -> dict[str, list[list[float]]]:
    return {
        "tau_alpamayo": trajectory,
        "tau_slow_down": slow_down(trajectory),
        "tau_stop": stop(trajectory),
        "tau_more_conservative_yield": slow_down(trajectory, factor=0.55),
        "tau_left_perturb": lateral_shift(trajectory, offset_m=0.35),
        "tau_right_perturb": lateral_shift(trajectory, offset_m=-0.35),
        "tau_speed_up_small": speed_up(trajectory),
        "tau_smooth_lane_keep": smooth_lane_keep(trajectory),
    }


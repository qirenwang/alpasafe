from __future__ import annotations

import hashlib
import time

from safeworld.alpamayo.prompts import build_prompt
from safeworld.counterfactuals.trajectory_perturb import slow_down, stop
from safeworld.data.schema import (
    AlpamayoProposal,
    CandidateTrajectory,
    DrivingSample,
    TopKAlpamayoProposal,
)

SUPPORTED_K_VALUES = (1, 2, 5, 10)

# Deterministic per-rank variations used to emulate the diversity of Alpamayo's
# native top-K proposals in dry-run mode. Rank 1 is the base behavior; later
# ranks are progressively more aggressive or more conservative maneuvers.
_RANK_VARIATIONS: list[tuple[float, float, str]] = [
    (1.00, 0.00, "Primary maneuver: follow the planned path at the planned speed."),
    (0.92, 0.00, "Alternative maneuver: reduce speed slightly and keep the same path."),
    (1.00, 0.12, "Alternative maneuver: keep speed and bias slightly left within the lane."),
    (1.00, -0.12, "Alternative maneuver: keep speed and bias slightly right within the lane."),
    (0.84, 0.06, "Alternative maneuver: slow down moderately with a small left bias."),
    (1.06, 0.00, "Alternative maneuver: increase speed slightly while keeping the lane."),
    (0.78, -0.06, "Alternative maneuver: slow down strongly with a small right bias."),
    (0.95, 0.20, "Alternative maneuver: ease off and shift left toward the lane edge."),
    (0.95, -0.20, "Alternative maneuver: ease off and shift right toward the lane edge."),
    (1.10, 0.05, "Alternative maneuver: make faster progress with a minor left bias."),
]


def _scale_trajectory(points: list[list[float]], scale: float, lateral_bias: float = 0.0) -> list[list[float]]:
    scaled: list[list[float]] = []
    for point in points:
        x = float(point[0]) * scale
        y = float(point[1]) + lateral_bias
        scaled.append([round(x, 4), round(y, 4)])
    return scaled


def _base_behavior(sample: DrivingSample) -> tuple[float, float, str]:
    tags = set(sample.scenario_tags)
    scale = 1.0
    lateral_bias = 0.0
    reasoning = "The route is clear, keep lane, maintain speed, and continue smoothly."
    if "pedestrian_crossing" in tags:
        scale = 1.08
        reasoning = (
            "A pedestrian crossing is visible ahead. The vehicle should yield and slow down, "
            "then proceed only if the crosswalk is clear."
        )
    elif "vehicle_cut_in" in tags:
        scale = 1.05
        lateral_bias = 0.15
        reasoning = (
            "A vehicle may cut in from the right. Keep lane, monitor the vehicle, and maintain "
            "a cautious gap."
        )
    elif "construction_or_blocked_lane" in tags:
        scale = 0.95
        lateral_bias = -0.25
        reasoning = (
            "Construction partially blocks the lane. Slow down and shift left within the lane "
            "to avoid cones."
        )
    elif "hard_brake" in tags:
        scale = 1.12
        reasoning = "Traffic ahead may slow. Yield if needed but keep progress."
    elif "low_light_or_glare" in tags:
        scale = 1.0
        reasoning = "Low-light glare reduces certainty. Slow down and prepare to yield."
    return scale, lateral_bias, reasoning


def _sample_jitter(sample_id: str, rank: int) -> float:
    digest = hashlib.sha256(f"{sample_id}:{rank}".encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) / 0xFFFFFFFF - 0.5) * 0.04


def dry_run_proposal(sample: DrivingSample, model_name: str) -> AlpamayoProposal:
    start = time.perf_counter()
    future = sample.future_ego_trajectory or [[0.1 * i, 0.0] for i in range(64)]
    scale, lateral_bias, reasoning = _base_behavior(sample)
    trajectory = _scale_trajectory(future, scale=scale, lateral_bias=lateral_bias)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return AlpamayoProposal(
        sample_id=sample.scene_id,
        model_name=model_name,
        prompt=build_prompt(sample),
        reasoning_text=reasoning,
        trajectory=trajectory,
        inference_latency_ms=round(elapsed_ms, 4),
        raw_output_path=None,
        metadata={"mode": "dry_run", "num_waypoints": len(trajectory)},
    )


def dry_run_topk_candidates(sample: DrivingSample, k: int) -> list[CandidateTrajectory]:
    future = sample.future_ego_trajectory or [[0.1 * i, 0.0] for i in range(64)]
    base_scale, base_lateral, base_reasoning = _base_behavior(sample)
    candidates: list[CandidateTrajectory] = []
    for rank in range(1, k + 1):
        var_scale, var_lateral, var_reasoning = _RANK_VARIATIONS[(rank - 1) % len(_RANK_VARIATIONS)]
        jitter = _sample_jitter(sample.scene_id, rank)
        scale = base_scale * var_scale + jitter
        lateral = base_lateral + var_lateral
        trajectory = _scale_trajectory(future, scale=scale, lateral_bias=lateral)
        reasoning = base_reasoning if rank == 1 else f"{base_reasoning} {var_reasoning}"
        candidates.append(
            CandidateTrajectory(
                candidate_id=f"alpamayo_candidate_{rank:02d}",
                candidate_source="alpamayo_native",
                candidate_rank=rank,
                trajectory=trajectory,
                reasoning_trace=reasoning,
                model_score=round(-0.10 * (rank - 1) + jitter, 4),
                raw_output_reference=None,
                metadata={
                    "mode": "dry_run",
                    "num_waypoints": len(trajectory),
                    "score_type": "synthetic_logprob_dry_run",
                },
            )
        )
    return candidates


def fallback_candidates(reference_trajectory: list[list[float]]) -> list[CandidateTrajectory]:
    specs = [
        ("fallback_stop", stop(reference_trajectory), "Backup controller: smooth early stop."),
        ("fallback_slow", slow_down(reference_trajectory), "Backup controller: slow down."),
        (
            "fallback_conservative_yield",
            slow_down(reference_trajectory, factor=0.55),
            "Backup controller: conservative yield.",
        ),
    ]
    return [
        CandidateTrajectory(
            candidate_id=candidate_id,
            candidate_source="fallback",
            candidate_rank=None,
            trajectory=trajectory,
            reasoning_trace=reasoning,
            model_score=None,
            raw_output_reference=None,
            metadata={"mode": "dry_run", "controller": "backup"},
        )
        for candidate_id, trajectory, reasoning in specs
    ]


class AlpamayoWrapper:
    def __init__(
        self,
        model_name: str,
        dry_run: bool = True,
        model_version: str = "1.5-10B",
        use_real_alpamayo: bool = False,
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version
        self.dry_run = dry_run
        self.use_real_alpamayo = use_real_alpamayo

    def _require_dry_run(self) -> None:
        if not self.dry_run or self.use_real_alpamayo:
            # Placeholder for real inference: loading the 10B Alpamayo checkpoint
            # requires licensed model access and a configured GPU runtime.
            raise RuntimeError(
                "Real Alpamayo loading is intentionally not automatic. Set "
                "topk.use_real_alpamayo=false / use --dry-run unless licensed model "
                "access and GPU runtime are configured."
            )

    def run(self, sample: DrivingSample) -> AlpamayoProposal:
        self._require_dry_run()
        return dry_run_proposal(sample, self.model_name)

    def run_topk(self, sample: DrivingSample, k: int) -> TopKAlpamayoProposal:
        if k < 1:
            raise ValueError("k must be >= 1")
        if k not in SUPPORTED_K_VALUES:
            raise ValueError(f"k={k} is outside the deployment-supported range {SUPPORTED_K_VALUES}")
        self._require_dry_run()
        start = time.perf_counter()
        candidates = dry_run_topk_candidates(sample, k)
        fallbacks = fallback_candidates(candidates[0].trajectory)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return TopKAlpamayoProposal(
            sample_id=sample.scene_id,
            k_requested=k,
            k_returned=len(candidates),
            alpamayo_model_name=self.model_name,
            alpamayo_model_version=self.model_version,
            inference_mode="dry_run",
            latency_ms=round(elapsed_ms, 4),
            gpu_memory_mb=None,
            candidates=candidates,
            fallback_candidates=fallbacks,
            metadata={"prompt": build_prompt(sample)},
        )

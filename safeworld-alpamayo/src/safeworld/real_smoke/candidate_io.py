"""Immutable load/validation of cached Alpamayo-native candidates (T24 retry).

This module is deliberately read-only with respect to candidates: it loads cached real candidates,
validates them, and converts one into a :class:`CandidateRolloutRequest` for a future rollout. It
never optimizes, perturbs, copies, or fabricates a candidate, and it emits no fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from safeworld.data.schema_v2 import CANDIDATE_SOURCE_NATIVE, CandidateRolloutRequest
from safeworld.real_smoke.reasoning_alignment import assert_unambiguous_reasoning

# Frozen Alpamayo trajectory contract (verified against real inference output shapes).
TRAJECTORY_FRAME = "ego_t0"
TRAJECTORY_HZ = 10.0
TRAJECTORY_STEPS = 64
TRAJECTORY_HORIZON_S = 6.4
DUPLICATE_TOL_M = 1e-3

__all__ = [
    "CachedAlpamayoCandidate",
    "load_cached_candidates",
    "assert_native_only",
    "assert_distinct_trajectories",
    "to_rollout_request",
]


@dataclass(frozen=True)
class CachedAlpamayoCandidate:
    """Immutable view of one cached Alpamayo-native candidate (frozen => cannot be mutated)."""

    candidate_id: str
    candidate_rank: int
    candidate_source: str
    reasoning_trace: str | None
    trajectory_xyz: tuple[tuple[float, float, float], ...]
    trajectory_frame: str
    trajectory_hz: float
    trajectory_steps: int
    traj_xyz_sha256: str
    seed: int | None = None

    def validate(self) -> None:
        if self.candidate_source != CANDIDATE_SOURCE_NATIVE:
            raise ValueError(
                f"candidate_source must be {CANDIDATE_SOURCE_NATIVE!r} (no fallback candidates), "
                f"got {self.candidate_source!r}"
            )
        if self.reasoning_trace is None or not str(self.reasoning_trace).strip():
            raise ValueError(f"candidate {self.candidate_id} must have exactly one reasoning trace")
        # Reject an un-split K-sample reasoning array: the real CLI requires exactly
        # one candidate-specific reasoning trace (T24R5). See reasoning_alignment.
        assert_unambiguous_reasoning(
            self.reasoning_trace, context=f"cached candidate {self.candidate_id}"
        )
        if self.trajectory_frame != TRAJECTORY_FRAME:
            raise ValueError(f"trajectory_frame must be {TRAJECTORY_FRAME!r}")
        if float(self.trajectory_hz) != TRAJECTORY_HZ:
            raise ValueError(f"trajectory_hz must be {TRAJECTORY_HZ}")
        if self.trajectory_steps != len(self.trajectory_xyz):
            raise ValueError("trajectory_steps must match the number of trajectory points")
        if self.trajectory_steps != TRAJECTORY_STEPS:
            raise ValueError(f"trajectory must have {TRAJECTORY_STEPS} steps (6.4 s @ 10 Hz)")
        for p in self.trajectory_xyz:
            if len(p) != 3 or not all(_is_finite(v) for v in p):
                raise ValueError(f"candidate {self.candidate_id} has a non-finite/invalid point")


def _is_finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


def load_cached_candidates(jsonl_path: str) -> list[CachedAlpamayoCandidate]:
    """Load and validate cached candidates from a JSONL file (read-only)."""
    out: list[CachedAlpamayoCandidate] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cand = CachedAlpamayoCandidate(
                candidate_id=row["candidate_id"],
                candidate_rank=int(row["candidate_rank"]),
                candidate_source=row["candidate_source"],
                reasoning_trace=row.get("reasoning_trace"),
                trajectory_xyz=tuple(tuple(float(x) for x in p) for p in row["trajectory_xyz"]),
                trajectory_frame=row["trajectory_frame"],
                trajectory_hz=float(row["trajectory_hz"]),
                trajectory_steps=int(row["trajectory_steps"]),
                traj_xyz_sha256=row["traj_xyz_sha256"],
                seed=row.get("seed"),
            )
            cand.validate()
            out.append(cand)
    return out


def assert_native_only(cands: list[CachedAlpamayoCandidate]) -> None:
    """Reject any non-native (fallback) candidate."""
    for c in cands:
        if c.candidate_source != CANDIDATE_SOURCE_NATIVE:
            raise ValueError(f"non-native candidate present: {c.candidate_id} ({c.candidate_source})")


def assert_distinct_trajectories(
    cands: list[CachedAlpamayoCandidate], tol_m: float = DUPLICATE_TOL_M
) -> None:
    """Reject duplicate trajectories within numerical tolerance (no manufactured duplicates)."""
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i].trajectory_xyz, cands[j].trajectory_xyz
            max_abs = max(abs(a[t][d] - b[t][d]) for t in range(len(a)) for d in range(3))
            if max_abs <= tol_m:
                raise ValueError(
                    f"candidates {cands[i].candidate_id} and {cands[j].candidate_id} are duplicate "
                    f"trajectories within {tol_m} m"
                )


def to_rollout_request(
    cand: CachedAlpamayoCandidate,
    *,
    sample_id: str,
    scene_id: str,
    initial_state_ref: str,
    rollout_backend: str = "alpasim",
    rollout_horizon_s: float = TRAJECTORY_HORIZON_S,
) -> CandidateRolloutRequest:
    """Build a rollout request from a cached candidate, preserving its values (no modification)."""
    cand.validate()
    req = CandidateRolloutRequest(
        sample_id=sample_id,
        scene_id=scene_id,
        candidate_id=cand.candidate_id,
        candidate_rank=cand.candidate_rank,
        candidate_source=cand.candidate_source,
        reasoning_trace=cand.reasoning_trace,
        trajectory=[list(p) for p in cand.trajectory_xyz],
        trajectory_frame=cand.trajectory_frame,
        trajectory_hz=cand.trajectory_hz,
        rollout_backend=rollout_backend,
        rollout_horizon_s=rollout_horizon_s,
        initial_state_ref=initial_state_ref,
        metadata={"seed": cand.seed, "source_sha256": cand.traj_xyz_sha256},
    )
    req.validate()
    return req

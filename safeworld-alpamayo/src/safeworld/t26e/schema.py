"""T26E-A derived-record schema and feature/label boundary.

The record structure keeps an explicit four-way separation:

- ``input``      — pre-rollout candidate features and observation references
                   only (the T26A/T26C executed contract defines the model
                   input feature tensor as the planned trajectory; reasoning
                   text is an input per method_spec_v2 W_theta(O, tau_i, r_i)).
- ``targets``    — everything produced by or after rollout execution: the
                   future-consequence target block and the official AlpaSim
                   scene score block. Never readable as features.
- ``metadata``   — identity and diagnostics (never model features by default).
- ``provenance`` — revisions, digests, and evaluation-mode flags.

``FORBIDDEN_INPUT_KEYS`` is the leakage guard: none of these may appear
anywhere inside the ``input`` section of any record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Post-rollout labels / outcomes / diagnostics (critical leakage rule, T26E-A).
FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "official_alpasim_scene_score",
        "progress_rel_to_total",
        "progress_clipped_rel",
        "progress_score",
        "progress_rel",
        "progress",
        "gt_dist_traveled_m",
        "dist_to_gt_trajectory",
        "dist_to_gt_location",
        "dist_traveled_m",
        "collision_at_fault",
        "collision_front",
        "collision_lateral",
        "collision_rear",
        "collision_any",
        "offroad",
        "wrong_lane",
        "safety_monitor_triggered",
        "failure_reason",
        "passed",
        "status",
        "score",
        "score_metrics",
        "official_score_metrics",
        "future_ego_states_global",
        "future_ego_states_ego_t0",
        "future_ego_velocity_rig_mps",
        "future_non_ego_actor_states_global",
        "future_non_ego_actor_states_ego_t0",
        "actor_valid_mask",
        "timestamps_us",
        "future_consequence_target",
        "official_score",
    }
)

# Identity/metadata keys that must not silently become features (metadata by
# default per the T26E-A contract; no frozen project contract requires any of
# them as a model input).
METADATA_ONLY_KEYS = frozenset(
    {
        "sample_id",
        "scene_id",
        "decision_timestamp_us",
        "decision_group_id",
        "candidate_id",
        "candidate_index",
        "rollout_id",
        "split",
    }
)

REQUIRED_TOP_LEVEL = ("sample_id", "decision_group_id", "split", "input", "metadata", "provenance")
LABELED_SPLITS = ("train", "val")


@dataclass(slots=True)
class T26ERecord:
    """One derived T26E-A record (labeled unless it is a sealed test input)."""

    sample_id: str
    scene_id: str
    decision_timestamp_us: int
    decision_group_id: str
    candidate_id: str
    candidate_index: int
    rollout_id: str
    split: str
    input: dict[str, Any]
    targets: dict[str, Any] | None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> T26ERecord:
        return cls(
            sample_id=row["sample_id"],
            scene_id=row["scene_id"],
            decision_timestamp_us=int(row["decision_timestamp_us"]),
            decision_group_id=row["decision_group_id"],
            candidate_id=row["candidate_id"],
            candidate_index=int(row["candidate_index"]),
            rollout_id=row["rollout_id"],
            split=row["split"],
            input=row["input"],
            targets=row.get("targets"),
            metadata=row.get("metadata", {}),
            provenance=row.get("provenance", {}),
        )


def _walk_keys(obj: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            keys |= _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            keys |= _walk_keys(v)
    return keys


def validate_record_structure(row: dict[str, Any], *, expect_targets: bool) -> list[str]:
    """Return a list of structural/leakage violations (empty == valid)."""
    problems: list[str] = []
    for k in REQUIRED_TOP_LEVEL:
        if k not in row:
            problems.append(f"missing top-level key: {k}")
    if problems:
        return problems

    input_keys = _walk_keys(row["input"])
    leaked = sorted(input_keys & FORBIDDEN_INPUT_KEYS)
    if leaked:
        problems.append(f"forbidden post-rollout keys inside input: {leaked}")
    meta_in_input = sorted(input_keys & METADATA_ONLY_KEYS)
    if meta_in_input:
        problems.append(f"metadata-only identifier keys inside input: {meta_in_input}")

    if expect_targets:
        t = row.get("targets")
        if not isinstance(t, dict):
            problems.append("labeled record missing targets section")
        else:
            score = t.get("official_score", {}).get("official_alpasim_scene_score")
            if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
                problems.append(f"official_alpasim_scene_score out of [0,1] or missing: {score!r}")
            if "future_consequence_target" not in t:
                problems.append("labeled record missing future_consequence_target")
    elif row.get("targets") is not None:
        problems.append("test-input record must not carry a targets section")

    return problems

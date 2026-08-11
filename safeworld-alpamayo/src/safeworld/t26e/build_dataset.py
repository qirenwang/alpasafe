"""Deterministic T26E-A dataset builder.

Joins (by ``sample_id``, with full secondary-identifier cross-checks) the
frozen T26D K8 future-consequence targets, candidate provenance, and the
official AlpaSim 196d21a scene scores into a training-ready derived dataset.

Read-only over all sources. Byte-deterministic: no wall-clock values are
written into any output file; identical inputs yield identical output bytes.

Usage:
    python -m safeworld.t26e.build_dataset [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
T26D = REPO / "outputs/t26d_k8_expanded"
SCORES = REPO / "outputs/t26d_k8_expanded_scores_alpasim_196d21a"
DEFAULT_OUT = REPO / "outputs/t26e_official_score_training_dataset_alpasim_196d21a"

TARGETS_F = T26D / "t26d_k8_future_consequence_targets.jsonl"
PROV_F = T26D / "t26d_k8_candidate_provenance.jsonl"
SPLIT_F = T26D / "t26d_k8_scene_level_split_manifest.json"
SCORES_F = SCORES / "alpasim_196d21a_official_scene_scores.jsonl"

ROLLOUT_REV = "a1f05bb628f3d1d19d79d44188e836e9108f98c6"
EVAL_REV = "196d21ab86593af121b055995d0185bb786d1f70"
SPLIT_ORDER = ("train", "val", "test")

FUTURE_TARGET_KEYS = (
    "timestamps_us",
    "future_ego_states_global",
    "future_ego_states_ego_t0",
    "future_ego_velocity_rig_mps",
    "future_non_ego_actor_states_global",
    "future_non_ego_actor_states_ego_t0",
    "actor_ids",
    "actor_valid_mask",
)


def canonical_digest(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def _group_id(scene_id: str, decision_timestamp_us: int) -> str:
    return f"{scene_id}@{decision_timestamp_us}"


def build_records() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Join sources into full labeled records; return (records, source_hashes)."""
    targets = load_jsonl(TARGETS_F)
    prov = load_jsonl(PROV_F)
    scores = load_jsonl(SCORES_F)
    split_manifest = json.loads(SPLIT_F.read_text())
    scene2split = {s: k for k, v in split_manifest["split"].items() for s in v}

    t_by = {r["sample_id"]: r for r in targets}
    p_by = {r["sample_id"]: r for r in prov}
    s_by = {r["sample_id"]: r for r in scores}
    if not (len(t_by) == len(p_by) == len(s_by) == len(targets) == 288):
        raise ValueError("expected 288 unique sample_ids in every source")

    records = []
    for sid in sorted(t_by):
        t, p, s = t_by[sid], p_by[sid], s_by[sid]
        # secondary identifier cross-checks (join is invalid if any disagrees)
        checks = [
            t["scene_id"] == p["scene_id"] == s["scene_id"],
            t["decision_timestamp_us"] == p["decision_timestamp_us"] == s["decision_timestamp_us"],
            t["candidate_id"] == p["candidate_id"] == s["candidate_id"],
            sid.endswith("#" + t["candidate_id"]),
            p["candidate_index"] == s["candidate_index"],
            p["rollout_id"] == s["rollout_id"],
            scene2split[t["scene_id"]] == s["split"],
            t["provenance"]["rollout_asl_sha256"]
            == p["rollout_asl_sha256"]
            == s["rollout_asl_sha256"],
        ]
        if not all(checks):
            raise ValueError(f"join cross-check failed for {sid}: {checks}")

        record = {
            "sample_id": sid,
            "scene_id": t["scene_id"],
            "decision_timestamp_us": t["decision_timestamp_us"],
            "decision_group_id": _group_id(t["scene_id"], t["decision_timestamp_us"]),
            "candidate_id": t["candidate_id"],
            "candidate_index": p["candidate_index"],
            "rollout_id": p["rollout_id"],
            "split": s["split"],
            "input": {
                # Pre-rollout candidate features (T26A/T26C executed contract:
                # the model input feature tensor is the planned trajectory;
                # reasoning is an input per method_spec_v2 W_theta(O, tau, r)).
                "planned_trajectory": t["planned_trajectory"],
                "planned_trajectory_frame": t["planned_trajectory_frame"],
                "planned_trajectory_hz": t["planned_trajectory_hz"],
                "planned_horizon_steps": t["planned_horizon_steps"],
                "candidate_reasoning_trace": t["candidate_reasoning_trace"],
                # Pre-rollout observation/scene references (references, not
                # feature tensors; O_history extraction is out of T26E-A scope).
                "observation_references": {
                    "map_or_scene_reference": t["map_or_scene_reference"],
                    "warmup_duration_us": t["warmup_duration_us"],
                },
            },
            "targets": {
                "future_consequence_target": {k: t[k] for k in FUTURE_TARGET_KEYS},
                "official_score": {
                    "official_alpasim_scene_score": s["official_alpasim_scene_score"],
                    "official_score_status": s["status"],
                    "official_score_passed": s["passed"],
                    "official_score_failure_reason": s["failure_reason"],
                    "official_score_metrics": s["official_score_metrics"],
                    "score_transformation": "identity",
                    "score_range": [0.0, 1.0],
                },
            },
            "metadata": {
                "decision_tag": t["decision_tag"],
                "candidate_source": t["candidate_source"],
                "origin": t["origin"],
                "rollout_semantics": t["rollout_semantics"],
                "quaternion_order": t["quaternion_order"],
                "rotation_handedness": t["rotation_handedness"],
                "world_model_target_kind": t["world_model_target_kind"],
                "native_metric_record_reference": t["native_metric_record_reference"],
                "coordinate_transform_roundtrip_max_m": t["coordinate_transform_roundtrip_max_m"],
            },
            "provenance": {
                "rollout_alpasim_git_revision": ROLLOUT_REV,
                "evaluator_alpasim_git_revision": EVAL_REV,
                "evaluation_mode": s["evaluation_mode"],
                "cross_version_re_evaluation": s["cross_version_re_evaluation"],
                "native_to_rollout_revision": s["native_to_rollout_revision"],
                "simulation_rerun": s["simulation_rerun"],
                "source_target_sha256": canonical_digest(t),
                "source_provenance_sha256": canonical_digest(p),
                "source_score_sha256": canonical_digest(s),
                "rollout_asl_sha256": p["rollout_asl_sha256"],
                "metrics_parquet_sha256": p["metrics_parquet_sha256"],
                "generated_post_eval_metrics_sha256": s["generated_post_eval_metrics_sha256"],
                "evaluator_config_sha256": s["evaluator_config_sha256"],
            },
        }
        if s["rollout_alpasim_git_revision"] != ROLLOUT_REV:
            raise ValueError(f"unexpected rollout revision for {sid}")
        if s["evaluator_alpasim_git_revision"] != EVAL_REV:
            raise ValueError(f"unexpected evaluator revision for {sid}")
        records.append(record)

    records.sort(
        key=lambda r: (
            SPLIT_ORDER.index(r["split"]),
            r["scene_id"],
            r["decision_timestamp_us"],
            r["candidate_index"],
            r["sample_id"],
        )
    )
    source_hashes = {
        str(TARGETS_F.relative_to(REPO)): file_sha256(TARGETS_F),
        str(PROV_F.relative_to(REPO)): file_sha256(PROV_F),
        str(SCORES_F.relative_to(REPO)): file_sha256(SCORES_F),
        str(SPLIT_F.relative_to(REPO)): file_sha256(SPLIT_F),
    }
    return records, source_hashes


def _strip_to_test_input(rec: dict[str, Any]) -> dict[str, Any]:
    """Sealed-test input view: no targets, no post-rollout outcome anywhere."""
    out = {k: rec[k] for k in rec if k != "targets"}
    prov = dict(out["provenance"])
    prov.pop("source_score_sha256", None)  # lives with the sealed labels
    out["provenance"] = prov
    out["targets"] = None
    return out


def _sealed_test_labels(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": rec["sample_id"],
        "scene_id": rec["scene_id"],
        "decision_timestamp_us": rec["decision_timestamp_us"],
        "decision_group_id": rec["decision_group_id"],
        "candidate_id": rec["candidate_id"],
        "candidate_index": rec["candidate_index"],
        "rollout_id": rec["rollout_id"],
        "split": rec["split"],
        "targets": rec["targets"],
        "provenance": rec["provenance"],
    }


def _score_stats(vals: list[float]) -> dict[str, float | int]:
    return {
        "n": len(vals),
        "non_null": sum(v is not None for v in vals),
        "min": min(vals),
        "max": max(vals),
        "mean": statistics.mean(vals),
        "std": statistics.pstdev(vals),
        "n_unique": len(set(vals)),
    }


def build(out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records, source_hashes = build_records()

    by_split: dict[str, list[dict[str, Any]]] = {k: [] for k in SPLIT_ORDER}
    for r in records:
        by_split[r["split"]].append(r)

    dump_jsonl(out_dir / "t26e_train_samples.jsonl", by_split["train"])
    dump_jsonl(out_dir / "t26e_val_samples.jsonl", by_split["val"])
    dump_jsonl(
        out_dir / "t26e_test_inputs.jsonl", [_strip_to_test_input(r) for r in by_split["test"]]
    )
    dump_jsonl(
        out_dir / "t26e_test_labels_sealed.jsonl", [_sealed_test_labels(r) for r in by_split["test"]]
    )

    index_rows = [
        {
            "sample_id": r["sample_id"],
            "decision_group_id": r["decision_group_id"],
            "scene_id": r["scene_id"],
            "decision_timestamp_us": r["decision_timestamp_us"],
            "candidate_id": r["candidate_id"],
            "candidate_index": r["candidate_index"],
            "rollout_id": r["rollout_id"],
            "split": r["split"],
            "labeled_file": {
                "train": "t26e_train_samples.jsonl",
                "val": "t26e_val_samples.jsonl",
                "test": "t26e_test_labels_sealed.jsonl (sealed)",
            }[r["split"]],
            "record_digest": canonical_digest(r),
        }
        for r in records
    ]
    dump_jsonl(out_dir / "t26e_all_sample_index.jsonl", index_rows)

    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(r["decision_group_id"], []).append(r)
    group_rows = []
    for gid in sorted(
        groups,
        key=lambda g: (
            SPLIT_ORDER.index(groups[g][0]["split"]),
            groups[g][0]["scene_id"],
            groups[g][0]["decision_timestamp_us"],
        ),
    ):
        mem = sorted(groups[gid], key=lambda r: r["candidate_index"])
        if [m["candidate_index"] for m in mem] != list(range(8)):
            raise ValueError(f"group {gid} incomplete")
        group_rows.append(
            {
                "decision_group_id": gid,
                "scene_id": mem[0]["scene_id"],
                "decision_timestamp_us": mem[0]["decision_timestamp_us"],
                "split": mem[0]["split"],
                "n_candidates": 8,
                "candidate_ids": [m["candidate_id"] for m in mem],
                "sample_ids": [m["sample_id"] for m in mem],
                "k_prefix_views": {"K2": [0, 1], "K5": [0, 1, 2, 3, 4], "K8": list(range(8))},
            }
        )
    dump_jsonl(out_dir / "t26e_decision_group_index.jsonl", group_rows)

    dump_json(
        out_dir / "t26e_feature_schema.json",
        {
            "section": "input",
            "model_input_feature_tensor": {
                "planned_trajectory": {
                    "shape": [64, 2],
                    "dtype": "float32",
                    "frame": "ego_t0_rig",
                    "hz": 10,
                    "note": "the only executed-contract feature tensor (T26A/T26C)",
                }
            },
            "additional_inputs_per_method_spec_v2": {
                "candidate_reasoning_trace": "str (r_i in W_theta(O_history, tau_i, r_i))",
                "observation_references": "scene/map references + warmup; O_history "
                "feature extraction is not defined by any frozen contract yet",
            },
            "metadata_only_identifiers": sorted(
                [
                    "sample_id",
                    "scene_id",
                    "decision_timestamp_us",
                    "decision_group_id",
                    "candidate_id",
                    "candidate_index",
                    "rollout_id",
                    "split",
                ]
            ),
            "forbidden_input_keys": "see safeworld/t26e/schema.py FORBIDDEN_INPUT_KEYS",
        },
    )
    dump_json(
        out_dir / "t26e_label_schema.json",
        {
            "section": "targets",
            "future_consequence_target": {k: "as frozen in T26D" for k in FUTURE_TARGET_KEYS},
            "official_score": {
                "official_alpasim_scene_score": {
                    "dtype": "float",
                    "range": [0.0, 1.0],
                    "transformation": "identity",
                    "additional_normalization": "none",
                },
                "official_score_status": "pass|fail (diagnostic)",
                "official_score_passed": "bool (diagnostic)",
                "official_score_failure_reason": "null|collision_at_fault|offroad (diagnostic)",
                "official_score_metrics": "official evaluator score_metrics dict (diagnostic)",
            },
            "diagnostics_are_not_training_objectives": True,
        },
    )

    split_manifest = json.loads(SPLIT_F.read_text())
    dump_json(
        out_dir / "t26e_split_manifest.json",
        {
            "scene_split": split_manifest["split"],
            "split_source": str(SPLIT_F.relative_to(REPO)),
            "split_source_sha256": source_hashes[str(SPLIT_F.relative_to(REPO))],
            "candidate_counts": {k: len(v) for k, v in by_split.items()},
            "group_counts": {
                k: sum(1 for g in group_rows if g["split"] == k) for k in SPLIT_ORDER
            },
            "scene_disjoint": True,
        },
    )

    join_contract = {
        "primary_join_key": "sample_id",
        "secondary_cross_checks": [
            "scene_id",
            "decision_timestamp_us",
            "candidate_id (+sample_id suffix)",
            "candidate_index",
            "rollout_id",
            "split (scene-level manifest)",
            "rollout_asl_sha256 (three-way)",
        ],
        "decision_group_id_construction": "scene_id + '@' + str(decision_timestamp_us)",
        "line_order_used_as_identity": False,
        "sources": source_hashes,
    }
    dump_json(out_dir / "t26e_join_contract.json", join_contract)

    tv = by_split["train"] + by_split["val"]
    qa = {
        "counts": {
            "train": len(by_split["train"]),
            "val": len(by_split["val"]),
            "test_inputs": len(by_split["test"]),
            "test_labels_sealed": len(by_split["test"]),
            "groups": {k: sum(1 for g in group_rows if g["split"] == k) for k in SPLIT_ORDER},
        },
        "score_target": {
            "name": "official_alpasim_scene_score",
            "transformation": "identity",
            "range": [0.0, 1.0],
            "train_stats": _score_stats(
                [
                    r["targets"]["official_score"]["official_alpasim_scene_score"]
                    for r in by_split["train"]
                ]
            ),
            "val_stats": _score_stats(
                [
                    r["targets"]["official_score"]["official_alpasim_scene_score"]
                    for r in by_split["val"]
                ]
            ),
            "test_policy": "sealed; no statistics reported",
            "train_val_all_finite_in_0_1": all(
                isinstance(
                    (v := r["targets"]["official_score"]["official_alpasim_scene_score"]),
                    (int, float),
                )
                and 0.0 <= v <= 1.0
                and math.isfinite(v)
                for r in tv
            ),
        },
    }
    dump_json(out_dir / "t26e_dataset_qa.json", qa)

    data_files = [
        "t26e_train_samples.jsonl",
        "t26e_val_samples.jsonl",
        "t26e_test_inputs.jsonl",
        "t26e_test_labels_sealed.jsonl",
        "t26e_all_sample_index.jsonl",
        "t26e_decision_group_index.jsonl",
        "t26e_feature_schema.json",
        "t26e_label_schema.json",
        "t26e_join_contract.json",
        "t26e_split_manifest.json",
        "t26e_dataset_qa.json",
    ]
    file_hashes = {f: file_sha256(out_dir / f) for f in data_files}
    dump_json(
        out_dir / "t26e_dataset_manifest.json",
        {
            "dataset": "T26E-A official-score training dataset (alpasim 196d21a labels)",
            "builder": "safeworld.t26e.build_dataset",
            "rollout_alpasim_git_revision": ROLLOUT_REV,
            "evaluator_alpasim_git_revision": EVAL_REV,
            "evaluation_mode": "offline_post_eval_existing_frozen_asl",
            "cross_version_re_evaluation": True,
            "native_to_rollout_revision": False,
            "simulation_rerun": False,
            "source_sha256": source_hashes,
            "file_sha256": file_hashes,
            "canonical_ordering": "split(train,val,test) then (scene_id, "
            "decision_timestamp_us, candidate_index, sample_id)",
        },
    )
    all_hashes = dict(file_hashes)
    all_hashes["t26e_dataset_manifest.json"] = file_sha256(out_dir / "t26e_dataset_manifest.json")
    dump_json(
        out_dir / "t26e_frozen_dataset_manifest.json",
        {"_note": "content hashes of the frozen T26E-A derived dataset", "file_sha256": all_hashes},
    )
    all_hashes["t26e_frozen_dataset_manifest.json"] = file_sha256(
        out_dir / "t26e_frozen_dataset_manifest.json"
    )
    return all_hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    hashes = build(args.out)
    print(json.dumps(hashes, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

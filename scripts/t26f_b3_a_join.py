#!/usr/bin/env python
"""T26F-B3-A Stage F: independently verify locks and build the frozen join."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


B3_0 = Path(
    "/storage/alpasafe/safeworld-alpamayo/artifacts/"
    "safeworld_t26f_b3_0_preregistration/20260718T180730Z"
)
SEALED = "t26e_test_labels_sealed.jsonl"
CONDITIONS = ("A_NORMAL", "B_NORMAL", "B_ZERO", "B_WRONG_SCENE")
TAGS = ("A", "B", "C")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    if SEALED in str(path):
        raise RuntimeError("sealed-label access refused")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload.encode()).hexdigest()


def as_scalar(value: np.ndarray) -> object:
    return value.item() if value.shape == () else value.tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "STAGE_C_GLOBAL_PREDICTION_LOCKED").is_file():
        raise RuntimeError("Stage-C global prediction lock is missing")
    if not (run_dir / "STAGE_E_TARGETS_LOCKED").is_file():
        raise RuntimeError("Stage-E target lock is missing")
    if (run_dir / "STAGE_F_TARGET_JOIN_COMPLETE").is_file():
        print("STAGE_F_TARGET_JOIN_ALREADY_COMPLETE")
        return 0

    # Reverify every separately locked prediction and target artifact before
    # opening either payload. This is the Stage-F separation boundary.
    pred_manifest_path = run_dir / "manifests/stageC_prediction_lock_manifest.json"
    target_manifest_path = run_dir / "manifests/stageE_target_manifest.json"
    pred_manifest = json.loads(pred_manifest_path.read_text())
    target_manifest = json.loads(target_manifest_path.read_text())
    pred_rows = pred_manifest["prediction_files"]
    if not pred_manifest["global_lock_complete"] or len(pred_rows) != 1200:
        raise RuntimeError("prediction lock is incomplete")
    bad_predictions = [
        row["relative_path"]
        for row in pred_rows
        if sha_file(Path(row["path"])) != row["sha256"]
    ]
    if bad_predictions:
        raise RuntimeError(f"prediction hash failures: {bad_predictions[:3]}")
    target_info = target_manifest["target_file"]
    target_path = Path(target_info["path"])
    if sha_file(target_path) != target_info["sha256"]:
        raise RuntimeError("target JSONL hash failure")
    bad_metrics = [
        row["path"]
        for row in target_manifest["posteval_metric_files"]
        if sha_file(Path(row["path"])) != row["sha256"]
    ]
    if bad_metrics:
        raise RuntimeError(f"post-eval metric hash failures: {bad_metrics[:3]}")
    summary = target_manifest["aggregate_summary"]
    if sha_file(Path(summary["path"])) != summary["sha256"]:
        raise RuntimeError("official aggregate summary hash failure")

    scene_doc = json.loads(
        (run_dir / "manifests/effective_primary_scene_inventory.json").read_text()
    )
    scenes = sorted(scene_doc["scenes"], key=lambda row: row["selection_rank"])
    scene_ids = [row["scene_id"] for row in scenes]
    if len(scene_ids) != 100 or len(set(scene_ids)) != 100:
        raise RuntimeError("frozen primary scene coverage failure")
    rank_of = {scene_id: rank for rank, scene_id in enumerate(scene_ids)}
    tag_of = {tag: index for index, tag in enumerate(TAGS)}

    targets = [json.loads(line) for line in target_path.read_text().splitlines()]
    if len(targets) != 2400:
        raise RuntimeError(f"target count is {len(targets)}")
    expected_keys = {
        (scene_id, tag, candidate)
        for scene_id in scene_ids
        for tag in TAGS
        for candidate in range(8)
    }
    actual_keys = {
        (row["scene_id"], row["decision_tag"], row["candidate_index"])
        for row in targets
    }
    if actual_keys != expected_keys:
        raise RuntimeError("target join-key coverage mismatch")

    score = np.empty((100, 3, 8), np.float32)
    progress = np.empty_like(score)
    collision = np.empty_like(score)
    offroad = np.empty_like(score)
    planned = np.empty((100, 3, 8, 64, 2), np.float32)
    future = np.zeros_like(planned)
    future_mask = np.zeros((100, 3, 8, 64), np.bool_)
    timestamps = np.empty((100, 3), np.int64)
    target_provenance = []
    for row in targets:
        s, g, c = rank_of[row["scene_id"]], tag_of[row["decision_tag"]], row["candidate_index"]
        if row["selection_rank"] != s or c not in range(8):
            raise RuntimeError("target ordering provenance mismatch")
        value = np.asarray(row["planned_trajectory"], dtype=np.float32)
        if value.shape != (64, 2) or not np.isfinite(value).all():
            raise RuntimeError("planned trajectory contract mismatch")
        fvalue = np.asarray(row["future_ego_states_ego_t0"], dtype=np.float32)
        n = int(row["future_valid_steps"])
        if fvalue.shape != (n, 2) or not 0 < n <= 64 or not np.isfinite(fvalue).all():
            raise RuntimeError("future target contract mismatch")
        score[s, g, c] = row["official_alpasim_scene_score"]
        progress[s, g, c] = row["progress_clipped_rel"]
        collision[s, g, c] = row["collision_at_fault"]
        offroad[s, g, c] = row["offroad"]
        planned[s, g, c] = value
        future[s, g, c, :n] = fvalue
        future_mask[s, g, c, :n] = True
        if c == 0:
            timestamps[s, g] = row["decision_timestamp_us"]
        elif timestamps[s, g] != row["decision_timestamp_us"]:
            raise RuntimeError("within-group decision timestamp mismatch")
        target_provenance.append(
            {
                "scene_id": row["scene_id"],
                "decision_tag": row["decision_tag"],
                "candidate_index": c,
                "sample_id": row["sample_id"],
                "rollout_id": row["rollout_id"],
                "candidate_input_sha256": row["candidate_input_sha256"],
                "rollout_asl_sha256": row["rollout_asl_sha256"],
                "posteval_metrics_sha256": row["posteval_metrics_sha256"],
            }
        )

    shape = (len(CONDITIONS), 3, 100, 3, 8)
    pred_score = np.empty(shape, np.float32)
    pred_progress = np.empty(shape, np.float32)
    collision_logit = np.empty(shape, np.float32)
    collision_probability = np.empty(shape, np.float32)
    offroad_logit = np.empty(shape, np.float32)
    offroad_probability = np.empty(shape, np.float32)
    pred_future = np.empty(shape + (64, 2), np.float32)
    seen_prediction_keys = set()
    condition_of = {condition: index for index, condition in enumerate(CONDITIONS)}
    prediction_provenance = []
    for row in pred_rows:
        condition, seed, scene_id = row["condition"], int(row["seed"]), row["scene_id"]
        key = (scene_id, seed, condition)
        if key in seen_prediction_keys:
            raise RuntimeError(f"duplicate prediction join key {key}")
        seen_prediction_keys.add(key)
        if condition not in condition_of or seed not in range(3) or scene_id not in rank_of:
            raise RuntimeError(f"unexpected prediction join key {key}")
        with np.load(row["path"], allow_pickle=False) as values:
            if as_scalar(values["condition"]) != condition or int(as_scalar(values["seed"])) != seed:
                raise RuntimeError(f"prediction internal provenance mismatch {key}")
            if values["predicted_score"].shape != (3, 8):
                raise RuntimeError(f"prediction shape mismatch {key}")
            expected_scene = np.full((3, 8), scene_id)
            expected_tags = np.repeat(np.asarray(TAGS)[:, None], 8, axis=1)
            expected_candidates = np.repeat(np.arange(8)[None, :], 3, axis=0)
            if not np.array_equal(values["scene_id"], expected_scene):
                raise RuntimeError(f"prediction scene keys mismatch {key}")
            if not np.array_equal(values["decision_tag"], expected_tags):
                raise RuntimeError(f"prediction tag keys mismatch {key}")
            if not np.array_equal(values["candidate_index"], expected_candidates):
                raise RuntimeError(f"prediction candidate order mismatch {key}")
            if not np.array_equal(values["decision_timestamp_us"], timestamps[rank_of[scene_id]]):
                raise RuntimeError(f"prediction/target timestamp join mismatch {key}")
            indices = (condition_of[condition], seed, rank_of[scene_id])
            for output, field in (
                (pred_score, "predicted_score"),
                (pred_progress, "predicted_progress"),
                (collision_logit, "collision_logit"),
                (collision_probability, "collision_probability"),
                (offroad_logit, "offroad_logit"),
                (offroad_probability, "offroad_probability"),
                (pred_future, "predicted_future"),
            ):
                value = values[field]
                if not np.isfinite(value).all():
                    raise RuntimeError(f"non-finite prediction {field} {key}")
                output[indices] = value
        prediction_provenance.append(
            {"scene_id": scene_id, "seed": seed, "condition": condition,
             "prediction_sha256": row["sha256"], "relative_path": row["relative_path"]}
        )
    expected_prediction_keys = {
        (scene_id, seed, condition)
        for scene_id in scene_ids
        for seed in range(3)
        for condition in CONDITIONS
    }
    if seen_prediction_keys != expected_prediction_keys:
        raise RuntimeError("prediction join-key coverage mismatch")

    if not np.isfinite(score).all() or not np.isfinite(pred_score).all():
        raise RuntimeError("joined arrays contain non-finite values")
    if not set(np.unique(collision)).issubset({0.0, 1.0}):
        raise RuntimeError("collision target is not binary")
    if not set(np.unique(offroad)).issubset({0.0, 1.0}):
        raise RuntimeError("offroad target is not binary")

    joined = run_dir / "datasets/stageF_joined_candidate_arrays.npz"
    joined.parent.mkdir(parents=True, exist_ok=True)
    temporary = joined.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            conditions=np.asarray(CONDITIONS),
            scene_ids=np.asarray(scene_ids),
            tags=np.asarray(TAGS),
            candidate_indices=np.arange(8, dtype=np.int16),
            decision_timestamp_us=timestamps,
            target_score=score,
            target_progress=progress,
            target_collision=collision,
            target_offroad=offroad,
            planned_trajectory=planned,
            target_future=future,
            target_future_mask=future_mask,
            predicted_score=pred_score,
            predicted_progress=pred_progress,
            collision_logit=collision_logit,
            collision_probability=collision_probability,
            offroad_logit=offroad_logit,
            offroad_probability=offroad_probability,
            predicted_future=pred_future,
        )
    os.replace(temporary, joined)
    joined_sha = sha_file(joined)
    provenance_path = run_dir / "datasets/stageF_join_provenance.json"
    provenance_sha = write_json(
        provenance_path,
        {
            "task": "t26f_b3_a_stageF_join_provenance",
            "created_utc": utc(),
            "target_records": target_provenance,
            "prediction_files": prediction_provenance,
        },
    )
    manifest = {
        "task": "t26f_b3_a_stageF_join_manifest",
        "created_utc": utc(),
        "prediction_manifest_sha256": sha_file(pred_manifest_path),
        "target_manifest_sha256": sha_file(target_manifest_path),
        "all_1200_prediction_hashes_reverified": True,
        "all_2400_posteval_metric_hashes_reverified": True,
        "target_jsonl_hash_reverified": True,
        "aggregate_summary_hash_reverified": True,
        "join_keys": ["scene_id", "decision_tag", "candidate_index"],
        "coverage": {"scenes": 100, "groups": 300, "candidates": 2400,
                     "seeds": 3, "conditions": list(CONDITIONS)},
        "joined_arrays": {"path": str(joined), "sha256": joined_sha,
                          "bytes": joined.stat().st_size},
        "provenance": {"path": str(provenance_path), "sha256": provenance_sha,
                       "bytes": provenance_path.stat().st_size},
        "sealed_labels_opened_hashed_copied": False,
    }
    manifest_sha = write_json(run_dir / "manifests/stageF_join_manifest.json", manifest)
    qa = {
        "task": "t26f_b3_a_stageF_join_qa",
        "created_utc": utc(),
        "checks": {
            "prediction_keys_exact": seen_prediction_keys == expected_prediction_keys,
            "target_keys_exact": actual_keys == expected_keys,
            "joined_hash_revalidates": sha_file(joined) == joined_sha,
            "provenance_hash_revalidates": sha_file(provenance_path) == provenance_sha,
            "all_values_finite": bool(np.isfinite(score).all() and np.isfinite(pred_score).all()),
            "sealed_labels_untouched": True,
        },
    }
    qa["all_pass"] = all(qa["checks"].values())
    write_json(run_dir / "qa/stageF_join_qa.json", qa)
    if not qa["all_pass"]:
        raise RuntimeError(f"Stage-F join QA failed: {qa}")
    (run_dir / "STAGE_F_TARGET_JOIN_COMPLETE").write_text(
        f"completed_utc: {utc()}\nmanifest_sha256: {manifest_sha}\n"
        f"joined_arrays_sha256: {joined_sha}\n"
    )
    print(f"STAGE_F_TARGET_JOIN_COMPLETE sha256={joined_sha}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOCKED_T26F_B3_A_TARGET_JOIN_FAILURE: {type(exc).__name__}: {exc}")
        raise

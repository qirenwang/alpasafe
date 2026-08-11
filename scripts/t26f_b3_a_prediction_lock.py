#!/usr/bin/env python
"""T26F-B3-A Stage C global prediction lock, with no target access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


B3_0 = Path(
    "/storage/alpasafe/safeworld-alpamayo/artifacts/"
    "safeworld_t26f_b3_0_preregistration/20260718T180730Z"
)
MODEL_SOURCE = Path(
    "/storage/alpasafe/safeworld-alpamayo/artifacts/"
    "safeworld_t26f_b1_scenelatent_bounded_pilot/20260716T175100Z/"
    "code_artifacts/t26f_b1_models.py"
)
MODEL_SOURCE_SHA256 = "81cf4fb82d18b95ae75f263e9bce5832e3a1e48ea66281d6853d446002443464"
CHECKPOINTS = {
    (arm, seed): B3_0 / f"models/final_ab/final_{arm}_seed{seed}.pt"
    for arm in ("A", "B")
    for seed in range(3)
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text)
    os.replace(temp, path)
    return hashlib.sha256(text.encode()).hexdigest()


def load_l3(path: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as handle:
        if list(handle.keys()) != ["L3"]:
            raise RuntimeError(f"non-L3 keys in {path}")
        value = handle.get_tensor("L3")
    if value.shape != torch.Size([4096]) or value.dtype != torch.bfloat16:
        raise RuntimeError(f"L3 contract mismatch in {path}")
    return value


def load_trajectories(path: str) -> torch.Tensor:
    document = json.loads(Path(path).read_text())
    value = np.asarray(
        [row["trajectory_xy"] for row in document["candidates"]], dtype=np.float32
    )
    if value.shape != (8, 64, 2) or not np.isfinite(value).all():
        raise RuntimeError(f"trajectory contract mismatch in {path}")
    return torch.from_numpy(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "STAGE_B_INPUTS_LOCKED").is_file():
        raise RuntimeError("STAGE_B_INPUTS_LOCKED is missing")
    if (run_dir / "STAGE_C_GLOBAL_PREDICTION_LOCKED").is_file():
        print("STAGE_C_GLOBAL_PREDICTION_ALREADY_LOCKED")
        return 0
    if (run_dir / "stageD_rollouts").exists() or (run_dir / "targets").exists():
        raise RuntimeError("rollout/target path exists before global prediction lock")
    if sha_file(MODEL_SOURCE) != MODEL_SOURCE_SHA256:
        raise RuntimeError("frozen SafeWorldV2 model source hash mismatch")
    sys.path.insert(0, str(MODEL_SOURCE.parent))
    from t26f_b1_models import SafeWorldV2, parameter_count

    checkpoint_manifest = json.loads(
        (B3_0 / "manifests/t26f_b3_0_final_checkpoint_manifest.json").read_text()
    )
    checkpoint_rows = []
    for key, path in CHECKPOINTS.items():
        relative = str(path.relative_to(B3_0))
        actual = sha_file(path)
        expected = checkpoint_manifest["sha256"][relative]
        if actual != expected:
            raise RuntimeError(f"checkpoint hash mismatch: {relative}")
        checkpoint_rows.append(
            {"arm": key[0], "seed": key[1], "path": str(path), "sha256": actual}
        )

    preprocessing = json.loads(
        (B3_0 / "manifests/t26f_b3_0_full_dev_preprocessing_manifest.json").read_text()
    )
    mean = torch.tensor(preprocessing["plan_mean_xy"], dtype=torch.float32)
    std = torch.tensor(preprocessing["plan_std_xy"], dtype=torch.float32)
    if preprocessing["sha256_of_values"] != "0f2b5d1532bf197f87bf43f2feb0375c64bd4849daa28d1bcacfeb410bc16ae7":
        raise RuntimeError("preprocessing value hash mismatch")

    stage_b = json.loads(
        (run_dir / "manifests/stageB_candidate_l3_manifest.json").read_text()
    )
    groups = stage_b["groups"]
    if len(groups) != 300:
        raise RuntimeError(f"Stage-B group count is {len(groups)}")
    group_by_key = {(row["scene_id"], row["decision_tag"]): row for row in groups}
    effective_inventory_path = run_dir / "manifests/effective_primary_scene_inventory.json"
    scenes = json.loads(effective_inventory_path.read_text())["scenes"]
    effective_mapping_path = run_dir / "manifests/effective_wrong_scene_l3_mapping.json"
    mapping_path = (
        effective_mapping_path
        if effective_mapping_path.is_file()
        else B3_0 / "scene_selection/t26f_b3_0_wrong_scene_l3_mapping.json"
    )
    mapping = json.loads(mapping_path.read_text())["mapping"]
    tags = ("A", "B", "C")

    model_cache = {}
    for arm in ("A", "B"):
        for seed in range(3):
            checkpoint = torch.load(CHECKPOINTS[(arm, seed)], map_location="cpu", weights_only=False)
            if checkpoint["arm"] != arm or checkpoint["seed"] != seed:
                raise RuntimeError(f"checkpoint interface mismatch {arm} seed{seed}")
            model = SafeWorldV2(arm)
            expected_count = 201_988 if arm == "A" else 734_596
            if parameter_count(model)["total"] != expected_count:
                raise RuntimeError(f"parameter count mismatch {arm}")
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.eval()
            model_cache[(arm, seed)] = model

    predictions = run_dir / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifests/stageC_prediction_lock_manifest.json"
    existing = {}
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        existing = {row["relative_path"]: row for row in prior.get("prediction_files", [])}
    files = []
    total = 100 * 3 * 4
    count = 0
    for scene in scenes:
        scene_id = scene["scene_id"]
        rank = scene["selection_rank"]
        trajectory_groups = [load_trajectories(group_by_key[(scene_id, tag)]["candidate_dump_path"]) for tag in tags]
        own_l3 = [load_l3(group_by_key[(scene_id, tag)]["shared_l3_path"]) for tag in tags]
        wrong_l3 = []
        wrong_donors = []
        for tag in tags:
            donor = mapping[tag][scene_id]
            if donor == scene_id:
                raise RuntimeError("wrong-scene mapping self-donor")
            wrong_donors.append(donor)
            wrong_l3.append(load_l3(group_by_key[(donor, tag)]["shared_l3_path"]))

        for seed in range(3):
            conditions = (
                ("A_NORMAL", "A", [None, None, None]),
                ("B_NORMAL", "B", own_l3),
                ("B_ZERO", "B", [torch.zeros(4096, dtype=torch.bfloat16)] * 3),
                ("B_WRONG_SCENE", "B", wrong_l3),
            )
            for condition, arm, latents in conditions:
                count += 1
                relative = f"rank{rank:03d}/seed{seed}_{condition}.npz"
                output = predictions / relative
                previous = existing.get(relative)
                if previous and output.is_file() and sha_file(output) == previous["sha256"]:
                    files.append(previous)
                    print(f"[{count:04d}/{total}] resume {relative}", flush=True)
                    continue
                model = model_cache[(arm, seed)]
                collected = {
                    "predicted_score": [],
                    "predicted_future": [],
                    "predicted_progress": [],
                    "collision_logit": [],
                    "collision_probability": [],
                    "offroad_logit": [],
                    "offroad_probability": [],
                }
                with torch.no_grad():
                    for trajectory, latent in zip(trajectory_groups, latents, strict=True):
                        result = model((trajectory - mean) / std, trajectory, latent)
                        collected["predicted_score"].append(result["predicted_score"].numpy())
                        collected["predicted_future"].append(result["predicted_future"].numpy())
                        collected["predicted_progress"].append(result["progress_pred"].numpy())
                        collected["collision_logit"].append(result["collision_logit"].numpy())
                        collected["collision_probability"].append(torch.sigmoid(result["collision_logit"]).numpy())
                        collected["offroad_logit"].append(result["offroad_logit"].numpy())
                        collected["offroad_probability"].append(torch.sigmoid(result["offroad_logit"]).numpy())
                tags_array = np.asarray(tags, dtype="U1")
                scene_array = np.full((3, 8), scene_id, dtype=f"U{len(scene_id)}")
                tag_array = np.repeat(tags_array[:, None], 8, axis=1)
                candidate_indices = np.repeat(np.arange(8, dtype=np.int16)[None, :], 3, axis=0)
                timestamps = np.asarray(
                    [group_by_key[(scene_id, tag)]["decision_timestamp_us"] for tag in tags],
                    dtype=np.int64,
                )
                prefix_mask = np.asarray(
                    [[index < k for index in range(8)] for k in (2, 5, 8)], dtype=np.bool_
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output,
                    **{key: np.stack(value) for key, value in collected.items()},
                    scene_id=scene_array,
                    decision_tag=tag_array,
                    decision_timestamp_us=timestamps,
                    candidate_index=candidate_indices,
                    candidate_order=np.arange(8, dtype=np.int16),
                    k_values=np.asarray([2, 5, 8], dtype=np.int16),
                    k_prefix_mask=prefix_mask,
                    arm=np.asarray(arm),
                    condition=np.asarray(condition),
                    seed=np.asarray(seed, dtype=np.int16),
                    wrong_scene_donor=np.asarray(wrong_donors if condition == "B_WRONG_SCENE" else ["", "", ""]),
                )
                row = {
                    "selection_rank": rank,
                    "scene_id": scene_id,
                    "seed": seed,
                    "arm": arm,
                    "condition": condition,
                    "relative_path": relative,
                    "path": str(output),
                    "sha256": sha_file(output),
                    "bytes": output.stat().st_size,
                    "wrong_scene_donors": wrong_donors if condition == "B_WRONG_SCENE" else None,
                }
                files.append(row)
                checkpoint = {
                    "task": "t26f_b3_a_stageC_prediction_lock",
                    "updated_utc": utc(),
                    "global_lock_complete": False,
                    "target_or_rollout_inputs_read": False,
                    "n_expected_files": 1200,
                    "n_prediction_files": len(files),
                    "prediction_files": files,
                }
                write_json(manifest_path, checkpoint)
                print(f"[{count:04d}/{total}] locked {relative}", flush=True)

    complete = {
        "task": "t26f_b3_a_stageC_prediction_lock",
        "locked_utc": utc(),
        "global_lock_complete": True,
        "before_first_stageD_rollout": True,
        "target_or_rollout_inputs_read": False,
        "n_scenes": 100,
        "n_seeds": 3,
        "conditions": ["A_NORMAL", "B_NORMAL", "B_ZERO", "B_WRONG_SCENE"],
        "n_expected_files": 1200,
        "n_prediction_files": len(files),
        "n_candidate_predictions_per_file": 24,
        "n_prediction_records_per_condition": 2400 * 3,
        "checkpoint_verification": checkpoint_rows,
        "model_source": str(MODEL_SOURCE),
        "model_source_sha256": MODEL_SOURCE_SHA256,
        "preprocessing_sha256_of_values": preprocessing["sha256_of_values"],
        "effective_inventory_sha256": sha_file(effective_inventory_path),
        "wrong_scene_mapping_path": str(mapping_path),
        "wrong_scene_mapping_sha256": sha_file(mapping_path),
        "prediction_files": files,
    }
    if len(files) != 1200 or len({row["relative_path"] for row in files}) != 1200:
        raise RuntimeError("prediction file coverage failure")
    digest = write_json(manifest_path, complete)
    qa = {
        "task": "t26f_b3_a_stageC_prediction_lock_qa",
        "created_utc": utc(),
        "checks": {
            "files_1200": len(files) == 1200,
            "all_hashes_revalidate": all(sha_file(Path(row["path"])) == row["sha256"] for row in files),
            "six_checkpoints_verified": len(checkpoint_rows) == 6,
            "model_source_verified": sha_file(MODEL_SOURCE) == MODEL_SOURCE_SHA256,
            "targets_not_read": True,
            "rollouts_not_started": not (run_dir / "stageD_rollouts").exists(),
            "wrong_scene_donor_differs": all(
                donor != row["scene_id"]
                for row in files
                if row["condition"] == "B_WRONG_SCENE"
                for donor in row["wrong_scene_donors"]
            ),
        },
    }
    qa["all_pass"] = all(qa["checks"].values())
    write_json(run_dir / "qa/stageC_prediction_lock_qa.json", qa)
    write_json(
        run_dir / "audit/six_checkpoint_verification.json",
        {
            "task": "t26f_b3_a_six_checkpoint_verification",
            "created_utc": utc(),
            "n_checkpoints": len(checkpoint_rows),
            "all_hashes_match_frozen_manifest": len(checkpoint_rows) == 6,
            "parameter_counts": {"A": 201988, "B": 734596},
            "frozen_model_source_sha256": MODEL_SOURCE_SHA256,
            "checkpoints": checkpoint_rows,
        },
    )
    if not qa["all_pass"]:
        raise RuntimeError(f"prediction-lock QA failure: {qa}")
    (run_dir / "STAGE_C_GLOBAL_PREDICTION_LOCKED").write_text(
        f"locked_utc: {utc()}\nmanifest_sha256: {digest}\n"
        "n_scenes: 100\nn_prediction_files: 1200\nfirst_stageD_rollout_allowed: true\n"
    )
    print(f"STAGE_C_GLOBAL_PREDICTION_LOCKED sha256={digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOCKED_T26F_B3_A_PREDICTION_LOCK_FAILURE: {type(exc).__name__}: {exc}")
        raise

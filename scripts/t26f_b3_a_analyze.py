#!/usr/bin/env python
"""T26F-B3-A frozen Stage-F metrics, endpoints, and interpretation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


B3_0 = Path(
    "/storage/alpasafe/safeworld-alpamayo/artifacts/"
    "safeworld_t26f_b3_0_preregistration/20260718T180730Z"
)
REPO = Path("/home/qiren/alpasafe/safeworld-alpamayo")
TRAIN_PILOT_SHA = "2194b71cf8e3f3f222f803f4101d2c754ae012b9c98749919e17682bc9bbc306"
SCENE_CV_SHA = "e65aa0293c591f98cc9229f123f2db4b451950fdfcedfd40edb00d6b18a4e764"
CONDITIONS = ("A_NORMAL", "B_NORMAL", "B_ZERO", "B_WRONG_SCENE")
K_VIEWS_LOCAL = {"K2": 2, "K5": 5, "K8": 8}
BOOT_N = 10_000
BOOT_SEED = 20260718
TIE_TOL = 1e-12


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
    if "t26e_test_labels_sealed.jsonl" in str(path):
        raise RuntimeError("sealed-label access refused")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=1, sort_keys=True, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload.encode()).hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def auroc(target: np.ndarray, probability: np.ndarray) -> float | None:
    positive, negative = probability[target > 0.5], probability[target <= 0.5]
    if len(positive) == 0 or len(negative) == 0:
        return None
    combined = np.concatenate([positive, negative])
    order = np.argsort(combined, kind="stable")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    for value in np.unique(combined):
        mask = combined == value
        ranks[mask] = ranks[mask].mean()
    return float(
        (ranks[: len(positive)].sum() - len(positive) * (len(positive) + 1) / 2)
        / (len(positive) * len(negative))
    )


def auprc(target: np.ndarray, probability: np.ndarray) -> float | None:
    labels = target > 0.5
    if labels.sum() == 0 or (~labels).sum() == 0:
        return None
    order = np.argsort(-probability, kind="stable")
    ordered = labels[order].astype(np.float64)
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float((precision * ordered).sum() / ordered.sum())


def paired_summary(deltas: np.ndarray, direction: str, scene_ids: list[str]) -> dict:
    if deltas.shape != (100,):
        raise RuntimeError(f"paired endpoint is not 100 scenes: {deltas.shape}")
    rng = np.random.default_rng(BOOT_SEED)
    indices = rng.integers(0, len(deltas), size=(BOOT_N, len(deltas)))
    means = deltas[indices].mean(axis=1)
    if direction == "negative":
        extreme = int((means >= 0).sum())
        hypothesis = "preregistered direction: mean < 0"
    elif direction == "positive":
        extreme = int((means <= 0).sum())
        hypothesis = "preregistered direction: mean > 0"
    else:
        raise ValueError(direction)
    sorted_delta = np.sort(deltas)
    trimmed = sorted_delta[10:-10]
    full_mean = float(deltas.mean())
    loso = (deltas.sum() - deltas) / (len(deltas) - 1)
    largest = int(np.argmax(np.abs(deltas)))
    return {
        "mean": full_mean,
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "one_sided_bootstrap_p": float((extreme + 1) / (BOOT_N + 1)),
        "one_sided_hypothesis": hypothesis,
        "n_resamples": BOOT_N,
        "bootstrap_seed": BOOT_SEED,
        "n_scenes": 100,
        "median_paired_scene_difference": float(np.median(deltas)),
        "trimmed_mean_10pct_descriptive": float(trimmed.mean()),
        "leave_one_scene_out_descriptive": {
            "min": float(loso.min()),
            "max": float(loso.max()),
            "mean": float(loso.mean()),
            "maximum_absolute_shift_from_full_mean": float(np.max(np.abs(loso - full_mean))),
        },
        "maximum_single_scene_contribution": {
            "scene_id": scene_ids[largest],
            "paired_delta": float(deltas[largest]),
            "contribution_to_mean": float(deltas[largest] / len(deltas)),
        },
        "improved_scenes": int((deltas < -TIE_TOL).sum()),
        "tied_scenes": int((np.abs(deltas) <= TIE_TOL).sum()),
        "worsened_scenes": int((deltas > TIE_TOL).sum()),
        "per_scene_delta": {
            scene_id: float(value) for scene_id, value in zip(scene_ids, deltas, strict=True)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "STAGE_F_TARGET_JOIN_COMPLETE").is_file():
        raise RuntimeError("Stage-F target join is missing")

    train_pilot = REPO / "src/safeworld/t26e/train_pilot.py"
    scene_cv = REPO / "src/safeworld/t26e/scene_cv.py"
    if sha_file(train_pilot) != TRAIN_PILOT_SHA or sha_file(scene_cv) != SCENE_CV_SHA:
        raise RuntimeError("frozen metric implementation hash mismatch")
    sys.path.insert(0, str(REPO / "src"))
    from safeworld.t26e.scene_cv import rank0_regret_per_group
    from safeworld.t26e.train_pilot import (
        K_VIEWS, candidate_metrics, future_metrics, group_metrics, spearman,
    )
    if K_VIEWS != K_VIEWS_LOCAL:
        raise RuntimeError(f"frozen K views mismatch: {K_VIEWS}")

    join_manifest_path = run_dir / "manifests/stageF_join_manifest.json"
    join_manifest = json.loads(join_manifest_path.read_text())
    arrays_path = Path(join_manifest["joined_arrays"]["path"])
    if sha_file(arrays_path) != join_manifest["joined_arrays"]["sha256"]:
        raise RuntimeError("joined-array hash mismatch")
    z = np.load(arrays_path, allow_pickle=False)
    conditions = tuple(z["conditions"].tolist())
    scene_ids = z["scene_ids"].tolist()
    tags = z["tags"].tolist()
    if conditions != CONDITIONS or len(scene_ids) != 100 or tags != ["A", "B", "C"]:
        raise RuntimeError("joined-array axis contract mismatch")
    target = z["target_score"].astype(np.float64)
    predicted = z["predicted_score"].astype(np.float64)
    if target.shape != (100, 3, 8) or predicted.shape != (4, 3, 100, 3, 8):
        raise RuntimeError("joined prediction/target shape mismatch")
    group_ids = [
        f"{scene_ids[s]}@{int(z['decision_timestamp_us'][s, g])}"
        for s in range(100) for g in range(3)
    ]
    meta = [
        {
            "sample_id": f"{group_id}#cand_{candidate}",
            "decision_group_id": group_id,
            "candidate_id": f"cand_{candidate}",
            "candidate_index": candidate,
        }
        for group_id in group_ids for candidate in range(8)
    ]
    target_flat = target.reshape(-1)

    metric_results: dict[str, dict[str, Any]] = {}
    group_regrets: dict[tuple[int, int, str], np.ndarray] = {}
    selections: dict[tuple[int, int, str], np.ndarray] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        metric_results[condition] = {}
        for seed in range(3):
            pred = predicted[condition_index, seed].reshape(-1)
            seed_entry: dict[str, Any] = {"k_views": {}}
            for k_name, k in K_VIEWS.items():
                gm = group_metrics(pred, target_flat, meta, k)
                candidate_mask = np.asarray([row["candidate_index"] < k for row in meta])
                cp, ct = pred[candidate_mask].copy(), target_flat[candidate_mask].copy()
                for start in range(0, len(cp), k):
                    cp[start : start + k] -= cp[start : start + k].mean()
                    ct[start : start + k] -= ct[start : start + k].mean()
                cm = candidate_metrics(pred[candidate_mask], target_flat[candidate_mask])
                r0 = rank0_regret_per_group(target_flat, meta, k)
                selected_scores = [row["selected_official_score"] for row in gm["per_group"]]
                best_scores = [row["best_official_score"] for row in gm["per_group"]]
                seed_entry["k_views"][k_name] = {
                    "selected_official_score_mean": float(np.mean(selected_scores)),
                    "mean_regret": gm["mean_selected_score_regret"],
                    "median_regret": gm["median_selected_score_regret"],
                    "top1_agreement": gm["top1_selection_accuracy"],
                    "pairwise_ranking_accuracy": gm["pairwise_ranking_accuracy"],
                    "group_centered_spearman": spearman(cp, ct),
                    "candidate_score_mae": cm["mae"],
                    "candidate_score_rmse": cm["rmse"],
                    "candidate_score_prediction_std": cm["prediction_std"],
                    "rank0_mean_regret": float(np.mean(list(r0.values()))),
                    "random_mean_regret": float(np.mean([
                        best - np.mean(target[s, g, :k])
                        for s in range(100) for g in range(3)
                        for best in [float(target[s, g, :k].max())]
                    ])),
                    "oracle_best_in_k_official_score_mean": float(np.mean(best_scores)),
                    "oracle_mean_regret": 0.0,
                    "regret_minus_rank0": gm["mean_selected_score_regret"]
                    - float(np.mean(list(r0.values()))),
                    "predicted_tie_count": gm["predicted_tie_count"],
                    "official_tie_count": gm["official_tie_count"],
                }
                per_group_by_id = {
                    row["decision_group_id"]: row for row in gm["per_group"]
                }
                group_regrets[(condition_index, seed, k_name)] = np.asarray(
                    [per_group_by_id[group_id]["selected_score_regret"] for group_id in group_ids],
                    dtype=np.float64,
                ).reshape(100, 3)
                selections[(condition_index, seed, k_name)] = np.asarray(
                    [per_group_by_id[group_id]["selected_candidate_index"] for group_id in group_ids],
                    dtype=np.int16,
                ).reshape(100, 3)
            pf = torch.from_numpy(z["predicted_future"][condition_index, seed]).reshape(-1, 64, 2)
            tf = torch.from_numpy(z["target_future"]).reshape(-1, 64, 2)
            tm = torch.from_numpy(z["target_future_mask"]).reshape(-1, 64)
            seed_entry["future"] = future_metrics(pf, tf, tm)
            seed_entry["candidate_score_global"] = candidate_metrics(pred, target_flat)
            seed_entry["auxiliary"] = {}
            for name, target_name, probability_name in (
                ("collision", "target_collision", "collision_probability"),
                ("offroad", "target_offroad", "offroad_probability"),
            ):
                binary_target = z[target_name].reshape(-1).astype(np.float64)
                probability = z[probability_name][condition_index, seed].reshape(-1).astype(np.float64)
                positives = int((binary_target > 0.5).sum())
                negatives = int((binary_target <= 0.5).sum())
                seed_entry["auxiliary"][name] = {
                    "positives": positives,
                    "negatives": negatives,
                    "auroc": auroc(binary_target, probability),
                    "auprc": auprc(binary_target, probability),
                    "brier": float(np.mean((probability - binary_target) ** 2)),
                    "prevalence": float((binary_target > 0.5).mean()),
                    "mean_predicted_probability": float(probability.mean()),
                }
            pred_progress = z["predicted_progress"][condition_index, seed].reshape(-1)
            target_progress = z["target_progress"].reshape(-1)
            progress_error = pred_progress - target_progress
            seed_entry["auxiliary"]["progress_clipped_rel"] = {
                "mae": float(np.abs(progress_error).mean()),
                "rmse": float(np.sqrt(np.mean(progress_error**2))),
                "spearman": spearman(pred_progress, target_progress),
            }
            metric_results[condition][f"seed_{seed}"] = seed_entry

    raw_plan = torch.from_numpy(z["planned_trajectory"]).reshape(-1, 64, 2)
    target_future = torch.from_numpy(z["target_future"]).reshape(-1, 64, 2)
    target_mask = torch.from_numpy(z["target_future_mask"]).reshape(-1, 64)
    raw_plan_future = future_metrics(raw_plan, target_future, target_mask)

    # Regret macro hierarchy is frozen: group -> scene -> seed -> paired scene.
    scene_macro: dict[str, dict[str, np.ndarray]] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        scene_macro[condition] = {}
        for k_name in K_VIEWS:
            scene_macro[condition][k_name] = np.stack([
                group_regrets[(condition_index, seed, k_name)].mean(axis=1)
                for seed in range(3)
            ])

    def comparison(left: str, right: str, k_name: str, direction: str) -> dict:
        left_array = scene_macro[left][k_name]
        right_array = scene_macro[right][k_name]
        delta = left_array.mean(axis=0) - right_array.mean(axis=0)
        result = paired_summary(delta, direction, scene_ids)
        result.update({
            "comparison": f"regret_{left} - regret_{right}",
            "k_view": k_name,
            "calculation_order": "group -> scene mean(3) -> seed mean(3) -> paired scene -> mean(100)",
            "per_seed_point": {
                f"seed_{seed}": float((left_array[seed] - right_array[seed]).mean())
                for seed in range(3)
            },
        })
        return result

    primary = {k: comparison("B_NORMAL", "A_NORMAL", k, "negative") for k in K_VIEWS}
    wrong = {k: comparison("B_WRONG_SCENE", "B_NORMAL", k, "positive") for k in K_VIEWS}
    zero = {k: comparison("B_ZERO", "B_NORMAL", k, "positive") for k in K_VIEWS}

    change_rates = {}
    for k_name in K_VIEWS:
        normal_selection = np.stack([selections[(1, seed, k_name)] for seed in range(3)])
        wrong_selection = np.stack([selections[(3, seed, k_name)] for seed in range(3)])
        zero_selection = np.stack([selections[(2, seed, k_name)] for seed in range(3)])
        change_rates[k_name] = {
            "wrong_vs_normal": float(np.mean(wrong_selection != normal_selection)),
            "zero_vs_normal": float(np.mean(zero_selection != normal_selection)),
            "denominator_seed_groups": int(normal_selection.size),
            "wrong_vs_normal_changed": int((wrong_selection != normal_selection).sum()),
            "zero_vs_normal_changed": int((zero_selection != normal_selection).sum()),
        }

    diagnostic_metric_changes = {}
    for k_name in K_VIEWS:
        diagnostic_metric_changes[k_name] = {}
        for variant in ("B_WRONG_SCENE", "B_ZERO"):
            spearman_per_seed = {
                f"seed_{seed}": (
                    metric_results[variant][f"seed_{seed}"]["k_views"][k_name][
                        "group_centered_spearman"
                    ]
                    - metric_results["B_NORMAL"][f"seed_{seed}"]["k_views"][k_name][
                        "group_centered_spearman"
                    ]
                )
                for seed in range(3)
            }
            variance_per_seed = {
                f"seed_{seed}": (
                    metric_results[variant][f"seed_{seed}"]["k_views"][k_name][
                        "candidate_score_prediction_std"
                    ] ** 2
                    - metric_results["B_NORMAL"][f"seed_{seed}"]["k_views"][k_name][
                        "candidate_score_prediction_std"
                    ] ** 2
                )
                for seed in range(3)
            }
            diagnostic_metric_changes[k_name][variant] = {
                "group_centered_spearman_change_vs_B_NORMAL_per_seed": spearman_per_seed,
                "group_centered_spearman_change_vs_B_NORMAL_mean": float(
                    np.mean(list(spearman_per_seed.values()))
                ),
                "candidate_score_variance_change_vs_B_NORMAL_per_seed": variance_per_seed,
                "candidate_score_variance_change_vs_B_NORMAL_mean": float(
                    np.mean(list(variance_per_seed.values()))
                ),
            }

    scene_doc = json.loads(
        (run_dir / "manifests/effective_primary_scene_inventory.json").read_text()
    )
    scene_rows = sorted(scene_doc["scenes"], key=lambda row: row["selection_rank"])
    primary_delta = np.asarray(list(primary["K8"]["per_scene_delta"].values()))
    cohorts = {}
    for field in ("country", "time_of_day", "ego_speed"):
        cohorts[field] = {}
        for value in sorted({row[field] for row in scene_rows}):
            mask = np.asarray([row[field] == value for row in scene_rows])
            cohorts[field][value] = {
                "n_scenes": int(mask.sum()),
                "Delta_BA_K8_mean": float(primary_delta[mask].mean()),
                "Delta_BA_K8_median": float(np.median(primary_delta[mask])),
            }

    effective_mapping_path = run_dir / "manifests/effective_wrong_scene_l3_mapping.json"
    mapping_path = (
        effective_mapping_path
        if effective_mapping_path.is_file()
        else B3_0 / "scene_selection/t26f_b3_0_wrong_scene_l3_mapping.json"
    )
    mapping = json.loads(mapping_path.read_text())
    mapping_values = mapping["mapping"]
    mapping_valid = all(
        set(mapping_values[tag]) == set(scene_ids)
        and set(mapping_values[tag].values()) == set(scene_ids)
        and all(recipient != donor for recipient, donor in mapping_values[tag].items())
        for tag in tags
    )
    technical_gates = {
        "stageA_assets_verified": (run_dir / "STAGE_A_ASSETS_VERIFIED").is_file(),
        "stageA_technical_validation": (run_dir / "STAGE_A_TECHNICAL_VALIDATION_PASS").is_file(),
        "stageB_inputs_locked": (run_dir / "STAGE_B_INPUTS_LOCKED").is_file(),
        "stageB_qa": json.loads((run_dir / "qa/stageB_generation_qa.json").read_text())["all_pass"],
        "stageC_global_prediction_lock": (run_dir / "STAGE_C_GLOBAL_PREDICTION_LOCKED").is_file(),
        "stageC_qa": json.loads((run_dir / "qa/stageC_prediction_lock_qa.json").read_text())["all_pass"],
        "stageD_rollouts_complete": (run_dir / "STAGE_D_ROLLOUTS_COMPLETE").is_file(),
        "stageD_qa": json.loads((run_dir / "qa/stageD_rollout_qa.json").read_text())["all_pass"],
        "stageE_targets_locked": (run_dir / "STAGE_E_TARGETS_LOCKED").is_file(),
        "stageE_qa": json.loads((run_dir / "qa/stageE_target_qa.json").read_text())["all_pass"],
        "stageF_join_qa": json.loads((run_dir / "qa/stageF_join_qa.json").read_text())["all_pass"],
        "metric_implementation_hashes": True,
        "wrong_scene_mapping_valid": mapping_valid,
    }
    all_gates = all(technical_gates.values())

    k8 = primary["K8"]
    confirmation_conditions = {
        "Delta_BA_K8_negative": k8["mean"] < 0,
        "K8_ci_upper_negative": k8["ci95_high"] < 0,
        "K2_point_le_plus_0_002": primary["K2"]["mean"] <= 0.002,
        "K5_point_le_plus_0_002": primary["K5"]["mean"] <= 0.002,
        "no_seed_K8_Delta_BA_gt_plus_0_005": max(k8["per_seed_point"].values()) <= 0.005,
        "all_technical_prediction_lock_evaluation_gates": all_gates,
    }
    if all(confirmation_conditions.values()):
        performance_status = "T26F_B3_EXTERNAL_PERFORMANCE_GAIN_CONFIRMED"
    elif k8["mean"] < 0:
        performance_status = "T26F_B3_DIRECTIONAL_EXTERNAL_GAIN_ONLY"
    elif k8["ci95_low"] > 0:
        # The frozen text defines reversal as materially worse with interval
        # support and supplies no separate numerical materiality threshold.
        # Therefore an interval-supported positive preregistered endpoint is
        # the direct, non-post-hoc operationalization.
        performance_status = "T26F_B3_EXTERNAL_REVERSAL"
    else:
        performance_status = "T26F_B3_NO_EXTERNAL_L3_GAIN"

    wrong_k8 = wrong["K8"]
    mechanism_conditions = {
        "Delta_WRONG_NORMAL_K8_positive": wrong_k8["mean"] > 0,
        "K8_ci_lower_positive": wrong_k8["ci95_low"] > 0,
        "selected_candidate_change_rate_reported": "K8" in change_rates,
        "mapping_valid_never_same_scene": mapping_valid,
    }
    if all(mechanism_conditions.values()):
        mechanism_status = "T26F_B3_SCENE_SPECIFIC_L3_USE_CONFIRMED"
    elif wrong_k8["mean"] > 0:
        mechanism_status = "T26F_B3_DIRECTIONAL_SCENE_SPECIFIC_L3_USE"
    else:
        mechanism_status = "T26F_B3_NO_SCENE_SPECIFIC_L3_SUPPORT"

    if performance_status == "T26F_B3_EXTERNAL_PERFORMANCE_GAIN_CONFIRMED":
        combined = (
            "performance confirmed + mechanism confirmed"
            if mechanism_status == "T26F_B3_SCENE_SPECIFIC_L3_USE_CONFIRMED"
            else "performance confirmed + mechanism unresolved"
        )
    elif performance_status == "T26F_B3_DIRECTIONAL_EXTERNAL_GAIN_ONLY":
        combined = "directional performance only"
    elif performance_status == "T26F_B3_EXTERNAL_REVERSAL":
        combined = "external reversal"
    else:
        combined = "no external L3 gain"

    created = utc()
    metric_output = {
        "task": "t26f_b3_a_frozen_metrics",
        "created_utc": created,
        "metric_sources": {
            "train_pilot_sha256": TRAIN_PILOT_SHA,
            "scene_cv_sha256": SCENE_CV_SHA,
        },
        "systems": metric_results,
        "raw_plan_future_baseline": raw_plan_future,
        "references_note": "RAND analytic uniform selector; R0 first candidate; ORACLE best official score in prefix",
    }
    endpoint_output = {
        "task": "t26f_b3_a_preregistered_scene_level_analysis",
        "created_utc": created,
        "primary_Delta_BA": primary,
        "mechanism_Delta_WRONG_NORMAL": wrong,
        "diagnostic_Delta_ZERO_NORMAL": zero,
        "selected_candidate_change_rates": change_rates,
        "diagnostic_metric_changes": diagnostic_metric_changes,
        "cohort_summaries_descriptive": cohorts,
        "wrong_scene_mapping": {
            "namespace": mapping.get("namespace"),
            "sha256": sha_file(mapping_path),
            "valid_bijection_per_tag_never_same_scene": mapping_valid,
        },
    }
    interpretation_output = {
        "task": "t26f_b3_a_frozen_interpretation",
        "created_utc": created,
        "performance_status": performance_status,
        "performance_confirmation_conditions": confirmation_conditions,
        "mechanism_status": mechanism_status,
        "mechanism_confirmation_conditions": mechanism_conditions,
        "combined_interpretation": combined,
        "technical_gates": technical_gates,
        "all_gates_pass": all_gates,
        "prohibited_claims_respected": True,
        "reversal_rule_note": "interval-supported positive Delta_BA (CI lower > 0); frozen contract provides no additional numerical materiality threshold",
    }
    for value in (metric_output, endpoint_output, interpretation_output):
        if not finite(value):
            raise RuntimeError("non-finite result detected")
    metric_sha = write_json(run_dir / "results/t26f_b3_a_metrics.json", metric_output)
    endpoint_sha = write_json(run_dir / "results/t26f_b3_a_scene_level_analysis.json", endpoint_output)
    interpretation_sha = write_json(
        run_dir / "results/t26f_b3_a_interpretation.json", interpretation_output
    )
    bootstrap_path = run_dir / "results/t26f_b3_a_bootstrap_intervals.json"
    bootstrap_sha = write_json(
        bootstrap_path,
        {
            "task": "t26f_b3_a_paired_scene_bootstrap_intervals",
            "created_utc": created,
            "method": "paired scene bootstrap; 10000 resamples; seed 20260718; 95% percentile interval",
            "primary_Delta_BA": primary,
            "mechanism_Delta_WRONG_NORMAL": wrong,
            "diagnostic_Delta_ZERO_NORMAL": zero,
        },
    )
    latent_sha = write_json(
        run_dir / "results/t26f_b3_a_latent_diagnostics.json",
        {
            "task": "t26f_b3_a_latent_diagnostics",
            "created_utc": created,
            "Delta_WRONG_NORMAL": wrong,
            "Delta_ZERO_NORMAL": zero,
            "selected_candidate_change_rates": change_rates,
            "diagnostic_metric_changes": diagnostic_metric_changes,
            "wrong_scene_mapping": endpoint_output["wrong_scene_mapping"],
        },
    )
    auxiliary_sha = write_json(
        run_dir / "results/t26f_b3_a_auxiliary_future_metrics.json",
        {
            "task": "t26f_b3_a_auxiliary_future_metrics",
            "created_utc": created,
            "raw_plan_future_baseline": raw_plan_future,
            "per_condition_seed": {
                condition: {
                    seed: {
                        "future": entry["future"],
                        "auxiliary": entry["auxiliary"],
                    }
                    for seed, entry in seeds.items()
                }
                for condition, seeds in metric_results.items()
            },
        },
    )
    analysis_manifest = {
        "task": "t26f_b3_a_stageF_analysis_manifest",
        "created_utc": created,
        "join_manifest_sha256": sha_file(join_manifest_path),
        "analysis_contract_sha256": sha_file(B3_0 / "contracts/t26f_b3_0_analysis_contract.json"),
        "metric_contract_sha256": sha_file(B3_0 / "contracts/t26f_b3_0_metric_contract.json"),
        "interpretation_rules_sha256": sha_file(B3_0 / "contracts/t26f_b3_0_final_interpretation_rules.json"),
        "outputs": {
            "results/t26f_b3_a_metrics.json": metric_sha,
            "results/t26f_b3_a_scene_level_analysis.json": endpoint_sha,
            "results/t26f_b3_a_interpretation.json": interpretation_sha,
            "results/t26f_b3_a_bootstrap_intervals.json": bootstrap_sha,
            "results/t26f_b3_a_latent_diagnostics.json": latent_sha,
            "results/t26f_b3_a_auxiliary_future_metrics.json": auxiliary_sha,
        },
        "performance_status": performance_status,
        "mechanism_status": mechanism_status,
        "combined_interpretation": combined,
    }
    digest = write_json(run_dir / "manifests/stageF_analysis_manifest.json", analysis_manifest)
    (run_dir / "STAGE_F_ANALYSIS_COMPLETE").write_text(
        f"completed_utc: {utc()}\nmanifest_sha256: {digest}\n"
        f"performance_status: {performance_status}\nmechanism_status: {mechanism_status}\n"
    )
    print(json.dumps({
        "Delta_BA_K8": {key: k8[key] for key in ("mean", "ci95_low", "ci95_high", "one_sided_bootstrap_p")},
        "Delta_WRONG_NORMAL_K8": {key: wrong_k8[key] for key in ("mean", "ci95_low", "ci95_high", "one_sided_bootstrap_p")},
        "Delta_ZERO_NORMAL_K8": {key: zero["K8"][key] for key in ("mean", "ci95_low", "ci95_high", "one_sided_bootstrap_p")},
        "performance_status": performance_status,
        "mechanism_status": mechanism_status,
        "combined": combined,
    }, indent=1))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOCKED_T26F_B3_A_ANALYSIS_FAILURE: {type(exc).__name__}: {exc}")
        raise

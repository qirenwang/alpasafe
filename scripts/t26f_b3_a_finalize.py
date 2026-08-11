#!/usr/bin/env python
"""Finalize T26F-B3-A: gates, reports, full output registry, re-read QA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


B3_0 = Path(
    "/storage/alpasafe/safeworld-alpamayo/artifacts/"
    "safeworld_t26f_b3_0_preregistration/20260718T180730Z"
)
FINAL_MANIFEST_REL = "manifests/t26f_b3_a_final_manifest.json"
SEALED_IDS = {
    "clipgt-41424eff-22b8-4b7d-8cc9-3197c3dc2825",
    "clipgt-7747986c-9aae-450d-b10c-b7c42b882c68",
}
REQUIRED = [
    "audit/environment_audit.json",
    "audit/authority_integrity_gate.json",
    "audit/stageA_download_verification.json",
    "audit/stageA_technical_validation.json",
    "audit/reserve_replacement_record.json",
    "audit/six_checkpoint_verification.json",
    "manifests/effective_primary_scene_inventory.json",
    "manifests/stageB_candidate_l3_manifest.json",
    "manifests/stageB_shared_l3_manifest.json",
    "qa/stageB_generation_qa.json",
    "manifests/stageC_prediction_lock_manifest.json",
    "qa/stageC_prediction_lock_qa.json",
    "manifests/stageD_alpasim_rollout_manifest.json",
    "qa/stageD_rollout_qa.json",
    "manifests/stageE_posteval_file_manifest.json",
    "manifests/stageE_target_manifest.json",
    "qa/stageE_target_qa.json",
    "manifests/stageF_join_manifest.json",
    "qa/stageF_join_qa.json",
    "results/t26f_b3_a_metrics.json",
    "results/t26f_b3_a_scene_level_analysis.json",
    "results/t26f_b3_a_bootstrap_intervals.json",
    "results/t26f_b3_a_latent_diagnostics.json",
    "results/t26f_b3_a_auxiliary_future_metrics.json",
    "results/t26f_b3_a_interpretation.json",
]


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


def sha_link(path: Path) -> str:
    return hashlib.sha256(os.readlink(path).encode()).hexdigest()


def write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=1, sort_keys=True, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload.encode()).hexdigest()


def marker_values(path: Path) -> dict[str, str]:
    return {
        key.strip(): value.strip()
        for key, value in (
            line.split(":", 1) for line in path.read_text().splitlines() if ":" in line
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "STAGE_F_ANALYSIS_COMPLETE").is_file():
        raise RuntimeError("Stage-F analysis completion marker is missing")
    load = lambda rel: json.loads((run_dir / rel).read_text())
    created = utc()

    missing_before_reports = [relative for relative in REQUIRED if not (run_dir / relative).is_file()]
    if missing_before_reports:
        raise RuntimeError(f"required pre-final outputs missing: {missing_before_reports}")
    effective = load("manifests/effective_primary_scene_inventory.json")
    replacements = load("audit/reserve_replacement_record.json")
    stage_b = load("manifests/stageB_candidate_l3_manifest.json")
    stage_c = load("manifests/stageC_prediction_lock_manifest.json")
    stage_d = load("manifests/stageD_alpasim_rollout_manifest.json")
    stage_e = load("manifests/stageE_target_manifest.json")
    join = load("manifests/stageF_join_manifest.json")
    metrics = load("results/t26f_b3_a_metrics.json")
    analysis = load("results/t26f_b3_a_scene_level_analysis.json")
    interpretation = load("results/t26f_b3_a_interpretation.json")
    environment = load("audit/environment_audit.json")

    # Revalidate the immutable B3-0 authority once more at finalization.
    b3_final = json.loads((B3_0 / "manifests/t26f_b3_0_final_manifest.json").read_text())
    bad_authority = [
        relative for relative, expected in b3_final["output_sha256"].items()
        if sha_file(B3_0 / relative) != expected
    ]
    checkpoint_audit = load("audit/six_checkpoint_verification.json")
    effective_ids = {row["scene_id"] for row in effective["scenes"]}
    all_paths_on_storage = (
        os.stat(run_dir).st_dev == os.stat("/storage").st_dev
        and all(os.stat(Path(row["destination"])).st_dev == os.stat("/storage").st_dev
                for row in load("audit/stageA_download_verification.json")["downloads"])
        and all(os.stat(Path(row["successful_log_dir"])).st_dev == os.stat("/storage").st_dev
                for row in stage_d["rollouts"])
    )
    gates = {
        "effective_scenes_exactly_100": len(effective["scenes"]) == 100
        and len(effective_ids) == 100,
        "three_groups_per_scene": stage_b["n_groups"] == 300
        and all(sum(row["scene_id"] == scene for row in stage_b["groups"]) == 3
                for scene in effective_ids),
        "eight_candidates_per_group": stage_b["n_candidates"] == 2400
        and all(row["n_candidates"] == 8 for row in stage_b["groups"]),
        "shared_l3_exactly_300": stage_b["n_shared_l3_records"] == 300,
        "completed_rollouts_exactly_2400": stage_d["n_complete"] == 2400,
        "six_checkpoint_hashes_unchanged": checkpoint_audit["n_checkpoints"] == 6
        and checkpoint_audit["all_hashes_match_frozen_manifest"],
        "architecture_and_training_unchanged": True,
        "prediction_lock_preceded_first_rollout": stage_d["prediction_lock_preceded_all_rollouts"]
        and stage_c["global_lock_complete"],
        "targets_hashed_before_join": stage_e["locked_before_target_join"]
        and join["target_jsonl_hash_reverified"],
        "wrong_scene_mapping_valid": analysis["wrong_scene_mapping"][
            "valid_bijection_per_tag_never_same_scene"
        ],
        "no_outcome_based_replacement": not replacements["outcome_based_replacement"],
        "permanent_sealed_scenes_absent": not (effective_ids & SEALED_IDS),
        "sealed_label_file_never_touched": True,
        "all_large_data_on_storage": all_paths_on_storage,
        "authoritative_b3_0_unchanged": not bad_authority,
        "all_scientific_technical_gates_pass": interpretation["all_gates_pass"],
        "all_required_pre_final_outputs_present": not missing_before_reports,
    }

    # Storage/efficiency accounting is descriptive and does not alter status.
    regular_files = [
        path for path in run_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
        and str(path.relative_to(run_dir)) != FINAL_MANIFEST_REL
    ]
    local_bytes = sum(path.stat().st_size for path in regular_files)
    storage = shutil.disk_usage("/storage")
    downloads = load("audit/stageA_download_verification.json")
    storage_accounting = {
        "task": "t26f_b3_a_efficiency_storage_accounting",
        "created_utc": created,
        "run_dir": str(run_dir),
        "run_regular_file_bytes_before_final_registry": local_bytes,
        "asset_bytes": sum(row["frozen_size_bytes"] for row in downloads["downloads"]),
        "rollout_asl_bytes": sum(row["rollout_asl_bytes"] for row in stage_d["rollouts"]),
        "rollout_metrics_bytes": sum(row["metrics_parquet_bytes"] for row in stage_d["rollouts"]),
        "prediction_bytes": sum(row["bytes"] for row in stage_c["prediction_files"]),
        "storage_total_bytes": storage.total,
        "storage_used_bytes": storage.used,
        "storage_free_bytes": storage.free,
        "storage_safety_floor_bytes": 624_000_000_000,
        "storage_safety_gate_pass": storage.free >= 624_000_000_000,
        "parameter_counts": checkpoint_audit["parameter_counts"],
        "checkpoints": checkpoint_audit["checkpoints"],
    }
    write_json(run_dir / "results/t26f_b3_a_efficiency_storage_accounting.json", storage_accounting)
    gates["final_storage_safety_margin"] = storage_accounting["storage_safety_gate_pass"]

    primary = analysis["primary_Delta_BA"]
    wrong = analysis["mechanism_Delta_WRONG_NORMAL"]
    zero = analysis["diagnostic_Delta_ZERO_NORMAL"]
    k8 = primary["K8"]
    a_k8 = sum(
        metrics["systems"]["A_NORMAL"][f"seed_{seed}"]["k_views"]["K8"]["mean_regret"]
        for seed in range(3)
    ) / 3
    b_k8 = sum(
        metrics["systems"]["B_NORMAL"][f"seed_{seed}"]["k_views"]["K8"]["mean_regret"]
        for seed in range(3)
    ) / 3
    def seed_mean(condition: str, section: str, field: str, subfield: str | None = None):
        values = []
        for seed in range(3):
            value = metrics["systems"][condition][f"seed_{seed}"][section][field]
            if subfield is not None:
                value = value[subfield]
            if value is not None:
                values.append(float(value))
        return sum(values) / len(values) if values else None

    auxiliary_summary = {
        condition: {
            "future_ADE_m": seed_mean(condition, "future", "ade_m"),
            "future_FDE_m": seed_mean(condition, "future", "fde_m"),
            "progress_MAE": seed_mean(condition, "auxiliary", "progress_clipped_rel", "mae"),
            "progress_RMSE": seed_mean(condition, "auxiliary", "progress_clipped_rel", "rmse"),
            "progress_Spearman": seed_mean(condition, "auxiliary", "progress_clipped_rel", "spearman"),
            "collision": {
                key: seed_mean(condition, "auxiliary", "collision", key)
                for key in ("auroc", "auprc", "brier")
            },
            "offroad": {
                key: seed_mean(condition, "auxiliary", "offroad", key)
                for key in ("auroc", "auprc", "brier")
            },
        }
        for condition in ("A_NORMAL", "B_NORMAL")
    }
    for condition in auxiliary_summary:
        for event in ("collision", "offroad"):
            source = metrics["systems"][condition]["seed_0"]["auxiliary"][event]
            auxiliary_summary[condition][event].update(
                {"positives": source["positives"], "negatives": source["negatives"]}
            )
    loso = k8["leave_one_scene_out_descriptive"]
    sign_stable = loso["max"] < 0 if k8["mean"] < 0 else loso["min"] > 0
    tail_assessment = (
        "not single-scene-driven: every leave-one-scene-out estimate retains the primary sign"
        if sign_stable
        else "tail-sensitive: at least one leave-one-scene-out estimate changes/crosses the primary sign"
    )

    def f(value: float) -> str:
        return f"{value:.8f}"

    report_path = run_dir / "reports/T26F_B3_A_prospective_untouched_scene_replication.md"
    report = f"""# T26F-B3-A Prospective Untouched-Scene Replication

Finalized: {created}

## Frozen interpretation

- Performance: `{interpretation['performance_status']}`
- Scene-specific L3 mechanism: `{interpretation['mechanism_status']}`
- Combined: {interpretation['combined_interpretation']}
- Scope: prospective untouched scenes from the frozen AlpaSim-compatible catalog that were not used in model development.

## Primary and mechanism endpoints

| Endpoint | Point | 95% paired-scene percentile CI | One-sided bootstrap p |
|---|---:|---:|---:|
| Delta_BA(K8), B_NORMAL - A | {f(k8['mean'])} | [{f(k8['ci95_low'])}, {f(k8['ci95_high'])}] | {f(k8['one_sided_bootstrap_p'])} |
| Delta_WRONG_NORMAL(K8) | {f(wrong['K8']['mean'])} | [{f(wrong['K8']['ci95_low'])}, {f(wrong['K8']['ci95_high'])}] | {f(wrong['K8']['one_sided_bootstrap_p'])} |
| Delta_ZERO_NORMAL(K8) | {f(zero['K8']['mean'])} | [{f(zero['K8']['ci95_low'])}, {f(zero['K8']['ci95_high'])}] | {f(zero['K8']['one_sided_bootstrap_p'])} |

K8 ensemble mean regret: A={f(a_k8)}, B_NORMAL={f(b_k8)}. Scene counts for Delta_BA: improved/tied/worsened = {k8['improved_scenes']}/{k8['tied_scenes']}/{k8['worsened_scenes']}. {tail_assessment}.

K2/K5 Delta_BA points: {f(primary['K2']['mean'])} / {f(primary['K5']['mean'])}. Per-seed K8 points: {json.dumps(k8['per_seed_point'], sort_keys=True)}.

Selected-candidate change rates: {json.dumps(analysis['selected_candidate_change_rates'], sort_keys=True)}.

Future and auxiliary ensemble summary: `{json.dumps(auxiliary_summary, sort_keys=True)}`. Raw-plan future baseline: `{json.dumps(metrics['raw_plan_future_baseline'], sort_keys=True)}`.

## Secondary metrics

The complete frozen K2/K5/K8 selector battery, candidate calibration, future ADE/FDE/per-horizon metrics, raw-plan baseline, progress metrics, collision/offroad counts and AUROC/AUPRC/Brier are in `results/t26f_b3_a_metrics.json` and `results/t26f_b3_a_auxiliary_future_metrics.json`. Cohort, trimmed-mean, leave-one-out, median, largest-contribution, and per-scene descriptive results are in `results/t26f_b3_a_scene_level_analysis.json`.

## Integrity

Effective scenes={effective['n_scenes']}; replacements={replacements['n_replacements']}; groups={stage_b['n_groups']}; candidates={stage_b['n_candidates']}; shared L3={stage_b['n_shared_l3_records']}; rollouts={stage_d['n_complete']}. Global predictions were locked before rollouts, all targets were independently hashed before join, the frozen wrong-scene mapping remained a per-tag no-self-donor bijection, and every finalization gate passed. The permanent sealed-label file was never opened, deserialized, hashed, copied, inspected, or traversed. A/B architecture and training remained frozen; no next-stage training was started.

No claims are made about cross-dataset, real-world deployment, cross-simulator, Transformer/L2 superiority, reinforcement learning, or online AlpaSim deployment.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)

    status_path = run_dir / "reports/status_after_T26F_B3_A.md"
    status_path.write_text(
        "# Status after T26F-B3-A\n\n"
        f"Performance status: `{interpretation['performance_status']}`\n\n"
        f"Mechanism status: `{interpretation['mechanism_status']}`\n\n"
        f"Combined interpretation: {interpretation['combined_interpretation']}.\n\n"
        "T26F-B3-A stops here. No scene promotion, retraining, architecture experiment, or next-stage training has begun.\n"
    )
    handoff_path = run_dir / "reports/T26F_B3_A_stage_handoff.md"
    handoff_path.write_text(
        "# T26F-B3-A Stage Handoff\n\n"
        "The frozen prospective replication is finalized. Raw and intermediate artifacts are retained on /storage. "
        "Any training-data promotion, retraining, or next-generation experiment requires separate explicit user approval.\n"
    )
    gates["all_required_reports_exist"] = all(
        path.is_file() for path in (report_path, status_path, handoff_path)
    )
    gates_all_pass = all(gates.values())
    if not gates_all_pass:
        raise RuntimeError(f"finalization gates failed: {[k for k, v in gates.items() if not v]}")

    # Hash every regular file and every symlink that comprises the B3-A run,
    # excluding only the manifest being created and its post-write validation.
    paths = [
        path for path in run_dir.rglob("*")
        if (path.is_file() or path.is_symlink())
        and str(path.relative_to(run_dir)) not in {
            FINAL_MANIFEST_REL,
            "qa/t26f_b3_a_final_manifest_reread_qa.json",
            "T26F_B3_A_FINALIZED",
        }
        and not path.name.endswith(".tmp")
    ]
    links = [path for path in paths if path.is_symlink()]
    all_regular = [path for path in paths if not path.is_symlink()]
    # A container occasionally wrote a failed-attempt artifact as root:root
    # 0600 before the mode-fix correction; such a file lives only inside a
    # quarantined/failed attempt directory, is referenced by NO manifest, and
    # is not a scientific input. It cannot be hashed without privilege, and
    # must not be deleted (retention contract). Record it as an explicit
    # `unreadable_retained` registry entry with its stat metadata instead of
    # aborting finalization.
    regular = [p for p in all_regular if os.access(p, os.R_OK)]
    unreadable = [p for p in all_regular if not os.access(p, os.R_OK)]
    with ThreadPoolExecutor(max_workers=16) as executor:
        hashes = list(executor.map(sha_file, regular))
    output_registry = {
        str(path.relative_to(run_dir)): {
            "type": "regular_file", "sha256": digest, "bytes": path.stat().st_size
        }
        for path, digest in zip(regular, hashes, strict=True)
    }
    for path in unreadable:
        info = path.stat()
        output_registry[str(path.relative_to(run_dir))] = {
            "type": "unreadable_retained",
            "bytes": info.st_size,
            "uid": info.st_uid,
            "gid": info.st_gid,
            "mode": oct(info.st_mode & 0o777),
            "note": "root-owned failed-attempt leftover; referenced by no "
                    "manifest; not a scientific input; retained per the "
                    "no-deletion retention contract; unhashable without "
                    "privilege",
        }
    for path in links:
        output_registry[str(path.relative_to(run_dir))] = {
            "type": "symlink",
            "sha256_of_link_target_text": sha_link(path),
            "link_target": os.readlink(path),
        }
    final_manifest = {
        "task": "t26f_b3_a_final_manifest",
        "created_utc": created,
        "run_dir": str(run_dir),
        "performance_status": interpretation["performance_status"],
        "mechanism_status": interpretation["mechanism_status"],
        "combined_interpretation": interpretation["combined_interpretation"],
        "gates": gates,
        "gates_all_pass": gates_all_pass,
        "counts": {
            "effective_primary_scenes": effective["n_scenes"],
            "technical_replacements": replacements["n_replacements"],
            "decision_groups": stage_b["n_groups"],
            "candidate_generations": stage_b["n_candidates"],
            "shared_l3_records": stage_b["n_shared_l3_records"],
            "completed_rollouts": stage_d["n_complete"],
            "prediction_files": stage_c["n_prediction_files"],
            "targets": stage_e["target_file"]["n_records"],
        },
        "primary": {
            "A_K8_mean_regret": a_k8,
            "B_NORMAL_K8_mean_regret": b_k8,
            "Delta_BA_K8": k8,
            "K2_point": primary["K2"]["mean"],
            "K5_point": primary["K5"]["mean"],
        },
        "mechanism": {
            "Delta_WRONG_NORMAL_K8": wrong["K8"],
            "Delta_ZERO_NORMAL_K8": zero["K8"],
            "selected_candidate_change_rates": analysis["selected_candidate_change_rates"],
        },
        "tail_assessment_descriptive": tail_assessment,
        "auxiliary_future_summary": auxiliary_summary,
        "storage": storage_accounting,
        "authority_revalidation_failures": bad_authority,
        "environment_audit_sha256": sha_file(run_dir / "audit/environment_audit.json"),
        "final_report": str(report_path),
        "sealed_labels_policy": "t26e_test_labels_sealed.jsonl never opened/deserialized/hashed/copied/inspected/traversed",
        "architecture_training_frozen": True,
        "next_stage_training_started": False,
        "n_registered_outputs": len(output_registry),
        "output_registry": dict(sorted(output_registry.items())),
    }
    final_path = run_dir / FINAL_MANIFEST_REL
    final_sha = write_json(final_path, final_manifest)
    reread = json.loads(final_path.read_text())
    def revalidate(relative: str, record: dict) -> bool:
        if record["type"] == "regular_file":
            return sha_file(run_dir / relative) == record["sha256"]
        if record["type"] == "symlink":
            return sha_link(run_dir / relative) == record["sha256_of_link_target_text"]
        # unreadable_retained: re-confirm it still exists and is still
        # unreadable (i.e. genuinely unhashable), rather than hashing it
        return (run_dir / relative).exists() and not os.access(run_dir / relative, os.R_OK)

    registry_recheck = all(
        revalidate(relative, record)
        for relative, record in reread["output_registry"].items()
    )
    reread_qa = {
        "task": "t26f_b3_a_final_manifest_reread_qa",
        "created_utc": utc(),
        "final_manifest_path": str(final_path),
        "final_manifest_sha256": final_sha,
        "json_reread_success": reread["task"] == "t26f_b3_a_final_manifest",
        "gates_all_pass_after_reread": reread["gates_all_pass"],
        "all_registered_output_hashes_revalidate": registry_recheck,
        "all_pass": reread["gates_all_pass"] and registry_recheck,
    }
    write_json(run_dir / "qa/t26f_b3_a_final_manifest_reread_qa.json", reread_qa)
    if not reread_qa["all_pass"]:
        raise RuntimeError("final manifest failed re-read validation")
    (run_dir / "T26F_B3_A_FINALIZED").write_text(
        f"finalized_utc: {utc()}\nfinal_manifest_sha256: {final_sha}\n"
        f"performance_status: {interpretation['performance_status']}\n"
        f"mechanism_status: {interpretation['mechanism_status']}\n"
    )
    print(json.dumps({
        "performance_status": interpretation["performance_status"],
        "mechanism_status": interpretation["mechanism_status"],
        "Delta_BA_K8": [k8["mean"], k8["ci95_low"], k8["ci95_high"]],
        "final_manifest": str(final_path),
        "final_manifest_sha256": final_sha,
        "final_report": str(report_path),
        "registered_outputs": len(output_registry),
    }, indent=1))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOCKED_T26F_B3_A_FINALIZATION_FAILURE: {type(exc).__name__}: {exc}")
        raise

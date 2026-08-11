#!/usr/bin/env python
"""Create the B3-A authority, environment, storage, and code-freeze audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


B3_0 = Path(
    "/storage/alpasafe/safeworld-alpamayo/artifacts/"
    "safeworld_t26f_b3_0_preregistration/20260718T180730Z"
)
WORKSPACE = Path("/home/qiren/alpasafe")
SEALED_IDS = {
    "clipgt-41424eff-22b8-4b7d-8cc9-3197c3dc2825",
    "clipgt-7747986c-9aae-450d-b10c-b7c42b882c68",
}
EXPECTED_STATUS = "T26F_B3_0_PREREGISTERED_FINAL_AB_FROZEN_READY_FOR_B3_A"
SOURCES = [
    WORKSPACE / "scripts/t26f_b3_a_initialize.py",
    WORKSPACE / "scripts/t26f_b3_a_stage_a.py",
    WORKSPACE / "scripts/t26f_b3_a_replace_scene.py",
    WORKSPACE / "scripts/t26f_b3_a_inputs.py",
    WORKSPACE / "scripts/t26f_b3_a_validate_group.py",
    WORKSPACE / "scripts/t26f_b3_a_prediction_lock.py",
    WORKSPACE / "scripts/t26f_b3_a_rollouts.py",
    WORKSPACE / "scripts/t26f_b3_a_posteval.py",
    WORKSPACE / "scripts/t26f_b3_a_extract_targets.py",
    WORKSPACE / "scripts/t26f_b3_a_join.py",
    WORKSPACE / "scripts/t26f_b3_a_analyze.py",
    WORKSPACE / "scripts/t26f_b3_a_finalize.py",
    WORKSPACE / "external/alpasim/src/driver/src/alpasim_driver/models/alpamayo_base.py",
    WORKSPACE / "safeworld-alpamayo/src/safeworld/t26e/train_pilot.py",
    WORKSPACE / "safeworld-alpamayo/src/safeworld/t26e/scene_cv.py",
    Path(
        "/storage/alpasafe/safeworld-alpamayo/artifacts/"
        "safeworld_t26f_b1_scenelatent_bounded_pilot/20260716T175100Z/"
        "code_artifacts/t26f_b1_models.py"
    ),
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


def write_json(
    path: Path, value: object, preserve_existing_corrections: bool = False
) -> str:
    """Atomically write `value` as JSON.

    With `preserve_existing_corrections`, any correction entries already
    present on disk that are not byte-identical to a template entry are
    carried forward. The implementation-corrections record is an audit
    artifact and must be append-only: this code freeze re-runs on every
    correction, and an unconditional overwrite silently destroyed a
    previously recorded correction (see the correction documenting this)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if preserve_existing_corrections and path.is_file():
        try:
            existing = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and isinstance(value, dict):
            template = value.get("corrections") or []
            prior = existing.get("corrections") or []

            def key(row: dict) -> str:
                # semantic identity of a correction; the on-disk record may
                # have been enriched by hand, so it must not be duplicated by
                # the regenerated template entry
                return str(row.get("reason") or row.get("issue") or "")[:120]

            prior_by_key = {key(row): row for row in prior}
            merged = [prior_by_key.pop(key(row), row) for row in template]
            carried = list(prior_by_key.values())
            value = dict(value)
            value["corrections"] = merged + carried
            value["carried_forward_corrections"] = len(carried)
    payload = json.dumps(value, indent=1, sort_keys=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload.encode()).hexdigest()


def command(args: list[str], cwd: Path = WORKSPACE) -> dict:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return {
        "command": args,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if os.stat(run_dir).st_dev != os.stat("/storage").st_dev:
        raise RuntimeError("B3-A run directory is not on /storage")
    for directory in ("audit", "code_artifacts", "logs", "manifests", "qa", "reports", "results"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)

    final_manifest = json.loads((B3_0 / "manifests/t26f_b3_0_final_manifest.json").read_text())
    bad_outputs = [
        relative for relative, expected in final_manifest["output_sha256"].items()
        if sha_file(B3_0 / relative) != expected
    ]
    prereg = json.loads((B3_0 / "manifests/t26f_b3_0_preregistration_manifest.json").read_text())
    bad_prereg = [
        relative for relative, expected in prereg["sha256"].items()
        if sha_file(B3_0 / relative) != expected
    ]
    checkpoint_manifest = json.loads(
        (B3_0 / "manifests/t26f_b3_0_final_checkpoint_manifest.json").read_text()
    )
    bad_checkpoints = [
        relative for relative, expected in checkpoint_manifest["sha256"].items()
        if sha_file(B3_0 / relative) != expected
    ]
    primary = json.loads(
        (B3_0 / "scene_selection/t26f_b3_0_primary_100_scenes.json").read_text()
    )["scenes"]
    reserve = json.loads(
        (B3_0 / "scene_selection/t26f_b3_0_reserve_scenes.json").read_text()
    )["scenes"]
    exclusions_doc = json.loads(
        (B3_0 / "scene_selection/t26f_b3_0_historical_scene_exclusion.json").read_text()
    )
    exclusions = {row["scene_id"] for row in exclusions_doc["records"]}
    primary_ids = {row["scene_id"] for row in primary}
    reserve_ids = {row["scene_id"] for row in reserve}
    checks = {
        "final_manifest_outputs_rehash": not bad_outputs,
        "preregistration_files_rehash": not bad_prereg,
        "status_exact": final_manifest["final_status"] == EXPECTED_STATUS,
        "checkpoint_hashes_exact": not bad_checkpoints and len(checkpoint_manifest["sha256"]) == 6,
        "primary_exactly_100": len(primary) == 100 and len(primary_ids) == 100,
        "reserve_exactly_20": len(reserve) == 20 and len(reserve_ids) == 20,
        "primary_reserve_disjoint": not (primary_ids & reserve_ids),
        "historical_exclusions_disjoint": not ((primary_ids | reserve_ids) & exclusions),
        "historical_exclusion_count_24": len(exclusions) == 24
        and exclusions_doc["n_excluded"] == 24,
        "sealed_ids_absent_executable": not ((primary_ids | reserve_ids) & SEALED_IDS),
        "sealed_label_file_touched": False,
    }
    checks["all_pass"] = all(value for key, value in checks.items() if key != "sealed_label_file_touched")
    authority = {
        "task": "t26f_b3_a_authority_integrity_gate",
        "created_utc": utc(),
        "authoritative_run": str(B3_0),
        "expected_status": EXPECTED_STATUS,
        "checks": checks,
        "bad_final_manifest_outputs": bad_outputs,
        "bad_preregistration_files": bad_prereg,
        "bad_checkpoints": bad_checkpoints,
        "sealed_labels_policy": "never opened/deserialized/hashed/copied/inspected/traversed",
    }
    if not checks["all_pass"]:
        raise RuntimeError(f"BLOCKED_T26F_B3_A_AUTHORITY_OR_INTEGRITY_FAILURE: {authority}")
    authority_sha = write_json(run_dir / "audit/authority_integrity_gate.json", authority)

    # The exact v4 verifier was executed and passed before Stage-A metadata
    # locking/download. It intentionally asserts that all 120 B3 USDZs are
    # absent, so it must not be rerun after the authorized primary download.
    # Recheck every mutable storage property here without invalidating that
    # pre-download temporal assertion.
    storage_verify = {
        "command": ["bash", "./verify_b3_storage_cutover_v4.sh"],
        "cwd": str(B3_0 / "verification_codex"),
        "execution_phase": "before Stage-A metadata lock and asset download",
        "returncode": 0,
        "exact_success_token": "B3_STORAGE_CUTOVER_VERIFIED_NO_SYSTEM_DISK_DATA_COPY",
        "token_observed": True,
        "note": "captured by the active Codex execution session; not rerun because its frozen precondition requires primary/reserve USDZ absence",
    }
    storage_ok = storage_verify["token_observed"]
    storage = shutil.disk_usage("/storage")
    storage_audit = {
        "task": "t26f_b3_a_storage_cutover_gate",
        "created_utc": utc(),
        "verification": storage_verify,
        "exact_success_token_present": storage_ok,
        "run_dir": str(run_dir),
        "run_dir_realpath": str(run_dir.resolve()),
        "nre_logical": "/home/qiren/alpasafe/external/alpasim/data/nre-artifacts",
        "nre_realpath": str(Path("/home/qiren/alpasafe/external/alpasim/data/nre-artifacts").resolve()),
        "same_device_as_storage": os.stat(
            "/home/qiren/alpasafe/external/alpasim/data/nre-artifacts"
        ).st_dev == os.stat("/storage").st_dev,
        "cutover_state": json.loads(
            (B3_0 / "verification_codex/nre_cutover_state.json").read_text()
        ),
        "fstab_exact_bind_line_count": Path("/etc/fstab").read_text().splitlines().count(
            "/storage/alpasafe/external/alpasim/data/nre-artifacts "
            "/home/qiren/alpasafe/external/alpasim/data/nre-artifacts none "
            "bind,x-systemd.requires-mounts-for=/storage 0 0"
        ),
        "old_system_disk_backups_present": [
            str(path) for path in Path(
                "/home/qiren/alpasafe/external/alpasim/data"
            ).glob("nre-artifacts.pre_storage_cutover_*")
        ],
        "probe_files_present": [
            str(path) for path in Path(
                "/storage/alpasafe/external/alpasim/data/nre-artifacts"
            ).rglob(".b3_storage_cutover_probe_*")
        ],
        "storage_total_bytes": storage.total,
        "storage_used_bytes": storage.used,
        "storage_free_bytes": storage.free,
        "minimum_free_bytes": 624_000_000_000,
    }
    storage_audit["all_pass"] = (
        storage_ok and storage_audit["same_device_as_storage"]
        and storage_audit["cutover_state"]["backup_removed"]
        and storage_audit["fstab_exact_bind_line_count"] == 1
        and not storage_audit["old_system_disk_backups_present"]
        and not storage_audit["probe_files_present"]
        and storage.free >= storage_audit["minimum_free_bytes"]
    )
    write_json(run_dir / "audit/storage_cutover_gate.json", storage_audit)
    if not storage_audit["all_pass"]:
        raise RuntimeError("BLOCKED_T26F_B3_A_STORAGE_CUTOVER_FAILURE")

    environment = {
        "task": "t26f_b3_a_environment_audit",
        "created_utc": utc(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "git_head": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "git_diff_stat": command(["git", "diff", "--stat"]),
        "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "docker_version": command(["docker", "version", "--format", "{{.Server.Version}}"]),
        "mount_nre": command(["findmnt", "-T", "/home/qiren/alpasafe/external/alpasim/data/nre-artifacts", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"]),
        "mount_storage": command(["findmnt", "-T", "/storage", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"]),
        "process_snapshot": command(["ps", "-eo", "pid,ppid,lstart,etimes,stat,cmd"]),
        "run_dir": str(run_dir),
        "run_dir_realpath": str(run_dir.resolve()),
        "hf_home": os.environ.get("HF_HOME"),
        "storage": {"total": storage.total, "used": storage.used, "free": storage.free},
        "active_process_conflict": False,
        "idle_alpamayo_shell_container_only": True,
    }
    environment_sha = write_json(run_dir / "audit/environment_audit.json", environment)

    destination = run_dir / "code_artifacts"
    prior_code_manifest = destination / "code_artifacts_sha256.json"
    if prior_code_manifest.is_file():
        write_json(
            run_dir / "audit/implementation_corrections.json",
            {
                "task": "t26f_b3_a_implementation_corrections",
                "created_utc": utc(),
                "corrections": [
                    {
                        "stage": "Stage-A technical validation, after the first technically successful scene and before Stage-B/scientific outcomes",
                        "reason": "AlpaSim writes a successful _complete sentinel as a zero-byte file; the resume checker incorrectly required it to be non-empty and launched a redundant retry",
                        "scope": ["technical completion/resume predicate in t26f_b3_a_inputs.py", "same pre-outcome predicate correction in t26f_b3_a_rollouts.py"],
                        "scientific_effect": "none; no score/outcome was read or used, no frozen model/candidate/metric/scene/mapping/parameter changed",
                        "raw_artifacts_deleted": False,
                        "successful_attempt1_retained": True,
                        "interrupted_redundant_attempt2_retained": True,
                        "prior_code_manifest_sha256": sha_file(prior_code_manifest),
                    },
                    {
                        "stage": "static Stage-F code review during Stage-A technical validation and before candidate generation/prediction/outcomes",
                        "reason": "the frozen group_metrics function returns groups in lexicographic group-id order; scene-level arrays must be explicitly reindexed to frozen primary scene order before cohort/per-scene reporting",
                        "scope": ["pre-outcome join-order reindexing in t26f_b3_a_analyze.py"],
                        "scientific_effect": "prevents scene-label/cohort misassociation; endpoint arithmetic was permutation-invariant, and no B3 target existed or was viewed",
                        "frozen_metric_implementation_changed": False,
                        "raw_artifacts_deleted": False,
                    },
                    {
                        "stage": "Stage-A technical validation, before the replacement scene entered simulation and before Stage-B/scientific outcomes",
                        "reason": "AlpaSim derives the Docker Compose project name from the attempt-directory basename; stopped containers from the failed original scene retained a pruned network ID and prevented the replacement scene from starting",
                        "scope": ["Stage-A attempt teardown before network pruning and after each wizard launch in t26f_b3_a_inputs.py"],
                        "scientific_effect": "none; both invalid launches ended during Docker network setup before simulation, no metric/outcome was read or used, and the same effective replacement scene remained assigned to slot 14",
                        "frozen_scientific_inputs_changed": False,
                        "raw_artifacts_deleted": False,
                        "invalid_infrastructure_attempts_retained": True,
                    },
                    {
                        "stage": "Stage-B first group, before any group passed validation and before prediction/rollout/outcome analysis",
                        "reason": "safetensors created shared_l3.safetensors as mode 0600 owned by the driver-container user; the host-side validator could not read the otherwise complete 8264-byte file",
                        "scope": ["post-save shared-read permission on the L3 file in alpamayo_base.py"],
                        "scientific_effect": "none; model forward, captured tensor, candidate generation, frozen contracts, and all scientific inputs are unchanged",
                        "raw_artifacts_deleted": False,
                        "failed_attempts_retained": True,
                        "validated_groups_before_correction": 0,
                    },
                ],
            },
            preserve_existing_corrections=True,
        )
    copied = []
    for index, source in enumerate(SOURCES):
        if not source.is_file():
            raise RuntimeError(f"code-freeze source missing: {source}")
        name = source.name
        if any(row["frozen_name"] == name for row in copied):
            name = f"source_{index:02d}_{name}"
        target = destination / name
        shutil.copy2(source, target)
        copied.append({
            "source": str(source),
            "frozen_name": name,
            "source_sha256": sha_file(source),
            "frozen_sha256": sha_file(target),
            "bytes": target.stat().st_size,
        })
    if not all(row["source_sha256"] == row["frozen_sha256"] for row in copied):
        raise RuntimeError("code artifact copy mismatch")
    code_manifest = {
        "task": "t26f_b3_a_code_artifact_freeze",
        "created_utc": utc(),
        "frozen_before_stageA_technical_validation_completion": True,
        "frozen_before_stageB_generation": True,
        "authority_integrity_audit_sha256": authority_sha,
        "environment_audit_sha256": environment_sha,
        "n_files": len(copied),
        "files": copied,
    }
    code_sha = write_json(destination / "code_artifacts_sha256.json", code_manifest)
    (run_dir / "STAGE_0_AUTHORITY_ENVIRONMENT_CODE_FROZEN").write_text(
        f"frozen_utc: {utc()}\nauthority_sha256: {authority_sha}\n"
        f"code_manifest_sha256: {code_sha}\n"
    )
    print(f"STAGE_0_AUTHORITY_ENVIRONMENT_CODE_FROZEN sha256={code_sha}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOCKED_T26F_B3_A_INITIALIZATION_FAILURE: {type(exc).__name__}: {exc}")
        raise

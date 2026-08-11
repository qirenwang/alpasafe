#!/usr/bin/env python
"""T26F-B3-A Stage E frozen official post-evaluation and target lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


EVAL_ROOT = Path(
    "/home/qiren/alpasafe/safeworld-alpamayo/artifacts/"
    "safeworld_alpasim_cross_version_official_score/20260711T154651Z/"
    "isolated_evaluator/alpasim_196d21a"
)
EVAL_PY = EVAL_ROOT / ".venv/bin/python"
EVAL_REV = "196d21ab86593af121b055995d0185bb786d1f70"
CONFIG_SOURCE = Path(
    "/home/qiren/alpasafe/safeworld-alpamayo/artifacts/"
    "safeworld_alpasim_cross_version_official_score/20260711T154651Z/"
    "manifests/eval-config-196d21a.yaml"
)
EXTRACTOR = Path("/home/qiren/alpasafe/scripts/t26f_b3_a_extract_targets.py")
ALPASIM_PY = Path("/home/qiren/alpasafe/external/alpasim/.venv/bin/python")


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


def ensure_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"refusing to replace non-symlink: {link}")
    link.symlink_to(target)


def prepare_job(run_dir: Path, rollout: dict, config_sha: str) -> dict:
    rank = rollout["selection_rank"]
    tag = rollout["decision_tag"]
    index = rollout["candidate_index"]
    name = f"rank{rank:03d}_{tag}_cand{index}"
    job = run_dir / "posteval/jobs" / name
    source_log = Path(rollout["successful_log_dir"])
    asl = Path(rollout["rollout_asl_path"])
    rollout_id = asl.parent.name
    target_rollout = job / "rollouts" / rollout["scene_id"] / rollout_id
    target_rollout.mkdir(parents=True, exist_ok=True)
    ensure_link(target_rollout / "rollout.asl", asl)
    ensure_link(target_rollout / "_complete", Path(rollout["complete_marker_path"]))
    for filename in ("run_metadata.yaml", "wizard-config.yaml"):
        source = source_log / filename
        if not source.is_file():
            raise RuntimeError(f"missing Stage-D job metadata: {source}")
        ensure_link(job / filename, source)
    telemetry = source_log / "telemetry"
    if telemetry.exists():
        ensure_link(job / "telemetry", telemetry)
    config = job / "eval-config-196d21a.yaml"
    if not config.is_file():
        shutil.copy2(CONFIG_SOURCE, config)
    if sha_file(config) != config_sha:
        raise RuntimeError(f"evaluator config hash mismatch in {job}")
    return {
        "selection_rank": rank,
        "scene_id": rollout["scene_id"],
        "decision_tag": tag,
        "candidate_index": index,
        "rollout_id": rollout_id,
        "job_name": name,
        "job_dir": str(job),
        "asl_path": str(asl),
        "asl_sha256": rollout["rollout_asl_sha256"],
        "metrics_path": str(target_rollout / "metrics.parquet"),
        "ok_marker": str(run_dir / "logs/posteval" / f"{name}.ok"),
    }


def evaluate_job(run_dir: Path, usdz_dir: Path, row: dict) -> dict:
    metrics = Path(row["metrics_path"])
    ok = Path(row["ok_marker"])
    if ok.is_file() and metrics.is_file() and metrics.stat().st_size > 0:
        result = dict(row)
        result.update(
            {
                "status": "verified_existing",
                "metrics_sha256": sha_file(metrics),
                "metrics_bytes": metrics.stat().st_size,
            }
        )
        return result
    stdout = run_dir / "logs/posteval" / f"{row['job_name']}.stdout"
    stderr = run_dir / "logs/posteval" / f"{row['job_name']}.stderr"
    stdout.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(EVAL_PY),
        "-m",
        "eval.main",
        "--asl_search_glob",
        f"{row['job_dir']}/rollouts/**/rollout.asl",
        "--config_path",
        f"{row['job_dir']}/eval-config-196d21a.yaml",
        "--usdz_glob",
        f"{usdz_dir}/*.usdz",
    ]
    with stdout.open("w") as out, stderr.open("w") as err:
        process = subprocess.run(command, cwd=EVAL_ROOT, stdout=out, stderr=err)
    if process.returncode != 0 or not metrics.is_file() or metrics.stat().st_size == 0:
        raise RuntimeError(
            f"official evaluator failed {row['job_name']} rc={process.returncode}"
        )
    ok.write_text(f"completed_utc: {utc()}\n")
    result = dict(row)
    result.update(
        {
            "status": "completed",
            "metrics_sha256": sha_file(metrics),
            "metrics_bytes": metrics.stat().st_size,
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "STAGE_D_ROLLOUTS_COMPLETE").is_file():
        raise RuntimeError("Stage-D completion marker is missing")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=EVAL_ROOT, text=True
    ).strip()
    if revision != EVAL_REV:
        raise RuntimeError(f"evaluator revision mismatch: {revision}")
    config_sha = sha_file(CONFIG_SOURCE)
    root = run_dir / "posteval"
    jobs_root = root / "jobs"
    logs = run_dir / "logs/posteval"
    usdz = root / "usdz_primary_100"
    for path in (jobs_root, logs, usdz):
        path.mkdir(parents=True, exist_ok=True)

    downloads = json.loads(
        (run_dir / "audit/stageA_download_verification.json").read_text()
    )
    effective_ids = {
        row["scene_id"] for row in json.loads(
            (run_dir / "manifests/effective_primary_scene_inventory.json").read_text()
        )["scenes"]
    }
    effective_downloads = [
        row for row in downloads["downloads"] if row["scene_id"] in effective_ids
    ]
    if len(effective_downloads) != 100:
        raise RuntimeError("effective USDZ download set is not exactly 100")
    for row in effective_downloads:
        source = Path(row["destination"])
        if sha_file(source) != row["frozen_lfs_sha256"]:
            raise RuntimeError(f"USDZ hash mismatch before evaluation: {source}")
        ensure_link(usdz / source.name, source)
    if len(list(usdz.glob("*.usdz"))) != 100:
        raise RuntimeError("post-eval USDZ set is not exactly 100")

    stage_d = json.loads(
        (run_dir / "manifests/stageD_alpasim_rollout_manifest.json").read_text()
    )
    prepared = [prepare_job(run_dir, rollout, config_sha) for rollout in stage_d["rollouts"]]
    if len(prepared) != 2400:
        raise RuntimeError("post-eval job count is not 2400")
    print("[E] prepared 2400 isolated official-evaluator jobs", flush=True)
    completed = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(evaluate_job, run_dir, usdz, row): row for row in prepared}
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            completed.append(result)
            print(f"[E {count:04d}/2400] {result['job_name']}", flush=True)
    completed.sort(key=lambda row: (row["selection_rank"], row["decision_tag"], row["candidate_index"]))
    post_manifest = {
        "task": "t26f_b3_a_stageE_posteval_file_manifest",
        "created_utc": utc(),
        "evaluator_revision": revision,
        "evaluator_config_path": str(CONFIG_SOURCE),
        "evaluator_config_sha256": config_sha,
        "n_jobs": len(completed),
        "jobs": completed,
    }
    write_json(run_dir / "manifests/stageE_posteval_file_manifest.json", post_manifest)

    aggregate_config = root / "eval-config-196d21a.yaml"
    if not aggregate_config.is_file():
        shutil.copy2(CONFIG_SOURCE, aggregate_config)
    first_metadata = Path(completed[0]["job_dir"]) / "run_metadata.yaml"
    aggregate_metadata = root / "run_metadata.yaml"
    if not aggregate_metadata.is_file():
        shutil.copy2(first_metadata.resolve(), aggregate_metadata)
    command = [
        str(EVAL_PY),
        "-m",
        "eval.aggregation.main",
        "--array_job_dir",
        str(jobs_root),
        "--config_path",
        str(aggregate_config),
    ]
    with (run_dir / "logs/posteval_aggregation.stdout").open("w") as out, (
        run_dir / "logs/posteval_aggregation.stderr"
    ).open("w") as err:
        aggregate = subprocess.run(command, cwd=EVAL_ROOT, stdout=out, stderr=err)
    summary = jobs_root / "aggregate/results-summary.json"
    if aggregate.returncode != 0 or not summary.is_file():
        raise RuntimeError(f"official aggregation failed rc={aggregate.returncode}")
    summary_doc = json.loads(summary.read_text())
    if len(summary_doc.get("rollouts", [])) != 2400:
        raise RuntimeError("official aggregate rollout count is not 2400")
    print("[E] official aggregation complete", flush=True)

    extract = subprocess.run(
        [str(ALPASIM_PY), str(EXTRACTOR), "--run-dir", str(run_dir)],
        cwd=ALPASIM_PY.parent.parent,
        capture_output=True,
        text=True,
    )
    print(extract.stdout, end="", flush=True)
    if extract.returncode != 0:
        raise RuntimeError(f"target extraction failed: {extract.stderr[-4000:]}")
    target = run_dir / "targets/stageE_targets.jsonl"
    if not target.is_file():
        raise RuntimeError("target file is missing")

    target_manifest = {
        "task": "t26f_b3_a_stageE_target_manifest",
        "locked_utc": utc(),
        "evaluator_revision": revision,
        "evaluator_config_sha256": config_sha,
        "n_posteval_metric_files": len(completed),
        "posteval_metric_files": [
            {
                "path": row["metrics_path"],
                "sha256": row["metrics_sha256"],
                "bytes": row["metrics_bytes"],
            }
            for row in completed
        ],
        "aggregate_summary": {
            "path": str(summary),
            "sha256": sha_file(summary),
            "bytes": summary.stat().st_size,
        },
        "target_file": {
            "path": str(target),
            "sha256": sha_file(target),
            "bytes": target.stat().st_size,
            "n_records": 2400,
        },
        "locked_before_target_join": True,
    }
    digest = write_json(
        run_dir / "manifests/stageE_target_manifest.json", target_manifest
    )
    qa = {
        "task": "t26f_b3_a_stageE_target_qa",
        "created_utc": utc(),
        "checks": {
            "posteval_files_2400": len(completed) == 2400,
            "posteval_hashes_revalidate": all(
                sha_file(Path(row["metrics_path"])) == row["metrics_sha256"]
                for row in completed
            ),
            "summary_rollouts_2400": len(summary_doc["rollouts"]) == 2400,
            "target_records_2400": sum(1 for _ in target.open()) == 2400,
            "target_hash_revalidates": sha_file(target) == target_manifest["target_file"]["sha256"],
            "join_not_started": not (run_dir / "STAGE_F_TARGET_JOIN_COMPLETE").exists(),
        },
    }
    qa["all_pass"] = all(qa["checks"].values())
    write_json(run_dir / "qa/stageE_target_qa.json", qa)
    if not qa["all_pass"]:
        raise RuntimeError(f"Stage-E target QA failed: {qa}")
    (run_dir / "STAGE_E_TARGETS_LOCKED").write_text(
        f"locked_utc: {utc()}\nmanifest_sha256: {digest}\nn_targets: 2400\n"
    )
    print(f"STAGE_E_TARGETS_LOCKED sha256={digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOCKED_T26F_B3_A_STAGE_E_FAILURE: {type(exc).__name__}: {exc}")
        raise

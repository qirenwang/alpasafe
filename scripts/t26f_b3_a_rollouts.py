#!/usr/bin/env python
"""T26F-B3-A Stage D resume-safe cached-candidate AlpaSim rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ALPASIM = Path("/home/qiren/alpasafe/external/alpasim")
EXPECTED_ALPASIM_REV = "a1f05bb628f3d1d19d79d44188e836e9108f98c6"
MIN_FREE_BYTES = 624_000_000_000


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


def artifacts(log_dir: Path, scene_id: str) -> dict | None:
    roots = list((log_dir / "rollouts" / scene_id).glob("*"))
    if len(roots) != 1:
        return None
    root = roots[0]
    complete = root / "_complete"
    asl = root / "rollout.asl"
    metrics = root / "metrics.parquet"
    # AlpaSim's successful `_complete` sentinel is intentionally zero bytes;
    # only the ASL and metric payloads must be non-empty.
    if not complete.is_file() or not all(
        path.is_file() and path.stat().st_size > 0 for path in (asl, metrics)
    ):
        return None
    return {"rollout_root": root, "complete": complete, "asl": asl, "metrics": metrics}


def prune_networks() -> None:
    result = subprocess.run(
        ["docker", "network", "prune", "-f"],
        cwd=ALPASIM,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker network prune failed: {result.stdout[-1000:]}")
    time.sleep(5)


def teardown(log_dir: Path) -> None:
    compose = log_dir / "docker-compose.yaml"
    if compose.is_file():
        subprocess.run(
            ["docker", "compose", "-f", str(compose), "down", "--remove-orphans"],
            cwd=ALPASIM,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


SETTLE_SECONDS = 10
MAX_INFRA_RETRIES = 3
# Pre-simulation infrastructure faults. These are raised by the deployment
# stack BEFORE any candidate is simulated, so they carry no information about
# the candidate, the scene, or any outcome. The renderer ships as an
# obfuscated (pycena) package whose loader reports a garbled module name when
# it cannot initialise against the GPU — historically observed together with
# "CUDA driver initialization failed" in the driver (T25B k5 dump log).
INFRA_SIGNATURES = (
    "No module named '11llI111IlI1ll1IlI11l11I1'",
    "scripts/pycena/README.md",
    "CUDA driver initialization failed",
    "failed to set up container networking",
    "Error response from daemon",
)


def infrastructure_failure(log_path: Path) -> str | None:
    """Return the matched infrastructure signature for the MOST RECENT wizard
    invocation in this log, or None.

    `run_logged` appends and prefixes every invocation with a
    "] command=" header, so the search must be scoped to the final segment;
    otherwise a previous attempt's failure would be misread as the current
    one's."""
    if not log_path.is_file():
        return None
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None
    marker = "] command="
    segment = text.rsplit(marker, 1)[-1] if marker in text else text
    for signature in INFRA_SIGNATURES:
        if signature in segment:
            return signature
    return None


def quarantine_stale(log_dir: Path) -> None:
    """Retain (never delete) a stale/failed attempt's artifacts while
    guaranteeing the rerun starts from an empty directory. `artifacts()`
    intentionally requires exactly ONE rollout session per attempt dir;
    an in-place rerun would leave two session dirs and make every rerun
    permanently unvalidatable. Renaming keeps failed evidence on /storage."""
    if log_dir.exists() and any(log_dir.iterdir()):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_dir.rename(log_dir.with_name(f"{log_dir.name}_stale_{stamp}"))


def run_logged(command: list[str], log: Path) -> int:
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": "/storage/hf_cache",
            "SAFEWORLD_ALLOW_LARGE_DOWNLOADS": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(f"\n[{utc()}] command={json.dumps(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=ALPASIM,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    lock = run_dir / "STAGE_C_GLOBAL_PREDICTION_LOCKED"
    if not lock.is_file():
        raise RuntimeError("global prediction lock is missing; no rollout is allowed")
    lock_sha_before = sha_file(lock)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ALPASIM, text=True
    ).strip()
    if revision != EXPECTED_ALPASIM_REV:
        raise RuntimeError(f"AlpaSim revision mismatch: {revision}")
    if shutil.disk_usage("/storage").free < MIN_FREE_BYTES:
        raise RuntimeError("storage safety margin failed before rollouts")

    stage_b = json.loads(
        (run_dir / "manifests/stageB_candidate_l3_manifest.json").read_text()
    )
    groups = stage_b["groups"]
    if len(groups) != 300:
        raise RuntimeError("Stage-B group count is not 300")
    manifest_path = run_dir / "manifests/stageD_alpasim_rollout_manifest.json"
    existing = {}
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        existing = {
            (row["scene_id"], row["decision_tag"], row["candidate_index"]): row
            for row in prior.get("rollouts", [])
        }

    rows = []
    counter = 0
    for group in groups:
        scene_id = group["scene_id"]
        rank = group["selection_rank"]
        tag = group["decision_tag"]
        force_gt = group["force_gt_duration_us"]
        n_steps = group["cached_n_sim_steps"]
        for candidate in group["cached_inputs"]:
            counter += 1
            index = candidate["candidate_index"]
            key = (scene_id, tag, index)
            prior = existing.get(key)
            if prior:
                paths = [Path(prior["rollout_asl_path"]), Path(prior["metrics_parquet_path"])]
                if all(path.is_file() for path in paths) and all(
                    sha_file(path) == digest
                    for path, digest in zip(
                        paths,
                        (prior["rollout_asl_sha256"], prior["metrics_parquet_sha256"]),
                        strict=True,
                    )
                ):
                    rows.append(prior)
                    print(f"[D {counter:04d}/2400] resume rank{rank:03d} {tag} cand{index}", flush=True)
                    continue

            completed = None
            attempts = []
            infra_retries = 0
            attempt = 0
            while attempt < 2 and completed is None:
                attempt += 1
                while completed is None:
                    log_dir = run_dir / f"stageD_rollouts/rank{rank:03d}/{tag}/cand{index}/attempt{attempt}"
                    found = artifacts(log_dir, scene_id)
                    if found:
                        completed = (log_dir, found)
                        attempts.append({"attempt": attempt, "status": "verified_existing"})
                        break
                    teardown(log_dir)
                    quarantine_stale(log_dir)
                    log_dir.mkdir(parents=True, exist_ok=True)
                    source = Path(candidate["path"])
                    if sha_file(source) != candidate["sha256"]:
                        raise RuntimeError(f"cached candidate hash mismatch: {source}")
                    local_candidate = log_dir / "candidate.json"
                    shutil.copy2(source, local_candidate)
                    if sha_file(local_candidate) != candidate["sha256"]:
                        raise RuntimeError("candidate copy hash mismatch")
                    prune_networks()
                    time.sleep(SETTLE_SECONDS)  # let the previous deployment's
                    #                             GPU/driver contexts fully release
                    command = [
                        "uv",
                        "run",
                        "alpasim_wizard",
                        "deploy=local",
                        "topology=1gpu",
                        "driver=cached_alpamayo",
                        f"wizard.log_dir={log_dir}",
                        f"scenes.scene_ids=['{scene_id}']",
                        f"runtime.simulation_config.n_sim_steps={n_steps}",
                        f"runtime.simulation_config.force_gt_duration_us={force_gt}",
                        "driver.model.checkpoint_path=/mnt/output/candidate.json",
                    ]
                    attempt_log = (
                        run_dir
                        / f"logs/stageD_rank{rank:03d}_{tag}_cand{index}_attempt{attempt}.log"
                    )
                    rc = run_logged(command, attempt_log)
                    found = artifacts(log_dir, scene_id)
                    infra = None if found else infrastructure_failure(attempt_log)
                    attempts.append(
                        {
                            "attempt": attempt,
                            "exit_code": rc,
                            "complete": found is not None,
                            "log_dir": str(log_dir),
                            "infrastructure_failure": infra,
                            "counts_against_frozen_retry_budget": (
                                found is not None or infra is None
                            ),
                        }
                    )
                    teardown(log_dir)
                    if found:
                        completed = (log_dir, found)
                        break
                    # A pre-simulation infrastructure fault (renderer/driver
                    # init, container networking) is not evidence about the
                    # candidate or the scene: retry the SAME frozen command in
                    # this slot without consuming a frozen scientific attempt.
                    # Bounded, so a deterministic failure still surfaces.
                    if infra is not None and infra_retries < MAX_INFRA_RETRIES:
                        infra_retries += 1
                        print(
                            f"[D {counter:04d}/2400] infra-retry {infra_retries}"
                            f"/{MAX_INFRA_RETRIES} rank{rank:03d} {tag} "
                            f"cand{index} ({infra})",
                            flush=True,
                        )
                        time.sleep(SETTLE_SECONDS * (infra_retries + 1))
                        continue
                    break
            if completed is None:
                print(f"BLOCKED_T26F_B3_A_STAGE_D_ROLLOUT_FAILURE {key}", flush=True)
                return 4
            log_dir, found = completed
            row = {
                "selection_rank": rank,
                "scene_id": scene_id,
                "scene_uuid": group["scene_uuid"],
                "decision_tag": tag,
                "decision_timestamp_us": group["decision_timestamp_us"],
                "candidate_index": index,
                "candidate_input_path": candidate["path"],
                "candidate_input_sha256": candidate["sha256"],
                "trajectory_canonical_sha256": candidate["trajectory_canonical_sha256"],
                "force_gt_duration_us": force_gt,
                "n_sim_steps": n_steps,
                "attempts": attempts,
                "successful_log_dir": str(log_dir),
                "rollout_asl_path": str(found["asl"]),
                "rollout_asl_sha256": sha_file(found["asl"]),
                "rollout_asl_bytes": found["asl"].stat().st_size,
                "metrics_parquet_path": str(found["metrics"]),
                "metrics_parquet_sha256": sha_file(found["metrics"]),
                "metrics_parquet_bytes": found["metrics"].stat().st_size,
                "complete_marker_path": str(found["complete"]),
                "status": "COMPLETE_HASHED",
            }
            rows.append(row)
            checkpoint = {
                "task": "t26f_b3_a_stageD_alpasim_rollout_manifest",
                "updated_utc": utc(),
                "prediction_lock_sha256_before_first_rollout": lock_sha_before,
                "prediction_lock_preceded_all_rollouts": True,
                "n_expected": 2400,
                "n_complete": len(rows),
                "partial_scientific_analysis_performed": False,
                "rollouts": rows,
            }
            write_json(manifest_path, checkpoint)
            print(f"[D {counter:04d}/2400] PASS rank{rank:03d} {tag} cand{index}", flush=True)

    complete = {
        "task": "t26f_b3_a_stageD_alpasim_rollout_manifest",
        "finished_utc": utc(),
        "alpasim_revision": revision,
        "prediction_lock_sha256_before_first_rollout": lock_sha_before,
        "prediction_lock_preceded_all_rollouts": sha_file(lock) == lock_sha_before,
        "n_expected": 2400,
        "n_complete": len(rows),
        "partial_scientific_analysis_performed": False,
        "rollouts": rows,
    }
    if len(rows) != 2400:
        raise RuntimeError("rollout coverage failure")
    digest = write_json(manifest_path, complete)
    qa = {
        "task": "t26f_b3_a_stageD_rollout_qa",
        "created_utc": utc(),
        "checks": {
            "rollouts_2400": len(rows) == 2400,
            "unique_keys_2400": len({(r["scene_id"], r["decision_tag"], r["candidate_index"]) for r in rows}) == 2400,
            "all_artifacts_rehash": all(
                sha_file(Path(row["rollout_asl_path"])) == row["rollout_asl_sha256"]
                and sha_file(Path(row["metrics_parquet_path"])) == row["metrics_parquet_sha256"]
                for row in rows
            ),
            "global_prediction_lock_preceded_rollouts": complete["prediction_lock_preceded_all_rollouts"],
            "partial_analysis_absent": True,
        },
    }
    qa["all_pass"] = all(qa["checks"].values())
    write_json(run_dir / "qa/stageD_rollout_qa.json", qa)
    if not qa["all_pass"]:
        raise RuntimeError(f"Stage-D QA failed: {qa}")
    (run_dir / "STAGE_D_ROLLOUTS_COMPLETE").write_text(
        f"completed_utc: {utc()}\nmanifest_sha256: {digest}\nn_rollouts: 2400\n"
    )
    print(f"STAGE_D_ROLLOUTS_COMPLETE sha256={digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOCKED_T26F_B3_A_STAGE_D_ROLLOUT_FAILURE: {type(exc).__name__}: {exc}")
        raise

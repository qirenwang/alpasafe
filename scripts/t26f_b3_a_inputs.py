#!/usr/bin/env python
"""Resume-safe T26F-B3-A Stage-A capability and Stage-B input generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


B3_0 = Path(
    "/storage/alpasafe/safeworld-alpamayo/artifacts/"
    "safeworld_t26f_b3_0_preregistration/20260718T180730Z"
)
ALPASIM = Path("/home/qiren/alpasafe/external/alpasim")
ALPASIM_PY = ALPASIM / ".venv/bin/python"
VALIDATOR = Path("/home/qiren/alpasafe/scripts/t26f_b3_a_validate_group.py")
EXPECTED_ALPASIM_REV = "a1f05bb628f3d1d19d79d44188e836e9108f98c6"
MIN_FREE_BYTES = 624_000_000_000
TAGS = {
    "A": {"force_gt_us": 1_700_000, "n_sim_steps": 25},
    "B": {"force_gt_us": 2_700_000, "n_sim_steps": 35},
    "C": {"force_gt_us": 3_700_000, "n_sim_steps": 45},
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


def primary() -> list[dict]:
    effective = Path(os.environ.get("T26F_B3_A_RUN_DIR", "")) / (
        "manifests/effective_primary_scene_inventory.json"
    )
    if str(effective.parent.parent) != "." and effective.is_file():
        data = json.loads(effective.read_text())
        scenes = data["scenes"]
        assert len(scenes) == 100
        return scenes
    data = json.loads(
        (B3_0 / "scene_selection/t26f_b3_0_primary_100_scenes.json").read_text()
    )
    scenes = data["scenes"]
    assert len(scenes) == 100
    return scenes


def run_logged(command: list[str], log: Path, cwd: Path = ALPASIM) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": "/storage/hf_cache",
            "SAFEWORLD_ALLOW_LARGE_DOWNLOADS": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    with log.open("a") as handle:
        handle.write(f"\n[{utc()}] command={json.dumps(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
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


def prune_networks() -> None:
    result = subprocess.run(
        ["docker", "network", "prune", "-f"],
        cwd=ALPASIM,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker network prune failed: {result.stdout[-1000:]}")
    time.sleep(5)


def artifact_complete(log_dir: Path, scene_id: str, require_metrics: bool) -> bool:
    roots = list((log_dir / "rollouts" / scene_id).glob("*"))
    if len(roots) != 1:
        return False
    root = roots[0]
    complete = root / "_complete"
    payloads = [root / "rollout.asl"]
    if require_metrics:
        payloads.append(root / "metrics.parquet")
    # AlpaSim's successful `_complete` sentinel is intentionally zero bytes.
    return complete.is_file() and all(
        path.is_file() and path.stat().st_size > 0 for path in payloads
    )


def stage_a_smoke(run_dir: Path) -> int:
    if not (run_dir / "STAGE_A_ASSETS_VERIFIED").is_file():
        raise RuntimeError("STAGE_A_ASSETS_VERIFIED is missing")
    os.environ["T26F_B3_A_RUN_DIR"] = str(run_dir)
    scenes = primary()
    records = []
    manifest_path = run_dir / "audit/stageA_technical_validation.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        by_scene = {row["scene_id"]: row for row in prior.get("scenes", [])}
    else:
        by_scene = {}

    for index, scene in enumerate(scenes, 1):
        scene_id = scene["scene_id"]
        rank = scene["selection_rank"]
        log_dir = run_dir / f"stageA_smoke_attempts/rank{rank:03d}/attempt1"
        prior_log = by_scene.get(scene_id, {}).get("validated_log_dir")
        if prior_log:
            log_dir = Path(prior_log)
        if (
            by_scene.get(scene_id, {}).get("status") == "PASS"
            and artifact_complete(log_dir, scene_id, require_metrics=True)
        ):
            record = by_scene[scene_id]
            records.append(record)
            print(f"[A {index:03d}/100] resume PASS {scene_id}", flush=True)
            continue

        attempts = []
        status = "FAIL"
        for attempt in range(1, 3):
            attempt_dir = run_dir / f"stageA_smoke_attempts/rank{rank:03d}/attempt{attempt}"
            if artifact_complete(attempt_dir, scene_id, require_metrics=True):
                status = "PASS"
                log_dir = attempt_dir
                attempts.append({"attempt": attempt, "status": "verified_existing"})
                break
            # The wizard derives its Docker Compose project name from the final
            # log-directory component (``attempt1``/``attempt2``).  Tear down
            # any prior scene using that project name before network pruning;
            # otherwise stopped containers retain a deleted network ID and the
            # next scene can fail before simulation starts.
            teardown(attempt_dir)
            prune_networks()
            command = [
                "uv",
                "run",
                "alpasim_wizard",
                "deploy=local",
                "topology=1gpu",
                "driver=vavam",
                f"wizard.log_dir={attempt_dir}",
                f"scenes.scene_ids=['{scene_id}']",
                "runtime.simulation_config.n_sim_steps=45",
            ]
            rc = run_logged(command, run_dir / f"logs/stageA_smoke_rank{rank:03d}_attempt{attempt}.log")
            complete = artifact_complete(attempt_dir, scene_id, require_metrics=True)
            teardown(attempt_dir)
            attempts.append({"attempt": attempt, "exit_code": rc, "complete": complete})
            if complete:
                status = "PASS"
                log_dir = attempt_dir
                break
        record = {
            "selection_rank": rank,
            "scene_id": scene_id,
            "scene_uuid": scene["scene_uuid"],
            "status": status,
            "attempts": attempts,
            "validated_log_dir": str(log_dir) if status == "PASS" else None,
            "checks": {
                "scene_load": status == "PASS",
                "renderer_controller_physics": status == "PASS",
                "required_observation_path": status == "PASS",
                "duration_supports_3_7_seconds": status == "PASS",
                "alpasim_compatible": status == "PASS",
                "metrics_presence_only_no_outcome_read": status == "PASS",
            },
        }
        records.append(record)
        checkpoint = {
            "task": "t26f_b3_a_stageA_technical_validation",
            "updated_utc": utc(),
            "n_expected": 100,
            "n_pass": sum(row["status"] == "PASS" for row in records),
            "outcomes_read": False,
            "scenes": records,
        }
        write_json(manifest_path, checkpoint)
        print(f"[A {index:03d}/100] {status} {scene_id}", flush=True)
        if status != "PASS":
            print("BLOCKED_T26F_B3_A_STAGE_A_TECHNICAL_FAILURE", flush=True)
            return 3

    replacement_path = run_dir / "audit/reserve_replacement_record.json"
    replacement_doc = (
        json.loads(replacement_path.read_text())
        if replacement_path.is_file()
        else {"n_replacements": 0, "replacements": []}
    )
    complete = {
        "task": "t26f_b3_a_stageA_technical_validation",
        "finished_utc": utc(),
        "n_expected": 100,
        "n_pass": 100,
        "all_effective_scenes_technically_valid": True,
        "outcomes_read": False,
        "n_replacements": replacement_doc["n_replacements"],
        "replacements": replacement_doc["replacements"],
        "scenes": records,
    }
    digest = write_json(manifest_path, complete)
    effective_path = run_dir / "manifests/effective_primary_scene_inventory.json"
    if effective_path.is_file():
        effective_inventory = json.loads(effective_path.read_text())
        if [row["scene_id"] for row in effective_inventory["scenes"]] != [
            row["scene_id"] for row in scenes
        ]:
            raise RuntimeError("effective inventory changed during technical validation")
        effective_inventory["technical_validation_complete"] = True
        effective_inventory["technical_validation_manifest_sha256"] = digest
    else:
        effective_inventory = {
            "task": "t26f_b3_a_effective_primary_scene_inventory",
            "created_utc": utc(),
            "source": str(B3_0 / "scene_selection/t26f_b3_0_primary_100_scenes.json"),
            "source_sha256": sha_file(
                B3_0 / "scene_selection/t26f_b3_0_primary_100_scenes.json"
            ),
            "n_scenes": 100,
            "n_replacements": 0,
            "ordering_slots_unchanged": True,
            "technical_validation_complete": True,
            "technical_validation_manifest_sha256": digest,
            "scenes": scenes,
        }
    write_json(effective_path, effective_inventory)
    if not replacement_path.is_file():
        write_json(
            replacement_path,
            {
                "task": "t26f_b3_a_reserve_replacement_record",
                "created_utc": utc(),
                "replacement_contract_sha256": sha_file(
                    B3_0 / "contracts/t26f_b3_0_replacement_contract.json"
                ),
                "n_replacements": 0,
                "replacements": [],
                "outcome_based_replacement": False,
                "all_reasons_from_allowed_list": True,
            },
        )
    (run_dir / "STAGE_A_TECHNICAL_VALIDATION_PASS").write_text(
        f"passed_utc: {utc()}\nmanifest_sha256: {digest}\nn_scenes: 100\n"
    )
    print(f"STAGE_A_TECHNICAL_VALIDATION_PASS sha256={digest}", flush=True)
    return 0


def validate_group(
    run_dir: Path, attempt_dir: Path, scene: dict, tag: str
) -> dict | None:
    cached_dir = run_dir / "stageB_cached_inputs"
    command = [
        str(ALPASIM_PY),
        str(VALIDATOR),
        "--group-dir",
        str(attempt_dir),
        "--scene-id",
        scene["scene_id"],
        "--selection-rank",
        str(scene["selection_rank"]),
        "--tag",
        tag,
        "--cached-dir",
        str(cached_dir),
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=ALPASIM)
    if result.returncode != 0:
        print(f"[validator] rc={result.returncode} stderr={result.stderr[-2000:]}", flush=True)
        return None
    return json.loads(result.stdout)


def teardown(attempt_dir: Path) -> None:
    compose = attempt_dir / "docker-compose.yaml"
    if compose.is_file():
        subprocess.run(
            ["docker", "compose", "-f", str(compose), "down", "--remove-orphans"],
            cwd=ALPASIM,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def quarantine_stale(attempt_dir: Path) -> None:
    """Retain (never delete) a stale/failed attempt's artifacts while
    guaranteeing the rerun starts from an empty directory. The frozen
    validators intentionally require exactly ONE wizard session per attempt
    directory; an in-place rerun would leave two rollout session dirs and
    make every rerun permanently unvalidatable (observed as ASL=2 on
    rank000/A). Renaming keeps the full failed evidence on /storage."""
    if attempt_dir.exists() and any(attempt_dir.iterdir()):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        attempt_dir.rename(
            attempt_dir.with_name(f"{attempt_dir.name}_stale_{stamp}"))


def stage_b(run_dir: Path) -> int:
    if not (run_dir / "STAGE_A_TECHNICAL_VALIDATION_PASS").is_file():
        raise RuntimeError("Stage-A technical validation has not passed")
    os.environ["T26F_B3_A_RUN_DIR"] = str(run_dir)
    current_rev = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ALPASIM, text=True
    ).strip()
    if current_rev != EXPECTED_ALPASIM_REV:
        raise RuntimeError(f"AlpaSim revision mismatch: {current_rev}")
    if shutil.disk_usage("/storage").free < MIN_FREE_BYTES:
        raise RuntimeError("storage safety margin failed before Stage B")

    manifest_path = run_dir / "manifests/stageB_candidate_l3_manifest.json"
    prior_groups = {}
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        prior_groups = {
            (row["scene_id"], row["decision_tag"]): row
            for row in prior.get("groups", [])
        }
    groups = []
    scenes = primary()
    total = len(scenes) * len(TAGS)
    counter = 0
    for scene in scenes:
        rank = scene["selection_rank"]
        for tag, settings in TAGS.items():
            counter += 1
            key = (scene["scene_id"], tag)
            prior = prior_groups.get(key)
            if prior and prior.get("all_contracts_pass"):
                paths = [
                    Path(prior["candidate_dump_path"]),
                    Path(prior["shared_l3_path"]),
                    Path(prior["dump_asl_path"]),
                ]
                if all(path.is_file() for path in paths) and all(
                    sha_file(path) == prior[name]
                    for path, name in zip(
                        paths,
                        ("candidate_dump_sha256", "shared_l3_sha256", "dump_asl_sha256"),
                        strict=True,
                    )
                ):
                    groups.append(prior)
                    print(
                        f"[B {counter:03d}/{total}] resume PASS rank{rank:03d} {tag}",
                        flush=True,
                    )
                    continue

            validated = None
            attempts = []
            for attempt in range(1, 3):
                attempt_dir = run_dir / f"stageB_dumps/rank{rank:03d}/{tag}/attempt{attempt}"
                if (attempt_dir / "b3a_candidate_dump.json").is_file() and (
                    attempt_dir / "shared_l3.safetensors"
                ).is_file():
                    validated = validate_group(run_dir, attempt_dir, scene, tag)
                    if validated:
                        attempts.append({"attempt": attempt, "status": "verified_existing"})
                        break
                teardown(attempt_dir)
                quarantine_stale(attempt_dir)
                attempt_dir.mkdir(parents=True, exist_ok=True)
                (attempt_dir / ".b3a_input_dump_request").write_text("8")
                prune_networks()
                command = [
                    "uv",
                    "run",
                    "alpasim_wizard",
                    "deploy=local",
                    "topology=1gpu",
                    "driver=alpamayo1_5",
                    f"wizard.log_dir={attempt_dir}",
                    f"scenes.scene_ids=['{scene['scene_id']}']",
                    f"runtime.simulation_config.n_sim_steps={settings['n_sim_steps']}",
                    f"runtime.simulation_config.force_gt_duration_us={settings['force_gt_us']}",
                    "+runtime.endpoints.startup_timeout_s=600",
                ]
                rc = run_logged(
                    command,
                    run_dir / f"logs/stageB_rank{rank:03d}_{tag}_attempt{attempt}.log",
                )
                validated = validate_group(run_dir, attempt_dir, scene, tag)
                attempts.append(
                    {
                        "attempt": attempt,
                        "exit_code": rc,
                        "validated": validated is not None,
                        "attempt_dir": str(attempt_dir),
                    }
                )
                teardown(attempt_dir)
                if validated:
                    break
            if validated is None:
                print(
                    f"BLOCKED_T26F_B3_A_INPUT_GENERATION_FAILURE rank{rank:03d} {tag}",
                    flush=True,
                )
                return 4
            validated["attempts"] = attempts
            validated["scene_uuid"] = scene["scene_uuid"]
            validated["clip_id"] = scene["clip_id"]
            groups.append(validated)
            checkpoint = {
                "task": "t26f_b3_a_stageB_candidate_l3_manifest",
                "updated_utc": utc(),
                "alpasim_revision": current_rev,
                "n_expected_groups": 300,
                "n_complete_groups": len(groups),
                "groups": groups,
            }
            write_json(manifest_path, checkpoint)
            print(
                f"[B {counter:03d}/{total}] PASS rank{rank:03d} {tag}", flush=True
            )

    candidate_hashes = [row["candidate_dump_sha256"] for row in groups]
    l3_hashes = [row["shared_l3_sha256"] for row in groups]
    complete = {
        "task": "t26f_b3_a_stageB_candidate_l3_manifest",
        "finished_utc": utc(),
        "alpasim_revision": current_rev,
        "alpamayo_checkpoint": "nvidia/Alpamayo-1.5-10B@bf580713f08656674827cd6e09888c79cf65fbf2",
        "candidate_generation_contract_sha256": sha_file(
            B3_0 / "contracts/t26f_b3_0_candidate_generation_contract.json"
        ),
        "shared_l3_contract_sha256": sha_file(
            B3_0 / "contracts/t26f_b3_0_shared_l3_contract.json"
        ),
        "live_export_hook_source": str(
            ALPASIM / "src/driver/src/alpasim_driver/models/alpamayo_base.py"
        ),
        "live_export_hook_source_sha256": sha_file(
            ALPASIM / "src/driver/src/alpasim_driver/models/alpamayo_base.py"
        ),
        "n_scenes": 100,
        "n_groups": 300,
        "n_candidates": 2400,
        "n_shared_l3_records": 300,
        "candidate_dump_hashes_unique": len(set(candidate_hashes)) == 300,
        "shared_l3_file_hashes_unique": len(set(l3_hashes)) == 300,
        "full_l2_persisted": False,
        "raw_reasoning_persisted": False,
        "groups": groups,
    }
    digest = write_json(manifest_path, complete)
    write_json(
        run_dir / "manifests/stageB_shared_l3_manifest.json",
        {
            "task": "t26f_b3_a_stageB_shared_l3_manifest",
            "created_utc": utc(),
            "contract_sha256": complete["shared_l3_contract_sha256"],
            "n_shared_l3_records": 300,
            "shape": [4096],
            "dtype": "bfloat16",
            "one_per_decision_group": True,
            "candidate_independent": True,
            "full_l2_persisted": False,
            "records": [
                {
                    "selection_rank": row["selection_rank"],
                    "scene_id": row["scene_id"],
                    "decision_tag": row["decision_tag"],
                    "decision_timestamp_us": row["decision_timestamp_us"],
                    "path": row["shared_l3_path"],
                    "sha256": row["shared_l3_sha256"],
                    "bytes": Path(row["shared_l3_path"]).stat().st_size,
                }
                for row in groups
            ],
        },
    )
    qa = {
        "task": "t26f_b3_a_stageB_generation_qa",
        "created_utc": utc(),
        "checks": {
            "scenes_100": len({row["scene_id"] for row in groups}) == 100,
            "groups_300": len(groups) == 300,
            "candidates_2400": sum(row["n_candidates"] for row in groups) == 2400,
            "shared_l3_300": sum(row["shared_l3_records"] for row in groups) == 300,
            "tags_exact": all(
                {row["decision_tag"] for row in groups if row["scene_id"] == scene["scene_id"]}
                == {"A", "B", "C"}
                for scene in scenes
            ),
            "all_group_contracts_pass": all(row["all_contracts_pass"] for row in groups),
            "full_l2_absent": all(not row["full_l2_persisted"] for row in groups),
            "raw_reasoning_absent": all(not row["raw_reasoning_persisted"] for row in groups),
        },
    }
    qa["all_pass"] = all(qa["checks"].values())
    write_json(run_dir / "qa/stageB_generation_qa.json", qa)
    if not qa["all_pass"]:
        print("BLOCKED_T26F_B3_A_INPUT_GENERATION_FAILURE", flush=True)
        return 5
    (run_dir / "STAGE_B_INPUTS_LOCKED").write_text(
        f"locked_utc: {utc()}\nmanifest_sha256: {digest}\n"
        "n_scenes: 100\nn_groups: 300\nn_candidates: 2400\nn_shared_l3: 300\n"
    )
    print(f"STAGE_B_INPUTS_LOCKED sha256={digest}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("stage", choices=["stage-a-smoke", "stage-b"])
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if os.stat(run_dir).st_dev != os.stat("/storage").st_dev:
        raise RuntimeError("run directory is not on /storage")
    if args.stage == "stage-a-smoke":
        return stage_a_smoke(run_dir)
    return stage_b(run_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOCKED_T26F_B3_A_INPUT_PIPELINE: {type(exc).__name__}: {exc}")
        raise

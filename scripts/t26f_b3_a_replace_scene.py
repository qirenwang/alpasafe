#!/usr/bin/env python
"""Apply one frozen, outcome-blind B3-A technical scene replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

import t26f_b3_a_stage_a as stage_a


B3_0 = Path(
    "/storage/alpasafe/safeworld-alpamayo/artifacts/"
    "safeworld_t26f_b3_0_preregistration/20260718T180730Z"
)
REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
COMMIT = "ebbb8d5b433bcf072451a55e3d8b86c5ed7a9396"
NAMESPACE = "T26F_B3_WRONG_SCENE_L3_V1"
ALLOWED_REASON = "frozen input construction irrecoverably fails"
ASSET_DIR = Path(
    "/home/qiren/alpasafe/external/alpasim/data/nre-artifacts/all-usdzs"
)
TMP_PARENT = Path(
    "/home/qiren/alpasafe/external/alpasim/data/nre-artifacts/.b3a-download-tmp"
)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_file(path: Path) -> str:
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


def source_scenes() -> list[dict]:
    return json.loads(
        (B3_0 / "scene_selection/t26f_b3_0_primary_100_scenes.json").read_text()
    )["scenes"]


def reserve_scenes() -> list[dict]:
    return json.loads(
        (B3_0 / "scene_selection/t26f_b3_0_reserve_scenes.json").read_text()
    )["scenes"]


def mapping_for(scene_ids: list[str]) -> dict[str, dict[str, str]]:
    output = {}
    for tag in ("A", "B", "C"):
        ordered = sorted(
            scene_ids,
            key=lambda scene_id: (
                hashlib.sha256(f"{NAMESPACE}|{tag}|{scene_id}".encode()).hexdigest(),
                scene_id,
            ),
        )
        output[tag] = {
            recipient: ordered[(index + 1) % len(ordered)]
            for index, recipient in enumerate(ordered)
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--failed-selection-rank", type=int, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if (run_dir / "STAGE_B_INPUTS_LOCKED").exists() or (run_dir / "predictions").exists():
        raise RuntimeError("replacement must precede Stage-B lock and predictions")
    technical_path = run_dir / "audit/stageA_technical_validation.json"
    technical = json.loads(technical_path.read_text())
    failure = next(
        row for row in technical["scenes"]
        if row["selection_rank"] == args.failed_selection_rank and row["status"] == "FAIL"
    )
    if len(failure["attempts"]) != 2 or any(row.get("exit_code") != 1 for row in failure["attempts"]):
        raise RuntimeError("failed scene does not satisfy the frozen two-attempt rule")
    evidence_logs = [
        run_dir / f"logs/stageA_smoke_rank{args.failed_selection_rank:03d}_attempt{attempt}.log"
        for attempt in (1, 2)
    ]
    if not all(path.is_file() for path in evidence_logs):
        raise RuntimeError("replacement evidence logs are missing")
    evidence_text = "\n".join(path.read_text(errors="replace") for path in evidence_logs)
    if evidence_text.count("route folds back on itself") < 2:
        raise RuntimeError("irrecoverable frozen route-construction evidence is absent")

    replacement_path = run_dir / "audit/reserve_replacement_record.json"
    if replacement_path.is_file():
        replacement_doc = json.loads(replacement_path.read_text())
        replacements = replacement_doc.get("replacements", [])
    else:
        replacements = []
    used_reserves = {row["replacement_scene_id"] for row in replacements}
    reserve = next(row for row in reserve_scenes() if row["scene_id"] not in used_reserves)
    reserve_rank = reserve["selection_rank"]

    hf_path = f"sample_set/26.01_release/{reserve['clip_id']}/{reserve['clip_id']}.usdz"
    result = HfApi().get_paths_info(
        REPO_ID, [hf_path], revision=COMMIT, repo_type="dataset", expand=False
    )
    if len(result) != 1 or result[0].path != hf_path:
        raise RuntimeError("reserve exact-path HF metadata query failed")
    info = result[0]
    lfs_sha = getattr(info.lfs, "sha256", None) if info.lfs else None
    lfs_size = getattr(info.lfs, "size", None) if info.lfs else None
    if not isinstance(lfs_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", lfs_sha):
        raise RuntimeError("reserve LFS hash metadata invalid")
    if lfs_size != info.size or not isinstance(info.size, int) or info.size <= 0:
        raise RuntimeError("reserve byte metadata invalid")
    asset = {
        "selection_rank": reserve_rank,
        "scene_id": reserve["scene_id"],
        "scene_uuid": reserve["scene_uuid"],
        "clip_id": reserve["clip_id"],
        "path": hf_path,
        "size_bytes": info.size,
        "blob_id": info.blob_id,
        "lfs_sha256": lfs_sha,
        "lfs_size_bytes": lfs_size,
    }
    destination = ASSET_DIR / f"{reserve['scene_uuid']}.usdz"
    attempts = []
    status = None
    if destination.is_file():
        checks = stage_a.verify(destination, asset)
        attempts.append({"kind": "existing_local_file", **checks})
        if checks["all_ok"]:
            status = "verified_existing"
    if status is None:
        for attempt in range(1, 3):
            try:
                if shutil.disk_usage("/storage").free < stage_a.MIN_FREE_BYTES:
                    raise RuntimeError("storage margin failed before reserve download")
                with tempfile.TemporaryDirectory(dir=TMP_PARENT) as temporary:
                    downloaded = Path(hf_hub_download(
                        REPO_ID, hf_path, repo_type="dataset", revision=COMMIT,
                        local_dir=temporary,
                    ))
                    checks = stage_a.verify(downloaded, asset)
                    attempts.append({"kind": f"download_attempt_{attempt}", **checks})
                    if checks["all_ok"]:
                        os.replace(downloaded, destination)
                        final = stage_a.verify(destination, asset)
                        if not final["all_ok"]:
                            raise RuntimeError("reserve final verification failed")
                        status = "downloaded_verified"
                        break
            except Exception as exc:
                attempts.append({"kind": f"download_attempt_{attempt}_exception",
                                 "error": repr(exc)[:1000]})
    if status not in {"verified_existing", "downloaded_verified"}:
        raise RuntimeError("earliest unused reserve asset could not be verified")
    download_record = {
        "selection_rank": reserve_rank,
        "effective_selection_rank": args.failed_selection_rank,
        "scene_id": reserve["scene_id"],
        "scene_uuid": reserve["scene_uuid"],
        "clip_id": reserve["clip_id"],
        "hf_path": hf_path,
        "source_commit": COMMIT,
        "frozen_size_bytes": info.size,
        "frozen_lfs_sha256": lfs_sha,
        "destination": str(destination),
        "attempts": attempts,
        "status": status,
        "replacement_reserve": True,
    }
    download_audit_path = run_dir / "audit/stageA_download_verification.json"
    download_audit = json.loads(download_audit_path.read_text())
    if not any(row["scene_id"] == reserve["scene_id"] for row in download_audit["downloads"]):
        download_audit["downloads"].append(download_record)
    download_audit.update({
        "updated_utc": utc(),
        "n_primary_verified": 100,
        "n_reserve_verified": len(used_reserves) + 1,
        "n_verified": 100 + len(used_reserves) + 1,
        "all_effective_assets_verified": True,
    })
    download_sha = write_json(download_audit_path, download_audit)
    write_json(
        run_dir / f"manifests/stageA_reserve_metadata_addendum_{len(replacements)+1}.json",
        {
            "task": "t26f_b3_a_reserve_metadata_addendum",
            "created_utc": utc(),
            "replacement_number": len(replacements) + 1,
            "repo_id": REPO_ID,
            "immutable_revision": COMMIT,
            "asset": asset,
        },
    )

    inventory_path = run_dir / "manifests/effective_primary_scene_inventory.json"
    if inventory_path.is_file():
        scenes = json.loads(inventory_path.read_text())["scenes"]
    else:
        scenes = source_scenes()
    replaced = dict(scenes[args.failed_selection_rank])
    effective_reserve = dict(reserve)
    effective_reserve.update({
        "selection_rank": args.failed_selection_rank,
        "frozen_reserve_selection_rank": reserve_rank,
        "replaces_scene_id": failure["scene_id"],
        "replacement_number": len(replacements) + 1,
    })
    scenes[args.failed_selection_rank] = effective_reserve
    if len(scenes) != 100 or len({row["scene_id"] for row in scenes}) != 100:
        raise RuntimeError("effective inventory coverage failure")
    inventory = {
        "task": "t26f_b3_a_effective_primary_scene_inventory",
        "created_utc": utc(),
        "source": str(B3_0 / "scene_selection/t26f_b3_0_primary_100_scenes.json"),
        "source_sha256": sha_file(
            B3_0 / "scene_selection/t26f_b3_0_primary_100_scenes.json"
        ),
        "n_scenes": 100,
        "n_replacements": len(replacements) + 1,
        "ordering_slots_unchanged": True,
        "scenes": scenes,
    }
    inventory_sha = write_json(inventory_path, inventory)

    mapping = mapping_for([row["scene_id"] for row in scenes])
    mapping_valid = all(
        set(mapping[tag]) == {row["scene_id"] for row in scenes}
        and set(mapping[tag].values()) == {row["scene_id"] for row in scenes}
        and all(recipient != donor for recipient, donor in mapping[tag].items())
        for tag in ("A", "B", "C")
    )
    if not mapping_valid:
        raise RuntimeError("repaired wrong-scene mapping QA failure")
    mapping_doc = {
        "task": "t26f_b3_a_wrong_scene_l3_mapping_addendum",
        "created_utc": utc(),
        "version": len(replacements) + 1,
        "namespace": NAMESPACE,
        "rule": "per decision tag, sort effective scenes by SHA256(namespace|tag|scene_id), tie-break scene_id, map to next cyclic scene",
        "effective_inventory_sha256": inventory_sha,
        "mapping": mapping,
        "qa": {"n_entries": 300, "per_tag_bijection": True,
               "never_same_scene": True, "tag_matched": True},
    }
    mapping_path = run_dir / "manifests/effective_wrong_scene_l3_mapping.json"
    mapping_sha = write_json(mapping_path, mapping_doc)

    evidence = {
        "attempt_logs": [
            {"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size}
            for path in evidence_logs
        ],
        "failed_asls": [
            {"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size}
            for attempt in (1, 2)
            for path in (run_dir / f"stageA_smoke_attempts/rank{args.failed_selection_rank:03d}/attempt{attempt}").glob(
                f"rollouts/{failure['scene_id']}/*/rollout.asl"
            )
        ],
        "exact_error": "Waypoint 201 fails sanity check: route folds back on itself with angle 180.00 deg",
    }
    replacement = {
        "replacement_number": len(replacements) + 1,
        "effective_selection_rank": args.failed_selection_rank,
        "failed_scene_id": failure["scene_id"],
        "failed_scene_uuid": failure["scene_uuid"],
        "failed_scene_record": replaced,
        "allowed_reason": ALLOWED_REASON,
        "decision_before_target_analysis": True,
        "outcomes_read_or_used": False,
        "attempts_exhausted": 2,
        "replacement_scene_id": reserve["scene_id"],
        "replacement_scene_uuid": reserve["scene_uuid"],
        "frozen_reserve_selection_rank": reserve_rank,
        "earliest_unused_reserve": True,
        "asset_download_sha256": download_record["frozen_lfs_sha256"],
        "evidence": evidence,
        "effective_inventory_sha256": inventory_sha,
        "repaired_mapping_sha256": mapping_sha,
    }
    replacements.append(replacement)
    replacement_doc = {
        "task": "t26f_b3_a_reserve_replacement_record",
        "created_utc": utc(),
        "replacement_contract_sha256": sha_file(
            B3_0 / "contracts/t26f_b3_0_replacement_contract.json"
        ),
        "n_replacements": len(replacements),
        "replacements": replacements,
        "outcome_based_replacement": False,
        "all_reasons_from_allowed_list": True,
    }
    replacement_sha = write_json(replacement_path, replacement_doc)
    (run_dir / "STAGE_A_ASSETS_VERIFIED").write_text(
        f"verified_utc: {utc()}\nmanifest_sha256: {download_sha}\n"
        f"n_primary_verified: 100\nn_reserve_verified: {len(replacements)}\n"
        f"effective_inventory_sha256: {inventory_sha}\n"
        f"replacement_record_sha256: {replacement_sha}\n"
    )
    print(json.dumps({
        "replacement_number": len(replacements),
        "failed_scene_id": failure["scene_id"],
        "replacement_scene_id": reserve["scene_id"],
        "frozen_reserve_selection_rank": reserve_rank,
        "allowed_reason": ALLOWED_REASON,
        "asset_status": status,
        "effective_inventory_sha256": inventory_sha,
        "mapping_sha256": mapping_sha,
    }, indent=1))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOCKED_T26F_B3_A_REPLACEMENT_FAILURE: {type(exc).__name__}: {exc}")
        raise

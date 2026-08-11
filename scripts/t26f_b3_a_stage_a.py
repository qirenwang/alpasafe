#!/usr/bin/env python
"""T26F-B3-A Stage A exact metadata lock, download, and asset verification.

This program is intentionally outcome-blind. It reads only the frozen B3-0
primary scene inventory, queries metadata for those exact immutable paths, and
downloads/verifies the corresponding USDZ assets. It never reads any target,
rollout, evaluator output, or sealed-label artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import huggingface_hub
import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile


B3_0 = Path(
    "/storage/alpasafe/safeworld-alpamayo/artifacts/"
    "safeworld_t26f_b3_0_preregistration/20260718T180730Z"
)
REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
COMMIT = "ebbb8d5b433bcf072451a55e3d8b86c5ed7a9396"
ASSET_DIR = Path(
    "/home/qiren/alpasafe/external/alpasim/data/nre-artifacts/all-usdzs"
)
TMP_PARENT = Path(
    "/home/qiren/alpasafe/external/alpasim/data/nre-artifacts/.b3a-download-tmp"
)
SEALED_IDS = {
    "clipgt-41424eff-22b8-4b7d-8cc9-3197c3dc2825",
    "clipgt-7747986c-9aae-450d-b10c-b7c42b882c68",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_ATTEMPTS = 2
MIN_FREE_BYTES = 624_000_000_000


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return hashlib.sha256(text.encode()).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def internal_uuid(path: Path) -> str | None:
    with zipfile.ZipFile(path) as archive, archive.open("metadata.yaml") as fh:
        metadata = yaml.safe_load(fh)
    return metadata.get("uuid")


def ensure_storage(path: Path) -> dict:
    real = path.resolve()
    storage_dev = os.stat("/storage").st_dev
    actual_dev = os.stat(path if path.exists() else path.parent).st_dev
    return {
        "logical_path": str(path),
        "realpath": str(real),
        "device": actual_dev,
        "storage_device": storage_dev,
        "on_storage": actual_dev == storage_dev,
    }


def load_primary() -> list[dict]:
    data = json.loads(
        (B3_0 / "scene_selection/t26f_b3_0_primary_100_scenes.json").read_text()
    )
    scenes = data["scenes"]
    ids = [row["scene_id"] for row in scenes]
    assert len(scenes) == 100 and len(set(ids)) == 100
    assert not (set(ids) & SEALED_IDS)
    assert [row["selection_rank"] for row in scenes] == list(range(100))
    return scenes


def exact_paths(scenes: list[dict]) -> list[str]:
    return [
        f"sample_set/26.01_release/{s['clip_id']}/{s['clip_id']}.usdz"
        for s in scenes
    ]


def query_metadata(run_dir: Path, scenes: list[dict]) -> dict:
    path = run_dir / "manifests/stageA_exact_asset_metadata.json"
    marker = run_dir / "STAGE_A_METADATA_LOCKED"
    if path.is_file() and marker.is_file():
        data = json.loads(path.read_text())
        if (
            data.get("resolved_commit_sha") == COMMIT
            and data.get("n_assets") == 100
            and [r["scene_id"] for r in data["assets"]]
            == [s["scene_id"] for s in scenes]
        ):
            print("STAGE_A_METADATA_ALREADY_LOCKED", flush=True)
            return data
        raise RuntimeError("existing Stage-A metadata lock is inconsistent")

    paths = exact_paths(scenes)
    api = HfApi()
    print(f"[metadata] querying {len(paths)} exact paths at {COMMIT}", flush=True)
    results = api.get_paths_info(
        REPO_ID, paths, revision=COMMIT, repo_type="dataset", expand=False
    )
    by_path = {row.path: row for row in results}
    if len(results) != 100 or set(by_path) != set(paths):
        raise RuntimeError(
            f"metadata path mismatch: n={len(results)}, "
            f"missing={sorted(set(paths) - set(by_path))[:5]}, "
            f"extra={sorted(set(by_path) - set(paths))[:5]}"
        )

    assets = []
    raw = []
    for scene, hf_path in zip(scenes, paths, strict=True):
        row = by_path[hf_path]
        if not isinstance(row, RepoFile) or not isinstance(row.size, int) or row.size <= 0:
            raise RuntimeError(f"invalid RepoFile metadata for {hf_path}")
        lfs = row.lfs
        digest = getattr(lfs, "sha256", None) if lfs else None
        lfs_size = getattr(lfs, "size", None) if lfs else None
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise RuntimeError(f"invalid LFS SHA256 for {hf_path}")
        if lfs_size != row.size:
            raise RuntimeError(f"LFS size mismatch for {hf_path}")
        assets.append(
            {
                "selection_rank": scene["selection_rank"],
                "scene_id": scene["scene_id"],
                "scene_uuid": scene["scene_uuid"],
                "clip_id": scene["clip_id"],
                "path": hf_path,
                "size_bytes": row.size,
                "blob_id": row.blob_id,
                "lfs_sha256": digest,
                "lfs_size_bytes": lfs_size,
            }
        )
        raw.append(
            {
                "path": row.path,
                "size": row.size,
                "blob_id": row.blob_id,
                "lfs": {"sha256": digest, "size": lfs_size},
                "type": "file",
            }
        )

    total = sum(row["size_bytes"] for row in assets)
    free = shutil.disk_usage("/storage").free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"storage margin failed: free={free}")
    manifest = {
        "task": "t26f_b3_a_stageA_exact_asset_metadata",
        "created_utc": utc(),
        "repo_id": REPO_ID,
        "repo_type": "dataset",
        "requested_revision": COMMIT,
        "resolved_commit_sha": COMMIT,
        "revision_is_immutable_commit": True,
        "huggingface_hub_version": huggingface_hub.__version__,
        "query_mechanism": "HfApi.get_paths_info(100 exact frozen primary paths, immutable revision, expand=False)",
        "n_assets": 100,
        "assets": assets,
        "total_download_bytes": total,
        "storage_free_bytes_before_download": free,
        "storage_margin_required_bytes": MIN_FREE_BYTES,
        "raw_response": raw,
    }
    digest = canonical_write(path, manifest)
    marker.write_text(
        f"locked_utc: {utc()}\nmanifest_sha256: {digest}\n"
        f"resolved_commit_sha: {COMMIT}\ntotal_download_bytes: {total}\n"
    )
    print(
        f"STAGE_A_METADATA_LOCKED assets=100 bytes={total} sha256={digest}",
        flush=True,
    )
    return manifest


def verify(path: Path, asset: dict) -> dict:
    record = {
        "bytes": path.stat().st_size,
        "bytes_match_frozen": path.stat().st_size == asset["size_bytes"],
    }
    record["sha256"] = sha_file(path) if record["bytes_match_frozen"] else None
    record["sha256_match_frozen_lfs"] = record["sha256"] == asset["lfs_sha256"]
    if record["sha256_match_frozen_lfs"]:
        try:
            value = internal_uuid(path)
        except Exception as exc:  # preserve exact technical evidence
            value = None
            record["internal_metadata_error"] = repr(exc)[:500]
        record["internal_metadata_uuid"] = value
        record["internal_uuid_match_catalog"] = value == asset["scene_uuid"]
    else:
        record["internal_metadata_uuid"] = None
        record["internal_uuid_match_catalog"] = False
    storage = ensure_storage(path)
    record["storage"] = storage
    record["all_ok"] = bool(
        record["bytes_match_frozen"]
        and record["sha256_match_frozen_lfs"]
        and record["internal_uuid_match_catalog"]
        and storage["on_storage"]
    )
    return record


def load_resume_records(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {row["scene_id"]: row for row in data.get("downloads", [])}


def download(run_dir: Path, metadata: dict) -> int:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    TMP_PARENT.mkdir(parents=True, exist_ok=True)
    storage_checks = {
        "asset_dir": ensure_storage(ASSET_DIR),
        "tmp_parent": ensure_storage(TMP_PARENT),
        "hf_home": ensure_storage(Path(os.environ["HF_HOME"])),
    }
    if not all(row["on_storage"] for row in storage_checks.values()):
        raise RuntimeError(f"non-storage Stage-A path: {storage_checks}")

    out_path = run_dir / "audit/stageA_download_verification.json"
    existing = load_resume_records(out_path)
    records: list[dict] = []
    assets = metadata["assets"]
    for index, asset in enumerate(assets, 1):
        dest = ASSET_DIR / f"{asset['scene_uuid']}.usdz"
        rec = {
            "selection_rank": asset["selection_rank"],
            "scene_id": asset["scene_id"],
            "scene_uuid": asset["scene_uuid"],
            "clip_id": asset["clip_id"],
            "hf_path": asset["path"],
            "source_commit": COMMIT,
            "frozen_size_bytes": asset["size_bytes"],
            "frozen_lfs_sha256": asset["lfs_sha256"],
            "destination": str(dest),
            "attempts": [],
        }
        prior = existing.get(asset["scene_id"])
        if prior and prior.get("status") in {"verified_existing", "downloaded_verified"} and dest.is_file():
            checks = verify(dest, asset)
            rec["attempts"].append({"kind": "resume_reverify", **checks})
            if checks["all_ok"]:
                rec["status"] = "verified_existing"
                records.append(rec)
                print(f"[{index:03d}/100] verified_existing {asset['scene_id']}", flush=True)
                continue
        elif dest.is_file():
            checks = verify(dest, asset)
            rec["attempts"].append({"kind": "existing_local_file", **checks})
            if checks["all_ok"]:
                rec["status"] = "verified_existing"
                records.append(rec)
                print(f"[{index:03d}/100] verified_existing {asset['scene_id']}", flush=True)
                continue
            rec["existing_file_failed_checks_not_used"] = True

        status = "failed_before_attempt"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                free = shutil.disk_usage("/storage").free
                if free < MIN_FREE_BYTES:
                    raise RuntimeError(f"storage margin failed before asset: free={free}")
                with tempfile.TemporaryDirectory(dir=TMP_PARENT) as temp_dir:
                    downloaded = Path(
                        hf_hub_download(
                            REPO_ID,
                            asset["path"],
                            repo_type="dataset",
                            revision=COMMIT,
                            local_dir=temp_dir,
                        )
                    )
                    checks = verify(downloaded, asset)
                    rec["attempts"].append(
                        {"kind": f"download_attempt_{attempt}", **checks}
                    )
                    if checks["all_ok"]:
                        os.replace(downloaded, dest)
                        final_checks = verify(dest, asset)
                        rec["final_destination_verification"] = final_checks
                        if not final_checks["all_ok"]:
                            raise RuntimeError("final destination re-verification failed")
                        status = "downloaded_verified"
                        break
                    status = "failed_verification"
            except Exception as exc:
                rec["attempts"].append(
                    {"kind": f"download_attempt_{attempt}_exception", "error": repr(exc)[:1000]}
                )
                status = "failed_exception"
        rec["status"] = status
        records.append(rec)
        checkpoint = {
            "task": "t26f_b3_a_stageA_download_verification",
            "updated_utc": utc(),
            "repo_id": REPO_ID,
            "revision_used_for_download": COMMIT,
            "revision_is_immutable_commit": True,
            "storage_checks": storage_checks,
            "n_expected": 100,
            "n_verified": sum(
                r.get("status") in {"verified_existing", "downloaded_verified"}
                for r in records
            ),
            "downloads": records,
        }
        canonical_write(out_path, checkpoint)
        print(f"[{index:03d}/100] {status} {asset['scene_id']}", flush=True)
        if status != "downloaded_verified":
            print("BLOCKED_T26F_B3_A_STAGE_A_ASSET_FAILURE", flush=True)
            return 3

    complete = {
        "task": "t26f_b3_a_stageA_download_verification",
        "started_or_resumed_utc": utc(),
        "finished_utc": utc(),
        "repo_id": REPO_ID,
        "revision_used_for_download": COMMIT,
        "revision_is_immutable_commit": True,
        "storage_checks": storage_checks,
        "n_expected": 100,
        "n_verified": 100,
        "all_primary_assets_verified": True,
        "downloads": records,
    }
    digest = canonical_write(out_path, complete)
    (run_dir / "STAGE_A_ASSETS_VERIFIED").write_text(
        f"verified_utc: {utc()}\nmanifest_sha256: {digest}\nn_verified: 100\n"
    )
    print(f"STAGE_A_ALL_100_ASSETS_VERIFIED sha256={digest}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("audit", "logs", "manifests", "qa", "reports", "code_artifacts"):
        (run_dir / name).mkdir(exist_ok=True)
    if not ensure_storage(run_dir)["on_storage"]:
        raise RuntimeError(f"B3-A run directory is not on /storage: {run_dir}")
    if shutil.disk_usage("/storage").free < MIN_FREE_BYTES:
        raise RuntimeError("storage safety margin failed")
    scenes = load_primary()
    metadata = query_metadata(run_dir, scenes)
    if args.metadata_only:
        return 0
    return download(run_dir, metadata)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOCKED_T26F_B3_A_STAGE_A: {type(exc).__name__}: {exc}", flush=True)
        raise

#!/usr/bin/env python
"""Validate one frozen T26F-B3-A candidate/shared-L3 decision group."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from alpasim_utils.logs import async_read_pb_log
from safetensors import safe_open


TAGS = {"A": 1_700_000, "B": 2_700_000, "C": 3_700_000}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


async def asl_metadata(path: Path) -> dict:
    start = None
    force_gt = None
    requests = []
    async for entry in async_read_pb_log(str(path)):
        which = entry.WhichOneof("log_entry")
        if which == "rollout_metadata":
            metadata = entry.rollout_metadata
            start = int(metadata.session_metadata.start_timestamp_us)
            force_gt = int(metadata.force_gt_duration)
        elif which == "driver_request":
            requests.append(int(entry.driver_request.time_now_us))
    return {"start_timestamp_us": start, "force_gt_duration_us": force_gt, "requests": requests}


def forbidden_keys(value: object, prefix: str = "") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if "reason" in key.lower() or key.lower() in {"cot", "chain_of_thought"}:
                found.append(path)
            found.extend(forbidden_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_keys(child, f"{prefix}[{index}]"))
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--selection-rank", type=int, required=True)
    parser.add_argument("--tag", choices=sorted(TAGS), required=True)
    parser.add_argument("--cached-dir", type=Path, required=True)
    args = parser.parse_args()

    group = args.group_dir.resolve()
    candidate_path = group / "b3a_candidate_dump.json"
    l3_path = group / "shared_l3.safetensors"
    asl_paths = list(group.glob(f"rollouts/{args.scene_id}/*/rollout.asl"))
    if not candidate_path.is_file() or not l3_path.is_file() or len(asl_paths) != 1:
        raise RuntimeError(
            f"missing input artifacts: candidate={candidate_path.is_file()} "
            f"L3={l3_path.is_file()} ASL={len(asl_paths)}"
        )

    document = json.loads(candidate_path.read_text())
    forbidden = forbidden_keys(document)
    if forbidden:
        raise RuntimeError(f"raw-reasoning fields persisted: {forbidden}")
    candidates = document.get("candidates", [])
    if len(candidates) != 8 or [c.get("sample_index") for c in candidates] != list(range(8)):
        raise RuntimeError("candidate count/order is not exact K=8 sample_index 0..7")
    if document.get("frame") != "ego_t0_rig" or document.get("output_frequency_hz") != 10:
        raise RuntimeError("candidate frame/frequency contract failure")
    if document.get("top_p") != 0.98 or document.get("temperature") != 0.6:
        raise RuntimeError("candidate decoding contract failure")
    trajectories = np.asarray([c["trajectory_xy"] for c in candidates], dtype=np.float32)
    headings = np.asarray([c["headings_rad"] for c in candidates], dtype=np.float32)
    if trajectories.shape != (8, 64, 2) or headings.shape != (8, 64):
        raise RuntimeError(
            f"candidate tensor contract failure: {trajectories.shape}, {headings.shape}"
        )
    if not np.isfinite(trajectories).all() or not np.isfinite(headings).all():
        raise RuntimeError("non-finite candidate tensor")
    min_pairwise = min(
        float(np.max(np.abs(trajectories[a] - trajectories[b])))
        for a in range(8)
        for b in range(a + 1, 8)
    )
    if not min_pairwise > 0:
        raise RuntimeError("candidate uniqueness contract failure")

    with safe_open(l3_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if keys != ["L3"]:
            raise RuntimeError(f"shared latent file keys are {keys}, expected only L3")
        l3 = handle.get_tensor("L3")
    if l3.shape != torch.Size([4096]) or l3.dtype != torch.bfloat16:
        raise RuntimeError(f"L3 contract failure: {l3.shape} {l3.dtype}")
    if not torch.isfinite(l3.float()).all():
        raise RuntimeError("non-finite L3")
    if document.get("shared_l3", {}).get("prefill_calls") != 1:
        raise RuntimeError("shared L3 was not captured from exactly one prompt prefill")
    if document.get("shared_l3", {}).get("full_L2_persisted") is not False:
        raise RuntimeError("full-L2 non-persistence assertion is absent")

    forbidden_files = [
        str(path.relative_to(group))
        for path in group.rglob("*")
        if path.is_file()
        and ("reason" in path.name.lower() or "l2" in path.name.lower() or "kv" in path.name.lower())
    ]
    if forbidden_files:
        raise RuntimeError(f"forbidden persisted files: {forbidden_files}")

    asl = asyncio.run(asl_metadata(asl_paths[0]))
    force_gt = TAGS[args.tag]
    requests = asl["requests"]
    decision = int(document["latest_timestamp_us"])
    pre = [value for value in requests if value < asl["start_timestamp_us"] + force_gt]
    if (
        asl["force_gt_duration_us"] != force_gt
        or pre
        or not requests
        or requests[0] != decision
    ):
        raise RuntimeError(f"decision-conditioning contract failure: {asl}")
    decision_rel_s = (decision - asl["start_timestamp_us"]) / 1e6
    cached_n_sim_steps = max(80, min(110, math.floor((decision_rel_s + 6.4 - 0.3) / 0.1)))

    args.cached_dir.mkdir(parents=True, exist_ok=True)
    cached_inputs = []
    for index, candidate in enumerate(candidates):
        output = args.cached_dir / f"rank{args.selection_rank:03d}_{args.tag}_{index}.json"
        payload = {
            "frame": "ego_t0_rig",
            "output_frequency_hz": 10,
            "candidate": {
                "sample_index": index,
                "trajectory_xy": candidate["trajectory_xy"],
                "headings_rad": candidate["headings_rad"],
                "horizon_T": 64,
            },
        }
        output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        cached_inputs.append(
            {
                "candidate_index": index,
                "path": str(output),
                "sha256": sha_file(output),
                "trajectory_canonical_sha256": canonical_sha(candidate["trajectory_xy"]),
            }
        )

    result = {
        "scene_id": args.scene_id,
        "selection_rank": args.selection_rank,
        "decision_tag": args.tag,
        "force_gt_duration_us": force_gt,
        "start_timestamp_us": asl["start_timestamp_us"],
        "decision_timestamp_us": decision,
        "decision_rel_s": decision_rel_s,
        "cached_n_sim_steps": cached_n_sim_steps,
        "n_candidates": 8,
        "candidate_shape": [8, 64, 2],
        "candidate_dtype": "float32",
        "candidate_frame": "ego_t0_rig",
        "frequency_hz": 10,
        "candidate_indices": list(range(8)),
        "min_pairwise_max_abs_m": min_pairwise,
        "candidate_dump_path": str(candidate_path),
        "candidate_dump_sha256": sha_file(candidate_path),
        "shared_l3_path": str(l3_path),
        "shared_l3_sha256": sha_file(l3_path),
        "shared_l3_tensor_sha256": hashlib.sha256(l3.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "shared_l3_shape": [4096],
        "shared_l3_dtype": "bfloat16",
        "shared_l3_records": 1,
        "full_l2_persisted": False,
        "raw_reasoning_persisted": False,
        "dump_asl_path": str(asl_paths[0]),
        "dump_asl_sha256": sha_file(asl_paths[0]),
        "cached_inputs": cached_inputs,
        "all_contracts_pass": True,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

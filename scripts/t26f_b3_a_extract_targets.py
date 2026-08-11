#!/usr/bin/env python
"""Extract frozen B3-A targets after official post-evaluation completes."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


REPO = Path("/home/qiren/alpasafe/safeworld-alpamayo")
ALPASIM = Path("/home/qiren/alpasafe/external/alpasim")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(ALPASIM / "src/utils"))
sys.path.insert(0, str(ALPASIM / "src/grpc"))

from alpasim_utils.logs import async_read_pb_log  # noqa: E402
from safeworld.real_smoke.asl_geometry import (  # noqa: E402
    aabb_xy_to_rig_xy,
    local_xy_to_ego_t0,
    quat_wxyz_to_yaw,
)


EVAL_REV = "196d21ab86593af121b055995d0185bb786d1f70"
ROLLOUT_REV = "a1f05bb628f3d1d19d79d44188e836e9108f98c6"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pose_fields(pose) -> dict:
    value, quat = pose.vec, pose.quat
    return {
        "pos_xyz": [float(value.x), float(value.y), float(value.z)],
        "quat_wxyz": [float(quat.w), float(quat.x), float(quat.y), float(quat.z)],
    }


async def read_asl(path: Path) -> dict:
    metadata = {}
    rig_to_aabb = None
    poses = []
    requests = []
    first_return = None
    async for entry in async_read_pb_log(str(path)):
        which = entry.WhichOneof("log_entry")
        if which == "rollout_metadata":
            value = entry.rollout_metadata
            session = value.session_metadata
            metadata = {
                "session_uuid": session.session_uuid,
                "scene_id": session.scene_id,
                "n_sim_steps": int(session.n_sim_steps),
                "start_timestamp_us": int(session.start_timestamp_us),
                "control_timestep_us": int(session.control_timestep_us),
                "force_gt_duration_us": int(value.force_gt_duration),
                "nre_version": getattr(session, "nre_version", ""),
                "nre_uuid": getattr(session, "nre_uuid", ""),
            }
            if value.HasField("transform_ego_coords_rig_to_aabb"):
                rig_to_aabb = pose_fields(value.transform_ego_coords_rig_to_aabb)
        elif which == "actor_poses":
            actor_poses = entry.actor_poses
            frame = {"timestamp_us": int(actor_poses.timestamp_us), "actors": {}}
            for actor in actor_poses.actor_poses:
                frame["actors"][actor.actor_id] = pose_fields(actor.actor_pose)
            poses.append(frame)
        elif which == "driver_request":
            requests.append(int(entry.driver_request.time_now_us))
        elif which == "driver_return" and first_return is None:
            first_return = [
                {"timestamp_us": int(point.timestamp_us), **pose_fields(point.pose)}
                for point in entry.driver_return.trajectory.poses
            ]
    return {
        "metadata": metadata,
        "rig_to_aabb": rig_to_aabb,
        "poses": poses,
        "requests": requests,
        "first_return": first_return,
    }


def extract_one(args: tuple[dict, dict, dict]) -> dict:
    rollout, group, summary = args
    asl_path = Path(rollout["rollout_asl_path"])
    parsed = asyncio.run(read_asl(asl_path))
    if not parsed["requests"] or parsed["rig_to_aabb"] is None or not parsed["first_return"]:
        raise RuntimeError(f"ASL structural content missing: {asl_path}")
    decision = parsed["requests"][0]
    if decision != group["decision_timestamp_us"]:
        raise RuntimeError(
            f"decision timestamp mismatch {decision} != {group['decision_timestamp_us']}"
        )
    if parsed["metadata"]["scene_id"] != rollout["scene_id"]:
        raise RuntimeError("ASL scene mismatch")
    anchor = parsed["first_return"][0]
    x0, y0 = anchor["pos_xyz"][:2]
    yaw0 = quat_wxyz_to_yaw(*anchor["quat_wxyz"])
    frames = sorted(
        [
            frame
            for frame in parsed["poses"]
            if frame["timestamp_us"] >= decision and "EGO" in frame["actors"]
        ],
        key=lambda frame: frame["timestamp_us"],
    )
    if not frames or len(frames) > 64:
        raise RuntimeError(f"future frame coverage invalid: {len(frames)}")
    aabb_xy = np.asarray([frame["actors"]["EGO"]["pos_xyz"][:2] for frame in frames])
    quats = [frame["actors"]["EGO"]["quat_wxyz"] for frame in frames]
    yaws = np.asarray([quat_wxyz_to_yaw(*quat) for quat in quats])
    rig_xy = aabb_xy_to_rig_xy(aabb_xy, yaws, parsed["rig_to_aabb"]["pos_xyz"][0])
    ego_t0 = local_xy_to_ego_t0(rig_xy, x0, y0, yaw0)
    metrics = summary["score_metrics"]
    required = {
        "progress_clipped_rel",
        "progress_rel",
        "progress_score",
        "collision_at_fault",
        "offroad",
        "gt_dist_traveled_m",
    }
    if not required.issubset(metrics):
        raise RuntimeError(f"official score metrics missing {sorted(required - set(metrics))}")
    score = summary["score"]
    if score is None or not 0 <= float(score) <= 1:
        raise RuntimeError(f"official score invalid: {score}")
    candidate = next(
        row for row in group["cached_inputs"] if row["candidate_index"] == rollout["candidate_index"]
    )
    candidate_doc = json.loads(Path(candidate["path"]).read_text())
    if any("reason" in key.lower() for key in candidate_doc.get("candidate", {})):
        raise RuntimeError("reasoning unexpectedly present in candidate target join source")
    return {
        "scene_id": rollout["scene_id"],
        "scene_uuid": rollout["scene_uuid"],
        "selection_rank": rollout["selection_rank"],
        "decision_tag": rollout["decision_tag"],
        "decision_timestamp_us": rollout["decision_timestamp_us"],
        "candidate_index": rollout["candidate_index"],
        "sample_id": (
            f"{rollout['scene_id']}@{rollout['decision_timestamp_us']}#"
            f"cand_{rollout['candidate_index']}"
        ),
        "rollout_id": Path(rollout["rollout_asl_path"]).parent.name,
        "official_alpasim_scene_score": float(score),
        "official_score_status": summary["status"],
        "official_score_passed": summary["passed"],
        "official_score_failure_reason": summary["failure_reason"],
        "official_score_metrics": metrics,
        "progress_clipped_rel": float(metrics["progress_clipped_rel"]),
        "collision_at_fault": float(metrics["collision_at_fault"]),
        "offroad": float(metrics["offroad"]),
        "planned_trajectory": candidate_doc["candidate"]["trajectory_xy"],
        "future_timestamps_us": [frame["timestamp_us"] for frame in frames],
        "future_ego_states_ego_t0": ego_t0.tolist(),
        "future_valid_steps": len(frames),
        "future_ego_states_global": [
            [*frame["actors"]["EGO"]["pos_xyz"], *frame["actors"]["EGO"]["quat_wxyz"]]
            for frame in frames
        ],
        "rollout_metadata": parsed["metadata"],
        "candidate_input_sha256": rollout["candidate_input_sha256"],
        "trajectory_canonical_sha256": rollout["trajectory_canonical_sha256"],
        "rollout_asl_path": str(asl_path),
        "rollout_asl_sha256": rollout["rollout_asl_sha256"],
        "posteval_metrics_path": summary["posteval_metrics_path"],
        "posteval_metrics_sha256": summary["posteval_metrics_sha256"],
        "rollout_alpasim_revision": ROLLOUT_REV,
        "evaluator_alpasim_revision": EVAL_REV,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    stage_d = json.loads(
        (run_dir / "manifests/stageD_alpasim_rollout_manifest.json").read_text()
    )
    stage_b = json.loads(
        (run_dir / "manifests/stageB_candidate_l3_manifest.json").read_text()
    )
    summary_path = run_dir / "posteval/jobs/aggregate/results-summary.json"
    summary = json.loads(summary_path.read_text())
    summary_by_id = {row["rollout_id"]: row for row in summary["rollouts"]}
    group_by_key = {
        (row["scene_id"], row["decision_tag"]): row for row in stage_b["groups"]
    }
    posteval_by_key = {}
    post_manifest = json.loads(
        (run_dir / "manifests/stageE_posteval_file_manifest.json").read_text()
    )
    for row in post_manifest["jobs"]:
        posteval_by_key[
            (row["scene_id"], row["decision_tag"], row["candidate_index"])
        ] = row
    work = []
    for rollout in stage_d["rollouts"]:
        rollout_id = Path(rollout["rollout_asl_path"]).parent.name
        score = dict(summary_by_id[rollout_id])
        post = posteval_by_key[
            (rollout["scene_id"], rollout["decision_tag"], rollout["candidate_index"])
        ]
        score["posteval_metrics_path"] = post["metrics_path"]
        score["posteval_metrics_sha256"] = post["metrics_sha256"]
        work.append(
            (
                rollout,
                group_by_key[(rollout["scene_id"], rollout["decision_tag"])],
                score,
            )
        )
    if len(work) != 2400:
        raise RuntimeError(f"target extraction work count is {len(work)}")
    with ThreadPoolExecutor(max_workers=12) as executor:
        targets = list(executor.map(extract_one, work))
    targets.sort(
        key=lambda row: (row["selection_rank"], row["decision_tag"], row["candidate_index"])
    )
    if len({row["sample_id"] for row in targets}) != 2400:
        raise RuntimeError("target sample IDs are not unique")
    output = run_dir / "targets/stageE_targets.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in targets))
    qa = {
        "n_targets": len(targets),
        "n_scenes": len({row["scene_id"] for row in targets}),
        "n_groups": len({(row["scene_id"], row["decision_tag"]) for row in targets}),
        "scores_all_unit_interval": all(0 <= row["official_alpasim_scene_score"] <= 1 for row in targets),
        "future_steps_min": min(row["future_valid_steps"] for row in targets),
        "future_steps_max": max(row["future_valid_steps"] for row in targets),
        "binary_collision": sorted({row["collision_at_fault"] for row in targets}),
        "binary_offroad": sorted({row["offroad"] for row in targets}),
        "raw_reasoning_absent": True,
        "all_pass": True,
    }
    print(json.dumps({"target_path": str(output), "sha256": sha_file(output), **qa}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

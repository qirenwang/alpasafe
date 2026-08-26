#!/usr/bin/env python
"""W0 Step 0 — read-only ASL probe.

Verifies, on a small sample of frozen scoring ASLs (FA2 + FN-A), that everything
the W0 extraction needs is really there: sha integrity, tick alignment of the
V2/V3 keyframes, EGO + traffic actor availability, and that our EGO chain
reproduces the frozen pipeline's future_ego_states_ego_t0 (FN-A label npz).

Read-only everywhere; writes only the report JSON inside the W0 study dir.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
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

FA2_DS = REPO / "artifacts/safeworld_t26g_fa2_prospective_full_l2_acquisition/20260730T190521Z/results/t26g_fa2_prospective_dataset.json"
FNA_DS = REPO / "artifacts/safeworld_t26g_fna_extension_acquisition/20260803T215243Z/results/t26g_fna_extension_dataset.json"

KF_ALL_US = {"2.1": 2_100_000, "3.2": 3_200_000, "4.2": 4_200_000, "6.3": 6_300_000, "6.4": 6_400_000}


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
    metadata: dict = {}
    rig_to_aabb = None
    poses: list[dict] = []
    requests: list[int] = []
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


def probe_one(tag: str, group: dict, cand: dict, label_npz: Path | None) -> dict:
    asl_path = Path(cand["rollout_asl_path"])
    report: dict = {"tag": tag, "group_id": group["group_id"], "candidate_index": cand["candidate_index"],
                    "official_score": cand["official_alpasim_scene_score"], "asl_path": str(asl_path)}

    sha = sha_file(asl_path)
    report["sha_match"] = sha == cand["rollout_asl_sha256"]
    if not report["sha_match"]:
        report["sha_expected"] = cand["rollout_asl_sha256"]
        report["sha_actual"] = sha
        return report

    parsed = asyncio.run(read_asl(asl_path))
    md = parsed["metadata"]
    decision = parsed["requests"][0] if parsed["requests"] else None
    report["metadata"] = md
    report["decision_ts_us"] = decision
    report["decision_matches_manifest"] = decision == group["decision_latest_timestamp_us"]
    report["force_gt_matches_manifest"] = md.get("force_gt_duration_us") == group.get("force_gt_duration_us")
    report["n_driver_requests"] = len(parsed["requests"])

    frames = sorted(parsed["poses"], key=lambda f: f["timestamp_us"])
    ts = np.array([f["timestamp_us"] for f in frames], dtype=np.int64)
    diffs = np.unique(np.diff(ts))
    report["n_actor_pose_frames"] = len(frames)
    report["frame_dt_unique_us"] = diffs.tolist()[:10]
    report["first_frame_rel_decision_us"] = int(ts[0] - decision)
    report["last_frame_rel_decision_us"] = int(ts[-1] - decision)

    post = [f for f in frames if f["timestamp_us"] >= decision]
    report["n_post_decision_frames"] = len(post)
    report["post_decision_has_ego_all"] = all("EGO" in f["actors"] for f in post)

    kf_report = {}
    for name, off in KF_ALL_US.items():
        target = decision + off
        exact = next((f for f in frames if f["timestamp_us"] == target), None)
        if exact is None:
            nearest = int(np.min(np.abs(ts - target)))
            kf_report[name] = {"exact": False, "nearest_off_us": nearest}
        else:
            actors = exact["actors"]
            ego = actors.get("EGO")
            n_other = 0
            if ego is not None:
                exy = np.array(ego["pos_xyz"][:2])
                for aid, ap in actors.items():
                    if aid == "EGO":
                        continue
                    if np.linalg.norm(np.array(ap["pos_xyz"][:2]) - exy) <= 50.0:
                        n_other += 1
            kf_report[name] = {"exact": True, "ego_present": ego is not None,
                              "n_actors": len(actors), "n_other_within_50m": n_other}
    report["keyframes"] = kf_report

    # EGO chain reproduction (reference convention) over post-decision frames
    anchor = parsed["first_return"][0]
    x0, y0 = anchor["pos_xyz"][:2]
    yaw0 = quat_wxyz_to_yaw(*anchor["quat_wxyz"])
    egoframes = [f for f in post if "EGO" in f["actors"]][:64]
    aabb_xy = np.asarray([f["actors"]["EGO"]["pos_xyz"][:2] for f in egoframes])
    yaws = np.asarray([quat_wxyz_to_yaw(*f["actors"]["EGO"]["quat_wxyz"]) for f in egoframes])
    rig_xy = aabb_xy_to_rig_xy(aabb_xy, yaws, parsed["rig_to_aabb"]["pos_xyz"][0])
    ego_t0 = local_xy_to_ego_t0(rig_xy, x0, y0, yaw0)
    report["ego_t0_n_steps"] = int(ego_t0.shape[0])
    report["ego_t0_final_xy"] = [round(float(v), 4) for v in ego_t0[-1]]

    if label_npz is not None and label_npz.exists():
        z = np.load(label_npz, allow_pickle=True)
        report["label_npz_keys"] = sorted(z.files)
        cand_key = None
        for k in z.files:
            if "future" in k:
                cand_key = k
        if cand_key is not None:
            arr = z[cand_key]
            report["label_future_shape"] = list(arr.shape)
            ci = cand["candidate_index"]
            if arr.ndim == 3 and arr.shape[0] >= ci + 1:
                ref = arr[ci][: ego_t0.shape[0]]
                report["ego_cross_check_max_abs_diff"] = float(np.max(np.abs(ref - ego_t0)))
    return report


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "probe"
    fa2 = json.loads(FA2_DS.read_text())
    fna = json.loads(FNA_DS.read_text())

    samples = []
    g0 = fa2["groups"][0]
    samples.append(("FA2_g0_c0", g0, g0["candidates"][0], None))
    # a hard-failure candidate (score==0) somewhere in FA2 → early-termination case
    hard = None
    for g in fa2["groups"]:
        for c in g["candidates"]:
            if c["official_alpasim_scene_score"] == 0.0:
                hard = ("FA2_hardfail", g, c, None)
                break
        if hard:
            break
    if hard:
        samples.append(hard)
    gn = fna["groups"][0]
    samples.append(("FNA_g0_c0", gn, gn["candidates"][0], Path(gn["label_npz_path"])))

    reports = [probe_one(*s) for s in samples]
    out = out_dir / "w0_step0_report.json"
    out.write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

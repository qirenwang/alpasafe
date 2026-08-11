#!/usr/bin/env python
"""W0 Step 1 — extract future-state targets u from the retained scoring ASLs.

Reads the two frozen dataset JSONs (read-only), sha-verifies every ASL against
its frozen manifest entry, extracts the 55-d future-state vector u at keyframes
{t+2.1, t+3.2, t+4.2, t+6.3} s (union of V2/V3, design amendment 001), and
appends one JSON line per candidate to a resumable .jsonl in the W0 study dir.
For FN-A candidates the ego rows are cross-checked against the frozen label npz.

Writes ONLY inside the W0 study dir. Any sha mismatch or structural defect is
recorded and the candidate skipped (counted); >0 sha mismatches ⇒ nonzero exit.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
import time
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
    rot2d,
)

W0 = Path(__file__).resolve().parent.parent
FA2_DS = REPO / "artifacts/safeworld_t26g_fa2_prospective_full_l2_acquisition/20260730T190521Z/results/t26g_fa2_prospective_dataset.json"
FNA_DS = REPO / "artifacts/safeworld_t26g_fna_extension_acquisition/20260803T215243Z/results/t26g_fna_extension_dataset.json"

KF_US = [2_100_000, 3_200_000, 4_200_000, 6_300_000]
KF_NAMES = ["2.1", "3.2", "4.2", "6.3"]
TICK_US = 100_000
N_NEIGH = 8
NEIGH_RADIUS_M = 50.0
EGO_XCHECK_TOL = 1e-4  # metres; frozen labels are float32


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
                "scene_id": session.scene_id,
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


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def build_u(parsed: dict, decision: int) -> dict:
    """u [4,55], termination flags, valid steps, ego_t0 64x2 (for cross-check)."""
    anchor = parsed["first_return"][0]
    x0, y0 = anchor["pos_xyz"][:2]
    yaw0 = quat_wxyz_to_yaw(*anchor["quat_wxyz"])
    rig_off = parsed["rig_to_aabb"]["pos_xyz"][0]

    by_ts = {f["timestamp_us"]: f for f in parsed["poses"]}
    post = sorted(
        [f for f in parsed["poses"] if f["timestamp_us"] >= decision and "EGO" in f["actors"]],
        key=lambda f: f["timestamp_us"],
    )[:64]
    if not post:
        raise RuntimeError("no post-decision EGO frames")
    # grid integrity: consecutive post frames must be exactly one tick apart
    pts = np.array([f["timestamp_us"] for f in post], dtype=np.int64)
    if not np.all(np.diff(pts) == TICK_US):
        raise RuntimeError("post-decision grid has gaps")
    valid_steps = len(post)

    aabb_xy = np.asarray([f["actors"]["EGO"]["pos_xyz"][:2] for f in post])
    yaws = np.asarray([quat_wxyz_to_yaw(*f["actors"]["EGO"]["quat_wxyz"]) for f in post])
    rig_xy = aabb_xy_to_rig_xy(aabb_xy, yaws, rig_off)
    ego_t0 = local_xy_to_ego_t0(rig_xy, x0, y0, yaw0)

    u = np.zeros((len(KF_US), 55), dtype=np.float32)
    term = np.zeros(len(KF_US), dtype=np.int8)
    for k, off in enumerate(KF_US):
        target = decision + off
        idx = off // TICK_US
        if idx >= valid_steps:  # terminated before keyframe: carry last + flag
            idx = valid_steps - 1
            term[k] = 1
            frame = post[idx]
        else:
            frame = post[idx]
            if frame["timestamp_us"] != target:
                raise RuntimeError(f"keyframe tick mismatch at {KF_NAMES[k]}")
        ego = frame["actors"]["EGO"]
        exy = np.array(ego["pos_xyz"][:2])
        eyaw = quat_wxyz_to_yaw(*ego["quat_wxyz"])
        # ego speed: finite difference over the preceding tick (world frame)
        prev = by_ts.get(frame["timestamp_us"] - TICK_US)
        if prev is not None and "EGO" in prev["actors"]:
            pxy = np.array(prev["actors"]["EGO"]["pos_xyz"][:2])
            espeed = float(np.linalg.norm(exy - pxy) / 0.1)
        else:
            espeed = 0.0
        dyaw = wrap(eyaw - yaw0)
        u[k, 0:2] = ego_t0[idx]
        u[k, 2] = np.cos(dyaw)
        u[k, 3] = np.sin(dyaw)
        u[k, 4] = espeed

        # neighbors: nearest N within radius, in ego-at-keyframe frame
        rows = []
        for aid, ap in frame["actors"].items():
            if aid == "EGO":
                continue
            nxy = np.array(ap["pos_xyz"][:2])
            d = float(np.linalg.norm(nxy - exy))
            if d > NEIGH_RADIUS_M:
                continue
            rel = (nxy - exy) @ rot2d(eyaw)
            nyaw = quat_wxyz_to_yaw(*ap["quat_wxyz"])
            ryaw = wrap(nyaw - eyaw)
            if prev is not None and aid in prev["actors"]:
                pn = np.array(prev["actors"][aid]["pos_xyz"][:2])
                nspeed = float(np.linalg.norm(nxy - pn) / 0.1)
            else:
                nspeed = 0.0
            rows.append((d, [float(rel[0]), float(rel[1]), float(np.cos(ryaw)), float(np.sin(ryaw)), nspeed, 1.0]))
        rows.sort(key=lambda r: r[0])
        for j, (_, feat) in enumerate(rows[:N_NEIGH]):
            u[k, 5 + 6 * j : 5 + 6 * (j + 1)] = feat
        u[k, 53] = float(term[k])
        u[k, 54] = min(len(rows), 30) / 10.0
    return {"u": u, "term": term, "valid_steps": valid_steps, "ego_t0": ego_t0}


def extract_candidate(job: dict) -> dict:
    cand, group, cohort, label_fut = job["cand"], job["group"], job["cohort"], job["label_fut"]
    asl_path = Path(cand["rollout_asl_path"])
    row: dict = {
        "cohort": cohort,
        "group_id": group["group_id"],
        "scene_id": group["scene_id"],
        "decision_tag": group["decision_tag"],
        "candidate_index": cand["candidate_index"],
        "sample_id": f"{group['group_id']}#c{cand['candidate_index']}",
        "official_score": float(cand["official_alpasim_scene_score"]),
    }
    sha = sha_file(asl_path)
    if sha != cand["rollout_asl_sha256"]:
        row["defect"] = "sha_mismatch"
        return row
    parsed = asyncio.run(read_asl(asl_path))
    decision = parsed["requests"][0] if parsed["requests"] else None
    if decision != group["decision_latest_timestamp_us"]:
        row["defect"] = "decision_mismatch"
        return row
    if parsed["metadata"]["scene_id"] != group["scene_id"]:
        row["defect"] = "scene_mismatch"
        return row
    try:
        built = build_u(parsed, decision)
    except RuntimeError as exc:
        row["defect"] = f"structural:{exc}"
        return row
    if label_fut is not None:
        ref = np.asarray(label_fut, dtype=np.float64)
        n = min(len(ref), built["ego_t0"].shape[0])
        diff = float(np.max(np.abs(ref[:n] - built["ego_t0"][:n])))
        row["ego_xcheck_max_abs_diff"] = diff
        if diff > EGO_XCHECK_TOL:
            row["defect"] = "ego_xcheck_failed"
            return row
    row["u"] = built["u"].tolist()
    row["terminated"] = built["term"].tolist()
    row["future_valid_steps"] = built["valid_steps"]
    return row


def main() -> int:
    out_path = W0 / "extracted/w0_u_targets.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        with out_path.open() as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["sample_id"])
                except Exception:
                    pass
    print(f"[w0-extract] resuming with {len(done)} rows already done", flush=True)

    jobs = []
    for cohort, ds_path in (("FA2", FA2_DS), ("FNA", FNA_DS)):
        ds = json.loads(ds_path.read_text())
        for group in ds["groups"]:
            label_by_ci = None
            if cohort == "FNA":
                z = np.load(group["label_npz_path"], allow_pickle=True)
                label_by_ci = z["future_ego_states_ego_t0"]
            for cand in group["candidates"]:
                sample_id = f"{group['group_id']}#c{cand['candidate_index']}"
                if sample_id in done:
                    continue
                jobs.append(
                    {
                        "cand": cand,
                        "group": group,
                        "cohort": cohort,
                        "label_fut": None if label_by_ci is None else label_by_ci[cand["candidate_index"]],
                    }
                )
    print(f"[w0-extract] {len(jobs)} candidates to extract", flush=True)

    lock = threading.Lock()
    counts = {"ok": 0, "defect": 0}
    t0 = time.time()
    with out_path.open("a") as out, ThreadPoolExecutor(max_workers=10) as pool:
        for row in pool.map(extract_candidate, jobs):
            with lock:
                out.write(json.dumps(row) + "\n")
                out.flush()
                counts["defect" if "defect" in row else "ok"] += 1
                n = counts["ok"] + counts["defect"]
                if n % 200 == 0:
                    rate = n / (time.time() - t0)
                    eta_h = (len(jobs) - n) / rate / 3600 if rate > 0 else -1
                    print(f"[w0-extract] {n}/{len(jobs)} ok={counts['ok']} defect={counts['defect']} eta={eta_h:.2f}h", flush=True)
    print(f"[w0-extract] DONE ok={counts['ok']} defect={counts['defect']} elapsed={(time.time()-t0)/3600:.2f}h", flush=True)
    return 0 if counts["defect"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

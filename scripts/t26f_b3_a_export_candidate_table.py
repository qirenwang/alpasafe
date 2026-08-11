#!/usr/bin/env python
"""Export the frozen T26F-B3-A joined candidate arrays as one flat
candidate-level table (2,400 rows) for offline inspection.

READ-ONLY derivation: every value is copied or arithmetically derived from
`datasets/stageF_joined_candidate_arrays.npz`, which the frozen Stage-F join
produced and the final manifest hashes. Nothing here re-runs a model,
re-reads a rollout, changes a metric, or alters any result; the frozen
analysis outputs remain the authoritative numbers.

Row identity: (scene_id, decision_tag, candidate_index) — 100 x 3 x 8.

Selection flags use the frozen selector semantics (deterministic argmax over
predicted official score within the K-prefix of the decision group), computed
per condition both per-seed at K8 and on the seed-mean score at K2/K5/K8.
ADE/FDE are masked over valid future timesteps exactly as the frozen future
metric does.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CONDITION_COLUMN = {
    "A_NORMAL": "A",
    "B_NORMAL": "B_normal",
    "B_ZERO": "B_zero",
    "B_WRONG_SCENE": "B_wrong",
}
K_VIEWS = {"K2": 2, "K5": 5, "K8": 8}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = args.run_dir / "datasets/stageF_joined_candidate_arrays.npz"
    z = np.load(source, allow_pickle=True)
    conditions = [str(c) for c in z["conditions"]]
    tags = [str(t) for t in z["tags"]]
    scenes = [str(s) for s in z["scene_ids"]]
    cand_idx = [int(c) for c in z["candidate_indices"]]
    n_seeds = z["predicted_score"].shape[1]

    score = z["predicted_score"]              # [cond, seed, scene, tag, cand]
    progress = z["predicted_progress"]
    coll_p = z["collision_probability"]
    off_p = z["offroad_probability"]
    pred_future = z["predicted_future"]       # [...,64,2]
    tgt_future = z["target_future"]           # [scene,tag,cand,64,2]
    tmask = z["target_future_mask"]           # [scene,tag,cand,64]
    tgt_score = z["target_score"]
    tgt_prog = z["target_progress"]
    tgt_coll = z["target_collision"]
    tgt_off = z["target_offroad"]
    plan = z["planned_trajectory"]
    ts = z["decision_timestamp_us"]

    # ---- displacement of every predicted future against the target future
    # (masked over valid timesteps; identical convention to the frozen metric)
    disp = np.linalg.norm(
        pred_future - tgt_future[None, None], axis=-1
    )                                          # [cond,seed,scene,tag,cand,64]
    mask = tmask[None, None].astype(np.float32)
    valid = np.maximum(mask.sum(axis=-1), 1.0)
    ade = (disp * mask).sum(axis=-1) / valid
    last = np.maximum(tmask.sum(axis=-1) - 1, 0)   # [scene,tag,cand]
    fde = np.take_along_axis(
        disp, last[None, None, ..., None], axis=-1
    ).squeeze(-1)

    # raw-plan baseline displacement (frozen reference)
    plan_disp = np.linalg.norm(plan - tgt_future, axis=-1)
    plan_ade = (plan_disp * tmask.astype(np.float32)).sum(axis=-1) / np.maximum(
        tmask.sum(axis=-1), 1)

    score_mean = score.mean(axis=1)            # [cond, scene, tag, cand]

    def selected(values: np.ndarray, k: int) -> np.ndarray:
        """Deterministic argmax over the first k candidates of each group."""
        pick = values[..., :k].argmax(axis=-1)
        flags = np.zeros(values.shape, dtype=bool)
        np.put_along_axis(flags, pick[..., None], True, axis=-1)
        return flags

    sel_mean = {  # [cond, scene, tag, cand] per K view, on the seed-mean score
        kv: selected(score_mean, k) for kv, k in K_VIEWS.items()
    }
    sel_seed_k8 = selected(score, 8)           # [cond, seed, scene, tag, cand]

    oracle = np.zeros(tgt_score.shape, dtype=bool)
    np.put_along_axis(oracle, tgt_score.argmax(axis=-1)[..., None], True, -1)

    header = [
        "scene_id", "decision_tag", "candidate_index", "decision_timestamp_us",
        "is_rank0", "is_oracle_best_K8",
        "official_target_score", "target_progress", "target_collision",
        "target_offroad", "raw_plan_future_ade_m",
    ]
    for cond in conditions:
        c = CONDITION_COLUMN[cond]
        for seed in range(n_seeds):
            header.append(f"{c}_pred_score_seed{seed}")
        header.append(f"{c}_pred_score_seedmean")
    for cond in conditions:
        c = CONDITION_COLUMN[cond]
        for kv in K_VIEWS:
            header.append(f"{c}_selected_{kv}_seedmean")
        for seed in range(n_seeds):
            header.append(f"{c}_selected_K8_seed{seed}")
    for cond in conditions:
        c = CONDITION_COLUMN[cond]
        header += [
            f"{c}_future_ade_m_seedmean", f"{c}_future_fde_m_seedmean",
            f"{c}_pred_progress_seedmean", f"{c}_pred_collision_prob_seedmean",
            f"{c}_pred_offroad_prob_seedmean",
        ]

    rows = []
    for si, scene in enumerate(scenes):
        for ti, tag in enumerate(tags):
            for ci, cand in enumerate(cand_idx):
                row = [
                    scene, tag, cand, int(ts[si, ti]),
                    int(cand == 0), int(oracle[si, ti, ci]),
                    float(tgt_score[si, ti, ci]), float(tgt_prog[si, ti, ci]),
                    float(tgt_coll[si, ti, ci]), float(tgt_off[si, ti, ci]),
                    float(plan_ade[si, ti, ci]),
                ]
                for ki in range(len(conditions)):
                    for seed in range(n_seeds):
                        row.append(float(score[ki, seed, si, ti, ci]))
                    row.append(float(score_mean[ki, si, ti, ci]))
                for ki in range(len(conditions)):
                    for kv in K_VIEWS:
                        row.append(int(sel_mean[kv][ki, si, ti, ci]))
                    for seed in range(n_seeds):
                        row.append(int(sel_seed_k8[ki, seed, si, ti, ci]))
                for ki in range(len(conditions)):
                    row += [
                        float(ade[ki, :, si, ti, ci].mean()),
                        float(fde[ki, :, si, ti, ci].mean()),
                        float(progress[ki, :, si, ti, ci].mean()),
                        float(coll_p[ki, :, si, ti, ci].mean()),
                        float(off_p[ki, :, si, ti, ci].mean()),
                    ]
                rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    meta = {
        "task": "t26f_b3_a_candidate_level_table",
        "derived_from": str(source),
        "source_sha256": sha_file(source),
        "read_only_derivation": True,
        "n_rows": len(rows),
        "n_columns": len(header),
        "row_identity": ["scene_id", "decision_tag", "candidate_index"],
        "conditions": conditions,
        "n_seeds": n_seeds,
        "selector_semantics": "deterministic argmax over predicted official "
                              "score within the K-prefix of the decision "
                              "group (frozen selector)",
        "ade_fde_semantics": "masked over valid future timesteps; FDE at the "
                             "last valid timestep (frozen future metric "
                             "convention)",
        "note": "convenience export for offline inspection; the frozen "
                "analysis outputs under results/ remain authoritative",
        "table_sha256": sha_file(args.out),
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True))
    print(f"wrote {len(rows)} rows x {len(header)} cols -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

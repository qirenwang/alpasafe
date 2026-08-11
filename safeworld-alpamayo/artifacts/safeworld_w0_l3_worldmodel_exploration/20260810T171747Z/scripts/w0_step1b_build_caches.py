#!/usr/bin/env python
"""W0 Step 1b — build L3 and trajectory caches for the 525-scene cohort.

Reads (read-only, sha-verified against the frozen manifests):
  - FA2: full_l2 safetensors (519 groups; contains keys L2 and L3) → take L3 [4096]
  - FN-A: l3_row safetensors (1,056 groups) → L3 [4096]
  - all 12,600 candidate trajectory JSONs (trajectory_xy 64x2, headings_rad 64)

Writes ONLY into the W0 study dir:
  extracted/w0_l3_cache.npz   {group_ids, l3 [1575,4096] fp32, cohort}
  extracted/w0_traj_cache.npz {sample_ids, group_ids, candidate_index, xy [N,64,2], heading [N,64], score [N]}
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO = Path("/home/qiren/alpasafe/safeworld-alpamayo")
W0 = Path(__file__).resolve().parent.parent
FA2_DS = REPO / "artifacts/safeworld_t26g_fa2_prospective_full_l2_acquisition/20260730T190521Z/results/t26g_fa2_prospective_dataset.json"
FNA_DS = REPO / "artifacts/safeworld_t26g_fna_extension_acquisition/20260803T215243Z/results/t26g_fna_extension_dataset.json"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    t0 = time.time()
    fa2 = json.loads(FA2_DS.read_text())
    fna = json.loads(FNA_DS.read_text())

    group_ids, l3_rows, cohorts = [], [], []
    sha_fail = 0
    for cohort, ds in (("FA2", fa2), ("FNA", fna)):
        for g in ds["groups"]:
            if cohort == "FA2":
                path, expect = Path(g["full_l2_path"]), g["full_l2_sha256"]
            else:
                path, expect = Path(g["l3_row_path"]), g["l3_row_sha256"]
            if sha_file(path) != expect:
                print(f"[l3-cache] SHA MISMATCH {path}", flush=True)
                sha_fail += 1
                continue
            with safe_open(str(path), framework="pt") as f:
                l3 = f.get_tensor("L3").to(torch.float32).numpy()
            assert l3.shape == (4096,)
            group_ids.append(g["group_id"])
            l3_rows.append(l3)
            cohorts.append(cohort)
        print(f"[l3-cache] {cohort} done, total {len(group_ids)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(
        W0 / "extracted/w0_l3_cache.npz",
        group_ids=np.array(group_ids),
        l3=np.stack(l3_rows).astype(np.float32),
        cohort=np.array(cohorts),
    )

    sample_ids, tgroup_ids, cand_idx, xys, heads, scores = [], [], [], [], [], []
    for ds in (fa2, fna):
        for g in ds["groups"]:
            for c in g["candidates"]:
                path = Path(c["candidate_path"])
                if sha_file(path) != c["candidate_sha256"]:
                    print(f"[traj-cache] SHA MISMATCH {path}", flush=True)
                    sha_fail += 1
                    continue
                doc = json.loads(path.read_text())
                cd = doc["candidate"]
                xy = np.asarray(cd["trajectory_xy"], dtype=np.float32)
                hd = np.asarray(cd["headings_rad"], dtype=np.float32)
                assert xy.shape == (64, 2) and hd.shape == (64,)
                sample_ids.append(f"{g['group_id']}#c{c['candidate_index']}")
                tgroup_ids.append(g["group_id"])
                cand_idx.append(c["candidate_index"])
                xys.append(xy)
                heads.append(hd)
                scores.append(float(c["official_alpasim_scene_score"]))
    np.savez_compressed(
        W0 / "extracted/w0_traj_cache.npz",
        sample_ids=np.array(sample_ids),
        group_ids=np.array(tgroup_ids),
        candidate_index=np.array(cand_idx, dtype=np.int32),
        xy=np.stack(xys),
        heading=np.stack(heads),
        score=np.array(scores, dtype=np.float64),
    )
    print(f"[caches] DONE l3={len(group_ids)} traj={len(sample_ids)} sha_fail={sha_fail} elapsed={(time.time()-t0)/60:.1f}min", flush=True)
    return 0 if sha_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

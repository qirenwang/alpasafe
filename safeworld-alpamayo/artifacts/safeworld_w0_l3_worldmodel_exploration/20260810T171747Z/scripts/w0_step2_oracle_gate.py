#!/usr/bin/env python
"""W0 Step 2 — oracle gate.

Question: if the score head could see the TRUE future scene states u at the
keyframes (perfect imagination), does selection improve over the matched
amortized control that sees only [L3, trajectory]?

Arms (byte-identical nets, same seeds/folds; only the u channel differs):
  ORACLE_V2 : input [L3proj, traj, u@{3.2,6.3}]          (true u)
  ORACLE_V3 : input [L3proj, traj, u@{2.1,4.2,6.3}]      (true u)
  AMORT     : same as ORACLE_V2 with u slots = 0 after standardization
              (mean-imputation semantics = uninformative in-distribution)

Endpoints: K8 scene-macro regret (primary), tie-aware top-1 (eps 1e-6),
group→scene→seed aggregation, paired-scene percentile bootstrap (n=10,000,
seed 20260810) on ORACLE−AMORT diffs. Folds: FN1 scene folds, read-only reuse.

Exploratory study; no frozen contract is touched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

W0 = Path(__file__).resolve().parent.parent
REPO = Path("/home/qiren/alpasafe/safeworld-alpamayo")
FOLDS = REPO / "artifacts/safeworld_t26g_fn1_l3_pathway_consolidation_experiment/20260809T173120Z/folds/t26g_fn1_scene_folds.json"

SEEDS = [0, 1, 2]
EPS = 1e-6
BOOT_N = 10_000
BOOT_SEED = 20260810
KF_IDX = {"V2": [1, 3], "V3": [0, 2, 3]}  # rows of u [4,55] = {2.1, 3.2, 4.2, 6.3}


def load_data():
    u_by_sid: dict[str, np.ndarray] = {}
    for line in (W0 / "extracted/w0_u_targets.jsonl").open():
        row = json.loads(line)
        if "defect" in row:
            continue
        u = np.asarray(row["u"], dtype=np.float32)
        # amendment 002: flag fires only on TRUE early termination (<62 steps);
        # the systematic 62-vs-64 cohort length difference is not termination.
        u[:, 53] = 1.0 if row["future_valid_steps"] < 62 else 0.0
        u_by_sid[row["sample_id"]] = u

    l3z = np.load(W0 / "extracted/w0_l3_cache.npz", allow_pickle=True)
    l3_by_gid = {g: v for g, v in zip(l3z["group_ids"], l3z["l3"])}

    tz = np.load(W0 / "extracted/w0_traj_cache.npz", allow_pickle=True)
    folds = json.loads(FOLDS.read_text())["fold_of_scene"]

    rows = []
    for i, sid in enumerate(tz["sample_ids"]):
        sid = str(sid)
        gid = str(tz["group_ids"][i])
        scene = gid.split("@")[0]
        if sid not in u_by_sid or gid not in l3_by_gid:
            continue
        xy = tz["xy"][i]
        hd = tz["heading"][i]
        traj = np.concatenate(
            [xy[::4].reshape(-1), xy[-1], np.cos(hd[::4]), np.sin(hd[::4])]
        ).astype(np.float32)
        rows.append(
            {
                "sid": sid,
                "gid": gid,
                "scene": scene,
                "fold": int(folds[scene]),
                "ci": int(tz["candidate_index"][i]),
                "traj": traj,
                "u": u_by_sid[sid],
                "l3": l3_by_gid[gid],
                "score": float(tz["score"][i]),
            }
        )
    return rows


class ScoreNet(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.l3_proj = nn.Sequential(nn.Linear(4096, 128), nn.GELU())
        self.mlp = nn.Sequential(
            nn.Linear(128 + feat_dim, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, l3, feats):
        return self.mlp(torch.cat([self.l3_proj(l3), feats], dim=-1)).squeeze(-1)


def fit_predict(rows, fold, seed, kf_rows, zero_u):
    train = [r for r in rows if r["fold"] != fold]
    test = [r for r in rows if r["fold"] == fold]

    def feats(rs):
        t = np.stack([r["traj"] for r in rs])
        u = np.stack([r["u"][kf_rows].reshape(-1) for r in rs])
        return t, u

    tr_t, tr_u = feats(train)
    te_t, te_u = feats(test)
    tr_l3 = np.stack([r["l3"] for r in train])
    te_l3 = np.stack([r["l3"] for r in test])

    def stdz(a, m, s):
        return (a - m) / s

    mt, st = tr_t.mean(0), tr_t.std(0) + 1e-6
    mu, su = tr_u.mean(0), tr_u.std(0) + 1e-6
    ml, sl = tr_l3.mean(0), tr_l3.std(0) + 1e-6
    tr_t, te_t = stdz(tr_t, mt, st), stdz(te_t, mt, st)
    tr_u, te_u = stdz(tr_u, mu, su), stdz(te_u, mu, su)
    tr_l3, te_l3 = stdz(tr_l3, ml, sl), stdz(te_l3, ml, sl)
    if zero_u:
        tr_u = np.zeros_like(tr_u)
        te_u = np.zeros_like(te_u)

    tr_f = np.concatenate([tr_t, tr_u], 1)
    te_f = np.concatenate([te_t, te_u], 1)
    y = np.array([r["score"] for r in train], dtype=np.float32)

    torch.manual_seed(seed)
    np.random.seed(seed)
    net = ScoreNet(tr_f.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    loss_fn = nn.SmoothL1Loss(beta=0.1)
    X_l3 = torch.tensor(tr_l3)
    X_f = torch.tensor(tr_f)
    Y = torch.tensor(y)
    n = len(train)
    for _ in range(60):
        perm = torch.randperm(n)
        for b in range(0, n, 512):
            idx = perm[b : b + 512]
            opt.zero_grad()
            loss = loss_fn(net(X_l3[idx], X_f[idx]), Y[idx])
            loss.backward()
            opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(te_l3), torch.tensor(te_f)).numpy()
    return {r["sid"]: float(p) for r, p in zip(test, pred)}


def endpoints(rows, preds):
    by_group: dict[str, list] = {}
    for r in rows:
        by_group.setdefault(r["gid"], []).append(r)
    scene_vals: dict[str, list] = {}
    for gid, rs in by_group.items():
        rs = sorted(rs, key=lambda r: r["ci"])
        scores = np.array([r["score"] for r in rs])
        p = np.array([preds[r["sid"]] for r in rs])
        sel = int(np.argmax(p))
        best = float(scores.max())
        regret = best - float(scores[sel])
        top1 = float(scores[sel] >= best - EPS)
        scene_vals.setdefault(rs[0]["scene"], []).append((regret, top1))
    out = {}
    for scene, vals in scene_vals.items():
        arr = np.array(vals)
        out[scene] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
    return out


def main() -> int:
    t0 = time.time()
    rows = load_data()
    scenes = sorted({r["scene"] for r in rows})
    print(f"[gate] rows={len(rows)} scenes={len(scenes)} ({time.time()-t0:.0f}s load)", flush=True)

    arms = {
        "ORACLE_V2": {"kf": KF_IDX["V2"], "zero": False},
        "ORACLE_V3": {"kf": KF_IDX["V3"], "zero": False},
        "AMORT": {"kf": KF_IDX["V2"], "zero": True},
    }
    per_arm_scene: dict[str, dict[str, dict]] = {a: {} for a in arms}
    for arm, cfg in arms.items():
        for seed in SEEDS:
            preds: dict[str, float] = {}
            for fold in range(5):
                preds.update(fit_predict(rows, fold, seed, cfg["kf"], cfg["zero"]))
            sv = endpoints(rows, preds)
            for scene, (reg, top) in sv.items():
                per_arm_scene[arm].setdefault(scene, []).append((reg, top))
            print(f"[gate] {arm} seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    summary: dict = {"n_rows": len(rows), "n_scenes": len(scenes), "arms": {}, "diffs": {}}
    scene_mean: dict[str, dict[str, np.ndarray]] = {}
    for arm in arms:
        m = {s: np.array(v).mean(0) for s, v in per_arm_scene[arm].items()}
        scene_mean[arm] = m
        arr = np.stack([m[s] for s in scenes])
        summary["arms"][arm] = {"K8_regret": float(arr[:, 0].mean()), "K8_top1": float(arr[:, 1].mean())}

    rng = np.random.default_rng(BOOT_SEED)
    for arm in ("ORACLE_V2", "ORACLE_V3"):
        d = np.stack([scene_mean[arm][s] - scene_mean["AMORT"][s] for s in scenes])
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean(0) for _ in range(BOOT_N)])
        summary["diffs"][f"{arm}-AMORT"] = {
            "regret_mean": float(d[:, 0].mean()),
            "regret_ci": [float(np.percentile(boots[:, 0], 2.5)), float(np.percentile(boots[:, 0], 97.5))],
            "top1_mean": float(d[:, 1].mean()),
            "top1_ci": [float(np.percentile(boots[:, 1], 2.5)), float(np.percentile(boots[:, 1], 97.5))],
        }

    out = W0 / "results/w0_step2_oracle_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[gate] DONE in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

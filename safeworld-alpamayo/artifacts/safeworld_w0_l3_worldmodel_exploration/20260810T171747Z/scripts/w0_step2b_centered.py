#!/usr/bin/env python
"""W0 Step 2b (amendment 003) — centered-objective diagnostic arms.

Separates two explanations of the Step-2 oracle result (true u HURT selection):
  (i)  u carries no incremental within-group ranking information, vs
  (ii) the absolute-score regression wastes it on level fitting.

Same nets/folds/seeds as Step 2; the ONLY change is the regression target:
score − group mean (argmax-equivalent within a group; the T26G-C group-mean /
candidate-residual lesson). Arms: ORACLE_V2_C, AMORT_C.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from w0_step2_oracle_gate import (  # noqa: E402
    BOOT_N, EPS, KF_IDX, SEEDS, W0, ScoreNet, endpoints, load_data,
)

BOOT_SEED = 20260810


def fit_predict_centered(rows, fold, seed, kf_rows, zero_u):
    train = [r for r in rows if r["fold"] != fold]
    test = [r for r in rows if r["fold"] == fold]

    gmean: dict[str, float] = {}
    gsum: dict[str, list] = {}
    for r in train:
        gsum.setdefault(r["gid"], []).append(r["score"])
    gmean = {g: float(np.mean(v)) for g, v in gsum.items()}

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
    y = np.array([r["score"] - gmean[r["gid"]] for r in train], dtype=np.float32)

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


def main() -> int:
    t0 = time.time()
    rows = load_data()
    scenes = sorted({r["scene"] for r in rows})
    print(f"[gate2b] rows={len(rows)} scenes={len(scenes)}", flush=True)

    arms = {
        "ORACLE_V2_C": {"kf": KF_IDX["V2"], "zero": False},
        "AMORT_C": {"kf": KF_IDX["V2"], "zero": True},
    }
    per_arm_scene: dict[str, dict[str, list]] = {a: {} for a in arms}
    for arm, cfg in arms.items():
        for seed in SEEDS:
            preds: dict[str, float] = {}
            for fold in range(5):
                preds.update(fit_predict_centered(rows, fold, seed, cfg["kf"], cfg["zero"]))
            sv = endpoints(rows, preds)
            for scene, (reg, top) in sv.items():
                per_arm_scene[arm].setdefault(scene, []).append((reg, top))
            print(f"[gate2b] {arm} seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    summary: dict = {"arms": {}, "diffs": {}}
    scene_mean: dict[str, dict[str, np.ndarray]] = {}
    for arm in arms:
        m = {s: np.array(v).mean(0) for s, v in per_arm_scene[arm].items()}
        scene_mean[arm] = m
        arr = np.stack([m[s] for s in scenes])
        summary["arms"][arm] = {"K8_regret": float(arr[:, 0].mean()), "K8_top1": float(arr[:, 1].mean())}

    rng = np.random.default_rng(BOOT_SEED)
    d = np.stack([scene_mean["ORACLE_V2_C"][s] - scene_mean["AMORT_C"][s] for s in scenes])
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean(0) for _ in range(BOOT_N)])
    summary["diffs"]["ORACLE_V2_C-AMORT_C"] = {
        "regret_mean": float(d[:, 0].mean()),
        "regret_ci": [float(np.percentile(boots[:, 0], 2.5)), float(np.percentile(boots[:, 0], 97.5))],
        "top1_mean": float(d[:, 1].mean()),
        "top1_ci": [float(np.percentile(boots[:, 1], 2.5)), float(np.percentile(boots[:, 1], 97.5))],
    }

    out = W0 / "results/w0_step2b_centered.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[gate2b] DONE in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def minimal_u(rows):
    """Amendment 004: shrink u to 4 outcome scalars per keyframe.

    Per keyframe: ego_x, ego_y, min neighbor distance (50 m cap when absent),
    terminated flag → [n_kf*4] instead of [n_kf*55].
    """
    for r in rows:
        u = r["u"]  # [4,55]
        mins = []
        for k in range(u.shape[0]):
            neigh = u[k, 5:53].reshape(8, 6)
            present = neigh[:, 5] > 0.5
            if present.any():
                d = np.linalg.norm(neigh[present, 0:2], axis=1).min()
            else:
                d = 50.0
            mins.append([u[k, 0], u[k, 1], d, u[k, 53]])
        r["u_min"] = np.asarray(mins, dtype=np.float32)
    return rows


def main_minimal() -> int:
    t0 = time.time()
    rows = minimal_u(load_data())
    for r in rows:
        r["u"] = r["u_min"]  # swap in the 4-d-per-kf u
    scenes = sorted({r["scene"] for r in rows})
    print(f"[gate2c] rows={len(rows)} scenes={len(scenes)}", flush=True)
    per_scene: dict[str, list] = {}
    for seed in SEEDS:
        preds: dict[str, float] = {}
        for fold in range(5):
            preds.update(fit_predict_centered(rows, fold, seed, KF_IDX["V2"], False))
        for scene, val in endpoints(rows, preds).items():
            per_scene.setdefault(scene, []).append(val)
        print(f"[gate2c] ORACLE_MIN_C seed {seed} done ({time.time()-t0:.0f}s)", flush=True)
    m = {s: np.array(v).mean(0) for s, v in per_scene.items()}
    arr = np.stack([m[s] for s in scenes])
    prev = json.loads((W0 / "results/w0_step2b_centered.json").read_text())
    amort = prev["arms"]["AMORT_C"]
    summary = {"ORACLE_MIN_C": {"K8_regret": float(arr[:, 0].mean()), "K8_top1": float(arr[:, 1].mean())},
               "AMORT_C_ref": amort}
    out = W0 / "results/w0_step2c_minimal.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0

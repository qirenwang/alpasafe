"""T26E-B pilot training driver (frozen contract; see configs/ in the run dir).

Modes:
    full      -- one fixed-seed training run (train=192, val=48)
    smoke     -- bounded one-TRAIN-group overfit smoke (no val/test data)
    baselines -- validation baselines (train-mean constant, rank-0, oracle)

The ONLY model input feature is ``input.planned_trajectory``. Everything else in
a record is a target (losses/eval) or metadata (joins, grouping, reporting,
selector tie-breaks). Sealed test labels are never opened (loader guard).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from safeworld.t26e.loader import T26EOfficialScoreDataset
from safeworld.t26e.model import SafeWorldCandidatePilot
from safeworld.t26e.selector_adapter import CandidateOfficialScorePrediction, select_within_group

HORIZON = 64
K_VIEWS = {"K2": 2, "K5": 5, "K8": 8}
HORIZON_SUMMARY_STEPS = (9, 19, 39, 63)  # 1s, 2s, 4s, 6.4s at 10 Hz
LR, WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS, PATIENCE = 3e-4, 1e-4, 8, 100, 15
HUBER = nn.SmoothL1Loss(beta=1.0)


# --------------------------------------------------------------------------- data
def tensorize(split: str) -> dict[str, Any]:
    """Build tensors. Feature source is exactly ``input.planned_trajectory``."""
    ds = T26EOfficialScoreDataset(split)
    n = len(ds)
    plan = torch.empty(n, HORIZON, 2, dtype=torch.float32)
    future = torch.zeros(n, HORIZON, 2, dtype=torch.float32)
    tmask = torch.zeros(n, HORIZON, dtype=torch.bool)
    score = torch.empty(n, dtype=torch.float32)
    meta = []
    for i, rec in enumerate(ds.records):
        plan[i] = torch.tensor(rec.input["planned_trajectory"], dtype=torch.float32)
        fct = rec.targets["future_consequence_target"]
        fut = torch.tensor(fct["future_ego_states_ego_t0"], dtype=torch.float32)
        t_r = len(fct["timestamps_us"])
        assert fut.shape == (t_r, 2) and 0 < t_r <= HORIZON
        future[i, :t_r] = fut
        tmask[i, :t_r] = True
        score[i] = float(rec.targets["official_score"]["official_alpasim_scene_score"])
        meta.append(
            {
                "sample_id": rec.sample_id,
                "decision_group_id": rec.decision_group_id,
                "candidate_id": rec.candidate_id,
                "candidate_index": rec.candidate_index,
            }
        )
    return {
        "plan": plan,
        "future": future,
        "tmask": tmask,
        "score": score,
        "meta": meta,
        "feature_source": "input.planned_trajectory",
    }


def fit_train_plan_stats(train_plan: torch.Tensor) -> dict[str, torch.Tensor]:
    """Train-only standardization statistics (T26C convention)."""
    return {
        "plan_mean": train_plan.mean(dim=(0, 1)),
        "plan_std": train_plan.std(dim=(0, 1)).clamp_min(1e-6),
    }


def standardize(plan: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    return (plan - stats["plan_mean"]) / stats["plan_std"]


# --------------------------------------------------------------------------- losses
def masked_future_huber(pred: torch.Tensor, target: torch.Tensor, tmask: torch.Tensor):
    m = tmask.unsqueeze(-1).expand_as(pred)
    return HUBER(pred[m], target[m])


def compute_losses(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
    l_future = masked_future_huber(
        out["predicted_future_ego_states_ego_t0"], batch["future"], batch["tmask"]
    )
    l_score = HUBER(out["predicted_official_alpasim_scene_score"], batch["score"])
    return {"future": l_future, "score": l_score, "total": 1.0 * l_future + 1.0 * l_score}


# --------------------------------------------------------------------------- metrics
def spearman(pred: np.ndarray, target: np.ndarray) -> float:
    """Spearman rank correlation with average ranks; 0.0 if either side is constant."""

    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x))
        r[order] = np.arange(1, len(x) + 1)
        for v in np.unique(x):
            mask = x == v
            r[mask] = r[mask].mean()
        return r

    if np.std(pred) == 0.0 or np.std(target) == 0.0:
        return 0.0
    rp, rt = ranks(pred), ranks(target)
    return float(np.corrcoef(rp, rt)[0, 1])


def candidate_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "spearman": spearman(pred, target),
        "prediction_min": float(pred.min()),
        "prediction_max": float(pred.max()),
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std()),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
    }


def future_metrics(pred: torch.Tensor, target: torch.Tensor, tmask: torch.Tensor):
    d = torch.linalg.norm(pred - target, dim=-1)  # [N,64] meters
    tm = tmask.float()
    ade = float((d * tm).sum(dim=1).div(tm.sum(dim=1)).mean())
    last = tm.sum(dim=1).long() - 1
    fde = float(d[torch.arange(d.shape[0]), last].mean())
    per_h = {}
    for step in HORIZON_SUMMARY_STEPS:
        valid = tmask[:, step]
        per_h[f"t{(step + 1) / 10.0:.1f}s"] = {
            "mean_displacement_m": float(d[valid, step].mean()) if valid.any() else None,
            "valid_fraction": float(valid.float().mean()),
        }
    return {
        "ade_m": ade,
        "fde_m": fde,
        "per_horizon": per_h,
        "valid_timestep_coverage": float(tm.mean()),
    }


def group_metrics(
    pred: np.ndarray, target: np.ndarray, meta: list[dict[str, Any]], k: int
) -> dict[str, Any]:
    """Grouped selector evaluation on the candidate_index < k prefix view."""
    groups: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        if m["candidate_index"] < k:
            groups.setdefault(m["decision_group_id"], []).append(i)
    per_group, regrets = [], []
    top1 = pair_correct = pair_total = pred_ties = off_ties = unique_sel = 0
    for gid in sorted(groups):
        idx = sorted(groups[gid], key=lambda i: meta[i]["candidate_index"])
        preds = [
            CandidateOfficialScorePrediction(
                sample_id=meta[i]["sample_id"],
                candidate_id=meta[i]["candidate_id"],
                final_reward=float(pred[i]),
                candidate_rank=meta[i]["candidate_index"],
            )
            for i in idx
        ]
        sel = select_within_group(preds)
        sel_i = next(i for i in idx if meta[i]["sample_id"] == sel.sample_id)
        off = target[idx]
        best_off = float(off.max())
        best_i = idx[int(np.argmax(off))]
        regret = best_off - float(target[sel_i])
        regrets.append(regret)
        is_top1 = math.isclose(float(target[sel_i]), best_off, abs_tol=0.0)
        top1 += is_top1
        p_tie = sum(math.isclose(p.final_reward, max(x.final_reward for x in preds)) for p in preds) > 1
        o_tie = sum(math.isclose(float(v), best_off) for v in off) > 1
        pred_ties += p_tie
        off_ties += o_tie
        unique_sel += not p_tie
        pc = pt = 0
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                ia, ib = idx[a], idx[b]
                if target[ia] == target[ib]:
                    continue
                pt += 1
                pc += bool((pred[ia] - pred[ib]) * (target[ia] - target[ib]) > 0)
        pair_correct += pc
        pair_total += pt
        per_group.append(
            {
                "decision_group_id": gid,
                "selected_candidate_id": meta[sel_i]["candidate_id"],
                "selected_candidate_index": meta[sel_i]["candidate_index"],
                "official_best_candidate_id": meta[best_i]["candidate_id"],
                "official_best_candidate_index": meta[best_i]["candidate_index"],
                "selected_official_score": float(target[sel_i]),
                "best_official_score": best_off,
                "selected_score_regret": regret,
                "top1_correct": bool(is_top1),
                "pairwise_correct": int(pc),
                "pairwise_total": int(pt),
                "predicted_score_tied": bool(p_tie),
                "official_score_tied": bool(o_tie),
            }
        )
    n_g = len(groups)
    return {
        "k": k,
        "n_groups": n_g,
        "top1_selection_accuracy": top1 / n_g,
        "pairwise_ranking_accuracy": pair_correct / pair_total if pair_total else None,
        "mean_selected_score_regret": float(np.mean(regrets)),
        "median_selected_score_regret": float(np.median(regrets)),
        "max_selected_score_regret": float(np.max(regrets)),
        "mean_oracle_gap": float(np.mean(regrets)),
        "uniquely_selected_group_count": unique_sel,
        "predicted_tie_count": pred_ties,
        "official_tie_count": off_ties,
        "per_group": per_group,
    }


# --------------------------------------------------------------------------- training
def checkpoint_key(entry: dict[str, float]) -> tuple:
    """Ordered comparison key: lower is better; earlier epoch breaks final ties."""
    return (
        entry["val_score_mae"],
        entry["val_k8_mean_regret"],
        entry["val_future_ade"],
        entry["val_total_loss"],
        entry["epoch"],
    )


@torch.no_grad()
def evaluate(model, data, stats) -> dict[str, Any]:
    model.eval()
    out = model(standardize(data["plan"], stats), data["plan"])
    losses = compute_losses(out, data)
    pred = out["predicted_official_alpasim_scene_score"].numpy()
    target = data["score"].numpy()
    return {
        "losses": {k: float(v) for k, v in losses.items()},
        "candidate": candidate_metrics(pred, target),
        "future": future_metrics(
            out["predicted_future_ego_states_ego_t0"], data["future"], data["tmask"]
        ),
        "groups": {name: group_metrics(pred, target, data["meta"], k)
                   for name, k in K_VIEWS.items()},
        "predictions": pred.tolist(),
    }


def param_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().numpy().tobytes())
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_one_seed(seed: int, run_dir: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, val = tensorize("train"), tensorize("val")
    stats = fit_train_plan_stats(train["plan"])
    model = SafeWorldCandidatePilot()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(seed)

    ckpt_dir = run_dir / f"checkpoints/seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_f = run_dir / f"metrics/seed_{seed}_epoch_history.jsonl"
    history_f.write_text("")

    best: dict[str, Any] | None = None
    bad = 0
    n = train["plan"].shape[0]
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        ep_losses = []
        for s in range(0, n, BATCH_SIZE):
            idx = perm[s : s + BATCH_SIZE]
            batch = {k: train[k][idx] for k in ("plan", "future", "tmask", "score")}
            opt.zero_grad()
            out = model(standardize(batch["plan"], stats), batch["plan"])
            losses = compute_losses(out, batch)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"non-finite train loss at epoch {epoch}")
            losses["total"].backward()
            opt.step()
            ep_losses.append({k: float(v) for k, v in losses.items()})
        va = evaluate(model, val, stats)
        entry = {
            "epoch": epoch,
            "val_score_mae": va["candidate"]["mae"],
            "val_k8_mean_regret": va["groups"]["K8"]["mean_selected_score_regret"],
            "val_future_ade": va["future"]["ade_m"],
            "val_total_loss": va["losses"]["total"],
        }
        with open(history_f, "a") as f:
            f.write(
                json.dumps(
                    {
                        **entry,
                        "train_loss_mean": {
                            k: float(np.mean([x[k] for x in ep_losses]))
                            for k in ("future", "score", "total")
                        },
                        "val_spearman": va["candidate"]["spearman"],
                        "val_k8_top1": va["groups"]["K8"]["top1_selection_accuracy"],
                    }
                )
                + "\n"
            )
        if best is None or checkpoint_key(entry) < checkpoint_key(best["entry"]):
            best = {"entry": entry, "epoch": epoch}
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                    "stats": {k: v.tolist() for k, v in stats.items()},
                    "entry": entry,
                },
                ckpt_dir / "best.pt",
            )
            bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break

    ckpt = torch.load(ckpt_dir / "best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    final_val = evaluate(model, val, stats)
    final_train = evaluate(model, train, stats)

    for split, data, ev in (("train", train, final_train), ("validation", val, final_val)):
        with open(run_dir / f"predictions/seed_{seed}_{split}_predictions.jsonl", "w") as f:
            for m, p, t in zip(data["meta"], ev["predictions"], data["score"].tolist()):
                f.write(
                    json.dumps(
                        {
                            **m,
                            "predicted_official_alpasim_scene_score": p,
                            "official_alpasim_scene_score": t,
                        }
                    )
                    + "\n"
                )
    (run_dir / f"metrics/seed_{seed}_candidate_metrics.json").write_text(
        json.dumps(
            {"train": final_train["candidate"], "val": final_val["candidate"]}, indent=1
        )
    )
    groups_slim = {
        k: {kk: vv for kk, vv in v.items() if kk != "per_group"}
        for k, v in final_val["groups"].items()
    }
    (run_dir / f"metrics/seed_{seed}_group_metrics.json").write_text(
        json.dumps({"val": final_val["groups"], "val_summary": groups_slim}, indent=1)
    )
    (run_dir / f"metrics/seed_{seed}_future_metrics.json").write_text(
        json.dumps({"train": final_train["future"], "val": final_val["future"]}, indent=1)
    )
    manifest = {
        "seed": seed,
        "selected_epoch": ckpt["epoch"],
        "early_stopping": {"patience": PATIENCE, "stopped_after_epoch": ckpt["epoch"] + bad},
        "checkpoint_path": str(ckpt_dir / "best.pt"),
        "checkpoint_sha256": file_sha256(ckpt_dir / "best.pt"),
        "parameter_hash": param_hash(model),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "val_score_mae": final_val["candidate"]["mae"],
        "val_score_rmse": final_val["candidate"]["rmse"],
        "val_spearman": final_val["candidate"]["spearman"],
        "val_future_ade": final_val["future"]["ade_m"],
        "val_future_fde": final_val["future"]["fde_m"],
        "val_k8": groups_slim["K8"],
        "val_k5": groups_slim["K5"],
        "val_k2": groups_slim["K2"],
        "preprocessing_stats": {k: v.tolist() for k, v in stats.items()},
    }
    (run_dir / f"manifests/seed_{seed}_checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=1)
    )
    return manifest


# --------------------------------------------------------------------------- baselines
def run_baselines(run_dir: Path) -> dict[str, Any]:
    train, val = tensorize("train"), tensorize("val")
    target = val["score"].numpy()
    const = float(train["score"].mean())
    const_pred = np.full_like(target, const)
    out = {
        "A_train_mean_constant": {
            "constant": const,
            "fit_split": "train",
            "candidate": candidate_metrics(const_pred, target),
            "note": "spearman defined as 0.0 for constant predictions",
            "groups": {
                name: group_metrics(const_pred, target, val["meta"], k)
                for name, k in K_VIEWS.items()
            },
        },
        "B_alpamayo_rank0": {},
        "C_val_oracle": {},
    }
    # B: rank-0 == constant predictions + selector tie-break to candidate_rank 0
    for name, k in K_VIEWS.items():
        gm = group_metrics(const_pred, target, val["meta"], k)
        assert all(g["selected_candidate_index"] == 0 for g in gm["per_group"])
        out["B_alpamayo_rank0"][name] = {kk: vv for kk, vv in gm.items() if kk != "per_group"}
        out["B_alpamayo_rank0"][name]["selection_rule"] = "candidate_index=0"
    # C: oracle upper bound — predict the official score itself (reporting only)
    for name, k in K_VIEWS.items():
        gm = group_metrics(target, target, val["meta"], k)
        out["C_val_oracle"][name] = {kk: vv for kk, vv in gm.items() if kk != "per_group"}
        out["C_val_oracle"][name]["note"] = "upper bound, not deployable"
    (run_dir / "baselines/t26e_b_validation_baselines.json").write_text(
        json.dumps(out, indent=1)
    )
    return out


# --------------------------------------------------------------------------- smoke
def run_smoke(run_dir: Path, steps: int = 200) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    train = tensorize("train")
    gid = train["meta"][0]["decision_group_id"]
    idx = [i for i, m in enumerate(train["meta"]) if m["decision_group_id"] == gid]
    assert len(idx) == 8
    batch = {k: train[k][idx] for k in ("plan", "future", "tmask", "score")}
    stats = fit_train_plan_stats(train["plan"])
    model = SafeWorldCandidatePilot()
    hash_before = param_hash(model)
    w_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    with torch.no_grad():
        out0 = model(standardize(batch["plan"], stats), batch["plan"])
        pred_before = out0["predicted_official_alpasim_scene_score"].tolist()
    trace, grad_norms = [], {}
    for step in range(steps):
        model.train()
        opt.zero_grad()
        out = model(standardize(batch["plan"], stats), batch["plan"])
        losses = compute_losses(out, batch)
        losses["total"].backward()
        if step == 0:
            grad_norms = {
                "w_theta": float(
                    sum(
                        p.grad.norm() ** 2
                        for p in model.w_theta.parameters()
                        if p.grad is not None
                    )
                    ** 0.5
                ),
                "g_phi": float(
                    sum(
                        p.grad.norm() ** 2
                        for p in model.g_phi.parameters()
                        if p.grad is not None
                    )
                    ** 0.5
                ),
            }
        opt.step()
        trace.append({k: float(v) for k, v in losses.items()})
    hash_after = param_hash(model)
    w_theta_updated = any(
        not torch.equal(w_before[n], p.detach())
        for n, p in model.named_parameters()
        if n.startswith("w_theta")
    )
    g_phi_updated = any(
        not torch.equal(w_before[n], p.detach())
        for n, p in model.named_parameters()
        if n.startswith("g_phi")
    )
    with torch.no_grad():
        out1 = model(standardize(batch["plan"], stats), batch["plan"])
        pred_after = out1["predicted_official_alpasim_scene_score"].tolist()

    ckpt_path = run_dir / "smokes/smoke_checkpoint.pt"
    torch.save({"model_state": model.state_dict()}, ckpt_path)
    model2 = SafeWorldCandidatePilot()
    model2.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state"])
    with torch.no_grad():
        pred_reload = (
            model2(standardize(batch["plan"], stats), batch["plan"])[
                "predicted_official_alpasim_scene_score"
            ].tolist()
        )
    result = {
        "group": gid,
        "sample_ids": [train["meta"][i]["sample_id"] for i in idx],
        "steps": steps,
        "seed": 0,
        "loss_first": trace[0],
        "loss_last": trace[-1],
        "all_losses_finite": all(math.isfinite(t["total"]) for t in trace),
        "score_loss_decreased": trace[-1]["score"] < trace[0]["score"],
        "total_loss_decreased": trace[-1]["total"] < trace[0]["total"],
        "grad_norm_first_step": grad_norms,
        "gradients_reach_w_theta": grad_norms["w_theta"] > 0,
        "gradients_reach_g_phi": grad_norms["g_phi"] > 0,
        "w_theta_parameters_updated": bool(w_theta_updated),
        "g_phi_parameters_updated": bool(g_phi_updated),
        "param_hash_before": hash_before,
        "param_hash_after": hash_after,
        "predictions_before": pred_before,
        "predictions_after": pred_after,
        "predictions_in_0_1": all(0.0 <= p <= 1.0 for p in pred_after),
        "checkpoint_sha256": file_sha256(ckpt_path),
        "checkpoint_reload_reproduces_predictions": pred_reload == pred_after,
        "loss_trace": trace,
    }
    (run_dir / "smokes/one_group_overfit_smoke.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "smoke", "baselines"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "smoke":
        r = run_smoke(args.run_dir)
        print(json.dumps({k: v for k, v in r.items() if k != "loss_trace"}, indent=1))
    elif args.mode == "baselines":
        r = run_baselines(args.run_dir)
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("groups",)}
                          for k, v in r.items()}, indent=1, default=str)[:2000])
    else:
        m = train_one_seed(args.seed, args.run_dir)
        print(json.dumps({k: v for k, v in m.items() if not k.startswith("preprocessing")},
                         indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

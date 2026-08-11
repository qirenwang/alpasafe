"""T26E-C candidate-only scene-generalization cross-validation.

Nested leave-one-scene-out CV over the 10 development scenes (original train +
original val) under the EXACT frozen T26E-B contract: same plan-only input,
same 20,801-parameter model, same two SmoothL1 losses at weight 1.0, same
AdamW/batch/epoch/patience configuration, same checkpoint metric and
tie-breaks, seeds {0,1,2}. No hyperparameter search, no architecture change.

Fold protocol (frozen before any optimization):
    sorted_scene_ids = lexicographic sort of the 10 development scene ids
    fold j: outer evaluation = sorted_scene_ids[j]
            inner validation = sorted_scene_ids[(j + 1) % 10]
            fold training    = the remaining 8 scenes

Leakage guards:
    - plan standardization is fit on fold-training scenes only;
    - checkpoint selection uses the inner-validation scene only;
    - outer-scene tensors live in an ``OuterSceneVault`` that refuses to
      release them until the fold checkpoint is frozen (runtime guard);
    - the two sealed test scenes never enter the development pool
      (asserted against the frozen split manifest, metadata only).

Modes: freeze | run | aggregate | reproduce
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from safeworld.t26e.b1_diag import (
    _margin_pairwise,
    _within_group_stats,
    scene_of,
)
from safeworld.t26e.model import SafeWorldCandidatePilot
from safeworld.t26e.train_pilot import (
    BATCH_SIZE,
    LR,
    MAX_EPOCHS,
    PATIENCE,
    WEIGHT_DECAY,
    K_VIEWS,
    candidate_metrics,
    checkpoint_key,
    compute_losses,
    evaluate,
    file_sha256,
    fit_train_plan_stats,
    future_metrics,
    group_metrics,
    param_hash,
    spearman,
    standardize,
    tensorize,
)

ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = ROOT / "outputs/t26e_official_score_training_dataset_alpasim_196d21a"
SEEDS = (0, 1, 2)
N_FOLDS = 10
BOOTSTRAP_SEED = 2607
BOOTSTRAP_RESAMPLES = 10_000

FROZEN_CONTRACT = {
    "task": "t26e_c_candidate_only_scene_cv",
    "inherits": "T26E-B frozen contract, unchanged",
    "input": "input.planned_trajectory [B,64,2] float32 ego_t0_rig 10Hz (only input)",
    "model": "SafeWorldCandidatePilot (W_theta -> G_phi), 20801 parameters",
    "losses": "L_total = 1.0*masked SmoothL1(future,beta=1) + 1.0*SmoothL1(score,beta=1)",
    "optimizer": {
        "name": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "seeds": list(SEEDS),
    },
    "checkpoint_selection": {
        "primary": "minimum inner-validation official-score MAE",
        "tie_breaks": [
            "lower inner-validation K8 mean selected-score regret",
            "lower inner-validation future ADE",
            "lower inner-validation total loss",
            "earlier epoch",
        ],
        "data": "inner-validation scene ONLY",
    },
    "preprocessing": "per-coordinate plan mean/std over FOLD-TRAINING scenes only",
    "preregistered_interpretation_rule": {
        "aggregation": "median over 3 seeds within each outer scene, then across scenes",
        "REPRODUCIBLE_PRELIMINARY_GAIN": [
            "macro mean model K8 regret < macro mean rank-0 K8 regret",
            "model K8 regret < rank-0 in >= 6 of 10 outer scenes",
            "macro median outer-scene K8 pairwise accuracy > 0.5",
        ],
        "SCENE_DEPENDENT_PRELIMINARY_GAIN": [
            "macro mean model K8 regret < rank-0",
            "macro median K8 pairwise accuracy > 0.5",
            "fewer than 6 of 10 scenes improve",
        ],
        "otherwise": "NO_CONSISTENT_GAIN",
        "global_spearman_role": "diagnostic only; not required for classification",
    },
    "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
    "forbidden": [
        "test scenes/labels",
        "architecture change",
        "loss change",
        "hyperparameter search",
        "metadata features",
        "outer-scene normalization/optimization/early-stopping/checkpoint-selection",
    ],
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ dev pool
def load_split_scenes() -> dict[str, list[str]]:
    manifest = json.loads((DATASET_ROOT / "t26e_split_manifest.json").read_text())
    return manifest["scene_split"]


def tensorize_dev() -> dict[str, Any]:
    """Development pool = original train + original val, scene id attached."""
    parts = [tensorize("train"), tensorize("val")]
    dev = {
        "plan": torch.cat([p["plan"] for p in parts]),
        "future": torch.cat([p["future"] for p in parts]),
        "tmask": torch.cat([p["tmask"] for p in parts]),
        "score": torch.cat([p["score"] for p in parts]),
        "meta": [
            {**m, "scene_id": scene_of(m["decision_group_id"])}
            for p in parts
            for m in p["meta"]
        ],
        "feature_source": "input.planned_trajectory",
    }
    scenes = load_split_scenes()
    dev_scene_ids = {m["scene_id"] for m in dev["meta"]}
    sealed = set(scenes["test"])
    if dev_scene_ids & sealed:
        raise RuntimeError("BLOCKED_T26E_C_TEST_SEALING: sealed scene in development pool")
    assert dev_scene_ids == set(scenes["train"]) | set(scenes["val"])
    assert len(dev["meta"]) == 240
    return dev


def subset(dev: dict[str, Any], idx: list[int]) -> dict[str, Any]:
    t = torch.tensor(idx, dtype=torch.long)
    return {
        "plan": dev["plan"][t],
        "future": dev["future"][t],
        "tmask": dev["tmask"][t],
        "score": dev["score"][t],
        "meta": [dev["meta"][i] for i in idx],
    }


# ------------------------------------------------------------- fold protocol
def build_fold_manifest() -> dict[str, Any]:
    scenes = load_split_scenes()
    dev_sorted = sorted(scenes["train"] + scenes["val"])
    assert len(dev_sorted) == N_FOLDS
    folds = []
    for j in range(N_FOLDS):
        outer = dev_sorted[j]
        inner = dev_sorted[(j + 1) % N_FOLDS]
        train = [s for s in dev_sorted if s not in (outer, inner)]
        assert len(train) == 8 and outer != inner
        folds.append(
            {
                "fold": j,
                "outer_evaluation_scene": outer,
                "inner_validation_scene": inner,
                "fold_training_scenes": train,
            }
        )
    outer_once = [f["outer_evaluation_scene"] for f in folds]
    inner_once = [f["inner_validation_scene"] for f in folds]
    assert sorted(outer_once) == dev_sorted and sorted(inner_once) == dev_sorted
    train_counts = {s: sum(s in f["fold_training_scenes"] for f in folds) for s in dev_sorted}
    assert all(c == 8 for c in train_counts.values())
    assert not (set(dev_sorted) & set(scenes["test"]))
    return {
        "task": "t26e_c_candidate_only_scene_cv",
        "created_utc": utcnow(),
        "development_scene_ids_sorted": dev_sorted,
        "sealed_test_scenes_excluded": scenes["test"],
        "rule": "fold j: outer=sorted[j], inner=sorted[(j+1) mod 10], train=remaining 8",
        "n_folds": N_FOLDS,
        "seeds": list(SEEDS),
        "expected_counts_per_fold": {
            "train": {"scenes": 8, "groups": 24, "candidates": 192},
            "inner_validation": {"scenes": 1, "groups": 3, "candidates": 24},
            "outer_evaluation": {"scenes": 1, "groups": 3, "candidates": 24},
        },
        "every_scene_serves": {"outer": 1, "inner": 1, "train": 8},
        "folds": folds,
    }


def fold_indices(
    dev: dict[str, Any], fold: dict[str, Any]
) -> tuple[list[int], list[int], list[int]]:
    train_scenes = set(fold["fold_training_scenes"])
    inner, outer = fold["inner_validation_scene"], fold["outer_evaluation_scene"]
    tr = [i for i, m in enumerate(dev["meta"]) if m["scene_id"] in train_scenes]
    iv = [i for i, m in enumerate(dev["meta"]) if m["scene_id"] == inner]
    oe = [i for i, m in enumerate(dev["meta"]) if m["scene_id"] == outer]
    assert len(tr) == 192 and len(iv) == 24 and len(oe) == 24
    assert not (set(tr) & set(iv)) and not (set(tr) & set(oe)) and not (set(iv) & set(oe))
    assert len(tr) + len(iv) + len(oe) == len(dev["meta"])
    return tr, iv, oe


class OuterSceneVault:
    """Runtime leakage guard: outer-scene tensors are inaccessible until the
    fold checkpoint has been frozen (Task 6 outer-label isolation)."""

    def __init__(self, dev: dict[str, Any], outer_idx: list[int]):
        self._dev = dev
        self._idx = outer_idx
        self.checkpoint_frozen = False
        self.checkpoint_frozen_utc: str | None = None
        self.first_read_utc: str | None = None

    def freeze_checkpoint(self) -> None:
        self.checkpoint_frozen = True
        self.checkpoint_frozen_utc = utcnow()

    def open(self) -> dict[str, Any]:
        if not self.checkpoint_frozen:
            raise RuntimeError(
                "BLOCKED_T26E_C_TARGET_LEAKAGE: outer-evaluation scene requested "
                "before the fold checkpoint was frozen"
            )
        if self.first_read_utc is None:
            self.first_read_utc = utcnow()
        return subset(self._dev, self._idx)


# ---------------------------------------------------------------- training
def train_fold_seed(
    fold_id: int,
    seed: int,
    train_data: dict[str, Any],
    inner_data: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """One frozen-contract training run. Sees fold-train + inner-val ONLY."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    stats = fit_train_plan_stats(train_data["plan"])  # fold-train only
    model = SafeWorldCandidatePilot()
    init_hash = param_hash(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(seed)

    ckpt_dir = run_dir / f"checkpoints/fold_{fold_id}_seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_f = run_dir / f"metrics/fold_{fold_id}_seed_{seed}_epoch_history.jsonl"
    history_f.write_text("")

    best: dict[str, Any] | None = None
    bad = 0
    n = train_data["plan"].shape[0]
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        ep_losses = []
        for s in range(0, n, BATCH_SIZE):
            idx = perm[s : s + BATCH_SIZE]
            batch = {k: train_data[k][idx] for k in ("plan", "future", "tmask", "score")}
            opt.zero_grad()
            out = model(standardize(batch["plan"], stats), batch["plan"])
            losses = compute_losses(out, batch)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(
                    f"BLOCKED_T26E_C_TRAINING_FAILURE: non-finite loss "
                    f"fold {fold_id} seed {seed} epoch {epoch}"
                )
            losses["total"].backward()
            opt.step()
            ep_losses.append({k: float(v) for k, v in losses.items()})
        ev = evaluate(model, inner_data, stats)
        entry = {
            "epoch": epoch,
            "val_score_mae": ev["candidate"]["mae"],
            "val_k8_mean_regret": ev["groups"]["K8"]["mean_selected_score_regret"],
            "val_future_ade": ev["future"]["ade_m"],
            "val_total_loss": ev["losses"]["total"],
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
                        "inner_spearman": ev["candidate"]["spearman"],
                        "inner_k8_top1": ev["groups"]["K8"]["top1_selection_accuracy"],
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
                    "fold": fold_id,
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
    inner_final = evaluate(model, inner_data, stats)
    return {
        "fold": fold_id,
        "seed": seed,
        "selected_epoch": ckpt["epoch"],
        "early_stopping": {"patience": PATIENCE, "stopped_after_epoch": ckpt["epoch"] + bad},
        "checkpoint_path": str(ckpt_dir / "best.pt"),
        "checkpoint_sha256": file_sha256(ckpt_dir / "best.pt"),
        "initialization_parameter_hash": init_hash,
        "parameter_hash": param_hash(model),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "preprocessing_stats": {k: v.tolist() for k, v in stats.items()},
        "inner_validation": {
            "score_mae": inner_final["candidate"]["mae"],
            "score_rmse": inner_final["candidate"]["rmse"],
            "spearman": inner_final["candidate"]["spearman"],
            "future_ade_m": inner_final["future"]["ade_m"],
            "future_fde_m": inner_final["future"]["fde_m"],
            "k8_mean_regret": inner_final["groups"]["K8"]["mean_selected_score_regret"],
            "k8_pairwise": inner_final["groups"]["K8"]["pairwise_ranking_accuracy"],
        },
    }


# ------------------------------------------------------------ outer metrics
def rank0_regret_per_group(
    target: np.ndarray, meta: list[dict[str, Any]], k: int
) -> dict[str, float]:
    groups: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        if m["candidate_index"] < k:
            groups.setdefault(m["decision_group_id"], []).append(i)
    out = {}
    for gid, idx in groups.items():
        best = max(float(target[i]) for i in idx)
        r0 = next(i for i in idx if meta[i]["candidate_index"] == 0)
        out[gid] = best - float(target[r0])
    return out


def outer_scene_metrics(
    ev: dict[str, Any], outer: dict[str, Any]
) -> dict[str, Any]:
    """Full Task 9 metric block for one fold/seed outer evaluation."""
    pred = np.array(ev["predictions"])
    target = outer["score"].numpy()
    meta = outer["meta"]

    gidx: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        gidx.setdefault(m["decision_group_id"], []).append(i)
    wg = [_within_group_stats(pred[idx], target[idx]) for idx in gidx.values()]
    centered_p, centered_t = pred.copy(), target.copy()
    for idx in gidx.values():
        centered_p[idx] -= centered_p[idx].mean()
        centered_t[idx] -= centered_t[idx].mean()

    rows = [
        {
            "decision_group_id": m["decision_group_id"],
            "predicted_official_alpasim_scene_score": float(pred[i]),
            "official_alpasim_scene_score": float(target[i]),
        }
        for i, m in enumerate(meta)
    ]
    k_views = {}
    for name, k in K_VIEWS.items():
        gm = group_metrics(pred, target, meta, k)
        r0 = rank0_regret_per_group(target, meta, k)
        k_views[name] = {
            "top1": gm["top1_selection_accuracy"],
            "pairwise": gm["pairwise_ranking_accuracy"],
            "mean_regret": gm["mean_selected_score_regret"],
            "median_regret": gm["median_selected_score_regret"],
            "max_regret": gm["max_selected_score_regret"],
            "rank0_mean_regret": float(np.mean(list(r0.values()))),
            "per_group_regret_vs_rank0": {
                g["decision_group_id"]: g["selected_score_regret"] - r0[g["decision_group_id"]]
                for g in gm["per_group"]
            },
        }

    cm = candidate_metrics(pred, target)
    return {
        "candidate": {
            "mae": cm["mae"],
            "rmse": cm["rmse"],
            "global_within_scene_spearman": cm["spearman"],
            "prediction_std": cm["prediction_std"],
            "target_std": cm["target_std"],
        },
        "k_views": k_views,
        "group_aware": {
            "mean_within_group_spearman": float(np.mean([w["spearman"] for w in wg])),
            "mean_within_group_kendall": float(np.mean([w["kendall_tau"] for w in wg])),
            "group_centered_spearman": spearman(centered_p, centered_t),
        },
        "margin_pairwise": _margin_pairwise(rows),
        "future": {
            "model_ade_m": ev["future"]["ade_m"],
            "model_fde_m": ev["future"]["fde_m"],
        },
    }


# ------------------------------------------------------------ fold baselines
def fold_baselines(
    fold_id: int,
    train_data: dict[str, Any],
    outer: dict[str, Any],
    stats: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Task 8 outer-scene baselines; nothing is fit on outer labels."""
    target = outer["score"].numpy()
    meta = outer["meta"]
    out: dict[str, Any] = {"fold": fold_id}

    def block(pred: np.ndarray, note: str) -> dict[str, Any]:
        cm = candidate_metrics(pred, target)
        kv = {}
        for name, k in K_VIEWS.items():
            gm = group_metrics(pred, target, meta, k)
            kv[name] = {
                "top1": gm["top1_selection_accuracy"],
                "pairwise": gm["pairwise_ranking_accuracy"],
                "mean_regret": gm["mean_selected_score_regret"],
            }
        return {
            "note": note,
            "mae": cm["mae"],
            "rmse": cm["rmse"],
            "spearman": cm["spearman"],
            "k_views": kv,
        }

    tr_scores = train_data["score"]
    for name, const in (
        ("fold_train_mean_constant", float(tr_scores.mean())),
        ("constant_0_5", 0.5),
        ("fold_train_median_constant", float(tr_scores.median())),
    ):
        out[name] = {
            "constant": const,
            **block(np.full_like(target, const), "constant; selector falls to rank-0"),
        }

    # untrained same-architecture model, fold-train-fitted preprocessing
    untr = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = SafeWorldCandidatePilot()
        model.eval()
        with torch.no_grad():
            o = model(standardize(outer["plan"], stats), outer["plan"])
        untr[f"seed_{seed}"] = block(
            o["predicted_official_alpasim_scene_score"].numpy(), "untrained init"
        )
    out["untrained_model"] = untr

    # candidate-index-0 Alpamayo selector (constant preds + rank-0 tie-break)
    const_pred = np.full_like(target, float(tr_scores.mean()))
    r0 = {}
    for name, k in K_VIEWS.items():
        gm = group_metrics(const_pred, target, meta, k)
        assert all(g["selected_candidate_index"] == 0 for g in gm["per_group"])
        r0[name] = {
            "top1": gm["top1_selection_accuracy"],
            "pairwise": None,
            "mean_regret": gm["mean_selected_score_regret"],
            "median_regret": gm["median_selected_score_regret"],
            "max_regret": gm["max_selected_score_regret"],
            "selection_rule": "candidate_index=0",
        }
    out["alpamayo_rank0"] = r0

    oracle = {}
    for name, k in K_VIEWS.items():
        gm = group_metrics(target, target, meta, k)
        oracle[name] = {
            "top1": gm["top1_selection_accuracy"],
            "mean_regret": gm["mean_selected_score_regret"],
            "note": "reporting-only upper bound",
        }
    out["oracle"] = oracle

    # raw plan as future prediction on the outer scene
    fm = future_metrics(outer["plan"], outer["future"], outer["tmask"])
    out["raw_plan_future"] = {"ade_m": fm["ade_m"], "fde_m": fm["fde_m"]}
    return out


# ------------------------------------------------------------------- run all
def run_fold_seed(
    dev: dict[str, Any],
    fold: dict[str, Any],
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    fold_id = fold["fold"]
    tr, iv, oe = fold_indices(dev, fold)
    train_data, inner_data = subset(dev, tr), subset(dev, iv)
    vault = OuterSceneVault(dev, oe)

    manifest = train_fold_seed(fold_id, seed, train_data, inner_data, run_dir)
    vault.freeze_checkpoint()

    # checkpoint frozen -> outer labels may now be read exactly once
    outer = vault.open()
    ckpt = torch.load(Path(manifest["checkpoint_path"]), weights_only=False)
    model = SafeWorldCandidatePilot()
    model.load_state_dict(ckpt["model_state"])
    stats = {k: torch.tensor(v) for k, v in ckpt["stats"].items()}
    ev = evaluate(model, outer, stats)
    om = outer_scene_metrics(ev, outer)
    raw_fm = future_metrics(outer["plan"], outer["future"], outer["tmask"])
    om["future"].update(
        {
            "raw_plan_ade_m": raw_fm["ade_m"],
            "raw_plan_fde_m": raw_fm["fde_m"],
            "ade_absolute_improvement_m": raw_fm["ade_m"] - om["future"]["model_ade_m"],
            "ade_relative_improvement": 1.0 - om["future"]["model_ade_m"] / raw_fm["ade_m"],
            "fde_absolute_improvement_m": raw_fm["fde_m"] - om["future"]["model_fde_m"],
            "fde_relative_improvement": 1.0 - om["future"]["model_fde_m"] / raw_fm["fde_m"],
        }
    )

    with open(run_dir / f"predictions/fold_{fold_id}_seed_{seed}_outer_predictions.jsonl", "w") as f:
        for m, p, t in zip(outer["meta"], ev["predictions"], outer["score"].tolist()):
            f.write(
                json.dumps(
                    {
                        **m,
                        "fold": fold_id,
                        "seed": seed,
                        "predicted_official_alpasim_scene_score": p,
                        "official_alpasim_scene_score": t,
                    }
                )
                + "\n"
            )

    manifest.update(
        {
            "outer_evaluation_scene": fold["outer_evaluation_scene"],
            "inner_validation_scene": fold["inner_validation_scene"],
            "checkpoint_frozen_utc": vault.checkpoint_frozen_utc,
            "outer_labels_first_read_utc": vault.first_read_utc,
            "outer": om,
        }
    )
    (run_dir / f"metrics/fold_{fold_id}_seed_{seed}_outer_metrics.json").write_text(
        json.dumps(manifest, indent=1)
    )
    return manifest


def mode_freeze(run_dir: Path) -> None:
    manifest = build_fold_manifest()
    path = run_dir / "fold_manifests/t26e_c_scene_fold_manifest.json"
    path.write_text(json.dumps(manifest, indent=1))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (run_dir / "fold_manifests/t26e_c_scene_fold_manifest.sha256").write_text(sha + "\n")
    contract_path = run_dir / "fold_configs/t26e_c_frozen_contract.json"
    contract_path.write_text(json.dumps(FROZEN_CONTRACT, indent=1))
    (run_dir / "fold_configs/t26e_c_frozen_contract.sha256").write_text(
        hashlib.sha256(contract_path.read_bytes()).hexdigest() + "\n"
    )
    for fold in manifest["folds"]:
        (run_dir / f"fold_configs/fold_{fold['fold']}.json").write_text(
            json.dumps({**fold, "seeds": list(SEEDS), "contract": "t26e_c_frozen_contract.json"}, indent=1)
        )
    lines = [
        "# T26E-C scene-CV fold protocol (frozen before optimization)",
        "",
        f"- frozen at: {manifest['created_utc']}",
        f"- fold manifest sha256: `{sha}`",
        "- development pool: 10 scenes (original 8 train + original 2 val), "
        "30 decision groups, 240 candidates",
        f"- sealed test scenes excluded: {manifest['sealed_test_scenes_excluded']}",
        "- rule: sort the 10 development scene ids lexicographically; "
        "fold j uses sorted[j] as the outer-evaluation scene, "
        "sorted[(j+1) mod 10] as the inner-validation scene, and the "
        "remaining 8 scenes for training.",
        "- every scene serves exactly once as outer, once as inner, 8x as train.",
        "- per fold: train 8 scenes/24 groups/192 candidates; inner 1/3/24; outer 1/3/24.",
        "- 3 seeds (0,1,2) per fold -> 30 full frozen-contract training runs.",
        "",
        "| fold | outer evaluation scene | inner validation scene |",
        "|---|---|---|",
    ] + [
        f"| {f['fold']} | {f['outer_evaluation_scene']} | {f['inner_validation_scene']} |"
        for f in manifest["folds"]
    ]
    (run_dir / "reports/t26e_c_scene_fold_protocol.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"frozen": True, "sha256": sha, "n_folds": len(manifest["folds"])}))


def mode_run(run_dir: Path, only_fold: int | None = None) -> None:
    manifest = json.loads(
        (run_dir / "fold_manifests/t26e_c_scene_fold_manifest.json").read_text()
    )
    dev = tensorize_dev()
    results_f = run_dir / "metrics/t26e_c_fold_seed_results.jsonl"
    for fold in manifest["folds"]:
        if only_fold is not None and fold["fold"] != only_fold:
            continue
        for seed in SEEDS:
            row = run_fold_seed(dev, fold, seed, run_dir)
            with open(results_f, "a") as f:
                slim = {k: v for k, v in row.items() if k != "preprocessing_stats"}
                f.write(json.dumps(slim) + "\n")
            print(
                f"fold {fold['fold']} seed {seed}: epoch {row['selected_epoch']}, "
                f"outer MAE {row['outer']['candidate']['mae']:.4f}, "
                f"K8 regret {row['outer']['k_views']['K8']['mean_regret']:.4f} "
                f"(rank0 {row['outer']['k_views']['K8']['rank0_mean_regret']:.4f})",
                flush=True,
            )
        # fold baselines (seed-independent constants + per-seed untrained inits);
        # outer labels are only touched after the fold's checkpoints were frozen
        tr, _, oe = fold_indices(dev, fold)
        train_data = subset(dev, tr)
        stats = fit_train_plan_stats(train_data["plan"])
        bl = fold_baselines(fold["fold"], train_data, subset(dev, oe), stats)
        bl["outer_evaluation_scene"] = fold["outer_evaluation_scene"]
        (run_dir / f"metrics/fold_{fold['fold']}_baselines.json").write_text(
            json.dumps(bl, indent=1)
        )


def mode_reproduce(run_dir: Path) -> None:
    """Task 12: repeat fold 0 / seed 0 and compare against the recorded run."""
    manifest = json.loads(
        (run_dir / "fold_manifests/t26e_c_scene_fold_manifest.json").read_text()
    )
    fold = manifest["folds"][0]
    dev = tensorize_dev()
    repro_dir = run_dir / "reproducibility"
    for sub in ("checkpoints", "metrics", "predictions"):
        (repro_dir / sub).mkdir(parents=True, exist_ok=True)
    row = run_fold_seed(dev, fold, 0, repro_dir)

    orig = json.loads((run_dir / "metrics/fold_0_seed_0_outer_metrics.json").read_text())
    orig_preds = (run_dir / "predictions/fold_0_seed_0_outer_predictions.jsonl").read_text()
    new_preds = (repro_dir / "predictions/fold_0_seed_0_outer_predictions.jsonl").read_text()
    checks = {
        "fold_membership_identical": (
            fold["fold_training_scenes"]
            == json.loads((run_dir / "fold_configs/fold_0.json").read_text())[
                "fold_training_scenes"
            ]
        ),
        "preprocessing_stats_identical": row["preprocessing_stats"]
        == orig["preprocessing_stats"],
        "initialization_hash_identical": row["initialization_parameter_hash"]
        == orig["initialization_parameter_hash"],
        "selected_epoch_identical": row["selected_epoch"] == orig["selected_epoch"],
        "predictions_identical": orig_preds == new_preds,
        "checkpoint_sha256_identical": row["checkpoint_sha256"]
        == orig["checkpoint_sha256"],
        "outer_metrics_identical": json.dumps(row["outer"], sort_keys=True)
        == json.dumps(orig["outer"], sort_keys=True),
        "final_parameter_hash_identical": row["parameter_hash"] == orig["parameter_hash"],
    }
    result = {
        "fold": 0,
        "seed": 0,
        "checks": checks,
        "reproducible": all(checks.values()),
        "deterministic_boundary": "CPU, torch num_threads=1, fixed seeds, "
        "identical library versions in the same venv",
    }
    (repro_dir / "t26e_c_reproducibility.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


# ---------------------------------------------------------------- aggregate
def mode_aggregate(run_dir: Path) -> None:
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics/t26e_c_fold_seed_results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == N_FOLDS * len(SEEDS), f"expected 30 rows, got {len(rows)}"

    # ---- scene level: median over seeds within each outer scene
    def med(fold_rows: list[dict[str, Any]], path: list[str]) -> float:
        vals = []
        for r in fold_rows:
            v: Any = r
            for p in path:
                v = v[p]
            vals.append(v)
        return float(np.median(vals))

    scene_level = []
    for j in range(N_FOLDS):
        frows = [r for r in rows if r["fold"] == j]
        assert len(frows) == 3
        scene = frows[0]["outer_evaluation_scene"]
        rank0_k8 = frows[0]["outer"]["k_views"]["K8"]["rank0_mean_regret"]
        baselines = json.loads((run_dir / f"metrics/fold_{j}_baselines.json").read_text())
        assert baselines["outer_evaluation_scene"] == scene
        entry = {
            "fold": j,
            "outer_scene": scene,
            "rank0_k8_mean_regret": rank0_k8,
            "model_k8_mean_regret": med(frows, ["outer", "k_views", "K8", "mean_regret"]),
            "model_k8_pairwise": med(frows, ["outer", "k_views", "K8", "pairwise"]),
            "model_k8_top1": med(frows, ["outer", "k_views", "K8", "top1"]),
            "rank0_k8_top1": baselines["alpamayo_rank0"]["K8"]["top1"],
            "model_mae": med(frows, ["outer", "candidate", "mae"]),
            "global_within_scene_spearman": med(
                frows, ["outer", "candidate", "global_within_scene_spearman"]
            ),
            "mean_within_group_spearman": med(
                frows, ["outer", "group_aware", "mean_within_group_spearman"]
            ),
            "mean_within_group_kendall": med(
                frows, ["outer", "group_aware", "mean_within_group_kendall"]
            ),
            "group_centered_spearman": med(
                frows, ["outer", "group_aware", "group_centered_spearman"]
            ),
            "model_ade_m": med(frows, ["outer", "future", "model_ade_m"]),
            "raw_plan_ade_m": frows[0]["outer"]["future"]["raw_plan_ade_m"],
            "model_fde_m": med(frows, ["outer", "future", "model_fde_m"]),
            "raw_plan_fde_m": frows[0]["outer"]["future"]["raw_plan_fde_m"],
        }
        for name in K_VIEWS:
            entry[f"{name.lower()}_mean_regret"] = med(
                frows, ["outer", "k_views", name, "mean_regret"]
            )
            entry[f"{name.lower()}_median_regret"] = med(
                frows, ["outer", "k_views", name, "median_regret"]
            )
            entry[f"{name.lower()}_max_regret"] = med(
                frows, ["outer", "k_views", name, "max_regret"]
            )
            entry[f"{name.lower()}_top1"] = med(frows, ["outer", "k_views", name, "top1"])
            entry[f"{name.lower()}_pairwise"] = med(
                frows, ["outer", "k_views", name, "pairwise"]
            )
            entry[f"{name.lower()}_rank0_mean_regret"] = frows[0]["outer"]["k_views"][name][
                "rank0_mean_regret"
            ]
        entry["delta_regret_scene"] = (
            entry["model_k8_mean_regret"] - entry["rank0_k8_mean_regret"]
        )
        scene_level.append(entry)

    # rank-0 top1 needs baseline metrics; computed in baselines file (joined later)
    deltas = np.array([s["delta_regret_scene"] for s in scene_level])
    improved = int((deltas < 0).sum())
    equal = int((deltas == 0).sum())
    worse = int((deltas > 0).sum())

    def macro(key: str) -> dict[str, float]:
        v = np.array([s[key] for s in scene_level], dtype=float)
        return {
            "macro_mean": float(v.mean()),
            "macro_median": float(np.median(v)),
            "std": float(v.std()),
            "min": float(v.min()),
            "max": float(v.max()),
        }

    macro_keys = [
        "model_k8_mean_regret",
        "rank0_k8_mean_regret",
        "model_k8_pairwise",
        "model_k8_top1",
        "model_mae",
        "global_within_scene_spearman",
        "mean_within_group_spearman",
        "mean_within_group_kendall",
        "group_centered_spearman",
        "delta_regret_scene",
        "model_ade_m",
        "raw_plan_ade_m",
        "model_fde_m",
        "raw_plan_fde_m",
    ] + [
        f"{n.lower()}_{f}"
        for n in K_VIEWS
        for f in ("mean_regret", "median_regret", "max_regret", "top1", "pairwise", "rank0_mean_regret")
    ]
    macros = {k: macro(k) for k in macro_keys}

    paired = {
        "definition": "delta_regret_scene = median_model_K8_regret - rank0_K8_regret; negative improves",
        "n_scenes_improved": improved,
        "n_scenes_equal": equal,
        "n_scenes_worse": worse,
        "mean_paired_difference": float(deltas.mean()),
        "median_paired_difference": float(np.median(deltas)),
        "largest_improvement": float(deltas.min()),
        "largest_degradation": float(deltas.max()),
        "per_scene": {
            s["outer_scene"]: s["delta_regret_scene"] for s in scene_level
        },
    }

    # ---- bootstrap over the 10 outer scenes
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pairwise_v = np.array([s["model_k8_pairwise"] for s in scene_level])
    top1_delta = np.array(
        [s["model_k8_top1"] - s["rank0_k8_top1"] for s in scene_level]
    )
    boots: dict[str, list[float]] = {
        "delta_regret_mean": [],
        "pairwise_mean": [],
        "top1_delta_mean": [],
    }
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, N_FOLDS, N_FOLDS)
        boots["delta_regret_mean"].append(float(deltas[idx].mean()))
        boots["pairwise_mean"].append(float(pairwise_v[idx].mean()))
        boots["top1_delta_mean"].append(float(top1_delta[idx].mean()))

    def ci(v: list[float]) -> dict[str, float]:
        a = np.array(v)
        return {
            "mean": float(a.mean()),
            "ci95_low": float(np.percentile(a, 2.5)),
            "ci95_high": float(np.percentile(a, 97.5)),
        }

    bootstrap = {
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "unit": "outer scene (n=10), seed-median aggregated",
        "mean_k8_regret_difference_vs_rank0": ci(boots["delta_regret_mean"]),
        "mean_k8_pairwise_accuracy": ci(boots["pairwise_mean"]),
        "mean_k8_top1_difference_vs_rank0": ci(boots["top1_delta_mean"]),
        "note": "diagnostic only; not a training decision",
    }

    # ---- pre-registered classification
    cond1 = macros["model_k8_mean_regret"]["macro_mean"] < macros["rank0_k8_mean_regret"]["macro_mean"]
    cond2 = improved >= 6
    cond3 = macros["model_k8_pairwise"]["macro_median"] > 0.5
    if cond1 and cond2 and cond3:
        classification = "REPRODUCIBLE_PRELIMINARY_GAIN"
    elif cond1 and cond3:
        classification = "SCENE_DEPENDENT_PRELIMINARY_GAIN"
    else:
        classification = "NO_CONSISTENT_GAIN"

    out = {
        "scene_level": scene_level,
        "macro": macros,
        "paired_k8_regret_vs_rank0": paired,
        "bootstrap": bootstrap,
        "preregistered_classification": {
            "cond1_macro_mean_regret_beats_rank0": cond1,
            "cond2_at_least_6_of_10_scenes_improved": cond2,
            "n_scenes_improved": improved,
            "cond3_macro_median_k8_pairwise_gt_0_5": cond3,
            "macro_median_k8_pairwise": macros["model_k8_pairwise"]["macro_median"],
            "classification": classification,
        },
    }
    (run_dir / "metrics/t26e_c_scene_macro_metrics.json").write_text(json.dumps(out, indent=1))
    print(
        json.dumps(
            {
                "classification": classification,
                "macro_model_k8_regret": macros["model_k8_mean_regret"]["macro_mean"],
                "macro_rank0_k8_regret": macros["rank0_k8_mean_regret"]["macro_mean"],
                "improved/equal/worse": [improved, equal, worse],
                "macro_median_pairwise": macros["model_k8_pairwise"]["macro_median"],
                "regret_delta_ci": bootstrap["mean_k8_regret_difference_vs_rank0"],
            },
            indent=1,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("freeze", "run", "aggregate", "reproduce"), required=True
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()
    {
        "freeze": lambda: mode_freeze(args.run_dir),
        "run": lambda: mode_run(args.run_dir, args.fold),
        "aggregate": lambda: mode_aggregate(args.run_dir),
        "reproduce": lambda: mode_reproduce(args.run_dir),
    }[args.mode]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

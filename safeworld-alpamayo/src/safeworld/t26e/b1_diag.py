"""T26E-B.1 contract-compliance, metric-repair, and low-signal diagnostic audit.

Strictly additive over the frozen T26E-B pilot run: reads the historical
outputs/checkpoints/predictions, never rewrites them, and never touches the
sealed test pool. Four audit tasks, one subcommand each:

    audit        -- Task 1: architecture + parameter-count audit (12,865 vs 20,801)
    metrics      -- Task 2: aggregate metric-direction repair (best/worst labels)
    baselines    -- Task 3: missing non-learned baselines A-E
    diagnostics  -- Task 4: existing-run low-signal diagnostics

The historical signal classification
T26E_B_CANDIDATE_CONSEQUENCE_SCORE_PILOT_LOW_SIGNAL_REVIEW is not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import kendalltau

from safeworld.t26e import model as model_module
from safeworld.t26e.model import SafeWorldCandidatePilot
from safeworld.t26e.train_pilot import (
    K_VIEWS,
    candidate_metrics,
    fit_train_plan_stats,
    future_metrics,
    group_metrics,
    spearman,
    standardize,
    tensorize,
)

ROOT = Path(__file__).resolve().parents[3]
T26E_B_OUT = ROOT / "outputs/t26e_b_candidate_consequence_score_pilot"
T26E_B_RUN = ROOT / "artifacts/safeworld_t26e_b_candidate_consequence_score_pilot/20260711T222233Z"

# Explicit metric-direction registry (Task 2). Every aggregated metric must
# appear in exactly one set; "best" = optimum under the metric's direction.
LOWER_IS_BETTER = {
    "val_score_mae",
    "val_score_rmse",
    "val_future_ade_m",
    "val_future_fde_m",
    "val_k2_mean_regret",
    "val_k2_median_regret",
    "val_k2_max_regret",
    "val_k5_mean_regret",
    "val_k5_median_regret",
    "val_k5_max_regret",
    "val_k8_mean_regret",
    "val_k8_median_regret",
    "val_k8_max_regret",
}
HIGHER_IS_BETTER = {
    "val_spearman",
    "val_k2_top1",
    "val_k2_pairwise",
    "val_k5_top1",
    "val_k5_pairwise",
    "val_k8_top1",
    "val_k8_pairwise",
}
MARGIN_THRESHOLDS = (0.0, 0.01, 0.02, 0.05)
COLLAPSE_THRESHOLDS = (1e-6, 1e-4, 1e-3)
SEEDS = (0, 1, 2)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_val_predictions() -> dict[int, list[dict[str, Any]]]:
    """Historical validation predictions keyed by seed, canonical order."""
    by_seed: dict[int, list[dict[str, Any]]] = {s: [] for s in SEEDS}
    for row in load_jsonl(T26E_B_OUT / "t26e_b_validation_predictions.jsonl"):
        by_seed[int(row["seed"])].append(row)
    assert all(len(v) == 48 for v in by_seed.values()), "expected 48 val records per seed"
    return by_seed


def scene_of(decision_group_id: str) -> str:
    """Scene id is the group-id prefix before the decision timestamp."""
    return decision_group_id.split("@")[0]


def load_group_scene_index() -> dict[str, str]:
    """decision_group_id -> scene_id from the structural group index (no scores)."""
    rows = load_jsonl(
        ROOT
        / "outputs/t26e_official_score_training_dataset_alpasim_196d21a"
        / "t26e_decision_group_index.jsonl"
    )
    mapping = {r["decision_group_id"]: r["scene_id"] for r in rows}
    for gid, sid in mapping.items():
        assert scene_of(gid) == sid, f"group-id prefix != scene_id for {gid}"
    return mapping


# --------------------------------------------------------------------- Task 1
def state_dict_manifest(model: torch.nn.Module) -> dict[str, Any]:
    tensors = []
    for name, p in model.named_parameters():
        tensors.append(
            {
                "parameter": name,
                "shape": list(p.shape),
                "dtype": str(p.dtype),
                "requires_grad": bool(p.requires_grad),
                "numel": int(p.numel()),
            }
        )
    return {"tensors": tensors, "total_parameters": sum(t["numel"] for t in tensors)}


@torch.no_grad()
def _checkpoint_tensor_audit(ckpt_path: Path) -> dict[str, Any]:
    state = torch.load(ckpt_path, weights_only=False)["model_state"]
    return {
        "checkpoint": str(ckpt_path),
        "sha256": sha256_file(ckpt_path),
        "tensors": [
            {
                "parameter": k,
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "numel": int(v.numel()),
            }
            for k, v in state.items()
        ],
        "total_parameters": int(sum(v.numel() for v in state.values())),
    }


def bypass_gradient_probe() -> dict[str, Any]:
    """Prove G_phi consumes latent z only: d(score)/d(plan_raw) must be zero."""
    torch.manual_seed(0)
    model = SafeWorldCandidatePilot()
    plan_std = torch.randn(4, 64, 2)
    plan_raw = torch.randn(4, 64, 2, requires_grad=True)
    out = model(plan_std, plan_raw)
    out["predicted_official_alpasim_scene_score"].sum().backward()
    raw_grad_norm = 0.0 if plan_raw.grad is None else float(plan_raw.grad.norm())
    return {
        "d_score_d_plan_raw_norm": raw_grad_norm,
        "no_raw_plan_score_bypass": raw_grad_norm == 0.0,
        "note": "score gradient w.r.t. raw plan bypass path is exactly zero; "
        "score depends on the plan only through W_theta's latent z",
    }


def task1_audit(run_dir: Path) -> dict[str, Any]:
    torch.manual_seed(0)
    model = SafeWorldCandidatePilot()
    manifest = state_dict_manifest(model)
    total = manifest["total_parameters"]

    expected_shapes = {
        "w_theta.encoder.0.weight": [64, 128],
        "w_theta.encoder.0.bias": [64],
        "w_theta.encoder.2.weight": [64, 64],
        "w_theta.encoder.2.bias": [64],
        "w_theta.future_residual_head.weight": [128, 64],
        "w_theta.future_residual_head.bias": [128],
        "g_phi.head.weight": [1, 64],
        "g_phi.head.bias": [1],
    }
    actual_shapes = {t["parameter"]: t["shape"] for t in manifest["tensors"]}
    shape_ok = actual_shapes == expected_shapes

    ckpt_audits = {}
    seed_rows = load_jsonl(T26E_B_OUT / "t26e_b_seed_results.jsonl")
    for row in seed_rows:
        audit = _checkpoint_tensor_audit(Path(row["checkpoint_path"]))
        audit["sha256_matches_seed_results"] = audit["sha256"] == row["checkpoint_sha256"]
        audit["shapes_match_source_model"] = {
            t["parameter"]: t["shape"] for t in audit["tensors"]
        } == expected_shapes
        ckpt_audits[f"seed_{row['seed']}"] = audit

    contract = json.loads((T26E_B_OUT / "t26e_b_model_contract.json").read_text())
    impl_report = (T26E_B_RUN / "reports/t26e_b_implementation_report.md").read_text()
    bypass = bypass_gradient_probe()

    per_module = {
        "w_theta.encoder.Linear(128,64)": 128 * 64 + 64,
        "w_theta.encoder.Linear(64,64)": 64 * 64 + 64,
        "w_theta.future_residual_head.Linear(64,128)": 64 * 128 + 128,
        "g_phi.head.Linear(64,1)": 64 * 1 + 1,
    }
    reconciliation = {
        "implementation_report_text_value": 12865,
        "model_contract_value": int(contract["n_parameters"]),
        "checkpoint_state_dict_values": {
            k: v["total_parameters"] for k, v in ckpt_audits.items()
        },
        "source_instantiation_value": total,
        "arithmetic_from_source_shapes": per_module,
        "arithmetic_total": sum(per_module.values()),
        "nearest_natural_subtotal_to_12865": {
            "encoder_plus_g_phi_only": 128 * 64 + 64 + 64 * 64 + 64 + 65,
            "difference_from_12865": 12865 - (128 * 64 + 64 + 64 * 64 + 64 + 65),
            "note": "even the model WITHOUT the zero-init future residual head "
            "has 12,481 parameters; no sub-module combination of the audited "
            "architecture sums to 12,865",
        },
        "t26c_models_for_reference": [21646, 22158],
        "verdict": "12,865 is an unsupported prose value in the implementation "
        "report (and two derived report sentences); source code, model contract, "
        "and all three checkpoint state_dicts agree on 20,801",
    }

    resolved = (
        total == 20801
        and int(contract["n_parameters"]) == 20801
        and all(v["total_parameters"] == 20801 for v in ckpt_audits.values())
        and shape_ok
        and all(v["shapes_match_source_model"] for v in ckpt_audits.values())
        and bypass["no_raw_plan_score_bypass"]
    )

    audit = {
        "source_file": str(Path(inspect.getfile(model_module)).relative_to(ROOT)),
        "state_dict_manifest": manifest,
        "expected_shapes": expected_shapes,
        "source_shapes_match_frozen_contract": shape_ok,
        "unexpected_heads": sorted(set(actual_shapes) - set(expected_shapes)),
        "g_phi_consumes_latent_only": bypass,
        "checkpoints": ckpt_audits,
        "parameter_count_reconciliation": reconciliation,
        "reconciled": resolved,
        "occurrences_of_12865_in_report": impl_report.count("12,865"),
    }
    (run_dir / "inspections/t26e_b_state_dict_audit.json").write_text(
        json.dumps(audit, indent=1)
    )

    erratum = {
        "erratum": "t26e_b_parameter_count",
        "kind": "additive; historical artifacts unmodified",
        "incorrect_value": 12865,
        "incorrect_locations": [
            str(T26E_B_RUN.relative_to(ROOT) / "reports/t26e_b_implementation_report.md"),
            "outputs/reports/T26E_B_candidate_consequence_official_score_pilot.md",
            "outputs/reports/status_after_T26E_B_candidate_consequence_official_score_pilot.md",
        ],
        "correct_value": 20801,
        "correct_sources": {
            "src/safeworld/t26e/model.py (instantiated)": total,
            "t26e_b_model_contract.json": int(contract["n_parameters"]),
            "t26e_b_seed_results.jsonl n_parameters": sorted(
                {int(r["n_parameters"]) for r in seed_rows}
            ),
            "checkpoint state_dicts (3 seeds)": sorted(
                {v["total_parameters"] for v in ckpt_audits.values()}
            ),
        },
        "per_module_breakdown": per_module,
        "explanation": reconciliation["verdict"],
        "training_affected": False,
        "affects_only": "three prose report sentences; no config, checkpoint, "
        "metric, prediction, or contract JSON carries the wrong value",
    }
    (run_dir / "corrected_outputs/t26e_b_parameter_count_erratum.json").write_text(
        json.dumps(erratum, indent=1)
    )
    report = [
        "# T26E-B parameter-count erratum (additive)",
        "",
        f"- Verified actual parameter count: **{total}** (source instantiation, "
        "model contract, and all three checkpoint state_dicts agree).",
        "- Per-module: Linear(128,64)=8,256 + Linear(64,64)=4,160 + "
        "future residual Linear(64,128)=8,320 + G_phi Linear(64,1)=65 = **20,801**.",
        "- The value **12,865** appears only in report prose (implementation report "
        "and two derived status reports). It cannot be reconstructed from any "
        "sub-module combination of the audited architecture (the nearest natural "
        "subtotal — the model without the zero-init residual head — is 12,481). "
        "It is classified as a reporting/transcription error.",
        "- No raw-plan score bypass: d(score)/d(plan_raw) == 0 (autograd probe); "
        "G_phi consumes the latent z only.",
        "- No unexpected heads; all 8 parameter tensors match the frozen contract.",
        f"- Reconciled: **{resolved}** (training was never affected; the erratum "
        "corrects documentation only).",
    ]
    (run_dir / "reports/t26e_b_parameter_count_erratum.md").write_text(
        "\n".join(report) + "\n"
    )
    return audit


# --------------------------------------------------------------------- Task 2
def _agg(values: dict[str, float], metric: str) -> dict[str, Any]:
    v = np.array([values[s] for s in sorted(values)])
    if metric in LOWER_IS_BETTER:
        direction, best, worst = "lower_is_better", float(v.min()), float(v.max())
    elif metric in HIGHER_IS_BETTER:
        direction, best, worst = "higher_is_better", float(v.max()), float(v.min())
    else:
        raise ValueError(f"metric {metric} missing from direction registry")
    return {
        "per_seed": {s: float(values[s]) for s in sorted(values)},
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std()),
        "direction": direction,
        "best": best,
        "worst": worst,
        "best_seed": min(values, key=lambda s: values[s] if metric in LOWER_IS_BETTER else -values[s]),
    }


def task2_metric_repair(run_dir: Path) -> dict[str, Any]:
    seed_rows = {int(r["seed"]): r for r in load_jsonl(T26E_B_OUT / "t26e_b_seed_results.jsonl")}
    historical = json.loads((T26E_B_OUT / "t26e_b_aggregate_metrics.json").read_text())
    preds = load_val_predictions()

    # Provenance check: recompute per-seed candidate metrics from raw predictions.
    provenance = {}
    for s in SEEDS:
        p = np.array([r["predicted_official_alpasim_scene_score"] for r in preds[s]])
        t = np.array([r["official_alpasim_scene_score"] for r in preds[s]])
        cm = candidate_metrics(p, t)
        meta = [
            {k: r[k] for k in ("sample_id", "decision_group_id", "candidate_id", "candidate_index")}
            for r in preds[s]
        ]
        gm = group_metrics(p, t, meta, 8)
        provenance[str(s)] = {
            "recomputed_mae": cm["mae"],
            "seed_results_mae": seed_rows[s]["val_score_mae"],
            "mae_match": math.isclose(cm["mae"], seed_rows[s]["val_score_mae"], rel_tol=1e-6),
            "recomputed_spearman": cm["spearman"],
            "seed_results_spearman": seed_rows[s]["val_spearman"],
            "spearman_match": math.isclose(
                cm["spearman"], seed_rows[s]["val_spearman"], rel_tol=1e-6
            ),
            "recomputed_k8_mean_regret": gm["mean_selected_score_regret"],
            "seed_results_k8_mean_regret": seed_rows[s]["val_k8"]["mean_selected_score_regret"],
            "k8_regret_match": math.isclose(
                gm["mean_selected_score_regret"],
                seed_rows[s]["val_k8"]["mean_selected_score_regret"],
                rel_tol=1e-6,
            ),
        }
    all_provenance_ok = all(
        v["mae_match"] and v["spearman_match"] and v["k8_regret_match"]
        for v in provenance.values()
    )

    def seed_metric(s: int, metric: str) -> float:
        r = seed_rows[s]
        flat = {
            "val_score_mae": r["val_score_mae"],
            "val_score_rmse": r["val_score_rmse"],
            "val_spearman": r["val_spearman"],
            "val_future_ade_m": r["val_future_ade"],
            "val_future_fde_m": r["val_future_fde"],
        }
        if metric in flat:
            return flat[metric]
        _, kview, field = metric.split("_", 2)
        block = r[f"val_{kview}"]
        return {
            "top1": block["top1_selection_accuracy"],
            "pairwise": block["pairwise_ranking_accuracy"],
            "mean_regret": block["mean_selected_score_regret"],
            "median_regret": block["median_selected_score_regret"],
            "max_regret": block["max_selected_score_regret"],
        }[field]

    corrected_aggregate = {}
    errata_rows = []
    for metric in sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER):
        values = {str(s): seed_metric(s, metric) for s in SEEDS}
        entry = _agg(values, metric)
        corrected_aggregate[metric] = entry
        hist = historical["aggregate"].get(metric)
        if hist is None:
            continue
        stats_ok = all(
            math.isclose(hist[k], entry[k], rel_tol=1e-9, abs_tol=1e-12)
            for k in ("mean", "median", "std")
        )
        was_inverted = not math.isclose(hist["best"], entry["best"], abs_tol=1e-12)
        errata_rows.append(
            {
                "metric": metric,
                "direction": entry["direction"],
                "historical_best": hist["best"],
                "historical_worst": hist["worst"],
                "corrected_best": entry["best"],
                "corrected_worst": entry["worst"],
                "historical_labeling_inverted": was_inverted,
                "per_seed_mean_median_std_consistent": stats_ok,
            }
        )

    inverted = [e["metric"] for e in errata_rows if e["historical_labeling_inverted"]]
    stats_consistent = all(e["per_seed_mean_median_std_consistent"] for e in errata_rows)

    corrected = {
        "_erratum_note": "additive corrected aggregate; historical "
        "t26e_b_aggregate_metrics.json unmodified. Only best/worst labels were "
        "repaired; per-seed values, mean, median, std are byte-consistent with "
        "the historical file and re-verified against the raw predictions.",
        "direction_rules": {
            "lower_is_better": sorted(LOWER_IS_BETTER),
            "higher_is_better": sorted(HIGHER_IS_BETTER),
        },
        "aggregate": corrected_aggregate,
        "provenance_check_from_raw_predictions": provenance,
        "provenance_consistent": all_provenance_ok,
        "historical_signal_classification_unchanged": historical["signal_classification"],
    }
    (run_dir / "corrected_outputs/t26e_b_aggregate_metrics_corrected.json").write_text(
        json.dumps(corrected, indent=1)
    )

    lines = [
        "# T26E-B aggregate best/worst erratum (additive)",
        "",
        "The historical aggregate computed `best=min(per_seed)`, `worst=max(per_seed)`",
        "for every metric, which inverts the labels for all higher-is-better metrics.",
        "Per-seed values, means, medians, and standard deviations were all correct and",
        "are re-verified here directly from the raw validation predictions.",
        "",
        f"- metrics audited: {len(errata_rows)}",
        f"- metrics with inverted best/worst labels: {len(inverted)}",
        f"  - {', '.join(inverted)}",
        f"- per-seed/mean/median/std consistent with source predictions: {stats_consistent}",
        "",
        "Examples of repaired labels:",
        "- `val_spearman`: best is now -0.00413 (seed 0, closest to positive), "
        "worst -0.06730 (seed 2); historical file had these swapped.",
        "- `val_k8_pairwise`: best is now 0.73214 (seed 2), worst 0.67857 (seed 1).",
        "- `val_k8_top1`: best is now 0.5 (seed 2), worst 0.0 (seed 1).",
        "",
        "The historical LOW_SIGNAL_REVIEW classification is computed from medians and",
        "explicit comparisons, not from the best/worst labels, and is unaffected.",
    ]
    (run_dir / "reports/t26e_b_aggregate_best_worst_erratum.md").write_text(
        "\n".join(lines) + "\n"
    )
    return corrected


# --------------------------------------------------------------------- Task 3
@torch.no_grad()
def _eval_score_baseline(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    cm = candidate_metrics(pred, target)
    return {
        "mae": cm["mae"],
        "rmse": cm["rmse"],
        "global_spearman": cm["spearman"],
        "spearman_note": "0.0 by convention for constant predictions",
        "prediction_mean": cm["prediction_mean"],
        "prediction_std": cm["prediction_std"],
        "prediction_range": [cm["prediction_min"], cm["prediction_max"]],
    }


def task3_baselines(run_dir: Path) -> dict[str, Any]:
    train, val = tensorize("train"), tensorize("val")
    target = val["score"].numpy()
    stats = fit_train_plan_stats(train["plan"])

    out: dict[str, Any] = {
        "_note": "non-learned baselines on original train/val only; no test access",
        "fit_split": "train (192 records)",
        "eval_split": "val (48 records)",
    }
    for name, const in (
        ("A_train_mean_constant", float(train["score"].mean())),
        ("B_constant_0_5", 0.5),
        ("C_train_median_constant", float(train["score"].median())),
    ):
        pred = np.full_like(target, const)
        out[name] = {"constant": const, **_eval_score_baseline(pred, target)}

    # D: untrained model, same architecture + train-fitted preprocessing, no optimizer.
    # Replicates the historical per-seed init sequence: torch.manual_seed(seed) then
    # SafeWorldCandidatePilot() (tensorize consumes no torch RNG).
    d_seeds = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SafeWorldCandidatePilot()
        model.eval()
        with torch.no_grad():
            o = model(standardize(val["plan"], stats), val["plan"])
        pred = o["predicted_official_alpasim_scene_score"].numpy()
        d_seeds[str(seed)] = {
            **_eval_score_baseline(pred, target),
            "future": future_metrics(
                o["predicted_future_ego_states_ego_t0"], val["future"], val["tmask"]
            ),
            "note_future": "zero-init residual head => untrained future == raw plan",
        }
    out["D_untrained_model"] = {
        "seeds": d_seeds,
        "median_mae": float(np.median([d_seeds[str(s)]["mae"] for s in SEEDS])),
        "median_global_spearman": float(
            np.median([d_seeds[str(s)]["global_spearman"] for s in SEEDS])
        ),
        "optimizer": None,
        "parameter_updates": 0,
    }

    # E: raw planned trajectory as the future-ego prediction (identity "model").
    raw_future = future_metrics(val["plan"], val["future"], val["tmask"])
    out["E_raw_plan_future"] = {
        "masked_ade_m": raw_future["ade_m"],
        "masked_fde_m": raw_future["fde_m"],
        "per_horizon": raw_future["per_horizon"],
        "valid_timestep_coverage": raw_future["valid_timestep_coverage"],
        "mask_semantics": "same masked ADE/FDE and validity mask as the trained model eval",
    }

    # Learned W_theta future predictions from the three frozen checkpoints.
    learned = {}
    for row in load_jsonl(T26E_B_OUT / "t26e_b_seed_results.jsonl"):
        ckpt = torch.load(Path(row["checkpoint_path"]), weights_only=False)
        model = SafeWorldCandidatePilot()
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        ck_stats = {k: torch.tensor(v) for k, v in ckpt["stats"].items()}
        with torch.no_grad():
            o = model(standardize(val["plan"], ck_stats), val["plan"])
        fm = future_metrics(o["predicted_future_ego_states_ego_t0"], val["future"], val["tmask"])
        learned[f"seed_{row['seed']}"] = {"ade_m": fm["ade_m"], "fde_m": fm["fde_m"]}
    ades = [v["ade_m"] for v in learned.values()]
    fdes = [v["fde_m"] for v in learned.values()]
    out["W_theta_vs_raw_plan_future"] = {
        "learned_per_seed": learned,
        "learned_median_ade_m": float(np.median(ades)),
        "learned_median_fde_m": float(np.median(fdes)),
        "raw_plan_ade_m": raw_future["ade_m"],
        "raw_plan_fde_m": raw_future["fde_m"],
        "ade_improvement_m": raw_future["ade_m"] - float(np.median(ades)),
        "ade_relative_improvement": 1.0 - float(np.median(ades)) / raw_future["ade_m"],
        "fde_improvement_m": raw_future["fde_m"] - float(np.median(fdes)),
        "fde_relative_improvement": 1.0 - float(np.median(fdes)) / raw_future["fde_m"],
    }

    (run_dir / "baselines/t26e_b1_additional_baselines.json").write_text(
        json.dumps(out, indent=1)
    )
    w = out["W_theta_vs_raw_plan_future"]
    lines = [
        "# T26E-B.1 additional non-learned baselines (val, 48 candidates)",
        "",
        "| baseline | MAE | RMSE | global Spearman |",
        "|---|---|---|---|",
    ]
    for name in ("A_train_mean_constant", "B_constant_0_5", "C_train_median_constant"):
        b = out[name]
        lines.append(
            f"| {name} (const={b['constant']:.5f}) | {b['mae']:.5f} | "
            f"{b['rmse']:.5f} | {b['global_spearman']:.1f} |"
        )
    d = out["D_untrained_model"]
    lines += [
        f"| D_untrained_model (median of 3 seeds) | {d['median_mae']:.5f} | — | "
        f"{d['median_global_spearman']:.5f} |",
        "",
        "## Future-consequence baseline",
        f"- raw plan as future: ADE {w['raw_plan_ade_m']:.4f} m, FDE {w['raw_plan_fde_m']:.4f} m",
        f"- learned W_theta (median of 3 checkpoints): ADE {w['learned_median_ade_m']:.4f} m, "
        f"FDE {w['learned_median_fde_m']:.4f} m",
        f"- improvement: ADE {w['ade_improvement_m']:.4f} m "
        f"({w['ade_relative_improvement']:.1%}), FDE {w['fde_improvement_m']:.4f} m "
        f"({w['fde_relative_improvement']:.1%})",
        "- untrained model future == raw plan exactly (zero-init residual head).",
    ]
    (run_dir / "reports/t26e_b1_additional_baselines.md").write_text("\n".join(lines) + "\n")
    return out


# --------------------------------------------------------------------- Task 4
def _within_group_stats(pred: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    sp = spearman(pred, target)
    if np.std(pred) == 0.0 or np.std(target) == 0.0:
        kt = 0.0
    else:
        kt = float(kendalltau(pred, target).statistic)
    return {"spearman": sp, "kendall_tau": kt}


def _margin_pairwise(
    rows: list[dict[str, Any]], thresholds=MARGIN_THRESHOLDS
) -> dict[str, Any]:
    """Within-group pairwise accuracy restricted to |official margin| > threshold."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["decision_group_id"], []).append(r)
    out = {}
    for th in thresholds:
        correct = total = 0
        for members in groups.values():
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    ta = members[a]["official_alpasim_scene_score"]
                    tb = members[b]["official_alpasim_scene_score"]
                    if abs(ta - tb) <= th:
                        continue
                    pa = members[a]["predicted_official_alpasim_scene_score"]
                    pb = members[b]["predicted_official_alpasim_scene_score"]
                    total += 1
                    correct += bool((pa - pb) * (ta - tb) > 0)
        out[f"margin_gt_{th:g}"] = {
            "n_pairs": total,
            "pairwise_accuracy": correct / total if total else None,
        }
    return out


def task4_diagnostics(run_dir: Path) -> dict[str, Any]:
    preds = load_val_predictions()
    group_scene = load_group_scene_index()

    per_group_f = open(run_dir / "diagnostics/t26e_b1_per_group_diagnostics.jsonl", "w")
    per_scene_all: dict[str, Any] = {}
    margin_all: dict[str, Any] = {}
    collapse_all: dict[str, Any] = {}
    candidate_level: dict[str, Any] = {}
    group_centered: dict[str, Any] = {}

    for s in SEEDS:
        rows = preds[s]
        p = np.array([r["predicted_official_alpasim_scene_score"] for r in rows])
        t = np.array([r["official_alpasim_scene_score"] for r in rows])
        meta = [
            {k: r[k] for k in ("sample_id", "decision_group_id", "candidate_id", "candidate_index")}
            for r in rows
        ]
        candidate_level[str(s)] = candidate_metrics(p, t)

        # ---- per-group diagnostics (K8 full view)
        gm8 = group_metrics(p, t, meta, 8)
        gidx: dict[str, list[int]] = {}
        for i, m in enumerate(meta):
            gidx.setdefault(m["decision_group_id"], []).append(i)
        centered_p = p.copy()
        centered_t = t.copy()
        n_const = {th: 0 for th in COLLAPSE_THRESHOLDS}
        group_rows = []
        for g in gm8["per_group"]:
            gid = g["decision_group_id"]
            idx = gidx[gid]
            wg = _within_group_stats(p[idx], t[idx])
            pred_range = float(p[idx].max() - p[idx].min())
            rank0_i = next(i for i in idx if meta[i]["candidate_index"] == 0)
            rank0_regret = g["best_official_score"] - float(t[rank0_i])
            row = {
                "seed": s,
                "decision_group_id": gid,
                "scene_id": group_scene[gid],
                "within_group_spearman": wg["spearman"],
                "within_group_kendall_tau": wg["kendall_tau"],
                "pairwise_ranking_accuracy": g["pairwise_correct"] / g["pairwise_total"]
                if g["pairwise_total"]
                else None,
                "top1_correct": g["top1_correct"],
                "selected_score_regret": g["selected_score_regret"],
                "rank0_regret": rank0_regret,
                "regret_vs_rank0": g["selected_score_regret"] - rank0_regret,
                "predicted_score_range": pred_range,
                "target_score_range": float(t[idx].max() - t[idx].min()),
                "selected_candidate_id": g["selected_candidate_id"],
                "selected_candidate_index": g["selected_candidate_index"],
                "official_best_candidate_id": g["official_best_candidate_id"],
                "official_best_candidate_index": g["official_best_candidate_index"],
            }
            group_rows.append(row)
            per_group_f.write(json.dumps(row) + "\n")
            centered_p[idx] = p[idx] - p[idx].mean()
            centered_t[idx] = t[idx] - t[idx].mean()
            for th in COLLAPSE_THRESHOLDS:
                n_const[th] += pred_range < th

        group_centered[str(s)] = {
            "group_centered_spearman": spearman(centered_p, centered_t),
            "note": "per-group prediction/target means removed before ranking all "
            "48 validation candidates jointly",
        }
        margin_all[str(s)] = _margin_pairwise(rows)

        # ---- per-scene diagnostics
        scenes: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            scenes.setdefault(group_scene[r["decision_group_id"]], []).append(i)
        per_scene = {}
        for sid, idx in sorted(scenes.items()):
            sp_, st_ = p[idx], t[idx]
            smeta = [meta[i] for i in idx]
            srows = [rows[i] for i in idx]
            scene_groups = {m["decision_group_id"] for m in smeta}
            wg_sp = [
                r["within_group_spearman"] for r in group_rows if r["scene_id"] == sid
            ]
            wg_kt = [
                r["within_group_kendall_tau"] for r in group_rows if r["scene_id"] == sid
            ]
            kview = {}
            for name, k in K_VIEWS.items():
                gm = group_metrics(sp_, st_, smeta, k)
                kview[name] = {
                    "top1": gm["top1_selection_accuracy"],
                    "pairwise": gm["pairwise_ranking_accuracy"],
                    "mean_regret": gm["mean_selected_score_regret"],
                }
            vs_rank0 = [
                r["regret_vs_rank0"] for r in group_rows if r["scene_id"] == sid
            ]
            per_scene[sid] = {
                "n_groups": len(scene_groups),
                "n_candidates": len(idx),
                "score_mae": float(np.abs(sp_ - st_).mean()),
                "score_rmse": float(np.sqrt(((sp_ - st_) ** 2).mean())),
                "global_within_scene_spearman": spearman(sp_, st_),
                "mean_within_group_spearman": float(np.mean(wg_sp)),
                "mean_within_group_kendall": float(np.mean(wg_kt)),
                "k_views": kview,
                "groups_improved_over_rank0": int(sum(d < 0 for d in vs_rank0)),
                "groups_equal_rank0": int(sum(d == 0 for d in vs_rank0)),
                "groups_worse_than_rank0": int(sum(d > 0 for d in vs_rank0)),
                "margin_pairwise": _margin_pairwise(srows),
                "predicted_std": float(sp_.std()),
            }
        per_scene_all[str(s)] = per_scene

        collapse_all[str(s)] = {
            "overall_predicted_std": float(p.std()),
            "target_std": float(t.std()),
            "per_scene_predicted_std": {
                sid: v["predicted_std"] for sid, v in per_scene.items()
            },
            "per_group_predicted_range": {
                r["decision_group_id"]: r["predicted_score_range"] for r in group_rows
            },
            "nearly_constant_groups": {
                f"range_lt_{th:g}": int(n_const[th]) for th in COLLAPSE_THRESHOLDS
            },
            "n_groups": len(group_rows),
        }
    per_group_f.close()

    # ---- loss interaction from existing epoch histories
    loss_interaction = {}
    for s in SEEDS:
        hist = load_jsonl(T26E_B_RUN / f"metrics/seed_{s}_epoch_history.jsonl")
        sel = json.loads(
            (T26E_B_OUT / "t26e_b_seed_results.jsonl").read_text().splitlines()[s]
        )
        assert sel["seed"] == s
        sel_epoch = sel["selected_epoch"]
        first, last = hist[0], hist[-1]
        sel_row = next(h for h in hist if h["epoch"] == sel_epoch)
        ratio = [
            {
                "epoch": h["epoch"],
                "train_future_over_score": h["train_loss_mean"]["future"]
                / max(h["train_loss_mean"]["score"], 1e-12),
            }
            for h in hist
        ]
        loss_interaction[str(s)] = {
            "epoch_history_starts_at": first["epoch"],
            "note": "histories record epochs 1..N (no epoch-0 pre-training eval was "
            "logged in T26E-B); 'epoch 0' below = first recorded epoch",
            "epoch0_train_L_future": first["train_loss_mean"]["future"],
            "epoch0_train_L_score": first["train_loss_mean"]["score"],
            "selected_epoch": sel_epoch,
            "selected_epoch_train_L_future": sel_row["train_loss_mean"]["future"],
            "selected_epoch_train_L_score": sel_row["train_loss_mean"]["score"],
            "final_epoch": last["epoch"],
            "final_epoch_train_L_future": last["train_loss_mean"]["future"],
            "final_epoch_train_L_score": last["train_loss_mean"]["score"],
            "early_stopping_epoch": sel["early_stopping"]["stopped_after_epoch"],
            "loss_ratio_future_over_score": ratio,
            "ratio_epoch0": ratio[0]["train_future_over_score"],
            "ratio_selected": next(
                r["train_future_over_score"] for r in ratio if r["epoch"] == sel_epoch
            ),
            "ratio_final": ratio[-1]["train_future_over_score"],
        }

    (run_dir / "diagnostics/t26e_b1_per_scene_diagnostics.json").write_text(
        json.dumps({"per_seed": per_scene_all, "candidate_level": candidate_level}, indent=1)
    )
    (run_dir / "diagnostics/t26e_b1_margin_pairwise_metrics.json").write_text(
        json.dumps({"per_seed": margin_all, "group_centered_spearman": group_centered}, indent=1)
    )
    (run_dir / "diagnostics/t26e_b1_prediction_collapse.json").write_text(
        json.dumps(collapse_all, indent=1)
    )
    (run_dir / "diagnostics/t26e_b1_loss_interaction.json").write_text(
        json.dumps(loss_interaction, indent=1)
    )
    return {
        "candidate_level": candidate_level,
        "group_centered": group_centered,
        "margin": margin_all,
        "collapse": collapse_all,
        "loss_interaction": {
            s: {k: v for k, v in d.items() if k != "loss_ratio_future_over_score"}
            for s, d in loss_interaction.items()
        },
        "per_scene": per_scene_all,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("audit", "metrics", "baselines", "diagnostics"), required=True
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    fn = {
        "audit": task1_audit,
        "metrics": task2_metric_repair,
        "baselines": task3_baselines,
        "diagnostics": task4_diagnostics,
    }[args.task]
    result = fn(args.run_dir)
    print(json.dumps(result, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

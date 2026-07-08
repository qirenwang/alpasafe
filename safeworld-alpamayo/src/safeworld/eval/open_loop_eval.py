from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from safeworld.auditor.gate import gate_action
from safeworld.data.loaders import (
    load_samples,
    load_targets,
    proposals_by_sample,
    topk_proposals_by_sample,
)
from safeworld.data.schema import CandidateEvaluation, CandidateSet
from safeworld.eval.metrics import classification_metrics
from safeworld.reasoning.rawc import compute_rawc
from safeworld.selection import SelectionConfig, all_selectors
from safeworld.utils.io import load_yaml, project_path, write_jsonl, write_markdown
from safeworld.world_model.model_v1 import EngineeredRiskModel


def _load_model(path: str) -> EngineeredRiskModel:
    payload = json.loads(project_path(path).read_text(encoding="utf-8"))
    return EngineeredRiskModel.from_dict(payload["model"])


def _predicate_flag(target: object, names: set[str]) -> float:
    for result in target.metadata.get("predicate_results", []):
        if result["name"] in names and not bool(result["is_safe"]):
            return 1.0
    return 0.0


def _markdown_table(title: str, metrics: dict[str, dict[str, float]]) -> str:
    lines = [f"## {title}", "", "| Method | FSR | Unsafe recall | Safe precision | AUROC | AUPRC | Brier | ECE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, row in metrics.items():
        lines.append(
            f"| {method} | {row['false_safe_rate']:.4f} | {row['unsafe_recall']:.4f} | "
            f"{row['safe_precision']:.4f} | {row['auroc']:.4f} | {row['auprc']:.4f} | "
            f"{row['brier']:.4f} | {row['ece']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_open_loop(config_path: str) -> dict[str, str]:
    config = load_yaml(config_path)
    inputs = config.get("inputs", {})
    outputs = config.get("outputs", {})
    gate_cfg = config.get("gate", {})
    samples = load_samples(inputs.get("index_path", "outputs/index/sample_index_mined.jsonl"))
    proposal_map = proposals_by_sample(inputs.get("proposal_path", "outputs/proposals/alpamayo_sample.jsonl"))
    targets = load_targets(inputs.get("target_path", "outputs/world_targets/sample_targets.jsonl"))
    model = _load_model(inputs.get("model_path", "outputs/models/world_model_v1.json"))
    sample_map = {sample.scene_id: sample for sample in samples}
    alpamayo_targets = [target for target in targets if target.candidate_id == "tau_alpamayo"]

    y_true = np.asarray([float(target.outcome_labels.get("unsafe", 0.0)) for target in alpamayo_targets])
    methods: dict[str, np.ndarray] = {}
    methods["Alpamayo alone"] = np.zeros(len(alpamayo_targets), dtype=float)
    methods["Alpamayo + kinematic checker"] = np.asarray(
        [
            _predicate_flag(
                target,
                {
                    "kinematic_acceleration_margin",
                    "kinematic_jerk_margin",
                    "curvature_margin",
                    "offroad_margin",
                    "comfort_score",
                },
            )
            for target in alpamayo_targets
        ],
        dtype=float,
    )
    methods["Alpamayo + TTC/RSS-like checker"] = np.asarray(
        [
            _predicate_flag(
                target,
                {"minimum_distance_margin", "ttc_margin", "rss_like_longitudinal_margin"},
            )
            for target in alpamayo_targets
        ],
        dtype=float,
    )
    rawc_scores: list[float] = []
    contradictions: list[tuple[str, list[str], float]] = []
    for target in alpamayo_targets:
        sample = sample_map[target.sample_id]
        proposal = proposal_map[target.sample_id]
        rawc = compute_rawc(
            proposal.reasoning_text,
            proposal.trajectory,
            sample.object_tracks,
            sample.map_context,
            sample.scenario_tags,
        )
        rawc_scores.append(rawc.score)
        contradictions.append((target.sample_id, rawc.contradictions, rawc.score))
    methods["Alpamayo + RAWC"] = 1.0 - np.asarray(rawc_scores, dtype=float)
    methods["Alpamayo + SafeWorld V1"] = np.asarray([model.predict_target(target) for target in alpamayo_targets])

    threshold = float(gate_cfg.get("risk_threshold", 0.45))
    method_metrics = {
        method: classification_metrics(y_true, score, threshold if "SafeWorld" in method or "RAWC" in method else 0.5)
        for method, score in methods.items()
    }

    gate_rows = []
    for target, risk, rawc_score in zip(alpamayo_targets, methods["Alpamayo + SafeWorld V1"], rawc_scores):
        min_margin = float(target.outcome_labels.get("min_margin", 0.0))
        decision = gate_action(
            target.candidate_id,
            float(risk),
            min_margin,
            float(rawc_score),
            risk_threshold=threshold,
            margin_threshold=float(gate_cfg.get("margin_threshold", 0.0)),
            rawc_threshold=float(gate_cfg.get("rawc_threshold", 0.65)),
        )
        gate_rows.append(decision.to_dict())

    e1 = "\n".join(
        [
            "# E1 - Open-loop trajectory safety auditing",
            "",
            f"Samples: {len(alpamayo_targets)}",
            f"Unsafe positives: {int(np.sum(y_true))}",
            "",
            _markdown_table("Safety auditor comparison", method_metrics),
            "Gate decisions:",
            "",
            "| Sample | Accepted | Reason | Risk | Min margin | RAWC | Selected |",
            "|---|---:|---|---:|---:|---:|---|",
            *[
                f"| {target.sample_id} | {row['accepted']} | {row['primary_reason']} | "
                f"{row['risk_probability']:.4f} | {row['minimum_margin']:.4f} | "
                f"{row['rawc_score']:.4f} | {row['selected_candidate_id']} |"
                for target, row in zip(alpamayo_targets, gate_rows)
            ],
            "",
            "Known limitations: actual unsafe labels are deterministic predicate proxies in dry-run mode.",
            "",
            "Reproduction command:",
            "`PYTHONPATH=src python -m safeworld.eval.open_loop_eval --config configs/eval_open_loop.yaml`",
            "",
        ]
    )
    write_markdown(outputs.get("e1_report_path", "outputs/reports/e1_open_loop_safety.md"), e1)

    safe_rawc = [score for score, label in zip(rawc_scores, y_true) if label < 0.5]
    unsafe_rawc = [score for score, label in zip(rawc_scores, y_true) if label >= 0.5]
    e2 = "\n".join(
        [
            "# E2 - Reasoning-Action-World Consistency",
            "",
            f"Mean RAWC safe: {np.mean(safe_rawc) if safe_rawc else 0.0:.4f}",
            f"Mean RAWC unsafe: {np.mean(unsafe_rawc) if unsafe_rawc else 0.0:.4f}",
            f"Contradiction rate: {np.mean([len(items) > 0 for _, items, _ in contradictions]):.4f}",
            "",
            "| Sample | RAWC | Contradictions |",
            "|---|---:|---|",
            *[
                f"| {sample_id} | {score:.4f} | {', '.join(items) if items else 'none'} |"
                for sample_id, items, score in contradictions
            ],
            "",
            "Ablation hooks implemented: remove_reasoning, remove_trajectory, remove_world_target.",
            "",
        ]
    )
    write_markdown(outputs.get("e2_report_path", "outputs/reports/e2_rawc.md"), e2)

    by_candidate: dict[str, list[float]] = defaultdict(list)
    by_candidate_label: dict[str, list[float]] = defaultdict(list)
    for target in targets:
        by_candidate[target.candidate_id].append(model.predict_target(target))
        by_candidate_label[target.candidate_id].append(float(target.outcome_labels.get("unsafe", 0.0)))
    e4_lines = [
        "# E4 - Counterfactual action consequence",
        "",
        "| Candidate | Mean predicted risk | Unsafe proxy rate |",
        "|---|---:|---:|",
    ]
    for candidate_id in sorted(by_candidate):
        e4_lines.append(
            f"| {candidate_id} | {np.mean(by_candidate[candidate_id]):.4f} | {np.mean(by_candidate_label[candidate_id]):.4f} |"
        )
    paired_variance = []
    for sample in samples:
        sample_targets = [target for target in targets if target.sample_id == sample.scene_id]
        risks = [model.predict_target(target) for target in sample_targets]
        if risks:
            paired_variance.append(float(np.var(risks)))
    e4_lines.extend(
        [
            "",
            f"Mean counterfactual risk variance: {np.mean(paired_variance) if paired_variance else 0.0:.4f}",
            "Known limitations: oracle counterfactual risk is a rule-based proxy in dry-run mode.",
            "",
        ]
    )
    write_markdown(outputs.get("e4_report_path", "outputs/reports/e4_counterfactuals.md"), "\n".join(e4_lines))

    tag_rows: dict[str, list[int]] = defaultdict(list)
    tag_scores: dict[str, list[float]] = defaultdict(list)
    for target, label, score in zip(alpamayo_targets, y_true, methods["Alpamayo + SafeWorld V1"]):
        sample = sample_map[target.sample_id]
        for tag in sample.scenario_tags:
            tag_rows[tag].append(int(label))
            tag_scores[tag].append(float(score))
    e6_lines = [
        "# E6 - Long-tail/OOD evaluation",
        "",
        "| Tag | Samples | Unsafe rate | SafeWorld AUROC | SafeWorld FSR |",
        "|---|---:|---:|---:|---:|",
    ]
    for tag in sorted(tag_rows):
        labels = np.asarray(tag_rows[tag], dtype=float)
        scores = np.asarray(tag_scores[tag], dtype=float)
        metrics = classification_metrics(labels, scores, threshold)
        e6_lines.append(
            f"| {tag} | {len(labels)} | {np.mean(labels):.4f} | {metrics['auroc']:.4f} | {metrics['false_safe_rate']:.4f} |"
        )
    e6_lines.append("")
    write_markdown(outputs.get("e6_report_path", "outputs/reports/e6_long_tail.md"), "\n".join(e6_lines))

    return {
        "e1": outputs.get("e1_report_path", "outputs/reports/e1_open_loop_safety.md"),
        "e2": outputs.get("e2_report_path", "outputs/reports/e2_rawc.md"),
        "e4": outputs.get("e4_report_path", "outputs/reports/e4_counterfactuals.md"),
        "e6": outputs.get("e6_report_path", "outputs/reports/e6_long_tail.md"),
    }


def _comfort_cost(target: object) -> float:
    for result in target.metadata.get("predicate_results", []):
        if result["name"] == "comfort_score":
            return max(0.0, -float(result["margin"]))
    return 0.0


def _progress_m(target: object) -> float:
    for result in target.metadata.get("predicate_results", []):
        if result["name"] == "progress_score":
            return float(result["details"].get("progress_m", 0.0))
    return 0.0


def build_candidate_evaluations(
    sample: object,
    candidate_set: CandidateSet,
    targets_by_candidate: dict[str, object],
    model: EngineeredRiskModel,
) -> list[CandidateEvaluation]:
    evaluations: list[CandidateEvaluation] = []
    for candidate in candidate_set.candidates:
        target = targets_by_candidate.get(candidate.candidate_id)
        if target is None:
            continue
        rawc = compute_rawc(
            candidate.reasoning_trace or "",
            candidate.trajectory,
            sample.object_tracks,
            sample.map_context,
            sample.scenario_tags,
        )
        evaluations.append(
            CandidateEvaluation(
                sample_id=sample.scene_id,
                candidate_id=candidate.candidate_id,
                candidate_source=candidate.candidate_source,
                candidate_rank=candidate.candidate_rank,
                predicted_risk=round(model.predict_target(target), 6),
                progress_score=round(_progress_m(target), 4),
                comfort_cost=round(_comfort_cost(target), 4),
                rawc_score=rawc.score,
                uncertainty=None,
                model_score=candidate.model_score,
                label_source="proxy_rule",
                outcome_labels=dict(target.outcome_labels),
                metadata={"rule_unsafe": int(target.outcome_labels.get("unsafe", 0))},
            )
        )
    return evaluations


def run_topk_selection(config_path: str, k: int | None = None) -> dict[str, str]:
    config = load_yaml(config_path)
    inputs = config.get("inputs", {})
    outputs = config.get("outputs", {})
    selection_cfg = SelectionConfig.from_dict(config.get("selection", {}))
    k = int(k or config.get("topk", {}).get("default_k", 10))
    proposal_path = inputs.get(
        "topk_proposal_path_template", "outputs/proposals/alpamayo_topk_k{k}.jsonl"
    ).format(k=k)
    target_path = inputs.get(
        "topk_target_path_template", "outputs/world_targets/topk_targets_k{k}.jsonl"
    ).format(k=k)
    log_path = outputs.get("topk_selection_log_template", "outputs/topk/selection_log_k{k}.jsonl").format(k=k)
    report_path = outputs.get("topk_report_template", "outputs/reports/topk_selection_k{k}.md").format(k=k)
    samples = load_samples(inputs.get("index_path", "outputs/index/sample_index_mined.jsonl"))
    proposal_map = topk_proposals_by_sample(proposal_path)
    if not proposal_map:
        raise RuntimeError(f"no top-K proposals found at {proposal_path}")
    targets = load_targets(target_path)
    if not targets:
        raise RuntimeError(f"no top-K world targets found at {target_path}")
    model = _load_model(inputs.get("model_path", "outputs/models/world_model_v1.json"))
    targets_by_sample: dict[str, dict[str, object]] = defaultdict(dict)
    for target in targets:
        targets_by_sample[target.sample_id][target.candidate_id] = target

    selectors = all_selectors()
    log_rows: list[dict[str, object]] = []
    per_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    unsafe_by_candidate: dict[str, dict[str, float]] = defaultdict(dict)
    for sample in samples:
        proposal = proposal_map.get(sample.scene_id)
        if proposal is None:
            continue
        candidate_set = CandidateSet.from_proposal(proposal)
        evaluations = build_candidate_evaluations(
            sample, candidate_set, targets_by_sample[sample.scene_id], model
        )
        if not evaluations:
            continue
        for evaluation in evaluations:
            unsafe_by_candidate[sample.scene_id][evaluation.candidate_id] = float(
                evaluation.outcome_labels.get("unsafe", 0.0)
            )
        for selector in selectors:
            decision = selector.select(sample.scene_id, proposal.k_returned, evaluations, selection_cfg)
            row = decision.to_dict()
            row["label_source"] = "proxy_rule"
            log_rows.append(row)
            per_method[selector.name].append(row)
    write_jsonl(log_path, log_rows)

    lines = [
        f"# Top-K selection protocol (K={k})",
        "",
        "All metrics below are computed from `proxy_rule` labels.",
        "**Dry-run proxy metric only; not scientific evidence.**",
        "",
        f"Samples: {len(per_method.get('top1', []))}",
        f"K: {k}",
        f"Selection log: `{log_path}`",
        "",
        "| Method | Fallback rate | Mean predicted risk | Selected proxy-unsafe rate | Mean progress | Mean comfort cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, rows in per_method.items():
        fallback_rate = np.mean([row["selected_candidate_source"] == "fallback" for row in rows])
        mean_risk = np.mean([row["predicted_risk"] or 0.0 for row in rows])
        unsafe_rate = np.mean(
            [unsafe_by_candidate[row["sample_id"]].get(row["selected_candidate_id"], 1.0) for row in rows]
        )
        mean_progress = np.mean([row["progress_score"] or 0.0 for row in rows])
        mean_comfort = np.mean([row["comfort_cost"] or 0.0 for row in rows])
        lines.append(
            f"| {method} | {fallback_rate:.4f} | {mean_risk:.4f} | {unsafe_rate:.4f} | "
            f"{mean_progress:.4f} | {mean_comfort:.4f} |"
        )
    lines.extend(
        [
            "",
            "Selection objective: `argmin [ safety_risk - lambda_progress * progress_score "
            "+ beta_comfort * comfort_cost + gamma_uncertainty * uncertainty ]`.",
            "If all Alpamayo-native candidates are unsafe, the selector falls back to "
            f"`{selection_cfg.fallback_candidate}` (configurable).",
            "",
            "`oracle_best_in_k` uses proxy rule labels in dry-run mode and is marked "
            "`proxy_oracle_only` in its decision reasons.",
            "",
            "Reproduction command:",
            "`PYTHONPATH=src python -m safeworld.eval.open_loop_eval --config configs/eval_open_loop.yaml --topk`",
            "",
        ]
    )
    write_markdown(report_path, "\n".join(lines))
    return {"selection_log": log_path, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_open_loop.yaml")
    parser.add_argument("--topk", action="store_true", help="run the top-K selection protocol")
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()
    if args.topk:
        paths = run_topk_selection(args.config, args.k)
    else:
        paths = run_open_loop(args.config)
    print("wrote reports: " + ", ".join(paths.values()))


if __name__ == "__main__":
    main()


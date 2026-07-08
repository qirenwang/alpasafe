from __future__ import annotations

import argparse
import json

import numpy as np

from safeworld.data.loaders import load_targets
from safeworld.eval.metrics import auprc, auroc, brier_score, expected_calibration_error
from safeworld.utils.io import ensure_parent, load_yaml, write_markdown
from safeworld.utils.seed import set_seed
from safeworld.world_model.model_v1 import fit_risk_model


def train_model(config_path: str) -> str:
    config = load_yaml(config_path)
    set_seed(int(config.get("run", {}).get("seed", 7)))
    target_path = config.get("outputs", {}).get("target_path", "outputs/world_targets/sample_targets.jsonl")
    model_path = config.get("outputs", {}).get("model_path", "outputs/models/world_model_v1.json")
    report_path = config.get("outputs", {}).get("e3_report_path", "outputs/reports/e3_world_model_v1.md")
    targets = load_targets(target_path)
    model_config = config.get("model", {})
    model, train_metrics = fit_risk_model(
        targets,
        learning_rate=float(model_config.get("learning_rate", 0.08)),
        epochs=int(model_config.get("epochs", 250)),
        l2=float(model_config.get("l2", 0.001)),
    )
    y_true = np.asarray([float(target.outcome_labels.get("unsafe", 0.0)) for target in targets])
    y_prob = np.asarray([model.predict_target(target) for target in targets])
    metrics = {
        **train_metrics,
        "auroc": auroc(y_true, y_prob),
        "auprc": auprc(y_true, y_prob),
        "brier": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    resolved = ensure_parent(model_path)
    resolved.write_text(json.dumps({"model": model.to_dict(), "metrics": metrics}, indent=2), encoding="utf-8")
    report = "\n".join(
        [
            "# E3 - Action-conditioned world model",
            "",
            "Model: engineered logistic risk model over candidate trajectory, reasoning keywords, and scenario tags.",
            f"Training targets: {len(targets)}",
            f"Model artifact: `{model_path}`",
            "",
            "| Metric | Value |",
            "|---|---:|",
            *[f"| {key} | {value:.4f} |" for key, value in metrics.items()],
            "",
            "Action conditioning: candidate-id and trajectory-derived speed/lateral/progress features are part of the feature vector.",
            "",
            "Known limitations: labels are deterministic predicate proxies in dry-run mode, not simulator oracle outcomes.",
            "",
            "Reproduction command:",
            "`PYTHONPATH=src python -m safeworld.world_model.train --config configs/world_model_v1.yaml`",
            "",
        ]
    )
    write_markdown(report_path, report)
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world_model_v1.yaml")
    args = parser.parse_args()
    model_path = train_model(args.config)
    print(f"wrote model: {model_path}")


if __name__ == "__main__":
    main()


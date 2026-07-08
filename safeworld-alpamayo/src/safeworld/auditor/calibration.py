from __future__ import annotations

import numpy as np

from safeworld.eval.metrics import brier_score, expected_calibration_error


def reliability_bins(y_true: list[float] | np.ndarray, y_prob: list[float] | np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    y = np.asarray(y_true, dtype=float)
    prob = np.asarray(y_prob, dtype=float)
    bins: list[dict[str, float]] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (prob >= low) & (prob < high if high < 1.0 else prob <= high)
        bins.append(
            {
                "low": float(low),
                "high": float(high),
                "count": float(np.sum(mask)),
                "confidence": float(np.mean(prob[mask])) if np.any(mask) else 0.0,
                "empirical_unsafe_rate": float(np.mean(y[mask])) if np.any(mask) else 0.0,
            }
        )
    return bins


def calibration_summary(y_true: list[float] | np.ndarray, y_prob: list[float] | np.ndarray) -> dict[str, float]:
    return {
        "brier": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }


def select_threshold_for_recall(
    y_true: list[float] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    target_recall: float = 0.8,
) -> float:
    y = np.asarray(y_true, dtype=float)
    prob = np.asarray(y_prob, dtype=float)
    for threshold in sorted(set(prob.tolist())):
        pred = prob >= threshold
        positives = y >= 0.5
        recall = float(np.sum(pred & positives) / max(np.sum(positives), 1))
        if recall >= target_recall:
            return float(threshold)
    return 0.5


from __future__ import annotations

import numpy as np


def _arrays(y_true: np.ndarray | list[float], y_score: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    if y.shape != s.shape:
        raise ValueError("y_true and y_score must have the same shape")
    return y, s


def false_safe_rate(y_true: np.ndarray | list[float], y_pred_unsafe: np.ndarray | list[float]) -> float:
    y, pred = _arrays(y_true, y_pred_unsafe)
    unsafe = y >= 0.5
    if not np.any(unsafe):
        return 0.0
    false_safe = unsafe & (pred < 0.5)
    return float(np.sum(false_safe) / np.sum(unsafe))


def unsafe_recall(y_true: np.ndarray | list[float], y_pred_unsafe: np.ndarray | list[float]) -> float:
    y, pred = _arrays(y_true, y_pred_unsafe)
    unsafe = y >= 0.5
    if not np.any(unsafe):
        return 0.0
    return float(np.sum(unsafe & (pred >= 0.5)) / np.sum(unsafe))


def safe_precision(y_true: np.ndarray | list[float], y_pred_unsafe: np.ndarray | list[float]) -> float:
    y, pred = _arrays(y_true, y_pred_unsafe)
    predicted_safe = pred < 0.5
    if not np.any(predicted_safe):
        return 0.0
    return float(np.sum((y < 0.5) & predicted_safe) / np.sum(predicted_safe))


def brier_score(y_true: np.ndarray | list[float], y_prob: np.ndarray | list[float]) -> float:
    y, prob = _arrays(y_true, y_prob)
    return float(np.mean((prob - y) ** 2))


def auroc(y_true: np.ndarray | list[float], y_score: np.ndarray | list[float]) -> float:
    y, score = _arrays(y_true, y_score)
    positives = score[y >= 0.5]
    negatives = score[y < 0.5]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    wins = 0.0
    total = len(positives) * len(negatives)
    for pos in positives:
        wins += float(np.sum(pos > negatives))
        wins += 0.5 * float(np.sum(pos == negatives))
    return float(wins / total)


def auprc(y_true: np.ndarray | list[float], y_score: np.ndarray | list[float]) -> float:
    y, score = _arrays(y_true, y_score)
    if not np.any(y >= 0.5):
        return 0.0
    order = np.argsort(-score)
    sorted_y = y[order]
    tp = np.cumsum(sorted_y >= 0.5)
    fp = np.cumsum(sorted_y < 0.5)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(float(np.sum(y >= 0.5)), 1.0)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapezoid(precision, recall))


def expected_calibration_error(
    y_true: np.ndarray | list[float],
    y_prob: np.ndarray | list[float],
    n_bins: int = 10,
) -> float:
    y, prob = _arrays(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (prob >= low) & (prob < high if high < 1.0 else prob <= high)
        if np.any(mask):
            conf = float(np.mean(prob[mask]))
            acc = float(np.mean(y[mask]))
            ece += float(np.sum(mask) / len(y)) * abs(conf - acc)
    return float(ece)


def classification_metrics(
    y_true: np.ndarray | list[float],
    y_score: np.ndarray | list[float],
    threshold: float,
) -> dict[str, float]:
    y, score = _arrays(y_true, y_score)
    pred = (score >= threshold).astype(float)
    return {
        "false_safe_rate": false_safe_rate(y, pred),
        "unsafe_recall": unsafe_recall(y, pred),
        "safe_precision": safe_precision(y, pred),
        "auroc": auroc(y, score),
        "auprc": auprc(y, score),
        "brier": brier_score(y, score),
        "ece": expected_calibration_error(y, score),
    }

from __future__ import annotations

import numpy as np


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-8
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return float(np.mean(-(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


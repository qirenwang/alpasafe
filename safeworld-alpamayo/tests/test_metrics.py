import numpy as np

from safeworld.eval.metrics import auprc, auroc, false_safe_rate, unsafe_recall


def test_basic_metrics() -> None:
    y = np.asarray([1, 1, 0, 0])
    score = np.asarray([0.9, 0.8, 0.2, 0.1])
    pred = score >= 0.5
    assert false_safe_rate(y, pred) == 0.0
    assert unsafe_recall(y, pred) == 1.0
    assert auroc(y, score) == 1.0
    assert auprc(y, score) > 0.9


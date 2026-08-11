"""Minimal scalar selector adapter for the T26E-B pilot.

Exposes the predicted official score as ``final_reward`` and applies the exact
documented RewardSelector ordering (selection/reward_selector.py:14-20):

1. highest ``final_reward``;
2. lower ``uncertainty`` — this pilot has NO uncertainty head, so ``uncertainty``
   is always ``None`` and is treated as +inf for every candidate (the selector's
   documented missing-uncertainty behavior); no values are fabricated;
3. lower ``candidate_rank`` (Alpamayo's own order; ``candidate_index`` metadata);
4. lexicographically smallest ``candidate_id``.

No NC/DAC/TTC/EP/Comfort components exist here and none are fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CandidateOfficialScorePrediction:
    """Scalar prediction contract: final_reward = predicted official score."""

    sample_id: str
    candidate_id: str
    final_reward: float
    uncertainty: None = None  # no uncertainty head in this pilot; never fabricated
    candidate_rank: int | None = None  # metadata only (candidate_index)


def _sort_key(pred: CandidateOfficialScorePrediction) -> tuple[float, float, float, str]:
    uncertainty = float("inf") if pred.uncertainty is None else float(pred.uncertainty)
    rank = float("inf") if pred.candidate_rank is None else float(pred.candidate_rank)
    return (-pred.final_reward, uncertainty, rank, pred.candidate_id)


def select_within_group(
    predictions: list[CandidateOfficialScorePrediction],
) -> CandidateOfficialScorePrediction:
    """Deterministic argmax with the documented RewardSelector tie-break chain."""
    if not predictions:
        raise ValueError("select_within_group requires at least one candidate prediction")
    return min(predictions, key=_sort_key)


def rank_within_group(
    predictions: list[CandidateOfficialScorePrediction],
) -> list[CandidateOfficialScorePrediction]:
    return sorted(predictions, key=_sort_key)

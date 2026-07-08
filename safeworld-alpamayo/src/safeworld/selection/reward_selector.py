"""SafeWorld v2 deterministic reward selector (method_spec_v2).

``RewardSelector`` implements ``i_star = argmax_i R_hat_i`` over **Alpamayo-native
candidates only** (decision #8: no fallback in the main selection set). It is a
different component from the :class:`~safeworld.reward.base.RewardHead`
(decision #7): the head predicts component/scalar rewards; the selector performs
deterministic selection.

This selector consumes *predicted* rewards
(:class:`~safeworld.reward.base.PredictedRewardComponents`). It must never be
handed evaluator-measured rewards; independent evaluation lives in
:mod:`safeworld.eval.evaluator_v2` (T23) and is reported separately.

Tie-breaking (documented & deterministic):

1. higher ``final_reward`` (R_hat_i) wins;
2. ties broken by *lower* predictive ``uncertainty`` (None treated as +inf);
3. remaining ties broken by *lower* ``candidate_rank`` (None treated as +inf),
   i.e. Alpamayo's own preference order;
4. final ties broken by lexicographically smallest ``candidate_id``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from safeworld.reward.base import PredictedRewardComponents

# This selector is a v2 component; it deliberately does not accept fallback.
NATIVE_SOURCE = "alpamayo_native"


@dataclass(slots=True)
class SelectionResult:
    """Result of a single deterministic argmax selection (see also T23 SelectionResultV2)."""

    sample_id: str
    k: int
    selected_candidate_id: str
    selected_final_reward: float
    selection_method: str = "reward_selector_argmax"
    ranking: list[str] = field(default_factory=list)
    tie_break_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rank_value(rank: int | None) -> float:
    return float(rank) if rank is not None else float("inf")


def _uncertainty_value(unc: float | None) -> float:
    return float(unc) if unc is not None else float("inf")


class RewardSelector:
    """Deterministic argmax-over-predicted-reward selector for native candidates."""

    name = "reward_selector"

    def select(
        self,
        candidates: list[Any],
        predicted_rewards: list[PredictedRewardComponents],
    ) -> SelectionResult:
        """Select i_star = argmax_i R_hat_i among Alpamayo-native candidates.

        Args:
            candidates: Alpamayo-native ``CandidateTrajectory``-like objects. Each
                must expose ``candidate_id``, ``candidate_source`` and
                ``candidate_rank``. Fallback candidates are rejected.
            predicted_rewards: predicted reward components, one per candidate,
                matched by ``candidate_id``.
        """
        if not candidates:
            raise ValueError("RewardSelector.select requires at least one candidate")

        rank_by_id: dict[str, int | None] = {}
        for cand in candidates:
            source = getattr(cand, "candidate_source", NATIVE_SOURCE)
            if source != NATIVE_SOURCE:
                raise ValueError(
                    "RewardSelector operates on Alpamayo-native candidates only; "
                    f"got candidate_source={source!r} (fallback is excluded from the "
                    "main selection set, decision #8)"
                )
            rank_by_id[cand.candidate_id] = getattr(cand, "candidate_rank", None)

        reward_by_id = {pr.candidate_id: pr for pr in predicted_rewards}
        missing = set(rank_by_id) - set(reward_by_id)
        if missing:
            raise ValueError(f"missing predicted rewards for candidates: {sorted(missing)}")

        def sort_key(cand_id: str) -> tuple[float, float, float, str]:
            pr = reward_by_id[cand_id]
            return (
                -float(pr.final_reward),
                _uncertainty_value(pr.uncertainty),
                _rank_value(rank_by_id[cand_id]),
                cand_id,
            )

        ordered = sorted(rank_by_id, key=sort_key)
        best_id = ordered[0]
        best = reward_by_id[best_id]

        tie_break_path = _explain_tie_break(best_id, ordered, reward_by_id, rank_by_id)

        return SelectionResult(
            sample_id=best.sample_id,
            k=len(rank_by_id),
            selected_candidate_id=best_id,
            selected_final_reward=float(best.final_reward),
            ranking=ordered,
            tie_break_path=tie_break_path,
            metadata={
                "candidate_source": NATIVE_SOURCE,
                "selected_components": best.components(),
                "selected_uncertainty": best.uncertainty,
            },
        )


def _explain_tie_break(
    best_id: str,
    ordered: list[str],
    reward_by_id: dict[str, PredictedRewardComponents],
    rank_by_id: dict[str, int | None],
) -> list[str]:
    """Record which tie-break levels were actually engaged for transparency."""
    if len(ordered) < 2:
        return ["unique_single_candidate"]
    first, second = reward_by_id[best_id], reward_by_id[ordered[1]]
    path: list[str] = []
    if first.final_reward != second.final_reward:
        path.append("final_reward")
        return path
    path.append("final_reward_tie")
    if _uncertainty_value(first.uncertainty) != _uncertainty_value(second.uncertainty):
        path.append("uncertainty")
        return path
    path.append("uncertainty_tie")
    if _rank_value(rank_by_id[best_id]) != _rank_value(rank_by_id[ordered[1]]):
        path.append("candidate_rank")
        return path
    path.append("candidate_rank_tie")
    path.append("candidate_id_lexicographic")
    return path

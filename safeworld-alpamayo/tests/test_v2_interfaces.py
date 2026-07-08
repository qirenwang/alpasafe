"""T21: v2 interface and reward-selector tests."""

import pytest

from safeworld.data.schema import CandidateTrajectory
from safeworld.reward.base import PredictedRewardComponents, RewardHead
from safeworld.selection import RewardSelector, SelectionResult
from safeworld.world_model.base import (
    ConsequenceWorldModel,
    FutureConsequencePrediction,
    ObservationHistory,
)


def _native(rank: int) -> CandidateTrajectory:
    return CandidateTrajectory(
        candidate_id=f"alpamayo_candidate_{rank:02d}",
        candidate_source="alpamayo_native",
        candidate_rank=rank,
        trajectory=[[0.1 * i, 0.0] for i in range(64)],
        reasoning_trace="keep lane",
    )


def _pred(cand_id: str, reward: float, unc: float | None = None) -> PredictedRewardComponents:
    return PredictedRewardComponents(
        sample_id="scene_0001",
        candidate_id=cand_id,
        nc=1.0,
        dac=1.0,
        ttc=1.0,
        ep=10.0,
        comfort=1.0,
        final_reward=reward,
        uncertainty=unc,
    )


def test_protocols_are_runtime_checkable() -> None:
    # A trivial structural implementation should satisfy the Protocols, and a
    # plain object should not. No fake/real world model is provided in this phase.
    class W:
        def predict_future(self, observation_history, candidate):  # noqa: D401, ANN001
            return FutureConsequencePrediction(
                sample_id="s", candidate_id="c", future_states=[[0.0, 0.0]], future_hz=10.0, horizon_s=6.4
            )

    class G:
        def predict_components(self, future_prediction, candidate):  # noqa: ANN001
            return _pred("c", 1.0)

    assert isinstance(W(), ConsequenceWorldModel)
    assert isinstance(G(), RewardHead)
    assert not isinstance(object(), ConsequenceWorldModel)
    assert not isinstance(object(), RewardHead)


def test_observation_history_default_frame() -> None:
    obs = ObservationHistory(
        sample_id="s", scene_id="sc", ego_history_states=[[0.0, 0.0]], history_hz=10.0
    )
    assert obs.history_frame == "ego_t0"


def test_reward_selector_argmax() -> None:
    cands = [_native(1), _native(2), _native(3)]
    preds = [
        _pred("alpamayo_candidate_01", 0.2),
        _pred("alpamayo_candidate_02", 0.9),
        _pred("alpamayo_candidate_03", 0.5),
    ]
    result = RewardSelector().select(cands, preds)
    assert isinstance(result, SelectionResult)
    assert result.selected_candidate_id == "alpamayo_candidate_02"
    assert result.k == 3
    assert result.ranking[0] == "alpamayo_candidate_02"
    assert result.tie_break_path == ["final_reward"]


def test_reward_selector_tie_break_uses_uncertainty_then_rank() -> None:
    cands = [_native(1), _native(2)]
    # Equal reward -> lower uncertainty wins.
    preds = [
        _pred("alpamayo_candidate_01", 0.7, unc=0.4),
        _pred("alpamayo_candidate_02", 0.7, unc=0.1),
    ]
    result = RewardSelector().select(cands, preds)
    assert result.selected_candidate_id == "alpamayo_candidate_02"
    assert result.tie_break_path == ["final_reward_tie", "uncertainty"]

    # Equal reward and uncertainty -> lower candidate_rank wins.
    preds2 = [
        _pred("alpamayo_candidate_01", 0.7, unc=0.2),
        _pred("alpamayo_candidate_02", 0.7, unc=0.2),
    ]
    result2 = RewardSelector().select(cands, preds2)
    assert result2.selected_candidate_id == "alpamayo_candidate_01"
    assert result2.tie_break_path == ["final_reward_tie", "uncertainty_tie", "candidate_rank"]


def test_reward_selector_rejects_fallback() -> None:
    fallback = CandidateTrajectory(
        candidate_id="fallback_stop",
        candidate_source="fallback",
        candidate_rank=None,
        trajectory=[[0.0, 0.0]],
        reasoning_trace="stop",
    )
    with pytest.raises(ValueError):
        RewardSelector().select([fallback], [_pred("fallback_stop", 1.0)])


def test_reward_selector_requires_matching_predictions() -> None:
    with pytest.raises(ValueError):
        RewardSelector().select([_native(1)], [])

"""T23: v2 schema, label-source, and evaluator-contract tests."""

import json

import pytest

from safeworld.data.schema_v2 import (
    CandidateRolloutRecord,
    CandidateRolloutRequest,
    EvaluationRewardComponents,
    FutureConsequenceTarget,
    SelectionResultV2,
    WeightProvenance,
    assert_real_label_source,
    validate_no_logged_label_duplication,
)
from safeworld.eval.evaluator_v2 import evaluate_record
from safeworld.reward.base import PredictedRewardComponents


def _record(candidate_id: str, label_source: str, future, **kw) -> CandidateRolloutRecord:
    base = dict(
        sample_id="scene_0001",
        scene_id="scene_0001",
        candidate_id=candidate_id,
        candidate_rank=1,
        candidate_source="alpamayo_native",
        reasoning_trace="keep lane",
        trajectory=[[0.1 * i, 0.0] for i in range(64)],
        trajectory_frame="ego_t0",
        trajectory_hz=10.0,
        rollout_backend="fixed_agent_replay",
        rollout_backend_version="0.0.0",
        label_source=label_source,
        rollout_success=True,
        failure_reason=None,
        future_ego_states=future,
        future_agent_states_or_occupancy=None,
        collision_events=[],
        drivable_area_events=[],
        ttc_curve=[3.0, 3.0],
        progress=12.0,
        comfort_components={"jerk": 0.1},
        metric_definition_version="v2_unfrozen",
    )
    base.update(kw)
    return CandidateRolloutRecord(**base)


# --- coordinate / horizon validation -------------------------------------------------


def test_request_requires_native_source_and_known_backend() -> None:
    req = CandidateRolloutRequest(
        sample_id="s",
        scene_id="sc",
        candidate_id="c",
        candidate_rank=1,
        candidate_source="alpamayo_native",
        reasoning_trace="r",
        trajectory=[[0.0, 0.0]],
        trajectory_frame="ego_t0",
        trajectory_hz=10.0,
        rollout_backend="alpasim",
        rollout_horizon_s=6.4,
        initial_state_ref="scene_0001@t0",
    )
    req.validate()
    req.candidate_source = "fallback"
    with pytest.raises(ValueError):
        req.validate()
    req.candidate_source = "alpamayo_native"
    req.rollout_backend = "not_a_backend"
    with pytest.raises(ValueError):
        req.validate()


def test_record_failure_requires_reason() -> None:
    with pytest.raises(ValueError):
        _record(
            "c", "fixed_agent_replay", [[0.0, 0.0]], rollout_success=False, failure_reason=None
        ).validate()


# --- label-source restrictions -------------------------------------------------------


def test_unknown_label_source_rejected() -> None:
    with pytest.raises(ValueError):
        _record("c", "made_up", [[0.0, 0.0]]).validate()


def test_assert_real_label_source_blocks_proxy() -> None:
    assert_real_label_source("alpasim_closed_loop")
    with pytest.raises(ValueError):
        assert_real_label_source("proxy_test_only")


def test_evaluator_blocks_proxy_in_real_mode() -> None:
    rec = _record("c", "proxy_test_only", [[0.0, 0.0]])
    with pytest.raises(ValueError):
        evaluate_record(rec)  # allow_proxy defaults False
    # Unit-test path may opt in explicitly.
    out = evaluate_record(rec, allow_proxy=True)
    assert out.label_source == "proxy_test_only"


def test_weight_provenance_rejects_proxy_and_requires_hashes() -> None:
    with pytest.raises(ValueError):
        WeightProvenance(
            component="RewardHead",
            initialization="random_seed_7",
            training_dataset_hash="abc",
            config_hash="def",
            checkpoint_hash="ghi",
            label_source="proxy_test_only",
            created_at="2026-06-20",
        ).validate()
    with pytest.raises(ValueError):
        WeightProvenance(
            component="RewardHead",
            initialization="random_seed_7",
            training_dataset_hash="",
            config_hash="def",
            checkpoint_hash="ghi",
            label_source="alpasim_closed_loop",
            created_at="2026-06-20",
        ).validate()


# --- separation of predicted vs evaluation reward ------------------------------------


def test_predicted_and_evaluation_reward_are_distinct_types() -> None:
    predicted = PredictedRewardComponents(
        sample_id="s", candidate_id="c", nc=1, dac=1, ttc=1, ep=1, comfort=1, final_reward=0.9
    )
    evaluated = EvaluationRewardComponents(
        sample_id="s",
        candidate_id="c",
        label_source="alpasim_closed_loop",
        nc=1,
        dac=1,
        ttc=1,
        ep=1,
        comfort=1,
        final_reward=0.9,
    )
    assert type(predicted) is not type(evaluated)
    assert not hasattr(predicted, "label_source")  # prediction carries no evaluator provenance
    assert hasattr(evaluated, "label_source")


def test_selection_result_v2_blocks_undisclosed_evaluation_basis() -> None:
    with pytest.raises(ValueError):
        SelectionResultV2(
            sample_id="s",
            k=2,
            selected_candidate_id="c",
            selection_method="argmax",
            reward_basis="evaluation",
            selected_predicted_reward=None,
            ranking=["c"],
            tie_break_path=[],
            diagnostic_only=False,
        ).validate()
    # Allowed when explicitly flagged as a diagnostic.
    SelectionResultV2(
        sample_id="s",
        k=2,
        selected_candidate_id="c",
        selection_method="best_achievable_in_sampled_set_by_evaluator",
        reward_basis="evaluation",
        selected_predicted_reward=None,
        ranking=["c"],
        tie_break_path=[],
        diagnostic_only=True,
    ).validate()


# --- leakage: no duplicated logged action --------------------------------------------


def test_logged_action_cannot_be_duplicated_across_candidates() -> None:
    same_future = [[float(i), 0.0] for i in range(64)]
    recs = [
        _record("c1", "logged_executed_action_only", same_future),
        _record("c2", "logged_executed_action_only", same_future),
    ]
    with pytest.raises(ValueError):
        validate_no_logged_label_duplication(recs)
    # Distinct futures are fine.
    recs[1] = _record("c2", "logged_executed_action_only", [[1.0, 1.0]])
    validate_no_logged_label_duplication(recs)


# --- JSONL round trip ----------------------------------------------------------------


def test_jsonl_round_trip() -> None:
    rec = _record("c", "alpasim_closed_loop", [[0.0, 0.0], [1.0, 0.0]])
    row = json.loads(json.dumps(rec.to_dict()))
    assert CandidateRolloutRecord.from_dict(row).candidate_id == "c"

    target = FutureConsequenceTarget(
        sample_id="s",
        candidate_id="c",
        label_source="nurec_candidate_replay",
        future_states=[[0.0, 0.0]],
        future_hz=10.0,
        horizon_s=6.4,
    )
    row2 = json.loads(json.dumps(target.to_dict()))
    assert FutureConsequenceTarget.from_dict(row2).horizon_s == 6.4

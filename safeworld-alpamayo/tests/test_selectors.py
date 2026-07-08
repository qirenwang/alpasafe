from safeworld.data.schema import CandidateEvaluation
from safeworld.selection import SelectionConfig, all_selectors
from safeworld.selection.selectors import (
    AlpamayoScoreSelector,
    OracleBestInKSelector,
    RandomTopKSelector,
    RAWCSelector,
    RuleCheckerSelector,
    SafeWorldSelector,
    Top1Selector,
)

REQUIRED_LOG_FIELDS = {
    "sample_id",
    "k",
    "selected_candidate_id",
    "selected_candidate_source",
    "selection_method",
    "predicted_risk",
    "progress_score",
    "comfort_cost",
    "rawc_score",
    "decision_reason",
}


def _evaluation(
    candidate_id: str,
    source: str,
    rank: int | None,
    risk: float,
    unsafe: int,
    rawc: float = 0.9,
    model_score: float | None = None,
    progress: float = 30.0,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        sample_id="scene_0001",
        candidate_id=candidate_id,
        candidate_source=source,
        candidate_rank=rank,
        predicted_risk=risk,
        progress_score=progress,
        comfort_cost=0.1,
        rawc_score=rawc,
        model_score=model_score,
        label_source="proxy_rule",
        outcome_labels={"unsafe": unsafe, "min_margin": 1.0 - unsafe},
        metadata={"rule_unsafe": unsafe},
    )


def _mixed_evaluations() -> list[CandidateEvaluation]:
    return [
        _evaluation("alpamayo_candidate_01", "alpamayo_native", 1, 0.8, 1, rawc=0.4, model_score=-0.0),
        _evaluation("alpamayo_candidate_02", "alpamayo_native", 2, 0.1, 0, rawc=0.9, model_score=-0.1),
        _evaluation("alpamayo_candidate_03", "alpamayo_native", 3, 0.3, 0, rawc=0.7, model_score=-0.2),
        _evaluation("fallback_stop", "fallback", None, 0.02, 0),
        _evaluation("fallback_slow", "fallback", None, 0.05, 0),
    ]


def _all_unsafe_evaluations() -> list[CandidateEvaluation]:
    return [
        _evaluation("alpamayo_candidate_01", "alpamayo_native", 1, 0.9, 1, rawc=0.2),
        _evaluation("alpamayo_candidate_02", "alpamayo_native", 2, 0.8, 1, rawc=0.3),
        _evaluation("fallback_stop", "fallback", None, 0.02, 0),
        _evaluation("fallback_slow", "fallback", None, 0.05, 0),
    ]


def test_all_selectors_produce_decision_with_required_fields() -> None:
    config = SelectionConfig()
    evaluations = _mixed_evaluations()
    for selector in all_selectors():
        decision = selector.select("scene_0001", 3, evaluations, config)
        row = decision.to_dict()
        assert REQUIRED_LOG_FIELDS.issubset(row.keys())
        assert row["selected_candidate_id"]
        assert row["decision_reason"]


def test_top1_selects_rank_one() -> None:
    decision = Top1Selector().select("scene_0001", 3, _mixed_evaluations(), SelectionConfig())
    assert decision.selected_candidate_id == "alpamayo_candidate_01"


def test_random_selector_is_deterministic_and_native() -> None:
    config = SelectionConfig(random_seed=7)
    first = RandomTopKSelector().select("scene_0001", 3, _mixed_evaluations(), config)
    second = RandomTopKSelector().select("scene_0001", 3, _mixed_evaluations(), config)
    assert first.selected_candidate_id == second.selected_candidate_id
    assert first.selected_candidate_source == "alpamayo_native"


def test_alpamayo_score_selector_uses_model_score() -> None:
    decision = AlpamayoScoreSelector().select("scene_0001", 3, _mixed_evaluations(), SelectionConfig())
    assert decision.selected_candidate_id == "alpamayo_candidate_01"
    no_scores = [
        _evaluation("alpamayo_candidate_01", "alpamayo_native", 1, 0.2, 0),
        _evaluation("alpamayo_candidate_02", "alpamayo_native", 2, 0.1, 0),
    ]
    decision = AlpamayoScoreSelector().select("scene_0001", 2, no_scores, SelectionConfig())
    assert decision.selected_candidate_id == "alpamayo_candidate_01"
    assert "rank" in decision.decision_reason


def test_rule_checker_skips_rule_violating_top1() -> None:
    decision = RuleCheckerSelector().select("scene_0001", 3, _mixed_evaluations(), SelectionConfig())
    assert decision.selected_candidate_id == "alpamayo_candidate_02"


def test_rawc_selector_picks_most_consistent() -> None:
    decision = RAWCSelector().select("scene_0001", 3, _mixed_evaluations(), SelectionConfig())
    assert decision.selected_candidate_id == "alpamayo_candidate_02"


def test_safeworld_selector_picks_lowest_objective() -> None:
    decision = SafeWorldSelector().select("scene_0001", 3, _mixed_evaluations(), SelectionConfig())
    assert decision.selected_candidate_id == "alpamayo_candidate_02"
    assert decision.selected_candidate_source == "alpamayo_native"


def test_safeworld_selector_falls_back_when_all_native_unsafe() -> None:
    config = SelectionConfig(fallback_candidate="fallback_stop")
    decision = SafeWorldSelector().select("scene_0001", 2, _all_unsafe_evaluations(), config)
    assert decision.selected_candidate_id == "fallback_stop"
    assert decision.selected_candidate_source == "fallback"
    config = SelectionConfig(fallback_candidate="fallback_slow")
    decision = SafeWorldSelector().select("scene_0001", 2, _all_unsafe_evaluations(), config)
    assert decision.selected_candidate_id == "fallback_slow"


def test_oracle_selector_marks_proxy_labels() -> None:
    decision = OracleBestInKSelector().select("scene_0001", 3, _mixed_evaluations(), SelectionConfig())
    assert decision.selected_candidate_id == "alpamayo_candidate_02"
    assert "proxy_oracle_only" in decision.decision_reason
    decision = OracleBestInKSelector().select("scene_0001", 2, _all_unsafe_evaluations(), SelectionConfig())
    assert decision.selected_candidate_source == "fallback"

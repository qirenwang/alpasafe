import json

import pytest

from safeworld.alpamayo.wrapper import AlpamayoWrapper
from safeworld.data.build_index import synthetic_samples
from safeworld.data.loaders import load_targets
from safeworld.data.schema import (
    CandidateEvaluation,
    CandidateSet,
    CandidateTrajectory,
    TopKAlpamayoProposal,
    WorldTarget,
)


def _native_candidate(rank: int = 1) -> CandidateTrajectory:
    return CandidateTrajectory(
        candidate_id=f"alpamayo_candidate_{rank:02d}",
        candidate_source="alpamayo_native",
        candidate_rank=rank,
        trajectory=[[0.1 * i, 0.0] for i in range(64)],
        reasoning_trace="keep lane",
        model_score=-0.1 * (rank - 1),
    )


def test_candidate_trajectory_round_trip() -> None:
    candidate = _native_candidate()
    row = candidate.to_dict()
    assert CandidateTrajectory.from_dict(row).candidate_id == "alpamayo_candidate_01"


def test_candidate_trajectory_rejects_unknown_source() -> None:
    candidate = _native_candidate()
    candidate.candidate_source = "made_up_source"
    with pytest.raises(ValueError):
        candidate.validate()


def test_topk_proposal_round_trip() -> None:
    proposal = TopKAlpamayoProposal(
        sample_id="scene_0001",
        k_requested=2,
        k_returned=2,
        alpamayo_model_name="nvidia/Alpamayo-1.5-10B",
        alpamayo_model_version="1.5-10B",
        inference_mode="dry_run",
        latency_ms=1.0,
        gpu_memory_mb=None,
        candidates=[_native_candidate(1), _native_candidate(2)],
        fallback_candidates=[
            CandidateTrajectory(
                candidate_id="fallback_stop",
                candidate_source="fallback",
                candidate_rank=None,
                trajectory=[[0.0, 0.0], [0.1, 0.0]],
                reasoning_trace="stop",
            )
        ],
    )
    row = json.loads(json.dumps(proposal.to_dict()))
    restored = TopKAlpamayoProposal.from_dict(row)
    assert restored.k_returned == 2
    assert restored.candidates[1].candidate_rank == 2
    assert restored.fallback_candidates[0].candidate_source == "fallback"


def test_topk_proposal_k_returned_must_match() -> None:
    proposal = TopKAlpamayoProposal(
        sample_id="scene_0001",
        k_requested=2,
        k_returned=3,
        alpamayo_model_name="m",
        alpamayo_model_version="v",
        inference_mode="dry_run",
        latency_ms=None,
        gpu_memory_mb=None,
        candidates=[_native_candidate(1)],
    )
    with pytest.raises(ValueError):
        proposal.validate()


def test_candidate_set_from_proposal_separates_sources() -> None:
    sample = synthetic_samples(1)[0]
    proposal = AlpamayoWrapper("dry", dry_run=True).run_topk(sample, 5)
    candidate_set = CandidateSet.from_proposal(proposal)
    assert len(candidate_set.native_candidates()) == 5
    assert {c.candidate_id for c in candidate_set.fallback_candidates()} == {
        "fallback_stop",
        "fallback_slow",
        "fallback_conservative_yield",
    }
    assert candidate_set.get("alpamayo_candidate_03").candidate_rank == 3


def test_candidate_evaluation_round_trip() -> None:
    evaluation = CandidateEvaluation(
        sample_id="scene_0001",
        candidate_id="alpamayo_candidate_01",
        candidate_source="alpamayo_native",
        candidate_rank=1,
        predicted_risk=0.2,
        progress_score=30.0,
        comfort_cost=0.0,
        rawc_score=0.9,
        label_source="proxy_rule",
        outcome_labels={"unsafe": 0},
    )
    assert CandidateEvaluation.from_dict(evaluation.to_dict()).predicted_risk == 0.2


def test_legacy_targets_marked_on_load(tmp_path) -> None:
    target = WorldTarget(
        sample_id="scene_0001",
        candidate_id="tau_alpamayo",
        candidate_trajectory=[[0.0, 0.0], [0.1, 0.0]],
        future_occupancy=None,
        future_min_distance=None,
        future_ttc=None,
        future_safety_margins={},
        outcome_labels={"unsafe": 0},
        metadata={},
    )
    path = tmp_path / "legacy_targets.jsonl"
    path.write_text(json.dumps(target.to_dict()) + "\n", encoding="utf-8")
    loaded = load_targets(str(path))
    assert loaded[0].metadata["candidate_source"] == "legacy_counterfactual_dry_run"

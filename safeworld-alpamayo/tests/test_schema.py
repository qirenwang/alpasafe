from safeworld.data.build_index import synthetic_samples
from safeworld.data.schema import AlpamayoProposal, DrivingSample, WorldTarget


def test_schema_round_trip() -> None:
    sample = synthetic_samples(1)[0]
    row = sample.to_dict()
    assert DrivingSample.from_dict(row).scene_id == sample.scene_id

    proposal = AlpamayoProposal(
        sample_id=sample.scene_id,
        model_name="dry",
        prompt="prompt",
        reasoning_text="keep lane",
        trajectory=sample.future_ego_trajectory or [[0.0, 0.0]],
        inference_latency_ms=1.0,
        raw_output_path=None,
        metadata={},
    )
    assert AlpamayoProposal.from_dict(proposal.to_dict()).sample_id == sample.scene_id

    target = WorldTarget(
        sample_id=sample.scene_id,
        candidate_id="tau_alpamayo",
        candidate_trajectory=proposal.trajectory,
        future_occupancy=None,
        future_min_distance=None,
        future_ttc=None,
        future_safety_margins={},
        outcome_labels={"unsafe": 0},
        metadata={},
    )
    assert WorldTarget.from_dict(target.to_dict()).candidate_id == "tau_alpamayo"


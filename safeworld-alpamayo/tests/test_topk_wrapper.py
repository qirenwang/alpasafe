import pytest

from safeworld.alpamayo.wrapper import AlpamayoWrapper
from safeworld.data.build_index import synthetic_samples


@pytest.mark.parametrize("k", [1, 2, 5, 10])
def test_dry_run_topk_returns_k_native_candidates(k: int) -> None:
    sample = synthetic_samples(2)[1]
    proposal = AlpamayoWrapper("nvidia/Alpamayo-1.5-10B", dry_run=True).run_topk(sample, k)
    assert proposal.k_requested == k
    assert proposal.k_returned == k
    assert proposal.inference_mode == "dry_run"
    assert proposal.latency_ms is not None
    assert [c.candidate_id for c in proposal.candidates] == [
        f"alpamayo_candidate_{rank:02d}" for rank in range(1, k + 1)
    ]
    assert all(c.candidate_source == "alpamayo_native" for c in proposal.candidates)
    assert [c.candidate_rank for c in proposal.candidates] == list(range(1, k + 1))
    assert all(len(c.trajectory) == 64 for c in proposal.candidates)
    assert all(c.reasoning_trace for c in proposal.candidates)


def test_fallback_candidates_kept_separate() -> None:
    sample = synthetic_samples(1)[0]
    proposal = AlpamayoWrapper("dry", dry_run=True).run_topk(sample, 2)
    fallback_ids = [c.candidate_id for c in proposal.fallback_candidates]
    assert fallback_ids == ["fallback_stop", "fallback_slow", "fallback_conservative_yield"]
    assert all(c.candidate_source == "fallback" for c in proposal.fallback_candidates)
    native_ids = {c.candidate_id for c in proposal.candidates}
    assert not native_ids.intersection(fallback_ids)


def test_topk_is_deterministic_per_sample() -> None:
    sample = synthetic_samples(1)[0]
    wrapper = AlpamayoWrapper("dry", dry_run=True)
    first = wrapper.run_topk(sample, 5)
    second = wrapper.run_topk(sample, 5)
    assert [c.trajectory for c in first.candidates] == [c.trajectory for c in second.candidates]
    assert [c.model_score for c in first.candidates] == [c.model_score for c in second.candidates]


def test_unsupported_k_rejected() -> None:
    sample = synthetic_samples(1)[0]
    with pytest.raises(ValueError):
        AlpamayoWrapper("dry", dry_run=True).run_topk(sample, 7)


def test_real_inference_requires_explicit_configuration() -> None:
    sample = synthetic_samples(1)[0]
    with pytest.raises(RuntimeError):
        AlpamayoWrapper("dry", dry_run=False).run_topk(sample, 1)
    with pytest.raises(RuntimeError):
        AlpamayoWrapper("dry", dry_run=True, use_real_alpamayo=True).run_topk(sample, 1)

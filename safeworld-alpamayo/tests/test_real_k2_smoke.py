"""T24 retry: tests for the real K=2 native-candidate smoke artifacts and plumbing.

These tests use the REAL cached candidates produced this run (no synthetic data). They are skipped
only if the artifact is absent, so the suite stays green on a fresh checkout.
"""

import csv
import dataclasses
import json
import os

import pytest

from safeworld.data.schema_v2 import (
    PROXY_LABEL_SOURCE,
    assert_real_label_source,
)
from safeworld.real_smoke.candidate_io import (
    CachedAlpamayoCandidate,
    assert_distinct_trajectories,
    assert_native_only,
    load_cached_candidates,
    to_rollout_request,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_K2 = os.path.join(_ROOT, "outputs", "real_smoke", "alpamayo_raw_k2.jsonl")
MANIFEST = os.path.join(_ROOT, "outputs", "real_smoke", "k2_scene_manifest.json")
METRIC_CSV = os.path.join(_ROOT, "outputs", "audits", "metric_availability_matrix_after_install.csv")

_have_raw = pytest.mark.skipif(not os.path.exists(RAW_K2), reason="real K=2 artifact not present")


@_have_raw
def test_exactly_two_native_candidates():
    cands = load_cached_candidates(RAW_K2)
    assert len(cands) == 2
    assert_native_only(cands)
    for c in cands:
        assert c.candidate_source == "alpamayo_native"


@_have_raw
def test_candidate_id_separation():
    cands = load_cached_candidates(RAW_K2)
    ids = [c.candidate_id for c in cands]
    assert len(set(ids)) == len(ids), "candidate ids must be distinct"


@_have_raw
def test_one_reasoning_each():
    cands = load_cached_candidates(RAW_K2)
    for c in cands:
        assert c.reasoning_trace is not None and c.reasoning_trace.strip()


@_have_raw
def test_horizon_frequency_and_finite():
    cands = load_cached_candidates(RAW_K2)
    for c in cands:
        assert c.trajectory_frame == "ego_t0"
        assert c.trajectory_hz == 10.0
        assert c.trajectory_steps == 64  # 6.4 s @ 10 Hz
        c.validate()  # finite check inside


@_have_raw
def test_duplicate_detection_passes_for_distinct():
    cands = load_cached_candidates(RAW_K2)
    assert_distinct_trajectories(cands)  # must not raise: the two real trajectories differ


@_have_raw
def test_jsonl_round_trip():
    cands = load_cached_candidates(RAW_K2)
    with open(RAW_K2) as f:
        raw = [json.loads(line) for line in f if line.strip()]
    assert len(raw) == len(cands)
    for r, c in zip(raw, cands):
        assert r["candidate_id"] == c.candidate_id
        assert r["traj_xyz_sha256"] == c.traj_xyz_sha256


@_have_raw
def test_observation_identical_across_candidates():
    man = json.load(open(MANIFEST))
    assert man.get("observation_identical_across_candidates") is True


@_have_raw
def test_to_rollout_request_preserves_trajectory():
    cands = load_cached_candidates(RAW_K2)
    req = to_rollout_request(
        cands[0], sample_id="s0", scene_id="clipgt-01d503d4", initial_state_ref="ref0"
    )
    assert req.trajectory == [list(p) for p in cands[0].trajectory_xyz]
    assert req.candidate_source == "alpamayo_native"


def test_cached_candidate_is_immutable():
    c = CachedAlpamayoCandidate(
        candidate_id="cand_0",
        candidate_rank=0,
        candidate_source="alpamayo_native",
        reasoning_trace="nudge left",
        trajectory_xyz=tuple((float(i), 0.0, 0.0) for i in range(64)),
        trajectory_frame="ego_t0",
        trajectory_hz=10.0,
        trajectory_steps=64,
        traj_xyz_sha256="deadbeef",
        seed=42,
    )
    c.validate()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.candidate_id = "tampered"  # type: ignore[misc]


def test_no_fallback_candidate_rejected():
    bad = CachedAlpamayoCandidate(
        candidate_id="fallback_stop",
        candidate_rank=1,
        candidate_source="fallback",
        reasoning_trace="stop",
        trajectory_xyz=tuple((0.0, 0.0, 0.0) for _ in range(64)),
        trajectory_frame="ego_t0",
        trajectory_hz=10.0,
        trajectory_steps=64,
        traj_xyz_sha256="x",
    )
    with pytest.raises(ValueError):
        bad.validate()
    with pytest.raises(ValueError):
        assert_native_only([bad])


def test_real_label_source_rejects_proxy():
    with pytest.raises(ValueError):
        assert_real_label_source(PROXY_LABEL_SOURCE, context="real K=2 smoke")
    # a NONE/placeholder label is not an allowed value either
    with pytest.raises(ValueError):
        assert_real_label_source("NONE")


@pytest.mark.skipif(not os.path.exists(METRIC_CSV), reason="metric matrix not present")
def test_evaluator_native_metric_names_preserved():
    with open(METRIC_CSV) as f:
        rows = list(csv.DictReader(f))
    by_sw = {r["safeworld_metric"]: r for r in rows}
    # Native collision names preserved (NOT renamed to NC).
    assert "collision_any" in by_sw["NC"]["native_alpasim_metric_names"]
    assert "offroad" in by_sw["DAC"]["native_alpasim_metric_names"]
    # TTC must NOT be falsely mapped to a native metric (no native TTC exists in v2026.5).
    assert by_sw["TTC"]["available_via_alpasim_when_unblocked"] == "partial"
    assert "min_distance_to_obstacle_m" in by_sw["TTC"]["native_alpasim_metric_names"]

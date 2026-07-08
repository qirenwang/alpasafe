"""T24.5 / T25A — tests for rollout semantics, reasoning alignment, provenance,
coordinate transforms, ASL targets, and metric-name preservation.

Unit tests run everywhere. Artifact-backed tests use the REAL v2.1 outputs and the
latest ASL-preflight run dir, and skip cleanly if those are absent (so the suite
stays green on a fresh checkout). No existing test is modified or weakened.
"""

import glob
import hashlib
import json
import os

import numpy as np
import pytest

from safeworld.data.rollout_semantics import (
    COMPLETED_K2_SMOKE_SEMANTICS,
    RolloutSemantics,
    migrate_label_source_to_semantics,
)
from safeworld.data.schema_v2 import (
    CandidateRolloutRecord,
    assert_real_label_source,
    validate_no_logged_label_duplication,
)
from safeworld.real_smoke.asl_geometry import (
    aabb_xy_to_rig_xy,
    ego_t0_to_local_xy,
    finite_difference_velocity,
    local_xy_to_ego_t0,
    quat_wxyz_to_yaw,
    rig_xy_to_aabb_xy,
)
from safeworld.real_smoke.reasoning_alignment import (
    ReasoningAmbiguityError,
    assert_unambiguous_reasoning,
    parse_serialized_reasoning_array,
    recover_candidate_reasoning,
    split_reasoning_by_sample_index,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "outputs", "real_smoke")


def _opt(path):
    return pytest.mark.skipif(not os.path.exists(path), reason=f"{path} absent")


def _canon_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _latest_run():
    runs = sorted(glob.glob(os.path.join(_ROOT, "artifacts", "safeworld_t24_5_asl_preflight", "*")))
    return runs[-1] if runs else None


# --------------------------------------------------------------------------- #
# 1. structured rollout semantics validation
# --------------------------------------------------------------------------- #
def test_rollout_semantics_completed_run_validates():
    COMPLETED_K2_SMOKE_SEMANTICS.validate()
    d = COMPLETED_K2_SMOKE_SEMANTICS.to_dict()
    assert d["ego_execution_mode"] == "cached_alpamayo_native_candidate"
    assert d["ego_policy_replanning"] is False
    assert d["traffic_sim_active"] is False
    assert d["interaction_assumption"] == "nonreactive_traffic"
    assert d["evaluation_source"] == "alpasim_native"


def test_rollout_semantics_rejects_incoherent():
    with pytest.raises(ValueError):
        RolloutSemantics(
            rollout_backend="alpasim", rollout_backend_version="x",
            evaluation_source="alpasim_native", simulation_loop_active=True,
            ego_execution_mode="cached_alpamayo_native_candidate", ego_policy_replanning=False,
            controller_active=True, physics_active=True, renderer_backend="nurec",
            traffic_sim_active=True,  # contradicts nonreactive_traffic
            non_ego_behavior="recorded_nonreactive", interaction_assumption="nonreactive_traffic",
        ).validate()
    with pytest.raises(ValueError):
        RolloutSemantics(
            rollout_backend="alpasim", rollout_backend_version="x",
            evaluation_source="BOGUS", simulation_loop_active=True,
            ego_execution_mode="cached_alpamayo_native_candidate", ego_policy_replanning=False,
            controller_active=True, physics_active=True, renderer_backend="nurec",
            traffic_sim_active=False, non_ego_behavior="recorded_nonreactive",
            interaction_assumption="nonreactive_traffic",
        ).validate()


# --------------------------------------------------------------------------- #
# 2. legacy record migration
# --------------------------------------------------------------------------- #
def test_legacy_label_source_migration():
    for ls in ("alpasim_closed_loop", "nurec_candidate_replay", "fixed_agent_replay",
               "logged_executed_action_only", "proxy_test_only"):
        sem = migrate_label_source_to_semantics(ls)
        sem.validate()
        assert sem.legacy_label_source == ls
        assert sem.metadata["migrated_from_label_source"] == ls
    with pytest.raises(ValueError):
        migrate_label_source_to_semantics("not_a_real_source")


# --------------------------------------------------------------------------- #
# 3 & 14. candidate-specific reasoning alignment + ambiguous rejection by CLI
# --------------------------------------------------------------------------- #
def test_parse_numpy_style_reasoning_array():
    s = "['Keep distance since stopped ahead'\n 'Keep distance since directly ahead']"
    elems = parse_serialized_reasoning_array(s)
    assert elems == ["Keep distance since stopped ahead", "Keep distance since directly ahead"]


def test_split_reasoning_requires_exact_count():
    s = "['a' 'b']"
    assert split_reasoning_by_sample_index(s, 2) == ["a", "b"]
    with pytest.raises(ReasoningAmbiguityError):
        split_reasoning_by_sample_index(s, 3)  # never pad/guess


def test_recover_candidate_reasoning_maps_by_sample_index():
    shared = "['reasoning A' 'reasoning B']"
    recs = [
        {"candidate_index": 0, "reasoning_text": shared},
        {"candidate_index": 1, "reasoning_text": shared},
    ]
    rec = {c.candidate_index: c.reasoning_trace for c in recover_candidate_reasoning(recs)}
    assert rec == {0: "reasoning A", 1: "reasoning B"}


def test_ambiguous_reasoning_rejected_single_accepted():
    with pytest.raises(ReasoningAmbiguityError):
        assert_unambiguous_reasoning("['a' 'b']")  # un-split array
    with pytest.raises(ReasoningAmbiguityError):
        assert_unambiguous_reasoning(None)
    assert_unambiguous_reasoning("Keep distance to the lead vehicle since it is stopped ahead")


def test_real_cli_loader_rejects_ambiguous_reasoning():
    """The real candidate loader (candidate_io) rejects un-split K-sample reasoning."""
    from safeworld.real_smoke.candidate_io import CachedAlpamayoCandidate

    bad = CachedAlpamayoCandidate(
        candidate_id="cand_0", candidate_rank=0, candidate_source="alpamayo_native",
        reasoning_trace="['reasoning A' 'reasoning B']",  # ambiguous array, not split
        trajectory_xyz=tuple((float(i), 0.0, 0.0) for i in range(64)),
        trajectory_frame="ego_t0", trajectory_hz=10.0, trajectory_steps=64,
        traj_xyz_sha256="x",
    )
    with pytest.raises(ReasoningAmbiguityError):
        bad.validate()


@_opt(os.path.join(_OUT, "alpamayo_raw_k2_candidate_reasoning_v2_1.jsonl"))
def test_real_reasoning_artifact_is_unambiguous_and_distinct():
    recs = [json.loads(ln) for ln in open(os.path.join(_OUT, "alpamayo_raw_k2_candidate_reasoning_v2_1.jsonl"))]
    assert len(recs) == 2
    traces = [r["reasoning_trace"] for r in recs]
    for t in traces:
        assert_unambiguous_reasoning(t)  # must not raise
    assert traces[0] != traces[1]  # the two candidates have distinct reasoning
    assert recs[0]["reasoning_trace"] == "Keep distance to the lead vehicle since it is stopped ahead"


# --------------------------------------------------------------------------- #
# 4. candidate / trajectory canonical hashing
# --------------------------------------------------------------------------- #
def test_canonical_hash_is_order_stable_and_value_sensitive():
    traj = [[0.1, 0.2], [0.3, 0.4]]
    assert _canon_sha(traj) == _canon_sha([[0.1, 0.2], [0.3, 0.4]])
    assert _canon_sha(traj) != _canon_sha([[0.1, 0.2], [0.3, 0.5]])


@_opt(os.path.join(_OUT, "candidate_rollout_provenance_k2_v2_1.jsonl"))
def test_provenance_trajectory_hash_matches_reasoning_artifact():
    prov = [json.loads(ln) for ln in open(os.path.join(_OUT, "candidate_rollout_provenance_k2_v2_1.jsonl"))]
    reas = [json.loads(ln) for ln in open(os.path.join(_OUT, "alpamayo_raw_k2_candidate_reasoning_v2_1.jsonl"))]
    by_i = {r["candidate_index"]: r for r in reas}
    for p in prov:
        i = p["candidate_index"]
        assert p["candidate_trajectory_canonical_sha256"] == by_i[i]["trajectory_canonical_sha256"]


# --------------------------------------------------------------------------- #
# 5. candidate-to-rollout provenance chain
# --------------------------------------------------------------------------- #
@_opt(os.path.join(_OUT, "candidate_rollout_provenance_k2_v2_1.jsonl"))
def test_provenance_chain_complete_and_linked():
    prov = [json.loads(ln) for ln in open(os.path.join(_OUT, "candidate_rollout_provenance_k2_v2_1.jsonl"))]
    assert len(prov) == 2
    for p in prov:
        # required custody fields present and non-empty
        for k in ("rollout_asl_sha256", "metrics_parquet_sha256", "candidate_trajectory_canonical_sha256",
                  "candidate_reasoning_sha256", "rollout_id", "session_uuid", "scene_uuid",
                  "usdz_sha256", "alpasim_git_revision"):
            assert p[k], f"missing {k}"
        assert p["complete_marker_present"] is True
        # the dump candidate anchored at logged t0 reproduces the executed plan
        assert p["candidate_anchor_to_executed_plan_max_abs_m"] < 1e-3
    # the two candidates map to two distinct rollouts
    assert prov[0]["rollout_id"] != prov[1]["rollout_id"]


# --------------------------------------------------------------------------- #
# 7 & 8 & 10. ASL timestamp ordering, coordinate round-trip, valid mask (real)
# --------------------------------------------------------------------------- #
def test_coordinate_transforms_round_trip_unit():
    rng = np.random.default_rng(0)
    xy = rng.normal(size=(20, 2)) * 10.0
    yaw0 = 0.37
    back = ego_t0_to_local_xy(local_xy_to_ego_t0(xy, 3.0, -2.0, yaw0), 3.0, -2.0, yaw0)
    assert np.abs(back - xy).max() < 1e-12
    yaw = np.full(20, 0.1)
    back2 = rig_xy_to_aabb_xy(aabb_xy_to_rig_xy(xy, yaw, 1.4675), yaw, 1.4675)
    assert np.abs(back2 - xy).max() < 1e-12


def test_quat_yaw_and_finite_diff():
    assert abs(quat_wxyz_to_yaw(1.0, 0.0, 0.0, 0.0)) < 1e-12
    # +90 deg about Z
    import math
    q = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]
    assert abs(quat_wxyz_to_yaw(*q) - math.pi / 2) < 1e-6
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    ts = np.array([0, 100_000, 200_000])  # 0.1 s steps
    vel = finite_difference_velocity(pos, ts)
    assert np.allclose(vel[:, 0], 10.0)  # 1 m / 0.1 s = 10 m/s


@pytest.mark.skipif(_latest_run() is None, reason="no ASL preflight run dir")
def test_real_actor_poses_timestamps_strictly_increasing():
    run = _latest_run()
    for i in (0, 1):
        ext = json.load(open(os.path.join(run, "extracted", f"asl_extraction_cand{i}.json")))
        ts = [f["timestamp_us"] for f in ext["actor_poses"]]
        assert ts == sorted(ts)
        assert all(b > a for a, b in zip(ts, ts[1:]))  # strictly increasing


@_opt(os.path.join(_OUT, "future_consequence_targets_k2_v2_1.jsonl"))
def test_real_target_roundtrip_and_valid_mask():
    targets = [json.loads(ln) for ln in open(os.path.join(_OUT, "future_consequence_targets_k2_v2_1.jsonl"))]
    assert len(targets) == 2
    for t in targets:
        # coordinate round-trip recorded and tight
        assert t["coordinate_transform_roundtrip_max_m"] < 1e-9
        # timestamps strictly increasing, ego_t0 anchored near origin
        ts = t["timestamps_us"]
        assert all(b > a for a, b in zip(ts, ts[1:]))
        assert abs(t["future_ego_states_ego_t0"][0][0]) < 0.05
        # valid mask: shape matches, has BOTH present and absent cells (actors enter/leave)
        vm = np.array(t["actor_valid_mask"])
        assert vm.shape == (len(ts), len(t["actor_ids"]))
        assert vm.max() == 1 and vm.min() == 0
        # occupancy not fabricated
        assert t["occupancy"] is None
        # JSONL round-trip
        assert json.loads(json.dumps(t))["candidate_id"] == t["candidate_id"]


# --------------------------------------------------------------------------- #
# 9. planned-vs-executed alignment (real)
# --------------------------------------------------------------------------- #
@_opt(os.path.join(_OUT, "planned_vs_executed_alignment_k2_v2_1.jsonl"))
def test_real_planned_vs_executed_alignment():
    recs = [json.loads(ln) for ln in open(os.path.join(_OUT, "planned_vs_executed_alignment_k2_v2_1.jsonl"))]
    assert len(recs) == 2
    for a in recs:
        assert a["number_of_aligned_steps"] >= 60
        assert a["timestamp_alignment_error_us"] == 0  # exact stamp match
        assert a["planned_vs_executed_ADE_m"] >= 0.0
        assert a["planned_vs_executed_max_error_m"] >= a["planned_vs_executed_ADE_m"]
        # the cached-candidate replay tracks its plan to well under a metre on average
        assert a["planned_vs_executed_ADE_m"] < 1.0
        assert a["t0_rig_est_vs_true_aabb_rig_residual_m"] < 0.05


# --------------------------------------------------------------------------- #
# 11 & 12. proxy rejection + logged-future duplication rejection (unchanged guards)
# --------------------------------------------------------------------------- #
def test_proxy_test_only_rejected_by_real_cli():
    with pytest.raises(ValueError):
        assert_real_label_source("proxy_test_only", context="real T24.5 CLI")


def _logged_record(cand_id, future):
    return CandidateRolloutRecord(
        sample_id="s0", scene_id="sc", candidate_id=cand_id, candidate_rank=0,
        candidate_source="alpamayo_native", reasoning_trace="r",
        trajectory=[[0.0, 0.0]], trajectory_frame="ego_t0", trajectory_hz=10.0,
        rollout_backend="logged_only", rollout_backend_version="v",
        label_source="logged_executed_action_only", rollout_success=True, failure_reason=None,
        future_ego_states=future, future_agent_states_or_occupancy=None,
        collision_events=[], drivable_area_events=[], ttc_curve=[], progress=0.0,
        comfort_components={}, metric_definition_version="v2_unfrozen",
    )


def test_logged_future_duplication_rejected():
    same = [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(ValueError):
        validate_no_logged_label_duplication([_logged_record("a", same), _logged_record("b", same)])
    # distinct futures are fine
    validate_no_logged_label_duplication(
        [_logged_record("a", [[1.0, 2.0]]), _logged_record("b", [[9.0, 9.0]])]
    )


# --------------------------------------------------------------------------- #
# 13. native metric-name preservation (real evaluator record)
# --------------------------------------------------------------------------- #
@_opt(os.path.join(_OUT, "evaluation_components_k2_v2_1.jsonl"))
def test_native_metric_names_preserved_and_no_fake_metrics():
    recs = [json.loads(ln) for ln in open(os.path.join(_OUT, "evaluation_components_k2_v2_1.jsonl"))]
    for e in recs:
        names = set(e["native_metric_names_present"])
        assert "collision_any" in names
        assert "min_distance_to_obstacle_m" in names
        # no fabricated TTC / Comfort, no renamed collision_at_fault
        assert "TTC" not in names and "Comfort" not in names
        assert "collision_at_fault" not in names
        assert "ABSENT" in e["collision_at_fault_status"]

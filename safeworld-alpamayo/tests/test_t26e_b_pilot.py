"""T26E-B pilot: model, selector-adapter, leakage, and grouped-eval tests."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from safeworld.t26e.loader import DEFAULT_ROOT  # noqa: E402
from safeworld.t26e.model import (  # noqa: E402
    CandidateTrajectoryConsequenceModel,
    OfficialSceneScoreHead,
    SafeWorldCandidatePilot,
)
from safeworld.t26e.selector_adapter import (  # noqa: E402
    CandidateOfficialScorePrediction,
    select_within_group,
)
from safeworld.t26e.train_pilot import (  # noqa: E402
    checkpoint_key,
    fit_train_plan_stats,
    group_metrics,
    tensorize,
)

DATASET_BUILT = (DEFAULT_ROOT / "t26e_frozen_dataset_manifest.json").exists()


def _plan(b: int = 4) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(b, 64, 2)


def test_model_shapes_and_ranges():
    model = SafeWorldCandidatePilot()
    plan = _plan(5)
    out = model(plan, plan)
    assert out["latent_consequence"].shape == (5, 64)
    assert out["predicted_future_ego_states_ego_t0"].shape == (5, 64, 2)
    score = out["predicted_official_alpasim_scene_score"]
    assert score.shape == (5,)
    assert torch.isfinite(score).all() and (score >= 0).all() and (score <= 1).all()


def test_w_theta_residual_zero_init():
    model = CandidateTrajectoryConsequenceModel()
    plan = _plan()
    _, future = model(plan, plan)
    assert torch.equal(future, plan)  # starts exactly at plan-as-future (T26C)


def test_g_phi_consumes_w_theta_latent_no_plan_bypass():
    model = SafeWorldCandidatePilot()
    plan = _plan()
    out = model(plan, plan)
    # G_phi's score must be a pure function of the latent produced by W_theta.
    direct = model.g_phi(out["latent_consequence"])
    assert torch.equal(direct, out["predicted_official_alpasim_scene_score"])
    # OfficialSceneScoreHead's only trainable input dimension is the 64-d latent.
    assert OfficialSceneScoreHead().head.in_features == 64


def test_no_fabricated_reward_components():
    fields = set(CandidateOfficialScorePrediction.__dataclass_fields__)
    assert fields == {"sample_id", "candidate_id", "final_reward", "uncertainty", "candidate_rank"}
    pred = CandidateOfficialScorePrediction("s", "cand_0", 0.5, candidate_rank=0)
    assert pred.uncertainty is None  # no invented uncertainty


def test_selector_adapter_tie_break_chain():
    def p(cid, reward, rank):
        return CandidateOfficialScorePrediction("s", cid, reward, candidate_rank=rank)

    # 1) highest final_reward wins
    assert select_within_group([p("cand_0", 0.2, 0), p("cand_1", 0.9, 1)]).candidate_id == "cand_1"
    # 2) uncertainty all None -> treated equal; 3) lower candidate_rank breaks tie
    assert select_within_group([p("cand_3", 0.5, 3), p("cand_2", 0.5, 2)]).candidate_id == "cand_2"
    # 4) lexicographic candidate_id as final tie-break
    assert (
        select_within_group([p("cand_b", 0.5, None), p("cand_a", 0.5, None)]).candidate_id
        == "cand_a"
    )


def test_checkpoint_tie_break_order():
    base = {
        "val_score_mae": 1.0,
        "val_k8_mean_regret": 1.0,
        "val_future_ade": 1.0,
        "val_total_loss": 1.0,
        "epoch": 5,
    }
    better_mae = {**base, "val_score_mae": 0.9, "epoch": 9}
    assert checkpoint_key(better_mae) < checkpoint_key(base)
    tie_regret = {**base, "val_k8_mean_regret": 0.5, "epoch": 9}
    assert checkpoint_key(tie_regret) < checkpoint_key(base)
    earlier = {**base, "epoch": 4}
    assert checkpoint_key(earlier) < checkpoint_key(base)


def test_grouped_k_prefix_metrics_toy():
    meta = [
        {
            "sample_id": f"g@1#cand_{i}",
            "decision_group_id": "g@1",
            "candidate_id": f"cand_{i}",
            "candidate_index": i,
        }
        for i in range(8)
    ]
    target = np.array([0.1, 0.9, 0.3, 0.4, 0.5, 0.2, 0.6, 0.0])
    pred_good = target.copy()
    for k, expect_top1 in ((2, 1.0), (5, 1.0), (8, 1.0)):
        gm = group_metrics(pred_good, target, meta, k)
        assert gm["top1_selection_accuracy"] == expect_top1
        assert gm["mean_selected_score_regret"] == 0.0
        assert gm["pairwise_ranking_accuracy"] == 1.0
    pred_const = np.zeros(8)  # constant -> selector picks rank 0
    gm = group_metrics(pred_const, target, meta, 8)
    assert gm["per_group"][0]["selected_candidate_index"] == 0
    assert gm["mean_selected_score_regret"] == pytest.approx(0.9 - 0.1)
    assert gm["predicted_tie_count"] == 1


def test_fixed_seed_ordering_deterministic():
    a = torch.randperm(192, generator=torch.Generator().manual_seed(0))
    b = torch.randperm(192, generator=torch.Generator().manual_seed(0))
    c = torch.randperm(192, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b) and not torch.equal(a, c)


@pytest.mark.skipif(not DATASET_BUILT, reason="T26E-A dataset not built")
class TestWithDataset:
    def test_feature_tensor_is_plan_only(self):
        data = tensorize("train")
        assert data["feature_source"] == "input.planned_trajectory"
        assert data["plan"].shape == (192, 64, 2) and data["plan"].dtype == torch.float32
        assert torch.isfinite(data["plan"]).all()
        # future/score/meta are targets+metadata, structurally separate tensors
        assert set(data) == {"plan", "future", "tmask", "score", "meta", "feature_source"}

    def test_targets_not_in_input_tensor(self):
        # The feature tensor must be reconstructable from input.planned_trajectory alone.
        import json

        data = tensorize("val")
        rows = [
            json.loads(line)
            for line in (DEFAULT_ROOT / "t26e_val_samples.jsonl").read_text().splitlines()
        ]
        ref = torch.tensor([r["input"]["planned_trajectory"] for r in rows], dtype=torch.float32)
        assert torch.equal(data["plan"], ref)

    def test_score_targets_identity_in_0_1(self):
        data = tensorize("val")
        assert (data["score"] >= 0).all() and (data["score"] <= 1).all()
        assert torch.isfinite(data["score"]).all()

    def test_train_only_preprocessing_stats(self):
        train = tensorize("train")
        stats = fit_train_plan_stats(train["plan"])
        assert stats["plan_mean"].shape == (2,) and stats["plan_std"].shape == (2,)
        val = tensorize("val")
        combined = fit_train_plan_stats(torch.cat([train["plan"], val["plan"]]))
        assert not torch.equal(stats["plan_mean"], combined["plan_mean"])  # val excluded

    def test_sealed_labels_still_refused(self):
        from safeworld.t26e.loader import SealedTestLabelError, _read_jsonl_guarded

        with pytest.raises(SealedTestLabelError):
            _read_jsonl_guarded(DEFAULT_ROOT / "t26e_test_labels_sealed.jsonl")

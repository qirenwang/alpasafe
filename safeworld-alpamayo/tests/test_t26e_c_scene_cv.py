"""T26E-C scene-CV fold-integrity, leakage-guard, and metric-direction tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from safeworld.t26e.b1_diag import (  # noqa: E402
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    _agg,
)
from safeworld.t26e.loader import DEFAULT_ROOT  # noqa: E402
from safeworld.t26e.model import SafeWorldCandidatePilot  # noqa: E402
from safeworld.t26e.scene_cv import (  # noqa: E402
    N_FOLDS,
    OuterSceneVault,
    build_fold_manifest,
    fold_indices,
    load_split_scenes,
    subset,
    tensorize_dev,
    train_fold_seed,
)
from safeworld.t26e.train_pilot import fit_train_plan_stats  # noqa: E402

DATASET_BUILT = (DEFAULT_ROOT / "t26e_frozen_dataset_manifest.json").exists()


# ------------------------------------------------------- parameter count
def test_parameter_count_is_20801_not_12865():
    model = SafeWorldCandidatePilot()
    total = sum(p.numel() for p in model.parameters())
    assert total == 20801
    assert total != 12865  # historical prose error, see T26E-B.1 erratum
    per_module = {
        "encoder.0": 128 * 64 + 64,
        "encoder.2": 64 * 64 + 64,
        "future_residual_head": 64 * 128 + 128,
        "g_phi.head": 64 * 1 + 1,
    }
    assert sum(per_module.values()) == 20801


# ------------------------------------------------------- metric directions
def test_metric_direction_registry_disjoint_and_complete():
    assert not (LOWER_IS_BETTER & HIGHER_IS_BETTER)
    assert "val_score_mae" in LOWER_IS_BETTER
    assert "val_k8_mean_regret" in LOWER_IS_BETTER
    assert "val_spearman" in HIGHER_IS_BETTER
    assert "val_k8_pairwise" in HIGHER_IS_BETTER
    assert "val_k8_top1" in HIGHER_IS_BETTER


def test_agg_best_worst_respect_direction():
    values = {"0": 0.1, "1": 0.3, "2": 0.2}
    lower = _agg(values, "val_score_mae")
    assert lower["best"] == 0.1 and lower["worst"] == 0.3
    higher = _agg(values, "val_spearman")
    assert higher["best"] == 0.3 and higher["worst"] == 0.1
    with pytest.raises(ValueError):
        _agg(values, "metric_not_in_registry")


# ------------------------------------------------------- fold protocol
def test_fold_manifest_each_scene_serves_once_outer_once_inner_eight_train():
    m = build_fold_manifest()
    dev = m["development_scene_ids_sorted"]
    assert len(dev) == N_FOLDS == 10 and dev == sorted(dev)
    outers = [f["outer_evaluation_scene"] for f in m["folds"]]
    inners = [f["inner_validation_scene"] for f in m["folds"]]
    assert sorted(outers) == dev and sorted(inners) == dev
    for j, f in enumerate(m["folds"]):
        assert f["outer_evaluation_scene"] == dev[j]
        assert f["inner_validation_scene"] == dev[(j + 1) % 10]
        assert len(f["fold_training_scenes"]) == 8
        assert f["outer_evaluation_scene"] not in f["fold_training_scenes"]
        assert f["inner_validation_scene"] not in f["fold_training_scenes"]
    train_counts = {s: sum(s in f["fold_training_scenes"] for f in m["folds"]) for s in dev}
    assert all(c == 8 for c in train_counts.values())


def test_no_sealed_test_scene_in_any_fold():
    m = build_fold_manifest()
    sealed = set(load_split_scenes()["test"])
    assert len(sealed) == 2
    for f in m["folds"]:
        members = {f["outer_evaluation_scene"], f["inner_validation_scene"]} | set(
            f["fold_training_scenes"]
        )
        assert not (members & sealed)
    assert not (set(m["development_scene_ids_sorted"]) & sealed)


# ------------------------------------------------------- outer-label vault
def test_outer_vault_blocks_access_before_checkpoint_freeze():
    dev = {
        "plan": torch.zeros(4, 64, 2),
        "future": torch.zeros(4, 64, 2),
        "tmask": torch.zeros(4, 64, dtype=torch.bool),
        "score": torch.zeros(4),
        "meta": [{"scene_id": "s", "decision_group_id": "s@1"} for _ in range(4)],
    }
    vault = OuterSceneVault(dev, [0, 1])
    with pytest.raises(RuntimeError, match="BLOCKED_T26E_C_TARGET_LEAKAGE"):
        vault.open()
    assert vault.first_read_utc is None
    vault.freeze_checkpoint()
    out = vault.open()
    assert out["plan"].shape[0] == 2 and vault.first_read_utc is not None


def test_trainer_signature_cannot_see_outer_scene():
    import inspect

    params = list(inspect.signature(train_fold_seed).parameters)
    assert params == ["fold_id", "seed", "train_data", "inner_data", "run_dir"]
    assert "outer" not in " ".join(params)


@pytest.mark.skipif(not DATASET_BUILT, reason="T26E-A dataset not built")
class TestWithDataset:
    def test_dev_pool_is_240_candidates_10_scenes_no_test(self):
        dev = tensorize_dev()
        scenes = {m["scene_id"] for m in dev["meta"]}
        assert len(dev["meta"]) == 240 and len(scenes) == 10
        assert not (scenes & set(load_split_scenes()["test"]))
        groups = {m["decision_group_id"] for m in dev["meta"]}
        assert len(groups) == 30

    def test_fold_indices_disjoint_and_counted(self):
        dev = tensorize_dev()
        m = build_fold_manifest()
        for f in m["folds"]:
            tr, iv, oe = fold_indices(dev, f)
            assert len(tr) == 192 and len(iv) == 24 and len(oe) == 24
            assert not (set(tr) & set(iv) | set(tr) & set(oe) | set(iv) & set(oe))
            # membership matches the declared scenes exactly
            assert {dev["meta"][i]["scene_id"] for i in tr} == set(f["fold_training_scenes"])
            assert {dev["meta"][i]["scene_id"] for i in iv} == {f["inner_validation_scene"]}
            assert {dev["meta"][i]["scene_id"] for i in oe} == {f["outer_evaluation_scene"]}
            # 3 groups x 8 candidates in each held-out scene
            for idx in (iv, oe):
                gids = [dev["meta"][i]["decision_group_id"] for i in idx]
                assert len(set(gids)) == 3
                assert all(gids.count(g) == 8 for g in set(gids))

    def test_normalization_is_fold_train_only(self):
        dev = tensorize_dev()
        m = build_fold_manifest()
        tr, iv, oe = fold_indices(dev, m["folds"][0])
        fold_train = subset(dev, tr)
        stats = fit_train_plan_stats(fold_train["plan"])
        manual_mean = fold_train["plan"].mean(dim=(0, 1))
        assert torch.equal(stats["plan_mean"], manual_mean)
        # stats fit on all development data must differ -> held-out scenes excluded
        all_stats = fit_train_plan_stats(dev["plan"])
        assert not torch.equal(stats["plan_mean"], all_stats["plan_mean"])
        # stats fit including the outer scene must differ too
        with_outer = fit_train_plan_stats(subset(dev, tr + oe)["plan"])
        assert not torch.equal(stats["plan_mean"], with_outer["plan_mean"])

    def test_metadata_is_not_tensorized(self):
        dev = tensorize_dev()
        assert dev["feature_source"] == "input.planned_trajectory"
        assert set(dev) == {"plan", "future", "tmask", "score", "meta", "feature_source"}
        # feature tensor holds exactly 64x2 floats per record: no room for ids
        assert dev["plan"].shape == (240, 64, 2)
        # identifiers live only in meta dicts, not in any tensor
        for key in ("plan", "future", "score"):
            assert dev[key].dtype in (torch.float32, torch.bool)

"""T26E-A derived dataset: structure, guards, leakage, and determinism tests.

These tests run against the frozen derived dataset under
outputs/t26e_official_score_training_dataset_alpasim_196d21a/ and are skipped
if it has not been built yet.
"""

from __future__ import annotations

import json
import math

import pytest

from safeworld.t26e.loader import (
    DEFAULT_ROOT,
    SealedTestLabelError,
    T26EGroupIndex,
    T26EOfficialScoreDataset,
    T26ETestInputDataset,
    _read_jsonl_guarded,
)
from safeworld.t26e.schema import FORBIDDEN_INPUT_KEYS, _walk_keys

pytestmark = pytest.mark.skipif(
    not (DEFAULT_ROOT / "t26e_frozen_dataset_manifest.json").exists(),
    reason="T26E-A derived dataset not built",
)


def test_split_counts_and_group_completeness():
    train = T26EOfficialScoreDataset("train")
    val = T26EOfficialScoreDataset("val")
    assert len(train) == 192 and len(val) == 48
    for ds, n_groups in ((train, 24), (val, 6)):
        groups = ds.groups()
        assert len(groups) == n_groups
        assert all(len(m) == 8 for m in groups.values())


def test_test_split_not_loadable_as_labeled():
    with pytest.raises(ValueError, match="not loadable"):
        T26EOfficialScoreDataset("test")


def test_sealed_labels_refused_by_all_loaders():
    with pytest.raises(SealedTestLabelError):
        _read_jsonl_guarded(DEFAULT_ROOT / "t26e_test_labels_sealed.jsonl")


def test_score_target_identity_range_dtype():
    for split in ("train", "val"):
        ds = T26EOfficialScoreDataset(split)
        for i, rec in enumerate(ds.records):
            v = ds.score_target(i)
            assert isinstance(v, float) and math.isfinite(v) and 0.0 <= v <= 1.0
            assert rec.targets["official_score"]["score_transformation"] == "identity"


def test_no_forbidden_keys_in_any_input_section():
    rows = []
    for name in ("t26e_train_samples.jsonl", "t26e_val_samples.jsonl", "t26e_test_inputs.jsonl"):
        rows += [json.loads(line) for line in (DEFAULT_ROOT / name).read_text().splitlines()]
    assert len(rows) == 288
    for row in rows:
        assert not (_walk_keys(row["input"]) & FORBIDDEN_INPUT_KEYS)


def test_test_inputs_carry_no_targets_or_outcomes():
    ds = T26ETestInputDataset()
    assert len(ds) == 48
    raw = [
        json.loads(line)
        for line in (DEFAULT_ROOT / "t26e_test_inputs.jsonl").read_text().splitlines()
    ]
    for row in raw:
        assert row["targets"] is None
        flat = json.dumps(row)
        assert "official_alpasim_scene_score" not in flat
        assert "failure_reason" not in flat
        assert "future_ego_states" not in flat


def test_deterministic_canonical_ordering():
    order = {"train": 0, "val": 1, "test": 2}
    idx = [
        json.loads(line)
        for line in (DEFAULT_ROOT / "t26e_all_sample_index.jsonl").read_text().splitlines()
    ]
    keys = [
        (order[r["split"]], r["scene_id"], r["decision_timestamp_us"], r["candidate_index"])
        for r in idx
    ]
    assert keys == sorted(keys)
    assert len({r["sample_id"] for r in idx}) == 288
    assert len({r["rollout_id"] for r in idx}) == 288


def test_group_index_is_structural_only():
    gi = T26EGroupIndex()
    assert len(gi.groups) == 36
    assert [len(gi.by_split(s)) for s in ("train", "val", "test")] == [24, 6, 6]
    flat = json.dumps(gi.groups)
    assert "score" not in flat and "passed" not in flat


def test_builder_determinism_rebuild(tmp_path):
    from safeworld.t26e.build_dataset import build, file_sha256

    rebuilt = build(tmp_path)
    frozen = json.loads((DEFAULT_ROOT / "t26e_frozen_dataset_manifest.json").read_text())
    primary = dict(frozen["file_sha256"])
    primary["t26e_frozen_dataset_manifest.json"] = file_sha256(
        DEFAULT_ROOT / "t26e_frozen_dataset_manifest.json"
    )
    assert rebuilt == primary

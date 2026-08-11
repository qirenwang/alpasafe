"""T26E-A dataset loaders (train/val labeled, test inputs, group index).

Hard guards:

- ``T26EOfficialScoreDataset`` accepts only ``split in {"train", "val"}`` and
  loads exactly the file for the requested split (a train loader never reads
  validation data unless a second loader is explicitly constructed).
- No loader in this module will open ``t26e_test_labels_sealed.jsonl`` — any
  path whose name marks it as sealed raises ``SealedTestLabelError``.
- Every record is structurally validated on load, including the leakage guard
  (no post-rollout key inside the ``input`` section).

Loaders are torch-free; they return plain Python structures in the dataset's
deterministic canonical order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safeworld.t26e.schema import T26ERecord, validate_record_structure

DEFAULT_ROOT = (
    Path(__file__).resolve().parents[3]
    / "outputs/t26e_official_score_training_dataset_alpasim_196d21a"
)
LABELED_FILES = {"train": "t26e_train_samples.jsonl", "val": "t26e_val_samples.jsonl"}
SEALED_MARKER = "labels_sealed"


class SealedTestLabelError(RuntimeError):
    """Raised on any attempt to open the sealed test-label artifact."""


def _read_jsonl_guarded(path: Path) -> list[dict[str, Any]]:
    if SEALED_MARKER in path.name:
        raise SealedTestLabelError(
            f"refusing to open sealed test labels: {path.name} "
            "(ALLOW_TEST_SCORE_ANALYSIS=0; sealed artifact is not loadable "
            "by training/validation/test-input loaders)"
        )
    return [json.loads(line) for line in path.read_text().splitlines()]


class T26EOfficialScoreDataset:
    """Labeled train/val dataset over the frozen T26E-A derived records."""

    def __init__(self, split: str, root: Path | str = DEFAULT_ROOT):
        if split not in LABELED_FILES:
            raise ValueError(
                f"split '{split}' is not loadable as a labeled T26E dataset: "
                f"only {tuple(LABELED_FILES)} (test labels are sealed)"
            )
        self.split = split
        self.root = Path(root)
        rows = _read_jsonl_guarded(self.root / LABELED_FILES[split])
        problems = [
            p
            for row in rows
            for p in validate_record_structure(row, expect_targets=True)
        ]
        if problems:
            raise ValueError(f"invalid records in {split}: {problems[:5]}")
        if any(r["split"] != split for r in rows):
            raise ValueError(f"record with wrong split found in {LABELED_FILES[split]}")
        self.records = [T26ERecord.from_dict(row) for row in rows]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> T26ERecord:
        return self.records[i]

    def score_target(self, i: int) -> float:
        assert self.records[i].targets is not None
        return float(
            self.records[i].targets["official_score"]["official_alpasim_scene_score"]
        )

    def groups(self) -> dict[str, list[T26ERecord]]:
        out: dict[str, list[T26ERecord]] = {}
        for rec in self.records:
            out.setdefault(rec.decision_group_id, []).append(rec)
        for gid, mem in out.items():
            mem.sort(key=lambda r: r.candidate_index)
            if [m.candidate_index for m in mem] != list(range(len(mem))):
                raise ValueError(f"non-contiguous candidate indices in group {gid}")
        return out


class T26ETestInputDataset:
    """Sealed-test input loader: records carry no targets of any kind."""

    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)
        rows = _read_jsonl_guarded(self.root / "t26e_test_inputs.jsonl")
        problems = [
            p
            for row in rows
            for p in validate_record_structure(row, expect_targets=False)
        ]
        if problems:
            raise ValueError(f"invalid test-input records: {problems[:5]}")
        self.records = [T26ERecord.from_dict(row) for row in rows]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> T26ERecord:
        return self.records[i]


class T26EGroupIndex:
    """Decision-group index (structural only; carries no score information)."""

    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)
        self.groups = _read_jsonl_guarded(self.root / "t26e_decision_group_index.jsonl")

    def by_split(self, split: str) -> list[dict[str, Any]]:
        return [g for g in self.groups if g["split"] == split]

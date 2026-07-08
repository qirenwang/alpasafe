# T09 - World model V1

Status: completed
Started: 2026-06-03 16:12 EDT
Completed: 2026-06-03 16:19 EDT
Owner: Codex

## Goal

Train baseline latent consequence model that predicts risk probabilities and
future safety curves.

## Scientific reason

V1 must validate action-conditioned consequence prediction without blocking on a
large generative video model.

## Inputs and assumptions

- Uses 96 dry-run WorldTarget records.
- Labels are deterministic predicate proxies.

## Files expected to be created or modified

- `src/safeworld/world_model/model_v1.py`
- `src/safeworld/world_model/losses.py`
- `src/safeworld/world_model/train.py`
- `src/safeworld/world_model/predict.py`

## Implementation plan

1. Extract engineered trajectory/reasoning/tag features.
2. Train a logistic risk model.
3. Save model JSON and E3 report.

## Acceptance criteria

Training runs on a small subset and saves metrics/checkpoint/config snapshot.

## Attempt log

### Attempt 1 - 2026-06-03 16:12 EDT
- Commands run: `PYTHONPATH=src python -m safeworld.world_model.train --config configs/world_model_v1.yaml`; `pytest`.
- Files changed: V1 model, losses, train, predict modules.
- Tests run: `pytest`.
- Results: trained engineered logistic action-conditioned risk model over candidate trajectory, reasoning keywords, and scenario tags.
- Problems: dry-run data is small and labels are deterministic, so high AUROC is a pipeline validation result, not a scientific claim.
- Decisions: keep V1 lightweight and action-conditioned; do not block on video generation.
- Next action: implement auditor and gate.

## Completion summary

Completed: 2026-06-03 16:19 EDT
Git diff summary: V1 model, training, and prediction modules added.
Commands run: training command above; `pytest`.
Test results: 7 passed.
Generated artifacts: `outputs/models/world_model_v1.json`, `outputs/reports/e3_world_model_v1.md`.
Known limitations: no held-out real validation set yet; dry-run E3 AUROC/AUPRC are 1.0000 on synthetic proxy labels.
Next task recommendation: T10 auditor and calibration.


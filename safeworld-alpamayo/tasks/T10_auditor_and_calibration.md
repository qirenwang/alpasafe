# T10 - Auditor, calibration, and gate

Status: completed
Started: 2026-06-03 16:13 EDT
Completed: 2026-06-03 16:19 EDT
Owner: Codex

## Goal

Implement calibrated risk auditor, ECE/Brier metrics, threshold selection, and
accept/reject gate.

## Scientific reason

The safety certificate should be calibrated and interpretable enough to support
runtime assurance decisions.

## Inputs and assumptions

- Uses dry-run model probabilities and predicate margins.
- Gate thresholds are config-driven.

## Files expected to be created or modified

- `src/safeworld/auditor/risk_model.py`
- `src/safeworld/auditor/calibration.py`
- `src/safeworld/auditor/gate.py`
- `src/safeworld/eval/metrics.py`
- `tests/test_metrics.py`

## Implementation plan

1. Add Brier/ECE/AUROC/AUPRC/FSR metrics.
2. Add reliability and threshold helpers.
3. Add gate decision dataclass and policy.
4. Validate through open-loop report.

## Acceptance criteria

Gate logs every decision and reason.

## Attempt log

### Attempt 1 - 2026-06-03 16:13 EDT
- Commands run: `PYTHONPATH=src python -m safeworld.eval.open_loop_eval --config configs/eval_open_loop.yaml`; `pytest`.
- Files changed: auditor, calibration, gate, metrics, metrics test.
- Tests run: `pytest`.
- Results: implemented Brier, ECE, reliability bins, recall threshold helper, calibrated-risk helper, and accept/reject gate.
- Problems: validation threshold selection is implemented as a helper but E1 used configured threshold 0.45 for this dry-run.
- Decisions: gate records accepted/rejected, reason, risk, min margin, RAWC, and selected fallback candidate.
- Next action: run open-loop experiments.

## Completion summary

Completed: 2026-06-03 16:19 EDT
Git diff summary: calibration, metrics, and gate modules added.
Commands run: open-loop eval command above; `pytest`.
Test results: 7 passed.
Generated artifacts: gate decision table in `outputs/reports/e1_open_loop_safety.md`.
Known limitations: calibrated threshold needs a real validation split for scientific reporting.
Next task recommendation: T11 open-loop experiments.


# T11 - Open-loop experiments

Status: completed
Started: 2026-06-03 16:14 EDT
Completed: 2026-06-03 16:19 EDT
Owner: Codex

## Goal

Run E0-E4 reports comparing Alpamayo alone, rule checkers, RAWC, and SafeWorld
V1.

## Scientific reason

This evaluates whether the auditor reduces false-safe actions and whether
action-conditioned risk changes across candidate trajectories.

## Inputs and assumptions

- 12 dry-run synthetic samples.
- 96 action-conditioned world targets.
- Labels are deterministic predicate proxies.

## Files expected to be created or modified

- `src/safeworld/eval/open_loop_eval.py`
- `src/safeworld/eval/closed_loop_eval.py`
- `src/safeworld/eval/ablations.py`
- `outputs/reports/e1_open_loop_safety.md`
- `outputs/reports/e2_rawc.md`
- `outputs/reports/e4_counterfactuals.md`
- `outputs/reports/e6_long_tail.md`

## Implementation plan

1. Compare Alpamayo alone, rules, RAWC, and SafeWorld V1.
2. Record FSR, unsafe recall, safe precision, AUROC/AUPRC, Brier, ECE.
3. Report counterfactual risk and long-tail breakdown.

## Acceptance criteria

Reports include false-safe rate, unsafe recall, AUROC/AUPRC, ECE, and OOD
breakdown.

## Attempt log

### Attempt 1 - 2026-06-03 16:14 EDT
- Commands run: `PYTHONPATH=src python -m safeworld.eval.open_loop_eval --config configs/eval_open_loop.yaml`; `PYTHONPATH=src python -m safeworld.eval.closed_loop_eval --config configs/eval_closed_loop.yaml`; `pytest`.
- Files changed: open-loop eval, closed-loop eval, ablations.
- Tests run: `pytest`.
- Results: generated E1/E2/E4/E6 reports. E1 SafeWorld V1: FSR 0.0000, unsafe recall 1.0000, AUROC 1.0000, ECE 0.0390 on 12 dry-run samples. E4 shows action-conditioned risk changes; `tau_stop` risk 0.0098 vs `tau_alpamayo` 0.7979.
- Problems: deterministic proxy labels make metrics optimistic.
- Decisions: report dry-run limitations explicitly in each report.
- Next action: configure real data/model or simulator/replay oracle.

## Completion summary

Completed: 2026-06-03 16:19 EDT
Git diff summary: open-loop and replay evaluation modules added.
Commands run: open-loop eval, closed-loop eval, and pytest commands above.
Test results: 7 passed.
Generated artifacts: `outputs/reports/e1_open_loop_safety.md`, `e2_rawc.md`, `e4_counterfactuals.md`, `e6_long_tail.md`.
Known limitations: E1-E6 are dry-run validation reports, not final scientific evidence.
Next task recommendation: T12 closed-loop/replay gate or real-data rerun.


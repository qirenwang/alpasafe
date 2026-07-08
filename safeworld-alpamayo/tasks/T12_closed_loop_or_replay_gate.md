# T12 - Closed-loop or replay gate

Status: completed
Started: 2026-06-03 16:18 EDT
Completed: 2026-06-03 16:19 EDT
Owner: Codex

## Goal

Evaluate gate in AlpaSim if available; otherwise implement offline replay gate
approximation and document limitation.

## Scientific reason

Closed-loop or replay validation is needed before making simulator-backed
safety-progress claims.

## Inputs and assumptions

- AlpaSim/NuRec closed-loop access was not configured.
- Dry-run artifacts can still validate gate wiring.

## Files expected to be created or modified

- `src/safeworld/eval/closed_loop_eval.py`
- `configs/eval_closed_loop.yaml`
- `outputs/reports/e5_closed_loop_or_replay_gate.md`

## Implementation plan

1. Check configured simulator/replay mode.
2. If unavailable, generate limitation report.
3. Keep gate integration ready for future replay/simulator runs.

## Acceptance criteria

Produces closed-loop/replay report with limitation stated clearly.

## Attempt log

### Attempt 1 - 2026-06-03 16:18 EDT
- Commands run: `PYTHONPATH=src python -m safeworld.eval.closed_loop_eval --config configs/eval_closed_loop.yaml`; `pytest`.
- Files changed: closed-loop/replay eval module and config.
- Tests run: `pytest`.
- Results: produced an E5 report documenting that AlpaSim/NuRec closed-loop access is not configured and offline replay/gate artifacts are available through E1/E4.
- Problems: no simulator or replay oracle accessible in this environment.
- Decisions: do not fabricate closed-loop collision/progress metrics; document limitation.
- Next action: add AlpaSim/NuRec integration when accessible.

## Completion summary

Completed: 2026-06-03 16:19 EDT
Git diff summary: closed-loop/replay placeholder report added.
Commands run: closed-loop eval command above; `pytest`.
Test results: 7 passed.
Generated artifacts: `outputs/reports/e5_closed_loop_or_replay_gate.md`.
Known limitations: no real closed-loop metrics without AlpaSim/NuRec access.
Next task recommendation: T13 report and figures after real-data or simulator run.


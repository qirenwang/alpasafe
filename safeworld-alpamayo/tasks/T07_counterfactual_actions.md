# T07 - Counterfactual candidate trajectories

Status: completed
Started: 2026-06-03 16:11 EDT
Completed: 2026-06-03 16:19 EDT
Owner: Codex

## Goal

Generate slow, stop, conservative yield, left/right perturbation, speed-up, and
lane-keep candidate trajectories.

## Scientific reason

The world model must be action-conditioned: for the same scene, changing the
candidate trajectory should change predicted future risk/consequence.

## Inputs and assumptions

- Candidate trajectories preserve waypoint count and timing.
- `tau` is the proposal notation for trajectory.

## Files expected to be created or modified

- `src/safeworld/counterfactuals/trajectory_perturb.py`
- `src/safeworld/counterfactuals/candidate_set.py`

## Implementation plan

1. Add trajectory perturbation functions.
2. Generate 8 candidates per proposal.
3. Verify through target building and E4 report.

## Acceptance criteria

All candidate trajectories preserve timing/shape and pass basic smoothing checks.

## Attempt log

### Attempt 1 - 2026-06-03 16:11 EDT
- Commands run: `PYTHONPATH=src python -m safeworld.world_model.dataset --config configs/world_model_v1.yaml --limit 96`; `PYTHONPATH=src python -m safeworld.eval.open_loop_eval --config configs/eval_open_loop.yaml`; `pytest`.
- Files changed: counterfactual perturbation and candidate-set modules.
- Tests run: `pytest`.
- Results: generated 8 candidates per sample: Alpamayo, slow, stop, conservative yield, left/right perturb, speed-up, smooth lane keep.
- Problems: first stop implementation was a hard truncation and made `tau_stop` unsafe through jerk/comfort.
- Decisions: changed `tau_stop` to a quintic smootherstep early-stop trajectory; final `tau_stop` mean predicted risk is 0.0098 and unsafe proxy rate is 0.0000.
- Next action: build world model dataset.

## Completion summary

Completed: 2026-06-03 16:19 EDT
Git diff summary: counterfactual generator added and stop smoothing corrected.
Commands run: world-target, eval, and pytest commands above.
Test results: 7 passed.
Generated artifacts: counterfactual targets in `outputs/world_targets/sample_targets.jsonl`, `outputs/reports/e4_counterfactuals.md`.
Known limitations: counterfactual oracle risk is rule-based, not simulator-derived.
Next task recommendation: T08 world model dataset.


# T05 - Safety predicates

Status: completed
Started: 2026-06-03 16:10 EDT
Completed: 2026-06-03 16:19 EDT
Owner: Codex

## Goal

Implement kinematic, TTC, RSS-like, offroad, min-distance, comfort, and progress
predicates.

## Scientific reason

The auditor should include interpretable safety margins, not only a black-box
classifier.

## Inputs and assumptions

- Trajectories are 64 waypoint `[x, y]` sequences in dry-run mode.
- Object tracks are simplified position sequences.

## Files expected to be created or modified

- `src/safeworld/geometry/trajectory.py`
- `src/safeworld/geometry/kinematics.py`
- `src/safeworld/geometry/safety_predicates.py`
- `src/safeworld/geometry/occupancy.py`
- `tests/test_trajectory.py`
- `tests/test_safety_predicates.py`

## Implementation plan

1. Implement trajectory derivatives and curvature.
2. Implement predicate result dataclass.
3. Implement required predicate functions.
4. Add unit tests for safe/unsafe synthetic cases.

## Acceptance criteria

Unit tests cover simple safe/unsafe synthetic trajectories.

## Attempt log

### Attempt 1 - 2026-06-03 16:10 EDT
- Commands run: `pytest`; `PYTHONPATH=src python -m safeworld.world_model.dataset --config configs/world_model_v1.yaml --limit 96`.
- Files changed: geometry and safety predicate modules plus tests.
- Tests run: `pytest`.
- Results: implemented acceleration, jerk, curvature, offroad, min-distance, TTC, RSS-like, progress, and comfort predicates.
- Problems: initial hard-stop counterfactual triggered jerk/comfort; addressed in T07 with smoother stop.
- Decisions: `unsafe` labels exclude progress-only failure but retain comfort/kinematic violations.
- Next action: implement RAWC.

## Completion summary

Completed: 2026-06-03 16:19 EDT
Git diff summary: geometry and safety predicate modules added with tests.
Commands run: `pytest`; world-target build command above.
Test results: 7 passed.
Generated artifacts: predicate results embedded in `outputs/world_targets/sample_targets.jsonl`.
Known limitations: lane/offroad and RSS checks are simplified dry-run approximations.
Next task recommendation: T06 RAWC.


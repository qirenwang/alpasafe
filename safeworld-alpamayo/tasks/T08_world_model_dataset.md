# T08 - World model dataset builder

Status: completed
Started: 2026-06-03 16:12 EDT
Completed: 2026-06-03 16:19 EDT
Owner: Codex

## Goal

Build action-conditioned consequence samples linking observation, reasoning,
candidate trajectory, and future risk labels.

## Scientific reason

This creates the supervised targets for the latent safety world model.

## Inputs and assumptions

- Uses mined sample index and dry-run Alpamayo proposals.
- Uses deterministic predicates as proxy labels in dry-run mode.

## Files expected to be created or modified

- `src/safeworld/world_model/dataset.py`
- `src/safeworld/geometry/occupancy.py`

## Implementation plan

1. Load samples and proposals.
2. Generate candidate trajectories.
3. Evaluate safety predicates per candidate.
4. Save WorldTarget JSONL.

## Acceptance criteria

Outputs a training JSONL file with `sample_id`, `candidate_id`, candidate
trajectory, inputs, and targets.

## Attempt log

### Attempt 1 - 2026-06-03 16:12 EDT
- Commands run: `PYTHONPATH=src python -m safeworld.world_model.dataset --config configs/world_model_v1.yaml --limit 96`; `pytest`.
- Files changed: world-target builder and path-aligned occupancy helper.
- Tests run: `pytest`.
- Results: built 96 action-conditioned `WorldTarget` records from 12 samples x 8 candidates.
- Problems: future occupancy is path-aligned binary occupancy, not semantic BEV.
- Decisions: include predicate results, min-distance/TTC curves, scenario tags, reasoning text, and object count in target metadata.
- Next action: train world model V1.

## Completion summary

Completed: 2026-06-03 16:19 EDT
Git diff summary: world-target builder added.
Commands run: world-target command above; `pytest`.
Test results: 7 passed.
Generated artifacts: `outputs/world_targets/sample_targets.jsonl` with 96 records.
Known limitations: targets are deterministic predicate proxies in dry-run mode.
Next task recommendation: T09 world model V1.


# T04 - Long-tail and OOD mining

Status: completed
Started: 2026-06-03 16:10 EDT
Completed: 2026-06-03 16:19 EDT
Owner: Codex

## Goal

Mine scenario tags from metadata, reasoning text, trajectory heuristics, and
available outcome labels.

## Scientific reason

Long-tail/OOD evaluation tests whether the world model and auditor provide more
value in challenging scenarios.

## Inputs and assumptions

- Dry-run samples include navigation text, object tracks, and synthetic future trajectories.
- No dataset-provided OOD labels were available in this pass.

## Files expected to be created or modified

- `src/safeworld/data/scenario_mining.py`
- `configs/data_physicalai.yaml`

## Implementation plan

1. Add keyword/object/trajectory heuristics.
2. Write mined index.
3. Write scenario counts and split summary.

## Acceptance criteria

Produces ID/OOD-style category counts and split files.

## Attempt log

### Attempt 1 - 2026-06-03 16:10 EDT
- Commands run: `PYTHONPATH=src python -m safeworld.data.scenario_mining --config configs/data_physicalai.yaml --limit 12`; `pytest`.
- Files changed: scenario mining module and data config outputs.
- Tests run: `pytest`.
- Results: mined tags for 12 samples and wrote category counts plus train/val/test split summary.
- Problems: no dataset-provided OOD labels available in dry-run mode.
- Decisions: use text/object/trajectory heuristics for first-pass long-tail tags.
- Next action: implement safety predicates.

## Completion summary

Completed: 2026-06-03 16:19 EDT
Git diff summary: scenario mining and count outputs added.
Commands run: scenario-mining command above; `pytest`.
Test results: 7 passed.
Generated artifacts: `outputs/index/sample_index_mined.jsonl`, `outputs/index/scenario_counts.json`, `outputs/index/split_summary.json`.
Known limitations: OOD tags are heuristic synthetic proxies.
Next task recommendation: T05 safety predicates.


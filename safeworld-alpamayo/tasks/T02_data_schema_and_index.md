# T02 - Data schema and index

Status: completed
Started: 2026-06-03 16:09 EDT
Completed: 2026-06-03 16:19 EDT
Owner: Codex

## Goal

Implement `DrivingSample`, `AlpamayoProposal`, `WorldTarget` schemas and a
dry-run/sample dataset index builder.

## Scientific reason

The world model and auditor need a reproducible schema connecting observations,
reasoning, trajectories, candidates, and future consequence labels.

## Inputs and assumptions

- Use synthetic samples until licensed PhysicalAI-AV access is configured.
- Preserve JSONL artifacts for traceability.

## Files expected to be created or modified

- `src/safeworld/data/schema.py`
- `src/safeworld/data/build_index.py`
- `src/safeworld/data/loaders.py`
- `src/safeworld/data/splits.py`
- `tests/test_schema.py`

## Implementation plan

1. Define dataclass schemas with validation.
2. Generate synthetic dry-run samples.
3. Write JSONL index.
4. Test schema round trip.

## Acceptance criteria

Index builder produces a valid JSONL sample index in `--dry-run` or `--limit`
mode.

## Attempt log

### Attempt 1 - 2026-06-03 16:09 EDT
- Commands run: `PYTHONPATH=src python -m safeworld.data.build_index --config configs/data_physicalai.yaml --dry-run --limit 12`; `pytest`.
- Files changed: data schema, loaders, split helper, synthetic index builder, schema tests.
- Tests run: `pytest`.
- Results: generated 12 synthetic `DrivingSample` rows at `outputs/index/sample_index.jsonl`.
- Problems: none.
- Decisions: schema uses dataclasses and explicit validation.
- Next action: implement Alpamayo dry-run proposal wrapper.

## Completion summary

Completed: 2026-06-03 16:19 EDT
Git diff summary: schema, loaders, synthetic index builder, schema tests added.
Commands run: build-index command above; `pytest`.
Test results: 7 passed.
Generated artifacts: `outputs/index/sample_index.jsonl` with 12 records.
Known limitations: synthetic scenes stand in for licensed PhysicalAI-AV data.
Next task recommendation: T03 Alpamayo inference wrapper.


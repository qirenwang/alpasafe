# T03 - Alpamayo inference wrapper

Status: completed
Started: 2026-06-03 16:10 EDT
Completed: 2026-06-03 16:19 EDT
Owner: Codex

## Goal

Implement a frozen Alpamayo 1.5 wrapper that reads normalized samples and stores
reasoning plus trajectory proposals.

## Scientific reason

SafeWorld treats Alpamayo as a frozen VLA planner/proposer and audits its
proposed action before execution.

## Inputs and assumptions

- Real 10B model loading requires licensed access and GPU runtime.
- This pass validates schema and provenance in dry-run mode.

## Files expected to be created or modified

- `src/safeworld/alpamayo/prompts.py`
- `src/safeworld/alpamayo/wrapper.py`
- `src/safeworld/alpamayo/run_inference.py`
- `src/safeworld/alpamayo/parse_outputs.py`

## Implementation plan

1. Build prompts from samples.
2. Add dry-run wrapper producing reasoning text and 64-waypoint trajectories.
3. Save proposals to JSONL.
4. Write E0 report.

## Acceptance criteria

Wrapper supports `--dry-run`, records model version, latency, prompt, and raw
output path/provenance.

## Attempt log

### Attempt 1 - 2026-06-03 16:10 EDT
- Commands run: `PYTHONPATH=src python -m safeworld.alpamayo.run_inference --config configs/inference_alpamayo.yaml --dry-run --limit 12`; `pytest`.
- Files changed: Alpamayo prompt/wrapper/run/parse modules.
- Tests run: `pytest`.
- Results: generated 12 dry-run `AlpamayoProposal` records with prompt, reasoning, 64-waypoint trajectory, model name, latency, and provenance.
- Problems: real 10B model not loaded by design.
- Decisions: wrapper raises for non-dry-run unless licensed model/GPU runtime are explicitly configured.
- Next action: mine long-tail tags.

## Completion summary

Completed: 2026-06-03 16:19 EDT
Git diff summary: Alpamayo dry-run wrapper and E0 report generation added.
Commands run: inference command above; `pytest`.
Test results: 7 passed.
Generated artifacts: `outputs/proposals/alpamayo_sample.jsonl`, `outputs/reports/e0_reproduction.md`.
Known limitations: E0 validates schema/profiling only, not real model quality.
Next task recommendation: T04 long-tail mining.


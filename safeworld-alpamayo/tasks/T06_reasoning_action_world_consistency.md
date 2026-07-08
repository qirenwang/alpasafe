# T06 - Reasoning-Action-World Consistency

Status: completed
Started: 2026-06-03 16:11 EDT
Completed: 2026-06-03 16:19 EDT
Owner: Codex

## Goal

Parse Alpamayo reasoning into claims and compute RAWC contradiction scores.

## Scientific reason

RAWC checks whether the reasoning trace, proposed action, and predicted/observed
world risk form a consistent causal chain.

## Inputs and assumptions

- Dry-run reasoning text is available in proposal JSONL.
- First version uses keyword-based claim extraction.

## Files expected to be created or modified

- `src/safeworld/reasoning/claim_schema.py`
- `src/safeworld/reasoning/parse_claims.py`
- `src/safeworld/reasoning/rawc.py`
- `tests/test_rawc.py`

## Implementation plan

1. Parse perception/risk/action/uncertainty claims.
2. Score contradiction types.
3. Report RAWC score and contradiction list.
4. Add unit tests.

## Acceptance criteria

RAWC tests cover yield/stop/keep-lane contradiction logic.

## Attempt log

### Attempt 1 - 2026-06-03 16:11 EDT
- Commands run: `pytest`; `PYTHONPATH=src python -m safeworld.eval.open_loop_eval --config configs/eval_open_loop.yaml`.
- Files changed: reasoning parser, RAWC scorer, RAWC test.
- Tests run: `pytest`.
- Results: RAWC detects yield-without-slowdown, stop-with-large-displacement, keep-lane drift, hazard mismatch, and critical hazard omission.
- Problems: keyword parser is intentionally simple for V1.
- Decisions: RAWC score is `1 - weighted_contradiction_score`.
- Next action: implement counterfactual candidates.

## Completion summary

Completed: 2026-06-03 16:19 EDT
Git diff summary: reasoning parser and RAWC scorer added.
Commands run: `pytest`; open-loop eval command above.
Test results: 7 passed.
Generated artifacts: `outputs/reports/e2_rawc.md`.
Known limitations: claim extraction is keyword based and not yet an LLM/NLP parser.
Next task recommendation: T07 counterfactual actions.


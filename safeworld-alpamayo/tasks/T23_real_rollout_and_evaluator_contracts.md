# T23 - Real rollout, future-state, reward, and evaluator contracts

Status: completed
Started: 2026-06-20 00:45 EDT
Completed: 2026-06-20 00:56 EDT
Owner: Claude Code

## Goal

Define schemas that prevent proxy-label leakage and clearly separate prediction
from independent evaluation.

## Scientific reason

If predicted reward and evaluator reward can be silently mixed, or one logged
future is duplicated across counterfactual candidates, results are not valid
evidence. Schemas must make these errors structurally impossible.

## Inputs and assumptions

- v2 label vocabulary: alpasim_closed_loop, nurec_candidate_replay,
  fixed_agent_replay, logged_executed_action_only, proxy_test_only.
- SCDS must NOT be locked until T22 confirms metric semantics (T22 = BLOCKED, so
  SCDS stays unfrozen).

## Files expected to be created or modified

- src/safeworld/data/schema_v2.py (all v2 dataclasses + validators)
- src/safeworld/eval/evaluator_v2.py (independent evaluator contract)
- tests/test_schema_v2.py

## Implementation plan

1. Add dataclasses: CandidateRolloutRequest/Record, FutureConsequenceTarget,
   (re-export FutureConsequencePrediction), RewardComponents,
   (re-export PredictedRewardComponents), EvaluationRewardComponents,
   SelectionResultV2, WeightProvenance.
2. Enforce: proxy_test_only barred from real CLI; no logged-label duplication;
   predicted vs evaluation reward distinct types; weight provenance hashes.
3. Evaluator exposes component metrics only; SCDS unlocked.
4. Tests for coord/horizon, label restrictions, separation, proxy block, JSONL
   round trip, deterministic argmax tie-break (in test_v2_interfaces.py).

## Acceptance criteria

- All schemas/contracts documented. [met]
- No scientific metric can silently use proxy_test_only. [met:
  `assert_real_label_source`, `evaluate_record(allow_proxy=False)`]
- Existing and new tests pass. [met]
- This log complete. [met]

## Attempt log

### Attempt 1 - 2026-06-20 00:45 EDT
- Commands run: `python -m pytest -q` -> 47 passed.
- Files changed: schema_v2.py, evaluator_v2.py, tests/test_schema_v2.py.
- Tests run: coordinate/horizon validation, label-source restriction, proxy
  block in real eval, predicted/evaluation separation, no logged duplication,
  JSONL round trip, weight provenance.
- Results: all green. Deterministic selector argmax/tie-break covered in
  tests/test_v2_interfaces.py (T21).
- Problems: a record test forgot to call `.validate()`; fixed.
- Decisions: SCDS left unlocked (`SCDS_LOCKED = False`) pending a GO from T22.
- Next action: T24 (will be BLOCKED).

## Completion summary

Completed: 2026-06-20 00:56 EDT
Git diff summary: +src/safeworld/data/schema_v2.py,
  +src/safeworld/eval/evaluator_v2.py, +tests/test_schema_v2.py.
Commands run: `python -m pytest -q`.
Test results: 47 passed.
Generated artifacts: v2 schema module, evaluator contract, tests.
Known limitations: evaluator component semantics are provisional until a real
  backend (T22 GO) fixes them; `final_reward` aggregate is diagnostic only.
Next task recommendation: T24.

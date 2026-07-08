# T21 - Method design freeze and code-path separation

Status: completed
Started: 2026-06-20 00:30 EDT
Completed: 2026-06-20 00:56 EDT
Owner: Claude Code

## Goal

Freeze the scientifically valid SafeWorld architecture (ConsequenceWorldModel ->
RewardHead -> RewardSelector) and separate it from legacy dry-run selector code.

## Scientific reason

The dry-run selectors (T16) and engineered V1 model (T09) are engineering
validation only. Before any real labels or training, the deployable method and
its naming rules must be frozen so a direct reward classifier is never mislabeled
a world model, and the diagnostic upper bound is never called an `oracle`.

## Inputs and assumptions

- Non-negotiable decisions #1-#10 (preserve history, K in {2,5,10}, frozen
  Alpamayo proposer, separate WorldModel/RewardHead/RewardSelector, native-only
  main set, no `oracle` naming, proxy labels test-only).
- No training this round (decision #11).

## Files expected to be created or modified

- docs/method_spec_v2.md, docs/terminology_v2.md,
  docs/related_work_collision_matrix.md, docs/data_and_label_assumptions.md
- src/safeworld/world_model/base.py (ConsequenceWorldModel Protocol)
- src/safeworld/reward/base.py + src/safeworld/reward/__init__.py (RewardHead)
- src/safeworld/selection/reward_selector.py (RewardSelector)
- src/safeworld/world_model/__init__.py, src/safeworld/selection/__init__.py (exports)
- tests/test_v2_interfaces.py

## Implementation plan

1. Add v2 abstract interfaces with NO training implementation and NO fake model.
2. Add deterministic RewardSelector (argmax predicted reward, native-only,
   documented tie-breaking).
3. Write the four design docs; mark legacy code `legacy_dry_run`.
4. Unit-test Protocols and selector; keep all existing tests green.

## Acceptance criteria

- Documentation internally consistent. [met]
- Interfaces type-checked (runtime_checkable Protocols) and unit-tested. [met]
- No fake world-model implementation introduced. [met]
- Existing tests still pass. [met]
- This log contains exact commands and files changed. [met]

## Attempt log

### Attempt 1 - 2026-06-20 00:30 EDT
- Commands run:
  - `python -m pytest -q` (baseline) -> 31 passed
  - inspected `src/safeworld/data/schema.py`, `selection/selectors.py`, init files
- Files changed: see "Files expected" above (all created/updated).
- Tests run: `python -m pytest -q` -> 47 passed (31 prior + 16 new).
- Results: v2 interfaces + reward selector + docs landed; legacy code retained.
- Problems: none (one test initially asserted on a constructor that does not
  auto-validate; fixed test to call `.validate()`).
- Decisions: legacy `OracleBestInKSelector` kept (decision #2) but documented as
  `legacy_dry_run`; v2 replacement is the `best_achievable_in_sampled_set_by_evaluator`
  diagnostic only.
- Next action: T22 audit.

## Completion summary

Completed: 2026-06-20 00:56 EDT
Git diff summary: +docs/method_spec_v2.md, +docs/terminology_v2.md,
  +docs/related_work_collision_matrix.md, +docs/data_and_label_assumptions.md,
  +src/safeworld/world_model/base.py, +src/safeworld/reward/{__init__,base}.py,
  +src/safeworld/selection/reward_selector.py, ~world_model/__init__.py,
  ~selection/__init__.py, +tests/test_v2_interfaces.py.
Commands run: `python -m pytest -q`.
Test results: 47 passed.
Generated artifacts: four docs + three interface modules + tests.
Known limitations: interfaces only; W_theta/G_phi training is T26, not now.
Next task recommendation: T22 AlpaSim/NuRec capability audit.

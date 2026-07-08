# T17 - Oracle labels and no-leakage evaluation

Status: superseded (by T23, 2026-06-20)

> Superseded: this task used the `oracle` label naming and proxy labels as a
> path to evaluation. The v2 design (T21/T23) replaces it with the v2
> label-source schema (`schema_v2.LABEL_SOURCES_V2`), bars `proxy_test_only`
> from the real CLI, and renames the diagnostic to
> `best_achievable_in_sampled_set_by_evaluator`. Log preserved per decision #2.
Started: not started
Completed: not completed
Owner: unassigned

## Goal

Separate proxy labels from real/simulator oracle labels and add leakage
auditing.

## Scientific reason

The current dry-run labels are deterministic predicate proxies. They are
useful for pipeline validation but cannot support final scientific claims.
Label provenance and feature/label leakage must be auditable before any
scientific conclusion is drawn.

## Inputs and assumptions

- Label groups: `proxy_rule_label`, `real_future_label`,
  `simulator_outcome_label`, `replay_oracle_label`.
- `label_source` field already exists on top-K targets and gate logs (T15/T16),
  currently always `proxy_rule`.
- Real/simulator/replay labels are unavailable in this environment; in that
  case the task must produce a limitation report, not fake metrics.

## Files expected to be created or modified

- `src/safeworld/eval/leakage_audit.py` (new)
- `src/safeworld/eval/open_loop_eval.py` (`--label-source` flag)
- `configs/eval_open_loop.yaml` (`eval_label_source`)
- `outputs/reports/leakage_audit.md` (generated)
- `tests/` leakage/label-separation tests

## Implementation plan

1. Add `eval_label_source: proxy_rule | real_future | simulator | replay` to
   the eval config and a `--label-source` CLI flag.
2. Make reports separate engineering dry-run metrics from scientific
   real/simulator-backed metrics; refuse to silently mix label sources.
3. Add a leakage audit listing, for every model input feature, whether it
   directly overlaps with target label computation; write
   `outputs/reports/leakage_audit.md`.
4. Enforce the hard rule that any metric computed from `proxy_rule_label`
   carries "dry-run proxy metric only; not scientific evidence".
5. When oracle labels are unavailable, emit a limitation report.

## Acceptance criteria

- Reports cannot silently mix proxy and oracle labels.
- Leakage audit file is generated.
- T17 task log is fully updated.

## Attempt log

(no attempts yet; task not started in this phase per user scope T14-T16 only)

## Completion summary

Not completed.

## Known limitations

Not started. No real/simulator/replay oracle labels are configured in this
environment.

## Next task recommendation

T18 SafeWorld top-K selector.

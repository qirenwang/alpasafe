# T19 - K sweep and recoverability evaluation

Status: superseded (by T27, 2026-06-20)

> Superseded: the recoverability/K-sweep evaluation was defined over proxy
> labels. The v2 plan (T27) runs K in {2,5,10} against **independent**
> simulator/replay metrics, not predicted/proxy reward. Log preserved per
> decision #2.

Started: not started
Completed: not completed
Owner: unassigned

## Goal

Run K sweep experiments for K = 1, 2, 5, 10.

## Scientific reason

We need to test whether increasing K improves safety recovery and whether
SafeWorld can identify safer alternatives within Alpamayo's proposal set.

## Inputs and assumptions

- Top-K proposals and targets for K=1,2,5,10 (T14/T15) and the selection
  protocol with all seven selectors (T16).
- Required metrics: top1_unsafe_rate, best_in_k_safe_availability,
  recoverable_failure_rate, safeworld_recovery_rate, selection_regret,
  false_safe_rate, unsafe_recall, progress_retention, comfort_cost,
  latency_overhead, k_returned_distribution.
- selection_regret = oracle_risk(selected_by_method) - oracle_risk(best_in_K).
- All dry-run metrics use proxy labels and must be marked as proxy dry-run;
  no scientific claim without real/simulator oracle.

## Files expected to be created or modified

- `src/safeworld/eval/topk_eval.py` (new K-sweep evaluator)
- `outputs/reports/e7_topk_k_sweep.md`
- `outputs/reports/e8_recoverability.md`
- `outputs/reports/e9_selector_comparison.md`
- tests for metric definitions

## Implementation plan

1. Implement the metric definitions above over gate logs and targets per K.
2. Run the sweep for K=1,2,5,10 in dry-run mode.
3. Generate the three reports; each must include engineering conclusion,
   scientific conclusion, limitations, label provenance (proxy vs oracle),
   K values tested, and exact commands run.

## Acceptance criteria

- K sweep runs in dry-run mode.
- Metrics are clearly marked as proxy dry-run if using proxy labels.
- No scientific claim is made without real/simulator oracle.
- T19 task log is fully updated.

## Attempt log

(no attempts yet; task not started in this phase per user scope T14-T16 only)

## Completion summary

Not completed.

## Known limitations

Not started. Depends on T17 label separation and T18 SafeWorld top-K selector.

## Next task recommendation

T20 real long-tail/OOD evaluation plan.

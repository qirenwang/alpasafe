# T20 - Real long-tail/OOD evaluation plan

Status: superseded (by T22/T25/T27, 2026-06-20)

> Superseded: the real-data integration plan is replaced by the concrete T22
> capability audit (which found `BLOCKED_MISSING_ASSETS_OR_API`) plus the T25
> dataset and T27 evaluation plans. Log preserved per decision #2.

Started: not started
Completed: not completed
Owner: unassigned

## Goal

Create a concrete plan for moving from dry-run top-K evaluation to real
Alpamayo / PhysicalAI-AV / NuRec / AlpaSim evaluation.

## Scientific reason

Publication claims require real Alpamayo outputs and real/simulator/replay
oracle labels. The current dry-run pipeline is engineering validation only.

## Inputs and assumptions

- Plan output: `outputs/reports/real_data_integration_plan.md`.
- The plan must specify: required datasets, required credentials/licenses,
  expected file paths, expected Alpamayo output format, how to cache real
  top-K outputs, how to compute real_future_label and
  simulator_outcome_label, how to evaluate long-tail/OOD categories, and
  expected GPU/memory/latency logging.
- Long-tail categories: intersection, pedestrian crossing, cut-in, blocked
  lane, construction, low light, adverse weather, ambiguous navigation,
  OOD geometry.
- First real experiment protocol: K=1,2,5,10; methods = Alpamayo top-1,
  top-K choose rank-1, top-K rule checker, top-K RAWC, top-K SafeWorld,
  Oracle best-in-K.
- Required final tables: (1) K sweep, (2) selector comparison,
  (3) long-tail/OOD breakdown, (4) latency and memory, (5) failure cases.

## Files expected to be created or modified

- `src/safeworld/eval/write_real_data_plan.py` (new)
- `outputs/reports/real_data_integration_plan.md` (generated)

## Implementation plan

1. Write the plan generator producing an actionable experiment protocol with
   all the elements above.
2. State explicitly that the current dry-run is not final evidence.

## Acceptance criteria

- The plan is written as an actionable experiment protocol.
- It explicitly says current dry-run is not final evidence.
- T20 task log is fully updated.

## Attempt log

(no attempts yet; task not started in this phase per user scope T14-T16 only)

## Completion summary

Not completed.

## Known limitations

Not started. Real Alpamayo 1.5 10B inference, PhysicalAI-AV data, NuRec, and
AlpaSim are not configured in this environment.

## Next task recommendation

Execute T17-T20 in order, then the cross-task `outputs/reports/topk_summary.md`.

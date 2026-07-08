# T27 - Real top-K evaluation

Status: planned (placeholder only — do not implement this phase)
Started: not started
Completed: not completed
Owner: unassigned

## Goal

Evaluate SafeWorld selection on real top-K with K in {2, 5, 10}.

## Scientific reason

The headline claim is whether SafeWorld selects a safer Alpamayo-native
candidate than Alpamayo's own default, judged by **independent** simulator/replay
metrics — not by predicted reward.

## Plan (not implemented)

- K in {2, 5, 10}.
- Main comparison:

  ```text
  Alpamayo default candidate
  vs
  SafeWorld-selected Alpamayo-native candidate
  ```

- Selection uses predicted reward (RewardSelector); scoring uses independent
  `evaluator_v2` / backend metrics (EvaluationRewardComponents), never predicted
  reward.
- Report `best_achievable_in_sampled_set_by_evaluator` as a diagnostic upper
  bound only (diagnostic_only=True), never as a deployable method.
- Long-tail / OOD breakdown, latency/memory, failure cases.

## Acceptance criteria (future)

- Real, independent-metric evaluation with predicted/evaluation reward strictly
  separated; no proxy labels.

## Attempt log

(none — placeholder; do not begin this phase)

## Completion summary

Not started. Blocked on T26 trained models.

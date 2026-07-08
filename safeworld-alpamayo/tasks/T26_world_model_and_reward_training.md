# T26 - World model and reward training

Status: planned (placeholder only — do not implement this phase)
Started: not started
Completed: not completed
Owner: unassigned

## Goal

Train only the ConsequenceWorldModel `W_theta` and RewardHead `G_phi`, with
Alpamayo 1.5 frozen.

## Scientific reason

The contribution requires reasoning-conditioned consequence prediction and
calibrated reward components; these must be learned from T25's candidate-
conditioned labels.

## Plan (not implemented)

- Freeze Alpamayo (proposal generator); train W_theta and G_phi only.
- Proposed objective:

  ```text
  L = lambda_state * L_future_state
    + lambda_reward * L_reward_components
    + lambda_rank  * L_pairwise_ranking
    + lambda_cal   * L_calibration
  ```

- Record `WeightProvenance` (initialization, training dataset hash, config hash,
  checkpoint hash, label_source) for every checkpoint.
- No proxy_test_only labels (barred by schema_v2 / assert_real_label_source).

## Acceptance criteria (future)

- Trained, calibrated W_theta + G_phi with full provenance; predicted reward
  kept separate from evaluation reward.

## Attempt log

(none — placeholder; do not begin this phase)

## Completion summary

Not started. Blocked on T25 dataset.

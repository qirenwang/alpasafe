# T18 - SafeWorld top-K selector

Status: superseded (by T21, 2026-06-20)

> Superseded: the dry-run SafeWorldSelector design is replaced by the v2
> `RewardSelector` (deterministic argmax over predicted reward, native-only,
> documented tie-breaking) in `src/safeworld/selection/reward_selector.py`.
> Log preserved per decision #2.

Started: not started
Completed: not completed
Owner: unassigned

## Goal

Adapt SafeWorld V1 from single/counterfactual-candidate evaluation to
Alpamayo-native top-K candidate evaluation.

## Scientific reason

SafeWorld should be evaluated as an external consequence critic and selector
over Alpamayo's candidate trajectories. SafeWorld V1 is an action-conditioned
latent consequence model, not a generative video/LiDAR world model.

## Inputs and assumptions

- SafeWorld input: scene observation summary, navigation command, Alpamayo
  candidate reasoning trace, Alpamayo candidate trajectory.
- SafeWorld output: predicted collision / close-encounter / offroad /
  rule-violation probabilities, predicted min-distance and TTC curves,
  predicted progress, predicted comfort cost, and an uncertainty estimate if
  available.
- The top-K selector ranks all Alpamayo-native candidates by predicted
  consequence risk; fallback candidates are used only if all native candidates
  are rejected (protocol from T16).

## Files expected to be created or modified

- `src/safeworld/world_model/` (top-K training/prediction over
  `topk_targets_k{k}.jsonl`)
- `src/safeworld/eval/open_loop_eval.py` (`--selector safeworld` path,
  per-candidate risk table, selected-candidate table)
- `configs/world_model_v1.yaml` (top-K training inputs)
- tests for K=1,2,5,10 ranking

## Implementation plan

1. Extend the world-model training path with `--topk` to fit on top-K targets.
2. Emit the full predicted-consequence output set per candidate.
3. Produce a per-candidate risk table and a selected-candidate table per K.
4. Verify ranking works for K=1,2,5,10.

## Acceptance criteria

- SafeWorld can rank K=1,2,5,10 candidates.
- Produces per-candidate risk table.
- Produces selected-candidate table.
- T18 task log is fully updated.

## Attempt log

(no attempts yet; task not started in this phase per user scope T14-T16 only)

## Completion summary

Not completed.

## Known limitations

Not started. V1 remains a lightweight engineered model; reports must call it
an action-conditioned latent consequence model.

## Next task recommendation

T19 K sweep and recoverability evaluation.

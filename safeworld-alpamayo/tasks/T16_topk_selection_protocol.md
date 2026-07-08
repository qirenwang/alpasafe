# T16 - Top-K selection protocol

Status: completed
Started: 2026-06-09 17:50 EDT
Completed: 2026-06-09 18:38 EDT
Owner: Claude Code

## Goal

Implement the top-K selection protocol and baseline selectors over Alpamayo's
candidate proposals.

## Scientific reason

The key experiment is not whether SafeWorld can reject one trajectory; it is
whether SafeWorld can select a safer trajectory from Alpamayo's own K
proposals. That requires a shared selection protocol and comparable baseline
selectors with full gate logging.

## Inputs and assumptions

- Top-K proposals from T14 (`outputs/proposals/alpamayo_topk_k{k}.jsonl`) and
  top-K world targets from T15 (`outputs/world_targets/topk_targets_k{k}.jsonl`).
- Selection objective:
  `argmin [safety_risk - lambda_progress*progress_score + beta_comfort*comfort_cost + gamma_uncertainty*uncertainty]`.
- If all Alpamayo-native candidates are unsafe, fall back to `fallback_stop`
  or `fallback_slow` depending on config.
- All dry-run labels are `proxy_rule`; selection metrics are dry-run proxy
  metrics only, not scientific evidence.

## Files expected to be created or modified

- `src/safeworld/selection/__init__.py`, `src/safeworld/selection/selectors.py`
  (seven selectors, `SelectionConfig`, `SelectionDecision`,
  `selection_objective`)
- `src/safeworld/eval/open_loop_eval.py` (`build_candidate_evaluations`,
  `run_topk_selection`, `--topk`/`--k` CLI)
- `configs/eval_open_loop.yaml` (`topk` and `selection` blocks)
- `src/safeworld/world_model/dataset.py` (sample-boundary limit fix)
- `tests/test_selectors.py`

## Implementation plan

1. Add `SelectionConfig` (risk_threshold, lambda_progress, beta_comfort,
   gamma_uncertainty, rawc_threshold, fallback_candidate, random_seed) and
   `SelectionDecision` carrying every required gate-log field.
2. Implement the seven selectors: Top1, RandomTopK (deterministic seeded),
   AlpamayoScore (model_score, rank fallback), RuleChecker (predicate pass),
   RAWC (consistency score with threshold fallback), SafeWorld (risk filter
   then objective argmin), OracleBestInK (proxy_oracle_only marking in
   dry-run; fallback when no safe native candidate exists).
3. Build `CandidateEvaluation` rows per candidate in `open_loop_eval`
   (predicted risk from the V1 engineered consequence model, progress,
   comfort cost, RAWC) and run all selectors per sample.
4. Write the gate log JSONL and a per-method comparison report explicitly
   marked as dry-run proxy only.

## Acceptance criteria

- All selectors produce a selected candidate and decision reason.
- Gate logs contain: sample_id, K, selected_candidate_id,
  selected_candidate_source, selection_method, predicted_risk,
  progress_score, comfort_cost, RAWC_score, decision_reason.
- `pytest` passes.
- T16 task log is fully updated.

## Attempt log

### Attempt 1 - 2026-06-09 17:50 EDT
- Commands run:
  - `PYTHONPATH=src python -m safeworld.alpamayo.run_inference --config configs/inference_alpamayo.yaml --dry-run --k 10 --limit 12`
  - `PYTHONPATH=src python -m safeworld.world_model.dataset --config configs/world_model_v1.yaml --topk --limit 120`
  - `PYTHONPATH=src python -m safeworld.eval.open_loop_eval --config configs/eval_open_loop.yaml --topk`
  - `pytest`
- Files changed: new `src/safeworld/selection/` package, open-loop eval
  extension, eval config, dataset-builder limit fix, `tests/test_selectors.py`.
- Tests run: `pytest` (31 passed) including required-gate-field checks for all
  seven selectors and fallback behavior for both configured fallbacks.
- Results: `outputs/topk/selection_log_k10.jsonl` has 70 rows
  (10 samples x 7 selectors) with all required gate-log fields;
  `outputs/reports/topk_selection_k10.md` written with per-method fallback
  rate, mean predicted risk, selected proxy-unsafe rate, mean progress, and
  mean comfort cost, marked "Dry-run proxy metric only; not scientific
  evidence."
- Problems: `--limit 120` originally truncated mid-sample (12 samples x 13
  candidates = 156 targets), leaving one sample without fallback targets and
  crashing selection with "no fallback candidate available". Fixed by honoring
  the limit at sample boundaries in `build_topk_targets` (130 targets = 10
  complete samples x 13 candidates).
- Decisions: random selector seeded via sha256 of `seed:sample_id:k` so logs
  are reproducible; SafeWorld fallback reason recorded as
  `all_native_candidates_exceed_predicted_risk_threshold`; oracle decisions
  carry `proxy_oracle_only` whenever every label_source is `proxy_rule`.
- Next action: T17 oracle labels and no-leakage evaluation.

## Completion summary

Completed: 2026-06-09 18:38 EDT
Git diff summary: new selection package (7 selectors, config, decision,
objective), top-K selection runner with gate log and report, eval config
selection block, dataset-builder sample-boundary limit fix, selector tests.
Commands run: K=10 inference, top-K target build, top-K selection eval,
`pytest`.
Test results: 31 passed.
Generated artifacts: `outputs/topk/selection_log_k10.jsonl` (70 rows),
`outputs/reports/topk_selection_k10.md`.
Dry-run observation (proxy only): top1/random_topk/alpamayo_score/rawc select
proxy-unsafe candidates on 0.80 of samples with no fallback;
rule_checker/safeworld/oracle_best_in_k fall back on 0.80 of samples with 0.0
proxy-unsafe selections. This reflects harsh synthetic predicates and is a
dry-run proxy metric only; not scientific evidence.
Known limitations: predicted risk comes from the engineered V1
action-conditioned latent consequence model on synthetic scenes; no real
Alpamayo outputs, no oracle labels; uncertainty term is 0 in dry-run.
Next task recommendation: T17 oracle labels and no-leakage evaluation.

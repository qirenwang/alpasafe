# T15 - Top-K candidate schema

Status: completed
Started: 2026-06-09 17:50 EDT
Completed: 2026-06-09 18:38 EDT
Owner: Claude Code

## Goal

Create a schema that cleanly separates (1) Alpamayo-native candidate
trajectories, (2) fallback safety trajectories, and (3) manually generated
stress-test trajectories.

## Scientific reason

The previous dry-run used manual counterfactual trajectories as the main
candidate set; that was only pipeline validation. The new scientific design
uses Alpamayo-native top-K proposals as the main candidate set, so candidate
provenance must be explicit and old artifacts must be distinguishable from new
ones.

## Inputs and assumptions

- `candidate_source` is one of `alpamayo_native | fallback | manual_stress_test
  | legacy_counterfactual_dry_run`.
- Existing T00-T12 artifacts (tau_* candidate ids in
  `outputs/world_targets/sample_targets.jsonl`) must keep loading unchanged.
- Fallbacks are backup controller candidates only, never main scientific
  candidates; manual perturbations are not used as main scientific candidates.

## Files expected to be created or modified

- `src/safeworld/data/schema.py` (`CandidateTrajectory`, `TopKAlpamayoProposal`,
  `CandidateSet`, `CandidateEvaluation`, source/mode/label-source constants)
- `src/safeworld/data/loaders.py` (top-K loaders; legacy marking on load)
- `src/safeworld/world_model/dataset.py` (`--topk` builder using Alpamayo
  top-K outputs as main candidates, fallbacks appended)
- `configs/world_model_v1.yaml` (top-K path templates)
- `tests/test_topk_schema.py`

## Implementation plan

1. Add `CandidateTrajectory` with candidate_id, candidate_source,
   candidate_rank, trajectory, reasoning_trace, optional model_score,
   raw_output_reference, metadata, and validation of the source vocabulary.
2. Add `TopKAlpamayoProposal` with provenance fields (model name/version,
   inference_mode, latency, gpu memory), native `candidates`, and separate
   `fallback_candidates`; `k_returned` must equal len(candidates).
3. Add `CandidateSet` (native/fallback accessors, `from_proposal`) and
   `CandidateEvaluation` (per-candidate predicted risk, progress, comfort,
   RAWC, label_source, outcome labels).
4. Mark legacy tau_* world targets as `legacy_counterfactual_dry_run` in
   `load_targets` without rewriting the files.
5. Add `build_topk_targets` so the world-model dataset builder consumes
   Alpamayo top-K proposals as main candidates with fallbacks appended only as
   backup controller candidates.

## Acceptance criteria

- Existing T00-T12 dry-run artifacts still load.
- New top-K artifacts validate against the schema.
- JSONL round-trip tests pass.
- T15 task log is fully updated.

## Attempt log

### Attempt 1 - 2026-06-09 17:50 EDT
- Commands run:
  - `PYTHONPATH=src python -m safeworld.data.build_index --config configs/data_physicalai.yaml --dry-run --limit 12`
  - `PYTHONPATH=src python -m safeworld.alpamayo.run_inference --config configs/inference_alpamayo.yaml --dry-run --k 10 --limit 12`
  - `pytest`
- Files changed: schema, loaders, dataset builder, world-model config, new
  `tests/test_topk_schema.py`.
- Tests run: `pytest` (31 passed) including round-trip tests for
  `CandidateTrajectory`, `TopKAlpamayoProposal`, `CandidateSet`,
  `CandidateEvaluation`, and the legacy-marking test.
- Results: 96 legacy targets in `outputs/world_targets/sample_targets.jsonl`
  load and all 96 are marked `legacy_counterfactual_dry_run`; new K=10
  proposals validate; top-K targets carry `candidate_source` in
  {alpamayo_native, fallback} only.
- Problems: none.
- Decisions: legacy marking happens at load time instead of rewriting old
  artifacts (the instruction is not to delete/modify existing dry-run
  artifacts); fallbacks live in a separate `fallback_candidates` field on the
  proposal rather than being interleaved with native candidates.
- Next action: T16 selection protocol on top of `CandidateEvaluation`.

## Completion summary

Completed: 2026-06-09 18:38 EDT
Git diff summary: four new schema dataclasses with validation and JSONL
round-trip support; source vocabulary constants; top-K loaders; load-time
legacy marking; top-K world-target builder.
Commands run: build_index dry-run, K=10 top-K inference, `pytest`.
Test results: 31 passed.
Generated artifacts: `outputs/world_targets/topk_targets_k10.jsonl` (130 rows:
10 samples x (10 native + 3 fallback)); validated
`outputs/proposals/alpamayo_topk_k{1,2,5,10}.jsonl`.
Known limitations: `manual_stress_test` is reserved in the vocabulary but no
manual stress-test candidates are generated in this phase; all labels on top-K
targets are `proxy_rule` dry-run proxies, not scientific evidence.
Next task recommendation: T16 top-K selection protocol.

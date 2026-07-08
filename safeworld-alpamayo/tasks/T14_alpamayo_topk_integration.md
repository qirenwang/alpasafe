# T14 - Alpamayo top-K integration

Status: completed
Started: 2026-06-09 17:50 EDT
Completed: 2026-06-09 18:38 EDT
Owner: Claude Code

## Goal

Modify the Alpamayo wrapper so it can represent K native Alpamayo trajectory
proposals per scene, with K in {1, 2, 5, 10}.

## Scientific reason

The new research question is: given Alpamayo's top-K trajectory proposals, can
an external action-conditioned consequence world model select a safer candidate
than the default top-1, especially in long-tail/OOD scenarios? Alpamayo is the
trajectory proposer; SafeWorld is the consequence critic and selector. The
wrapper must therefore store K native candidates per scene, not one.

## Inputs and assumptions

- K is in the deployment-supported range {1, 2, 5, 10}; K=10 is the largest
  value used in this phase, not the only one.
- Real 10B Alpamayo inference is NOT loaded. Dry-run mode generates K plausible
  Alpamayo-native-style candidates for pipeline validation only.
- Dry-run candidates are named `alpamayo_candidate_01..K` and are not labeled
  as manual counterfactuals.
- Fallback candidates (`fallback_stop`, `fallback_slow`,
  `fallback_conservative_yield`) are kept separate from native candidates.

## Files expected to be created or modified

- `configs/inference_alpamayo.yaml` (topk config block, output templates)
- `src/safeworld/alpamayo/wrapper.py` (`run_topk`, dry-run K-candidate generation,
  fallback generation, real-inference placeholder)
- `src/safeworld/alpamayo/run_inference.py` (`--k` flag, `run_topk_inference`)
- `src/safeworld/alpamayo/parse_outputs.py` (`parse_topk_proposal`)
- `tests/test_topk_wrapper.py`

## Implementation plan

1. Add `topk: {enabled, k_values, default_k, use_real_alpamayo, dry_run}` to
   `configs/inference_alpamayo.yaml`.
2. Add `AlpamayoWrapper.run_topk(sample, k)` returning a `TopKAlpamayoProposal`
   (schema from T15) storing sample_id, k_requested, k_returned,
   alpamayo_model_name, alpamayo_model_version, inference_mode, latency_ms,
   gpu_memory_mb (None in dry-run), native candidate list, and separate
   fallback candidate list.
3. Dry-run generation: rank 1 uses the scenario-conditioned base behavior;
   ranks 2..K apply deterministic per-rank scale/lateral variations plus a
   per-sample hash jitter, with synthetic logprob-style `model_score`
   (marked `synthetic_logprob_dry_run` in metadata).
4. Keep the real-inference path as an explicit RuntimeError placeholder gated
   by `dry_run` / `topk.use_real_alpamayo`.
5. Extend `run_inference.py` with `--k`; without `--k` the legacy single
   proposal path is unchanged.

## Acceptance criteria

- Running dry-run inference with K=1,2,5,10 creates valid proposal JSONL files.
- Each sample stores K native Alpamayo-style candidates.
- `pytest` passes.
- T14 task log is fully updated.

## Attempt log

### Attempt 1 - 2026-06-09 17:50 EDT
- Commands run:
  - `PYTHONPATH=src python -m safeworld.alpamayo.run_inference --config configs/inference_alpamayo.yaml --dry-run --k 1 --limit 12`
  - `PYTHONPATH=src python -m safeworld.alpamayo.run_inference --config configs/inference_alpamayo.yaml --dry-run --k 2 --limit 12`
  - `PYTHONPATH=src python -m safeworld.alpamayo.run_inference --config configs/inference_alpamayo.yaml --dry-run --k 5 --limit 12`
  - `PYTHONPATH=src python -m safeworld.alpamayo.run_inference --config configs/inference_alpamayo.yaml --dry-run --k 10 --limit 12`
  - `PYTHONPATH=src python -m safeworld.alpamayo.run_inference --config configs/inference_alpamayo.yaml --dry-run --limit 12` (legacy path regression)
  - `pytest`
- Files changed: wrapper, run_inference, parse_outputs, inference config,
  new `tests/test_topk_wrapper.py`.
- Tests run: `pytest` (31 passed); loader validation script confirming all
  four JSONL files parse as `TopKAlpamayoProposal` with the expected candidate
  ids and separate fallback list.
- Results: `outputs/proposals/alpamayo_topk_k{1,2,5,10}.jsonl` (12 rows each)
  and `outputs/reports/topk_inference_k{1,2,5,10}.md` written. k_returned == k
  for every sample. Legacy `alpamayo_sample.jsonl` path unchanged.
- Problems: none in this task (a downstream limit-truncation issue surfaced in
  T16 and was fixed there).
- Decisions: `run_topk` rejects K outside {1,2,5,10} to enforce the
  deployment-supported range; gpu_memory_mb stays None in dry-run rather than
  fabricating a number.
- Next action: T15 schema (implemented together since the wrapper returns the
  T15 `TopKAlpamayoProposal` type).

## Completion summary

Completed: 2026-06-09 18:38 EDT
Git diff summary: top-K config block; `AlpamayoWrapper.run_topk` with
deterministic dry-run K-candidate generation, separate fallback candidates, and
real-inference placeholder; `--k` CLI; `parse_topk_proposal`; wrapper tests.
Commands run: the four `--k` inference commands above, the legacy command, and `pytest`.
Test results: 31 passed.
Generated artifacts: `outputs/proposals/alpamayo_topk_k{1,2,5,10}.jsonl`,
`outputs/reports/topk_inference_k{1,2,5,10}.md`.
Known limitations: dry-run candidates are synthetic Alpamayo-native-style
proposals for pipeline validation only; they are not real Alpamayo 1.5 outputs
and no scientific claims can be made from them. gpu_memory_mb and
raw_output_reference are only populated in real mode (not yet configured).
Next task recommendation: T15 top-K candidate schema (co-implemented), then T16
selection protocol.

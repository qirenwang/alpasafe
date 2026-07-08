# T24 - Real Alpamayo K=2, one-scene smoke test

Status: blocked
Started: 2026-06-20 00:52 EDT
Completed: 2026-06-20 00:56 EDT (as BLOCKED report)
Owner: Claude Code

## Goal

Attempt the smallest real end-to-end experiment (K=2, one scene) without model
training.

## Scientific reason

A single real, plumbed-through K=2 case is the minimum proof that independent
candidate-conditioned labels can be produced. It validates the data path before
any dataset generation (T25) or training (T26).

## Inputs and assumptions

Preconditions: (a) licensed local real scene, (b) Alpamayo 1.5 weights
configured, (c) T22 established >=1 valid rollout/evaluation mode.

## Files expected to be created or modified

- outputs/real_smoke/k2_scene_manifest.json (only if GO)
- outputs/real_smoke/alpamayo_raw_k2.jsonl (only if GO)
- outputs/real_smoke/candidate_rollouts_k2.jsonl (only if GO)
- outputs/real_smoke/evaluation_components_k2.jsonl (only if GO)
- outputs/reports/real_k2_single_scene_smoke.md (only if GO)
- outputs/reports/real_k2_single_scene_smoke_BLOCKED.md (if any precondition fails)

## Implementation plan

1. Check preconditions.
2. If all met: run real K=2 inference, cache raw outputs, parse 2 native
   candidates, run T22-approved rollout from identical initial state, compute
   evaluator components, write artifacts.
3. If any precondition fails: write a transparent BLOCKED report with the exact
   operator next command. No synthetic data, no training.

## Acceptance criteria

- Real results or a transparent BLOCKED report. [met: BLOCKED report]
- Both candidates from same scene state. [n/a — blocked before rollout]
- Label source not proxy_test_only. [met — no labels fabricated]
- No fallback candidate included. [met]
- No model training. [met]
- This log complete. [met]

## Attempt log

### Attempt 1 - 2026-06-20 00:52 EDT
- Commands run: precondition checks (from T22 audit): AlpaSim/NuRec absent,
  Alpamayo checkpoint not local, no cached PhysicalAI-AV clip.
- Files changed: outputs/reports/real_k2_single_scene_smoke_BLOCKED.md.
- Tests run: n/a.
- Results: precondition (c) fails (T22 = BLOCKED_MISSING_ASSETS_OR_API);
  (b) fails (no checkpoint); (a) unverified. Smoke test cannot run.
- Problems: external assets missing; checkpoint download >1 GB unauthorized.
- Decisions: emit BLOCKED report; generate NO synthetic artifacts (decision #11).
- Next action: human operator runs the unblock steps in the BLOCKED report.

## Completion summary

Completed: 2026-06-20 00:56 EDT (BLOCKED).
Git diff summary: +outputs/reports/real_k2_single_scene_smoke_BLOCKED.md.
Commands run: precondition checks only.
Test results: n/a.
Generated artifacts: BLOCKED report (no real_smoke/*.jsonl written by design).
Known limitations: no independent rollout backend, no checkpoint, no local scene.
Next task recommendation: human runs unblock steps 1-4; then re-run T22 to flip
  to GO and execute T24. Do NOT start T25-T27 (out of scope this phase).

# T24R3 — Real Alpamayo 1.5 single-scene inference

Status: completed (standalone; NOT AlpaSim-integrated)
Date: 2026-06-21 (UTC)
Run dir: `artifacts/safeworld_alpasim_t24_retry/20260621T014528Z/`
Owner: Claude Code

## Goal
Validate that official Alpamayo 1.5 inference completes on a real clip.

## Result
- COMPLETED via the standalone `alpamayo1_5` package (the AlpaSim-integrated path is docker/sudo
  blocked — T24R2). Real reasoning + real 64-step ego_t0 trajectory produced from clip
  `030c760c-…` (`nvidia/PhysicalAI-Autonomous-Vehicles`), fully offline from cache.
- Model `nvidia/Alpamayo-1.5-10B` @ `f11cd25…`; CFG-nav disabled; seed 42; peak GPU 21.67 GiB;
  model inference 1.14 s; minADE vs logged future 0.375 m. No large download (cache delta 0).
- Cosmos-Reason2-8B weights NOT needed on this path (config only).

## Files
- `outputs/real_smoke/alpamayo15_single_candidate_raw.jsonl`
- `outputs/real_smoke/alpamayo15_single_candidate_manifest.json`
- `outputs/reports/alpamayo15_single_scene_inference.md`
- run smoke: `smoke/run_standalone_k2_inference.py`, `logs/standalone_k2_approachB.log`

## Caveat
Open-loop generation only; no simulator rollout, no evaluator metric, no rollout label.

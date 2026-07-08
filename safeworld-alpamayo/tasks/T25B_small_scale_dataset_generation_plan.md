# T25B — Small-scale dataset generation (plan + limited execution)

Status: BLOCKED (2026-07-06, UTC) — `BLOCKED_SMALL_SCALE_GENERATION`
Depends on: T25A (GO_T25_SMALL_SCALE) — `artifacts/safeworld_t24_5_asl_preflight/20260629T030551Z/`
Run dir: `artifacts/safeworld_t25b_small_scale_dataset/20260706T033402Z/`
Owner: Claude Code

## Goal
Pre-register a small-scale dataset plan and, if feasible, generate a small REAL
dataset of `(scene_id, decision_timestamp_us, candidate_id)` samples with the frozen
cached-candidate rollout semantics — **without training any model** and without
fabrication.

## What happened
1. **Plan pre-registered before generation** — `manifests/t25b_preregistered_dataset_plan.json`,
   `outputs/reports/T25B_preregistered_dataset_plan.md` (K=5, single scene, scene-level
   split, sample unit, required fields, no-training/no-fabrication confirmations).
2. **Assets inspected** — exactly **1** production scene cached locally; more scenes
   need gated ~2 GB downloads (not authorized). Only the 2 pilot cached-candidate
   rollouts match the required semantics. Docker+GPU+images+model all present.
3. **Bounded generation attempted** (K=5 forward-pass dump on the one scene) —
   **FAILED**: docker-compose GPU device reservation → driver container
   `CUDA driver initialization failed` (runtime regression; direct `docker run
   --gpus` works). HF cache delta = 0 B; no `k2_dump.json`; nothing fabricated.
4. **Pilot QA'd** as the only real data — `outputs/real_smoke/t25b_small_scale_dataset_qa.json`,
   `t25b_small_scale_dataset_manifest.json` (1 scene, 1 decision, K=2, 2/2 rollouts
   parsed, valid-mask sparsity 0.08, native names preserved, no winner, no fabrication).

## Result
- New real samples generated this run: **0** (generation blocked by runtime regression).
- Additional scenes: blocked by gated downloads. Single scene ⇒ **no leakage-free
  scene-level split** is possible (blocks T25C split-freeze regardless).
- No model trained; T26/T27 not executed; tests 79 passed; ruff clean.

## Decision
`BLOCKED_SMALL_SCALE_GENERATION` (immediate: docker-compose GPU-runtime regression +
only 1 local scene). Deeper structural blocker for T25C: `BLOCKED_SCENE_SPLIT_POLICY`
(≥2 distinct scenes required). **T26 training remains NOT AUTHORIZED.**

## Files
- plan: `manifests/t25b_preregistered_dataset_plan.json`, `outputs/reports/T25B_preregistered_dataset_plan.md`
- assets: `inspections/t25b_available_assets_inspection.md`, `manifests/t25b_available_assets.json`
- attempt: `logs/k5_dump.log`, `logs/generation_attempt_diagnosis.txt`, `code_artifacts/run_k5_dump.sh`
- QA: `outputs/real_smoke/t25b_small_scale_dataset_qa.json`, `t25b_small_scale_dataset_manifest.json`,
  `reports/t25b_dataset_qa.md`
- reports: `outputs/reports/T25B_small_scale_dataset_generation.md`, `outputs/reports/status_after_T25B.md`

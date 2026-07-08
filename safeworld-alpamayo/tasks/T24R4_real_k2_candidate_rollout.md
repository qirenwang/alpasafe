# T24R4 — Exactly K=2 native candidates + rollout

Status: COMPLETED (2026-06-22, after real Docker unblock) — K=2 generation + execution + rollouts done
Date: 2026-06-21 (UTC); re-attempted and COMPLETED 2026-06-22 (UTC)
Run dir: `artifacts/safeworld_alpasim_t24_retry/20260621T014528Z/`
Latest run dir: `artifacts/safeworld_alpasim_t24_retry/20260622T183057Z/`
Owner: Claude Code

## 2026-06-22 (later) — COMPLETED after real Docker unblock
Docker genuinely unblocked (qiren added to `docker` group; compose+buildx installed). Full Path B
executed end-to-end: official single-scene validation on `clipgt-01d503d4-…` (vavam, scene_ids, NOT
public_2601) SUCCEEDED (_complete, metrics.parquet, usdz sha256 84612ff0…, components
alpasim-base:0.87.0 + nre-ga:26.02). Real Alpamayo 1.5 closed-loop in-sim verified. K=2 native
candidates regenerated from ONE identical forward pass on the validated scene (distinct: max 6.21m,
mean L2 1.73m), cached as NEW artifact (old 030c760c candidates NOT reused). Built minimal
`CachedAlpamayoCandidateModel` (no inference/optimizer/fallback; fails on coord/horizon/frame
mismatch; reasoning as metadata). Ran candidate 0 and 1 from IDENTICAL initial state (fingerprints
EQUAL 4ed59ee444c8…). Real outputs written: candidate_rollouts_k2.jsonl, evaluation_components_k2.jsonl,
k2_initial_state_fingerprints.json, k2/alpamayo_raw_k2_alpasim_scene.jsonl, and
outputs/reports/real_k2_single_scene_smoke_COMPLETED.md. Native metric names preserved (no TTC/Comfort
rename/claim). No safety claim from one scene. Run dir: `artifacts/safeworld_alpasim_t24_retry/20260622T194814Z/`.

## 2026-06-22 re-attempt (after claimed Docker/sudo unblock)
Premise FALSE — Docker still unusable without sudo (qiren not in `docker` group; no compose
plugin; sudo password-gated). Step 1 gate FAILED → entire chain blocked. Path A (execute cached
candidates) FORBIDDEN: cached clip_id `030c760c-…` ≠ AlpaSim default `clipgt-01d503d4-…`, no
official mapping. Path B blocked (needs Step 3 validated scene → needs Docker). No outputs
fabricated; no large download; no `_COMPLETED.md`. See updated
`outputs/reports/real_k2_single_scene_smoke_BLOCKED.md` and run dir
`logs/step1_docker_gate{.log,_status.json}`. Code: `BLOCKED_DOCKER_RUNTIME_UNAVAILABLE_NO_SUDO`.

## Goal
Obtain exactly K=2 Alpamayo-native candidates from one identical observation and run each from an
identical initial simulator state.

## Result
- **Generation: DONE (approach B).** Two seeded calls `[42,1234]`, same observation (verified by
  identical input_ids + ego-history hashes), `num_traj_samples=1` each → 2 candidates, each with one
  reasoning trace + one executable trajectory, `candidate_source=alpamayo_native`, ego_t0/10Hz/64/6.4s,
  both finite, trajectories distinct (max 1.19 m, mean L2 0.56 m). Reasoning text identical across
  seeds (documented; not engineered). Raw cached: `outputs/real_smoke/alpamayo_raw_k2.jsonl`,
  `outputs/real_smoke/k2_scene_manifest.json`.
- **Execution/rollout: BLOCKED.** No AlpaSim closed loop (docker/sudo). No
  candidate_rollouts_k2.jsonl / evaluation_components_k2.jsonl / k2_initial_state_fingerprints.json
  written (would fabricate non-existent results). label_source = NONE.

## Decision
`BLOCKED_RUNTIME_INCOMPATIBILITY` for execution; native K=2 generation API validated (so NOT
`BLOCKED_NO_NATIVE_K2_GENERATION_API`).

## Files
- `outputs/real_smoke/alpamayo_raw_k2.jsonl`, `outputs/real_smoke/k2_scene_manifest.json`
- `outputs/reports/real_k2_single_scene_smoke_RETRY.md`

# T24R5 — Post-smoke semantics, provenance & reasoning-alignment correction

Status: COMPLETED (2026-06-28, UTC)
Depends on: T24R4 COMPLETED (`artifacts/safeworld_alpasim_t24_retry/20260622T194814Z/`)
Run dir: `artifacts/safeworld_t24_5_asl_preflight/20260629T030551Z/`
Owner: Claude Code

## Goal
Correct the rollout terminology, candidate-specific reasoning/provenance gaps, and config/metric
provenance left after the completed K=2 AlpaSim smoke — without retraining, without re-generating the
K=2 dataset, and without overwriting any historical artifact.

## Scope (this task)
1. Freeze precise structured **rollout semantics** (replace the too-coarse single `label_source`):
   backend, evaluation source, sim-loop/controller/physics active, ego execution mode, replanning,
   renderer, traffic, non-ego behaviour, interaction assumption. New `RolloutSemantics` object +
   backward-compatible migration; v2.1 docs supersede (preserve) the v2 docs.
2. Audit & fix the **candidate-specific reasoning alignment** ambiguity (both raw records stored the
   same 2-element reasoning array). Prove `reasoning[i] ↔ trajectory[i]` from the model source, or
   stop with `BLOCKED_REASONING_CANDIDATE_ALIGNMENT`.
3. Build the full **candidate-to-outcome chain of custody** (raw → cached candidate → driver input →
   rollout config → ASL → metrics, all SHA256-linked), an immutable `latest_success_manifest.json`,
   and preserve the external AlpaSim patch + driver/dump-hook sources.
4. Normalise **config & metric provenance**: pull `n_sim_steps/control_timestep_us/force_gt_duration_us`
   from ASL metadata; extract the real per-session random seed; reconcile `collision_at_fault`.

## Result (COMPLETED)
- **Reasoning alignment: PROVEN, deterministically, no rerun.** `Alpamayo1_5.sample_trajectories_
  from_data_with_vlm_rollout` reshapes both the trajectories and the CoT text tokens with the same
  `[B, num_traj_sets, num_traj_samples]` layout ("to match trajectory shape"); the dump-hook bug
  indexed the `num_traj_sets` axis (size 1) instead of `num_traj_samples`, so both records received
  the full 2-element array. The preserved array order IS the sample order, so:
  cand0 ↔ "Keep distance to the lead vehicle since it is stopped ahead";
  cand1 ↔ "Keep distance to the lead vehicle since it is directly ahead in our lane".
  Corrected artifact: `outputs/real_smoke/alpamayo_raw_k2_candidate_reasoning_v2_1.jsonl`.
- **Provenance chain: COMPLETE.** Dump candidate anchored at the logged t0 reproduces the driver's
  logged plan to max |Δ| = 0.000000 m (cand0) / 0.000001 m (cand1) — the cached K=2 candidates ARE
  what AlpaSim executed. `outputs/real_smoke/candidate_rollout_provenance_k2_v2_1.jsonl` +
  `latest_success_manifest.json`.
- **Semantics: frozen** (`docs/alpasim_rollout_semantics_v2_1.md`, `RolloutSemantics` object).
- **Config/metric provenance: corrected** — `n_sim_steps=80`, `control_timestep_us=100000`,
  `force_gt_duration_us=1700000` from ASL; real per-session seeds (cand0=1160834997,
  cand1=1139370374, NOT "presumed identical"); `collision_at_fault` is **absent** from native
  `metrics.parquet` and is removed/flagged (only collision_any/front/lateral/rear exist).

## Decision
`GO` for T25A future-target preflight (this run). Reasoning alignment proven; provenance complete.

## Files
- docs: `method_spec_v2_1.md`, `terminology_v2_1.md`, `data_and_label_assumptions_v2_1.md`,
  `alpasim_rollout_semantics_v2_1.md`
- code: `src/safeworld/data/rollout_semantics.py`, `src/safeworld/real_smoke/reasoning_alignment.py`
- outputs: `outputs/real_smoke/alpamayo_raw_k2_candidate_reasoning_v2_1.jsonl`,
  `candidate_rollout_provenance_k2_v2_1.jsonl`, `latest_success_manifest.json`,
  `candidate_rollouts_k2_v2_1.jsonl`, `evaluation_components_k2_v2_1.jsonl`
- report: `outputs/reports/T24R5_post_smoke_semantics_and_provenance.md`

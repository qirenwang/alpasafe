# T25A — Single-scene FutureConsequenceTarget preflight (ASL extraction)

Status: COMPLETED (2026-06-28, UTC)
Depends on: T24R5 (semantics + provenance + reasoning alignment)
Run dir: `artifacts/safeworld_t24_5_asl_preflight/20260629T030551Z/`
Owner: Claude Code

## Goal
Convert the two completed K=2 AlpaSim rollouts into schema-valid, fully auditable
`FutureConsequenceTarget` pilot records containing **real** structured future-state targets parsed
from the AlpaSim `.asl` logs — without fabricating any label and without starting full T25 dataset
generation.

## Method (real data only)
- Parse `rollout.asl` for cand0/cand1 with the official reader
  `alpasim_utils.logs.async_read_pb_log` (no OCR / text scraping).
- Inventory message types (counts, never assumed); extract `RolloutMetadata`, every `ActorPoses`
  frame (raw global `local`-frame AABB poses, quat wxyz), and the real per-session random seed.
- Derive ego-t0 frame states through a single documented transform (`local`→`ego_t0_rig`), recording
  quaternion order (wxyz) and handedness (right-handed ENU/rig). Verify the transform round-trips.
- Build per-candidate `FutureConsequenceTarget` with global + ego-t0 ego states, non-ego actor
  states, actor IDs + valid masks, real timestamps, and provenance hashes. No occupancy fabricated
  (tracks/poses are an admissible structured world-model target).
- Align each planned candidate (ego_t0_rig → ASL frame via the verified t0 ego pose) with the
  executed ego future; report ADE/FDE/max/timestamp-error/aligned-step count with no invented
  pass threshold.

## Result (COMPLETED)
- **Real ASL found for both candidates**; 82 ActorPoses frames each; message types present include
  rollout_metadata, actor_poses, driver_request/return, controller_request/return,
  physics_request/return, render_request, driver_session_request (with random_seed).
- **Future ego states extracted** (global + ego-t0); **non-ego actor states extracted** (5 present at
  t0 of 40 defined; valid masks track enter/leave). Coordinate transform verified to ~1e-6 m
  round-trip; rig_est == true rig (error model off).
- **Planned-vs-executed measured**: cand0 ADE=0.104 m / FDE=0.323 m; cand1 ADE=0.170 m / FDE=0.888 m;
  62/64 steps aligned (last 2 planned steps run past rollout end); timestamp error = 0 µs.
- Targets: `outputs/real_smoke/future_consequence_targets_k2_v2_1.jsonl`;
  alignment: `outputs/real_smoke/planned_vs_executed_alignment_k2_v2_1.jsonl`.
- Native evaluator metrics kept in a **separate** file from future-state targets.

## Decision
`GO_T25_SMALL_SCALE_DATASET_GENERATION` (targets are real, schema-valid, auditable).
**T26 training remains NOT AUTHORIZED in this phase.**

## Files
- code: `src/safeworld/real_smoke/asl_geometry.py`, `.../asl_targets.py`
- outputs: `outputs/real_smoke/future_consequence_targets_k2_v2_1.jsonl`,
  `planned_vs_executed_alignment_k2_v2_1.jsonl`
- run dir: `inspections/asl_message_inventory.json`, `extracted/asl_extraction_cand{0,1}.json`,
  `extracted/asl_timeline_cand{0,1}.json`
- report: `outputs/reports/T25A_single_scene_future_target_preflight.md`

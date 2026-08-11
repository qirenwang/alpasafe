# W0 — L3 world-model exploration (exploratory study, NOT part of the T26G confirmatory line)

Written BEFORE any extraction or training (2026-08-10). Exploratory discipline:
design decisions fixed here ex ante; anything changed later is recorded as an
amendment at the bottom, never edited in place.

## Question

Does putting future-state prediction ON the inference path (WoTE-style explicit
imagination, conditioned on L3) buy selection performance that the amortized
pathway (closed by FN1 as tied-with-geometry) could not?

## Fixed design decisions

- **Keyframes (geometry-determined, NOT data-driven, per user directive):**
  - V2 (primary): t+3.2 s, t+6.4 s after decision — uniform Δ=3.2 s, 2 steps (= WoTE's {H/2, H} structure on our 6.4 s horizon).
  - V3 (arm): t+2.1 s, t+4.2 s, t+6.3 s — uniform Δ=2.1 s, 3 steps (6.4/3 is off-grid; 2.1 s is the nearest uniform on-grid step; 0.1 s shortfall at horizon end accepted).
  - Trajectory grid fact (verified from frozen candidate files): horizon_T=64, output_frequency_hz=10, points at t+0.1..t+6.4 s, frame ego_t0_rig.
- **Future-state vector u per keyframe** (finalized after Step-0 probe; intended):
  ego position/yaw/speed in ego_t0_rig; nearest-N other actors (N=8) relative
  position/yaw/speed with presence mask; per-keyframe `terminated_before_kf` flag
  (rollout ended early — carries last-state + flag, never fabricated).
- **Supervision:** u regressed by generator T (option 1 of the recovered plan);
  official_alpasim_scene_score remains the ONLY selection objective;
  official_score_metrics stay diagnostics-only (unchanged contract inheritance).
- **Score head consumes** [z_t = L3, û at keyframes, planned trajectory] — deletion
  test intentionally FAILS (prediction on the score path); the ZERO-imagination
  ablation verifies it actually fails.
- **Endpoints:** K8 scene-macro regret (primary), tie-aware top-1 (co-primary),
  group→scene→seed aggregation, paired-scene percentile bootstrap CI n=10,000.
  Exploratory: no gates, no sealed-set contact, informal comparisons allowed.
- **Cohort:** the 525-scene frozen cohort (FA2 173 + FN-A 352), read-only.
  Scene-level 5-fold CV × 3 seeds; fold assignment generated fresh with recorded
  seed 20260810 unless FN1's fold file is trivially reusable read-only (record which).

## Step sequence (verify before advancing)

0. Read-only ASL probe (this dir /probe): extractability, tick alignment, u spec finalization, ego cross-check vs FN-A label npz.
1. Extract u for all 12,600 scored rollouts (both V2 and V3 keyframe sets in one pass), sha-verifying every ASL against the frozen manifests. Output: one parquet/npz per cohort in this dir.
2. Oracle gate: score head on TRUE u vs matched amortized control (same folds/seeds). Oracle ≤ amortized ⇒ CLOSE LINE. Else headroom quantified.
3. Imagination variant: T(z, traj segment) recurrent, V2 primary, V3 arm; ablations ZERO-imagination and oracle-swap.
4. (conditional) full-L2 conditioning on FA2 519 groups, directional only.

## Isolation guarantees

- All writes under this directory only; /storage payloads opened read-only.
- Frozen dataset JSONs, FN1/FM1 artifacts, sealed 35 scenes, t26e/ code: untouched.
- No new downloads, no AlpaSim runs, no Alpamayo forwards.
- Every ASL read is sha256-checked against the frozen manifest entry; mismatch ⇒ HALT.

## Amendments

**001 (2026-08-10, from Step-0 probe, before any extraction):** the rollout
actor-pose grid is t+0.0 .. t+6.3 s (64 frames @ 0.1 s); t+6.4 s does not exist
in the ASL stream. V2 keyframes therefore become **{t+3.2, t+6.3}** (midpoint +
last available state; steps 3.2/3.1 s). V3 {2.1, 4.2, 6.3} unchanged — all
exact ticks. Extraction stores the union {2.1, 3.2, 4.2, 6.3}.

**002 (2026-08-10, from Step-1 QA, before any training):** FN-A rollouts have
62 post-decision steps (end t+6.1 s) vs FA2's 64 (end t+6.3 s) — a systematic
acquisition-config difference, NOT termination; zero rollouts end before step
62 in the whole cohort. The 4th keyframe is therefore reinterpreted as
"rollout end state" (t+6.3 FA2 / t+6.1 FNA; 0.2 s cohort offset accepted and
recorded), and `terminated_before_kf` is redefined to fire only when
valid_steps < 62 (currently never). Applied at load time in Step 2+ (flag
column recomputed from future_valid_steps); extraction files unchanged.

**Step-1 QA (2026-08-10):** 12,600/12,600 rows, 0 defects, unique sample_ids,
cohorts FA2 4,152 / FNA 8,448; FNA ego cross-check vs frozen labels: median
1.9e-6, max 7.6e-6 over all 8,448; u NaN/Inf zero; neighbor presence median
8/8 at every keyframe (189 zero-neighbor rows); score zeros 784 (6.2%),
median 0.565 — consistent with frozen freeze stats.

**Step-0 verdict (3 samples: FA2 normal, FA2 hard-fail, FN-A hard-fail):**
sha 3/3, decision/force-GT manifest match 3/3, post-decision grid clean 0.1 s
64 frames in all (hard-fail rollouts still run full length), EGO present in all
frames, neighbors within 50 m: 2–31, ego_t0 chain reproduces frozen FN-A label
npz to max|diff| 3.4e-6 (float32 storage noise). u finalized per keyframe (55-d):
ego [x, y, cosΔyaw, sinΔyaw, speed] (ego_t0 frame) + nearest-8 neighbors within
50 m [rel_x, rel_y, cos relyaw, sin relyaw, speed, present] (ego-at-keyframe
frame) + [terminated_before_kf, n_within_50m/10]. Termination policy: carry
last available frame + flag=1; mid-grid gaps would be counted as defects.

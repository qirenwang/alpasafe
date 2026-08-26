# W0 — L3 world-model exploration: final report (2026-08-10)

**Terminal status: `W0_ORACLE_GATE_FAILED_CLOSE_EXPLICIT_IMAGINATION_LINE`**

Question: does putting future-state prediction ON the inference path (WoTE-style
imagination over L3) buy selection performance the amortized pathway could not?
Answer: **no — and the oracle experiment shows why nothing trained can.**

## Results (525 scenes, 5-fold scene CV × 3 seeds, FN1 folds reused read-only)

| Arm | u channel (true future states) | Objective | K8 regret | K8 top-1 |
|---|---|---|---|---|
| AMORT (control) | zeroed | score | 0.0329 | 74.3% |
| AMORT_C (control) | zeroed | centered | 0.0345 | 73.7% |
| ORACLE_V2 | 110-d true u @ {3.2, end} | score | 0.0362 | 51.3% |
| ORACLE_V3 | 165-d true u @ {2.1, 4.2, end} | score | 0.0385 | 48.8% |
| ORACLE_V2_C | 110-d true u @ {3.2, end} | centered | 0.0408 | 45.5% |
| ORACLE_MIN_C | 4 scalars/kf true u | centered | 0.0332 | 73.8% |

Paired-scene bootstrap (n=10,000, seed 20260810), oracle − control:
- ORACLE_V2−AMORT: regret +0.0034 CI [−0.0009, +0.0074]; top-1 −23.1 pp CI [−25.5, −20.6]
- ORACLE_V3−AMORT: regret +0.0056 CI [+0.0010, +0.0098]; top-1 −25.5 pp CI [−27.9, −23.1]
- ORACLE_V2_C−AMORT_C: regret +0.0063 CI [+0.0019, +0.0104]; top-1 −28.1 pp CI [−30.5, −25.7]
- ORACLE_MIN_C−AMORT_C: regret −0.0012, top-1 +0.1 pp (≈ null)

Controls reproduce the FN1 frontier (L3 arm 0.0321 / 74.9%) — pipeline validity.

## Mechanism (diagnosed, not conjectured)

1. **Alignment verified**: executed state at t+3.2 s matches own plan (p50 0.22 m)
   vs group-mates' plans (p50 0.92 m). No misalignment bug — and candidates are
   geometrically tightly bunched (~0.9 m apart at t+3.2 s).
2. **Within-group outcome variation is rare**: collision/offroad outcomes are
   IDENTICAL across all 8 candidates in ~93% of groups (FA2 7.5%/7.7%,
   FN-A 5.9%/6.2% differ). Within-group score ranking is dominated by progress
   (spread p50 ≈ 0.055) and dist-to-GT (p50 ≈ 0.8 m) — trajectory-geometry terms.
3. **Full-dim true future states act as an overfittable noise channel**: each
   candidate's u comes from its own stochastic rollout; within a group, u
   differences are mostly rollout noise, so the head memorizes train-set
   idiosyncrasies and scrambles within-group order at test time (top-1 −23 pp
   while regret only +0.003: picks stay in the good cluster but stop hitting
   the exact best). Centering the objective does NOT rescue it (rules out the
   level-fitting explanation).
4. **Distilled to outcome essentials (4 scalars/kf), true future adds ≈ nothing**
   — consistent with (2): the groups where outcomes differ (~6–8%) are too few,
   and the progress component is already inferable from (plan, L3).
5. **Conceptual closure**: at inference, any trained imagination T(z, traj) is a
   deterministic function of the same inputs the amortized head already sees —
   it adds inductive bias, never information. The only information-adding
   version is the oracle (true u), and it failed. Hence no trained Step-3
   variant can beat this gate; Steps 3/4 are void.

## Why this coheres with FN1

FN1 found L3 ≈ tuned geometry MLP on both endpoints. W0's decomposition explains
it: within-group ranking signal on this cohort is mostly trajectory geometry
(+progress); scene/future information discriminates in only a small minority of
groups. The K=8 same-policy sampling regime produces counterfactual spread too
small for future-state evaluation to matter — unlike WoTE's 256 heterogeneous
anchors, where futures genuinely diverge.

## What would change the answer (recorded for any future line)

- A candidate generator with real behavioral diversity (lane change vs stop vs
  go), i.e., large within-group future divergence.
- Endpoints weighted toward outcome-critical groups (the 6–8%) rather than
  scene-macro averages.
- Score components with within-group variation beyond geometry/progress.

## Integrity

12,600/12,600 ASLs sha-verified, 0 defects; FN-A ego cross-check vs frozen
labels max 7.6e-6 over all 8,448; all writes confined to this study dir;
frozen datasets/payloads/code untouched; sealed 35 scenes untouched;
design amendments 001–004 recorded before the affected computations ran.

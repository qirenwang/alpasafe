# SafeWorld T26G-FN0 — FINAL Fable 5 Read-Only Preregistration Prompt
## L3 Pathway Consolidation & Promotion: verify the FM1 authority as finalized,
## resolve the promotion power gap from the locked FM1 predictions, freeze a
## two-question four-arm design (adequately-powered legacy superiority on
## regret + a controlled selector-loss test for the top-1 gap), plan the
## extension acquisition, and generate the FN-A acquisition and FN1 execution
## prompts — NO TRAINING, NO ACQUISITION, NO NEW ENDPOINTS

## Execution authorization

T26G-FM1 completed with the terminal status:

```text
T26G_FM1_NO_FULL_L2_GAIN_CLOSE_LIGHTWEIGHT_READER_LINE
```

The full-L2 reader line is CLOSED and stays closed. FN0 does NOT reopen it.
FN0's subject is what FM1 ALSO established, prospectively and with CI
support on both primary endpoints, on 173 untouched scenes:

```text
M3  LASTTOKEN − NULL   regret −0.0069 CI [−0.0124, −0.0012]
                       top-1  +0.0996 CI [+0.0668, +0.1330]
M5  LASTTOKEN − LEGACY regret −0.0051 CI [−0.0114, +0.0005]   (3/3 seeds
                       negative, 4/5 folds negative; best absolute regret
                       of all four arms: 0.0371 vs legacy 0.0422)
                       top-1  −0.0334 CI [−0.0713, +0.0039]
                       (K2/K5 top-1 CI-supported worse)
```

The user authorizes **T26G-FN0 only**. FN0 may:

1. audit the finalized FM1/FM0/FA2 authority chain read-only;
2. recompute, from the LOCKED FM1 predictions, the per-scene paired deltas
   and their SDs, and resolve the extension sample size by the frozen
   FA0 power method;
3. freeze the FN design: arms, objective variants, checkpoint policy,
   folds recipe (INCLUDING the inner-validation rule this time), endpoints,
   statistics and promotion gates, BEFORE any new outcome exists;
4. freeze the extension-cohort acquisition plan (eligibility, selection
   order, N, pipeline identity = the frozen FA2 package plus its two
   recorded repairs);
5. generate, hash and register TWO gated execution prompts: FN-A
   (extension acquisition) and FN1 (training + evaluation, gated on FN-A);
6. run synthetic-only conformance for the one new training variant
   (the selector-loss arm) and re-register budgets.

FN0 does NOT authorize:

```text
training any arm; computing any new endpoint on any cohort
scene download, Alpamayo generation, AlpaSim/NuRec/evaluator execution
opening the 35-scene confirmation cohort (IDs stay ID-only)
re-running, re-tuning or re-interpreting ANY closed FM1 gate
modifying any finalized record (FM1, FM0, FA2, or earlier)
starting FN-A or FN1
```

Stop at the first genuine blocker and finalize FN0 in the same response.
Repository: /home/qiren/alpasafe/safeworld-alpamayo
Generated-code discipline: every emitted Python module passes
`ruff check --select F821` before registration (standing gate).

---

## 0. Position, and on-disk facts verified before this prompt was issued

The executor re-verifies everything below; expected state, not surprises.

**0.1 Authoritative FM1 record**

```text
FM1_REC = artifacts/safeworld_t26g_fm1_full_l2_matched_selector_experiment/
          20260803T132326Z
status/T26G_FM1_FINALIZED   (exact terminal status above; gates 3/10:
                             G5 causal use, G9 budget, G10 integrity PASS)
manifests/t26g_fm1_final_manifest.json + reread QA (0 mismatches)
manifests/t26g_fm1_prediction_lock_manifest.json  (hash-registered
    predictions for 4 arms x 3 seeds; the ONLY admissible source for the
    Phase-1 power recomputation)
manifests/t26g_fm1_derived_folds.json  (the inherited inner-validation
    rule, source-hashed to the T26G-A expanded-split manifest)
results/t26g_fm1_analysis.json / t26g_fm1_promotion_gate.json
```

**0.2 Upstream chain (unchanged)**: FM0 20260803T060900Z FINALIZED
(design contracts, 519 label arrays, wrong-scene bijection, derived FA2
payload pointer); FA2 20260730T190521Z FINALIZED (dataset sha
`e84ddde8c24991a40935cfc464f7f60755f99ef44a8934db3d95d00715056e96`,
173/519/4,152); pins Alpamayo `bf580713…`, AlpaSim `a1f05bb6…`,
evaluator `196d21ab…`, catalog csv `b1cc2d2f…` (916 rows, 790 eligible).

**0.3 The instability history is CLOSED, not open.** Three loss/training
defects were found and repaired upstream and must not be re-litigated:
the rank/centered gradient conflict (fixed by the C0 decomposition + D0
bounded residual), the Spearman-first checkpoint mis-selection (fixed by
the E1 CONSTRAINED_SELECTOR_FIRST archive), and the selector-loss
gradient pathology at confident errors (documented in E0). FM1's M3 shows
the surviving stack is stable prospectively. FN tests exactly two NEW
questions and nothing else:

```text
Q1  POWER: is the LASTTOKEN regret advantage over the retrained legacy
    reference real at adequate power?  (M5 point −0.0051 nearly excludes
    zero at n=173; the cohort, not the model, is the current limit)
Q2  TOP-1: does the frozen E1 selector loss close the CI-supported
    K2/K5 top-1 deficit vs legacy WITHOUT giving back the regret
    advantage?  (E6 precedent: the selector loss produced the E-line's
    only joint gates-1+2 pass, +0.0829 CI-positive top-1; its known
    harms are recorded in E4 and must be gated against)
```

**0.4 Seen-once honesty clause.** The 173 FA2 scenes' outcomes were
analyzed exactly once by FM1. They remain development data. Every FN
promotion gate that quotes the enlarged cohort must ALSO hold
directionally in the untouched-extension subset (per-cohort consistency,
the E1 gate-8 pattern). The 35-scene confirmation cohort remains sealed
and ID-only until an FN1 winner and every threshold are frozen.

**0.5 The 35-scene reserve contract carries over unchanged**:
`T26G_FM_FROZEN_UNTOUCHED_CONFIRMATION_COHORT_V1` (FM0 contract). FM1
produced no winner, so it was never opened. Its sole purpose is now the
confirmation of the FN1 winner (FN2), never FN training or gating.

---

## 1. FN0 record root

```text
FN0_REC = artifacts/safeworld_t26g_fn0_l3_pathway_consolidation_preregistration/<UTC>
```

Subdirs: audit/ code_artifacts/ contracts/ folds/ logs/ manifests/
prompts/ qa/ reports/ results/ splits/ status/. System disk, MB-scale.
No upstream record is written.

---

## 2. Phase 0 — authority integrity (read-only)

Verify: FM1 FINALIZED status string exact and gates record matches 0.1;
FM1 final manifest rereads (spot re-hash 20 registered outputs including
all 12 prediction npz); FM1 prediction-lock manifest rereads and every
prediction file re-hashes; FM0/FA2 statuses and dataset sha unchanged;
capture implementation sha `6d43679e…` unchanged; the FM0 35-scene
contract unchanged; no FN record already exists; full pin inventory
re-recorded. Blocker: BLOCKED_T26G_FN0_AUTHORITY_OR_INTEGRITY_FAILURE.

---

## 3. Phase 1 — power resolution from the locked predictions (read-only)

Recompute per-scene K8 paired deltas for M5 (LASTTOKEN − LEGACY, regret
and top-1) and M3, from the LOCKED FM1 prediction npz files, using the
frozen shared helper t26g_e_topoflist and the frozen aggregation
(group -> scene mean -> seed mean -> paired scene). Verify the recomputed
means/CIs equal the locked FM1 analysis values bit-for-bit in mean (this
is a reproduction check, not a new endpoint).

Then resolve the extension size by the FROZEN FA0 method (two-sided
alpha 0.05, power 0.80, z = 1.96 + 0.8416):

```text
N_total = ceil( ((1.96 + 0.8416) * SD_paired_regret_M5 / MDE_FN)^2 )
MDE_FN  = the FM1-observed M5 regret point effect, recomputed here
          (expectation ~0.0051; the recomputed value is authoritative)
N_new   = N_total − 173, +20% attrition allowance (FA0 convention),
          capped by the eligible pool
```

Expectation (to be recomputed, not assumed): SD_paired ≈ 0.040 →
N_total ≈ 480, N_new ≈ 307 before attrition, ~369 with allowance.
Eligible pool: 790 catalog-eligible − 173 primary − 35 reserve − 4 smoke
= 578; the frozen FA scene_key order (sha256("T26G-FA|" + catalog_revision
+ "|" + scene_id)) continues where FA2's selection stopped — no new
selection rule. If N_new (with allowance) exceeds 578, take 578 and
record the achieved power. Also record the top-1 MDE achieved at N_total
for the Q2 comparison. Write
contracts/t26g_fn0_power_and_sample_size_contract.json (+ QA).
Storage estimate: FA2 scaled by N_new/173 (payload ≈ 1.2 TB × ratio;
verify /storage free ≥ 1.5× estimate or block).

---

## 4. Phase 2 — arms and objective variants (frozen before any new outcome)

Four arms, 5 folds, 3 seeds = 60 runs on the ENLARGED cohort:

```text
P_L3_CKPT_CONS      LASTTOKEN architecture, checkpoint-only T26G-D
                    objective  (byte-for-byte the FM1 LASTTOKEN arm)
P_L3_SEL_CONS       SAME architecture; the exact frozen E1 selector
                    objective replacing L_rank_margin (tau, coefficient
                    and code re-hashed from the E1 record; NO re-tuning)
P_NULL_CKPT_CONS    matched no-scene control (FM1 NULL arm, unchanged)
P_LEGACY            t26f_b1_models.SafeWorldV2('A'), historical T26G-B
                    recipe, retrained on the enlarged cohort
```

Architecture: the frozen F0B SafeWorldF1 graph exactly as instantiated by
the FM1 package (1,708,871 params/arm, equality asserted); visibility
LASTTOKEN / NULL only — the FULL mode exists in code but is
CONTRACTUALLY UNREACHABLE in FN (the reader line is closed). The FM1
package modules are the base; adaptation is dataset plumbing + the
selector-loss wiring for P_L3_SEL_CONS only, taken from the registered
t26g_e_losses.py / t26g_e_models.py loss path (re-hashed; the E0
selector-loss contract sha `4a9e0c88…` is the design anchor).

**Nomination rule (frozen now, deterministic):** P_L3_SEL_CONS is the
nomination arm IF it passes every gate; otherwise P_L3_CKPT_CONS is the
nomination arm IF it passes every gate; otherwise no nomination. No other
arm is nominable. Both candidate arms' gate evaluations are computed
exactly once from the same locked predictions.

Checkpoint policy: the exact E1 CONSTRAINED_SELECTOR_FIRST archive for
all three trio arms (counterexample asserted); legacy keeps its
historical rule. Input boundary, cast policy, tie semantics
(eps 1e-6 + exact-equality pairwise), refit rule: inherited unchanged.

---

## 5. Phase 3 — folds (inner-validation rule frozen EXPLICITLY this time)

```text
namespace "T26G-FN": fold_key = sha256("T26G-FN|" + scene_id)
sort ALL enlarged-cohort scenes by fold_key ascending; fold = rank mod 5
inner-validation = the lowest-ranked max(12, floor(0.15 * n_train))
scenes of each outer-training set under the SAME key (the inherited
T26G-A rule, now first-class in the FN0 contract, closing the FM0 gap)
seeds {0,1,2}; 60 runs; smoke + 35 reserve scenes structurally absent
```

Write splits (computable for the 173 now; the extension scenes are
appended deterministically by FN1 after FN-A, using this frozen formula —
no discretion), plus the run-population and leakage contracts.

---

## 6. Phase 4 — endpoints, statistics, gates

Endpoints/aggregation/bootstrap: unchanged from FM0/FM1 (K8 primary
regret, co-primary tie-aware top-1; K2/K5 prefixes by candidate_index;
10,000 paired scene bootstraps; NEW frozen seed = the FN0 record date as
integer). Prediction lock before any endpoint; analyzer sha registered.

Comparison graph:

```text
N1  nominated − P_LEGACY            (promotion comparisons)
N2  nominated − P_NULL_CKPT_CONS    (scene value re-verification)
N3  P_L3_SEL_CONS − P_L3_CKPT_CONS  (the Q2 controlled contrast)
N4  P_L3_CKPT_CONS − P_LEGACY       (reported for both candidates)
```

Ten gates, margins REREAD from the frozen E0/F0B/FA2/FM0 contracts
(0.005 / 0.010 / 0.013 / 0.046 — hash-register the sources; numerals
here are expectations):

```text
G1  N1 regret SUPERIORITY: delta < 0 AND 95% CI high < 0
G2  N1 top-1 NON-INFERIORITY: point delta >= −0.013(*) AND CI low >
    −0.046; (*) the point clause uses the frozen regret-materiality-
    scaled convention resolved in FN0 from the frozen constants, BEFORE
    FN1 — record the written resolution
G3  N1 pairwise non-inferiority: CI low > −0.010
G4  N2 scene value: regret CI high < 0 AND top-1 CI low > 0 (the M3
    replication must reproduce on the enlarged cohort)
G5  IF the nominated arm is P_L3_SEL_CONS: N3 top-1 delta > 0 with CI
    low > 0 AND N3 regret CI high < +0.005 (the selector loss must BUY
    top-1 and NOT SELL the regret advantage); vacuous for P_L3_CKPT_CONS
G6  causal scene use: WRONG_SCENE and ZERO ablations on the nominated
    arm (FN wrong-scene bijection frozen in FN0, "T26G-FN-WRONG|" ring)
G7  K-consistency at the frozen materialities
G8  robustness: seeds, folds, LOSO, half-effect rule, AND the 0.4
    per-cohort consistency clause (direction holds in the untouched-
    extension subset on both primaries)
G9  budget: unchanged ceilings; GPU preflight gate (CUDA has been
    flapping on this host — the runner refuses to train while wedged)
G10 integrity: 60/60, isolation, lock-before-endpoints, all joins/
    hashes verified, reserve/sealed untouched, manifests reread
```

On PASS: FN1 generates (never executes) the FN2 confirmation
preregistration — the frozen winner evaluated ONCE on the 35-scene
cohort after its own FA2-style acquisition. On FAIL of both candidates:

```text
T26G_FN1_NO_STABLE_L3_PROMOTION_CLOSE_CONS_SELECTOR_LINE
```

and the CONS selector line closes (the same finality the reader line
received). Exactly two FN1 terminal outcomes:

```text
T26G_FN1_L3_PROMOTABLE_READY_FOR_FN2_CONFIRMATION
T26G_FN1_NO_STABLE_L3_PROMOTION_CLOSE_CONS_SELECTOR_LINE
```

---

## 7. Phase 5 — extension acquisition plan (FN-A, generated not executed)

FN-A reuses the FA2 pipeline byte-for-byte: the FA1A-registered modules
plus the two recorded mid-run repairs (post-repair hashes `7ed2da24…`,
`2a460589…`, `4d02a4e7…`), the same pins, the same capture patch
(re-hash `6d43679e…`), the same concurrency contract, the same
record/payload separation, the same FA2-style resumable runner
(run/watch/stop/resume, lock file, heartbeat, per-item completion).
Differences: N_new scenes from the frozen continuation of the FA scene
order; full-L2 capture retained ONLY as the L3 row provenance (store the
[1,4096] last row + its parent-hash proof; the full sequence is NOT
retained — the reader line is closed and 2 TB of dead payload is not);
labels materialized inline by the registered B3-A extractor semantics at
acquisition time (the FM0 Phase-7 lesson: persist closure from the
start). FN-A terminal statuses:

```text
T26G_FNA_EXTENSION_COHORT_ACQUIRED_READY_FOR_FN1
BLOCKED_T26G_FNA_* (acquisition-specific blockers, FA2 conventions)
```

## 8. Phase 6 — synthetic conformance + budget

Synthetic-only: the selector-loss arm's wiring (loss value finite,
gradient nonzero where the E0 pathology predicts zero — assert the
KNOWN failure mode is detected and logged, not silently accepted),
parameter equality unchanged, determinism fixture re-hashed, budget
ceilings re-measured on GPU (refuse if wedged, record the flap).

## 9. Phase 7 — FN prompts generation and FN0 finalization

Generate + hash prompts/SafeWorld_T26GFNA_…_Acquisition_Prompt_<DATE>.md
and prompts/SafeWorld_T26GFN1_…_Execution_Prompt_<DATE>.md (FN1 hard-
gated on FN-A's terminal status and on the in-session preflights, FM1
§9-style execution vehicle included). Finalize with final manifest,
reread QA, reports, and exactly one status:

```text
T26G_FN0_L3_PATHWAY_CONSOLIDATION_PREREGISTERED_READY_FOR_FNA
```

or the precise BLOCKED_T26G_FN0_* blocker. Required outputs mirror the
FM0 §12 inventory (authority audit, power contract, arm/objective/
checkpoint/tie/fold/endpoint/gate/wrong-scene/acquisition contracts,
conformance + budget results, all QA, both prompts, final manifest,
status).

## 10. Final terminal response

State plainly: (1) FN0 status; (2) the recomputed M5/M3 per-scene SDs,
the resolved MDE, N_total, N_new and achieved power; (3) the four arms
and the frozen nomination rule; (4) the selector-loss identity (tau,
coefficient, code hashes); (5) folds incl. the now-explicit inner-val
rule; (6) all ten gates with reread margins; (7) the FN-A plan (N, order
continuation, storage estimate, L3-only retention decision); (8) both
generated prompt paths + sha256; (9) confirmation that no training,
acquisition, endpoint, reserve or sealed access occurred; (10) the sole
next action — the user launches FN-A. Do not train. Do not acquire. Do
not touch the 35 scenes.

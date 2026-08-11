# SafeWorld T26G-FM0 — FINAL Fable 5 Read-Only Preregistration Prompt
## Full-L2 Matched Selector Study: verify the FA2 authority as it actually
## exists on disk, refreeze the confirmation cohort, freeze arms / objective /
## splits / endpoints / gates, materialize the one missing auxiliary label,
## run synthetic conformance and budget checks, and generate the T26G-FM1
## execution prompt — NO TRAINING, NO PREDICTIONS, NO ENDPOINTS

## Execution authorization

T26G-FA2 completed with the terminal status:

```text
T26G_FA2_PROSPECTIVE_FULL_L2_COHORT_ACQUIRED_READY_FOR_SELECTOR_PREREGISTRATION
```

The user authorizes **T26G-FM0 only**. FM0 may:

1. audit the finalized FA2 record and payload read-only, against the
   authority set FA2 actually publishes (see §0 — FA2 has no final manifest);
2. refreeze the 35 unused reserve scenes as a protected confirmation cohort;
3. recover and freeze the matched reader architecture from the F0B contingent
   package and the legacy selector graph from frozen T26G-B/E1 references;
4. freeze the training objective, checkpoint policy, tie semantics, folds,
   seeds, endpoints, statistics and all promotion gates BEFORE any label
   statistic beyond the already-seen aggregate score distribution is read;
5. materialize exactly one missing auxiliary label (future_ego_states) from
   the frozen FA2 scoring ASLs, read-only, AFTER all design freezes;
6. run synthetic-only architecture conformance and budget measurement;
7. generate, hash and register the T26G-FM1 execution prompt.

FM0 does NOT authorize:

```text
training any arm on real official_alpasim_scene_score targets
cross-validation, refits, real model predictions
computing any endpoint, regret, top-1 or arm comparison
opening or using the 35 reserve scenes (beyond ID-only protection contract)
new scene download, Alpamayo generation, AlpaSim/NuRec rollout, evaluator run
modifying the frozen FA2 cohort or any finalized record
sealed-label or protected-cohort access
changing architecture, loss, folds, sample size or gates in response to the
  already-seen FA2 score distribution
starting T26G-FM1
```

Stop at the first genuine blocker and finalize FM0 in the same response.
Repository: /home/qiren/alpasafe/safeworld-alpamayo

---

## 0. Position, and on-disk facts verified before this prompt was issued

These facts were checked directly on the machine on 2026-08-03. The executor
must re-verify them, but must treat them as the expected state — not as
surprises, and in the case of 0.2 not as a blocker.

**0.1 Authoritative FA2 roots**

```text
FA2_REC = /home/qiren/alpasafe/safeworld-alpamayo/artifacts/
          safeworld_t26g_fa2_prospective_full_l2_acquisition/20260730T190521Z
FA2_PAY = /storage/alpasafe/safeworld-alpamayo/payloads/
          safeworld_t26g_fa2_prospective_full_l2_acquisition/20260730T190521Z
```

**0.2 FA2 publishes NO final manifest and NO payload pointer.**
`FA2_REC/manifests/` is empty. FA2's authority set is:

```text
status/T26G_FA2_FINALIZED                     (exact status string above)
results/t26g_fa2_prospective_dataset.json
    sha256 e84ddde8c24991a40935cfc464f7f60755f99ef44a8934db3d95d00715056e96
    (internally registers every full-L2 / dump-ASL / candidate / score hash;
     10/10 checks true, 0 defects)
results/t26g_fa2_full_official_scores.json
results/t26g_fa2_full_state.json
logs/t26g_fa2_full_pipeline_status.txt        (PIPELINE_COMPLETE, rc=0)
```

Code identity is carried by the FA1A stage, not by FA2:

```text
artifacts/safeworld_t26g_fa1a_fa2_launch_amendment/20260730T190521Z/
    manifests/t26g_fa1a_fa2_launch_pointer.json
    manifests/t26g_fa1a_final_manifest.json
       sha256 7b69708df0587c97473430845e1b2ee54a06df4eab7fdcb40dfadd685e2ff17d
FA2_REC/audit/t26g_fa2_mid_run_repair_001.json
FA2_REC/audit/t26g_fa2_mid_run_repair_002.json
```

The two mid-run repairs are the ONLY accepted code deltas versus the
FA1A-registered package (001: estimate-schema fix; 002: sha_file import +
capture MAX_INFRA_RETRIES 2->6). Both are recorded as zero-scientific-impact.

**0.3 Auxiliary-label state is already known.**
`progress_clipped_rel`, `collision_at_fault`, `offroad` (plus
`gt_dist_traveled_m`, `dist_to_gt_trajectory`, `progress_rel`,
`progress_score`) are materialized per candidate inside the dataset's
`official_score_metrics`, verified 4,152/4,152. `future_ego_states_ego_t0`
is NOT materialized anywhere in the dataset. FM0 materializes it itself
(§9); it does not halt to request a separate stage for a known fact.

**0.4 Cohort identity**

```text
173 scenes / 519 groups / 4,152 candidates / K=8
primary_ids_sha256  c6c94732c5f93ed90fee7deea1c13ff25671fa7aff16a8b1c53dc26849c13fa9
reserve_ids_sha256  478dfce3f322121e79ba2df31689cc6edaa9c2007bff75f3ef928c5a981a8945
smoke exclusion     4 scenes, ENGINEERING_SMOKE_ONLY (FA2_REC/splits/...)
L2                  [3086,4096] bf16, 519/519 sha unique, L3 == L2[-1] bit-exact
pins                Alpamayo bf580713f08656674827cd6e09888c79cf65fbf2
                    AlpaSim  a1f05bb628f3d1d19d79d44188e836e9108f98c6
                    evaluator 196d21ab86593af121b055995d0185bb786d1f70
```

**0.5 Scientific position.** E1 established scene-specific L3 selector value
against matched controls but failed the absolute-reference gates. The
remaining hypothesis: the single final prompt token over-compresses scene
information; a candidate-conditioned reader over the full prompt-only L2 may
retrieve trajectory-relevant evidence L3 alone does not preserve. FA2 proves
data closure only. FM1 tests the hypothesis. FM0 freezes FM1.

**0.6 The already-seen score distribution** (median 0.548, group spread
median 0.066, 168/519 > 0.1) proves oracle headroom exists. It proves
nothing about FULL's ability to exploit it, and from this point forward it
may not influence any design constant.

---

## 1. FM0 record root

Create (system disk, MB-scale policy; no /storage payload tree — the only
bulk work is a streaming read of existing FA2 ASLs):

```text
FM0_REC = /home/qiren/alpasafe/safeworld-alpamayo/artifacts/
          safeworld_t26g_fm0_full_l2_matched_selector_preregistration/<UTC_TIMESTAMP>
```

Subdirs: audit/ arrays/ code_artifacts/ contracts/ folds/ logs/ manifests/
prompts/ qa/ reports/ results/ splits/ status/
Synthetic conformance tensors go under results/synthetic/, small, fixed-seed.
No whole-run symlink. No upstream record is modified. The E1 lesson stands:
never re-run a Phase-0 audit post-finalization without a phase flag.

Generated-code discipline for FM0 and for everything FM0 emits into the FM1
package: every Python module must pass an undefined-name scan
(`ruff check --select F821`) before being registered. Both FA2 mid-run
crashes were F821-catchable; this is now a standing gate.

---

## 2. Phase 0 — FA2 and authority integrity (read-only)

Verify, using the §0.2 authority set:

1. FINALIZED status string exact; pipeline log ends PIPELINE_COMPLETE rc=0;
2. the frozen dataset rereads at its recorded sha256; internal checks
   all_pass, 0 defects; counts 173/519/4152; K=8; 3 groups per scene;
3. spot re-verification: for the 12 lexicographically smallest group_ids,
   re-hash the full-L2 file, dump ASL and all 8 candidate jsons against the
   dataset records; for 24 deterministically chosen candidates re-hash the
   scoring ASL and metrics parquet;
4. all 519 L2 sha unique; shape [3086,4096] uniform; L3 == full_L2[-1]
   bit-exact re-verified by loading the 12 spot-check tensors;
5. same-prefill association: the capture-time proof lives in each group's
   candidate_generation dump (`fa_full_l2.captured_in_same_predict_call...`,
   `prefill_calls == 1`); re-verify on the 12 spot-check groups;
6. cohort hygiene: zero overlap of the 173 with the 4 smoke scenes, the 122
   development scenes, the 2 permanent sealed scenes, and the T26G-A 100+20
   protected cohort (scene_id, uuid AND clip_id);
7. code identity: FA2 module hashes match the FA1A launch pointer except for
   the two recorded repairs; the capture implementation is the UNCOMMITTED
   working-tree patch to
   external/alpasim/src/driver/src/alpasim_driver/models/alpamayo_base.py —
   hash the live file and compare against the baseline registered in
   FA1_REC/audit/t26g_fa_capture_boundary_audit.json
   (FA1_REC = artifacts/safeworld_t26g_fa_prospective_full_l2_cohort/
   20260730T020706Z, final manifest sha 367f243e926a797c6110796268
   7003ae1c6a969245315289c8294c8449b37ddd);
8. upstream statuses unchanged: E0/E1 (E1 run 20260729T014122Z), F0, F0B
   (20260729T173643Z), FP0, FR0, FS0, FA1, FA1A;
9. no selector training has started; no reserve-scene access has occurred;
10. record a FULL hash inventory (no prefixes): Alpamayo weight-file shas
    from the HF cache, AlpaSim revision, evaluator revision, 26.01 catalog
    csv sha, FA2 dataset sha, capture implementation sha.

Then DERIVE, read-only, FM0's own FA2 payload pointer:
walk FA2_PAY, record path/bytes for every file and sha256 for every
scientific file re-verified against the dataset registry, and write it to
FM0_REC/manifests/t26g_fm0_derived_fa2_payload_pointer.json. FA2 itself is
never written.

Write: audit/t26g_fm0_authority_integrity_audit.json,
reports/t26g_fm0_authority_integrity.md, qa/t26g_fm0_authority_integrity_qa.json.
Blocker: BLOCKED_T26G_FM0_AUTHORITY_OR_INTEGRITY_FAILURE.

---

## 3. Phase 1 — refreeze the untouched confirmation cohort

The 35 reserve scenes were never used by FA2. Freeze them as:

```text
T26G_FM_FROZEN_UNTOUCHED_CONFIRMATION_COHORT_V1   (ID-only, N=35)
```

Until one winning FM1 model and every threshold are fully frozen: no asset
download, no content opening, no Alpamayo, no AlpaSim/NuRec/evaluator, no
candidates or scores, no use in folds or normalization, no qualitative
inspection. They serve exactly one future purpose: a separately authorized
FM2 prospective confirmation acquisition + evaluation of the frozen winner.
Their FA2 "replacement pool" role is retired — FA2 is complete, replacement
can never occur again.

Write contracts/t26g_fm0_untouched_confirmation_cohort_contract.json (+ QA).
Any evidence a reserve scene was already used for model selection:
BLOCKED_T26G_FM0_RESERVE_COHORT_CONTAMINATION.

---

## 4. Phase 2 — arms, architecture and input boundary

**4.1 Four arms, one nomination arm.**

```text
P_FULL_L2_CONS               (sole nomination arm)
P_LASTTOKEN_L3_MATCHED_CONS
P_NULL_SCENE_MATCHED_CONS
P_A_LEGACY_MATCHED           (absolute reference, retrained on this cohort)
```

**4.2 Matched-trio construction rule — no architecture search.**
All three matched arms instantiate the SAME module graph, recovered from the
F0B contingent package (run 20260729T173643Z; 1,708,871 trainable params per
arm; 12/12 conformance QA), adapted ONLY in dataset plumbing (FA2 paths,
L=3086). FM0 must either (A) prove one exact F0B configuration satisfies all
prospective-data contracts and freeze it byte-for-byte, or (B) identify a
genuine incompatibility and block
(BLOCKED_T26G_FM0_ARCHITECTURE_OR_MATCHED_CONTROL_FAILURE) without proposing
a substitute. No hidden-size / depth / pooling / token-pruning search.

The single difference between the matched arms is the key/value sequence the
frozen candidate query cross-attends to:

```text
FULL       H_g = full_L2           [3086, 4096]
LASTTOKEN  H_g = full_L2[-1:]      [1, 4096]     (same FA2 capture, not any
                                                  historical L3 shard)
NULL       H_g = zeros, fixed, non-trainable [1, 4096]
```

Parameter counts are therefore exactly equal by construction; assert
equality. Candidate side (from F0B, subject to byte-audit): trajectory
encoder over planned_trajectory [64,2] -> candidate query -> one lightweight
cross-attention block over H_g -> candidate-conditioned consequence latent
-> heads: future ego-state [<=64,2], progress, collision, offroad, and the
official-score head with the frozen C0 group-mean / candidate-residual
decomposition and the frozen D0 bounded residual S*tanh(raw/S), S = 0.67.
Record the F0-B trap: the score-residual head is ZERO-INITIALIZED, so any
init-time score probe is vacuous and must never be cited as evidence.

**4.3 Candidate independence (conformance-tested per matched arm):**
score(scene, candidate_i) invariant to candidate permutation and to the
presence/absence of unrelated candidates; no cross-candidate attention or
set encoder; one shared scene K/V cache may be reused across the K
candidates.

**4.4 P_A_LEGACY_MATCHED.** Reconstruct the EXACT legacy selector graph from
the frozen references the E-line compared against (T26G-B run
20260724T022521Z frozen reference models + registered model code; verify
identity against E1's A_legacy reference by hash). Retrain on this cohort
under the same folds, seeds and leakage rules, with its own historical
objective and checkpoint rule — do not retrofit the new objective into it.
If the graph, objective, preprocessing or checkpoint rule cannot be
recovered without invention:
BLOCKED_T26G_FM0_LEGACY_REFERENCE_NOT_RECONSTRUCTABLE. Do not substitute a
newly designed baseline.

**4.5 Input boundary (frozen).** Permitted model inputs:

```text
planned_trajectory [64,2] float32
scene representation per arm (full L2 / last token / deterministic null)
```

Forbidden inputs: raw observation or camera tensors, map, scene ID,
timestamp, candidate index/rank, reasoning tokens or text, generation RNG
state, rollout ASL content, native metrics, all auxiliary targets, official
score, any diagnostic or split identifier. Observation/prompt closure is
provenance only. bf16 L2 is stored as-is; the loader casts to fp32 at batch
assembly (frozen cast policy).

Write: contracts/t26g_fm0_architecture_contract.json,
contracts/t26g_fm0_arm_contract.json,
qa/t26g_fm0_architecture_conformance_qa.json,
results/t26g_fm0_parameter_count.json.

---

## 5. Phase 3 — objective, checkpoint and tie semantics

**5.1 Objective — checkpoint-only consequence objective (decision recorded).**
FULL / LASTTOKEN / NULL train with the exact T26G-D consequence objective:

```text
L_total = 1.00 L_rank_margin + 0.50 L_centered + 0.10 L_group_mean
        + 0.25 L_future + 0.10 L_progress + 0.10 L_collision
        + 0.10 L_offroad
```

Every coefficient, SmoothL1 beta (C0 scale-matching), margin, normalization
and pooling detail is recovered from the registered D/E code and re-hashed;
the prose above is not authoritative.

Decision record, frozen now and not revisitable after results: the E1
selector loss is deliberately NOT used. Basis: (a) the D-line R2 arms passed
gate 1 with CI support under this objective on both backbones — the failure
was interval width at 122 scenes, and this cohort holds 173 scenes powered
for the frozen MDEs; (b) E1's E4 edge showed the selector loss materially
worsens no-scene regret (CI fully positive), and E3 showed no absolute
improvement of the L3 arm; (c) FM1 isolates exactly one variable —
representation visibility — relative to a consequence-objective baseline.
The E6 finding (selector loss produced the first joint gates-1+2 pass for
L3) is acknowledged; if FM1 fails, any selector-loss × full-L2 combination
is a NEW hypothesis requiring its own preregistered stage, not a post-hoc
switch.

**5.2 Checkpoint policy.** FULL/LASTTOKEN/NULL use the exact E1 archive
implementation of CONSTRAINED_SELECTOR_FIRST, including the eligible-epoch
archive, rising-band pruning, incumbent-vacation behaviour and tie-break
key. The legacy arm keeps its historical checkpoint rule.

**5.3 Tie semantics.** top-of-list tie epsilon = 1e-6; pairwise endpoint
uses E1's exact-equality semantics; one shared implementation for inner
validation and final analysis.

Write: contracts/t26g_fm0_objective_checkpoint_contract.json,
contracts/t26g_fm0_tie_semantics_contract.json, + both QA files.

---

## 6. Phase 4 — folds, seeds, run population

Five deterministic scene-level outer folds over the 173 primary scenes,
fully specified (no examples, no discretion):

```text
fold_key = sha256("T26G-FM|" + scene_id)
sort the 173 scenes by fold_key ascending
fold_id  = sorted_index mod 5            ->  sizes 35/35/35/34/34
all 3 groups of a scene inherit its fold
seeds    = {0, 1, 2}
runs     = 4 arms x 5 folds x 3 seeds = 60
```

No scene appears in both inner-training and outer-test of a fold. Every
normalization statistic, calibration and checkpoint-selection quantity is
computed from outer-training scenes only. The 4 smoke scenes and 35
confirmation scenes appear in no fold (structural: folds are built from the
primary list). Freeze a full-development refit rule for later use; do not
execute it. Write splits/t26g_fm0_scene_folds.json,
folds/t26g_fm0_fold_composition.json,
contracts/t26g_fm0_run_population_contract.json,
qa/t26g_fm0_split_and_leakage_qa.json.

---

## 7. Phase 5 — prediction lock, endpoints, statistics

**Prediction lock (executed in FM1, frozen here).** All outer-fold
predictions — IDs, arm/fold/seed, predicted score, residuals and bounded
transforms, auxiliary predictions, NORMAL plus the FULL ablation conditions
— are written and hash-registered before any endpoint is computed;
STAGE_PREDICTIONS_LOCKED only after every hash verifies.

**Endpoints.** Primary: K8 scene-macro selected official_alpasim_scene_score
regret (lower better). Co-primary: K8 tie-aware top-1 agreement (higher
better). Safety/mechanism: K8 pairwise accuracy, best-vs-second-best
inversion, best-vs-rest pairwise, selected true rank, selected oracle gap,
score-margin diagnostics, future ADE/FDE, auxiliary metrics. K=2 and K=5
reported from deterministic candidate prefixes frozen before results; a
candidate's predicted score is never recomputed under a different set.

**Aggregation and inference (E1 convention).** group -> scene mean (3
groups) -> seed mean -> paired scene comparison -> mean over 173; 10,000
paired scene bootstraps, bootstrap seed 20260803, 95% CIs;
improved/tied/worsened counts, median paired delta, 10% trimmed mean,
leave-one-scene-out range, max single-scene contribution, per-fold and
per-seed tables. Undefined metrics stay undefined and are counted.

Write contracts/t26g_fm0_prediction_lock_contract.json,
contracts/t26g_fm0_endpoint_and_statistics_contract.json.

---

## 8. Phase 6 — comparison graph and the ten gates

Comparison graph:

```text
M1 FULL - LASTTOKEN      M2 FULL - NULL       M3 LASTTOKEN - NULL
M4 FULL - A_LEGACY_MATCHED                    M5 LASTTOKEN - A_LEGACY_MATCHED
```

Nomination arm: P_FULL_L2_CONS only. Gates, frozen before FM1; margins are
REREAD from the frozen E1/FA0 contracts and hash-registered (the numerals
here are expectations, not sources):

```text
G1  M1 regret:   delta < 0 with 95% CI high < 0
G2  M1 top-1:    delta > 0 with 95% CI low  > 0
G3  M2:          CI-supported FULL superiority over NULL on BOTH primary
                 endpoints
G4  M4 safety:   regret non-inferiority CI high < +0.005 (or superiority);
                 pairwise non-inferiority at margin 0.010; top-1 margin
                 resolved by written rule from the frozen materiality
                 constant (0.046) BEFORE FM1
G5  mechanism:   WRONG_SCENE_FULL_L2 and ZERO_FULL_L2 each worsen at least
                 one primary endpoint in the preregistered direction
                 (materiality 0.013 regret / 0.046 top-1), with no
                 contradictory material improvement on the other; the
                 wrong-scene donor map is a deterministic scene-level
                 bijection frozen in FM0
                 (contracts/t26g_fm0_wrong_scene_mapping.json); no
                 candidate-specific generated hidden state may be used
G6  K-consistency: no material reversal across K=2/5/8 on either primary
                 endpoint
G7  robustness:  no seed material reversal; no fold systematically opposed;
                 LOSO stays in the favorable region; no single scene
                 contributes more than half the primary effect
G8  non-collapse: FULL's future/safety auxiliary predictions not materially
                 worse than LASTTOKEN's (margins frozen before FM1)
G9  budget:      trainable params <= 2,000,000; median incremental K8
                 selector latency <= 10 ms, p95 <= 15 ms; peak incremental
                 inference memory <= 1 GiB; one group-level L2
                 projection/KV cache shared across K
G10 integrity:   60/60 runs; outer-fold isolation; lock before endpoints;
                 all FA2 joins/hashes verified; no reserve/protected/sealed
                 access; all manifests reread
```

PASS requires G1–G10. If FM1 fails, the lightweight full-L2 reader line
CLOSES — no post-hoc tuning, no gate reinterpretation, no re-run of the same
reader on outer-fold outcomes. Write
contracts/t26g_fm0_comparison_graph_contract.json,
contracts/t26g_fm0_promotion_gate_contract.json,
contracts/t26g_fm0_wrong_scene_mapping.json,
qa/t26g_fm0_gate_resolution_qa.json.

---

## 9. Phase 7 — future-state materialization (only after all freezes)

Known state (§0.3): three auxiliary labels already exist; ONLY
future_ego_states_ego_t0 is missing. Materialize it now, read-only:

```text
source     the 4,152 frozen FA2 scoring rollout ASLs (no simulation)
semantics  the registered T26F-B3-A extractor
           (/home/qiren/alpasafe/scripts/t26f_b3_a_extract_targets.py;
            AlpaSim venv reader alpasim_utils.logs; decision anchor = first
            driver_request timestamp; aabb->rig via the anchor first_return
            pose; frames >= decision, <= 64)
adaptation paths only; adapted code registered and F821-clean
validation per candidate: ASL scene match, decision-timestamp match, frame
           count in [1,64], finite values
outputs    FM0_REC/arrays/ (per group [8,<=64,2] fp32 + valid lengths),
           manifests/t26g_fm0_future_state_manifest.json (hashes),
           results/t26g_fm0_label_audit.json, QA
cost       streaming ~544 GB, CPU-only, parallel readers, CPU <= 16, 1-3 h
```

Zero tolerance: any unextractable candidate ->
BLOCKED_T26G_FM0_AUXILIARY_LABEL_CLOSURE_FAILURE; no imputation, no
in-training re-derivation. Reading these arrays must not alter any Phase 2–6
frozen constant. Also freeze the label-mapping contract
(contracts/t26g_fm0_label_mapping_contract.json): the four auxiliary labels
supervise consequence heads only and never reconstruct the official score;
NC/DAC/TTC/Comfort/EP/PDMS, NAVSIM/nuPlan and WoTE rewards remain barred.

---

## 10. Phase 8 — synthetic conformance, budget, and the FM1 concurrency plan

**Synthetic conformance (no real targets).** Fixed-seed synthetic tensors
prove: shapes and masks at L=3086; parameter-count equality of the matched
trio; candidate-independence invariances (§4.3); bit-reproducibility of one
fixed (fold, seed) fixture under the frozen determinism flags.

**Budget measurement vs G9 ceilings.** On real L2 inputs with synthetic
targets: forward-only K8 latency (median/p95, warmups and synchronization
recorded), peak incremental inference memory, loader throughput with the 13
GiB L2 store RAM-cached vs mmap (freeze the strategy), fwd+bwd peak VRAM per
training worker.

**FM1 training-concurrency plan (frozen here, executed as FM1 preflight).**
Benchmark TRAIN_WORKERS in {1, 2, 4}, each worker a distinct (fold, seed)
shard, bounded to <= 40 optimizer steps on synthetic targets. Eligibility: 0
OOM, 0 numerical/hash failures, total CPU <= 16 with per-worker thread caps,
safe VRAM/host-RAM headroom, and the determinism check — the designated
fixture re-run under concurrent load must be bit-identical to its solo run.
Selection: highest eligible level with >= 10% aggregate throughput gain over
the previous eligible level, else the previous. The rollout worker count (8)
does not transfer to training; do not assume it. Freeze every remaining
training scalar (batch/group size, grad accumulation, optimizer, LR,
weight decay, clip, max epochs, patience, AMP policy, loader workers) from
the F0B recipe; any scalar F0B leaves unresolved is resolved ONCE by a
written outcome-blind rule or blocks.

Write: contracts/t26g_fm0_training_and_compute_contract.json,
results/t26g_fm0_synthetic_budget_benchmark.json,
qa/t26g_fm0_synthetic_architecture_dry_run_qa.json.

---

## 11. Phase 9 — FM1 prompt generation and FM0 finalization

Only if every FM0 phase closed, generate
prompts/SafeWorld_T26GFM1_FINAL_Fable5_FullL2_Matched_Selector_Execution_Prompt_<DATE>.md
and register its full sha256 + size. It must carry: the exact authoritative
paths and hashes from Phase 0; the four frozen arms and code hashes; the
60-run population; objective/checkpoint/tie contracts; folds and seeds; the
TRAIN_WORKERS preflight; prediction lock; endpoints/statistics; the
wrong-scene/zero ablations; all ten gates applied exactly once from frozen
contracts; record/payload separation; resume and failure behaviour; the
F821 code gate; the reserve-cohort prohibition; and exactly two terminal
outcomes:

```text
T26G_FM1_FULL_L2_PROMOTABLE_READY_FOR_FM2_CONFIRMATION
T26G_FM1_NO_FULL_L2_GAIN_CLOSE_LIGHTWEIGHT_READER_LINE
```

On PASS, FM1 may only GENERATE the FM2 confirmation preregistration for the
35 frozen scenes — never access them. Do not launch FM1.

Finalize FM0: hash-registered final manifest (post-manifest files excluded),
reread QA, reports/T26G_FM0_full_l2_matched_selector_preregistration.md,
reports/status_after_T26G_FM0.md, and exactly one status:

```text
T26G_FM0_FULL_L2_MATCHED_SELECTOR_PREREGISTERED_READY_FOR_T26G_FM1
```

or the precise blocker:

```text
BLOCKED_T26G_FM0_AUTHORITY_OR_INTEGRITY_FAILURE
BLOCKED_T26G_FM0_RESERVE_COHORT_CONTAMINATION
BLOCKED_T26G_FM0_AUXILIARY_LABEL_CLOSURE_FAILURE
BLOCKED_T26G_FM0_LEGACY_REFERENCE_NOT_RECONSTRUCTABLE
BLOCKED_T26G_FM0_ARCHITECTURE_OR_MATCHED_CONTROL_FAILURE
BLOCKED_T26G_FM0_GATE_OR_STATISTICAL_RESOLUTION_FAILURE
```

---

## 12. Required outputs

```text
audit/t26g_fm0_authority_integrity_audit.json
audit/t26g_fm0_dataset_and_label_closure_audit.json

contracts/t26g_fm0_untouched_confirmation_cohort_contract.json
contracts/t26g_fm0_label_mapping_contract.json
contracts/t26g_fm0_architecture_contract.json
contracts/t26g_fm0_arm_contract.json
contracts/t26g_fm0_objective_checkpoint_contract.json
contracts/t26g_fm0_tie_semantics_contract.json
contracts/t26g_fm0_run_population_contract.json
contracts/t26g_fm0_prediction_lock_contract.json
contracts/t26g_fm0_endpoint_and_statistics_contract.json
contracts/t26g_fm0_comparison_graph_contract.json
contracts/t26g_fm0_promotion_gate_contract.json
contracts/t26g_fm0_wrong_scene_mapping.json
contracts/t26g_fm0_training_and_compute_contract.json

splits/t26g_fm0_scene_folds.json
folds/t26g_fm0_fold_composition.json
arrays/ + manifests/t26g_fm0_future_state_manifest.json
manifests/t26g_fm0_derived_fa2_payload_pointer.json

results/t26g_fm0_parameter_count.json
results/t26g_fm0_label_audit.json
results/t26g_fm0_synthetic_budget_benchmark.json

qa/t26g_fm0_authority_integrity_qa.json
qa/t26g_fm0_untouched_confirmation_cohort_qa.json
qa/t26g_fm0_dataset_and_label_closure_qa.json
qa/t26g_fm0_architecture_conformance_qa.json
qa/t26g_fm0_objective_checkpoint_qa.json
qa/t26g_fm0_tie_semantics_qa.json
qa/t26g_fm0_split_and_leakage_qa.json
qa/t26g_fm0_gate_resolution_qa.json
qa/t26g_fm0_synthetic_architecture_dry_run_qa.json
qa/t26g_fm0_no_training_or_protected_access_qa.json
qa/t26g_fm0_final_manifest_reread_qa.json

reports/T26G_FM0_full_l2_matched_selector_preregistration.md
reports/status_after_T26G_FM0.md
manifests/t26g_fm0_final_manifest.json
status/T26G_FM0_FINALIZED
prompts/SafeWorld_T26GFM1_..._Execution_Prompt_<DATE>.md
```

---

## 13. Final terminal response

State plainly: (1) final FM0 status; (2) the FA2 authority set as verified,
with the dataset sha and derived payload-pointer result; (3) cohort counts,
shapes, joins, full hash inventory; (4) reserve-cohort protection; (5) the
label audit result — three pre-existing labels verified and the
future-state materialization count; (6) frozen architecture and exact
parameter counts; (7) the four arms and why only FULL is nominable; (8) the
objective and checkpoint policy, with the recorded rationale for
checkpoint-only; (9) tie semantics; (10) folds/seeds/60 runs; (11)
endpoints; (12) comparison graph and all ten gates with the reread margins;
(13) synthetic conformance and budget numbers vs G9; (14) the FM1 prompt
path, sha256 and size; (15) confirmation that no training, prediction,
endpoint, acquisition, rollout, evaluator run, reserve, sealed or protected
access occurred; (16) the sole next action — the user launches FM1 — not
executed.

Do not train. Do not compute endpoints. Do not touch the 35 confirmation
scenes. Do not start FM1.

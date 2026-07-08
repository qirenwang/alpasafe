# SafeWorld-Alpamayo Task Index

This is the canonical task log location for the project. Future experiment
attempts should be recorded only in this project-root `tasks/` directory.

Note: T00-T12 logs were reconstructed on 2026-06-03 after the duplicate
`researchs/SafeWorld_Alpamayo_Codex/tasks` directory was removed.

| Task | Status | Main files | Last updated | Notes |
|---|---|---|---|---|
| T00 | completed | `safeworld-alpamayo/` scaffold | 2026-06-03 | Importable package and scripts |
| T01 | completed | configs, utils | 2026-06-03 | Dry-run configs and YAML/JSONL helpers |
| T02 | completed | data schema/index | 2026-06-03 | 12-row synthetic sample index |
| T03 | completed | alpamayo wrapper | 2026-06-03 | 12 dry-run proposals and E0 report |
| T04 | completed | scenario mining | 2026-06-03 | Mined tags, category counts, split summary |
| T05 | completed | safety predicates | 2026-06-03 | Kinematic/TTC/RSS/offroad predicates with tests |
| T06 | completed | RAWC | 2026-06-03 | Claim parser and RAWC report |
| T07 | completed | counterfactual actions | 2026-06-03 | 8 candidates/sample; smooth stop fixed |
| T08 | completed | world model dataset | 2026-06-03 | 96 action-conditioned world targets |
| T09 | completed | world model V1 | 2026-06-03 | Engineered logistic V1 model and E3 report |
| T10 | completed | auditor/calibration/gate | 2026-06-03 | ECE/Brier metrics and gate decision logging |
| T11 | completed | open-loop eval | 2026-06-03 | E1/E2/E4/E6 dry-run experiment reports |
| T12 | completed | closed-loop/replay | 2026-06-03 | E5 limitation report; no AlpaSim configured |
| T13 | not_started | reports/figures | 2026-06-03 | Needs real-data or simulator-backed results |
| T14 | completed | alpamayo top-K wrapper | 2026-06-09 | `run_topk` K in {1,2,5,10}; dry-run proposals for all four K |
| T15 | completed | top-K candidate schema | 2026-06-09 | candidate_source vocabulary; legacy tau_* marked on load |
| T16 | completed | selection protocol | 2026-06-09 | 7 selectors; gate log + report (dry-run proxy only) |
| T17 | superseded | oracle labels / leakage audit | 2026-06-20 | Superseded by T23 (uses `oracle` naming + proxy labels; replaced by label-source schema + `best_achievable_in_sampled_set_by_evaluator`). Log preserved. |
| T18 | superseded | safeworld top-K selector | 2026-06-20 | Superseded by T21 `RewardSelector` (dry-run SafeWorldSelector design). Log preserved. |
| T19 | superseded | K sweep / recoverability | 2026-06-20 | Superseded by T27 plan (proxy-label recoverability). Log preserved. |
| T20 | superseded | real long-tail/OOD plan | 2026-06-20 | Superseded by T22 audit + T25/T27 plans. Log preserved. |
| T21 | completed | method_spec_v2 + v2 interfaces | 2026-06-20 | ConsequenceWorldModel/RewardHead/RewardSelector frozen; 4 docs; legacy marked `legacy_dry_run` |
| T22 | completed | AlpaSim/NuRec audit | 2026-06-20 | **BLOCKED_MISSING_ASSETS_OR_API**: no AlpaSim/NuRec; physical_ai_av data-only |
| T23 | completed | rollout/eval contracts | 2026-06-20 | schema_v2 + evaluator_v2; proxy barred from real CLI; predicted vs eval reward separated |
| T24 | blocked | real K=2 smoke | 2026-06-20 | BLOCKED report; no synthetic data; operator unblock steps documented |
| T24R1 | completed | access/env preflight | 2026-06-21 | HF access GO (all 3 repos); Alpamayo-1.5-10B IS cached (T22 looked at wrong path); infra sub-blockers found |
| T24R2 | blocked | AlpaSim single-scene bring-up | 2026-06-21 | **BLOCKED_RUNTIME_INCOMPATIBILITY**: alpasim v2026.5 pinned in external/; docker needs sudo (user not in docker group) + compose/buildx plugins; deps are large-download gated |
| T24R3 | completed | real Alpamayo 1.5 inference | 2026-06-21 | Standalone (NOT AlpaSim) real inference completed offline from cache; peak 21.7 GiB; no large download |
| T24R4 | completed | real K=2 native candidates + rollout | 2026-06-22 | K=2 native generation + cached-candidate execution + 2 AlpaSim rollouts DONE; `…/20260622T194814Z/` |
| T24R5 | completed | post-smoke semantics + provenance | 2026-06-28 | Reasoning↔trajectory alignment PROVEN (no rerun); full candidate→outcome provenance chain; structured RolloutSemantics; config/metric provenance corrected; v2.1 docs supersede v2 |
| T25A | completed | single-scene future-target preflight | 2026-06-28 | Real ASL parsed (both cands); FutureConsequenceTarget pilot (global+ego_t0, actor valid-masks); planned-vs-executed measured; **GO_T25_SMALL_SCALE**; T26 still NOT authorized |
| T25B | blocked | small-scale dataset generation | 2026-07-06 | Plan pre-registered; K=5 generation attempted → **BLOCKED_SMALL_SCALE_GENERATION** (docker-compose GPU-runtime regression + only 1 local scene); pilot QA'd; T26 still NOT authorized |
| T25B-diag | completed | blocker diagnosis | 2026-07-06 | GPU compose issue RESOLVED (transient, cleared by reboot; verified matrix); scene-split still blocked; **READY_RETRY_T25B_GENERATION** |
| T25B-retry | completed | bounded gen on existing scene | 2026-07-06 | 3 decisions × K=5 = **15/15 real samples** (100% ASL parse, anchor ≤1.9e-6 m, provenance complete); **GO_T25_OFFICIAL_SCENE_SELECTION**; T25C split still blocked; T26 NOT authorized |
| T25 | completed | multi-scene dataset + T25C split freeze | 2026-07-07 | 5 official scenes (4 downloaded, 7.54 GB); **75/75 samples** (5×3×K=5, 100% ASL parse); QA PASSED; **T25C split FROZEN** (3/1/1, leakage-free); **T25_COMPLETE_READY_FOR_T26_AUTHORIZATION**; T26 NOT authorized |
| T26 | planned | world model + reward training | 2026-06-20 | Placeholder only; not implemented this phase |
| T27 | planned | real top-K evaluation | 2026-06-20 | Placeholder only; not implemented this phase |

Last updated: 2026-07-07


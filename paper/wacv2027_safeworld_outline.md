# WACV 2027 Paper Skeleton — SafeWorld / AlpaSafe Trajectory Selector

> **DRAFT STATUS 2026-08-03**: full first draft now lives in
> `paper/wacv2027/` (main + supp, auto-number pipeline `make_numbers.py`,
> FN1 outcome toggles in main.tex) — see its `README.md` for the Overleaf
> file map and remaining TODOs. This outline stays the planning source.
>
> Working outline for the writing session. Target: WACV 2027 Round 2
> (registration Aug 21, submission **Aug 28, 2026** AoE). Sections marked
> `[DONE]` are backed by finalized, hash-registered results; `[PENDING-FN1]`
> get filled ~Aug 10 and have BOTH outcome branches pre-planned. 中文批注为
> 写作决策提示,正文骨架为论文语言(英文)。

---

## 0. Title candidates(选一,或混搭)

1. *Your Planner Already Knows: Frozen VLA Representations as Amortized
   Trajectory Evaluators*
2. *One Token Is Enough: Trajectory Selection from a Frozen Driving VLA's
   Prefill State*
3. *SafeWorld: Amortized Online Trajectory Evaluation without a World Model*

核心卖点排序(摘要与导言按此排):**(a)** 免费表征 + 1.7M/1.3ms 头即可选优;
**(b)** 反直觉负结果:整段序列不如末 token;**(c)** 证据标准(CI/预注册/
因果消融/前瞻封存)远超领域惯例;**(d)** [PENDING] 对强几何基线的足够功效判定。

## 1. Abstract (slot template)

- E2E planners emit K candidates; deployment picks one; minADE@K silently
  assumes an oracle selector — the gap is unmeasured in most work.
- Prior online evaluation (WoTE) trains an explicit BEV world model to
  imagine futures per candidate.
- We show the generator itself already computed the needed scene summary:
  the last token of a frozen 10B driving VLA's prompt prefill (L3), read by
  a 1.7M-param decomposed value head at 1.3 ms, yields a causally-verified,
  prospectively replicated selection signal (regret −0.007, top-1 +0.10,
  95% CIs excluding zero, on 173 untouched scenes).
- Surprisingly, the FULL 3086-token hidden sequence is CI-supported WORSE
  than its own last row.
- [PENDING-FN1 slot: adequately-powered verdict vs a trained
  trajectory-geometry baseline on 525 scenes → 两分支各预写一句]
- All studies preregistered with frozen gates; sealed 35-scene cohort.

## 2. Introduction

1. 开局钩子:minADE@K 的"上帝选择器"假设——报告的是 potential,部署的是
   actual;差值就是 selected regret,本文的研究对象。
2. Explicit route (WoTE): purpose-trained world model, online imagination,
   submetric reward supervision.(概念定位;按 8/9 决定,WoTE 实验数字一律不进本文)
3. Our question: does a driving VLA's own forward pass already encode the
   scene knowledge a selector needs? (representation-reuse, zero VLA
   training)
4. Same-prefill capture:候选与表征来自**同一次** prefill(比特级 L3=L2[-1]
   证明)——排除"表征与候选异源"混淆。
5. Contributions (4 条):
   - C1 Same-prefill capture protocol + open evaluation program design
     (preregistration, sealed cohorts, causal ablations) 「方法学」
   - C2 Frozen-VLA last-token representation carries scene-specific,
     causally-verified selection signal (CI-supported, prospective) 「正结果」
   - C3 Full-sequence reader is CI-supported worse than last-token-only —
     "more context hurts" under matched capacity 「负结果」
   - C4 [PENDING-FN1] The adequately-powered incumbent test (N=466 power
     0.80): либо promotion evidence, либо a clean boundary statement on
     when representation reuse pays. 两分支都写成贡献。

## 3. Related Work

- **Online trajectory evaluation**: WoTE (ICCV'25, arXiv:2504.01941) —
  256 K-Means anchors, 2-step autoregressive BEV imagination (t+2s/t+4s),
  imitation + NC/DAC/TTC/Comfort/EP simulation rewards; PDM/rule-based
  scorers; Hydra-MDP-style distillation.(仅架构理念对比;不引用其实验数据) 定位句:explicit imagination vs
  **amortized consequence knowledge** — we occupy the far end of that
  spectrum (0 inference-time rollout steps).
- **E2E planning & multi-modal output** (minADE convention critique).
- **Foundation-model representation probing/reuse** (frozen features →
  downstream heads);本文首次将其用于闭环轨迹选优并给出因果证据。
- **Evaluation rigor in AV/ML**(预注册在本领域的缺位)。

## 4. Method

- 4.1 Setting: AlpaSim closed-loop; official scene score ∈[0,1] as sole
  supervision (submetric aggregates deliberately barred — anti-gaming);
  scenes × 3 decision groups (timing tags A/B/C) × K=8 sampled candidates;
  24 scoring rollouts per scene.
- 4.2 **Same-prefill L3 capture**: hook inside the production predict();
  exactly-one-prefill asserted; L3 = last row of final-layer hidden states
  [4096]; bit-exact provenance chain (capture-time tensor hashes).
- 4.3 **Selector head** (SafeWorldF1, 1.7M params): score decomposed as
  group-mean + bounded candidate residual S·tanh(·/S) (fixes the
  group-offset artifact + rank/centered gradient conflict — cite internal
  repair line as design rationale); candidate independence (no
  cross-candidate attention); consequence heads (future ego [64,2],
  progress, collision, offroad) as **auxiliary-only** supervision =
  amortized world knowledge (deletion test: inference score bit-identical
  without them).
- 4.4 Objectives: checkpoint objective (rank-margin family) and selector
  objective (regret-weighted best-vs-rest logistic; τ=0.043, w=0.026
  derived mechanically, no tuning; gradient saturates exactly where
  expected-regret softmax vanishes — E0 criterion 5, 可作一个小 lemma/图).
- 4.5 Protocol: 5-fold scene CV × 3 seeds; inner-validation-only
  checkpointing (CONSTRAINED_SELECTOR_FIRST); prediction hash-locking
  before any endpoint; paired scene bootstrap (10k) CIs; preregistered
  gates frozen before outcomes.

## 5. Experimental Program(独立小节卖方法学)

- Cohort ladder: 122 dev → **173 prospective untouched** (FA2) →
  **+352 extension** (FN-A, running) → **35 sealed confirmation** (never
  opened until winner frozen).
- Metrics: selected regret & tie-aware top-1 @K8 (+K2/K5), scene-macro;
  与 minADE 的关系:regret ≡ oracle-selector gap actually achieved.
- Arms & ablations: scene-informed vs NULL (matched), vs trajectory-
  geometry-only legacy (202k MLP, same labels/folds), WRONG-SCENE swap &
  ZERO ablations, FULL vs LASTTOKEN visibility.

## 6. Results

- 6.1 `[DONE]` **Scene signal is real & causal** — Table 1 (M3: regret
  −0.0069 CI[−0.0124,−0.0012]; top-1 +0.0996 CI[+0.0668,+0.1330], 173
  untouched scenes) + ablation bars (wrong-scene/zero degrade); E1
  lineage as internal replication.
- 6.2 `[DONE]` **One token beats the sequence** — M1: FULL−LASTTOKEN
  regret +0.0085 CI[+0.0030,+0.0143], top-1 −0.0527 CI[−0.0822,−0.0238].
  讨论假说:reader capacity vs 3086-token dilution;matched-params 设计
  使解释收敛到"信息可及性"而非容量。
- 6.3 `[PENDING-FN1]` **The incumbent test at power 0.80** — design: N
  resolved from locked predictions (SD 0.0396, MDE 0.00514 → 466 scenes);
  branch A (pass): CI-supported superiority + non-inferior top-1;
  branch B (fail): boundary result — representation adds signal (6.1)
  but not enough to displace a tuned geometry baseline; report both
  candidates (checkpoint vs selector objective) + N3 trade-off contrast.
- 6.4 `[PENDING-FN1]` **Selector-loss trade** (buy top-1 without selling
  regret? margin +0.005) — E1 precedent (+0.083 top-1) as prior.
- 6.5 `[DONE]` **Efficiency** — Table 2, TWO rows (勘误 2026-08-03: 原稿的
  1.29ms/158MB 其实是 **FULL 序列读取头**,出自 FM1 gpu_gate): deployed
  LASTTOKEN path = 1.19 ms med / 1.21 ms p95, 3.5 MB (FN0 conformance,
  input [1,4096]) vs FULL reader = 1.29/1.30 ms, 165.2 MB — 47× 显存差
  且更省的那条臂反而更准,本身就是卖点 (representation is free;按 8/9 决定
  Table 2 不含 WoTE 行,WoTE 只在 Related Work 作概念定位).
- 6.6 `[PENDING-FN2]` sealed-cohort confirmation(只有 6.3 过门才存在)。

## 7. Discussion & Limitations

- Internal benchmark (AlpaSim official score) — no NavSim/Bench2Drive
  comparability; we trade leaderboard placement for closed-loop official
  scoring + statistical rigor. 直说,别藏。
- Absolute effects are small; that is the honest scale of selection gains
  at K=8 with a strong generator — and exactly why CIs/power matter.
- Single VLA (Alpamayo-1.5-10B), single simulator; K=8 (vs WoTE's 256).
- Amortized↔explicit spectrum: middle points (e.g., score consuming the
  predicted future) untested — future work.

## 8. Figures / Tables plan

| # | 内容 | 素材来源 |
|---|---|---|
| Fig.1 | Teaser: two routes to online evaluation (explicit imagination vs our amortized head);只标注我们自己的延迟 | 手绘 |
| Fig.2 | Same-prefill capture + provenance chain (bit-exact L3=L2[-1]) | FA2 capture 合约 |
| Fig.3 | Forest plot: all primary deltas w/ 95% CI (M1/M3/M5 → N1-N4) | FM1/FN1 analysis JSON |
| Fig.4 | Ablations: wrong-scene / zero / lasttoken-diag | FM1 ablations |
| Tab.1 | Main endpoints w/ CIs per comparison per cohort | 同上 |
| Tab.2 | Efficiency & budget vs WoTE | FN0 conformance JSON |
| Tab.3 | Cohort ladder + preregistration registry (prompt shas) | 各阶段 status |
| Supp | Ten gates, selector-loss derivation, negative-results timeline (compressed-L2, offset artifact, replay non-bit-exactness) | E0/C0/D0/FP0 records |

## 9. Claims inventory(写作会话的红绿灯总表)

| Claim | 状态 | 证据文件 |
|---|---|---|
| L3 carries causal scene-specific selection signal | 🟢 结案 | FM1 `results/t26g_fm1_analysis.json` (M3, ablations) |
| Full sequence worse than last token | 🟢 结案 | 同上 (M1) |
| Selector-loss gradient geometry (saturate vs vanish) | 🟢 结案 | E0 contract + FN0 `results/t26g_fn0_conformance_and_budget.json` |
| Efficiency ceilings | 🟢 结案 | FN0 conformance / FM1 gpu_gate |
| Beats geometry baseline at power 0.80 | 🟡 FN1 (~8/10) | FN1 record (未来) |
| Selector loss buys top-1 w/o selling regret | 🟡 FN1 | 同上 |
| Sealed-cohort confirmation | 🟡 FN2(条件性) | — |

## 10. 数字取用路径(写作时直接读这些 JSON,不要手抄)

```text
ART=/home/qiren/alpasafe/safeworld-alpamayo/artifacts
FM1: $ART/safeworld_t26g_fm1_full_l2_matched_selector_experiment/20260803T132326Z/results/t26g_fm1_analysis.json
FN0 power: $ART/safeworld_t26g_fn0_l3_pathway_consolidation_preregistration/20260803T210139Z/contracts/t26g_fn0_power_and_sample_size_contract.json
FN0 conformance(延迟/参数): 同记录 results/t26g_fn0_conformance_and_budget.json
E0 selector loss: $ART/safeworld_t26g_e0_top_of_list_selector_resolution/20260727T042825Z/contracts/t26g_e0_selector_loss_contract.json
FN-A(进行中): $ART/safeworld_t26g_fna_extension_acquisition/20260803T215243Z
WoTE 事实核对: arXiv:2504.01941 (ICCV 2025)
```

## 11. 时间线

```text
8/9   FN-A 终态(采集)     8/10-11  FN1 终审(训练+十门)
8/12  初稿冲刺开始          8/21     Round 2 注册
8/25  内部定稿              8/28     提交(AoE)
```

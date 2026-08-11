# WACV 2027 SafeWorld 论文稿 — 工作目录

目标:WACV 2027 Round 2(注册 8/21,提交 **8/28 AoE**)。
骨架来源:`../wacv2027_safeworld_outline.md`(取数路径、红绿灯总表、时间线都在那)。

## 文件 → Overleaf 对应关系

| 本地 | Overleaf | 说明 |
|---|---|---|
| `main.tex` | `main.tex` | 覆盖模板版(标题/作者/输入表已换,含 FN1 开关) |
| `preamble.tex` | `preamble.tex` | 覆盖,或只把 "PAPER-SPECIFIC ADDITIONS" 块追加进模板版 |
| `sec/0_abstract.tex` … `sec/6_discussion.tex` | `sec/` | 全部新文件;模板的 `2_formatting.tex`/`3_finalcopy.tex` 从 main.tex 移除引用即可 |
| `sec/numbers.tex` | `sec/numbers.tex` | **自动生成,勿手改**(见下) |
| `main.bib` | `main.bib` | 覆盖 |
| `supp.tex` | 新文件 | 补充材料,单独编译 |
| `figs/*.pdf` | `figs/` 目录 | 四张图的矢量版(只传 PDF;PNG 是预览) |

模板自带的 `wacv.sty`、`ieeenat_fullname.bst` 不动。

## 数字管线(零手抄)

所有正文数字来自锁定 JSON,经 `make_numbers.py` 生成宏:

```bash
python3 make_numbers.py                # 当前:FN1 槽位输出红色 [FN1] 占位
python3 make_numbers.py --fn1 <FN1_analysis.json>   # FN1 定稿后
```

来源(hash 注册记录):FM1 analysis / FM1 gpu_gate / FN0 power / FN0
conformance / E0 selector-loss contract(具体路径见 numbers.tex 头部注释)。

**注意**:骨架 §6.5 手写的 "1.29ms/158MB" 实为 **FULL 序列读取头**(FM1
gpu_gate);**部署的 LASTTOKEN 路径**是 1.19ms/3.5MB(FN0 conformance)。
Table 2 两条都列,对比本身是卖点(47×)。

## FN1 定稿流程(~8/10–11)

1. `python3 make_numbers.py --fn1 <FN1 记录>/results/..._analysis.json`
2. `main.tex` + `supp.tex`:`\fnonedonetrue`;过门则再 `\fnonepasstrue`
3. §6.3/6.4 双分支已预写,只需删占位段、填 `\todo{}`(门逐条结果、被提名臂)
4. 摘要/导言 C4 的分支句自动切换

## 图管线(已完成,与数字管线同源)

四张图全部由 `figs/make_fig*.py` 生成(PDF 进论文 + PNG 预览),数据图直接
读锁定 JSON,示意图里的延迟/参数量也从 JSON 取:

```bash
cd figs
python3 make_fig1_teaser.py      # Fig.1 两条路线 teaser(单栏,AV 风格 v2)
python3 make_fig_capture.py      # Fig.2 same-prefill 采集+出处链(单栏,§4.2)
python3 make_fig_arch.py         # Fig.3 架构主图,figure* 跨双栏(§4.3,AV 风格 v2)
python3 make_fig_results.py      # Fig.4 主对比柱状图;FN1 后加 --fn1 <analysis.json>
python3 make_fig4_ablations.py   # Fig.5 消融柱状图
```

**风格 v2(2026-08-09,按用户反馈)**:统计图从 forest 点线改为 CV 会议
惯用的柱状+95%CI 误差棒(数值不变,几何变);示意图走 AV 论文视觉语言
(`fancy.py`:透视道路+轨迹扇、相机堆、transformer 渐变堆叠、token 条
末位高亮、雪花冻结徽章、逐候选分数条、执行回显)。架构图里的分数条与
轨迹为示意(图注已声明),所有实测数字仍从锁定 JSON 读。
**注意**:`figure_prompts.html` 与 `safeworld_figures.pptx` 仍是 v1 风格,
风格定稿后需重新生成。

(架构图 2026-08-08 重做:原合并版 fig2_method 拆为 fig_capture +
fig_arch;fig_arch 端到端展示 冻结VLA一次前向 → 候选与 h 双产物 →
逐候选共享权重头(×8 堆叠卡)→ 分解输出与选择器,效率脚注取自 FN0
conformance。文件名与最终图号不必一致,LaTeX 自动编号。)

**网页版迭代辅助(2026-08-08)**:
- `figs/figure_prompts.html` — 自包含 prompt 包:五张图各一段独立完整
  prompt(数据/配色/布局全写死)+ 当前预览,贴进网页版 Claude 可重绘为
  SVG 并对话式修改;改动结论需回填 make 脚本,脚本仍是唯一真源
- `figs/safeworld_figures.pptx` — 图集 PPT:每页备注栏含同款 prompt;
  第 5 页为架构图原生形状版(框/箭头/文字均可在 PowerPoint 直接编辑)

配色:蓝/红 = CI 排除 0 的方向语义,灰 = CI 跨零;实心/空心为冗余编码
(灰度打印与色盲安全,双极已过验证器)。字体 Nimbus Roman 匹配正文 Times。
FN1 定稿后:`make_fig3_forest.py --fn1 …` 会用 N1–N4 实数替换灰色 pending 带。

## 未完事项(按优先级)

1. **引文核对**:main.bib 里所有 `TODO-VERIFY` 条目(尤其 WoTE 作者列表、
   Alpamayo 公开引用、WoTE 18.7ms 延迟数字——Fig.1/Table 2 也引用了它)
2. **E1 内部复现数字**:supp 与 §6.1/6.4 的 `\todo`,需从 E1 记录 JSON 取
   (可扩展 make_numbers.py)
3. **supp 负结果时间线 / registry 摘要**:标了 todo 的数值要回记录核对,
   digest 补全
4. **命名决策**:AlpaSim / Alpamayo 是否匿名化(preamble 里 `\simname`
   `\vlaname` 一处改)
5. **篇幅**:当前正文估 ~7.5–8 页(无图),加图后需裁剪;优先砍 §2 related
   与 §4 program 的重叠、§5 results 各小节的过渡句
6. 提交前 `grep -rn 'fnpending\|TODO' sec/ supp.tex` 必须为空

## 红绿灯(骨架 §9 的落地状态)

- 🟢 §6.1 场景信号因果成立(M3 + 消融)— 已写,数字已接
- 🟢 §6.2 一个 token 胜过整段序列(M1 + 诊断)— 已写,数字已接
- 🟢 §6.5 效率(双路径对比)— 已写,数字已接
- 🟡 §6.3/6.4 FN1 两分支 — 预写完毕,等数字
- 🟡 §6.6 sealed 确认 — 条件性小节(仅过门后存在)

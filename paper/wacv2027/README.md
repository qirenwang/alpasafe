# WACV 2027 AlpaSafe 论文稿 — 工作目录

> **命名(2026-08-24 定)**:全文统一用 **AlpaSafe**,不再使用 SafeWorld
> (代码侧的历史 run 目录/artifacts 路径保留原名,不动——那些是 hash 锁定
> 的 provenance 路径)。

> **2026-08-12 目录合并**:本目录现在是**唯一正版**,官方模板文件
> (`wacv.sty`、`ieeenat_fullname.bst`、`rebuttal.tex`、`sec/2_formatting.tex`、
> `sec/3_finalcopy.tex`、`README_template.md`)已并入,自成可编译项目
> (LaTeX Workshop 或 `latexmk -pdf main.tex` 直接跑)。
> 已删除:`AlphSafe_official/`、`AlphSafe_ready/`、`AlphSafe_ready.zip`、
> `safeworld_wacv2027_overleaf.zip`(均为本目录的派生副本)。
> `../AlphSafe.zip` 保留为官方模板原始存档。
>
> **模板合并规则**:main.tex / preamble.tex 是**官方模板原文 + 标记的最小
> 插入**(ALPASAFE ADDITIONS 块;官方注释与宏包一律未动)。批注宏用模板
> 自带的 `\TODO{}`;wacv.sty 已含 booktabs/subcaption 等,我们只额外加载
> cleveref(必须在 hyperref 之后)。
>
> **交付到 Overleaf(2026-08-24 起)**:不打包 zip,变更文件手动复制到
> Overleaf 对应路径(本目录自含 sty/bst,首次建项目也是逐个文件上传)。

目标:WACV 2027 Round 2(注册 8/21,提交 **8/28 AoE**)。
骨架来源:`../wacv2027_safeworld_outline.md`(取数路径、红绿灯总表、时间线都在那)。

## 文件 → Overleaf 对应关系

| 本地 | Overleaf | 说明 |
|---|---|---|
| `main.tex` | `main.tex` | 覆盖模板版(标题/作者/输入表已换,含 FN1 开关) |
| `preamble.tex` | `preamble.tex` | 覆盖,或只把 "PAPER-SPECIFIC ADDITIONS" 块追加进模板版 |
| `sec/0_abstract.tex` … `sec/6_discussion.tex` | `sec/` | 全部新文件;模板的 `2_formatting.tex`/`3_finalcopy.tex` 从 main.tex 移除引用即可 |
| `main.bib` | `main.bib` | 覆盖 |
| `supp.tex` | 新文件 | 补充材料,单独编译 |
| `figs/*.pdf` | `figs/` 目录 | 四张图的矢量版(只传 PDF;PNG 是预览) |

模板自带的 `wacv.sty`、`ieeenat_fullname.bst` 不动。

## 数字:直接写在正文里(2026-08-12 起)

原先的 `sec/numbers.tex` 宏层**已移除**,177 处宏全部替换成字面数字。
查数字、追来源、改数字看 **`../data_source/`**:

- `../data_source/paper_numbers.md` — 按章节组织的全部数字 + 来源标注
- `../data_source/locked_records/` — 六份原始 hash 锁定 JSON
- `../data_source/extract_numbers.py` — 记录更新时重新生成主表

改一个数字时注意它常在多处出现(如 −0.0069 出现在摘要/导言/结果/讨论 4 处),
用 `grep -rn "0.0069" sec/` 确认改全。

<details><summary>历史:原 numbers.tex 管线(已废弃)</summary>

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

</details>

## FN1 状态(已完成 2026-08-09)

FN1 未过门:两个候选臂均 G1(regret 优越性)+ G8(稳健性/队列一致性)失败,
selector 候选另加 G5 失败 → **走边界结论分支**。开关已置为
`\fnonedonetrue` + `\fnonepassfalse`(main.tex / supp.tex)。
数字已写入正文;N1/N2 引用的是 **checkpoint 候选**那一组
(见 `../data_source/paper_numbers.md` 第 5 节,两组值都列了)。

剩余:§5.3/5.4 的 `\TODO{}`(逐门结果、被提名臂措辞)、supp 门表加结果列。

## 图管线(已完成,与数字管线同源)

> **架构图改用手工版(2026-08-24)**:`3_method.tex` 的 fig:arch 现在引用
> `figs/AlpaSafe.png`(Overleaf 上手工维护,由 SafeWorld.png 改名而来)。
> 仓库里的 `figs/AlpaSafe.png` 只是本地编译占位(fig_arch.png 的副本),
> **不要复制到 Overleaf 覆盖真图**。`make_fig_arch.py`/`fig_arch.pdf`
> 保留作历史脚本真源,但当前未被正文引用。

四张图全部由 `figs/make_fig*.py` 生成(PDF 进论文 + PNG 预览),数据图直接
读锁定 JSON,示意图里的延迟/参数量也从 JSON 取:

```bash
cd figs
python3 make_fig1_teaser.py      # (已弃用 2026-08-24:teaser 图从论文删除,脚本仅存档)
python3 make_fig_capture.py      # Fig.2 same-prefill 采集(单栏;2026-08-24 重画:去掉 SHA/出处链,只留一次前向两产物 + h=H[-1])
python3 make_fig_arch.py         # Fig.3 架构主图,figure* 跨双栏(§4.3,AV 风格 v2)
python3 make_fig_results.py      # Fig.4 主对比柱状图;FN1 后加 --fn1 <analysis.json>
python3 make_fig4_ablations.py   # Fig.5 消融柱状图
```

**风格 v2(2026-08-09,按用户反馈)**:统计图从 forest 点线改为 CV 会议
惯用的柱状+95%CI 误差棒(数值不变,几何变);示意图走 AV 论文视觉语言
(`fancy.py`:透视道路+轨迹扇、相机堆、transformer 渐变堆叠、token 条
末位高亮、雪花冻结徽章、逐候选分数条、执行回显)。架构图里的分数条与
轨迹为示意(图注已声明),所有实测数字仍从锁定 JSON 读。
**注意**:`figure_prompts.html` 与 `alpasafe_figures.pptx` 仍是 v1 风格,
风格定稿后需重新生成。

(架构图 2026-08-08 重做:原合并版 fig2_method 拆为 fig_capture +
fig_arch;fig_arch 端到端展示 冻结VLA一次前向 → 候选与 h 双产物 →
逐候选共享权重头(×8 堆叠卡)→ 分解输出与选择器,效率脚注取自 FN0
conformance。文件名与最终图号不必一致,LaTeX 自动编号。)

**网页版迭代辅助(2026-08-08)**:
- `figs/figure_prompts.html` — 自包含 prompt 包:五张图各一段独立完整
  prompt(数据/配色/布局全写死)+ 当前预览,贴进网页版 Claude 可重绘为
  SVG 并对话式修改;改动结论需回填 make 脚本,脚本仍是唯一真源
- `figs/alpasafe_figures.pptx` — 图集 PPT:每页备注栏含同款 prompt;
  第 5 页为架构图原生形状版(框/箭头/文字均可在 PowerPoint 直接编辑)

配色:蓝/红 = CI 排除 0 的方向语义,灰 = CI 跨零;实心/空心为冗余编码
(灰度打印与色盲安全,双极已过验证器)。字体 Nimbus Roman 匹配正文 Times。
**待办**:`make_fig_results.py --fn1 <FN1 analysis.json>` 尚未跑,
图里 N1–N4 仍是灰色 pending 带,需要刷新。

## 未完事项(按优先级)

1. **引文核对**:main.bib 里其余 `TODO-VERIFY` 条目(尤其 WoTE 作者列表、
   WoTE 18.7ms 延迟数字——Fig.1/Table 2 也引用了它);Alpamayo 公开引用
   已核实修正为 arXiv:2511.00088(2026-08-24)
2. **E1 内部复现数字**:supp 与 §5.1/5.4 的 `\TODO`,需从 E1 记录 JSON 取
   (取到后加进 `../data_source/`)
3. **supp 负结果时间线 / registry 摘要**:标了 todo 的数值要回记录核对,
   digest 补全
4. **命名决策**:AlpaSim / Alpamayo 是否匿名化(preamble 里 `\simname`
   `\vlaname` 一处改)
5. **篇幅(2026-08-24 晚实测,teaser 已删)**:正文到第 9 页左栏底,
   超 8 页限 ≈ 1 栏(~0.5 页),不砍会 desk reject;候选刀口:§3.4
   "Why not expected regret?" 段压缩或移 supp、Table 1(cohort ladder)
   移 supp、§4 Arms 段与 Table 3 图注的重复、Discussion 各段各收 1–2 句、
   (备选)Fig.1 capture 图与架构图(a)面板信息重叠,整图可删省 ~1/3 栏
6. 提交前 `grep -rn 'fnpending\|TODO' sec/ supp.tex` 必须为空

## 红绿灯(骨架 §9 的落地状态)

- 🟢 §6.1 场景信号因果成立(M3 + 消融)— 已写,数字已接
- 🟢 §6.2 一个 token 胜过整段序列(M1 + 诊断)— 已写,数字已接
- 🟢 §6.5 效率(双路径对比)— 已写,数字已接
- 🟢 §5.3/5.4 FN1 边界结论 — 数字已接,措辞待收尾
- ⚫ §5.6 sealed 确认 — 不存在(FN1 未过门,35 场景保持封存)

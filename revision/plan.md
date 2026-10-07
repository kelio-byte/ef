# Fragment-level 后续纠错机制验证计划

创建日期：2026-08-21。
目标：验证 SPE-M500 的较好生成结果是否真的来自“早期错误后可由 SUB/DEL 修复”，而不只是首次编辑或 token completion 更准确。

> **方法修正（2026-08-22）**：早先基于单条 Levenshtein traceback 的
> `first-off-oracle` 与旧版 P2 结果只保留作历史记录，不能作为最终机制结论。
> 不同编辑顺序可能都能到达同一目标；旧版错误干预也经常没有让实际 token 距离变差。
> 最终结论以本文后面的顺序无关重分析和新版配对干预为准。

## 执行状态

| 顺序 | 现在要做什么 | 完成标志 | 是否需要 GPU |
|---|---|---|---|
| **P0（已完成）** | 新增一个**只读诊断脚本**：逐条记录 Euler 的首个真实编辑、后续 INS/SUB/DEL、到 target 的 token 编辑距离，并以正式的 canonical SMILES 判定最终是否命中。先在 10 个 reaction、每个 2 条轨迹上 smoke test。 | 40 个自动测试通过；操作实现、批量距离和 damage=1 smoke 均通过。 | 已完成 |
| **P1（已完成）** | 用顺序无关的距离进展规则，对 Atom@600K 和 SPE-M500@490K 各跑 `1,000 reaction × 9 trajectories × 3 seeds` 的**自然轨迹**；另以 seed=42 覆盖同一 1,000 个 reaction 的全部 20 个 augmentation。 | 已得到 full/partial/neutral/harmful 首事件比例及 reaction-cluster bootstrap；20 个写法中 M500 的有害首事件均更少。 | 已完成 |
| **P2（已完成）** | 只保留自然首事件中实际完整改善的路径；在同一位置/类型只改一个 INS/SUB token，并要求首事件距离恰好恶化 1；与原路径严格配对比较。 | 2 个模型 × 2 条条件 × 3 seeds，共 12 个 run；damage 全部为 1。 | 已完成 |
| **P3（按停止规则不执行）** | 原计划生成轨迹 HTML；但 P1/P2 已足以回答机制问题，且停止规则未满足继续研究条件，因此不追加可视化采样。 | `revision/report.md` 给出支持/不支持后续纠错机制的结论。 | 不需要 GPU |

**当前 P0、P1、P2 已全部完成。** P3 可视化按预设停止规则不再追加；本轮不重新训练、不扫 checkpoint、不引入新 sampler。

旧版 P2 的结果目录 `revision/results/intervention/` 不用于最终结论；新版结果位于
`revision/results/intervention_order_invariant_v3/`，汇总位于
`revision/results/intervention_order_invariant_v3_summary/`。

## 1. 问题与可证伪假设

当前已知：改进后 global R-SMILES + SPE-M500@490K 在 dev 上是当前 fragment-level 主 baseline；但这**不能自动证明** M500 的更多 SUB 能带来后续纠错。

需要区分：

- **H1：首次编辑优势。** M500 只是更少在第一次真实编辑时选错位置、类型或 token。
- **H2：后续纠错优势。** 即使第一次真实编辑错误，M500 也更常在后续通过 SUB/DEL 回到 target。

本文重点验证 H2。若数据只支持 H1，则汇报时应将原始 Motivation 改写为“改善初始局部编辑/token completion”，不能声称“增强后续纠错”。

## 2. 冻结对象与统一协议

| 项目 | 固定选择 |
|---|---|
| Atom 对照 | `new_checkpoints/checkpoint_step600000.pt` |
| Fragment 对照 | `new_checkpoints/spe_m500_checkpoints/checkpoint_step490000.pt` |
| 表示 | 改进后 global R-SMILES；各自 tokenizer 的投影文件 |
| 数据 | `evaluation_v2/dev_unique1000_aug20` 的 1,000 个 reaction block |
| augmentation | 每个 block 固定取第 1 条，即每种表示 1,000 输入；不重复计入 20 条增强视图 |
| sampler | 普通 Euler，N=9，100 steps，cubic，`event_prob_mode=poisson` |
| seeds | 42、7、123 |
| 最终正确性 | 与正式评估一致：拼接 token、inverse global alignment、RDKit canonical SMILES 比较 |

不用 R9K1M2、Euler-Beam 或 structured sampler；本任务只检验表示对普通 Euler 轨迹的影响。

## 3. 现有代码可复用部分与必要修正

可复用：

- `sample_euler(..., record_first_events=True)`：可低成本记录首次真实事件。
- `sample_euler_with_first_step_intervention(...)`：已有正常、强制正确首编辑、强制错误首编辑的基础框架。
- `compute_oracle_model_output(...)`：可针对每个动态状态重新计算 oracle。

不能直接使用旧 `scripts/first_event_impact_analysis.py` 的汇总结果，原因：

1. 历史结果使用旧 Atom checkpoint、test 子集和 linear scheduler，不是当前比较对象。
2. 旧 `event_set_correct` 只比较编辑位置集合，不能保证 INS/SUB token 正确。
3. 旧汇总直接比较 decoded token 字符串，而非正式 canonical SMILES。
4. 它只记录首次事件，不能说明之后是否发生了实际纠错。

实现原则：新建独立诊断入口，不改普通训练和正式评估路径。推荐新增：

```text
scripts/trajectory_correction_analysis.py
edit_flows/analysis/trajectory_correction.py
```

所有输出写入本目录下的 `results/`、`logs/`、`figures/` 与 `report.md`。

## 4. 指标定义

### 4.1 首次真实事件

忽略 no-op；每条 Euler path 的第一步非空编辑称为 first event。

对每个实际 action，以当前状态 `x_t` 对 target `x_1` 动态计算 oracle。一个 action 为 oracle-consistent，当且仅当：

- 编辑位置是 oracle 正位置；
- 编辑类型为 oracle 支持的 INS/SUB/DEL 类型；
- 若为 INS/SUB，采样 token 在 oracle token support 内（非仅比较 argmax）。

一个 first event 为 **fully oracle-consistent**，当且仅当其中所有 action 都满足上述条件。另行记录 anchor 的 position/type/token 正确性，但不能以 anchor 代替完整事件判断。

### 4.2 主要统计量

| 名称 | 定义 | 回答的问题 |
|---|---|---|
| First-off-oracle rate | `P(first event 非 fully oracle-consistent)` | 谁更少早期犯错？ |
| Clean-path success | `P(final hit | first event fully oracle-consistent)` | 后续正常建模能力如何？ |
| Natural recovery rate | `P(final hit | first event off-oracle)` | 自然出现早期错误后谁更能恢复？ |
| Recovery share | 成功路径中 first-off-oracle 的占比 | 成功是否允许经偏离路径达成？ |
| Post-error SUB/DEL rate | first-off-oracle 后各路径的后续 SUB/DEL 数量与比例 | 是否实际使用可能的纠错操作？ |
| Edit-distance rebound | 首次错误后，后续事件使 token edit distance 降低的比例 | 是否真的向 target 回归？ |
| Validity | 条件 invalid/valid rate、最终 edit distance | 恢复失败是化学无效还是近似错误？ |

`Natural recovery rate` 是验证 H2 的主观测指标；不能只看总体 Top-1 或 SUB 占比。

## 5. P0：实现审计与 smoke test

### 工作

1. 实现 canonical final scoring，复用正式 global SMILES 解码规则。
2. 实现 token-aware、全事件的 oracle-consistency 判定。
3. 实现轻量级后续事件 trace：只保存 `step/t`、INS/SUB/DEL 数、编辑前后 token edit distance、最终 canonical 结果；**不保存完整 logits**。
4. 加入 `--seed`，保证每个 run 可复现；不同 intervention mode 前重置随机数。
5. 写单元测试：
   - INS/SUB token 错误不能被判为正确；
   - 只比较位置而 token 错误时，fully-consistent 为 false；
   - canonical 等价的最终 SMILES 被判为 hit；
   - 开启 compact recorder 不改变同 seed 普通 Euler 的输出；
   - M500 token 拼接后可正确进入 global canonicalization。

### Smoke protocol

- 每个模型 10 个 reaction、N=2、seed=42。
- 检查：行索引和 target 对应、无 OOV、诊断前后最终 token 输出逐行一致、结果 JSON 可重算汇总指标。

### 通过条件

- 所有新增测试通过；
- 同 seed 的 normal sampler 最终输出与未记录模式完全一致；
- Atom 与 M500 的同一 reaction ID 指向同一原始反应。

## 6. P1：自然轨迹相关性分析

### 运行

对 Atom@600K 与 M500@490K 分别在 1,000 个固定 reaction × 9 paths × 3 seeds 上运行，共 27,000 条轨迹/模型。

每个 seed 单独保存：

```text
revision/results/natural/{atom,m500}/seed_{42,7,123}/summary.json
revision/results/natural/{atom,m500}/seed_{42,7,123}/per_reaction.jsonl
revision/logs/natural_*.log
```

### 汇总

- 每个 seed 独立报告 4.2 的全部指标。
- 按 reaction block 做 paired/cluster bootstrap，输出 M500 − Atom 的差、95% CI。
- 不把 9 条 path 当成独立 reaction；同一个 reaction 的 9 条轨迹在 bootstrap 中整体重采样。

### P1 的解释边界

P1 只说明相关性。若 M500 的自然 recovery rate 更高，仍可能因为“较容易恢复的样本恰好更多发生首错”；因此必须进入 P2。

### P1 顺序无关重分析结果

为避免把某一条 Levenshtein traceback 当成唯一正确编辑顺序，首事件改按实际距离变化分类：
完整改善、部分改善、距离不变、距离变差。Atom 与 M500 的完整改善比例几乎相同
（`64.24%` vs `64.45%`，差异不显著）；M500 的距离变差比例更低（`19.01%` vs
`25.99%`），但距离不变比例更高（`16.34%` vs `9.66%`）。因此修正后的 P1 只能说明
M500 更少出现明确的有害首事件，不能再声称它拥有更高的“正确首编辑率”。自然轨迹的
最终命中率为 Atom `35.32%`、M500 `36.38%`。完整表格见
`revision/results/natural/order_invariant_summary/summary.md`。

随后用固定 seed=42 对同一 1,000 个 reaction 的全部 20 个 R-SMILES augmentation 复核。统计时先在 reaction 内聚合 20×9 条轨迹，再以 reaction 为 bootstrap 单位。M500 的距离变差比例比 Atom 低 `7.12 pp`（95% CI `[-8.24, -6.04] pp`），且 20 个 view 的差值均为负；首事件平均距离变化增加 `0.0782`（95% CI `[+0.0563, +0.1007]`），20 个 view 均为正。完整改善率仍无显著差异。这排除了“仅第 1 个 augmentation 写法造成现象”的解释，但不改变 P1 只能说明首事件相关性、不能证明后续纠错机制的边界。完整结果见 `revision/results/augmentation_robustness/summary/summary.md`。

## 7. P2：顺序无关的错误 token completion 因果干预

### 为什么不能只用已有 `force_wrong_first`

现有模式优先构造 oracle 位置外的高分错误 anchor，验证的是“错误位置”破坏性；原始 Motivation 具体讨论的是“前面插入了错误 token 后，能否通过 SUB/DEL 修正”。两者不同。

### 新干预模式

旧版 `force_correct_completion_first` / `force_wrong_completion_first` 使用固定 oracle
anchor，不能排除编辑顺序问题，因此只保留为历史结果。新版在第一次真实事件发生时：

- `progress_compatible_first`：只保留实际距离恰好减少有效 action 数量的自然事件，作为配对控制；
- `force_harmful_completion_first`：保持同一位置、类型和其他 action，只改一个 INS/SUB token，
  并要求首事件后的距离比控制状态恰好增加 1。

每条轨迹不强行制造不可匹配的错误；按实际类型统计 INS/SUB 覆盖和结果。两种模式按
`seed/reaction/path` 配对，只汇报两边都 applied 的路径。

### 运行与主指标

仍使用 1,000 reaction、N=9、3 seeds、普通 Euler。核心因果量：

```text
Matched harmful-path hit drop =
P(final hit | progress-compatible control)
- P(final hit | matched harmful completion)
```

同时报告：

- 配对控制到 harmful 的逐路径最终命中率下降；
- 强制错误后后续 SUB/DEL 的数量和 edit-distance rebound；
- Atom 与 M500 各自的恢复率，再报告 M500 − Atom 的 cluster-bootstrap CI。

跨 tokenizer 的“一个错误 fragment”和“一个错误 atom”化学破坏程度不完全相同。因此主要结论应先比较每个模型从 forced-correct 到 forced-wrong 的相对恢复能力，再谨慎比较二者差异；不得将两者称为完全相同的化学扰动。

### P2 顺序无关配对干预结果

新版 P2 只在自然首事件实际“完整改善”的路径上进行干预，并保持首事件的所有位置、
动作类型和其他 action 不变；只替换一个 INS/SUB token。候选 token 必须让首事件后的
token 编辑距离比控制路径恰好增加 1。两种条件按 `seed/reaction/path` 配对，只统计
两边都成功应用的路径。

| 模型 | 配对路径 | 配对反应 | 控制最终命中率（逐路径） | 有害最终命中率（逐路径） | 命中率下降 | 控制→有害后续 SUB/DEL | 控制→有害后续距离下降 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Atom@600K | 13,406 | 864 | 53.57% | 0.74% | 52.83 pp | 0.328 → 0.376 | 2.909 → 2.155 |
| SPE-M500@490K | 13,775 | 838 | 52.52% | 2.13% | 50.39 pp | 0.934 → 1.010 | 2.154 → 1.775 |

两模型的有害条件命中率都接近 0，说明一个立即变差的首 token 通常会让后续路径
无法回到正确答案。虽然有害组后续 SUB/DEL 数量略有增加，但后续距离下降事件反而
减少；这不支持“后续动作数量增加就代表成功纠错”。所有已应用干预的 damage 均为 1。
完整 JSON/Markdown 见 `revision/results/intervention_order_invariant_v3_summary/`。

## 8. P3：预注册规则下的轨迹可视化

原计划从 P1/P2 已生成的结果中，不人工挑选样例，而按固定规则随机抽取：

- M500 首错后恢复成功：4 个；
- M500 首错后未恢复：4 个；
- Atom 首错后恢复成功：4 个；
- Atom 首错后未恢复：4 个。

对每个样例使用现有轨迹 HTML，再补充 token-aware 事件标签。输出放在：

```text
revision/figures/trajectory_html/
```

可视化仅解释聚合结果，不作为主要统计证据。本轮因 P1/P2 已明确不支持原始 H2，按停止规则不生成额外 HTML。

## 9. 结论门槛与停止规则

### 可以支持原始 Motivation

须同时满足：

1. P1 中 M500 的 natural recovery rate 高于 Atom，且 3 个 seed 方向一致；
2. P2 中 M500 的 forced-error recovery 也高于 Atom，或至少显著高于其自身的随机/错误基线；
3. M500 的恢复成功路径在后续 SUB/DEL 与 edit-distance 回落上有明确富集，而不是只比 Atom 多做无效 SUB；
4. reaction-cluster bootstrap 的主差异 95% CI 不跨 0，或至少结果明确标为“趋势、证据不足”。

### 不支持时如何改写结论

- 若 M500 只是 First-off-oracle rate 更低，而 Natural/Forced recovery 不高：收益主要来自更好的早期位置/token completion，不是后续纠错。
- 若 M500 的 recovery 也不高且强制错误后同样崩溃：原始 Motivation 被否定；保留“效率和局部 tokenization 平衡”作为 M500 的理由。
- 若自然相关性支持而因果干预不支持：不做机制性宣称，仅报告观察到的相关性。

### 停止规则

本轮不做新的训练、不扫 M500 checkpoint、不引入新 sampler。P0/P1/P2 已完成并停止；由于 P1/P2 没有同时支持 H2，不进入专门强化纠错训练的分支。

### 本轮最终判定（顺序无关版本）

P1 没有显示 M500 的完整改善比例高于 Atom；它只显示 M500 的明确有害首事件更少，
同时距离不变事件更多。P2 中两模型的 matched-control 逐路径最终命中率分别为 `53.57%/52.52%`，
有害干预后降为 `0.74%/2.13%`；M500 的后续 SUB/DEL 虽从 `0.934` 增至 `1.010`，
后续距离下降却从 `2.154` 降至 `1.775`。因此原始 H2（SPE-M500 更擅长首错后的
SUB/DEL 纠错）不成立；后续改进以减少有害首事件、提升候选覆盖和采样效率为主。

## 10. 文件组织与最终交付

```text
revision/
├── plan.md                       # 本计划
├── results/                      # JSON/JSONL、每 seed 原始汇总
├── logs/                         # 命令与标准输出
├── figures/trajectory_html/      # P3 HTML
├── report.md                     # 结果、表格、解释与最终结论
└── commands.md                   # 实际冻结运行命令、checkpoint SHA256、数据 SHA256
```

最终 `report.md` 必须明确写出：是否支持原始 Motivation、证据强度、所有负结果、与当前 SPE-M500 baseline 的关系，以及是否值得进入“纠错训练”改进分支。

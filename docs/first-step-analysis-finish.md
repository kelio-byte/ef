# First-Step Analysis 实验结果总结

## 1. 实验背景与目标

本项目使用 Edit Flows 进行化学逆合成（product → reactant）。Oracle 实验已证明方法本身可行（Top-1 ~93%），但模型生成效果远低于 oracle（Top-1 ~45-47%）。本轮分析的目标是诊断：**模型的"第一步"编辑是否准确，以及第一步对最终结果的决定程度**。

"第一步"拆成两个独立的实验：

- **实验 1（静态初始预测）**：固定输入 `x_t = x_0`，不执行采样，只看模型 forward 输出是否能命中 oracle 的应编辑位置/类型/token
- **实验 2（第一次真实编辑）**：运行 Euler 采样，记录轨迹中第一次实际发生的 edit event，分析其与最终结果的关系，并通过干预实验验证因果关系

---

## 2. 实验设置

### 2.1 模型与数据

| 项目 | 内容 |
|------|------|
| 模型 | `checkpoints/USPTO_50K_PtoR_aug20_#global#/2026-06-08_17-20-39/checkpoint_step1680000.pt` |
| 训练配置 | `use_rate_reparam: true`, `scheduler: cubic`, 10 层 Transformer, hidden_dim=256 |
| 数据来源 | `USPTO_50K_PtoR_aug20_#global#/test`，dedup20 后随机种子 42 抽取 1000 条 |
| 数据路径 | `analysis_subsets/USPTO_50K_PtoR_aug20_#global#/test_dedup_seed42_1000/` |
| 采样配置 | `scheduler: linear`, `n_steps: 100`, `event_prob_mode: poisson` |
| GPU | NVIDIA A100-SXM4-40GB (GPU 0) |

### 2.2 使用的脚本

| 实验 | 脚本 |
|------|------|
| 实验 1 | `scripts/first_step_forward_analysis.py` |
| 实验 2 相关性 | `scripts/first_event_impact_analysis.py --mode correlation` |
| 实验 2 干预 | `scripts/first_event_impact_analysis.py --mode intervention` |

---

## 3. 实验 1：静态初始预测分析

### 3.1 实验设计

- 固定 `x_t = x_0`（产物），不执行任何采样
- 在 7 个时间点 (`t = 0, 1e-3, 1e-2, 5e-2, 0.1, 0.2, 0.3`) 上运行模型 forward
- 同时输出 **base**（模型原始输出）和 **effective**（经 `apply_rate_parameterization` 后的实际采样速率）两套分数
- 与 `compute_oracle_model_output(x_t, x_1, t)` 的结果对比

### 3.2 指标定义

#### A. 位置级指标

将每个位置的模型总编辑倾向定义为 `score(j) = λ_ins(j) + λ_sub(j) + λ_del(j)`。oracle 正位置定义为该位置在 X-space 聚合后存在非零编辑需求。

| 指标 | 含义 |
|------|------|
| **Center Hit@1** | 模型最高编辑倾向的位置是否命中 oracle 正位置 |
| **Center Hit@3** | 模型 top-3 编辑倾向位置是否命中 oracle 正位置 |
| **Center Hit@5** | 模型 top-5 编辑倾向位置是否命中 oracle 正位置 |
| **Center MRR** | 第一个 oracle 正位置在模型排序中的倒数排名（MRR） |
| **Position AP** | 将"该位置是否需要编辑"视为多标签检索，计算 Average Precision |

注意：这里的 "center" 不是严格的化学图反应中心原子，而是在 Edit Flows token 空间下与代码一致的"应编辑位置"定义。所有位置指标默认排除 BOS。

#### B. 编辑类型指标

在模型 top-1 位置（anchor position）上，比较模型预测类型与 oracle 类型：

| 指标 | 含义 |
|------|------|
| **Type Acc@oracle-pos** | anchor 位置上的 argmax 编辑类型（ins/sub/del）是否正确 |

#### C. Token 级指标

仅对 oracle 标记为 insert 或 substitute 的 anchor 位置统计：

| 指标 | 含义 |
|------|------|
| **Ins Token Acc@1** | insert token 的 top-1 准确率 |
| **Ins Token Acc@5** | insert token 的 top-5 准确率 |
| **Sub Token Acc@1** | substitute token 的 top-1 准确率 |
| **Sub Token Acc@5** | substitute token 的 top-5 准确率 |

#### D. 完整首编辑指标

| 指标 | 含义 |
|------|------|
| **Full First-Edit Acc** | 位置正确 + 类型正确 + token 正确（del 类型只需类型正确） |

### 3.3 结果

1000 条样本，batch_size=64，各时间点结果如下：

| 指标 | t=0 | t=0.001 | t=0.01 | t=0.05 | t=0.1 | t=0.2 | t=0.3 |
|------|-----|---------|--------|--------|-------|-------|-------|
| Center Hit@1 | 0.782 | 0.780 | 0.789 | 0.793 | **0.797** | 0.785 | 0.776 |
| Center Hit@3 | 0.926 | 0.924 | 0.922 | 0.920 | 0.917 | 0.913 | 0.911 |
| Center Hit@5 | 0.962 | 0.963 | 0.964 | 0.966 | **0.967** | 0.965 | 0.960 |
| Center MRR | 0.859 | 0.859 | 0.863 | 0.865 | **0.866** | 0.859 | 0.851 |
| Position AP | 0.845 | 0.845 | 0.848 | 0.848 | **0.850** | 0.843 | 0.836 |
| Type Acc@oracle-pos | 0.774 | 0.772 | 0.780 | 0.784 | **0.788** | 0.775 | 0.765 |
| Ins Token Acc@1 | 0.460 | 0.464 | 0.471 | **0.478** | 0.475 | 0.463 | 0.459 |
| Ins Token Acc@5 | 0.992 | 0.992 | 0.992 | 0.991 | 0.987 | 0.984 | 0.984 |
| Sub Token Acc@1 | 1.000 | 1.000 | 1.000 | 0.909 | 0.889 | 1.000 | 1.000 |
| **Full First-Edit Acc** | 0.366 | 0.368 | 0.378 | **0.385** | 0.384 | 0.370 | 0.363 |

**关于 base vs effective：** 当前模型使用 `use_rate_reparam: true`，`apply_rate_parameterization()` 对速率做全局缩放 `v = k(t) * v'`，不改变位置间或类型间的相对排序。因此 base 和 effective 在所有指标上完全一致。

### 3.4 分析

1. **位置感知能力强**：Center Hit@1 ~78-80%，Center Hit@5 ~96%。模型在 `x_t = x_0` 时能较准确地定位应编辑位置。

2. **时间稳定性好**：指标从 t=0 到 t=0.3 变化很小，说明模型对时间输入的敏感度适中。指标在 t≈0.05-0.1 处略有峰值，与 `#global#` 平均编辑距离约 5（首个关键编辑可能在 t≈0.2 附近）的估计趋势一致。

3. **Token 预测是主要瓶颈**：Ins Token Acc@1 仅 ~46-48%，远低于位置命中率。这意味着即使模型知道"在哪里编辑"，也往往不知道"插入什么"。

4. **Full First-Edit Acc 仅 ~37-39%**：综合考虑位置+类型+token 后，完整首编辑正确率较低。主要拖累因素是 insert token 准确率。

5. **Sub Token 样本量小**：Sub Token Acc@1 波动较大（0.889-1.000），因为数据中 substitute 类型样本极少。

### 3.5 Token 分布 KL 散度分析

除了 Top-1/5 Acc，还引入了 KL 散度来直接衡量模型预测的 token 分布与 oracle 分布之间的差异。

#### 动机

Top-1 Acc 只看 argmax，Top-5 Acc 只看覆盖。但之前 Oracle Loss 分析的核心诊断是"总量级对了但分布不够锐"——KL 散度可以直接量化模型分布到底有多"不锐"。如果模型在正确 token 上分配的概率不够集中（即使正确 token 在 top-5 内），KL 值就会偏高。

#### 计算方式

两个口径：

| 口径 | 计算位置 | 含义 |
|------|---------|------|
| **Ins KL@anchor** | anchor 位置，模型预测 ins 且 oracle 确认 ins | 与 Ins Token Acc 口径一致，"模型认为该 insert 时分布准不准" |
| **Ins KL@oracle-pos** | 所有 oracle 标记为 insert 的位置 | 覆盖面更广，"所有该 insert 的位置分布准不准" |

计算方式：`KL(oracle || model) = Σ_i P_oracle(i) * (log P_oracle(i) - log P_model(i))`，通过 `F.kl_div(model_log_probs, oracle_log_probs, log_target=True)` 实现。KL 值越低越好，0 表示完美匹配。Oracle 分布在正确 token 上接近 one-hot，因此 KL ≈ -log P_model(correct_token)。

**对照基线**：若模型输出均匀分布，`KL(oracle || uniform) ≈ log(V) ≈ 4.6 nat`（词表约 100 token）。

Sub Token 样本量极小（部分 t 仅 1-2 个样本），以下仅报告 Ins 结果。

#### 结果

| 指标 | t=0 | t=0.001 | t=0.01 | t=0.05 | t=0.1 | t=0.2 | t=0.3 |
|------|-----|---------|--------|--------|-------|-------|-------|
| Ins KL@anchor | 0.316 | 0.315 | 0.315 | **0.310** | 0.312 | 0.348 | 0.379 |
| Ins KL@oracle-pos | 0.402 | 0.402 | 0.404 | 0.412 | 0.424 | 0.455 | 0.498 |
| Ins Token Acc@1 | 0.460 | 0.464 | 0.471 | **0.478** | 0.475 | 0.463 | 0.459 |
| Ins Token Acc@5 | 0.992 | 0.992 | 0.992 | 0.991 | 0.987 | 0.984 | 0.984 |

#### 分析

1. **KL 远低于随机基线（0.3 vs 4.6 nat），模型确实学到了有意义的分布**。KL 0.31 nat 对应模型在正确 token 上概率约 exp(-0.31) ≈ 73%。不是随机猜，但离 oracle 的 ~100% 还有明显差距。

2. **KL 量化了"分布不锐"的程度**：Ins Token Acc@5 ≈ 99%（正确 token 几乎总在 top-5），但 Acc@1 仅 ~46%，KL ~0.3 nat。三者一致表明：模型把概率分散在了多个候选 token 上，没有足够自信地指向正确 token。

3. **KL@oracle-pos > KL@anchor（0.40 vs 0.32）**：anchor 位置是模型"最有把握"的子集（模型已正确识别为 insert），全覆盖口径下分布更分散。这说明模型只在它"确认"的位置上分布较锐，在其他 oracle-insert 位置上更不确定。

4. **随时间 t 退化**：Ins KL@anchor 从 t=0 的 0.316 升至 t=0.3 的 0.379，KL@oracle-pos 从 0.402 升至 0.498。更大 t 对应更大 k(t) 放大系数，模型分布质量下降——与 k(t) 放大后模型困惑度增加的假设一致。

5. **与现有诊断一致**：之前结论"总量级对了但分布不够锐"在 token 层面被 KL 散度定量证实——模型分布不是均匀噪声（那样 KL 会是 4.6 nat），而是"大致对但不够集中"（KL ~0.3-0.5 nat）。

#### 输出文件

新增 KL 指标的结果保存在：

```text
analysis_outputs/first_step/test_dedup_seed42_1000/forward_kl/summary.json
```

---

## 4. 实验 2：第一次真实编辑影响分析

### 4.1 实验设计

运行完整 Euler 采样（n_steps=100），关注轨迹中第一次实际发生的 edit event。

#### "第一次真实编辑"的定义

在每轮 Euler step 中，通过 `_sample_edit_actions()` 采样 `ins_mask / del_mask / sub_mask`。第一次真实编辑定义为轨迹中最早出现的、包含任意编辑事件的 step。该 step 内的全部编辑集合称为 **first event set**。

为便于分析和可视化，额外定义 **anchor edit**：first event set 中第一个位置的编辑。

#### 第一次事件记录内容

每个 sample 的 `first_event` 保存：

| 字段 | 含义 |
|------|------|
| `first_event_step_idx` | 第一次事件所在的 step 近似索引 |
| `first_event_t` | 第一次事件发生前的时间 t |
| `n_first_events` | 该 step 内同时发生的事件数 |
| `event_positions` | 该 step 内发生编辑的位置集合 |
| `anchor_pos` / `anchor_type` / `anchor_token` | anchor 编辑的位置/类型/token |
| `center_hit` | anchor 位置是否命中 oracle 正位置 |
| `type_correct` | anchor 类型是否正确 |
| `token_correct` | anchor token 是否正确 |
| `event_set_correct` | 当前事件集合是否等于 oracle 正位置集合（严格口径） |

#### Oracle 对照

对每个 step 开始时的当前状态 `x_t`，调用 `compute_oracle_model_output(x_t, x_1, t)` 得到该状态下理论最优编辑。

### 4.2 相关性分析（不干预采样）

#### 设置

- `n_samples=10`（每条样本重复采样 10 次）
- 不改任何采样逻辑，只记录第一次事件
- 总计 10,000 条轨迹

#### 结果

| 指标 | 数值 |
|------|------|
| 总轨迹数 | 10,000 |
| Top-1 Exact Match | 43.86% |
| First Event Set 正确数 | 2,985 (29.85%) |
| First Event Set 错误数 | 7,015 (70.15%) |
| **P(final correct \| first event set correct)** | **63.48%** |
| **P(final correct \| first event set wrong)** | **35.51%** |

#### 分析

1. 第一次事件正确的概率仅 ~30%，与实验 1 中 Full First-Edit Acc ~37% 的趋势一致（真实采样因随机性和多事件并发，正确率更低）。

2. 当第一次事件正确时，最终成功率提升至 **63.5%**（约为整体 43.9% 的 1.45 倍）。

3. 当第一次事件错误时，最终成功率降至 **35.5%**。条件概率比 P(correct\|first correct) / P(correct\|first wrong) ≈ **1.79×**，说明第一步正确性对最终结果有显著影响。

4. 但即使第一步正确，仍有 ~36.5% 的失败率——说明后续步骤的速率建模也需改进。

### 4.3 干预分析（因果验证）

#### 设置

三种模式各运行 10,000 条轨迹：

| 模式 | 含义 |
|------|------|
| `normal` | 不干预，完全按模型采样 |
| `force_correct_first` | 第一次事件步，将该样本的编辑替换为 oracle 正确 anchor edit，后续恢复模型采样 |
| `force_wrong_first` | 第一次事件步，将该样本的编辑替换为模型高分但 oracle 错误的 anchor edit，后续恢复模型采样 |

注意：干预采用的是"首个 anchor edit"口径，不是强制执行 oracle 的完整 event set。`force_wrong_first` 选取模型排序最高但不在 oracle 正位置中的位置，取该位置的 top-1 错误类型和 token，更贴近真实失败模式。

#### 结果

| 模式 | Top-1 Acc | First Event Correct 数 | Δ vs normal |
|------|-----------|----------------------|-------------|
| normal | 43.43% | 3,008 / 10,000 | — |
| **force_correct_first** | **60.82%** | 3,891 / 10,000 | **+17.39%** |
| force_wrong_first | 0.88% | 0 / 10,000 | −42.55% |

#### 详细条件概率

| 模式 | P(correct \| first correct) | P(correct \| first wrong) |
|------|---------------------------|--------------------------|
| normal | 61.60% | 35.61% |
| force_correct_first | 69.90% | 55.03% |
| force_wrong_first | N/A | 0.88% |

#### 分析

1. **`force_correct_first` 带来 +17.4pp 的大幅提升**：Top-1 从 43.4% 跃升至 60.8%。这是强因果证据——**第一步编辑是当前模型的主要瓶颈之一**。

2. **`force_wrong_first` 几乎完全破坏结果**：Top-1 降至 0.88%，说明模型对开局错误**几乎没有纠错能力**。一旦第一步走错，整条轨迹几乎不可能恢复。

3. **修正第一步后仍远低于 oracle (~93%)**：60.8% 距离 93% 仍有 ~32pp 差距。这说明：
   - 第一步是重要瓶颈，但不是唯一瓶颈
   - 后续步的速率预测也存在明显问题（否则 force_correct_first 应接近 oracle）

4. `force_correct_first` 模式下，first event set correct 也仅从 30.1% 升至 38.9%，因为干预只替换了 anchor edit 而非整个 event set。这也解释了为什么该模式下 P(final correct) 最高只有 69.9% 而非更高。

---

## 5. 跨实验综合分析

### 5.1 三步诊断框架

| 层次 | 实验 | 核心发现 |
|------|------|----------|
| 模型认知 | 实验 1（静态） | 位置感知强 (~79%)，token 预测弱 (~47%)，KL ~0.3 nat 量化"分布不锐" |
| 采样行为 | 实验 2 相关性 | 首次事件正确率 ~30%，正确则成功率 1.79× |
| 因果验证 | 实验 2 干预 | 修正首步 +17pp，破坏首步 → 0.9% |

### 5.2 核心结论

1. **模型具有初始反应中心感知能力**：在固定 `x_t = x_0` 时，能以 ~79% 的准确率找到应编辑位置。这说明训练让模型学到了"产物中哪里需要变化"。

2. **Token 预测是当前最突出的短板**：Insert Token Acc@1 仅 ~46%，远低于位置命中率。KL 散度 ~0.31 nat（对比随机基线 ~4.6 nat）表明模型学到了有意义的分布，但概率质量分散在多个候选 token 上（正确 token 概率仅 ~73% vs oracle ~100%）。这正是"总量级对了但分布不够锐"的具体表现。

3. **第一步正确性对最终结果有决定性影响**：
   - 第一步正确 → 成功率 63.5%
   - 第一步错误 → 成功率 35.5%
   - 强制修正第一步 → 成功率 60.8% (+17.4pp)
   - 强制破坏第一步 → 成功率 0.9%

4. **模型缺乏自我纠错能力**：`force_wrong_first` 降至 0.9% 表明，一旦开局出错，后续采样几乎无法补救。Euler 采样的离散化和 clamp 机制也无法纠正这个偏差。

5. **第一步不是唯一瓶颈**：修正第一步后 Top-1 仅到 60.8%（oracle 93%），说明中后期步骤的速率建模同样不足。

### 5.3 与现有结论的一致性

这些结果与之前 `oracle-analysis.md` 和 `loss-analysis.md` 的诊断高度一致：

- Oracle 证明方法可行 → 模型速率学习是主瓶颈
- Oracle loss 分析显示"总量对、分布不锐" → 实验中表现为位置找对但 token 不对，KL ~0.3 nat 定量确认分布不锐程度
- 当前主配置 `use_rate_reparam: true` 下，base 与 effective 分数完全一致（全局缩放不改相对排序），说明速率重参数化对静态预测阶段的模型偏好无影响
- KL 随时间退化（t=0: 0.316 → t=0.3: 0.379），与 k(t) 放大后模型困惑度增加的假设一致

---

## 6. 后续建议

### 6.1 优先级排序

基于本轮结果，建议按以下优先级推进：

1. **改进 token 级预测**（最直接）
   - 实验 1 中 Ins Token Acc@1 仅 ~46%，KL ~0.31 nat 表明概率质量分散
   - 可能的改进方向：调整 ins/sub head 的参数化、增强 position-wise token distribution 的锐度、增加 token 级监督信号（如直接优化 KL 或加入温度调节）

2. **改善中后期步骤的速率建模**
   - `force_correct_first` 只能到 60.8%，距离 oracle 93% 仍有差距
   - 建议做前 N 步 oracle 替换曲线（`oracle 前 1/2/5 步 + model 后续`），定位"需要多少步 oracle 引导才能接近 oracle 上限"

3. **做 `use_rate_reparam: false` 对照实验**
   - 当前所有实验基于 `use_rate_reparam: true` 模型
   - 对照模型可验证速率重参数化是否改善了初始反应中心感知

4. **扩展到完整 test 集和 standard 数据集**
   - 当前 1000 条子集结论需在更大规模上验证稳定性

### 6.2 已完成的输出文件

| 实验 | 输出目录 |
|------|----------|
| 实验 1 | `analysis_outputs/first_step/test_dedup_seed42_1000/forward/` |
| 实验 1 (KL) | `analysis_outputs/first_step/test_dedup_seed42_1000/forward_kl/` |
| 实验 2 相关性 | `analysis_outputs/first_step/test_dedup_seed42_1000/event_correlation/` |
| 实验 2 干预 | `analysis_outputs/first_step/test_dedup_seed42_1000/event_intervention/` |

每个目录包含 `summary.json`（聚合指标）、`per_example.pt` / `per_sample_events.pt`（逐样本明细）和 `report.md`（简要摘要）。

---

## 7. 一句话总结

模型已经学会了"在哪里编辑"（位置命中 ~79%），但还没学会"编辑成什么"（Token Acc ~46%，KL ~0.3 nat 表明概率分散而非随机）；第一步走对能让成功率翻近一倍（63.5% vs 35.5%），强制修正第一步能将 Top-1 从 43% 提升至 61%，但即使第一步完全正确，模型仍远未达到 oracle 上限——中后期的速率预测同样需要改进。

---

## 8. 参考：Oracle 正位置分布

实验 1 的指标依赖 oracle 正位置标签。以下是对 `test_dedup_seed42_1000` 数据集上 oracle 正位置数量的统计（计算方式：固定 `x_t = x_0`，`t = 0`，调用 `compute_oracle_model_output`，取总速率 > 1e-6 的位置）。

### 8.1 汇总统计

| 指标 | 数值 |
|------|------|
| 样本数 | 1000 |
| Mean | 2.17 |
| Median | 2.0 |
| Std | 2.22 |
| Min / Max | 1 / 35 |
| Q25 / Q75 | 1.0 / 2.0 |

### 8.2 分布

| 正位置数 | 样本数 | 占比 |
|----------|--------|------|
| 1 | 409 | 40.9% |
| 2 | 357 | 35.7% |
| 3 | 150 | 15.0% |
| 4 | 45 | 4.5% |
| 5 | 6 | 0.6% |
| 6–10 | 15 | 1.5% |
| 11–20 | 16 | 1.6% |

### 8.3 对实验 1 指标的影响

- 每个样本平均约 46 个 token，正位置仅占约 **4.7%**。随机猜中概率约 2.2%（1/46）。
- **76.6%** 的样本只有 1–2 个正位置。这意味着 Center Hit@1 ~79% 是在大多数情况下"从 ~46 个候选中精准挑对 1 个"的结果，信号很强。
- 长尾中约 3% 的样本（6+ 正位置）对应编辑距离较大的困难样本，可能是 Position AP 被拉低的主要来源。

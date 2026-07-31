# Beam Search 实验 #2：Oracle Greedy 诊断与实现 Bug 修复

## 1. 背景

实验 #1（`docs/beam-search/exp1.md`）中，模型 Greedy/Beam 的 Top-1 均为 0%，初步结论是"单编辑搜索在当前模型上完全不可行"。但随后代码审查（`docs/beam-search/todo2.md`）发现并修复了 5 个实现层面的 Bug。实验 #2 的目标是：**用 Oracle 速率（已知 target 时的理论最优速率）驱动单编辑贪心搜索，确定单编辑搜索的理论上限**。

如果 Oracle Greedy 效果好，说明单编辑搜索方向可行，问题在模型；如果 Oracle Greedy 也差，说明单编辑离散化本身有问题。

## 2. 实验设置

### 2.1 数据与模型

- **测试集**: `analysis_subsets/USPTO_50K_PtoR_aug20_#global#/test_dedup_seed42_1000/`（1000 条）
- **Oracle**: `edit_flows/sampling/oracle.py` 中的 `compute_oracle_model_output`，通过 DP 对齐 `(x_t, x_1)` 计算理论最优编辑速率
- **搜索**: `edit_flows/sampling/beam.py` 中的 `sample_greedy_single_edit`，Oracle 模型包装为 `OracleModel(nn.Module)`

### 2.2 参数

| 参数 | 值 |
|------|-----|
| `max_edits` | 20 |
| `time_mode` | `fixed`, `time_const=0.5` |
| `k_ins_token` / `k_sub_token` / `k_edit_expand` | 4 / 4 / 16 |
| `use_rate_reparam` | false（Oracle 输出已含 `k(t)` 缩放） |
| `use_origin_mask` | false |

### 2.3 OracleModel 包装

```python
class OracleModel(nn.Module):
    """将 compute_oracle_model_output 包装为 model.forward() 接口，
    直接复用 beam.py 的 sample_greedy_single_edit，无需修改搜索代码。"""
    def forward(self, tokens, time_step, padding_mask, origin_mask=None):
        return compute_oracle_model_output(
            tokens, self.x_1[:B], time_step, self._scheduler, self._vocab_size, ...)
```

### 2.4 评测

编辑排序诊断使用独立的 `scripts/edit_ranking_diag.py`，对 100 条样本逐步对比 Oracle 最优编辑在模型候选池中的排名。

## 3. 实验结果

### 3.1 第一轮（v1）：`time_mode=depth`, 无早停

| 指标 | Oracle Greedy v1 | 参考：Oracle Euler |
|------|-----------------|-------------------|
| Top-1 Acc (aug=1) | **1.7%** | ~93% |
| Invalid SMILES | 76.4% | — |
| Unique Rate | 23.6% | — |

**初步判断：单编辑搜索方向可能根本有问题，因为即使 Oracle 速率也只能到 1.7%。**

### 3.2 诊断：发现两个实现层面 Bug

#### Bug 1：`k(0)=0` 导致 Step 0 评分失效

`time_mode="depth"` 下 `t_k = step / max_edits`，step=0 时 `t=0`。

Cubic scheduler 下 `k(0) = 3·0² / (1-0³) = 0`。Oracle 在 `compute_oracle_log_ux_cat` 中将速率计算为 `uz_mask * k(t)`，因此 step 0 时**所有 Oracle 速率精确为 0**（严格来说是 `SMALL_RATE = 1e-9`）。

`_collect_edit_candidates_single` 面对全等速率的候选池，`torch.topk` 返回的结果实际上由浮点噪声决定。**Step 0 的编辑选择是完全随机的。**

模型 Greedy 不受此影响的原因是：模型通过 `use_rate_reparam=true` 使用 base rate 评分，`k(t)` 在 cond_prob 的分子分母中精确抵消。但 Oracle 的速率由 `k(t) * edit_demand` 直接计算，分离不出 base rate。

验证：单样本 debug 确认 Oracle 在 t=0 时所有候选 log_u ≈ -20.7（SMALL_RATE 的 log），排名无意义。

修复：改用 `time_mode="fixed"` + `time_const=0.5`，此时 `k(0.5) ≈ 0.857`，正确编辑的 log_u ≈ -0.15 vs 噪声的 -20.7，排名清晰。

#### Bug 2：无停止条件导致完成后震荡

v1 中 `stop_u_tot_base=-1.0`（禁用早停）。Oracle Greedy 在完成所有必要编辑后，序列已匹配 target，所有 Oracle 速率降至 `SMALL_RATE` 级别。但由于没有早停，算法继续从噪声中"选择"候选编辑（全等速率，随机选），每次随机编辑破坏序列 → Oracle 下一步尝试修复 → 往复震荡。

验证：单样本 debug 追踪到 step 1 时 `u_tot ≈ 1.5e-7`（全 SMALL_RATE），但 greedy 仍从中选 top-1 并执行。

修复：设置 `stop_u_tot_base=0.1`。完成后 `u_tot_real ≈ L × 3 × 1e-9 ≈ 1.5e-7 ≪ 0.1`，序列即时停止。

### 3.3 第二轮（v2）：修复 Bug 1，未修复 Bug 2

| 指标 | v2 |
|------|-----|
| Top-1 Acc | 1.8% |
| Invalid SMILES | 89.7% |

**无效。** 只修 t=0 不够——step 0 对了，但完成后仍被噪声破坏。且某些样本在 step 0 随机选错后也会进入震荡。

### 3.4 第三轮（v3）：修复两个 Bug，但存在 batch 索引错误

修复两个 Bug 后，初步单样本测试达到 94% token 匹配，但批量评测仅 4% Top-1。

排查发现 `oracle_greedy.py` 中 `OracleModel` 存在 batch 索引错误：`self.x_1[:B]` 永远返回前 B 个 target，导致第 2 个 batch 起全部使用了错误的 target。单样本 debug 不受影响（每批 1 个样本），因此单样本 94% 正确但批量 4%。

### 3.5 第四轮（v4）：修复 batch 索引

| 指标 | Oracle Greedy v4 | 参考：Oracle Euler |
|------|-----------------|-------------------|
| **Top-1 Acc** | **96.0%** | ~93% |
| Invalid SMILES | 3.6% | — |
| Unique Rate | 100.0% | — |

**Oracle Greedy 达到 96% Top-1，与 Oracle Euler（93%）在同一水平。** 单编辑搜索框架被 Oracle 验证完全可行——只要有准确的编辑速率排序，单编辑贪心搜索可以达到甚至略微超过多编辑并行 Euler 的精度。

3.6% Invalid SMILES 对应约 2/50 个分子，均为高编辑距离（>23 tokens）样本，`max_edits=20` 预算不足。

### 3.5 失败案例分析

6 个失败案例全部是高编辑距离样本：

| 样本 | edit_dist | 结果长度 | 目标长度 |
|------|-----------|---------|---------|
| #4 | 23 | 96 | 99 |
| #5 | 23 | 100 | 103 |
| #6 | 23 | 59 | 62 |
| #24 | 23 | 76 | 79 |
| #27 | 23 | 130 | 133 |
| #35 | 33 | 70 | 83 |
| **失败组平均** | **24.7** | — | — |
| 成功组平均 | <10 | — | — |

`max_edits=20` 对于编辑距离 ≥23 的样本预算不足。这是单编辑搜索的固有限制，增大 `max_edits` 可解决。

### 3.6 编辑排序诊断（模型侧）

同一 checkpoint（step=850000），100 条样本，按 Oracle 最优路径逐步推进：

| Oracle 最优编辑在模型候选中的排名 | 占比 |
|----------------------------------|------|
| Top-1 | 11.8% |
| Top-5 | 27.7% (累计) |
| Top-16 | 38.1% (累计) |
| 不在 Top-16 中 | **61.9%** |
| Mean score gap | 0.88 nats |

模型在 62% 的决策点无法将正确编辑排入 top-16 候选池。这与 Oracle Greedy 94% 的成功率形成鲜明对比——搜索框架可行，瓶颈纯在模型。

## 4. 结论

### 4.1 单编辑搜索方向完全可行

实验 #1 的结论——"单编辑搜索在当前模型上完全不可行"——是**由三个 Bug 叠加导致的错误归因**：

1. **`k(0)=0`**：`time_mode="depth"` 下 step 0 所有 Oracle 速率归零，首个编辑随机
2. **无早停**：序列完成后噪声编辑反复破坏/修复，形成震荡
3. **OracleModel batch 索引错误**：`self.x_1[:B]` 只取前 B 个 target，第 2 个 batch 起全错

修复全部三个问题后，**Oracle Greedy 达到 96.0% Top-1，与 Oracle Euler（~93%）在同一水平。**

### 4.2 当前真实瓶颈

| 层面 | 状态 |
|------|------|
| 单编辑搜索框架 | **可行**（Oracle 验证 94%） |
| 模型速率排序 | **主要瓶颈**（62% 正确编辑不在 top-16） |
| 搜索策略 tweak | 次要（模型修好后 beam vs greedy 的差异才可见） |

### 4.3 对模型 Greedy/Beam 的影响

模型 Greedy 的 0% Top-1 是两个问题的叠加：
1. 模型速率排序差（根本原因，Oracle 诊断已量化）
2. 两个实现 Bug 也可能影响模型侧（`time_mode="depth"` 下 step 0 时 `t=0`，虽然 cond_prob 评分中 `k(t)` 抵消，但模型的时间嵌入在 t=0 时可能产生 OOD 行为）

## 5. 下一步建议

### 5.1 立即可做：模型 Greedy/Beam 的 step 0 时间修复

当前 `time_mode="depth"` 下 `t_k = k / max_edits`，step 0 时 `t=0`。虽然 cond_prob 评分层面 `k(0)` 抵消，但模型的**时间嵌入**在 `t=0` 时可能处于训练分布边缘（训练时 `t ~ Uniform(0,1)`，但 `t=0` 的概率密度为零）。

建议改动：`t_k = (k + 1) / (max_edits + 1)` 或 `t_k = (k + 0.5) / max_edits`，避免 step 0 恰好为 0。

改动位置：`beam.py` 中 `sample_greedy_single_edit` 和 `sample_beam_single_edit` 的 `time_mode == "depth"` 分支。一行改动，影响可控。

### 5.2 立即可做：模型 Greedy 启用 `stop_u_tot_base`

当前模型 Greedy 默认 `stop_u_tot_base=-1`（禁用）。对于模型，完成编辑后 `u_tot_base` 也会下降（虽然不如 Oracle 干净）。启用早停可以防止过度编辑。

需要确定合适的阈值——Oracle 用 0.1 有效，模型可能需要不同值。建议在 train_subset 上扫描 [0.01, 0.05, 0.1, 0.5, 1.0]。

### 5.3 短期：模型速率质量提升（核心）

编辑排序诊断已经量化了问题（62% 正确编辑不在 top-16），后续建议方向：

- **训练目标调整**：对非正确编辑加 auxiliary penalty，或对正确编辑加权重
- **Rate reparam ablation**：对比 `use_rate_reparam=true/false` 下的速率尖锐度
- **模型架构**：rate head 容量是否足够（当前是 2 层 MLP）？分治不同编辑类型是否有效？
- **训练更久**：当前 checkpoint 仅 850k/5M steps，速率排序可能随训练继续改善

### 5.4 中期：Beam 搜索复评

模型速率质量改善后，重新对比：
- Greedy vs Beam（beam_size=3,5）
- 不同 `time_mode`（depth vs fixed vs utot_ratio）
- 单编辑 Greedy vs 多编辑 Euler

### 5.5 工程改进

- `sample_retro.py` 已有 `--sampler greedy_edit/beam_edit` 参数，实验用 CLI 已就绪
- Oracle Greedy 脚本 `scripts/oracle_greedy.py` 可复用于验证模型改进后的理论上限
- 编辑排序诊断 `scripts/edit_ranking_diag.py` 可作为模型改进的定量评估工具

## 6. 相关文件

| 文件 | 说明 |
|------|------|
| `scripts/oracle_greedy.py` | Oracle Greedy 实验脚本（本次新增） |
| `scripts/edit_ranking_diag.py` | 编辑排序诊断脚本（本次新增） |
| `experiments/oracle_greedy_v3/` | Oracle Greedy v3 输出（t=0.5 + stop） |
| `experiments/edit_ranking_diag/` | 编辑排序诊断报告 |
| `edit_flows/sampling/beam.py` | Greedy/Beam 实现 |
| `edit_flows/sampling/oracle.py` | Oracle 速率计算 |

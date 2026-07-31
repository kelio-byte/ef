# Beam Search 实验 #3：Edit Ranking 诊断修正与重新评估

## 1. 背景

实验 #2（`docs/beam-search/exp2.md`）的核心结论之一是：

- 模型在 oracle 轨迹上的 edit ranking 极差：Top-1 仅 11.8%，62% 的正确编辑不在模型 Top-16 候选池中

但 `docs/beam-search/todo3.md` 的代码审查发现，该结论依赖的诊断脚本（`scripts/edit_ranking_diag.py`）存在两个会系统性压低结果的 bug：

1. **时间边界点问题**：`time_mode="depth"` 下 step 0 对应 `t=0`，cubic scheduler 下 `k(0)=0` 导致评分失效
2. **Oracle tie 口径过严**：只取 argmax 单个 edit 作为真值，忽略了 oracle 侧可能存在多个同分最优 edit

虽然这两处 bug 已在当前代码中修复，但实验 #3 的目标更根本：**修正一个更严重的实验设计缺陷——oracle 轨迹收敛后噪声步骤污染统计指标**。

## 2. 问题诊断：Oracle 轨迹噪声

### 2.1 现象

使用修正后的脚本（tie-aware + interior time mapping）在 1000 条样本上运行 ranking 诊断，设置 `max_edits=20`。结果发现：

- **所有 1000 条样本都跑满了 20 步**（per-sample step count: 100% at 16+）
- Detail trace 显示大量样本后期进入 BOS 位置的 **ins ↔ del 震荡循环**（如 `(ins,1,))` → `(del,2,<None>)` → 重复）
- 这使得整体统计中 60% 的 oracle steps 落在 step≥8 区间，污染了分步指标

### 2.2 根因

Oracle 在 `compute_oracle_log_ux_cat` 中的速率计算逻辑：

- 所有位置的速率初始化为 `SMALL_RATE = 1e-9`
- 有编辑需求的位置叠加 `k(t) * 1.0`
- 当 `x_t == x_1`（序列已收敛到目标）时，编辑需求为零，**所有速率均为 SMALL_RATE**

此时 `_collect_edit_candidates_single` 仍然会返回候选列表（所有候选的 log_u 近似相等，均为 `LOG_SMALL_RATE ≈ -20.72`）。`torch.topk` 从中"随机"选取 top-1，造成了 BOS 区域的往返震荡。

这本质上是 Oracle Greedy 实验（exp2 3.2 节 Bug 2）中"无早停导致震荡"同一现象的再现——只是发生在 ranking 诊断脚本而非 greedy 采样中。

### 2.3 真编辑与噪声编辑的理论区分

候选编辑的 score = `log_u(e) - log_u_tot`。`k(t)` 在分子分母中精确抵消：

| 场景 | score | 说明 |
|------|-------|------|
| 真编辑 (n_active=1) | ~0.00 | 仅一个位置需要编辑 |
| 真编辑 (n_active=2) | ~-0.69 | 两个位置同时需要编辑 |
| 真编辑 (n_active=5) | ~-1.61 | 五个位置（已属罕见） |
| 真编辑 (n_active=10) | ~-2.30 | 极端情况（<3% 样本） |
| **噪声** (L=50, 收敛后) | **~-5.01** | 所有速率均为 SMALL_RATE |
| **噪声** (L=20, 收敛后) | **~-4.09** | 短序列噪声上界 |

**真编辑 score > -2.3，噪声编辑 score < -4.0，两者之间约 2 nat 的清晰间隔。**

## 3. 实现修改

### 3.1 新增脚本：`scripts/edit_ranking_diag_v2.py`

基于 `scripts/edit_ranking_diag.py` 做了以下改进：

1. **时间映射**：统一使用 `_depth_time_value(step, max_edits) = (step + 1) / (max_edits + 1)`，与 `beam.py` 一致，避免 `t=0` 边界
2. **Tie-aware oracle 匹配**：收集 oracle top score ± 1e-6 范围内的所有 edits 作为真值集合，模型命中其中任意一个即算命中（沿用 v1 修复）
3. **Per-step-bin 分桶统计**：按 `step=0 / 1-3 / 4-7 / >=8` 四档独立统计 ranking 指标（实验 B）
4. **Score 阈值早停（本轮新增）**：在 oracle candidates 收集后，检查 top candidate 的 score，若 `step > 0 and score < -3.0` 则判定为噪声并终止轨迹

### 3.2 关键代码改动

在 `edit_ranking_diag_v2.py` 的主循环中（约第 270 行），在 oracle candidates 收集后、ranking 统计前：

```python
if not oracle_cands:
    break

# Stop when oracle has no meaningful edits left.
# Real edits: score > -2.3. Noise (SMALL_RATE): score < -4.0.
# Threshold -3.0 gives ≥ 1 nat safety margin.
# Exclude step 0: rare extreme samples may have n_active ≥ 20.
if step > 0 and oracle_cands[0].score < -3.0:
    break

n_steps += 1
```

阈值选择依据：

- score = log_u - log_u_tot，k(t) 在分子分母中精确抵消
- 真编辑 score = -log(n_active) ∈ [-0.0, -2.3]，其中 n_active ∈ [1, 10]
- 噪声 score = -log(L*3) ∈ [-4.1, -6.4]，其中 L ∈ [20, 200]
- 阈值 -3.0 在最坏情况下仍有 >1 nat 的安全间隔
- `step > 0` 排除第一步（罕见极端样本可能有 n_active ≥ 20）

### 3.3 文件变更

| 文件 | 变更 |
|------|------|
| `scripts/edit_ranking_diag_v2.py` | 新增，约 330 行 |
| `experiments/edit_ranking_diag_v2/test_dedup_seed42_1000/` | 新输出目录 |
| `experiments/edit_ranking_diag_v2/train_subset_1000/` | 新输出目录 |

## 4. 实验设置

### 4.1 模型与数据

| 项目 | 内容 |
|------|------|
| Checkpoint | `2026-06-08_17-20-39/checkpoint_step1680000.pt` |
| 训练配置 | `use_rate_reparam: true`, `scheduler: cubic`, hidden_dim=256, 10 layers |
| 测试集 | `test_dedup_seed42_1000`（1000 条，来自 `USPTO_50K_PtoR_aug20_#global#/test`） |
| 训练子集 | `train_subsets/USPTO_50K_PtoR_aug20_#global#/test`（取前 1000 条） |
| GPU | NVIDIA A100-SXM4-40GB (GPU 2/3) |

### 4.2 参数

| 参数 | 值 |
|------|-----|
| `max_edits` | 20 |
| `k_ins_token` / `k_sub_token` / `k_edit_expand` | 4 / 4 / 16 |
| `use_rate_reparam` | true（从 checkpoint 读取） |
| `time_input` | t |
| 时间映射 | `(step + 1) / (max_edits + 1)` |
| Score 早停阈值 | -3.0（step > 0） |

### 4.3 执行命令

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 scripts/edit_ranking_diag_v2.py \
  --checkpoint checkpoints/.../checkpoint_step1680000.pt \
  --data_dir analysis_subsets/.../test_dedup_seed42_1000 \
  --n_samples 1000 --max_edits 20 \
  --output_dir experiments/edit_ranking_diag_v2/test_dedup_seed42_1000 \
  --device cuda
```

## 5. 实验结果

### 5.1 噪声过滤效果

Score 阈值早停使 oracle 轨迹长度从强行 20 步大幅缩减：

| 数据集 | 修正前总步数 | 修正后总步数 | 减少 |
|--------|:---:|:---:|:---:|
| Test | 20,000 (20.0/sample) | 5,546 (5.5/sample) | -72.3% |
| Train-subset | 20,000 (20.0/sample) | 7,227 (7.2/sample) | -63.9% |

### 5.2 修正后总体指标

| 指标 | Test | Train-subset |
|------|:---:|:---:|
| **Overall Top-1** | **77.9%** | **83.7%** |
| Overall Top-5 (cum) | 89.7% | 94.1% |
| **Overall Top-16 (cum)** | **96.4%** | **99.4%** |
| Not in top-16 | 3.6% | 0.6% |
| Mean score gap | 0.23 nats | 0.12 nats |

### 5.3 Per-step-bin 分解（修正后）

| Step Bin | Test Top-1 | Test Top-16 | Train Top-1 | Train Top-16 |
|----------|:---:|:---:|:---:|:---:|
| step=0 (n=1000) | 71.1% | 94.1% | 83.4% | 100.0% |
| step=1-3 | 80.0% | 97.9% | 77.3% | 99.7% |
| step=4-7 | 68.5% | 95.0% | 73.8% | 97.7% |
| step>=8 | 87.5% | 96.9% | 96.6% | 100.0% |

### 5.4 修正前后对比

| 指标 | 修正前 (Test) | 修正后 (Test) | 修正前 (Train) | 修正后 (Train) |
|------|:---:|:---:|:---:|:---:|
| Top-1 | 22.7% | **77.9%** | 31.4% | **83.7%** |
| Top-16 (cum) | 38.5% | **96.4%** | 45.8% | **99.4%** |
| Not in top-16 | 61.5% | **3.6%** | 54.2% | **0.6%** |
| Mean score gap | 1.27 | **0.23** | 0.89 | **0.12** |

## 6. 讨论

### 6.1 与 exp2 结论的关系

实验 #2 的"模型排序极差（62% 不在 top-16）"结论被**推翻**。该数字是两个因素叠加的结果：

1. 诊断脚本的 tie 口径和时间边界 bug（已在 #2 和 #3 之间修复）
2. 更根本的：**72% 的 oracle 步骤是收敛后噪声**，污染了整体统计

修正后，模型 edit ranking 实际上**非常强**：Test Top-16 = 96.4%，Train Top-16 = 99.4%。

### 6.2 与 first-step analysis 的一致性

本次结果与 `first-step-analysis2` 修正后的结论高度吻合：

| 来源 | 指标 | Test | Train |
|------|------|:---:|:---:|
| first-step-analysis2 | Full First-Edit Acc (step 0) | 71.7% | 89.0% |
| 本次 (exp3) | Edit ranking Top-1 (step 0) | 71.1% | 83.4% |

两个独立指标在 step 0 上均收敛到 ~71%（Test）/ ~83-89%（Train），增强了结论的可信度。

### 6.3 "轨迹退化"假说被否定

修正前数据显示"step 0 Top-1=71%，step>=8 Top-1=11%"，看似是轨迹退化。修正后 step>=8 的 Top-1 反而是所有 bin 中最高的（87.5% Test / 96.6% Train）。这说明：

- 之前的"退化"完全是噪声污染造成的假象
- 真实规律是：**step 0 是唯一从 `x_0` 出发的步，oracle 候选分布最广，模型排序难度最大；后续步的 oracle 状态更接近 `x_1`，模型更容易识别正确编辑**
- 不存在"越走越差"的轨迹退化

### 6.4 剩余问题

1. **泛化差距**：Test Top-1 77.9% vs Train 83.7%，约 6pp gap。这与 first-step analysis 观察到的位置/类型泛化问题一致

2. **3.6% not-in-top-16 失败案例**：Test 上仍有 199/5,546 个决策点模型无法将正确编辑排入 top-16。值得进一步拆解这些案例（实验 C）

3. **Oracle 轨迹 vs 模型自身轨迹**：本次诊断沿 oracle 轨迹推进，oracle 轨迹上的 ranking 强并不能保证模型自身 greedy 轨迹上的 ranking 也强。Oracle 轨迹的中间状态可能与模型自身采样产生的状态分布不同（OOD 问题），这是实验 E 要回答的核心问题

## 7. 结论

1. **模型 edit ranking 实际上很强**：修正口径后，Overall Top-1 = 77.9% (Test) / 83.7% (Train)，Top-16 = 96.4% / 99.4%

2. **实验 #2 的"模型排序极差"结论是错误的**，其根因是 oracle 轨迹收敛后噪声步骤（约占 72%）污染了统计指标

3. **不存在"轨迹退化"**：所有 step bin 的 Top-16 均稳定在 94-100%

4. **score 阈值（-3.0）是有效的噪声过滤机制**：利用了真编辑与噪声编辑之间 ~2 nat 的天然 score 间隔，无需参数调优

5. **与 first-step analysis 结论完全一致**：模型第一步能力不差（Top-1 ~71-83%），主要 gap 在 train→test 泛化

## 8. 相关文件

| 文件 | 说明 |
|------|------|
| `scripts/edit_ranking_diag_v2.py` | 修正后诊断脚本（tie-aware + step 分解 + score 早停） |
| `experiments/edit_ranking_diag_v2/test_dedup_seed42_1000/` | Test 集输出（ranking_report.txt + summary.json） |
| `experiments/edit_ranking_diag_v2/train_subset_1000/` | Train-subset 输出 |
| `docs/beam-search/todo3.md` | 本轮实验计划 |
| `docs/beam-search/exp2.md` | 上一轮实验（含被推翻的结论） |

---

## 9. 实验 D：小规模 Greedy/Beam 复现

### 9.1 目的

在 exp3 确认模型 edit ranking 实际很强（Top-16=96.4%）之后，验证两件事：

1. `depth` 时间修复 + `stop_u_tot_base` 早停对真实生成是否有实质影响
2. Beam search 能否将 Top-1→Top-16 的 ~18.5pp ranking 差距转化为实际生成精度提升

### 9.2 实验设置

| 项目 | 内容 |
|------|------|
| Checkpoint | `2026-06-08_17-20-39/checkpoint_step1680000.pt`（与 exp3 相同） |
| 数据 | `test_dedup_seed42_1000` 前 200 条 |
| GPU | NVIDIA A100-SXM4-40GB (GPU 2) |
| `max_edits` | 20 |
| `time_mode` | `depth`（使用 `_depth_time_value` interior mapping） |
| `k_ins/k_sub/k_edit_expand` | 4 / 4 / 16 |

**扫描配置**（15 组）：

| 参数 | 取值 |
|------|------|
| Sampler | `greedy_edit`, `beam_edit` (size=3, 5, 8) |
| `stop_u_tot_base` | -1（禁用）, 0.01, 0.05, 0.1, 0.5 |

### 9.3 执行

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python experiments/exp3_beam_d/run_exp_d.py
```

采样耗时：greedy ~10s/config, beam3 ~5-9min/config, beam5 ~7-14min/config, beam8 ~12-19min/config。总计约 90 分钟。

评测使用独立的 `experiments/exp3_beam_d/eval_minimal.py`（因 `score_#global#.py` 在 `beam_size=1` 时存在 IndexError，需绕过）。

### 9.4 结果

#### 9.4.1 完整结果表

| Config | Top-1 | Invalid | Correct (N) |
|--------|:-----:|:-------:|:-----------:|
| greedy_stop-1 | 4.0% | 43.5% | 8 |
| greedy_stop0.01 | 29.0% | 10.5% | 58 |
| greedy_stop0.05 | 32.0% | 9.0% | 64 |
| greedy_stop0.1 | 34.0% | 8.5% | 68 |
| greedy_stop0.5 | **35.5%** | **7.5%** | 71 |
| | | | |
| beam3_stop-1 | 4.5% | 45.5% | 9 |
| beam3_stop0.01 | 31.0% | 11.5% | 62 |
| beam3_stop0.05 | 34.0% | 10.0% | 68 |
| beam3_stop0.1 | 35.5% | 9.5% | 71 |
| | | | |
| beam5_stop-1 | 4.0% | 40.5% | 8 |
| beam5_stop0.01 | 31.5% | 10.0% | 63 |
| beam5_stop0.05 | 34.5% | 8.0% | 69 |
| beam5_stop0.1 | **36.0%** | 8.0% | **72** |
| | | | |
| beam8_stop-1 | 3.0% | 41.5% | 6 |
| beam8_stop0.05 | 34.5% | 7.5% | 69 |

**最佳整体**：beam5_stop0.1（Top-1=36.0%, Invalid=8.0%）
**最佳 Greedy**：greedy_stop0.5（Top-1=35.5%, Invalid=7.5%）
**Beam Δ over Greedy（同 stop=0.1）**：+2.0pp（36.0% vs 34.0%）

#### 9.4.2 早停（stop_u_tot_base）分析

| stop_u_tot_base | Greedy Top-1 | Greedy Invalid |
|:---:|:---:|:---:|
| -1（禁用） | 4.0% | 43.5% |
| 0.01 | 29.0% | 10.5% |
| 0.05 | 32.0% | 9.0% |
| 0.1 | 34.0% | 8.5% |
| 0.5 | 35.5% | 7.5% |

几点观察：

1. **早停不是可选项，是必需品**。不禁用早停时，模型会一直编辑到 `max_edits=20`，即使序列已经完成或已经出错。~43% 的 invalid SMILES 和 3-4% Top-1 说明模型在完成必要编辑后持续进行破坏性编辑。

2. **最轻微的阈值（0.01）已经产生巨大效果**：Invalid 从 43.5% → 10.5%（-33pp），Top-1 从 4.0% → 29.0%（+25pp）。这说明模型的 `u_tot_base` 在完成编辑后确实会下降，0.01 足以滤除大部分"过度编辑"。

3. **阈值越高越好，但边际递减**：0.01→0.1 涨幅 5pp，0.1→0.5 涨幅仅 1.5pp。合理的最优区间在 0.1-0.5。

4. **高阈值不会过度早停**：stop=0.5 的 Top-1 最高（35.5%），Invalid 最低（7.5%）。如果高阈值导致"编辑不够"而提前停止，Top-1 应该下降而不是上升。实际观察到的趋势说明模型更倾向于"过度编辑"而非"编辑不足"。

#### 9.4.3 Beam Search 分析

在匹配的 stop 阈值下，Beam 相对 Greedy 的提升：

| stop | Greedy | Beam-3 | Beam-5 | Δ (Beam5 - Greedy) |
|:---:|:---:|:---:|:---:|:---:|
| 0.01 | 29.0% | 31.0% | 31.5% | +2.5pp |
| 0.05 | 32.0% | 34.0% | 34.5% | +2.5pp |
| 0.1 | 34.0% | 35.5% | 36.0% | +2.0pp |

Beam 收益约 +2pp，稳定但有限。这远低于 exp3 ranking 诊断中 Top-1（77.9%）→ Top-16（96.4%）的 18.5pp 理论差距。

#### 9.4.4 Beam Size Scaling

在 stop=0.05 下观察 beam size 的边际收益：

| Beam Size | Top-1 |
|:---:|:---:|
| 1 (Greedy) | 32.0% |
| 3 | 34.0% (+2.0pp) |
| 5 | 34.5% (+0.5pp) |
| 8 | 34.5% (+0.0pp) |

Beam size 从 3 到 8 几乎没有额外收益。这表明模型候选池中排名 2-8 的编辑在模型自身轨迹上并不比 top-1 显著更好——这与 oracle 轨迹上 Top-1=77.9%、Top-5=89.7% 的清晰分层形成对比。

### 9.5 讨论

#### 9.5.1 为什么 Beam 收益远小于 exp3 ranking 诊断的预期？

exp3 的 ranking 诊断测量的是**沿 oracle 轨迹**的 per-step edit ranking。模型自己的采样一旦在某一步选了非最优编辑，后续状态就偏离了 oracle 轨迹。在这些 OOD 状态上，模型的 edit ranking 很可能显著退化，导致 beam 中保留的"备选路径"也无法有效纠正。

用一个简化模型来理解：

```
exp3 oracle 轨迹:  Top-1=77.9%,  Top-16=96.4%  (每个 oracle 状态)
                                                  ↓
                                           Beam 最多可捕获 18.5pp
                                                  ↓
实际模型轨迹:      第一步正确率 ~72%（first-step analysis 的 Full First-Edit Acc）
                  后续步在 OOD 状态上 ranking 退化
                  ↓
            Beam 实际收益: +2pp
```

核心矛盾是：**模型在 oracle 状态上会排序，但在自己产生的"错误状态"上可能完全不知所措**。

#### 9.5.2 Per-step 78% → Final 35% 的 Gap

如果平均需要 ~5 步编辑，per-step 独立 77.9% 准确率下的期望最终成功率约为 0.779^5 ≈ 29%。加上：
- 模型轨迹偏离 oracle 轨迹后 per-step 准确率 < 78%
- ~8% 的 residual invalid SMILES
- 部分样本编辑步数 >5

最终 35.5% 与这个估算基本一致。这说明 **per-step 排序 78% 本身不足以支撑高最终准确率**，多步误差累积是更根本的限制。

#### 9.5.3 早停机制的本质

`stop_u_tot_base` 阈值利用的是模型的 `u_tot_base`（总基础速率）在完成编辑后会下降的性质。当模型认为"没什么需要改了"时，所有位置的编辑速率都降低，`u_tot_base` 下降。这与 oracle 实验（exp2 3.2 节 Bug 2）中观察到的现象一致：oracle 收敛后 `u_tot ≈ 1.5e-7`，远低于正常编辑时的水平。

但模型的 `u_tot_base` 下降不如 oracle 干净（oracle 收敛后所有速率为 `SMALL_RATE=1e-9`，模型不会这么极端），因此需要扫阈值来找到最优平衡点。

### 9.6 对后续实验的影响

实验结果直接影响了原 `todo3` 中剩余实验的优先级：

1. **实验 E（oracle vs 模型轨迹诊断）优先级大幅提前**。这是解释"78% → 35%"差距最关键的下一步：在模型自身 greedy 轨迹的每个中间状态上，模型的 per-step edit ranking 是否显著低于 oracle 轨迹上的 77.9%？

2. **实验 C（位置/类型/token 分解）暂缓**。per-step 排序本身不差（Top-1=78%, Top-16=96%），拆解"第一步错在哪"不如先搞清楚"为什么多步累积损失这么大"。

3. **stop_u_tot_base 的自动调优**值得考虑。当前需要在验证集上手动扫阈值（0.01-0.5），但 `u_tot_base` 下降点在不同样本上可能不同。一个可能的方向是 per-sample 自适应阈值（如 `stop_u_tot_base_ratio = u_tot_base(t) / u_tot_base(t=0)`）。

### 9.7 新增文件

| 文件 | 说明 |
|------|------|
| `experiments/exp3_beam_d/run_exp_d.py` | 实验 D 主脚本（采样 + 评测编排） |
| `experiments/exp3_beam_d/eval_minimal.py` | 独立评测脚本（绕过 score_#global#.py 的 beam_size=1 bug） |
| `experiments/exp3_beam_d/data/` | 200 条样本子集 |
| `experiments/exp3_beam_d/outputs/` | 15 组配置的采样输出 + eval.log |
| `experiments/exp3_beam_d/summary.txt` | 结果汇总 |

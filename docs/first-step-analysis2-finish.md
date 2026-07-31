# First-Step Analysis 第二轮：指标修正、新数据集与跨数据集对比

## 1. 本轮背景

第一轮 first-step analysis（`docs/first-step-analysis-finish.md`）中，一个关键结论是「模型能找到编辑位置（Center Hit@1 ~79%），但不知道编辑成什么 token（Ins Token Acc@1 ~46%）」。

但是 Ins Token Acc@1 ~46% 与 Ins KL 散度仅 ~0.09 之间存在矛盾——KL 低说明模型分布与 oracle 分布高度一致，Acc 低暗示模型预测错误。此外，对照项目（R-SMILES Transformer，autoregressive seq2seq）在同数据集上 Top-1 可达 ~64.8%，提供了外部基线。

本轮围绕三个方向推进：

1. **构建 baseline-correct 新数据集**：筛选 R-SMILES baseline 正确预测的样本，排除「过于困难」的样本
2. **发现并修复 Token Acc 指标 bug**：oracle token 分布在多 GAP 聚合位置存在 tie，argmax 选取导致大量假阴性
3. **跨三个数据集的系统对比**：训练集子集、测试集随机子集、baseline-correct 子集

---

## 2. 新数据集：baseline-correct 子集

### 2.1 动机

对照项目（R-SMILES Transformer, `/data6/duanbh/desktop/retrosynthesis/`）在同数据集 `USPTO_50K_PtoR_aug20_#global#` 上 Top-1 约 64.8%。过滤出 baseline 能做对的样本，可以：
- 排除极端困难样本的噪声
- 在「baseline 可解」的样本上，更清楚地观察 Edit Flows 与 baseline 的差距本质

### 2.2 构建方法

脚本：`scripts/create_baseline_correct_subset.py`

**数据流**：
1. 取全量 test 集 100140 条输入（5007 products × 20 augmentations），每条作为一个独立数据点
2. 取 baseline 原始预测 `average_model_56-60-results.txt`（100140 × 10 beam = 1,001,400 行），每 10 行取第一个作为 top-1
3. 对预测和 target 分别做 `inverse_global_align()` → RDKit 去 atom map → canonical SMILES 转换
4. Exact match 判定正确，共 60565 条（60.5%）
5. 随机种子 42 抽取 1000 条

**输出**（`analysis_subsets/USPTO_50K_PtoR_aug20_#global#/baseline_correct_seed42_1000/`）：

| 文件 | 内容 |
|------|------|
| `src-test.txt` | product SMILES（tokenized） |
| `tgt-test.txt` | target SMILES（tokenized） |
| `baseline_top1.txt` | baseline top-1 预测（canonical SMILES） |
| `meta.json` | 构建参数与原始索引 |

### 2.3 关键数字

| 指标 | 数值 |
|------|------|
| 总输入 | 100140 |
| 有效预测（parseable SMILES） | 99805（99.7%） |
| Baseline 正确（canonical match） | 60565（60.5%） |
| 从中抽取 | 1000 |

60.5% 的 per-line 正确率与 baseline 的 64.8% per-product 正确率一致（per-line 不做跨 augmentation 聚合，略低）。

---

## 3. Token Acc 指标修复

### 3.1 问题诊断

原 `Ins Token Acc@1` 的计算逻辑：

```python
target_token = int(oracle["ins_token"][i, anchor_pos].item())  # oracle argmax
correct = top5_tokens[0] == target_token
```

其中 `oracle["ins_token"]` 来自 `extract_oracle_event_set()` 的 `log_ins_probs.argmax(dim=-1)`。

**根因**：当多个 Z-space GAP token（需插入不同 token）经 `rm_gap_tokens` 聚合到同一个 X position 时，oracle 在该位置的 insert 分布是**多峰的**（如 token A 概率 0.5、token B 概率 0.5），而非 one-hot。`argmax` 在 tie 时总是选 index 最小的 token，导致模型预测另一个 tied token 被错误判为「错误」。

实验验证（100 样本）：
- 65.6%（59/90）的 anchor 位置存在 top-1 tie
- 其中 69%（41/59）tie 中，模型 top-1 在 tied set 内但被 argmax 判为错误
- 仅 17.8% 的位置 oracle 为 one-hot

### 3.2 修复方案

将严格匹配改为「模型 top-1 是否在 oracle 最大概率 token 集合中」：

```python
ora_probs = torch.exp(oracle_out[1][i, anchor_pos])
ora_max_prob = ora_probs.max()
ora_valid_tokens = set(
    (ora_probs >= ora_max_prob - 1e-6).nonzero(as_tuple=True)[0].tolist()
)
correct = top1_token in ora_valid_tokens
```

Acc@5 也对应修改：检查 top-5 是否与 oracle valid token 集合有交集。

修改位置：`scripts/first_step_forward_analysis.py` 第 233-264 行。

---

## 4. 主结果：三个数据集对比

### 4.1 数据集说明

| 数据集 | 路径 | 来源 | 规模 |
|--------|------|------|------|
| **Train-subset** | `train_subsets/USPTO_50K_PtoR_aug20_#global#/test/` | 训练集 dedup20 后 1000 条 | 1000 |
| **Test-original** | `analysis_subsets/.../test_dedup_seed42_1000/` | 测试集 dedup20 + seed42 随机 1000 条 | 1000 |
| **Test-BC** | `analysis_subsets/.../baseline_correct_seed42_1000/` | 测试集中 baseline top-1 正确，seed42 随机 1000 条 | 1000 |

所有实验使用同一模型 checkpoint：`checkpoint_step1680000.pt`（`use_rate_reparam: true`）。

### 4.2 核心指标（t=0，修正后）

| 指标 | Train-subset | Test-original | Test-BC |
|------|:---:|:---:|:---:|
| Center Hit@1 | **93.1%** | 78.2% | 90.7% |
| Center Hit@3 | 99.0% | 92.6% | 98.1% |
| Center Hit@5 | 99.6% | 96.2% | 99.6% |
| Position AP | 95.2% | 84.5% | 93.6% |
| Type Acc@oracle-pos | **92.7%** | 77.4% | 90.1% |
| Ins Token Acc@1 | 96.1% | 92.5% | **97.2%** |
| Ins Token Acc@5 | 100.0% | 99.2% | 99.9% |
| **Full First-Edit Acc** | **89.0%** | 71.7% | 87.5% |

### 4.3 指标修正前后对比（Test-original，t=0）

| 指标 | 修正前（有 bug） | 修正后（tie-aware） | 变化 |
|------|:---:|:---:|:---:|
| Ins Token Acc@1 | 46.0% | **92.5%** | **+46.5pp** |
| Ins Token Acc@5 | 99.2% | 99.2% | — |
| Full First-Edit Acc | 36.6% | **71.7%** | **+35.1pp** |

### 4.4 指标随 t 变化（t=0.1 峰值）

| 指标 | Train-subset | Test-original | Test-BC |
|------|:---:|:---:|:---:|
| Center Hit@1 | 94.7% | 79.7% | 92.6% |
| Type Acc | 93.9% | 78.8% | 92.3% |
| Ins Token Acc@1 | 95.5% | 93.4% | 96.9% |
| Full First-Edit Acc | 89.7% | 73.6% | 89.5% |

所有指标在 t≈0.05-0.1 处略有峰值，与 `#global#` 平均编辑距离约 5（首个关键编辑在 t≈0.2 附近）的估计趋势一致。

---

## 5. 关键分析与诊断更新

### 5.1 之前的核心诊断已被修正

| 之前结论 | 修正后结论 |
|----------|------------|
| 「模型找不到该编辑成什么 token」 | **Token 预测是模型最强能力之一**（Ins Token Acc@1 92-97%） |
| Full First-Edit Acc ~37% | **Full First-Edit Acc ~72%（Test）~89%（Train）** |
| Token 分布不够锐 | 分布尖锐但之前被 argmax tie 污染的指标掩盖了 |

### 5.2 当前瓶颈重新定位

修正后的指标表明，Edit Flows 第一步的能力瓶颈按优先级排序如下：

1. **位置感知过拟合**（Train 93% → Test 78%，-15pp）：是主要泛化差距来源
2. **编辑类型判断**（Train 93% → Test 77%，-15pp）：同样存在泛化差距
3. **Token 预测轻微过拟合**（Train 96% → Test 93%，-3pp）：泛化最好，几乎不是瓶颈

### 5.3 跨数据集规律

- **Train → Test 泛化差距 17pp**（Full Acc 89.0% → 71.7%）：几乎全部来自位置和类型指标的下降
- **Test-BC ≈ Train**（Full Acc 87.5% vs 89.0%）：baseline 能做对的样本，Edit Flows 的第一步几乎与训练集上一样好
- **Ins Token Acc@1 在三组数据上均 >92%**：进一步确认为最强指标

### 5.4 与之前干预实验的一致性

之前 `force_correct_first` 将 Top-1 从 43% 提升至 61%（+17pp）。现在看到 Full First-Edit Acc 在 Test 上为 72%，这意味着：
- 第一步整体正确率不低（72%），但并非所有「第一步正确的样本」最终都成功
- `force_correct_first` 提升 17pp 与第一步错误率 ~28% 之间存在对应关系
- 中后期步骤的累积错误仍是最终结果从 72%（第一步正确）降至 43%（最终正确）的主要原因

### 5.5 天花板效应

即使在训练集上，Full First-Edit Acc = 89%，未达到 100%。剩余约 11% 的可能来源：
- 多 GAP 聚合导致 oracle 自身多峰（即使修正后仍会有部分位置争议）
- t=0 不一定是所有样本的最佳预测点
- 极少数极端样本

---

## 6. 与对照项目（R-SMILES Transformer）的对比

| 维度 | Edit Flows | R-SMILES Transformer |
|------|-----------|---------------------|
| 范式 | 非自回归编辑生成 | 自回归 seq2seq |
| Test Top-1 正确率 | ~43-47%（最终生成） | ~64.8% |
| 第一步 Full Acc（Test） | 71.7% | —（无对应指标） |
| 第一步 Full Acc（Test-BC） | 87.5% | 100%（按构造） |
| Token 级预测 | 在所有位点同时预测分布 | 逐 token 自回归生成 |

Edit Flows 在静态第一步预测上的表现（Full Acc ~72-89%）说明编辑范式下的第一步并非主要问题。最终生成质量差距（43% vs 65%）应更多归因于：
- 多步采样的误差累积
- 中后期步骤的速率建模不足
- Euler 离散化和采样随机性

---

## 7. 相关文件索引

### 新数据集

- `analysis_subsets/USPTO_50K_PtoR_aug20_#global#/baseline_correct_seed42_1000/`
- 构建脚本：`scripts/create_baseline_correct_subset.py`

### 分析输出

| 数据集 | 输出目录 |
|--------|----------|
| Train-subset | `analysis_outputs/first_step/train_subset/forward/` |
| Test-original | `analysis_outputs/first_step/test_dedup_seed42_1000/forward/` |
| Test-BC | `analysis_outputs/first_step/baseline_correct/forward/` |

每个目录包含 `summary.json`、`per_example.pt`、`report.md`。

### 修改的文件

- `scripts/first_step_forward_analysis.py` — Token Acc 指标从 strict argmax 改为 tie-aware

---

## 8. 一句话总结

修正 Token Acc 指标后，诊断结论完全反转：Edit Flows 模型在第一步的 token 预测能力很强（Ins Token Acc@1 92-97%），主要弱项是位置感知的泛化（Train 93% → Test 78%）。在 baseline 可解的简单样本上，Edit Flows 第一步几乎达到训练集水平（Full Acc 87.5% vs 89.0%）。最终生成质量差距（43% vs 65%）的核心原因应从多步采样误差积累和后续步骤速率建模角度继续深挖。

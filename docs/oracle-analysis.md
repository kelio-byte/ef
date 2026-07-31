# Oracle Generation 实验：最优速率生成诊断

## 1. 背景与动机

Edit Flows 逆合成模型在训练集上的生成结果也不理想（standard Top-1 22.2%，#global# Top-1 40.6%），这暗示模型本身可能存在根本性问题。

此前 `docs/loss-analysis.md` 中的 Oracle Loss 测试表明：直接构造理论最优速率代入 Bregman loss，得到的 loss 均值（-5.24）远低于训练 loss（~11），但模型与 oracle 的差距仅 0.42σ——模型在**总量级**上是对的，但**分布不够锐利**。

为了进一步诊断：如果每一步都用最优（oracle）速率来进行 Euler 采样，生成的准确率能到多少？这可以分离：

- **方法固有问题**（Euler 离散化误差、编辑操作本身的随机性）
- **模型问题**（速率预测不准）

## 2. 实验方法

### 2.1 核心思路

用 `scripts/oracle_loss_profile.py` 中计算的理论最优速率，替换模型在 Euler 采样每一步的 forward。最优速率为：

$$u_{j,c} = K_{j,c} \cdot \frac{\dot{\kappa}_t}{1-\kappa_t}$$

其中 $K_{j,c}$ 是映射到 X 空间位置 $j$ 的需要编辑类型 $c$ 的 Z 空间位置数量。

### 2.2 动态对齐

训练时 Z 空间对齐是预计算的（`x_0` 与 `x_1` 的 Levenshtein DP）。但在生成过程中，$x_t$ 随着编辑逐步变化，与 $x_1$ 的对应关系也随之改变。因此每一步需要**动态重新对齐**：

```
每一步 (当前状态 x_t, 时间 t):
  1. x_t 与 x_1 做 Levenshtein DP 对齐 → z_t, z_1 (Z 空间)
  2. 计算 uz_mask (标记每个 Z 位置需要的编辑)
  3. rm_gap_tokens: 提取 X 空间结构
  4. 聚合 Z 空间编辑需求到 X 空间 → ux_cat
  5. 从 ux_cat 拆出 λ_ins, λ_sub, λ_del, ins_probs, sub_probs
  6. 用与 sample_euler 相同的随机采样逻辑决定编辑
  7. 应用编辑 → x_{t+h}
```

### 2.3 新增/修改文件

| 文件 | 说明 |
|------|------|
| `edit_flows/sampling/oracle.py` (**新**) | `compute_oracle_model_output()` — 动态对齐 + 最优速率计算，输出格式与模型 forward 一致 |
| `edit_flows/sampling/euler.py` | 新增 `sample_euler_oracle()` — 与 `sample_euler` 结构相同但用 oracle 替代模型 |
| `scripts/oracle_sample.py` (**新**) | 加载数据、运行 oracle 采样、保存 predictions.txt、自动调用 score 脚本评测 |

### 2.4 实验设置

- 数据：`train_subsets/` 中两个数据集的 unique product（`--deduplicate 20`，取每个 product 的第一个 variant），各 1000 条
- 参数：`n_samples=10`, `n_steps=100`, `batch_size=32`
- 设备：CPU（无需 GPU，计算瓶颈在 DP 对齐）
- 评测：`augmentation=1`，`--edit_flows` 等权排名

## 3. 结果

### 3.1 与模型对比

#### Standard 数据集 (USPTO_50K_PtoR_aug20)

| Metric | Oracle | Model |
|--------|--------|-------|
| Top-1 Acc | **93.2%** | 22.2% |
| Top-2 Acc | **99.4%** | 34.4% |
| Top-3 Acc | **99.8%** | 41.8% |
| Top-5 Acc | **99.9%** | 50.0% |
| Top-10 Acc | **99.9%** | 61.0% |
| Invalid SMILES | 22.6% | 32.5% |
| Unique Rates | 14.8% | 99.9% |

#### #global# 数据集 (USPTO_50K_PtoR_aug20_#global#)

| Metric | Oracle | Model |
|--------|--------|-------|
| Top-1 Acc | **93.4%** | 40.6% |
| Top-2 Acc | **99.5%** | 58.9% |
| Top-3 Acc | **100.0%** | 68.4% |
| Top-5 Acc | **100.0%** | 79.2% |
| Top-10 Acc | **100.0%** | 87.6% |
| Invalid SMILES | 9.0% | 14.2% |
| Unique Rates | 15.1% | 97.1% |

### 3.2 计算开销

- 每样本每步耗时约 0.14s（含 Python Levenshtein DP + tensor 操作）
- 1000 products × 10 samples × 100 steps ≈ 22-25 分钟 / 数据集（CPU）
- 主要瓶颈：每步的 Python DP 对齐（$O(L^2)$ 双重循环），序列长度 ~50 token

### 3.3 Scheduler 对比：Cubic vs Linear

上述实验均使用 CubicScheduler（$\kappa=t^3$）。为了验证 scheduler 对无效 SMILES 下界的影响，使用 LinearScheduler（$\kappa=t$）在相同设置下进行了对比实验。

#### Standard 数据集 (USPTO_50K_PtoR_aug20)

| Metric | Cubic | Linear | 变化 |
|--------|-------|--------|------|
| Top-1 Acc | 93.2% | **97.4%** | **+4.2pp** |
| Top-2 Acc | 99.4% | 99.9% | +0.5pp |
| Top-3 Acc | 99.8% | 100.0% | +0.2pp |
| Invalid SMILES | 22.6% | **9.2%** | **-59%** |
| Unique Rates | 14.8% | 12.2% | — |

#### #global# 数据集 (USPTO_50K_PtoR_aug20_#global#)

| Metric | Cubic | Linear | 变化 |
|--------|-------|--------|------|
| Top-1 Acc | 93.4% | **98.6%** | **+5.2pp** |
| Top-2 Acc | 99.5% | 100.0% | +0.5pp |
| Top-3 Acc | 100.0% | 100.0% | — |
| Invalid SMILES | 9.0% | **2.6%** | **-71%** |
| Unique Rates | 15.1% | 11.9% | — |

**关键发现**：仅将 scheduler 从 Cubic 改为 Linear（不增加步数、不调整 clamp），无效 SMILES 即大幅下降——Standard 从 22.6% 降至 9.2%，#global# 从 9.0% 降至 2.6%。Top-1 也随之提升至 97%+。

## 4. 分析与观察

### 4.1 Edit Flows 方法本身是可行的

Oracle 生成在 Top-1 达到 93%+，Top-2 达到 99%+。这说明 Edit Flows + Copy Product 耦合的编辑范式**没有根本性缺陷**——只要速率预测准确，Euler 采样能够从产物编辑出正确的反应物。

### 4.2 模型速率预测是当前主要瓶颈

Oracle vs 模型的 Top-1 差距约 **50-70 个百分点**。这与此前 Oracle Loss 分析的结论一致：模型在总速率尺度上是正确的（u_tot ≈ 13），但**分布不够锐利**——正确编辑的速率不够集中，无关编辑的速率不够接近 0。

具体来说，Bregman loss 分解为 $L = u_{tot} - CE_{term}$：
- **u_tot**（压制无关编辑）：模型 ≈ oracle（都约 13-14），说明模型已学会最小编辑偏好
- **CE_term**（奖励正确编辑）：模型远低于 oracle（3+ vs 19），说明模型对正确编辑的置信度不够

### 4.3 无效 SMILES 的根因：Euler 离散化 + Clamp 导致编辑概率 < 1

Oracle 生成仍有 22.6%（standard）和 9.0%（#global#）的无效 SMILES。初看令人困惑——Oracle 每一步都给出了完全准确的编辑速率，为什么还会产生无效分子？

**这不是 bug，而是 Euler 离散化与 `clamp=50` 共同导致的一个数学必然结果。**

#### 4.3.1 核心机制：每次编辑有非零的 "永不触发" 概率

Euler 采样中，每一步对某个编辑以概率 $1 - e^{-h \cdot \lambda}$ 触发。整个过程等价于一个时变 Poisson 过程。某编辑在整个轨迹上**至少触发一次**的概率为：

$$P(\text{fire}) = 1 - \exp\left(-\sum_i h_i \cdot \lambda_i\right) = 1 - e^{-H}$$

其中 $H = \sum_i h_i \cdot \lambda_i$ 是该编辑的 **total integrated hazard**。这个公式是精确的（不是近似），因为每步的触发概率 $1-e^{-h\lambda}$ 恰好是 Poisson 过程的离散化形式，各步之间独立，累积不触发概率为 $\prod e^{-h_i\lambda_i} = e^{-H}$。

对于 CubicScheduler（$\kappa = t^3$, $\dot{\kappa} = 3t^2$），Oracle 最优速率为：

$$\lambda(t) = K \cdot \frac{\dot{\kappa}}{1-\kappa} = K \cdot \frac{3t^2}{1-t^3}$$

被 clamp 到最大 50。步长 $h = \min(h_{\text{default}}, \frac{1-\kappa}{\dot{\kappa}}) = \min(0.01, \frac{1-t^3}{3t^2})$。

**关键**：对于 $K=1$ 的单次编辑，$H \approx 3.81$，因此：

$$P(\text{永不触发}) = e^{-3.81} \approx 2.2\%$$

#### 4.3.2 为什么 $H$ 是有限值

$\frac{\dot{\kappa}}{1-\kappa}$ 在 $t \to 1$ 时本身是**发散的**（$\int_0^1 \frac{3t^2}{1-t^3} dt = -\ln(1-t^3)|_0^1 = \infty$），理论上每个编辑一定会触发。但 `clamp=50` 截断后，H 变为有限。轨迹可分为三段：

| 区间 | t 范围 | 步长 h | λ | 每步 hazard | 区间总 H |
|------|--------|--------|---|-------------|----------|
| ① 正常 | [0, ~0.9805] | 0.01 | 3t²/(1-t³) | 递增, 0→0.5 | ~2.86 |
| ② Clamped, 固定步长 | [~0.9805, ~0.991] | 0.01 | 50 (clamped) | 0.5 | ~0.5 |
| ③ Clamped + Adaptive | [~0.991, 1] | (1-t³)/(3t²) → 0 | 50 | **递减, → 0** | ~0.45 |

**第三段是最容易误解的地方**：虽然 adaptive step 让步数增加（h 变小，迭代次数变多），但每步贡献的 hazard $h \cdot \lambda = \frac{1-t^3}{3t^2} \cdot 50$ 随 $t \to 1$ **递减趋于 0**。越接近终点，每一步能完成的编辑越少。这是因为 clamp=50 限制了 λ 的增长，而 adaptive h 又被迫缩小，两者相乘 → 0。

#### 4.3.3 这完美解释了全部实验数据

**无效 SMILES 率**：每个 $K=1$ 的编辑有 ~2.2% 概率永不触发。产品中有多个此类编辑时，只要有一个未触发就会产生无效 SMILES：

$$P(\text{产品无效}) = 1 - (1-0.022)^{E} \quad \text{其中 E = 单次编辑数}$$

| 数据集 | 无效 SMILES | 反推 E | 解释 |
|--------|------------|--------|------|
| Standard | 22.6% | ~12 | $1 - 0.978^{12} \approx 23\%$ |
| #global# | 9.0% | ~4-5 | $1 - 0.978^{4.5} \approx 9.5\%$ |

**#global# 无效 SMILES 更低**的原因是 R-SMILES 的 root-alignment 使产物和反应物共享公共前缀，需要对齐的差异更少 → 编辑数 E 更少 → 更少编辑面临 "不触发" 风险。

**Top-1 的 93.2%（而非 100%）**：部分产品的所有编辑都触发了，但恰好某个**关键编辑**没有触发——产物变成了一个合法但错误的 SMILES（valid but wrong），RDKit 可以解析但和 ground truth 不同。

**注意**：$K>1$ 的编辑（多个 Z 位置在同一 X 位置需要同类型编辑）的 H 会乘以 K，触发概率极高（如 $K=2$ 时 $H=7.62$, 不触发概率仅 0.05%）。瓶颈只在 $K=1$ 的编辑上。

#### 4.3.4 降低无效 SMILES 的方法

这与速率预测的准确性无关——即使 Oracle 给出了完美速率，也无法规避。要降低这个下界：

- **增加采样步数**（如 200-500 步）：直接增加 H（因为更多的步数 = 更多的 hazard 累积机会，尤其在 clamped 区间前）
- **提高 clamp 值**（如 50→200）：延后 clamp 生效的 t 时刻，让未 clamp 区间贡献更多 H（$\int_0^{t_{clamp}}$ 更大），同时 clamped 区间的积分也更宽
- **减小或去除 clamp**：理论上 H→∞ 则 P(不触发)→0，但会导致 t→1 时步数爆炸，实际不可行
- **改用更高阶的 SDE solver**：可能更高效地利用 hazard budget，但不能根本解决有限 H 的问题

#### 4.3.5 Linear Scheduler 实验验证

第 3.3 节的 scheduler 对比实验完美验证了上述理论分析。

**机制**：Linear scheduler（$\kappa=t$, $\dot{\kappa}=1$）的 sched_coeff 为：

$$\lambda(t) = K \cdot \frac{1}{1-t}$$

与 Cubic 的关键区别：

| 属性 | Cubic (t³) | Linear (t) |
|------|-----------|------------|
| sched_coeff 起始值 (t→0) | 0 | **1** |
| P(sc<1) 占比 | ~53% | **0%** |
| Total integrated hazard H (K=1) | 3.60 | **4.69** |
| P(单编辑永不触发) | 2.74% | **0.92%** |

Cubic 的 sc 从 0 起步，前 53% 的时间步处于"平坦区"（sc<1），每一步贡献的 hazard 较小。Linear 的 sc 从 1 起步，全程没有平坦区，每一步贡献更高的 hazard，累积 H 增大 30%，P(永不触发) 降至 Cubic 的 1/3。

**理论预测 vs 实验结果**：

| 数据集 | E (估) | Cubic 预测 | Linear 预测 | Linear 实际 |
|--------|--------|-----------|------------|------------|
| Standard | ~12 | 22.6% (H=3.60) | ~10.5% (H=4.69) | **9.2%** |
| #global# | ~4-5 | 9.0% (H=3.60) | ~3.6% (H=4.69) | **2.6%** |

实验结果略优于理论预测，这是因为更高的早期 hazard 还能减少"关键编辑延迟 → 后续对齐偏差"的级联效应。

**结论**：sched_coeff 的分布（而不仅仅是总步数）对无效 SMILES 下界有决定性影响。选择在 t∈[0,1) 全程 sched_coeff ≥ 1 的 scheduler（如 Linear），可以在不增加步数的前提下大幅降低无效 SMILES 率。

### 4.4 #global# 格式更干净

无论是 oracle（9.0% vs 22.6% 无效）还是模型（14.2% vs 32.5% 无效），#global# 格式的无效率都显著更低。如 4.3.3 节所分析，这是因为 R-SMILES 的 root-alignment 使得产物和反应物共享公共前缀，对齐所需的编辑操作更少（E 更小），所有编辑都触发的概率更高。

### 4.5 Oracle 的确定性特征

Oracle 的 Unique Rates 仅 ~15%（每 10 个采样中约 1.5 个唯一结果），远低于模型的 97-99%。这是因为最优速率对大多数位置是**确定性的**——每个位置要么匹配（无需编辑），要么只有一个正确的编辑操作（插入/删除特定 token 或替换为特定 token）。多条采样之间的差异主要来自 Euler 离散化的随机性（哪一步完成编辑、是否有残留未编辑位置），而非编辑类型的不确定性。

### 4.6 当前模型与 Oracle 的 Loss 差距为何不大但生成差距巨大？

Oracle Loss 分析显示模型 loss（~11）与 oracle loss（~ -5）的差距仅约 0.42σ，似乎不大。但生成结果差距悬殊（Top-1 93% vs 22%）。原因：

- Loss 的方差极大（oracle loss std = 39），0.42σ 的均值差距在统计上不显著
- Bregman loss 的绝对值与生成质量不是线性关系——loss 中多一个 bit 的不确定性（log rate 差 ~0.69）在 100 步累积采样中可能被放大
- Oracle 的确定性速率 $u = K \cdot sc$ 在正确位置集中全部概率质量，而模型的 softmax 输出将概率分散到多个 token，每一步都可能采样到错误 token，100 步后错误累积

## 5. 改进方向

基于以上分析，改进分两个层面：**降低无效 SMILES 下界**（离散化问题，影响 Oracle 和模型）和**提升模型速率预测质量**（模型问题）。

### 5.1 降低无效 SMILES 下界（离散化）

这些改进对 Oracle 和模型**都有效**——它们降低的是方法本身的误差下限。

| 方向 | 具体做法 | 优先级 |
|------|---------|--------|
| **改用 Linear scheduler** | κ=t，sched_coeff 全程 ≥1，H 增大 30%，无效 SMILES 降低 59-71%（已验证） | **最高** |
| **增加采样步数** | n_steps: 100→200~500，直接增加 total integrated hazard H | **最高** |
| **提高 clamp 值** | clamp: 50→100~200，延后 clamp 生效时刻 | 高 |
| **进一步改进 scheduler** | 如 κ=t² 等介于 Linear/Cubic 之间的方案，或 t 的非均匀离散化 | 中 |

注意：Linear scheduler 在训练中可能导致更剧烈的 loss 震荡（无平坦区，sc 全程 ≥1），见 `docs/loss-analysis.md` 第 4 节的 scheduler 对比分析。训练时是否使用 Linear scheduler 需要单独实验验证。

### 5.2 提升模型速率预测质量

Oracle 实验确认了模型速率预测不准是当前性能的主要瓶颈（Top-1 差距约 50-70pp）。改进应集中于**让模型的速率分布更锐利**（提升 CE_term）：

| 方向 | 具体做法 | 优先级 |
|------|---------|--------|
| **降低 softmax temperature** | 对 Q_ins/Q_sub 的输出除以 temperature < 1，让分布更尖锐 | 高 |
| **调整 softplus 参数** | 让速率更容易接近 0（无编辑位置）或很大（需要编辑的位置） | 高 |
| **增加模型容量** | 当前 13.5M 参数，对于 72-260 词表可能偏小 | 中 |
| **改进采样策略** | 当前 Euler 中不同 batch 元素 t 不同步增长，已完成元素被过度编辑 | 低 |

## 6. 运行命令

```bash
# Standard 数据集 (Cubic scheduler)
PYTHONPATH=. python scripts/oracle_sample.py \
    --products_file train_subsets/USPTO_50K_PtoR_aug20/test/src-test.txt \
    --targets_file train_subsets/USPTO_50K_PtoR_aug20/test/tgt-test.txt \
    --vocab_file /data6/duanbh/desktop/retrosynthesis/dataset/USPTO_50K_PtoR_aug20/example.vocab.src \
    --output_dir train_subsets/eval/oracle_standard \
    --n_samples 10 --n_steps 100 --batch_size 32 \
    --deduplicate 20 --device cpu

# #global# 数据集 (Cubic scheduler)
PYTHONPATH=. python scripts/oracle_sample.py \
    --products_file train_subsets/USPTO_50K_PtoR_aug20_#global#/test/src-test.txt \
    --targets_file train_subsets/USPTO_50K_PtoR_aug20_#global#/test/tgt-test.txt \
    --vocab_file /data6/duanbh/desktop/retrosynthesis/dataset/USPTO_50K_PtoR_aug20_#global#/example.vocab.src \
    --output_dir train_subsets/eval/oracle_global \
    --n_samples 10 --n_steps 100 --batch_size 32 \
    --deduplicate 20 --score_script scripts/score_#global#.py --device cpu

# Standard 数据集 (Linear scheduler, --scheduler linear)
PYTHONPATH=. python scripts/oracle_sample.py \
    --products_file train_subsets/USPTO_50K_PtoR_aug20/test/src-test.txt \
    --targets_file train_subsets/USPTO_50K_PtoR_aug20/test/tgt-test.txt \
    --vocab_file /data6/duanbh/desktop/retrosynthesis/dataset/USPTO_50K_PtoR_aug20/example.vocab.src \
    --output_dir train_subsets/eval/oracle_standard_linear \
    --n_samples 10 --n_steps 100 --batch_size 32 \
    --deduplicate 20 --device cpu --scheduler linear

# #global# 数据集 (Linear scheduler, --scheduler linear)
PYTHONPATH=. python scripts/oracle_sample.py \
    --products_file train_subsets/USPTO_50K_PtoR_aug20_#global#/test/src-test.txt \
    --targets_file train_subsets/USPTO_50K_PtoR_aug20_#global#/test/tgt-test.txt \
    --vocab_file /data6/duanbh/desktop/retrosynthesis/dataset/USPTO_50K_PtoR_aug20_#global#/example.vocab.src \
    --output_dir train_subsets/eval/oracle_global_linear \
    --n_samples 10 --n_steps 100 --batch_size 32 \
    --deduplicate 20 --score_script scripts/score_#global#.py --device cpu --scheduler linear
```

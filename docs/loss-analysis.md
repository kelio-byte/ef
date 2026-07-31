# Loss 分析与诊断

## 1. 当前训练 Loss 的现象

训练日志（USPTO_50K_PtoR_aug20, step ~1.1M / 5M total）显示以下特征：

| 现象 | 数值 |
|------|------|
| Loss 典型范围 | 5 ~ 16 |
| Loss 震荡幅度 | 邻近 step 间可差 5-10（如 step 1,112,100: -0.3，step 1,112,300: 16.2） |
| 出现负 loss | 有（step 1,112,100: -0.30） |
| u_tot（总编辑速率） | 10 ~ 17 |
| ins/del/sub 分解 | 大致均衡，ins 略高 |

核心疑问：
1. 负 loss 是 bug 还是理论正常行为？
2. Loss 大幅波动是方法固有属性还是实现问题？

---

## 2. Oracle Loss 测试方法

### 2.1 思路

Bregman 散度在条件速率处取得最小值。如果我们不经过神经网络，而是**直接构造理论最优的速率输出**代入 loss 函数，就能得到 loss 的理论下界。观察这个「Oracle Loss」的分布，可以分离：

- **方法固有方差**（来自 t 采样、z_t 采样、对齐结构）
- **模型差距**（网络容量、优化不足造成的额外 loss）

### 2.2 构造最优速率

Bregman loss 对单样本的形式为：

$$L(\theta) = \sum_{y \neq x_t} u^\theta(y|x_t) - \sum_{i} \mathbf{1}[z_1^i \neq z_t^i] \cdot \frac{\dot{\kappa}_t}{1-\kappa_t} \cdot \log u^\theta(\text{correct}_i | x_t)$$

对每个 X 空间位置 j 和编辑通道 c，令：

$$K_{j,c} = \text{映射到位置 j 且需要编辑 c 的 Z 位置数量}$$

则最优速率为：

$$u^\theta_{j,c} = K_{j,c} \cdot \frac{\dot{\kappa}_t}{1-\kappa_t}$$

推导：对 $u_{j,c}$ 求偏导，$\partial L/\partial u = 1 - K \cdot \text{sched\_coeff} / u = 0 \Rightarrow u = K \cdot \text{sched\_coeff}$。

实现要点：
1. 运行 `prepare_batch` 获取 `uz_mask`（标记每个 Z 位置需要什么编辑）
2. 使用 `fill_gap_tokens_with_repeats` 相同的 Z→X 映射，将 Z 空间的编辑需求聚合回 X 空间
3. 聚合时每个编辑需求贡献 `sched_coeff` 的速率
4. 无编辑需求的通道设为极小值（$10^{-9}$）
5. 将构造的 `log_ux_cat` 代入 `bregman_loss` 计算

测试脚本：`scripts/oracle_loss_profile.py`

### 2.3 结果

200 个 batch（batch_size=128，共 25,600 个样本）：

```
Oracle Loss 分布 (κ_t = t³, clamp=50)
═══════════════════════════════════════
  Mean:         -5.24
  Std:          39.1
  Min/Max:      -728 / +72
  Median:       1.62
  负 loss 占比:  23.0%
  u_tot mean:   13.7
  CE term mean: 18.9
  sched_coeff mean: 3.81 ± 8.79
  #edits mean:  10.6 ± 10.1
```

**关键发现：**

| | Oracle (理论最优) | 实际训练 (step ~1.1M) |
|---|---|---|
| Loss mean | **-5.24** | ~11 |
| Loss range | [-728, +72] | [-0.3, +16] |
| u_tot | 13.7 | 10-17 |
| Gap vs Oracle | — | 16.2 (仅 **0.42σ**) |

---

## 3. 各项指标的含义

### 3.1 sched_coeff = κ̇/(1-κ)

时间权重系数，决定了编辑监督信号的强度。

- **sc < 1**：loss 贡献为正且平缓（导数 < 0.5），模型处于「安全区」
- **sc = e ≈ 2.718**：单编辑 loss 贡献 = 0（正负分界线）
- **sc > e**：单编辑 loss 贡献为负且急剧增长（导数 → -∞），模型处于「负 loss 区」
- **sc → ∞ (t→1)**：被 clamp 到 50 截断

### 3.2 u_tot（第一项）

$$\sum_{y \neq x_t} u^\theta(y|x_t)$$

所有可能编辑的速率之和。作用：**压制无关编辑**，形成「最小编辑偏好」。Oracle 的 u_tot ≈ 13.7，训练模型也在此范围（10-17），说明模型已学会正确的总速率尺度。

### 3.3 CE term（第二项）

$$\sum_i \text{mask}_i \cdot \frac{\dot{\kappa}_t}{1-\kappa_t} \cdot \log u^\theta(\text{correct}_i | x_t)$$

对正确编辑的对数速率进行奖励。Oracle 的 CE term = 18.9，训练模型约 3——差距在这里。模型对正确编辑的速率估计偏保守（log rate 偏低），导致 CE 项不够大（不够正）。

### 3.4 Loss 方差的来源

Loss 与各因素的相关系数：

```
                loss     u_tot   ce_term  sched_coeff  n_edits
loss            1.00    -0.74    -0.97     -0.53         0.25
sched_coeff    -0.53     0.32     0.49      1.00        -0.30
n_edits         0.25     0.11    -0.13     -0.30         1.00
```

- **ce_term 主导 loss 方差**（r = -0.97），而 ce_term 由 sched_coeff 驱动
- **n_edits 与 loss 弱正相关**（编辑越多 loss 越大），但影响远小于 sched_coeff
- 核心机制：t 采样 → sched_coeff 变化 → CE 项剧烈变化 → loss 震荡

### 3.5 负 loss 的成因

单编辑的最优 loss 贡献 = `sc * (1 - log(sc))`。

当 sc > e ≈ 2.718 时，`1 - log(sc) < 0`，贡献为负。这是**理论必然结果**，不是 bug——最优速率本身就产生负 loss。

---

## 4. 不同 κ_t 调度器的对比

### 4.1 三种调度器

| 调度器 | κ_t | κ̇_t | sched_coeff |
|--------|-----|------|-------------|
| Cubic | t³ | 3t² | 3t²/(1-t³) |
| Quadratic | t² | 2t | 2t/(1-t²) |
| Linear | t | 1 | 1/(1-t) |

### 4.2 sched_coeff 分布对比

```
                        Cubic (t³)    Quad (t²)   Linear (t¹)
─────────────────────────────────────────────────────────────
Mean sc                   3.88          4.25         4.93
Median sc                 0.86          1.33         2.01
P(sc < 1)                53.0%         41.3%         0.0%
P(sc > e)                26.3%         30.2%        37.0%
P(clamped@50)            ~2%           ~2%          ~2%
Warm-up (t 使 sc<1)     t < 0.53      t < 0.41     无
```

### 4.3 为什么 Cubic 最稳定

sc 对 loss 的影响是**高度非线性**的：

| sc | 单编辑 loss | |∂loss/∂sc||
|----|------------|------------|
| 0.1 | +0.33 | 2.3 |
| 0.5 | +0.85 | 0.7 |
| 1.0 | +1.00 | 0.0 |
| 2.0 | +0.61 | 0.3 |
| 2.72 | 0 | 1.0 |
| 5.0 | -3.05 | 2.6 |
| 10.0 | -13.0 | 3.3 |
| 50.0 | -145.6 | 4.9 |

- **sc < 1**：loss 平坦，梯度温和
- **sc > e**：loss 陡峭，梯度急剧增大

Cubic 将 53% 样本压在 sc < 1 的平坦区，提供了天然的梯度稳定机制。

Linear 的问题：
- sc 最小值为 1，**永远无法进入平坦区**
- 每个样本都在陡峭区或负 loss 区，训练全程高压
- 没有「先学会压低无关速率」的渐进课程

Quadratic 介于中间：41% 在平坦区，但比 Cubic 少 12 个百分点。

### 4.4 进一步方向：t⁴

如果追求更大稳定性，可考虑 `κ_t = t⁴`：
- `sc = 4t³/(1-t⁴)`
- P(sc<1) ≈ 60%，median sc ≈ 0.6
- 代价：编辑信号更晚出现，可能降低收敛速度

---

## 5. 下一步改进方向

### 5.1 调整 clamp 值（低风险，推荐先试）

当前 `clamp=50`，约 2% 样本触及。这些样本贡献了损失中最极端的负值（-700+）。

- 降至 20 或 30：削短极端 loss 尾部，减少梯度尖峰
- 风险：t→1 时的边界条件约束变弱——但这 2% 样本影响有限

### 5.2 改进采样策略

当前 Euler 采样中，不同 batch 元素的 t 不同步增长，已完成的元素被过度编辑（Bug 2）。修复后可能间接改善训练-推理一致性。

### 5.3 模型侧优化

Oracle 测试表明模型与最优 loss 仅差 0.42σ——模型在**总量级**上是对的，但**分布不够锐利**（正确编辑的速率不够集中，无关编辑的速率不够接近 0）。可能的改进：

- 调整 softplus 的温度参数，让速率更容易接近 0 或很大
- 对 Q_ins/Q_sub 使用更低的 temperature（sharpen softmax）
- 增加模型容量（当前 13.5M 参数）

### 5.4 时间采样策略

当前 t ~ Uniform(0, 1)。考虑从更关注边界的分布采样：
- 在 t→1 附近增加采样密度（加强对目标分布的约束）
- 或在 t→0 附近减少采样（降低噪音样本比例）
- 这会影响 sched_coeff 的经验分布，间接改变方差

### 5.5 训练超参

- 当前仅训练了 ~1.1M / 5M steps（22%），继续训练 loss 可能进一步下降
- Noam scheduler 的 warmup（8000 steps）相对总步数偏短，warmup 期间 loss 从 158 降至 85，大部分下降发生在前 100 步
- 可尝试适当增大 warmup 或使用 cosine decay

---

## 6. 运行测试

```bash
# Oracle loss 分布分析
PYTHONPATH=. python scripts/oracle_loss_profile.py \
    --config configs/retro.yaml --num_batches 200 --device cpu

# Scheduler 对比
python scripts/compare_schedulers.py
```

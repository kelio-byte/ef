# Sample Scheduler Analysis: time_input & clamp_kappa 对比实验分析

## 1. 问题背景

两个 checkpoint 在相同训练配置下仅 `time_input` 和 `clamp_kappa` 不同，eval 结果出现明显差异：

| 实验 | time_input | clamp_kappa | Top-1 | Top-10 | Unique Rate |
|------|-----------|-------------|-------|--------|-------------|
| 13-32-29 | `kappa` | `true` | 47.993% | 85.321% | 84.839% |
| 13-34-45 | `t` | `false` | 50.210% | 84.622% | 78.935% |

原始猜想是"仅有 time_input 区别"，但实际上两个配置参数同时变化，需要分离分析。

## 2. 代码审查结论

### 2.1 训练/采样一致性：无 bug

审查了以下关键路径，确认训练侧和采样侧保持一致：

- **训练侧 time 计算** (`trainer.py:98`)：`t_model = scheduler(t) if time_input == "kappa" else t`
- **采样侧 time 计算** (`euler.py:96, _compute_model_time`)：`time_input == "kappa"` 时返回 `sample_scheduler(t)`，`time_input == "t"` 且 scheduler 相同时返回 `t`
- **rate scale**：loss (`loss.py:21`) 和采样 (`euler.py:117, apply_rate_parameterization`) 都通过同一个 `get_rate_scale` 函数计算，传入的都是原始 `t`（非 `t_model`），参数透传一致

### 2.2 无 bug 不代表无问题

虽然训练/采样没有 mismatched logic，但两种配置下的**训练动态和采样行为有实质性差异**。

## 3. clamp_kappa 的影响（主要因素）

### 3.1 机制

`get_rate_scale` 在两种 `clamp_kappa` 下行为不同：

```
clamp_kappa=false:  k(t) = min( deriv(t) / (1−κ(t)), clamp_max )
                          = min( 3t² / (1−t³), 50 )
                          峰值 = 50 (在 k(t) 超过 50 时被截断)

clamp_kappa=true:   k(t) = deriv(t) · min( 1 / (1−κ(t)), clamp_max )
                          = 3t² · min( 1/(1−t³), 50 )
                          峰值 = 3 × 50 = 150 (deriv(t→1) → 3)
```

关键差异：`clamp_kappa=true` 在 t→1 时的 rate scale 是 `false` 的 3 倍（150 vs 50）。

### 3.2 影响：训练

`use_rate_reparam=true` 下，loss = `k(t) · (u_tot - ce_term)`。

`clamp_kappa=true` 让接近 t=1 的样本 loss 权重高达 150，是 `false` 的 3 倍。但 t→1 时多数 token 已收敛到 target（κ(t)→1，z_t 大部分来自 z_1），有效编辑信号反而弱。高权重 + 弱信号 → 梯度噪声大，可能导致模型在末尾时间步学得不稳定。

### 3.3 影响：采样

采样时 `apply_rate_parameterization` 将模型预测的 base rate 乘以相同的 `k(t)`。t→1 时：
- `false`：rate scale 上限 50
- `true`：rate scale 上限 150

过高的 scale 让采样末尾步骤编辑过于激进，已正确的 token 可能被错误修改。

### 3.4 Eval 数据佐证

kappa+clamp (true) 版本 Unique Rate 更高（84.8% vs 78.9%），但 Top-1 更低（48.0% vs 50.2%）。更高 diversity + 更低 accuracy = 模型的编辑决策更随机/不稳定，与末尾过度编辑的推断一致。

## 4. time_input 的影响（次要但存在）

### 4.1 机制

`time_input` 决定传给 `SinusoidalTimeEmbedding` 的值：

- `t`：直接传入原始 t ∈ [0, 1]，分布均匀
- `kappa`：传入 κ(t) = t³ ∈ [0, 1]，分布集中在低值区（50% 样本 κ < 0.125）

### 4.2 影响

Sinusoidal embedding 使用 log-spaced 频率（1 ~ 10000⁻¹），当输入 κ(t) 较小时（如 t=0.1 → κ=0.001），embedding 各维度几乎为常数（sin≈0, cos≈1），时间步之间难以区分。

同时，高 κ 区域（κ > 0.9，对应 t > 0.965）训练中仅占约 3.5% 样本，模型在此区域学得不充分，而采样最后约 35 步恰好在高 κ 区域执行关键编辑。

### 4.3 与 clamp_kappa 的交互

`time_input=kappa` 让模型在末尾（高 κ）学得不好，`clamp_kappa=true` 又让末尾 rate scale 放大到 150。两者叠加使得末尾编辑既不准又激进，可能是 kappa+clamp 组合表现最差的原因。

## 5. _compute_model_time 中一个潜在问题（当前实验未触发）

`euler.py:105-114` 中，当 `use_rate_reparam=false` 且 training/sampling scheduler 不同时：

```python
k_train = get_rate_scale(t_model, train_scheduler, ...)
```

此处 `t_model` 可能是 κ(t)（当 `time_input=kappa` 时），但 `get_rate_scale` 内部会再做一次 `scheduler(t_model)`，导致二次映射。当前实验 `use_rate_reparam=true` 未触发此路径，但未来若组合 `use_rate_reparam=false` + `time_input=kappa` + 不同 scheduler 时会出错。

## 6. 建议实验方案

做两因素交叉实验，分离 `time_input` 和 `clamp_kappa` 的独立效应：

| # | time_input | clamp_kappa | 目的 |
|---|-----------|-------------|------|
| A | `t` | `false` | 当前最佳 baseline |
| B | `kappa` | `false` | 仅测 time_input |
| C | `t` | `true` | 仅测 clamp_kappa |
| D | `kappa` | `true` | 当前组合（预期最差）|

建议先在 `train_subsets` 上小规模训练（如 500k steps），快速暴露差异。

观察指标：
- Loss 曲线稳定性（特别是训练后期）
- u_tot 分布随时间步的变化
- Top-1 / Top-10 / Unique Rate
- 各时间步的编辑统计（是否某段时间编辑异常多）

预期：
- 如果 `clamp_kappa=true` 是主因，则 C 和 D 应明显差于 A 和 B
- 如果 `time_input=kappa` 是主因，则 B 和 D 应差于 A 和 C
- 如果两者有交互，D 会明显差于任何单因素变体

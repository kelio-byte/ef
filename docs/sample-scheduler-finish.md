# 跨 Scheduler 采样：理论推导与实现总结

## 1. 背景

Edit Flows 中，scheduler 的 κ(t) 控制从 source 到 target 的"进度"。不同的 κ(t) 函数（如 Cubic κ=t³、Linear κ=t）对应不同的时间参数化，但它们描述的是同一条从产物到反应物的编辑路径。

训练时的速率缩放系数为：

$$k(t) = \frac{\dot{\kappa}_t}{1-\kappa_t}$$

实验发现，Oracle 用 Linear scheduler 采样（即使只改 scheduler 不改步数），无效 SMILES 从 22.6% 降至 9.2%（Standard），Top-1 从 93.2% 升至 97.4%。这引发了两个核心问题：

1. **能否训练用一种 scheduler、采样用另一种 scheduler？**
2. **如果可行，如何保证跨 scheduler 采样时模型输出的速率是正确的？**

本文对上述问题给出完整的理论推导和实现方案。

---

## 2. 理论推导

### 2.1 为什么 scheduler 在理论上等价

训练损失在 rate reparam（`use_rate_reparam=true`）下为：

$$\tilde{L} = k(t) \cdot (u'_{\text{tot}} - CE')$$

对 t 求期望，做变量替换 $s = \kappa(t)$，$ds = \dot{\kappa}_t dt$：

$$E_t[\tilde{L}] = \int_0^1 \frac{\dot{\kappa}_t}{1-s} \cdot \ell(v') \cdot \frac{ds}{\dot{\kappa}_t} = \int_0^1 \frac{\ell(v')}{1-s} ds$$

**κ̇ 精确消掉了。** 所有 scheduler 在 κ 空间里具有完全相同的训练目标。scheduler 仅仅是 [0,1] 区间上的一种"时钟"重参数化，不影响最优解。

然而，将训练好的模型用于不同 scheduler 采样时，有两个实际问题需要解决：**模型输入的时间值**和**模型输出的速率值**。

### 2.2 问题一：模型时间输入的分布偏移

模型以时间 t 为输入（通过 SinusoidalTimeEmbedding）。训练时：

- Cubic scheduler：t_cubic → κ = t_cubic³
- 模型学到映射 $(x_t, t_{\text{cubic}}) \to v'$

采样时若换用 Linear scheduler：

- Linear scheduler：t_lin → κ = t_lin
- 同样的 t=0.5，Cubic 下 κ=0.125（状态接近产物），Linear 下 κ=0.5（状态已半程）
- 模型看到 t=0.5 会"以为"状态还应接近产物，但实际状态已大不相同

**解决方案：t 映射。** 注意到中间状态 $z_t$ 的分布只依赖 κ，不依赖 t：

$$z_t[j] = \begin{cases} z_1[j] & \text{w.p. } \kappa(t) \\ z_0[j] & \text{otherwise} \end{cases}$$

因此 $x_t$ 的条件分布为 $p(x_t \mid \kappa)$，与 scheduler 无关。只需将当前的 κ 映射回训练 scheduler 下的等效 t：

$$t_{\text{model}} = \kappa_{\text{train}}^{-1}(\kappa_{\text{sample}}(t))$$

对 Cubic 训练 + Linear 采样：

$$t_{\text{model}} = (t_{\text{linear}})^{1/3}$$

这样模型看到的 $(x_t, t_{\text{model}})$ 对符合其训练分布。

对于新模型，更干净的方式是直接让模型以 κ(t) 为输入（`time_input: "kappa"`）。因为 κ 直接表示"混合进度"（0=产物，1=反应物），天然 scheduler-invariant，不再需要上述映射。

### 2.3 问题二：无 rate reparam 时的速率缩放

`use_rate_reparam=false` 时，模型直接预测真实 CTMC 速率 v：

$$v_{\text{model}} \approx K \cdot k_{\text{train}}(t_{\text{model}})$$

速率缩放系数 k(t) 被模型内化到了输出中。即使做了 t 映射修复了输入分布，模型输出的速率仍然对应 $k_{\text{train}}$，而非采样所需的 $k_{\text{sample}}$：

| 采样 t_lin | κ | t_model | 模型预测 v ≈ | 正确速率 v* = |
|-----------|-----|--------|-------------|-------------|
| 0.5 | 0.5 | 0.794 | K · k_cubic(0.794) = K · 3.78 | K · k_linear(0.5) = K · 2.0 |

模型预测是正确值的约 1.9 倍。

**解决方案：速率校正。** 在模型输出上乘以校正因子：

$$v_{\text{corrected}} = v_{\text{model}} \cdot \frac{k_{\text{sample}}(t)}{k_{\text{train}}(t_{\text{model}})}$$

推导：

$$v_{\text{model}} \approx K \cdot k_{\text{train}}(t_{\text{model}}) \quad \Rightarrow \quad v_{\text{corrected}} \approx K \cdot k_{\text{sample}}(t)$$

本质是在采样时手动「除掉 bake 进去的 $k_{\text{train}}$，乘上正确的 $k_{\text{sample}}$」，等价于在采样侧做了一次 rate reparam 的解耦操作。

**注意：** 此校正只针对 `log_rates`（$\lambda_{\text{ins}}, \lambda_{\text{sub}}, \lambda_{\text{del}}$），不涉及 `log_ins_probs` / `log_sub_probs`——token 分布是纯内容问题，与 scheduler 无关。

当 `use_rate_reparam=true` 时，模型输出的是 base rate v'，经由 `apply_rate_parameterization` 乘以 $k_{\text{sample}}$。这直接得到 $v = K \cdot k_{\text{sample}}$，无需额外校正。

### 2.4 完整的 t → v 采样流程

```
t (采样时间)
 │
 ├── κ_sample(t)
 │      │
 │      └── t_model = κ_train⁻¹(κ_sample) ──→ model(x_t, t_model) → v_out
 │
 ├── k_sample = k(t, sample_scheduler)
 │
 └── k_train  = k(t_model, train_scheduler)
           │
           │   ┌─ use_rate_reparam=True:   v = v_out · k_sample     (原有)
           ├───┤
           │   └─ use_rate_reparam=False:  v = v_out · k_sample / k_train  (新增校正)
           ▼
     最终 CTMC 速率: v = K · k_sample(t)   ← 始终正确
```

---

## 3. Clamp 的 κ̇ 无关化

### 3.1 问题

原来的 clamp 直接作用在 k(t) 上：

$$k_{\text{eff}}(t) = \min\left(\frac{\dot{\kappa}}{1-\kappa}, C\right)$$

做变量替换时：

$$\int \min\left(\frac{\dot{\kappa}}{1-\kappa}, C\right) \cdot \ell \, dt = \int \min\left(\frac{1}{1-s}, \frac{C}{\dot{\kappa}}\right) \cdot \ell \, ds$$

**κ̇ 留在了分母里**，不同的 $\dot{\kappa}$ 导致不同 scheduler 在 clamped 区域有不同的损失权重。

### 3.2 解决方案

将 clamp 移到 $1/(1-\kappa)$ 上（通过 `clamp_kappa: true` 配置）：

$$k_{\text{eff}}(t) = \dot{\kappa} \cdot \min\left(\frac{1}{1-\kappa}, C\right)$$

变量替换后：

$$\int \dot{\kappa} \cdot \min\left(\frac{1}{1-\kappa}, C\right) \cdot \ell \, dt = \int \min\left(\frac{1}{1-s}, C\right) \cdot \ell \, ds$$

**κ̇ 完美消掉。** 所有 scheduler 在 κ 空间完全相同。

此时 $C$ 直接控制 $1/(1-\kappa)$ 的上限。对 Cubic scheduler（$\dot{\kappa} \to 3$），$C=17$ 时有效 k_max ≈ 51，与旧行为一致。

---

## 4. 实现改动

### 4.1 新增配置项

```yaml
# retro.yaml 新增项
sample_scheduler: cubic     # 采样时使用的 scheduler（默认等于 scheduler）
time_input: t               # 模型时间输入："t"（原始时间）或 "kappa"（κ(t)）
clamp_kappa: false          # true：clamp 在 1/(1-κ) 上；false：clamp 在 full k(t) 上
clamp_max: 50.0             # clamp 上限
```

### 4.2 新增/修改的核心函数

**`KappaScheduler.inverse(kappa)`**（`scheduler.py`）
```python
# CubicScheduler:  κ = t³  →  t = κ^(1/3)
# LinearScheduler: κ = t   →  t = κ
```

**`_compute_model_time(t, sample_scheduler, time_input, train_scheduler)`**（`euler.py`）

计算传给模型的 t 值：
```python
if time_input == "kappa":
    return sample_scheduler(t)       # 模型看到 κ
elif train_scheduler is None or same_type(sample, train):
    return t                          # 同 scheduler，直接传 t
else:
    kappa = sample_scheduler(t)
    return train_scheduler.inverse(kappa)  # t 映射
```

**`get_rate_scale(t, scheduler, clamp_kappa, clamp_max)`**（`rate_scale.py`）
```python
if clamp_kappa:
    k = κ̇ · min(1/(1-κ), clamp_max)    # 新模式：κ̇ 可消解
else:
    k = min(κ̇/(1-κ), clamp_max)        # 旧模式（默认）
```

**速率校正**（`euler.py` 中 `sample_euler` 内）

当 `use_rate_reparam=False` 且跨 scheduler 采样时：
```python
k_sample = get_rate_scale(t, scheduler, ...)
k_train = get_rate_scale(t_model, train_scheduler, ...)
log_rates = log_rates + log(k_sample / k_train)
```

### 4.3 触及的文件

| 文件 | 改动 |
|------|------|
| `edit_flows/core/scheduler.py` | 新增 `inverse()` 抽象方法，`CubicScheduler` / `LinearScheduler` 实现 |
| `edit_flows/core/rate_scale.py` | `get_rate_scale()` 新增 `clamp_kappa`；`apply_rate_parameterization()` 透传 |
| `edit_flows/training/loss.py` | `bregman_loss()` 透传 `clamp_kappa`, `clamp_max` |
| `edit_flows/training/trainer.py` | `train_step()` 新增 `time_input`, `clamp_kappa`, `clamp_max` |
| `edit_flows/sampling/euler.py` | 新增 `_compute_model_time()`；`sample_euler()` 新增 `time_input`, `train_scheduler`, 速率校正 |
| `edit_flows/sampling/oracle.py` | 改用 `get_rate_scale`，支持 `clamp_kappa` |
| `scripts/train_retro.py` | 读取新配置并传入 `train_step` |
| `scripts/sample_retro.py` | 新增 `--scheduler` CLI；构建 train/sample 两个 scheduler 并传入 `sample_euler` |
| `scripts/eval_retro.py` | 新增 `--scheduler` CLI，透传至 `sample_retro.py` |
| `scripts/oracle_sample.py` | 已有 `--scheduler`，新增 `--clamp_kappa`, `--clamp_max` |
| `scripts/oracle_loss_profile.py` | 改用 `get_rate_scale`，支持新配置 |
| `configs/retro.yaml` | 新增 `sample_scheduler`, `time_input`, `clamp_kappa`, `clamp_max` |
| `configs/retro-example.yaml` | 同上 |

---

## 5. 向后兼容

所有新参数都有兼容默认值：
- `time_input: "t"` — 行为与旧版完全相同
- `clamp_kappa: false` — 使用旧 clamp 公式
- `clamp_max: 50.0` — 旧默认值
- `sample_scheduler` — 默认等于 `scheduler`

### 5.1 兼容矩阵

| 场景 | 配置 | 行为 |
|------|------|------|
| **旧 config 直接跑训练** | 不加任何新字段 | 完全不变 |
| **旧 checkpoint 采样** | `--checkpoint <old.pt>`，不加 `--scheduler` | 使用 checkpoint 中的原始 `scheduler`，行为不变 |
| **旧 checkpoint + 新 scheduler 采样** | `--scheduler linear` | 自动做 t 映射 + (如需要) 速率校正 |
| **旧 checkpoint + 新 scheduler 采样 + rate_reparam** | 配置已有 `use_rate_reparam: true` | 只做 t 映射，无需速率校正 |
| **新训练 (`time_input: "kappa"`)** | `time_input: kappa` | 模型直接学 κ 输入，跨 scheduler 采样零额外开销 |
| **新训练 (`clamp_kappa: true`)** | `clamp_kappa: true, clamp_max: 17.0` | 训练损失在 κ 空间严格 scheduler-invariant |

### 5.2 典型实验命令

```bash
# 旧 Cubic 模型，用 Linear scheduler 采样（自动 t 映射 + 速率校正）
PYTHONPATH=. python scripts/sample_retro.py \
    --checkpoint checkpoints/.../checkpoint_step1000000.pt \
    --products_file test/src-test.txt \
    --scheduler linear --n_samples 10 --device cuda

# Oracle 实验：Linear scheduler + κ 空间 clamp
PYTHONPATH=. python scripts/oracle_sample.py \
    --products_file test/src-test.txt --targets_file test/tgt-test.txt \
    --vocab_file /path/to/example.vocab.src \
    --output_dir eval/oracle_linear --n_samples 10 --n_steps 100 \
    --scheduler linear --clamp_kappa --clamp_max 17.0

# 端到端评测（带 scheduler 覆盖）
PYTHONPATH=. python scripts/eval_retro.py \
    --checkpoint checkpoints/.../checkpoint_step1000000.pt \
    --scheduler linear --n_samples 10 --device cuda
```

---

## 6. 总结

跨 scheduler 采样的所有问题都可以在采样侧解决，不需重新训练：

1. **时间映射**（$t \to \kappa \to t_{\text{model}}$）：修复模型输入分布偏移
2. **速率校正**（$v \cdot k_{\text{sample}} / k_{\text{train}}$）：修复 `use_rate_reparam=false` 时的速率缩放
3. **Clamp 重参数化**（`clamp_kappa: true`）：让训练损失对所有 scheduler 严格等价

对于未来的训练，推荐使用 `time_input: "kappa"` + `clamp_kappa: true`，使训练与 scheduler 选择完全解耦，此后切换 scheduler 只需改配置，无需任何映射或校正。

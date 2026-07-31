# 速率重参数化：公式与最终实现总结

## 1. 背景

当前 Edit Flows 训练目标中，时间相关系数为：

$$
k(t) = \frac{\dot{\kappa}_t}{1-\kappa_t}
$$

实际实现中还会做 clamp：

$$
k_{\text{eff}}(t) = \min\left(\frac{\dot{\kappa}_t}{1-\kappa_t+\varepsilon}, 50\right)
$$

在原始参数化下，模型直接预测真实 CTMC 速率 $v$。这意味着模型既要学习：

- **内容相关部分**：当前位置是否需要编辑、该编辑的类别和 token
- **时间尺度部分**：随着 $t \to 1$，速率如何被 $k(t)$ 放大

为了解耦这两部分，引入速率重参数化。

---

## 2. 原始 loss

对单个位置/单个编辑通道，设：

- $u$：目标“基准速率”或监督信号
- $v$：模型预测的真实速率
- $k$：由 scheduler 决定的时间系数

原始 loss 可以写成：

$$
L(v) = v - k u \log v
$$

对整条序列、所有编辑通道求和后，就是当前实现中的 Bregman-style loss：

$$
L = u_{\text{tot}} - CE
$$

其中：

$$
u_{\text{tot}} = \sum v
$$

$$
CE = k \sum \log v_{\text{correct}}
$$

对应代码中的非重参数化分支：

- [loss.py](/data3/duanbh/desktop/edit-flows/edit_flows/training/loss.py)

```python
u_tot = ux_cat.sum(dim=(1, 2))
ce_term = (log_uz_cat * uz_mask * sched_coeff.unsqueeze(-1)).sum(dim=(1, 2))
per_sample_loss = u_tot - ce_term
```

---

## 3. 重参数化推导

令模型不再直接输出真实速率 $v$，而是输出一个**基准速率** $v'$：

$$
v = k v'
$$

代入原始 loss：

$$
L(v') = k v' - k u \log(k v')
$$

展开：

$$
L(v') = k v' - k u \log k - k u \log v'
$$

整理得：

$$
L(v') = -k u \log k + k (v' - u \log v')
$$

其中：

$$
-k u \log k
$$

是与模型参数无关的常数项。因此训练时可以省掉，只优化：

$$
\tilde{L}(v') = k (v' - u \log v')
$$

推广到整条序列后，最终采用的重参数化训练目标是：

$$
\tilde{L} = k \left(u'_{\text{tot}} - CE'\right)
$$

其中：

$$
u'_{\text{tot}} = \sum v'
$$

$$
CE' = \sum \log v'_{\text{correct}}
$$

注意：这里 $k$ 是**每个样本自己的时间系数**，不是 batch 共享常数。

---

## 4. 为什么最终采用“每样本乘 k”的实现

这次实现过程中有两种可能的写法：

### 写法 A：先把 $v'$ 还原为 $v$

先在 `train_step` 中做：

$$
\log v = \log v' + \log k
$$

然后把还原后的真实速率 $v$ 传给原始 loss。

### 写法 B：训练时直接对 $v'$ 计算 loss

不在 `train_step` 中提前乘回 $k$，而是在 loss 内部直接计算：

$$
\tilde{L} = k (u'_{\text{tot}} - CE')
$$

最终采用的是 **写法 B**，原因有三点：

1. **语义更清楚**
   - 模型输出始终解释为 base rate $v'$
   - 训练目标直接对应去掉常数项后的重参数化目标

2. **loss 更容易分析**
   - 不再混入常数项 $-k u \log k$
   - 不同实验间的 loss 曲线更容易解释

3. **负值更少**
   - 原始 loss 中较大的负值有一部分来自常数项
   - 去掉常数项后，数值分布通常更平稳

---

## 5. 最终训练实现

### 5.1 scheduler 系数

统一在：

- [rate_scale.py](/data3/duanbh/desktop/edit-flows/edit_flows/core/rate_scale.py)

中定义：

```python
def get_rate_scale(t, scheduler, clamp_max=50.0, eps=1e-8):
    scale = scheduler.derivative(t) / (1 - scheduler(t) + eps)
    return torch.clamp(scale, max=clamp_max)
```

这对应：

$$
k_{\text{eff}}(t) = \min\left(\frac{\dot{\kappa}_t}{1-\kappa_t+\varepsilon}, 50\right)
$$

### 5.2 loss 分支

最终 `bregman_loss()` 的两个分支为：

#### 非重参数化

```python
u_tot = ux_cat.sum(dim=(1, 2))
ce_term = (log_uz_cat * uz_mask * sched_coeff.unsqueeze(-1)).sum(dim=(1, 2))
per_sample_loss = u_tot - ce_term
```

对应：

$$
L = u_{\text{tot}} - k \sum \log v_{\text{correct}}
$$

#### 重参数化

```python
u_tot = ux_cat.sum(dim=(1, 2))
ce_term = (log_uz_cat * uz_mask).sum(dim=(1, 2))
per_sample_loss = sched_coeff.squeeze(-1) * (u_tot - ce_term)
```

对应：

$$
\tilde{L} = k \left(u'_{\text{tot}} - CE'\right)
$$

其中：

- `ux_cat = exp(log_ux_cat)` 表示 base rate $v'$
- `log_uz_cat` 也是 base rate 的 log-space 映射
- `ce_term` 前面不再额外乘 `k`，而是在最后整体乘每个样本自己的 `k`

### 5.3 train_step 中不提前乘回 k

最终训练路径中，`train_step()` 保持：

```python
log_rates, log_ins_probs, log_sub_probs = model(x_t, t, x_pad_mask)
```

随后直接构造 base-rate 版本的 `log_ux_cat`，不做：

```python
log_rates_eff = apply_rate_parameterization(...)
```

这保证了训练时 `bregman_loss()` 看到的就是 $v'$ 而不是 $v$。

实现位置：

- [trainer.py](/data3/duanbh/desktop/edit-flows/edit_flows/training/trainer.py)

---

## 6. 最终采样实现

训练时模型输出的是 $v'$，但 **Euler 采样必须使用真实 CTMC 速率 $v$**。因此在采样时需要乘回时间系数：

$$
v = k v'
$$

对应代码：

- [euler.py](/data3/duanbh/desktop/edit-flows/edit_flows/sampling/euler.py)

```python
log_rates, log_ins_probs, log_sub_probs = model(x_t, t, x_pad_mask)
log_rates = apply_rate_parameterization(
    log_rates, t, scheduler, use_rate_reparam=use_rate_reparam,
)
rates = torch.exp(log_rates)
```

这里：

```python
apply_rate_parameterization(log_base_rates, t, scheduler, use_rate_reparam=True)
```

会执行：

$$
\log v = \log v' + \log k
$$

从而恢复真实速率。

因此最终语义是：

- **训练时**
  - `use_rate_reparam=false`：模型输出真实速率 $v$
  - `use_rate_reparam=true`：模型输出 base rate $v'$

- **采样时**
  - `use_rate_reparam=false`：直接使用模型输出
  - `use_rate_reparam=true`：先乘回 $k(t)$，再作为真实速率执行 CTMC 采样

---

## 7. 配置开关

新增配置项：

- [retro.yaml](/data3/duanbh/desktop/edit-flows/configs/retro.yaml)
- [retro-example.yaml](/data3/duanbh/desktop/edit-flows/configs/retro-example.yaml)
- [default.yaml](/data3/duanbh/desktop/edit-flows/configs/default.yaml)

```yaml
use_rate_reparam: true
```

含义：

- `false`：使用原始参数化，模型直接预测真实速率 $v$
- `true`：使用重参数化，模型预测 base rate $v'$，训练时优化
  $$
  k(u'_{\text{tot}} - CE')
  $$
  ，采样时恢复真实速率 $v = kv'$

`train_retro.py` 与 `sample_retro.py` 都直接读取该参数，因此训练与生成可以保持一致。

---

## 8. 日志指标的解释

在 `use_rate_reparam=true` 时：

- 训练 loss 是 **base-rate 重参数化 loss**
- 但日志中的 `u_tot / u_ins / u_del / u_sub` 仍然统计的是**乘回 $k$ 之后的有效真实速率**

这是通过 `train_step()` 末尾的：

```python
log_rates_eff = apply_rate_parameterization(...)
```

完成的。

这样做的目的：

1. 保持与旧实验的 `u_tot` 可比
2. 让训练日志反映真实采样时会使用的速率量级

因此要注意：

- `loss` 与 `u_tot` 在 `use_rate_reparam=true` 时不再对应同一个参数化空间
- `loss` 用于优化分析
- `u_tot` 用于和旧实验、采样行为做量级对比

---

## 9. 结论

这次速率重参数化的最终形式为：

### 模型输出

$$
v'
$$

### 训练目标

$$
\tilde{L} = k(t)\left(u'_{\text{tot}} - CE'\right)
$$

其中去掉了与模型参数无关的常数项：

$$
-k u \log k
$$

### 采样速率

$$
v = k(t) v'
$$

最终实现的核心特点：

1. **内容与时间尺度解耦**
2. **训练 loss 更平稳、更易分析**
3. **采样时仍严格使用真实 CTMC 速率**
4. **通过 `use_rate_reparam` 配置开关可与原始实现直接对比**

---

## 10. 相关文件

| 文件 | 作用 |
|------|------|
| [rate_scale.py](/data3/duanbh/desktop/edit-flows/edit_flows/core/rate_scale.py) | 统一定义 $k(t)$ 与 log-space 乘回 |
| [loss.py](/data3/duanbh/desktop/edit-flows/edit_flows/training/loss.py) | 重参数化 loss 的最终实现 |
| [trainer.py](/data3/duanbh/desktop/edit-flows/edit_flows/training/trainer.py) | 训练路径：不提前乘回 $k$，日志中统计有效真实速率 |
| [euler.py](/data3/duanbh/desktop/edit-flows/edit_flows/sampling/euler.py) | 采样路径：恢复真实速率 $v = kv'$ |
| [retro.yaml](/data3/duanbh/desktop/edit-flows/configs/retro.yaml) | 通过 `use_rate_reparam` 控制开关 |

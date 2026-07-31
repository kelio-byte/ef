# Fix 1: BOS 训练-推理不一致 & 采样过度编辑

## 问题诊断

结合理论文档 `docs/edit-flows.md` 审查实现，发现两个明确 bug。

### Bug 1: BOS token 训练-推理不一致

**训练路径** (`edit_flows/training/trainer.py:36-37`)：

```python
z_0_padded[:, 0] = bos_token
z_1_padded[:, 0] = bos_token
```

`prepare_batch` 在 Z 空间序列前插入 BOS_TOKEN,使得 `x_t`（经 `rm_gap_tokens` 后）始终以 BOS 开头。模型位置 0 的 embedding 和编辑速率被训练用于处理 Z 空间中 BOS 之后的 GAP 插入（即第一条产物 token 之前的插入）。

**推理路径** (`scripts/sample_retro.py:33-41`，修复前）：

```python
x_0 = torch.full((B, max_len), pad_token, ...)
for i, ids in enumerate(product_ids):
    x_0[i, :len(ids)] = torch.tensor(ids, ...)
```

产物 SMILES token 直接作为 `x_0`，无 BOS 前缀。模型位置 0 看到的是第一个产物 token（如 `C`），而非训练时的 BOS，导致：

- 位置 0 的 embedding 与训练分布不匹配，编辑速率预测失准
- 任何需要在产物开头之前插入 token 的情况被遗漏

**修复**：`_make_batch` 在序列开头插入 `BOS_TOKEN`：

```python
x_0 = torch.full((B, max_len + 1), pad_token, ...)
x_0[:, 0] = bos_token
for i, ids in enumerate(product_ids):
    x_0[i, 1:1 + len(ids)] = torch.tensor(ids, ...)
```

`_ids_to_str` 已有 BOS/PAD 过滤逻辑，输出中不会残留 BOS。

### Bug 2: 已完成元素在 Euler 采样中被过度编辑

**问题** (`edit_flows/sampling/euler.py:47`)：

```python
while (t < 1.0).any():
```

循环条件为「batch 中任意样本未完成则继续」，但 `adapt_h` 是 per-sample 的。不同样本可能在不同步数到达 `t >= 1.0`，已完成的样本在后续步数中仍被应用编辑操作。

**修复**：在编辑 mask 计算后，对 `t >= 1.0` 的样本将 mask 置为 False：

```python
done = (t >= 1.0).squeeze(-1)
if done.any():
    ins_mask[done] = False
    del_sub_mask[done] = False
```

## 修改文件

| 文件 | 修改 |
|------|------|
| `scripts/sample_retro.py:33-41` | `_make_batch` 添加 BOS 前缀 |
| `edit_flows/sampling/euler.py:71-74` | 对已完成样本屏蔽编辑 mask |

## 验证

修复后运行冒烟测试（batch_size=2, n_samples=1），采样无报错，输出 SMILES 不含 BOS 残留。随后启动两个 checkpoint 的完整评测在后台运行。

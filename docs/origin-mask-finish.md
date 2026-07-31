# Origin Mask：实现总结与实验指南

## 1. 动机

当前模型看到 `x_t` 时，并不知道每个 token 是从 `x_0`（原始产物）来的还是从 0~t 之间的编辑过程来的。但这一信息在训练和采样过程中都是**精确已知的**：

- 训练时：`z_t` 的每个位置是否仍携带 `z_0` 中的原始 token，可以由 `z_t` 与 `z_0` 直接比较得到
- 采样时：每次 substitute / insert / delete 都明确改变了哪些位置

直觉上，这个信息可能帮助模型更好地学习编辑速率——例如，已经被编辑过的位置（已到达 target 状态）通常不需要再编辑，而仍来自 `x_0` 的位置可能需要继续编辑。

## 2. 实现方案

### 2.1 核心思路

为 `x_t` 中每个 token 位置新增一个二值标记：

- `1`（True）= 来自 `x_0`（原始产物 token，尚未被编辑过）
- `0`（False）= 已被编辑过（来自扩散过程中的 insert / substitute 等操作）

该标记以 **learnable embedding** 形式注入到 token embedding 层，与 token embedding、time embedding、position encoding 求和后送入 Transformer。

### 2.2 训练侧 (`prepare_batch`)

```
z_t[j] == z_0[j] 且 z_0[j] != GAP  → 原始   → origin_mask = True
否则                               → 已编辑 → origin_mask = False
```

这里**不能**直接用 `~pick_z1` 作为 `origin_mask`。原因是当 `z_0[j] == z_1[j]` 时，即使该位置实际上从未被编辑，`pick_z1` 仍然会随机取 `True/False`，从而把 unchanged token 随机标成 edited，造成训练/采样语义不一致。

当前实现做法是：

1. `sample_cond_zt(..., return_pick=True)` 仍返回 `pick_z1`，用于保留路径采样信息
2. `origin_mask_z` 不再由 `pick_z1` 生成，而是直接定义为：

```python
origin_mask_z = (z_t == z_0_padded) & (z_0_padded != GAP_TOKEN)
```

3. 再通过 `project_mask_z_to_x` 投影到 X 空间（与 `rm_gap_tokens` 相同的 `dest_col` 逻辑），得到与 `x_t` 同形的 `origin_mask`

这样 BOS 和所有 unchanged 位置都会稳定保持 `True`，与采样侧“只要没真的经历 edit，就仍是 original”的语义一致。

### 2.3 采样侧 (`sample_euler`)

```
t=0:   origin_mask = all True（全部来自 x_0）

每步：
  substitute at j  → origin_mask[j] = False
  insert new token → origin_mask 对应位置 = False
  delete token     → 随 token 一起删掉
```

Insert/delete 的空间变换复用现有 `apply_ins_del_operations`（将 `origin_mask` 转为 long tensor 传入，`ins_tokens` 全填 0 表示 False）。

实现上还有一个容易踩坑的点：不能把 `origin_mask.long()` 直接传给 `apply_ins_del_operations`，因为 `0` 会同时表示：

- edited (`False`)
- PAD

这会在 insert/delete/replace 后把长度计算搞乱。当前实现改为先转成三值 marker：

- `1` = original
- `0` = edited
- `2` = pad

再调用 `apply_ins_del_operations(..., pad_token=2)`，最后恢复成布尔 `origin_mask = (marker == 1)`。

### 2.4 模型侧 (`EditFlowsTransformer`)

```python
# __init__
if use_origin_mask:
    self.origin_embedding = nn.Embedding(2, hidden_dim)

# forward
if origin_mask is not None:
    token_emb = token_emb + self.origin_embedding(origin_mask.long())
```

Embedding(2, hidden_dim) 只有 2×hidden_dim 个参数（约 512 for hidden_dim=256），几乎不增加模型容量。

## 3. 修改的文件

| 文件 | 改动 |
|------|------|
| `edit_flows/core/z_space.py` | `sample_cond_zt` 新增 `return_pick`；新增 `project_mask_z_to_x` |
| `edit_flows/core/__init__.py` | 导出 `project_mask_z_to_x` |
| `edit_flows/__init__.py` | 导出 `project_mask_z_to_x` |
| `edit_flows/models/transformer.py` | 新增 `use_origin_mask` 参数、`origin_embedding`、`forward` 接受 `origin_mask` |
| `edit_flows/models/interface.py` | Protocol `forward` 签名新增 `origin_mask=None` |
| `edit_flows/training/trainer.py` | `prepare_batch` 计算 origin_mask；`train_step` 透传至模型 |
| `edit_flows/sampling/euler.py` | `sample_euler` 初始化并追踪 origin_mask，透传至模型 |
| `scripts/train_retro.py` | 透传 `use_origin_mask` 到模型构造和 `prepare_batch` |
| `scripts/sample_retro.py` | 透传 `use_origin_mask` 到模型构造和 `sample_euler` |
| `configs/retro.yaml` | 新增 `use_origin_mask: false` |
| `configs/retro-example.yaml` | 新增 `use_origin_mask: false` |
| `tests/conftest.py` | `DummyModel.forward` 签名更新 |
| `tests/test_integration.py` | `SmallMLP.forward` 签名更新 |
| `tests/core/test_z_space.py` | 新增 `test_return_pick`、`TestProjectMaskZtoX` 测试 |
| `tests/training/test_trainer.py` | 新增 `prepare_batch` 的 origin-mask 语义测试 |
| `tests/sampling/test_euler.py` | 新增 substitute / insert / delete / replace 下的 origin-mask 演化测试 |

## 4. 配置与使用

### 4.1 配置开关

```yaml
# configs/retro.yaml
use_origin_mask: true   # 开启 origin mask 实验（默认 false）
```

### 4.2 训练

```bash
# 使用默认配置（含 use_origin_mask: true）
PYTHONPATH=. python scripts/train_retro.py \
    --config configs/retro.yaml --device cuda

# 或通过命令行覆盖（如果配置中未设置）
# 当前通过 yaml 配置控制，不提供 CLI 覆盖
```

训练时模型会额外看到每个 token 是 "original" 还是 "edited"。

### 4.3 采样

```bash
PYTHONPATH=. python scripts/sample_retro.py \
    --checkpoint checkpoints/.../checkpoint_step5000000.pt \
    --products_file test/src-test.txt \
    --n_samples 10 --device cuda
```

采样脚本从 checkpoint 配置中读取 `use_origin_mask`，自动决定是否追踪 origin_mask 并传给模型。**无需额外 CLI 参数**。

### 4.4 评测

```bash
PYTHONPATH=. python scripts/eval_retro.py \
    --checkpoint checkpoints/.../checkpoint_step5000000.pt \
    --n_samples 10 --device cuda
```

同正常流程，透传至 `sample_retro.py`。

## 5. 向后兼容

| 场景 | 行为 |
|------|------|
| 旧 config / 旧 checkpoint（无 `use_origin_mask` 字段） | 默认 `false`，模型不创建 `origin_embedding`，行为完全不变 |
| 新训练 `use_origin_mask: true` | 模型创建 embedding（+512 参数），checkpoint 保存此配置 |
| 新 checkpoint 采样 | 自动从 checkpoint 配置读取，追踪 origin_mask 并传给模型 |
| 新 checkpoint 用旧版代码采样 | 旧版 `sample_euler` 不接受 `use_origin_mask` → 报错（不应出现） |

**注意**：`use_origin_mask` 是**训练时架构选择**，不能在采样时临时开启或关闭。checkpoint 中的配置决定了模型是否有 `origin_embedding` 权重。

## 6. 建议的实验方案

### 6.1 核心对比实验

**A/B test**：相同配置下，仅改变 `use_origin_mask`：

| 实验 | `use_origin_mask` | 其他配置 |
|------|-------------------|----------|
| Baseline | `false` | 当前最佳配置（`use_rate_reparam: true` 等） |
| +Origin Mask | `true` | 同上，其余不变 |

**训练规模**：建议先在 `train_subsets` 上小规模训练（如 total_steps=500k），快速暴露差异。

### 6.2 评测指标

- Standard Top-1 / Top-10 accuracy
- `#global#` Top-1 / Top-10 accuracy
- Invalid SMILES rate
- Unique rate
- 与 oracle gap 的对比（越小越好）

### 6.3 观察要点

1. **Loss 曲线**：origin mask 组是否收敛更快或最终 loss 更低
2. **u_tot 分布**：origin mask 是否让模型对 original vs edited 位置输出更分化的速率
3. **生成质量**：Top-1 是否有显著抬升
4. **与 rate reparam 的交互**：当前默认 `use_rate_reparam: true`，origin mask 在此基础上的增量效果

### 6.4 可能的后续分析

如果 origin mask 有帮助，可以进一步分析：

- 对 original / edited 位置分别统计模型输出的速率分布
- 观察模型是否学会了 "edited → 低速率、original → 高速率" 的模式
- 如果效果不明显，考虑将 origin mask 信息以更强的方式注入（如 attention bias）

## 7. 设计决策与权衡

1. **Embedding 而非单向量**：用 `Embedding(2, hidden_dim)` 给 original 和 edited 各一个可学习向量，比单个 `original_marker` 向量更灵活（参数仅多 256 个，可忽略）
2. **注入层级**：仅注入到 token embedding 求和层，不改动 attention 结构。保持模型主体不变，最小化干扰
3. **训练/采样一致性**：训练时的 origin_mask 必须定义为“该 token 是否仍为原始产物 token”，而不是“这一步混合采样是否选中了 `z_1` 分支”。前者才能与采样时的实际编辑历史一致
4. **不参与 loss**：origin_mask 仅作为模型输入特征，不参与 loss 计算（loss 仍为 Bregman divergence）

## 8. 修正记录

在初版实现中，有两个后来修正的问题：

1. **训练侧语义错误**：最初使用 `origin_mask_z = ~pick_z1`。这会让 `z_0 == z_1` 的 unchanged 位置随机变成 edited。现已改为基于 `(z_t == z_0)` 的定义。
2. **采样侧 PAD 冲突**：最初直接把 `origin_mask.long()` 送入 `apply_ins_del_operations`，导致 `False=0` 与 `PAD=0` 冲突。现已改为三值 marker 追踪。

当前测试已显式约束这两个行为：

- identical `x_0 / x_1` 时，训练侧 `origin_mask` 必须全为 `True`
- 插入得到的新 token 必须是 `False`
- substitute / insert / delete / replace 后，采样侧 `origin_mask` 必须与真实编辑历史一致

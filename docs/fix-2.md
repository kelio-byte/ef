# Fix 2: Pre-Norm 残差修复与词表辅助函数清理

## 问题背景

在对照原始论文 [Edit Flows: Flow Matching with Edit Operations](https://arxiv.org/abs/2506.09018)、现有实现文档以及代码后，定位到两个需要优先修复的问题：

1. `PreNormEncoderLayer` 的残差连接实现不符合标准 Pre-LN Transformer 写法
2. `utils/tokens.py` 中仍残留旧的 `+3` 词表大小辅助函数，与当前 4 个特殊 token 约定不一致

其中第 1 项属于高优先级问题，因为它直接影响模型的表达能力、梯度路径和训练稳定性；第 2 项属于中优先级问题，虽然当前主流程未直接依赖错误返回值，但它反映出代码里仍有旧设定遗留，后续很容易再次引入不一致。

---

## Bug 1: Pre-Norm 残差连接写错

### 原实现

文件：`edit_flows/models/transformer.py`

修复前的核心逻辑是：

```python
x = self.norm1(src)
x = x + self.dropout1(self.self_attn(x, x, x, ...)[0])
x = x + self.dropout2(self.linear2(...self.norm2(x)...))
```

这会导致第一条残差分支的基底从原始输入 `src` 变成了 `norm1(src)`。

### 为什么这是问题

标准的 Pre-LN Transformer encoder block 应该是：

```python
x = src + Attn(LN(src))
x = x + FFN(LN(x))
```

也就是说：

- LayerNorm 只应该作用在子层输入上
- 残差分支本身必须保留原始未归一化的 `src`

修复前的写法实际上改变了残差路径，不再是标准 Pre-LN 结构。这不是简单的“实现风格差异”，而是模型函数形式发生了变化，可能带来：

- 残差直通路径被破坏
- 梯度传播性质变差
- 与 OpenNMT 风格的目标结构不一致
- 训练可以继续跑，但学到的表示更差

这类问题足以解释“loss 看起来可训练，但生成质量明显不佳”的一部分现象。

### 修复方案

改为标准 Pre-LN 残差：

```python
x = src + self.dropout1(
    self.self_attn(self.norm1(src), self.norm1(src), self.norm1(src), ...)[0]
)
x = x + self.dropout2(
    self.linear2(self.dropout(self.activation(self.linear1(self.norm2(x)))))
)
```

### 修改位置

- `edit_flows/models/transformer.py`

---

## Bug 2: 词表辅助函数仍使用旧的 `+3` 约定

### 原实现

文件：`edit_flows/utils/tokens.py`

修复前：

```python
def model_vocab_size(real_vocab_size: int) -> int:
    return real_vocab_size + 3

def z_vocab_size(real_vocab_size: int) -> int:
    return real_vocab_size + 3
```

但当前项目已经明确使用 4 个特殊 token：

- `PAD_TOKEN = 0`
- `BOS_TOKEN = 1`
- `GAP_TOKEN = 2`
- `UNK_TOKEN = 3`

因此完整词表大小应为：

```python
real_vocab_size + 4
```

### 为什么这是问题

当前训练/采样主流程多数地方已经直接使用 `load_vocab()` 返回的 `model_vocab`，所以这个错误未必直接破坏现有实验结果。但它仍然是潜在风险：

- 辅助函数含义与真实 token 约定不一致
- 后续新脚本若误调用这些函数，可能再次引入 off-by-one / off-by-four 问题
- 文档、测试和代码之间的口径会继续分裂

### 修复方案

统一改为：

```python
def model_vocab_size(real_vocab_size: int) -> int:
    return real_vocab_size + 4

def z_vocab_size(real_vocab_size: int) -> int:
    return real_vocab_size + 4
```

### 修改位置

- `edit_flows/utils/tokens.py`

---

## 新增测试

为了避免这两个问题以后再次回归，补充了两个针对性测试。

### 1. Pre-Norm 残差语义测试

文件：`tests/models/test_transformer.py`

思路：

- 将 attention 输出强制为 0
- 将 FFN 权重清零
- 此时整个 block 的输出应严格等于输入

如果实现仍是错误的 `norm(src) + ...` 路径，那么输出不会等于原始 `src`，测试会失败。

### 2. 词表辅助函数测试

文件：`tests/utils/test_tokens.py`

检查：

- `pad_token_id / bos_token_id / gap_token_id` 返回值与常量一致
- `model_vocab_size(68) == 72`
- `z_vocab_size(68) == 72`

---

## 验证结果

本次修复后，运行了与改动直接相关的测试和一组轻量回归测试。

### 定向测试

```bash
pytest -q tests/models/test_transformer.py tests/utils/test_tokens.py
```

结果：

```text
3 passed
```

### 回归测试

```bash
pytest -q tests/test_integration.py tests/sampling/test_euler.py tests/core/test_z_space.py
```

结果：

```text
21 passed
```

合计：

```text
24 passed
```

说明本次修改没有破坏已有的基础训练、Z 空间处理和 Euler 采样路径。

---

## 当前结论

这次修复解决了两个明确问题：

1. 模型主干中的 Pre-LN 残差路径错误
2. 特殊 token 数量与词表辅助函数定义不一致

其中第 1 项更重要，因为它是一个真正会影响模型学习能力的结构性错误。修复后，后续若训练效果仍不理想，就更有理由把注意力集中到 loss、速率参数化、采样离散化和 retro 任务适配本身，而不是继续怀疑 Transformer 主干写法。

---

## 仍可能遗留的潜在问题

虽然这次修复了中高优先级问题，但目前仍有几类潜在问题没有被完全排除。

### 1. Bregman loss 的尺度项是否与论文完全一致

当前实现：

- `u_tot = sum(exp(log_ux_cat))`
- `ce_term = sched_coeff * log u(correct)`

见 `edit_flows/training/loss.py`。

这套实现与现有 `oracle_loss_profile.py` 和 `oracle_sample.py` 的分析是内部自洽的，但是否与论文中的严格公式、记号缩放和训练目标完全一致，仍值得再做一次逐式核对。

潜在风险：

- 若 `u_tot` / `ce_term` 的时间系数缩放存在偏差，模型可能会学到“总速率量级差不多，但正确编辑分布不够尖锐”的次优解

可能方向：

- 对照论文公式逐项检查 `u_tot` 是否也应带时间系数或其他常数项
- 将论文记号、代码变量、oracle 推导写成同一份对照表

### 2. 速率头参数化可能仍偏保守

当前模型使用：

- `log_rates = log(softplus(raw_rates))`
- `log_ins_probs = log_softmax(ins_logits)`
- `log_sub_probs = log_softmax(sub_logits)`

这会让模型天然更倾向于输出平滑分布。结合你已有实验，模型与 oracle 的主要差距更像是“正确编辑不够尖锐”，而不是“完全不知道该编辑哪里”。

潜在风险：

- 无关编辑压不够低
- 正确 token 的替换/插入概率不够集中

可能方向：

- 尝试对 `ins/sub` logits 引入 temperature
- 尝试替换或调整 rate head 参数化方式
- 观察 CE term 是否能更接近 oracle

### 3. Euler 离散化误差仍然构成方法下界

你已有 oracle 实验已经说明，即使使用最优速率，`standard` 上仍有约 22.6% invalid SMILES，`#global#` 上约 9.0%。

这部分不是模型 bug，而是当前离散化、`clamp=50` 和有限步数共同造成的误差下界。

潜在风险：

- 即便模型学得更好，也会撞上这个下界

可能方向：

- 增加 `n_steps`
- 提高 `clamp`
- 系统比较不同 scheduler

### 4. Retro 任务下的对齐目标仍可能不是最优任务表达

当前采用的是：

- `x_0 = product`
- `x_1 = reactant`
- Z 空间通过 Levenshtein 对齐，再学习局部编辑

oracle 结果说明这条路线是可行的，但不代表它是 retro 上最优的条件表达方式。尤其 standard 与 `#global#` 的显著差距说明：表示方式本身会强烈影响编辑难度。

潜在风险：

- 普通 tokenized SMILES 对齐成本太高
- 某些反应模式需要更结构化的条件表达

可能方向：

- 继续优先使用 `#global#` / root-aligned 表示
- 对比不同字符串表示的平均编辑数、oracle invalid rate、oracle Top-1

---

## 建议的下一步修改顺序

建议按以下顺序推进，而不是同时改很多东西：

1. 基于当前修复版本做小规模复训
2. 先验证 train subset 上的 Top-1、CE term、oracle gap 是否明显改善
3. 若有改善，再继续长训或扩实验
4. 若改善有限，优先深挖 loss 公式与 rate 参数化
5. 最后再集中做采样步数 / clamp / scheduler 的系统比较

这样可以把“架构实现 bug”和“方法本身或训练目标问题”尽量拆开，避免不同因素混在一起。

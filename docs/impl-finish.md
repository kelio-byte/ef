# Edit Flows: 实现总结与使用指南

## 1. 论文背景

Edit Flows 是一种**非自回归**序列生成框架，将序列生成建模为序列空间上的**连续时间马尔可夫链 (CTMC)**，通过三种原子编辑操作——**插入 (insert)、删除 (delete)、替换 (substitute)**——在可变长度序列之间进行传输。

核心训练技巧：引入增广空间 Z（包含 GAP token 的对齐空间），在 Z 空间定义简单的逐 token 条件概率路径，通过 Theorem 3.1 的边际化保证训练梯度正确回传到 X 空间的模型。

### 关键公式

- **条件概率路径**: $p_t(z_t^i|z_0^i, z_1^i) = (1-\kappa_t)\delta_{z_0^i} + \kappa_t\delta_{z_1^i}$，其中 $\kappa_t = t^3$
- **编辑速率**: $u^\theta(\text{ins}(x,i,a)|x) = \lambda^{\text{ins}}_{t,i} \cdot Q^{\text{ins}}_{t,i}(a)$（插入/替换类似，删除只有 $\lambda$）
- **Bregman 散度损失**: $\mathcal{L} = \mathbb{E}[ \sum_y u^\theta - \sum_i \mathbf{1}_{[z_1^i\neq z_t^i]} \cdot \frac{\dot{\kappa}_t}{1-\kappa_t} \cdot \log u^\theta ]$

---

## 2. 项目结构

```
edit-flows/
├── edit_flows/                    # 核心库
│   ├── __init__.py                # 统一导出
│   ├── core/                      # 数学模型（与网络无关）
│   │   ├── scheduler.py           # κ_t 调度器
│   │   ├── coupling.py            # π(x₀,x₁) 耦合分布
│   │   ├── alignment.py           # 序列对齐 (编辑距离DP)
│   │   └── z_space.py             # Z↔X 空间映射 + 条件路径采样
│   ├── models/                    # 模型实现（可插拔）
│   │   ├── interface.py           # 模型接口（Protocol）
│   │   └── transformer.py         # 参考 Transformer 实现
│   ├── training/                  # 训练流程
│   │   ├── loss.py                # Bregman 散度损失
│   │   └── trainer.py             # prepare_batch + train_step
│   ├── sampling/                  # 生成/采样
│   │   ├── ops.py                 # apply_ins_del_operations
│   │   └── euler.py               # Euler-Maruyama 采样循环
│   └── utils/                     # 通用工具
│       ├── tokens.py              # 特殊 token 常量 + 辅助函数
│       └── helpers.py             # x2prob, sample_p, 可视化
├── tests/                         # 测试（镜像结构）
│   ├── conftest.py                # 共享 fixture（DummyModel 等）
│   ├── core/                      # scheduler / coupling / alignment / z_space
│   ├── training/                  # loss / trainer
│   ├── sampling/                  # ops / euler
│   └── test_integration.py        # 端到端集成测试
├── scripts/                       # 入口脚本
│   ├── train.py                   # 训练入口
│   └── sample.py                  # 采样/生成入口
├── configs/                       # 实验配置
│   └── default.yaml               # 默认超参数
└── pyproject.toml                 # 项目元信息 + 依赖
```

---

## 3. 核心模块详解

### 3.1 Token 约定

| Token | ID | 说明 |
|-------|-----|------|
| `PAD_TOKEN` | 0 | 填充标记，模型词表内 |
| `BOS_TOKEN` | 1 | 序列起始标记，模型词表内 |
| `GAP_TOKEN` | 2 | Z 空间对齐标记，**模型不可见** |
| 真实 token | 3, 4, ..., V+2 | 模型词表内 |

模型 `vocab_size = V_real + 3`（PAD + BOS + GAP(未使用但占位) + V_real 个真实 token）。

Z 空间 `x2prob` 使用的 vocab_size 与模型相同（= V_real + 3），GAP=2 在其中合法。

### 3.2 特殊 Token 工具函数 (`utils/tokens.py`)

```python
from edit_flows.utils import PAD_TOKEN, BOS_TOKEN, GAP_TOKEN

# 辅助函数（给定真实 token 数量，计算各 ID 和 vocab_size）
from edit_flows.utils.tokens import bos_token_id, pad_token_id, gap_token_id
from edit_flows.utils.tokens import model_vocab_size, z_vocab_size
```

### 3.3 调度器 (`core/scheduler.py`)

- **`CubicScheduler(a=1.0, b=1.0)`**: $\kappa_t = t^3$，导数 $\dot{\kappa}_t = 3t^2$
- **`LinearScheduler()`**: $\kappa_t = t$
- 实现自己的调度器只需继承 `KappaScheduler`，实现 `__call__` 和 `derivative`

### 3.4 耦合分布 (`core/coupling.py`)

| 类 | 用途 | x₀ 来源 |
|----|------|---------|
| `EmptyCoupling()` | 论文默认 | 空序列 |
| `UniformCoupling(min_len, max_len, vocab_size)` | 去噪 | 均匀随机序列 |
| `GeneratorCoupling(generator_fn)` | seq2seq | 外部函数 |
| `ExtendedCoupling(n_insert, vocab_size)` | 扩展先验 | 在 x₁ 中插入噪声 |

### 3.5 序列对齐 (`core/alignment.py`)

- **`opt_align_xs_to_zs`**: Levenshtein 编辑距离 + DP 回溯，产生最优对齐
- **`naive_align_xs_to_zs`**: 简单右填充 GAP
- **`shifted_align_xs_to_zs`**: x₁ 平移到 x₀ 右侧

### 3.6 Z 空间操作 (`core/z_space.py`)

| 函数 | 作用 | 方向 |
|------|------|------|
| `rm_gap_tokens(z)` | 移除 GAP token，得到 x_t + 各 mask | Z → X |
| `rv_gap_tokens(x, masks)` | 回插 GAP token（`rm_gap_tokens` 逆操作） | X → Z |
| `fill_gap_tokens_with_repeats(ux, masks)` | 将 X 空间速率映射回 Z 空间 | X → Z（训练用） |
| `make_ut_mask_from_z(z_t, z_1)` | 标记 Z 空间中使 z→z₁ 的正确编辑 | 构建训练目标 |
| `sample_cond_zt(z_0, z_1, t, vocab_size, kappa_fn)` | 从条件路径采样 z_t | 训练用 |

### 3.7 模型接口 (`models/interface.py`)

唯一的模型接口约定——`forward` 签名：

```python
def forward(
    self,
    tokens: Tensor,        # (B, L), token IDs
    time_step: Tensor,     # (B, 1), t ∈ [0, 1]
    padding_mask: Tensor,  # (B, L), bool, True=PAD位置
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Returns:
        rates:     (B, L, 3)  — λ_ins, λ_sub, λ_del (正数, softplus)
        ins_probs: (B, L, V)  — Q_ins 插入 token 分布 (softmax)
        sub_probs: (B, L, V)  — Q_sub 替换 token 分布 (softmax)
    """
```

参考实现：`EditFlowsTransformer`（双向 TransformerEncoder + 正弦时间嵌入）。

### 3.8 训练 (`training/`)

**`prepare_batch(x_0, x_1, scheduler, align_fn, vocab_size)`**：完整训练数据准备
1. 对齐 → (z_0, z_1)
2. 添加 BOS 前缀
3. 采样 t ~ Uniform(0, 1)
4. 从条件路径采样 z_t
5. rm_gap_tokens → x_t + masks
6. make_ut_mask_from_z → uz_mask

**`train_step(model, batch_data, scheduler, optimizer)`**：单步训练
1. 模型 forward → rates, ins_probs, sub_probs
2. 拼接 ux_cat = [λ_ins·Q_ins, λ_sub·Q_sub, λ_del]
3. 计算 Bregman loss
4. 反向传播 + 梯度裁剪 (max_norm=1.0)

**`bregman_loss(ux_cat, z_gap_mask, z_pad_mask, uz_mask, t, scheduler)`**：
- 第一项 `u_tot`：压制所有速率（最小编辑偏好）
- 第二项 `-CE_term`：拉高正确编辑速率（方向监督）
- `sched_coeff = κ̇/(1-κ)` 限制为最大 50

### 3.9 采样 (`sampling/`)

**`sample_euler(model, x_0, scheduler, n_steps, max_seq_len)`**：Euler-Maruyama 采样循环
1. 从 x₀ 开始，t=0
2. 每步：模型预测 rates → 计算编辑概率 → 采样编辑 → 并行应用
3. 自适应步长：$h_{adapt} = \min(h, (1-\kappa_t)/\dot{\kappa}_t)$
4. 直到 t ≥ 1

**`apply_ins_del_operations(x_t, ins_mask, del_mask, ins_tokens)`**：并行应用编辑
- 同一位置 ins+del → 替换
- 计算累积偏移确定新位置
- 支持 batch 操作

---

## 4. 使用方法

### 4.1 训练

```bash
python scripts/train.py --config configs/default.yaml --device cuda
```

或在代码中：

```python
from edit_flows import (
    CubicScheduler, EmptyCoupling, opt_align_xs_to_zs,
    EditFlowsTransformer, prepare_batch, train_step,
)

model = EditFlowsTransformer(vocab_size=V + 3, hidden_dim=512, ...)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = CubicScheduler()
coupling = EmptyCoupling()

for step in range(total_steps):
    # 获取 batch x_0, x_1 (token IDs, 不含 BOS)
    x_0, x_1 = coupling.sample(x_1_data)
    batch = prepare_batch(x_0, x_1, scheduler, opt_align_xs_to_zs, vocab_size=V)
    metrics = train_step(model, batch, scheduler, optimizer)
```

### 4.2 采样/生成

```bash
python scripts/sample.py --checkpoint checkpoint.pt --n_samples 4
```

或在代码中：

```python
from edit_flows import CubicScheduler, sample_euler

model.eval()
scheduler = CubicScheduler()
x_0 = torch.empty((n_samples, 0), dtype=torch.long)  # 空先验
results, trajectory = sample_euler(
    model, x_0, scheduler, n_steps=100, max_seq_len=512,
)
```

---

## 5. 扩展框架

### 5.1 替换模型

只需实现 `forward(tokens, time_step, padding_mask) -> (rates, ins_probs, sub_probs)` 签名即可：

```python
class MyCustomModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        # ... 你的网络结构 ...

    def forward(self, tokens, time_step, padding_mask):
        # tokens:   (B, L)  token IDs
        # time_step: (B, 1)  t ∈ [0, 1]
        # padding_mask: (B, L)  True=PAD位置
        #
        # 返回:
        #   rates:     (B, L, 3)  正数 (建议 softplus)
        #   ins_probs: (B, L, V)  概率分布，dim=-1 求和=1 (建议 softmax)
        #   sub_probs: (B, L, V)  概率分布，dim=-1 求和=1 (建议 softmax)
        ...
        return rates, ins_probs, sub_probs
```

之后直接用 `train_step` 和 `sample_euler`：

```python
model = MyCustomModel(vocab_size=V + 3)
metrics = train_step(model, batch_data, scheduler, optimizer)  # 直接使用
```

### 5.2 自定义调度器

```python
class MyScheduler(KappaScheduler):
    def __call__(self, t: Tensor) -> Tensor:
        return t ** 2  # κ_t = t²

    def derivative(self, t: Tensor) -> Tensor:
        return 2 * t    # κ̇_t = 2t
```

### 5.3 自定义耦合分布

```python
class MyCoupling(Coupling):
    def sample(self, x1: Tensor) -> tuple[Tensor, Tensor]:
        # 返回 (x_0, x_1)，长度可以不同
        x0 = ...  # 自定义先验生成逻辑
        return x0, x1
```

### 5.4 自定义对齐策略

对齐函数签名为 `(x_0: Tensor, x_1: Tensor) -> (z_0: Tensor, z_1: Tensor)`：

```python
def my_align_fn(x_0, x_1):
    # x_0, x_1: (B, L) token IDs (含 PAD)
    # 返回 z_0, z_1: 等长，GAP_TOKEN 标记插入/删除位置
    ...
    return z_0, z_1
```

### 5.5 集成 HuggingFace 模型

```python
from transformers import AutoModel
import torch.nn as nn

class HFEditFlowsModel(nn.Module):
    def __init__(self, hf_model_name: str, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.backbone = AutoModel.from_pretrained(hf_model_name)
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.rates_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3),
        )
        self.ins_head = nn.Linear(hidden_dim, vocab_size)
        self.sub_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens, time_step, padding_mask):
        b, l = tokens.shape
        out = self.backbone(
            input_ids=tokens, attention_mask=~padding_mask,
        ).last_hidden_state
        t_emb = self.time_mlp(self.time_embed(time_step)).unsqueeze(1).expand(-1, l, -1)
        h = out + t_emb

        rates = F.softplus(self.rates_head(h))
        ins_probs = F.softmax(self.ins_head(h), dim=-1)
        sub_probs = F.softmax(self.sub_head(h), dim=-1)

        m = (~padding_mask).unsqueeze(-1).float()
        return rates * m, ins_probs * m, sub_probs * m
```

---

## 6. 训练流程详解

```
每个训练步:

1. 采样 (x₀, x₁) ~ π(x₀, x₁)              ← Coupling
2. 对齐得到 (z₀, z₁)                        ← Alignment (DP编辑距离)
3. 添加 BOS 前缀                             ← prepare_batch
4. 采样 t ~ Uniform(0, 1)                   ← prepare_batch
5. 采样 z_t ~ p_t(·|z₀, z₁)                ← 逐 token 混合路径 (sample_cond_zt)
6. x_t = rm_gap_tokens(z_t)                 ← 去 GAP
7. 模型: x_t, t → rates, Q_ins, Q_sub      ← forward
8. ux_cat = [λ_ins·Q_ins, λ_sub·Q_sub, λ_del]  ← train_step
9. uz_cat = fill_gap_tokens_with_repeats()  ← X → Z 映射
10. uz_mask = make_ut_mask_from_z()          ← 标记正确编辑
11. loss = u_tot - (log_uz * uz_mask * κ̇/(1-κ)).sum()  ← Bregman
12. loss.backward() + clip_grad + step()
```

## 7. 采样流程详解

```
1. 初始化: x₀ ~ prior, t = 0
2. While t < 1:
   a. rates, Q_ins, Q_sub = model(x_t, t)
   b. h_adapt = min(h, (1-κ_t)/κ̇_t)          ← 自适应步长
   c. P(ins) = 1 - exp(-h_adapt · λ_ins)
   d. P(del/sub) = 1 - exp(-h_adapt · (λ_sub + λ_del))
      P(del) = P(del/sub) · λ_del/(λ_sub + λ_del)
      P(sub) = P(del/sub) · λ_sub/(λ_sub + λ_del)
   e. 从 Q_ins/Q_sub 采样具体 token
   f. x_t = apply_ins_del_operations(x_t, masks, tokens)
   g. t += h_adapt
```

## 8. 已知限制与后续方向

| 方面 | 当前实现 | 论文完整版 |
|------|---------|-----------|
| 数据集 | 通用 token 序列 | 文本/代码生成 |
| 模型规模 | 参考 Transformer (~17M) | 280M / 1.3B Llama |
| Classifier-Free Guidance | 未实现 | ✓ |
| Localized Edit Flows | 未实现 | ✓ (+48% Pass@1 on code) |
| Reverse Rates (自校正采样) | 未实现 | ✓ |
| Random alignment | 未实现（仅 DP 对齐） | ✓ |

以上扩展功能可在当前框架上以最小改动添加：
- **CFG**：在 `sample_euler` 中分别计算 conditional/unconditional rates，组合后采样
- **Localized Edit Flows**：扩展 `core/z_space.py` 中的辅助 CTMC，更新 `training/loss.py`
- **Reverse Rates**：新增一个反向模型，在 `sampling/euler.py` 中加入自校正步

---

## 9. 测试

```bash
# 运行全部 59 个测试
python -m pytest tests/ -v

# 仅运行核心模块测试
python -m pytest tests/core/ -v

# 运行集成测试
python -m pytest tests/test_integration.py -v
```

测试覆盖：
- **core 单元测试** (22): scheduler 端点/单调/导数、coupling 形状/特征、alignment 编辑距离一致性、z_space round-trip/mask 正确性
- **training 单元测试** (7): loss 正性/梯度流、prepare_batch 输出完整性、train_step loss 下降和指标
- **sampling 单元测试** (8): 纯插入/删除/替换/混合编辑操作、自适应步长、Euler 采样轨迹
- **集成测试** (4): 端到端训练无 NaN、过拟合收敛、模型输出签名、采样形状

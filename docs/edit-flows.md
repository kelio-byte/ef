# Edit Flows: Flow Matching with Edit Operations

> 论文: [Edit Flows: Flow Matching with Edit Operations](https://arxiv.org/abs/2506.09018) (Havasi et al.)
> 实现代码: `main.py`, `flow.py`, `utils.py`

---

## 1. 问题背景

Edit Flows 是一种**非自回归**序列生成框架。它将序列生成建模为序列空间上的**连续时间马尔可夫链 (CTMC)**，通过三种原子编辑操作——**插入 (insert)、删除 (delete)、替换 (substitute)**——在可变长度序列之间进行传输。

与传统的离散流匹配 (DFM) 不同：DFM 在固定长度的 token 空间上操作（逐 token 独立），而 Edit Flows 天然支持变长序列。

状态空间定义为不超过最大长度 N 的所有可能序列：\(\mathcal{X} = \bigcup_{n=0}^N \mathcal{T}^n\)，其中 \(\mathcal{T}\) 是大小为 M 的词汇表。

---

## 2. 核心概念：Continuous-Time Markov Chain (CTMC)

CTMC 的行为由其**速率矩阵**（rate matrix）\(u_t(y|x)\) 完全定义——表示在时刻 \(t\)，从状态 \(x\) 转移到状态 \(y\) 的瞬时速率（probability per unit time）。分布 \(p_t\) 的演化遵循 Kolmogorov 前向方程：

\[
\frac{d}{dt} p_t(x) = \sum_{x' \neq x} p_t(x') u_t(x|x') - p_t(x) \sum_{x' \neq x} u_t(x'|x)
\]

对于无穷小时间步 \(dt\)，转移概率为：

\[
p_{t+dt}(y|x) = \delta_{x,y} + u_t(y|x) \cdot dt + o(dt)
\]

---

## 3. 三种编辑操作

Edit Flows 只允许序列间「差一个编辑操作」的转移。三种原子操作定义如下：

| 操作 | 符号 | 定义 |
|------|------|------|
| **插入** | \(\text{ins}(x, i, a)\) | 在位置 i 右侧插入 token a |
| **删除** | \(\text{del}(x, i)\) | 删除第 i 个 token |
| **替换** | \(\text{sub}(x, i, a)\) | 将第 i 个 token 替换为 a |

三者**互斥**（产生不同的序列），因此可以独立参数化。

---

## 4. 速率参数化

模型对每个位置预测两个层面的输出：

| 输出 | 维度 | 含义 |
|------|------|------|
| \(\lambda_{t,i}^{\text{ins}}, \lambda_{t,i}^{\text{sub}}, \lambda_{t,i}^{\text{del}}\) | \(3\) | 三种编辑的总速率（正标量） |
| \(Q_{t,i}^{\text{ins}}(a \mid x)\) | \(V\) | 插入时选哪个 token 的概率分布 |
| \(Q_{t,i}^{\text{sub}}(a \mid x)\) | \(V\) | 替换时选哪个 token 的概率分布 |

实际的转移速率为：

\[
\begin{aligned}
u_t^\theta(\text{ins}(x,i,a) \mid x) &= \lambda_{t,i}^{\text{ins}}(x) \cdot Q_{t,i}^{\text{ins}}(a \mid x) \\
u_t^\theta(\text{sub}(x,i,a) \mid x) &= \lambda_{t,i}^{\text{sub}}(x) \cdot Q_{t,i}^{\text{sub}}(a \mid x) \\
u_t^\theta(\text{del}(x,i) \mid x) &= \lambda_{t,i}^{\text{del}}(x)
\end{aligned}
\]

每个位置输出一个 \((2V + 1)\) 维向量（`main.py:574`）：
```
u_i = [λ_ins · Q_ins(0), ..., λ_ins · Q_ins(V-1),   ← V 个插入速率
       λ_sub · Q_sub(0), ..., λ_sub · Q_sub(V-1),   ← V 个替换速率
       λ_del                                          ← 1 个删除速率]
```

---

## 5. 训练

### 5.1 核心困难

直接定义 \(x_0 \to x_1\) 之间的条件概率路径是困难的——存在指数级的编辑路径组合。

### 5.2 解决方案：增广空间 Z

引入增广空间 \(\mathcal{Z} = (\mathcal{T} \cup \{\varepsilon\})^N\)，其中 \(\varepsilon\) 是特殊的 **gap token**（不在真实词表中，代码中为 `GAP_TOKEN = 130`）。

#### 对齐 (Alignment)

将 \(x_0, x_1\) 对齐到等长的 \(z_0, z_1\)。对齐方式决定了编辑操作的解释：

- \(z_0[i] = \varepsilon, z_1[i] = c\) → 需要**插入**
- \(z_0[i] = c, z_1[i] = \varepsilon\) → 需要**删除**
- \(z_0[i] = c_1, z_1[i] = c_2\) → 需要**替换**

代码提供了三种对齐（`utils.py`）：

| 方法 | 策略 |
|------|------|
| `naive_align_xs_to_zs` | 简单右填充 gap 到等长 |
| `shifted_align_xs_to_zs` | 将 \(x_1\) 平移到 \(x_0\) 右侧 |
| `opt_align_xs_to_zs` | Levenshtein 编辑距离 + DP 回溯（最优对齐） |

#### 条件概率路径

在 Z 空间中，条件路径定义为逐 token 独立的混合：

\[
p_t(z_t^i \mid z_0^i, z_1^i) = (1 - \kappa_t) \cdot \delta_{z_0^i} + \kappa_t \cdot \delta_{z_1^i}
\]

其中 \(\kappa_t = t^3\)（`CubicScheduler`，`flow.py:129-138`）。

Z 空间中的条件速率由此推导得出（仅当 \(z_t^i \neq z_1^i\) 时非零）：

\[
u_t^{\text{cond}}(z \mid z_t, z_0, z_1) = \sum_{i=1}^N \mathbf{1}[z_t^i \neq z_1^i] \cdot \frac{\dot{\kappa}_t}{1-\kappa_t} \cdot \delta_{z_1^i}(z^i) \cdot \delta_{z_t}(z^{\neg i})
\]

移除 gap token 的映射 \(f: \mathcal{Z} \to \mathcal{X}\) 将 z 映射回原始空间：\(x = f(z)\)。

### 5.3 Theorem 3.1：辅助过程的边际化

论文核心定理：在增广空间 \((x, z)\) 上定义的 CTMC 可以被边际化得到一个在 \(\mathcal{X}\) 上的有效 CTMC。关键性质——Bregman 散度在增广空间和原始空间中对 \(u^\theta\) 的梯度**完全相同**。

这使得我们可以：
1. 在 Z 空间定义简单的逐 token 条件速率
2. 在 X 空间运行模型（模型不需要理解 gap token）
3. 通过 `fill_gap_tokens_with_repeats` 映射计算损失，梯度正确回传

### 5.4 训练流程

```
1. 采样 (x_0, x_1) ~ π(x_0, x_1)           ← 耦合分布
2. 对齐得到 (z_0, z_1)                      ← 编辑距离对齐
3. 采样 t ~ Uniform(0, 1)
4. 采样 z_t ~ p_t(·|z_0, z_1)              ← 逐token混合路径
5. x_t = f(z_t)                            ← 去除gap token
6. 模型: 输入 x_t, t → 输出 rates, Q       ← Transformer
7. ux_cat = [λ_ins·Q_ins, λ_sub·Q_sub, λ_del]
8. uz_cat = fill_gap_tokens_with_repeats()  ← X→Z映射
9. 构建 uz_mask (标记哪些操作使 z_t→z_1)
10. 计算 Bregman 散度损失
```

### 5.5 损失函数：Bregman 散度

对于负熵 \(F(p) = \sum_x p(x)\log p(x)\)，Bregman 散度退化为 KL 散度。在一个无穷小时间步 \(dt\) 内，学习速率的转移分布与条件速率的转移分布之间的 KL 散度为：

\[
\mathcal{L} = \mathbb{E}_{t, (x_0,x_1), x_t} \left[
\sum_{y \neq x_t} u_t^\theta(y|x_t)
- \sum_{y \neq x_t} u_t^{\text{cond}}(y|x_t) \log u_t^\theta(y|x_t)
\right]
\]

代入 Z 空间条件速率，得到具体的可计算形式（论文 Eq. 23）：

\[
\mathcal{L}(\theta) = \mathbb{E} \left[ \sum_{x \neq x_t} u_t^\theta(x|x_t) -
\sum_{i=1}^N \mathbf{1}[z_1^i \neq z_t^i] \cdot \frac{\dot{\kappa}_t}{1-\kappa_t} \cdot \log u_t^\theta(x(z_t, i, z_1^i) \mid x_t) \right]
\]

代码实现（`main.py:603-607`）：

```python
sched_coeff = scheduler.derivative(t) / (1 - scheduler(t))
loss = u_tot - (log_uz_cat * uz_mask * sched_coeff).sum(dim=(1, 2))
loss = loss.mean()
```

#### 损失的两项各有什么作用

**第一项** \(\sum_{y \neq x} u_t^\theta(y|x)\)：**正则化 / 速率压制**

惩罚所有从 \(x_t\) 出发的编辑操作速率之和。迫使模型只在必要的地方输出高速率，形成「最小编辑偏好」——类似于连续流匹配中的动能最小化效应。

**第二项** \(-u_t^{\text{cond}} \cdot \log u_t^\theta\)：**方向性监督**

只在能使 \(z_t\) 更接近 \(z_1\) 的编辑上产生梯度。`uz_mask` 标记了 Z 空间中哪些是「正确的编辑」：

| \(z_t^i \to z_1^i\) | 操作 | 监督信号 |
|---------------------|------|---------|
| GAP → c | 插入 c | 拉高 \(\lambda^{\text{ins}}, Q^{\text{ins}}(c)\) |
| c → GAP | 删除 | 拉高 \(\lambda^{\text{del}}\) |
| c₁ → c₂ | 替换为 c₂ | 拉高 \(\lambda^{\text{sub}}, Q^{\text{sub}}(c_2)\) |
| c → c (不变) | 无需编辑 | 无监督，仅靠第一项压制 |

**权重** \(\dot{\kappa}_t/(1-\kappa_t)\)：时间自适应缩放

- t 很小：\(\kappa_t \approx 0\)，权重 ≈ 0，不急于编辑
- t 接近 1：\(\kappa_t \to 1\)，权重 → ∞，强制边界条件 \(p_1 = p_{\text{data}}\)

### 5.6 耦合分布 (Coupling)

从 \(\pi(x_0, x_1)\) 采样先验和目标序列对。代码实现了多种耦合（`flow.py`）：

| 耦合 | \(p(x_0)\) | 用途 |
|------|-----------|------|
| `EmptyCoupling` | \(\delta_\varnothing\)（空序列） | 论文默认的 Edit Flow |
| `GeneratorCoupling` | 生成函数（如低频正弦波） | seq2seq 任务 |
| `ExtendedCoupling` | 在 \(x_1\) 中随机插入噪声 | 扩展先验 |
| `UniformCoupling` | 均匀随机序列 | 去噪任务 |

### 5.7 模型架构

`SimpleEditFlowsTransformer`（`main.py:158-272`）:

- **输入**: token 序列 \(x_t\) + 时间步 \(t\)
- **嵌入**: Token Embedding + 正弦时间嵌入 + 位置嵌入
- **骨干**: 多层 TransformerEncoder（双向自注意力，非因果）
- **输出头**:
  - `rates_out`: (B, L, 3) → softplus 激活（保证 >0）
  - `ins_logits_out`: (B, L, V) → softmax → \(Q^{\text{ins}}\)
  - `sub_logits_out`: (B, L, V) → softmax → \(Q^{\text{sub}}\)

时间嵌入采用正弦位置编码风格（`SinusoidalTimeEmbedding`），后接两层 MLP + SiLU。

---

## 6. 生成（采样）

### 6.1 Euler 采样流程

从 \(t=0\) 开始，用一阶 Euler 步近似求解 CTMC，直到 \(t=1\)：

```
1. 初始化: x_0 ~ p(x_0), t = 0
2. While t < 1:
   a. 模型前向: λ_ins, λ_sub, λ_del, Q_ins, Q_sub = model(x_t, t)
   b. 自适应步长: h_adapt = min(h, (1-κ_t)/κ̇_t)
   c. 每种编辑独立采样:
      P(ins)  = 1 - exp(-h_adapt · λ_ins)
      P(del/sub) = 1 - exp(-h_adapt · (λ_sub + λ_del))
      P(del) = P(del/sub) · λ_del/(λ_sub + λ_del)
      P(sub) = P(del/sub) · λ_sub/(λ_sub + λ_del)
   d. 插入/替换从 Q_ins/Q_sub 采样具体 token
   e. 并行应用所有编辑操作
   f. t += h_adapt
```

### 6.2 自适应步长

```python
h_adapt = min(h, (1 - scheduler(t)) / scheduler.derivative(t))
```

这个约束确保步长不超过概率路径的变化速率，防止在 t→1 时出现数值不稳定。当 \(t\) 接近 1 时，\(\kappa_t \to 1\)，\(1-\kappa_t\) 变小，自适应步长自动减小。

### 6.3 并行编辑应用

`apply_ins_del_operations`（`main.py:700-766`）：

- 同一位置触发 insert + delete → 视为替换
- 计算累计偏移量，确定 token 的新位置
- 所有操作同时应用（非自回归的核心优势）

---

## 7. X 空间与 Z 空间的映射机制

### 7.1 去 Gap 映射：`rm_gap_tokens`

```python
x_t, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z_t)
```

- 从 \(z_t\) 中移除所有 GAP_TOKEN
- 用 PAD_TOKEN 右填充到等长
- 保存 gap 掩码和 pad 掩码用于后续 X→Z 回映射

### 7.2 X→Z 回映射：`fill_gap_tokens_with_repeats`

模型只在 X 空间（无 gap）上运行，但损失需要在 Z 空间计算。每个 gap 位置的速率使用**前一个非 gap 位置的速率**：

```python
non_gap_mask = ~z_gap_mask
indices = non_gap_mask.cumsum(dim=1) - 1    # gap处不累加，索引停留在前驱non-gap
uz = ux[batch_indices, indices]              # 重复填充
```

这意味着多个连续 gap 共享同一个 X 空间位置的模型输出。

### 7.3 BOS token 的作用

每个序列开头添加 BOS token（`main.py:408-411`），确保 Z 空间中位置 0 永远是 non-gap，为所有开头的 gap 提供可靠的锚点。否则若开头是 gap，cumsum 会导致索引出界或语义错乱。

---

## 8. 训练-生成的对偶关系总结

```
                       训练                                        生成

       Z空间条件速率 u^cond(·|z_t, z₀, z₁)
       │  格式: 在需要编辑的位置 i:
       │  u^cond = κ̇/(1-κ) · δ(z_t^i → z_1^i)
       │
       ▼
  ┌────────────────────────────────────────────────┐
  │  Bregman散度 = Σu^θ  -  κ̇/(1-κ) · Σ log u^θ │
  │                                                    │
  │  · Σu^θ:    压低所有速率 → 最小编辑偏好            │
  │  · CE项:     拉高正确编辑速率 → 方向引导           │
  │  · κ̇/(1-κ): 时间权重 → 边界条件约束               │
  └────────────────────────────────────────────────┘
       │
       ▼
      学习到的 X空间速率 u^θ(·|x_t)
       │  模型输出: λ_ins, λ_sub, λ_del, Q_ins, Q_sub
       │
       ▼
       Euler采样: P(edit) = 1 - exp(-h_adapt · λ)
       │  插入/删除/替换 同时并行应用
       │
       ▼
      最终生成序列 x_1
```

---

## 9. 细节问答

### Q1: 每个位置的 u 是一个概率分布吗？λ 的总和为 1 吗？

**不是。** u 是速率向量（正实数），不是概率分布。三个 λ 各自独立，总和可以是任意正数。

三个层次：
- **λ_ins, λ_sub, λ_del**：正标量，决定编辑**多频繁**发生（单位时间速率）。通过 softplus 保证 >0，无归一化约束。
- **Q_ins(a), Q_sub(a)**：概率分布，softmax 输出，和恒为 1。决定**如果**编辑发生，**选哪个** token。
- **实际速率**：`λ × Q`，即 2V+1 维向量，各分量之和 = λ_ins + λ_sub + λ_del，无上界。

类比：λ 是泊松过程的强度参数，可以 >1。生成时通过 \(P = 1 - e^{-h\lambda}\) 将速率转为概率。

### Q2: 训练 loss 中 \(\mathbf{1}[z_1^i \neq z_t^i]\) 的作用是什么？

它标记了 Z 空间中**需要编辑的位置**。两点作用：

- **z_t^i ≠ z_1^i**：CE 项有梯度，拉高正确方向的编辑速率。第一项同时负责压制速率（防止过高）。
- **z_t^i = z_1^i**：CE 项掩码为 0，无监督信号。仅靠第一项的速率压制将三个 λ 推向 0。

这就是「最小编辑偏好」的来源：模型只在位置确实需要编辑时才产生高速率，已经正确的 token 被自动推向「不动」。

### Q3: 连续多个 gap 需要插入不同 token，如何处理？

多个 gap 在损失中都映射到**同一个前驱 X 空间位置**的模型输出。每个 gap 的 CE 监督强度各为 1，在**梯度层面**叠加：

```
Z空间:  [BOS,  a,  GAP1, GAP2, GAP3,  b]
X空间:  [BOS,  a,                      b]
               ↑ 三个gap都映射到这里

CE梯度累加:
  ∂L/∂log_λ_ins[a]   += 3 · κ̇/(1-κ)      ← λ_ins 被三重强化
  ∂L/∂logit_X[a]     += κ̇/(1-κ)           ← Q_ins(X) ↑
  ∂L/∂logit_Y[a]     += κ̇/(1-κ)           ← Q_ins(Y) ↑
  ∂L/∂logit_Z[a]     += κ̇/(1-κ)           ← Q_ins(Z) ↑
```

**不额外归一化**。Q_ins 始终由 softmax 自动归一化。梯度层面三者公平竞争，λ_ins 的 3 倍强化让模型自动学会这个位置需要更高的插入速率。

**生成时**不存在 gap——每次插入后序列变长，新 token 获得独立位置，模型用新上下文重新决策。多步迭代自然处理连续插入。

### Q4: non-gap 自己要删除 + 后面有 gap 要插入，这种复杂情况如何处理？

同一个 X 空间位置的输出承担**多种编辑的监督信号**：

```
z_t:  [BOS,  a,   GAP,  b  ]
z_1:  [BOS,  GAP, X,    Y  ]
             ↑删  ↑插   ↑替

X空间 a 位置:
  DELETE mask（Z位置"a"）： CE拉高 λ_del
  INSERT X mask（Z位置"GAP"）：CE拉高 λ_ins, Q_ins(X)

两股梯度累加到同一组参数 → 模型学会"删除自己"+"后面插入X"
```

不同的 Z 空间位置各自标记各自的正确操作，梯度回传时在不同位置提取不同通道的 logit，自然叠加。

### Q5: 为什么要用 BOS token？

BOS 确保 Z 空间中位置 0 永远是 non-gap，为所有开头的 gap 提供 `fill_gap_tokens_with_repeats` 的可靠锚点。否则若开头是 gap，cumsum-1 会出界（可能 -1 被 clamp 到 0），导致语义错位的速率引用。

### Q6: 为什么用 Bregman 散度而非直接回归速率？

直接回归边际速率的期望 \(\mathbb{E}_{p_t(x_0,x_1|x_t)}[u_t(x|x_t, x_0, x_1)]\) 是 intractable 的——需要对所有可能的 \((x_0, x_1)\) 配对求期望。

Bregman 散度允许我们在**不需要显式计算期望**的情况下，通过条件速率的对数来学习边际速率。它是一个凸函数的广义距离——当取负熵时退化为 KL 散度。其关键性质（Theorem 3.1）：在增广空间的梯度等同于边际化后空间的梯度，使得 X 空间的模型可以在 Z 空间的监督下正确训练。

---

## 10. 代码结构索引

| 模块 | 文件 | 关键内容 |
|------|------|---------|
| 耦合分布 | `flow.py:24-114` | EmptyCoupling, GeneratorCoupling, ExtendedCoupling, UniformCoupling |
| 调度器 | `flow.py:117-138` | KappaScheduler, CubicScheduler (κ_t = t³) |
| 序列对齐 | `utils.py:10-88` | _align_pair, naive/opt/shifted_align_xs_to_zs |
| GAP 处理 | `utils.py:91-121` | rm_gap_tokens, rv_gap_tokens |
| 模型 | `main.py:133-272` | SimpleEditFlowsTransformer, SinusoidalTimeEmbedding |
| 训练 | `main.py:516-636` | 批采样、速率计算、X→Z映射、损失计算 |
| 采样 | `main.py:700-910` | apply_ins_del_operations, get_adaptive_h, Euler循环 |
| 损失掩码 | `main.py:417-468` | make_ut_mask_from_z, fill_gap_tokens_with_repeats |

---

## 11. 本实现的简化

| 方面 | 论文 | 本实现 |
|------|------|--------|
| 数据集 | 文本/代码生成 | 合成正弦波离散序列 |
| 对齐 | 随机对齐 + 最优对齐 | Levenshtein 编辑距离对齐 |
| 模型规模 | 280M / 1.3B Llama | 小型 Transformer (~17M) |
| Classifier-Free Guidance | ✓ | ✗ |
| Localized Edit Flows | ✓ (+48% Pass@1 on code) | ✗ |
| Reverse Rates / Stationary Component | ✓ (自校正采样) | ✗ |

核心的训练-采样框架（增广空间对齐 → 逐 token 混合路径 → 边际化 → Bregman 损失 → Euler 采样）完全按论文实现。

---

## 参考文献

1. Gat et al., *Discrete Flow Matching*, arXiv:2407.15595
2. Havasi et al., *Edit Flows: Flow Matching with Edit Operations*, arXiv:2506.09018
3. Le Bellier, *Introduction to Flow Matching*, GitHub: lebellig/discrete-fm

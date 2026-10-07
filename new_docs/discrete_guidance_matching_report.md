# Discrete Guidance Matching：面向 Edit Flows / Euler-Beam 的借鉴报告

> 论文：`DISCRETE GUIDANCE MATCHING: EXACT GUIDANCE FOR DISCRETE FLOW MATCHING`
> （ICLR 2026，Wan et al.）
>
> 本报告依据项目中的
> [`2026--Discrete Guidance Matching Exact Guidance for Discrete Flow Matching.pdf`](</root/autodl-tmp/edit_flows/PDF/2026--Discrete%20Guidance%20Matching%20Exact%20Guidance%20for%20Discrete%20Flow%20Matching.pdf>)
> 整理。这里的“exact”指在论文假设成立、指导函数准确且 CTMC 采样实现正确时，
> 指导后的转移率/后验对应目标分布；不等于有限步 Euler 数值误差和神经网络近似误差
> 自动消失。

## 0. 一句话结论

我们当前 Euler-Beam 的主要缺口不是“分支数还不够大”，而是模型始终在采样原始
Edit Flows 分布 `p(reactants | product)`，然后用 branch log-mass、状态合并和
`changed_state_bonus` 做启发式搜索；它没有一个能够把化学偏好或 reward 系统地转化为
目标分布 `q` 的机制。

这篇论文提供了一个适合研究的方向：冻结原始 flow 模型，训练一个正值 guidance 网络
估计密度比 `r=q/p` 的条件期望，在每个离散状态上一次性重加权所有候选后验，再用同一
套 CTMC 编辑机制采样。它有望比“生成很多候选再 rerank”更有效率、更有理论依据，
但不能直接复制到当前代码：我们的 Edit Flows 支持变长序列、insert/delete/replace，
而论文的主公式按固定维度坐标状态书写；同时当前 checkpoint 没有独立保存完整 product
condition。因此必须先完成状态和编辑动作的形式化，再进入化学实验。

## 1. Motivation：我们自己的推理方法有什么问题？

### 1.1 当前模型和采样器实际在做什么

对一个固定产物 `c`，当前 checkpoint 学到的是一个源分布，可以记作

```text
p_c(y) = p(reactants y | product c)
```

模型在每个状态 `x_t` 输出三类编辑 rate 和 token proposal：

- `lambda_ins(i)`、`Q_ins(token | i)`：在位置附近插入 token；
- `lambda_sub(i)`、`Q_sub(token | i)`：替换 token；
- `lambda_del(i)`：删除 token。

Euler 依据这些 rate 做随机编辑；Euler-Beam 则让每个父分支生成若干 child，再按
`log_mass`/路径概率合并、排序和剪枝。`changed_state_bonus`、`stochastic_noop` 等
是搜索启发式，它们可以改变保留哪些样本，但不是从一个明确的目标分布重新推导出的
CTMC 转移率。

### 1.2 具体缺口

#### 问题 A：搜索扩展不等于目标分布指导

增大 `R`、`K` 或 `M` 只是增加候选轨迹数量。它不能回答：

```text
如何让“更可能有效、能解释产物、前向反应得分高”的反应物，
在每一个中间状态就获得更大的生成概率？
```

目前的 bonus 是固定启发式，且对所有非原始状态近似一视同仁；它并不等于一个
product-conditioned chemical reward。`M=3` 时指标明显下降也说明，单纯增加随机
child 会扩大搜索噪声，而不会自动带来正确的质量分配。

#### 问题 B：事后 rerank 的代价和偏差

如果把一个 reward model 只用于最终候选排序，必须先生成大量完整反应物；如果把它
用于每一个可能的下一步编辑，则每个状态可能有 `O(L|V|)` 个插入/替换候选，再加删除
候选。逐候选调用 reward model 或额外模型 forward 会非常慢。

更重要的是，事后 top-k 保留只是在有限 proposal 集合中挑样本，不保证得到

```text
q_c(y) ∝ p_c(y) r(c,y)
```

这样的目标分布；分支数变了，排序结果和偏差也会变。

#### 问题 C：离散状态下的一阶近似并不可靠

连续空间可以用梯度近似 reward 对状态的影响；SMILES 编辑是离散跳转，插入、删除和
替换不是小的欧氏位移。对每个 token 的 reward 变化使用一阶 Taylor 近似，可能既不
准确，也无法自然处理序列长度改变。

这正是论文的切入点：在离散 CTMC 中直接构造目标后验/转移率，而不是用离散状态之间
的欧氏梯度近似。

#### 问题 D：当前模型缺少独立的完整 product condition

当前 checkpoint 的 `use_origin_mask=False`，模型主要看到当前被编辑的状态。对于
逆合成而言，真正需要的是固定 `product c` 下的条件分布。当前 copy-product 路径有
帮助，但当早期编辑删除或替换 product token 后，模型不一定还能完整访问原始 product。

因此，即使我们训练 guidance，也必须明确 guidance 是否接收完整 product；否则 reward
可能把不同 product 的中间状态混在一起。

### 1.3 我们真正希望解决的目标

把当前源分布 `p_c(y)` 变成一个可解释的目标分布：

```text
q_c(y) = p_c(y) r(c,y) / Z(c)
```

其中 `r(c,y) >= 0` 是密度比或其未归一化版本，`Z(c)` 不需要显式计算。`r` 可以由
化学 reward、前向反应模型得分、validity/atom-conservation 等组合而来，但必须在
采样时不使用 test target，避免标签泄漏。

## 2. 论文方法是什么？

### 2.1 直白凝练的解释

论文方法可以压缩为五句话：

1. 先有一个已经训练好的离散 flow 模型，它会从源分布 `p` 生成样本。
2. 定义想要的目标分布 `q`，并用 `r(x)=q(x)/p(x)` 表示“这个最终样本应该被偏爱多少”。
3. 训练一个 guidance 网络，告诉我们：在当前中间状态下，如果某个位置最终取某个
   token，它平均会带来多大的 `r`。
4. 每个采样步把原模型的 token 后验乘上这个 guidance，再归一化；因此高 reward 的
   后续从中间阶段就更容易被采到。
5. 这一重加权来自 CTMC 的精确后验公式，不需要为每个候选状态单独做一次 reward
   forward，也不需要在离散 token 空间使用梯度近似。

它不是 beam search 的替代品。它负责改变“应该往哪里跳”的概率；beam 可以作为额外
的有限搜索层，但不再承担全部的目标偏好建模。

### 2.2 论文的详细对象和假设

论文把最终离散样本写成 `x_1`，当前状态写成 `x_t`，每个样本有 `D` 个离散坐标，
每个坐标的 token 状态属于有限集合 `S`。源模型有条件后验

```text
p^d_{1|t}(s | x_t)
```

表示在看到当前状态后，第 `d` 个坐标最终取 token `s` 的概率。源模型同时定义了
条件编辑 rate `u^d_{p,t}(z_d, x_d | x_1)`。

论文的关键假设是：

1. 目标分布的 support 是源分布 support 的子集，即 `q_1 << p_1`，这样密度比有定义；
2. 源和目标使用相同的条件概率路径 `p_{t|1}=q_{t|1}`；
3. guidance 网络能够足够准确地估计所需的条件期望。

第一个条件在我们的任务中意味着：如果化学 reward 给某个反应物非零权重，原始
Edit Flows 也必须有机会生成它。一个原模型概率严格为零的反应物，单靠 guidance
不能凭空创造出来。

### 2.3 Theorem 1：posterior-based guidance

对每个坐标和候选 token，论文定义正值 guidance：

```text
H^d_t(s, x_t)
  = E_{p(x_{1,\setminus d} | x^d_1=s, x_t=x_t)} [ r(x_1) ]
```

直观上，它不是问“当前 token 的 reward 是多少”，而是问：在当前状态下强制第 `d`
个最终 token 为 `s`，其余未知部分仍按源模型后验补全时，最终密度比的条件平均是多少。

于是目标后验可以精确写成：

```text
q^d_{1|t}(s | x_t)
  = H^d_t(s, x_t) p^d_{1|t}(s | x_t)
    / Σ_{s'∈S} H^d_t(s', x_t) p^d_{1|t}(s' | x_t)
```

也就是“源后验 × 条件 guidance，再归一化”。最终在该目标后验下采样 `x_1`，并使用
相同的条件 CTMC rate，就能得到目标路径。

这比直接把 reward 乘到某一个已采样 child 上更完整：它在生成下一个跳转之前同时考虑
所有可能 token 的相对权重。

### 2.4 为什么只需要一次批量 forward

论文的 guidance 网络一次输入当前状态 `x_t`，输出所有坐标、所有 token 的正值矩阵
`H_t(x_t) ∈ R_+^{D×|S|}`。原模型也一次输出 `p_{1|t}` 或相应 rate。然后在 GPU 上
逐元素相乘和归一化：

```text
base posterior  ─┐
                 ├─ elementwise multiply → normalize → sample
guidance H      ─┘
```

因此复杂度主要是一次 base model forward 加一次 guidance model forward（如果共享
Transformer trunk，也可以合并为一次主干 forward），而不是对每个候选 token 逐个调用
模型。论文表格把这类实现记为每步一次 guidance function evaluation。

### 2.5 Guidance 网络如何训练

论文用 Bregman divergence 学习正值条件期望。选择
`F(h)=h log h` 后，忽略与参数无关的常数，源数据训练目标为：

```text
L_{h,p}(θ) = E [ Σ_d ( h^d_θ(x^d_1, x_t)
                         - r(x_1) log h^d_θ(x^d_1, x_t) ) ]
```

最优解正是所需的条件期望。实际实现应使用 `softplus` 或 `exp` 保证 `h>0`，并对
`log h` 做数值保护。

如果有目标分布样本，论文再加入一个 regularization：在目标样本上直接训练由
`H · p_{1|t}` 归一化得到的 guided posterior。其形式可以理解为：

```text
L_h(θ) = L_{h,p}(θ) + λ L_{h,q}(θ)
```

这里的 `λ` 是目标样本正则强度，不是我们 Euler-Beam 中的 `changed_state_bonus`，
二者不能混用。

### 2.6 采样和“exact”的边界

采样时，论文先根据重加权后验采样最终 token，再按对应 conditional rate 产生跳转。
为避免简单 Euler 概率在大 rate 时超出 `[0,1]`，附录使用 always-valid 形式，跳转概率
为

```text
1 - exp(-h λ)
```

这与我们当前 Euler/Euler-Beam 的 Poisson event probability 方向一致。

论文还讨论了 rate-based guidance，但它要求源/目标 corruption rate 相同，条件更强；
在一般离散 flow 中，posterior-based Theorem 1 更灵活。对于我们的变长 Edit Flow，
第一阶段应优先借鉴 posterior-based 版本，而不是直接套 rate-based 版本。

## 3. 适合我们的任务吗？如何借鉴？

### 3.1 适配性判断

| 论文条件/组件 | 我们当前项目 | 判断 |
|---|---|---|
| 离散状态、CTMC 跳转 | token 序列上的 insert/substitute/delete rate | 方向匹配 |
| 已训练 source flow | `checkpoint_step600000.pt` | 已有 |
| token 后验/编辑 proposal | `log_ins_probs`、`log_sub_probs`、三类 rate | 部分匹配 |
| 目标/源密度比 `r` | 当前没有 reward/density-ratio 模型 | 必须新增 |
| 固定维度坐标 | 我们支持变长编辑和 GAP/Z 空间 | 需形式化，不能直接复制公式 |
| 条件变量 | product 是任务条件，但当前模型没有独立 product encoder | guidance 必须显式接收 product |
| 逐候选 cost | 当前 branch child 已经批量化，但没有 reward guidance | 有效率收益空间 |
| 论文的同一 conditional path | 我们的 aligned Z-space path 可能可对应，但需验证 | 先做 toy proof/实验 |

### 3.2 最值得借鉴的部分

#### 借鉴一：从“branch 排序”转向“后验重加权”

当前每个 child 先随机生成，再由 beam 选择；论文建议在采样前就对所有可行编辑的
posterior 做 guidance。这样 `M` 的作用从“盲目增加随机候选”变成“在被 guidance
重排后的分布上增加 Monte Carlo 覆盖”。这更可能改善 top-1，同时不会因为单纯扩大
`M` 而将大量质量预算花在低 reward child 上。

#### 借鉴二：优先 posterior-based，而不是一阶梯度或逐候选 reward

我们的状态是离散 token 和编辑操作，插入/删除没有自然的连续梯度。论文的后验公式
可以直接在 token/action 维度进行向量化，避免为每个假想编辑复制一次 Transformer。

#### 借鉴三：冻结现有 checkpoint，新增独立 guidance adapter

这样可以把“原始模型能力”和“guidance 能力”分开归因：

- `guidance=0` 应退化为当前 Euler/Euler-Beam；
- guidance 网络训练失败时不会污染 `checkpoint_step600000.pt`；
- 可以只改推理层和 guidance checkpoint，不重训原始 Edit Flows。

### 3.3 不能直接照搬的部分

#### 难点 A：变长 Edit Flow 的动作空间

论文坐标 `d` 的状态改变通常保持维度 `D`。我们的操作会改变序列长度：

```text
insert(i, v), substitute(i, v), delete(i)
```

必须先定义一个固定的“动作坐标空间”。推荐方案是：在 aligned Z-space 中将 GAP 作为
合法状态，令每个对齐位置拥有真实 token/GAP 状态；然后把 Z-space 的目标状态映射回
当前 X-space 的 insert/substitute/delete action。若某个 X-space action 无法唯一对应
一个 Z-space transition，就不能声称使用了论文的 exact posterior guidance，只能称为
action-level approximate guidance。

#### 难点 B：`H` 不是现有 `Q_ins/Q_sub` 的简单 temperature

Q sharpening 只改变同一 operation 内 token 分布的尖锐程度；论文的 `H` 还需要根据
整个 product-conditioned 后验估计最终 reward，并可能同时改变 insert/substitute/delete
三类动作的相对质量。不能把 `q_temperature` 当成论文 guidance 的替代品。

#### 难点 C：需要明确 reward 与 density ratio

论文假设 `r=q/p` 已知或可学习。对逆合成，我们还没有现成的 reward model。可以考虑：

- RDKit validity、价态/结构约束等规则 reward；
- reactants → product 的 forward reaction model likelihood；
- atom conservation、反应中心合理性、组件数量等反应级 reward；
- 用真实训练反应作为正样本、Euler 生成的负样本训练 product-conditioned reward model。

第一阶段不能把 test target 直接作为 reward 输入，否则 top-k 会发生标签泄漏。

#### 难点 D：当前 product condition 不完整

guidance 网络至少应输入 `(product c, current edit state x_t, t)`。如果只输入 `x_t`，
它无法可靠区分不同 product 对同一中间 token 状态的偏好。独立 product encoder、
cross-attention 或冻结 product memory 都可以作为第一版；这属于 guidance adapter 的
结构，不要求立即重训 base checkpoint。

### 3.4 推荐的第一版适配形式

我们不建议一开始重写 `euler_beam.py`，而是新增一个受保护的
`posterior_guided_euler.py`（或在 `euler.py` 上增加显式 sampler 选项）：

1. 冻结 `checkpoint_step600000.pt`，保持原始 rate/Q 输出不变。
2. 在 aligned Z-space 定义固定坐标、GAP 状态和 action↔coordinate 映射。
3. 新增 guidance 网络 `H_φ(c, x_t, t)`，输出每个位置、每个目标 token/GAP 的正值。
4. 用源训练对 `(product, aligned reactants, z_t)` 和 `r(product, reactants)` 训练
   `L_{h,p}`；有可信目标样本后再加入 `λ L_{h,q}`。
5. 采样时计算

   ```text
   p_guided(s | c, x_t)
     = normalize( H_φ(s, c, x_t, t) · p_base(s | x_t, t) )
   ```

   再把它映射到当前可执行编辑 action，并使用 `1-exp(-hλ)` 的批量跳转实现。
6. 先运行单路径 guided Euler，再将同一 action proposal 接入 Euler-Beam；beam 只负责
   在 guided proposal 上维护有限分支，不再额外叠加未经校准的 reward bonus。

这条路径既保留当前工程的向量化优势，也能清楚区分“论文形式化 exact 版本”和“工程
上的 action-level 近似版本”。

## 4. 详细任务安排

以下任务按“先证明机制、再训练 reward、最后扩大实验”的顺序执行。所有新 checkpoint
和 YAML 都放在独立目录，历史 checkpoint 与历史结果只读保留。

### DG-0：形式化与可行性审计

**目标**：确认论文 Theorem 1 能否在我们的 aligned Z-space / Edit operation 上成立。

具体工作：

1. 写出当前 `log_rates + Q` 对应的 action-level `u_p(a | c,x_t,t)`。
2. 定义 `a=(position, operation, token)` 与 Z-space 坐标状态之间的双向映射。
3. 证明或实验检查 source/target 是否共享同一 conditional path；记录 GAP、BOS、PAD、
   variable-length 的边界规则。
4. 明确 guidance 的 product 输入方式和 reward 支持范围。

交付物：公式文档、action mapping 表、一个明确结论：`posterior-exact` 或
`action-level approximate`。在该结论完成前不改正式采样器。

### DG-1：基线接口和可观测性

**目标**：不引入 guidance 时，新的接口逐字节/逐 seed 复现当前 sampler。

具体工作：

- 抽取 `base posterior/rate` 计算和 action probability 分解为可测试函数；
- 记录每步 parent 数、action 数、base forward 次数、guidance forward 次数；
- 保持 `seed=42`、`n_steps=100`、TF32 设置和现有 output layout；
- 加单元测试：`H=1` 时 guided posterior 等于 base posterior，guidance 关闭时输出与
  当前 Euler 对齐。

验收：tiny 上相同 seed 的输出完全一致或差异有明确的浮点原因；Top-1～10、invalid、
coverage 和 wall time 作为 baseline manifest 保存。

### DG-2：已知密度比的 toy / oracle 验证

**目标**：在不涉及化学 reward 的小离散系统中验证公式和 sampler。

具体工作：

1. 构造小词表、固定长度序列和已知源分布 `p`；手工定义 `r`，可直接计算目标 `q`。
2. 训练或直接提供 oracle `H`，比较 base posterior、guided posterior 和经验终态
   分布。
3. 比较 posterior-based 与 naive reward rerank / first-order 近似。
4. 测试不同 `h`、rate 大小和重复采样数，检查无效概率和 KL/TV 距离。

验收：guided 经验分布明显更接近解析 `q`；`H=1` 回退；高 rate 下不出现概率越界。
这是决定是否继续化学适配的硬门槛。

### DG-3：化学 reward / density-ratio 定义

**目标**：在不使用 test target 的前提下得到可训练的 `r(c,y)`。

推荐分两级：

- **规则版 pilot**：RDKit validity、价态检查、组件/原子守恒等低成本 reward，用于先测
  采样机制；
- **模型版**：训练 product-conditioned forward/reward model，用真实 train reaction
  为正样本，用冻结 base Euler/Euler-Beam 产生的有效/无效候选为负样本。

需要确定 reward 标度、温度 `τ`、`r=exp(R/τ)` 的 clipping/log-domain 保护，并在独立
validation 上校准。不要使用 test 的 target 计算 guidance label。

验收：reward 与 validation 的 validity/forward consistency 有相关性；记录 reward
模型本身的 AUROC、校准和对插入偏差的影响。

### DG-4：训练 guidance adapter

**目标**：冻结 base checkpoint，只训练 `H_φ`。

建议初版：

- 输入：product condition、当前 X/Z state、时间 `t`；
- 输出：位置×token/GAP 的 positive `H`；
- 主损失：论文 `L_{h,p}`；
- 第二阶段再 sweep `λ L_{h,q}`，候选 `λ∈{0,0.2,0.5,1.0}`；
- guidance 学习率、步数和结构单独写入新的 `configs/guidance_retro.yaml`，不修改
  `configs/retro.yaml`。

验证 guidance 的正值、均值、极端值、`H` clipping 命中率和 train/validation gap。

### DG-5：单路径 guided Euler

**目标**：先验证 guidance 改变的是概率分配，而不是 beam 代码造成的结果。

实现顺序：

1. 新增 `posterior_guided_euler`，保留当前 `sample_euler` 不变；
2. base/guidance forward 按 batch 向量化；
3. 通过 exact posterior 重加权采样 action；
4. 使用当前 Poisson/always-valid event update；
5. 先不使用 `changed_state_bonus`，避免把 guidance 与旧 heuristic 混在一起。

对照配置固定：base Euler、guided Euler、base Euler-Beam。每个配置使用相同 product
区间、seed、n_steps 和输出数量。

### DG-6：接入 Euler-Beam

**目标**：在 guided proposal 上使用有限 beam，而不是用 beam 代替 guidance。

实验矩阵至少包括：

| 组别 | sampler | 说明 |
|---|---|---|
| A | Euler | 当前源分布基线 |
| B | Euler-Beam R9K1M2 | 当前工程推荐搜索基线 |
| C | Guided Euler | 只验证 guidance |
| D | Guided Beam R9K1M2 | guidance + 搜索 |

初始 sweep：`guidance_strength/τ`、`λ`、是否使用 reward clipping；先在 tiny 做 smoke，
再在 mini-1001 做选择。每项必须报告 Top-1～10、invalid、coverage、平均 reward、
模型 forward 次数、guidance forward 次数和 wall time。

### DG-7：完整验证和结论

只有 DG-0～DG-6 通过后，才在完整 `src-test` 上跑一次最终配置。报告中同时给出：

- 是否达到与 base 相同的 source support；
- Top-1～10 和 invalid/coverage 的变化；
- guidance 带来的 reward 提升是否伴随 exact-match 下降；
- 每个 product 的输出多样性、重复率和 branch 合流情况；
- 总时间和每一步 forward 数，确认“理论上的一次批量 guidance”确实落到实现效率。

建议的晋级标准不是只看一个 top-1：至少要求 exact-match 不显著下降、invalid 不上升，
并且 reward/coverage 有可重复改善；若只提高 reward 却降低 top-k exact accuracy，应把
它记录为“偏好 steering 成功但不适合当前 benchmark”，不能直接替换 baseline。

## 5. 风险、停止条件和不应做的事

1. 不要把论文的 `λ` regularization、reward temperature `τ` 和 Euler-Beam 的
   `changed_state_bonus` 视为同一个变量。
2. 不要直接在现有 `euler_beam.py` 中同时加入 reward、new posterior、new beam score；
   必须通过独立 sampler 和独立 YAML 做单变量对照。
3. 如果 DG-0 证明变长动作无法满足论文同一 conditional path，应诚实标记为近似方法，
   不使用“exact guidance”作为方法名或结论。
4. 如果 reward 模型对 invalid/插入偏差没有校准，先修 reward，不要用增大 guidance
   strength 掩盖问题。
5. `checkpoint_step600000.pt`、现有 baseline 结果和测试 target 只读；新方法失败时
   可以完整回退。

## 最终建议

这篇论文值得借鉴，但第一步不是马上训练一个大型 reward model，也不是直接把 reward
乘到 Euler-Beam 的分数上。最稳妥的路线是：

```text
Z-space/action 形式化
    → 已知 r 的 toy exact 验证
    → 规则 reward pilot
    → product-conditioned guidance adapter
    → guided Euler
    → guided Euler-Beam
    → mini 选择 + full src-test 验证
```

如果 toy 和 DG-0 不能通过，就保留当前 Euler-Beam，把论文结论记录为“理论上相关但
当前变长 Edit Flow 暂不满足适配条件”；如果通过，则 guidance adapter 是比继续盲目
增大 `R/K/M` 更有研究价值、也更容易解释的下一条方法分支。

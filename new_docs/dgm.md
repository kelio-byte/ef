# DGM 在 Edit Flows / Euler-Beam 中的适配方案

状态：阶段 1 synthetic mechanics 已通过；阶段 2 reward 评估接口已完成；阶段 3 正式
train/validation guidance 数据已生成并审计通过；阶段 4 action-level guidance 训练和
held-out 观测已通过最低门槛；阶段 5 ordinary Euler action-level adapter 已实现，identity
和 validation-200 off/on 对照均完成。当前实现机制通过，但 validity reward 尚未通过准确率
收益门槛；阶段 7 的 forward reaction reward 已完成 checkpoint/tokenizer/方向 smoke、
批量 reward adapter 和 pilot guidance/β 校准，但尚未通过完整 Top-k 门槛，默认采样仍
关闭 guidance。2026-08-08 又完成 5,000-step 单变量复核和代码全链路审计：更长训练没有
消除 Top-1 与 Top-3/10/Oracle 的权衡，因此不继续盲目增加 step。

**2026-08-11 反应级评估更新。** 旧 validation-A/B 已不再承担方法有效性结论。按照新的
1,000 个独立反应开发集协议，普通 Euler 的 Top-1/Top-10/候选覆盖率为
58.2% / 83.5% / 86.6%。终点正向重排把 Top-10 提至 85.0%，但 Top-1 降至 57.6%；旧
大规模 guidance 的 Top-1 为 55.8%；修正为共享中间状态数据后的 guidance 有 Top-10
84.9% 和覆盖率 87.5% 的信号，但 Top-1 仍降至 56.2%（相对基线 −2.0 个百分点，配对
95% 区间 [−3.9, −0.2]）。后续将训练状态覆盖扩展到 step 10/30/50/70/90、并加入共享状态内
排序／校准的模型虽然在离线验证中改善了排序，但开发集 Top-1 仍为 56.4%（相对基线 −1.8
个百分点，配对 95% 区间 [−3.7, 0.0]），Top-10 为 84.2%。为排除仅仅欠训练，又在不改数据、
网络、reward、损失或推理参数的前提下，将该模型训练延长至 2,000 step：held-out guidance
排序准确率从长训练 control 的 59.07% 提至 63.75%，但开发集 Top-1 仍为 56.7%（相对基线
−1.5 个百分点，配对 95% 区间 [−3.3, +0.3]），Top-10 为 83.8%。因此没有候选通过“Top-1 不下降且深层指标改善”的确认集门槛，
确认集和最终验证集尚未使用，默认采样继续关闭 guidance。完整协议、时间和全部配对统计见
`new_docs/dgm_evaluation_v2.md`。

**2026-08-11 reward 审计与校准更新。** 对五时间点 guidance 的已有终点做只读的 canonical target
匹配后，forward-beam reward 在 200 个 held-out 反应上有 0.6971 的全局正确性 AUC、0.7308 的同一
共享状态组内 AUC，说明它有稳定但不充分的方向；46.05% 的错误终点仍得到正 reward。随后在一个完全
隔离的 200-reaction 训练 holdout 上，P1 线性校准把**全局** AUC 从 0.7073 提到 0.7655，但同一共享状态
组内 AUC 仅从 0.6836 到 0.6843，且同候选池 Top-1/Top-3 从 41.5/76.5% 降到 40.0/73.0%。预注册的 P2
仅追加 teacher-forced likelihood 后，同组 AUC 升至 0.7093、误报率略降，但同池 Top-1/Top-3 仍降至
41.0/73.0%。两次均未通过 gate，reward 校准支线已经关闭：不重训 guidance、不跑开发集，也不扫描更多
endpoint 特征或超参数。协议、防泄漏规则和完整结果见 `new_docs/dgm_reward_quality_protocol.md`。

本文面向本项目的实际实现，说明如何把
`Discrete Guidance Matching (DGM)` 适配到当前的 Edit Flows 推理流程。文中把
论文中的严格公式和第一阶段可落地的工程近似明确区分，避免把一个启发式加权方法误称为
“exact DGM”。

## 0. 先给出结论

当前已有的 Edit Flows checkpoint 不需要重新训练。DGM 是训练完成后的推理 guidance：

```text
冻结基础 Edit Flows pθ
        │
        ├── 生成基础样本并计算 reward R(c, y)
        │                 │
        │                 └── 训练 guidance model hψ
        │
        └── 推理时：基础编辑分布 × guidance 权重 → guided 编辑分布
```

这里有三个不同对象：

1. **基础模型 `pθ`**：当前 `checkpoint_step600000.pt`，输入 product，生成 reactants。
2. **reward evaluator `R`**：判断一个完整反应物候选有多好，可以是 RDKit 规则、forward
   reaction model 或其他独立化学评分器。它不是 guidance model。
3. **guidance model `hψ`**：学习“当前中间状态下，各个候选后续对终点 reward 的预期”，
   推理时把这个预期转成权重。

推荐的总顺序是：

```text
先冻结并测量基础 baseline
→ 先做 reward 接口和 synthetic 验证
→ 再做便宜的 validity reward
→ 再训练 guidance model
→ 先接普通 Euler，再接 Euler-Beam/SMC
→ 最后接昂贵的 forward-model reward
```

不要一开始就同时引入 forward model、learned guidance、beam 参数和新的 reward 组合，
否则即使指标变化，也无法判断究竟是哪一部分造成的。

---

## 1. DGM 的核心对象

设产物为 `c`，反应物为 `y`。

### 1.1 基础生成分布

当前 Edit Flows 近似的是：

```text
pθ(y | c)
```

它是推理时的基础分布，也可以叫 source/proposal distribution。它不是完美的真实分布，
因为它可能生成非法 SMILES、错误反应物，或者没有覆盖某些正确答案。

数据集中的真实反应物可以帮助训练基础模型和 reward evaluator，但不能把基础模型输出
直接称为理想目标分布。

### 1.2 reward-tilted 目标分布

DGM 不要求我们显式写出一个完整的目标分布 `q(y|c)`。我们选择一个非负 reward：

```text
R(c, y) >= 0
```

然后定义：

```text
qβ(y | c) = pθ(y | c) Rβ(c, y) / Zβ(c)
```

其中：

```text
Zβ(c) = Σ_y pθ(y | c) Rβ(c, y)
```

是归一化常数。于是：

```text
qβ(y|c) / pθ(y|c) ∝ Rβ(c,y)
```

常数 `Zβ(c)` 不需要计算，因为采样时会重新归一化。

常用的 reward 形式是：

```text
Rβ(c,y) = exp(β S(c,y))
```

`S` 是化学评分，`β` 是 guidance 强度。`β` 太大可能导致所有粒子集中在少数候选上，
使 ESS、Oracle 或 Top-10 下降。

### 1.3 reward 不能使用 test target

“准确率更高”可以作为研究目标，但不能在 test 推理时用“是否等于 test target”直接
构造 reward。这会发生标签泄漏。

可以使用：

- RDKit 合法性、价态、原子守恒等不读取 target 的规则；
- 只用训练集训练的 forward reaction model；
- 在 validation 上训练/校准的可行性模型；
- 从训练/validation 数据估计的化学一致性分数。

正式 test 只用于最后报告 Top-1～10、Oracle、invalid 等结果，不能参与 reward 或
`β` 调参。

---

## 2. DGM 的完整 pipeline

### 2.1 训练前：冻结基础模型和定义 reward

先固定：

```text
checkpoint
seed
n_steps
R/K/M
batch_size
TF32/FP32
```

当前建议基础配置仍为已经验证过的：

```text
R9K1M2
n_runs=9
n_branches=1
n_children=2
n_steps=100
seed=42
stochastic_noop
temperature=1.0
bonus=0.5
```

reward 先只选一个主来源。每个 reward 都要记录版本、数据边界、数值范围和计算成本。

### 2.2 训练 guidance model：数据如何产生

对训练或 validation 中的 product `c`，先用冻结的基础模型采样完整反应物 `y`：

```text
y ~ pθ(. | c)
```

第一版应使用普通 Euler 的独立采样来近似 `pθ`，不要先经过 beam 排序再拿结果训练
guidance。Beam 会改变样本分布，导致训练时的 proposal 不再是原始 `pθ`。

随后计算：

```text
r = R(c, y)
```

再使用已有 Edit Flows 的 path 构造中间状态：

```text
(product c, final reactant y)
→ alignment / aligned Z-space
→ 随机采样 t
→ 构造 x_t
```

每个训练样本最终包含：

```text
(c, x_t, t, y, r)
```

### 2.3 guidance model 输出什么

论文中的 guidance 不是一个标量，而是对每个候选最终 token 的正值权重：

```text
H_t[d, s]
```

它的含义是：

> 当前状态是 `x_t` 时，如果位置 `d` 最终取 token `s`，完整结果的 reward 期望是多少？

网络至少应看到：

```text
product c
current state x_t
time t
```

只输入 `x_t` 通常不够，因为同一个中间序列在不同 product 条件下可能对应不同的正确
反应物。

第一版工程实现可输出：

```text
H_ins(i, token)
H_sub(i, token)
H_del(i)
```

更接近论文的严格版本则在 aligned Z-space 输出：

```text
H[d, token_or_gap]
```

所有权重必须为正，可以使用：

```python
H = softplus(raw_H) + epsilon
```

#### 推荐的第一版 guidance model 架构

第一版不直接改造当前 10 层的 Edit Flows 主模型，而是训练一个独立、较小的
product-conditioned guidance adapter。这样可以冻结基础 checkpoint，也便于做
`guidance off/on` 的严格对照。

推荐结构如下：

```text
product tokens c
    → product encoder：2 层 Transformer，hidden=256，heads=8，FFN=1024
    → masked mean/pooling 得到 product context

current tokens x_t + time embedding t + product context
    → state encoder：4 层 Transformer，hidden=256，heads=8，FFN=1024
    → 每个当前序列位置的 hidden state

hidden state
    ├─ insert head：MLP 256 → 256 → V_edit，softplus 得 H_ins
    ├─ substitute head：MLP 256 → 256 → V_edit，softplus 得 H_sub
    └─ delete head：MLP 256 → 256 → 1，softplus 得 H_del
```

其中 `V_edit` 使用当前 Edit Flows 的词表大小，不能使用 Molecule Transformer 的词表。
插入位置的定义也必须沿用当前 Euler 的位置约定。第一版总参数量预计约 5～10M，远小于
基础 Edit Flows checkpoint；训练时冻结基础模型，只更新 guidance 参数。

这不是论文固定坐标公式的最终 Z-space 实现，而是可验证的 action-level 近似。只有当
它通过概率闭合和 synthetic 测试后，才考虑把 product encoder/state encoder 改成更严格
的 aligned Z-space guidance。

### 2.4 guidance model 的训练目标

对终点真实 token 对应的 guidance 输出，使用正值 Bregman loss：

```text
L_h = H - r log(H)
```

在满足论文假设时，最优解是对应候选的条件 reward 期望。实际训练时需要：

- `r` 作为常量，不对 reward 反向传播；
- 对 padding、GAP 和无效坐标做 mask；
- 对 `H` 做正值和数值稳定保护；
- 记录 reward 分布、`H` 分布和校准误差；
- 用不重叠的 validation products 检查 guidance 是否真的能预测未来 reward。

如果有高质量目标样本，可以额外加入 guided posterior 的交叉熵正则；但第一版不加入，
避免把普通监督学习误判为 DGM 收益。

#### 训练数据量和时间估计

guidance 训练不需要把 800,000 条 augmentation 行全部复制成独立样本。第一版可以对每个
unique product 生成一个基础 Euler 终点 `y`，保存 `(c, y, reward)`，训练时在线重新采样
`t` 和 `x_t`。这样同一个终点可以被多个时间点复用，也避免把同一 product 的 augmentation
随机拆到 train/validation 两侧。

在 RTX 3090 上的保守估计如下；实际时间以阶段 3 的小批量 benchmark 为准：

| 阶段 | 数据规模 | guidance 训练 | 其他主要时间 |
|---|---:|---:|---:|
| synthetic smoke | 已知 toy 分布 | 小于 5 分钟 | 小于 5 分钟 |
| pilot | 约 2,000 products | 5～20 分钟 | 基础采样/RDKit 约 5～20 分钟 |
| validation 规模 | 约 10,000 products | 10～45 分钟 | 基础采样约 20～90 分钟 |
| train 规模 | 约 20,000～40,000 unique products | 20～90 分钟 | 基础采样约 1～4 小时 |

这些估计不包含旧版 Molecular Transformer 的兼容改造。初步 benchmark（3090 上仍有
约 20% `alive.py` 占用）显示，默认第一版模型约 5.26M 参数，batch=64、product/state
长度约 96/128 时，一次 forward+backward 约 40ms；因此 guidance 模型本身通常不是
主要瓶颈，基础终点采样、SMILES reward 和 forward reward 才可能占主要时间。正式估计
仍以阶段 3 的真实长度 benchmark 为准。

### 2.5 DGM 推理 pipeline

每个 Euler 时间步执行：

```text
1. 基础 Edit Flows forward
   → λ_ins, λ_sub, λ_del, Q_ins, Q_sub

2. guidance forward
   → H_ins, H_sub, H_del 或 Z-space H

3. 构造基础动作速率
   u_ins(i,a) = λ_ins(i) Q_ins(i,a)
   u_sub(i,a) = λ_sub(i) Q_sub(i,a)
   u_del(i)   = λ_del(i)

4. 用 H 重加权并归一化
   guided action ∝ base action × H

5. 按 guided 分布执行编辑

6. 重复直到 t=1
```

论文严格版本是在离散终点 posterior 上做重加权，再由对应 conditional rate 产生
CTMC 跳转。直接把当前的 `Q_ins/Q_sub` 乘上 `H` 是第一阶段的 action-level 近似，
在完成 Z-space 映射证明之前不能称为 exact DGM。

推理仍可使用 `n_steps=100`。DGM 改变的是每一步往哪里跳的概率，不改变训练时的
`total_steps=600000`。

### 2.6 关闭 guidance 时必须退化为 baseline

实现必须满足：

```text
guidance off → 与当前 Euler 完全一致
guidance weight 为常数 → 与当前基础分布一致
```

这是最重要的回归测试。否则后续指标变化可能来自 seed、编辑概率或路径实现错误，
而不是 DGM。

---

## 3. 适配到当前 Edit Flows 的具体难点

### 难点一：我们的序列长度会变化

论文通常按固定坐标讨论 token 状态，而我们的动作包括：

```text
insert(i, token)
substitute(i, token)
delete(i)
```

插入和删除会改变长度，因此必须决定 guidance 是：

1. 在 aligned Z-space 的固定位置上计算；还是
2. 直接对 insert/substitute/delete action 加权。

第一阶段建议先实现第 2 种，快速验证工程闭环；同时保留清晰的“approximate”标记。
如果它有收益，再实现 Z-space 版本以接近论文理论。

### 难点二：当前模型输出的是 rate + token proposal

当前输出不是一个单独的 token probability，而是：

```text
编辑类型的总速率 λ
编辑 token 的条件分布 Q
```

因此不能只修改 `Q` 就结束。需要明确：

- guidance 是否同时影响 insert/substitute/delete 的相对概率；
- rate 是否需要重新归一化；
- Poisson event probability `1-exp(-hλ)` 如何保持合法；
- path log-prob 应该记录 guided proposal 还是原始 proposal。

如果按新分布采样，却仍按旧分布记录 log-prob，后续 beam/SMC 权重会不一致。

### 难点三：product 条件必须保留

当前 checkpoint 没有 `origin_embedding`，且 `use_origin_mask=False`。编辑进行后，
仅看当前状态可能无法恢复完整 product 信息。

guidance model 第一版应显式接收 product；如果基础模型也需要 product condition，
则要在推理状态中单独保存 product，而不能假设它仍然存在于被编辑序列中。

### 难点四：训练 guidance 的样本必须来自正确 proposal

训练密度比 guidance 时，样本应近似来自 `pθ`。如果先用 Euler-Beam 的排序和剪枝，
样本分布已经变成 beam 分布，训练目标会被改变。

因此：

- guidance 数据生成优先用独立 Euler；
- beam 只作为最终推理对照；
- 如果以后必须使用其他 proposal，必须同时记录 proposal log-prob 并做 importance correction。

### 难点五：reward 计算可能比基础模型更慢

RDKit 规则较便宜；forward Transformer 可能对每个候选都需要一次 forward。

第一版要：

- 批量计算 reward；
- 缓存相同 SMILES；
- 只在终点计算 terminal reward；
- 记录 reward 调用次数和 wall time；
- 不要先在每个中间候选上逐个调用 forward model。

### 难点六：权重可能坍缩

如果 `β` 或 reward 尺度过大：

```text
少数粒子权重占据全部质量
→ ESS 接近 1
→ 多样性下降
→ Top-10/Oracle 反而下降
```

因此每次实验都要报告：Top-1～10、Oracle、invalid、true unique、ESS、reward 分布、
resampling 次数和 wall time，不能只看 Top-1。

---

## 4. 我们必须遵守的实操推进顺序

以下步骤是固定顺序。前一步没有通过，就不进入下一步。

### 阶段 0：冻结 baseline，建立可复现基线

工作内容：

1. 固定新 checkpoint、seed、`n_steps=100`、TF32/FP32、batch size。
2. 固定 R9K1M2 和当前评分脚本。
3. 在 validation-A 保存 baseline predictions、metadata 和完整指标。
4. 为同一批 product 保存基础 Euler 的独立样本。

通过标准：

- 相同 seed 能复现 predictions SHA；
- guidance 关闭时结果与当前 Euler 一致；
- 记录 Top-1～10、Oracle、invalid、true unique、wall、显存。

本阶段不训练任何新模型。

### 阶段 1：实现 reward 接口和 synthetic DGM

先不接真实化学 reward，构造一个已知目标分布的小型离散 toy task：

```text
已知 p(x)、已知 R(x)、已知 q(x) ∝ p(x)R(x)
```

验证：

- `p × R` 的归一化是否正确；
- guidance 输出是否为正；
- guidance 关闭是否退化为 p；
- 采样频率是否接近已知 q；
- ESS 和 importance weight 是否正确。

如果 synthetic 不能通过，不进入真实 SMILES 实验。

#### 阶段 1 当前实现记录（2026-08-07）

已完成低风险的代数和 reward 接口子阶段：

- 新增 `edit_flows/guidance/dgm.py`：正值 guidance、`p × H` 后验重加权、正值
  Bregman loss；
- 新增 `edit_flows/guidance/rewards.py`：不读取 test target 的 RDKit validity reward；
- 新增 `retro_tokenized_validity_reward()`：先去掉 token 空格并执行 Edit Flows 的
  inverse global alignment，再交给 RDKit，避免把序列化格式误判为化学非法；支持调用方
  cache，适配 augmentation 中的重复候选；
- 新增 5 个 synthetic/接口测试；全部通过；
- 现有 `euler_smc` 和 Transformer 相关回归测试 12 个全部通过。
- 第一版 `ProductConditionedGuidance` smoke 已通过：默认配置参数量约 5.26M，随机输入
  的输出形状为 `H_ins[B,L,V]`、`H_sub[B,L,V]`、`H_del[B,L,1]`，且全部为有限正值；
  guidance 模型 forward/backward 和参数规模测试通过。
- 新增已知两步 categorical chain 的 synthetic rollout：用精确条件 guidance 恢复已知
  terminal `q ∝ pR`，并检查 proposal 下的 normalizer estimate 和 ESS；该测试通过。

这还不等于完成 guidance model 训练或真实 Euler 接入。下一步进入阶段 2 的 validation
SMILES reward/terminal twisting，同时保留 synthetic 测试作为回归门槛。

### 阶段 2：先接便宜的 validity reward

实现一个纯函数：

```text
terminal_reward(products, reactants) -> finite reward tensor
```

第一版只使用 RDKit 合法性、价态等不依赖 test target 的规则。暂时不追求 Top-1 提升，
主要验证：

- reward 能否批量计算；
- invalid 是否如预期变化；
- reward 是否导致 ESS 坍缩；
- guidance 接口、metadata 和路径概率是否闭合。

这一步可以先使用 terminal twisting，不必立即训练 learned guidance。

#### 阶段 2 当前验证记录（2026-08-07）

已在 validation 的前 200 个原始反应上复用一组已有的 R9K1M2、`n_steps=100` 预测做
reward-only 诊断；reward 只读取采样终点，不读取 target，target 只在之后的评分步骤用于
报告指标。预测文件共有 36,000 行（200 reactions × 20 augmentations × 9 candidates）。

- 正确流程是 `retro_tokenized_validity_reward()`：先 compact token，再 inverse global
  align，最后调用 RDKit；直接对带空格的原始行评分会把表示格式错误计入 invalid，已明确
  禁止该用法；
- 36,000 个终点中 31,700 个 RDKit-valid，valid rate 为 **88.06%**；去重后 15,850
  个 normalized candidates，cache hit rate 为 **55.97%**；
- 单进程 reward 评估耗时约 **5.89 s**，开启调用方 cache 后约 **2.82 s**（同一进程，含
  global-align 与 RDKit）；输出完全一致；
- 同一预测文件的评分对照为 Top-1/2/3/10 = **70.0/84.5/90.5/93.5%**，Oracle-any
  **95.5%**。这些是 baseline 诊断值，不是 validity twisting 带来的提升；尚未运行
  reward-guided sampler，因此不把这组数字误称为 reward-guided 提升。
- 新增 `terminal_twist_target_increment()` 与 `apply_terminal_twist()`：在已有终点粒子上
  只施加 `exp(βR)` 的 target/proposal 比值；synthetic 测试验证指数倾斜、`β=0` identity
  limit、ESS/evidence 以及输入校验均通过。它仍是独立 adapter，不会改变默认
  Euler/Euler-Beam。
- 新增隔离入口 `run_euler_smc_terminal_twist()`，真实 smoke 使用 validation 前 5 个原始
  反应、9 粒子、100 steps、`β=1`、CUDA/TF32。终点 ESS 为 **7.29–9.00**（均值约 8.43），
  无重采样；proposal 与 terminal-twist 的 Top-1～9、Oracle **80%**、valid rate **82.22%**
  完全相同，两个 rollout 合计约 **13.6s**。这是预期的：terminal-only reward 只重排已采到
  的终点，不能增加 Oracle 覆盖；在该小样本上 validity 排序也没有改变前 9 名。

结论：reward 接口、表示转换、批量评估、缓存和纯 terminal-twisting 数学 smoke 已通过；
真实多步 adapter 也已通过接口/数值 smoke，但 validity terminal reward 暂不作为准确率
改进方法。下一步转入阶段 3，训练能在中间状态预测未来 reward 的 guidance model；仍需
固定总候选预算并报告 proposal/target 权重、ESS、invalid 和 Top-1～10。若只改变 reward
而没有固定总候选预算，不能把结果解释为 DGM 收益。

### 阶段 3：生成 guidance 训练数据

使用冻结的基础 Euler 生成 train-guidance 和 validation-guidance 样本：

```text
product c
→ y ~ pθ(.|c)
→ reward R(c,y)
→ aligned path
→ 随机 t 和 x_t
```

数据划分必须按 product 划分，不能让同一个 product 的 augmentation 同时出现在训练和
validation 两边。

需要保存：

```text
product tokens
generated reactant tokens
intermediate state x_t
t
reward
alignment / mask
seed
baseline checkpoint id
```

#### 阶段 3 当前实现记录（2026-08-07）

已完成不依赖 GPU 的数据格式和生成器实现：

- `edit_flows/guidance/data.py` 提供 `sample_intermediate_states()`，沿用
  `opt_align_xs_to_zs → sample_cond_zt → rm_gap_tokens`，输入/输出都显式保留 BOS，返回
  可直接送入 action-level guidance adapter 的 `x_t`；同时提供记录校验、`.pt` 保存/加载和
  padding collate；
- `scripts/generate_guidance_data.py` 只调用普通 Euler，按 `augmentation` 先取每个原始
  product 的一个代表行，绝不读取 target，也不使用 Euler-Beam；每条记录保存 product、
  `x_t`、`t`、generated terminal、reward、source/sample/time index、采样和 coupling seed；
- 默认 reward 是 RDKit validity，tokenized/global-aligned 终点会先 compact + inverse
  align；metadata 记录 checkpoint、scheduler、n_steps、n_samples、time_samples、seed 和
  batch RNG 作用域；
- CPU 单元测试 `14 passed`（含 4 个 data tests），脚本 `--help` 和静态编译通过；真实
  checkpoint smoke 已完成：5 个 validation product、20 steps、每条轨迹取 2 个状态，共
  10 条记录，reward positive **4/10**、均值 **0.4**，GPU wall **约 4.5s**；`.pt` 加载、
  padding/collate 和 metadata 检查通过。
- 正式 train 生成首次运行到第 120/626 批时发现一条终态第 0 列被采样编辑、丢失 BOS。
  这是采样器的结构性边界问题，不是输入数据损坏：训练耦合始终固定 BOS，而 Euler 原先
  只屏蔽 PAD。已在普通 Euler 和 Euler-Beam 的 action sampler 中统一屏蔽位置 0，并加入
  回归测试；失败批次（原始 product index 8019）修复后短复现不再产生非 BOS 终态。这样
  既保证 guidance 数据格式合法，也避免 DGM 与 baseline 使用不同的序列语义。
- 修复后正式数据已完整生成并通过全量 CPU 审计：train 为 **40,003 products / 80,006
  records**，GPU wall **1475.5s（24.6min）**；validation 为 **5,001 products / 10,002
  records**，GPU wall **182.5s（3.0min）**。两份数据均为每个 product 两个中间时间点，
  `t=0.0099..0.9901`，所有 product/state/terminal 的第 0 列为 BOS，记录与 product
  索引一一对应且无跨 split 重叠。RDKit validity reward 为二值：train positive
  **70,180/80,006 = 87.72%**，validation **8,778/10,002 = 87.76%**；metadata、
  scheduler、checkpoint、seed 和 record 数均通过检查。

推荐的首次 smoke（运行前先停止 `alive.py`，结束后立即重启）是：

```bash
python scripts/generate_guidance_data.py \
  --checkpoint new_checkpoints/checkpoint_step600000.pt \
  --products_file "datasets/USPTO_50K_PtoR_aug20_#global#/val/src-val.txt" \
  --output /tmp/dgm_guidance_val5.pt \
  --augmentation 20 --max_products 5 \
  --n_steps 20 --n_samples 1 --time_samples 2 \
  --batch_size 2 --device cuda --seed 42
```

该 smoke 只验证数据条数、reward 分布、状态长度、seed/metadata 和运行时间，不使用
validation target 选择超参，也不训练 guidance model。阶段 3 的“接口、正式生成和结构审计”
门槛已通过，下一步进入阶段 4 真实 guidance 训练。

### 阶段 4：训练最小 guidance model

第一版建议冻结基础 Transformer，只训练一个 guidance adapter/head：

```text
(product, x_t, t)
→ small guidance head
→ H_ins, H_sub, H_del
```

不能把一个 scalar reward 无差别广播到所有 action：这样只能学习“当前状态的平均 reward”，
无法告诉模型哪个后继更好。当前实现先用 `x_t` 与 terminal 的 optimal alignment 生成
insert/substitute/delete 稀疏 action mask，再对选中的 action 使用 `background + reward`、
其余合法 action 使用小的正 background，训练正值 Bregman loss。它仍是 action-level
近似，不宣称是严格 Z-space DGM。

必须评估：

- guidance loss；
- reward 与 `H` 的相关性；
- 不同时间 `t` 的校准误差；
- 高 reward 样本是否得到更高 guided probability；
- guidance forward 的耗时和显存。

#### 阶段 4 当前实现记录（2026-08-07）

- 新增 `edit_flows/guidance/targets.py`：从 state/terminal 构造三类 action mask，并把
  reward 转为正值 action targets；
- 新增 `edit_flows/guidance/training.py`：统一 forward、mask、Bregman loss、梯度裁剪和
  train/eval step；不修改基础 Edit Flows checkpoint；
- 新增 targets/training 测试；当前 DGM/SMC/模型相关 CPU 回归共 **35 passed**；
- 这一步修正了“scalar reward 广播到所有 action”的潜在正确性问题。正式 train/validation
  数据已生成；alignment mask 在 CPU 构造后再搬到
  GPU，避免每个训练 batch 在 GPU 上执行 Python/DP 对齐。
- 新增 `scripts/train_guidance.py`：独立加载 guidance `.pt`，冻结基础 checkpoint 不参与
  训练，使用 AdamW、梯度裁剪、validation 截断和 TensorBoard `train/*`、`validation/*`
  标量，并保存独立 `guidance_final.pt`、`config.json`。在 CPU tiny smoke（2 steps、
  hidden=16、1+1 层）上运行 **0.186s**，loss `0.5374→0.5215`，validation loss
  `0.5129→0.4999`；TensorBoard 依赖已在新 `ef` 环境安装，`pip check` 无冲突。
- 新增 selected-action guidance 均值及其与 reward 的 batch Pearson 诊断。真实 CUDA
  10-step smoke 用时 **2.38s**，loss `0.7446→0.2913`，validation loss `0.3445→0.2725`；
  step-10 selected guidance mean 为 train **0.4800**、validation **0.4249**，相关系数
  分别 **-0.371/-0.024**。由于只有 10 steps 且相关性按 batch 计算，这只是观测链路通过，
  不能据此宣称 guidance 已学到有效 reward 方向；正式训练需在完整 validation 上重新统计。
- 1,000-step pilot 暴露并验证了 action sparsity 的影响：原等权背景损失的全 validation
  selected-action H 均值仅 **0.0064**、selected-only reward-H 相关 **0.145**（全体含无
  selected action 行为 **0.046**）；把背景项降为 `background_loss_weight=0.01` 后，
  selected H 均值升至 **0.8277**、selected-only 相关升至 **0.327**，H 范围约
  `8.6e-5..1.328`。两次均为 1,000 steps、同 seed、batch=32，训练 wall **150.5s vs
  155.1s**，峰值 allocated 均约 **1.16GB**。balanced loss 的 raw validation loss
  约 **0.801**，不能与等权 loss **0.00417** 直接比较；当前保留 balanced 版本，后续以
  selected action 校准和真实 guided rollout 判断收益。
- 使用 balanced loss 完成一 epoch（2,500 steps）的真实训练，checkpoint 保存在
  `/root/autodl-tmp/dgm_guidance_runs/epoch1_balanced/guidance_final.pt`（不进 Git）。训练
  wall **392.7s**，峰值 CUDA allocated/reserved **1.17/1.91GB**；validation 子集 loss
  从 **0.818** 降到 **0.775**。全 validation 10,002 条记录中 8,685 条有 selected
  action，selected H 均值 **0.8629**，selected-only reward-H 相关 **0.3599**；按 t 的
  四个区间相关为 **0.361/0.369/0.355/0.355**，H 全局范围 `0.00154..1.287`。评估
  forward wall **21.3s**。这满足阶段 4 的训练稳定性与时间校准门槛，但不代表 Top-k 已提升；
  下一步必须做 ordinary Euler guidance off/on 对照。

推荐的真实数据训练 smoke（正式数据生成完成后执行）为：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ef
unset OMP_NUM_THREADS
python scripts/train_guidance.py \
  --train_data /root/autodl-tmp/dgm_guidance_data/train_validity.pt \
  --val_data /root/autodl-tmp/dgm_guidance_data/val_validity.pt \
  --output_dir /root/autodl-tmp/dgm_guidance_runs/smoke \
  --device cuda --batch_size 32 --max_steps 10 --val_interval 5 --val_batches 2 \
  --background_loss_weight 0.01
```

通过阶段 4 的最低门槛是：loss/validation loss 有限、梯度有限、输出权重为正、validation
可运行且 checkpoint/TensorBoard 可读取；准确率提升要等阶段 5 的 off/on 推理对照，不能
把训练 loss 下降直接称为 DGM 指标收益。

### 阶段 5：先接普通 Euler，不接 Beam

在普通 Euler 上做最小 A/B（这里的 `n_samples=3` 是 3 条独立 Euler 轨迹，不是
Euler-Beam 的 `R3K3`）：

```text
同一 checkpoint
同一 product
同一 seed
同一 n_steps
同一总样本预算
guidance off vs guidance on
```

先验证 action-level approximate guidance 的正确性，重点检查：

- `H=constant` 是否严格退化；
- 采样概率和记录的 log-prob 是否一致；
- Poisson event probability 是否仍在合法范围；
- guidance 是否只改变预期方向，而不是改变序列操作语义。

#### 阶段 5 当前实现与 smoke（2026-08-08）

- 新增 `edit_flows/guidance/sampling.py` 的 `apply_action_guidance()`：把
  `u=λ·Q·H^β` 分解到 insert/substitute/delete 三类 action，再按每个位置的原始总
  edit rate 归一化。这样 `β=0` 或 H 为常数时 rates/posteriors 都严格保持 baseline；
  非常数 H 只改变 action 类型/token 的相对质量，不改变该位置总体编辑强度。纯函数测试
  覆盖 identity、常数 H、rate 保持和 posterior 归一化。
- `sample_euler()` 新增可选 `guidance_model/guidance_product/guidance_beta`，
  `scripts/sample_retro.py` 新增 `--guidance_checkpoint/--guidance_beta`，当前只允许
  ordinary Euler；guidance checkpoint 的 vocab、配置、SHA 和 beta 写入 sampling metadata。
- 同时修复 sample_retro 普通 Euler 的 seed 语义：`--seed` 现在实际调用 global torch RNG，
  metadata 标记 `seed_applied_to_sampler=true`；这使 baseline 与 beta=0 可以做字节级对照。
- 固定新 checkpoint、seed=42、100 steps、batch=20、20 个输入行（一个 augmentation
  block）、每 product 两个输出：baseline 与 guidance beta=0 的 prediction SHA 完全相同，
  `cmp` 通过；baseline wall **2.376s**、beta=0 **2.317s**。beta=1 改变 25/40 行，wall
  **2.952s**、peak allocated/reserved **0.297/0.415GB**，输出 40 行且均保留正常格式；
 该单 reaction block 的 RDKit-valid 为 baseline **25/40 (62.5%)**、beta=1 **24/40
  (60.0%)**，Top-k 不能在一条 reaction 上解释，暂不据此淘汰 guidance。随后在未参与
 training 的 validation reaction 200–399（200 个完整反应、ordinary Euler 的
 `n_samples=3`、100 steps、seed=42）完成了固定预算 off/on：

| 配置 | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Oracle | invalid@1/2/3 | mean final rank | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| guidance off | 51.0% | 66.5% | 72.0% | 77.0% | 83.5% | 86.5% | 11.675/11.225/12.425% | 2.474 | 253.0s |
| validity guidance, β=1 | 51.0% | 64.5% | 71.5% | 78.5% | 83.5% | 86.5% | 11.875/11.875/10.875% | 2.416 | 381.2s |

guided 输出约 **1.51×** baseline wall；Top-1、Top-3、Top-10 和 Oracle 没有提升，Top-2
下降 2 个百分点，只有 rank tail 和 invalid 分布出现小幅变化。因此 mechanics/metadata/
identity 门槛通过，但 validity reward 的真实准确率门槛失败，不把它设为默认 guidance，也
不继续在 Euler-Beam 上叠加这个弱 reward。下一步优先检查阶段 7 的独立 forward reaction
model：先完成 checkpoint/tokenizer/方向 smoke，再封装 terminal forward reward。

### 阶段 6：固定预算下接 Euler-Beam / Euler-SMC

只有阶段 5 通过后，才把 guidance 放入当前 R9K1M2：

- 固定总 child budget；
- 不同时调整 `R/K/M`、temperature 和 changed-state bonus；
- 比较 guidance off/on；
- 报告 ESS、祖先多样性、merge/resample 次数。

此时要明确 guidance 是：

```text
先重加权 proposal，再生成 child
```

还是：

```text
生成 child 后再用 reward rerank
```

二者不能混写成同一种方法。

### 阶段 7：接入 forward reaction model reward

只有 validity reward 和 guidance mechanics 通过后，才接 forward reward。

推荐顺序：

1. 先确认已有可用的 Molecule Transformer/forward reaction checkpoint；
2. 确认它的训练数据、tokenizer、词表、输入输出格式和 checkpoint 配置；
3. 在独立 validation 上检查 forward model 的基本准确率；
4. 封装批量、缓存的 `forward_reward`；
5. 只做 terminal reward；
6. 在 validation-A 调整很小的 `β` 候选集合，在 validation-B 复核；
7. 最后才在 test 上运行一次。

#### 阶段 7 当前实现与 validation 诊断（2026-08-08）

- 新增 `edit_flows/forward/molecular_transformer.py`：在不安装旧版 `torchtext`/OpenNMT
  的前提下重建 OpenNMT 0.4.1 的 encoder、decoder、attention、LayerNorm、位置编码和
  generator，并对 checkpoint 做 strict state-dict 加载；新增官方 SMILES tokenizer 和
  `scripts/forward_model_smoke.py`。
- 新增 `edit_flows/forward/reward.py`：批量把 Edit Flows `#global#` 候选转换为普通 SMILES，
  用 teacher-forced、长度归一化的 `log p(product | reactants)` 作为 raw forward score，
  支持 caller-owned pair cache；`positive_forward_reward()` 只在需要 DGM 正值目标时做
  `exp(score / temperature)`，不把负 log-likelihood 直接广播成 action target。
- 兼容加载、tokenizer 和 reward 单元测试 **4 passed**。200 个 validation unique reactions
  的方向 smoke 中，正确的 reactants→product 平均 score **-2.0448**，交换方向
  product→reactants 为 **-3.3193**，正确方向胜出 **92.0%**；因此 checkpoint、词表和
  方向达到进入 reward 实验的最低门槛。
- 600 个已有 validation 候选的 batch reward smoke 使用 460 个唯一 pair，CUDA wall
  **0.504s**；完全命中 cache 的重复调用 **0.018s**，输出逐元素完全一致。
- 作为“是否可以直接 rerank”的反证诊断，在 validation reaction 200–399 的 12,000 条
  ordinary-Euler `n_samples=3` × augmentation 候选上直接按 raw forward score 排序，Top-1/2/3/5/10 为
  **36.0/51.5/64.0/72.0/84.5%**（累计），baseline 为
  **51.0/66.5/72.0/77.0/83.5%**，Oracle 均为 **86.5%**。它只略改善 Top-10，明显损害
  Top-1/Top-2，说明 forward score 目前是有方向信息但未校准的弱 reward；不能把直接
  rerank 设为方法结论，也不能据此在 test 上调温度。下一步是在 validation-A/B 固定小的
  reward temperature/β 候选，先训练或校准 terminal guidance，再做 off/on 对照。

##### 阶段 7 pilot guidance 与 β 对照（2026-08-08）

为避免把 forward score 直接当作排序器，先用生成的 `train_forward_t1.pt` /
`val_forward_t1.pt` 训练独立的 action-level guidance adapter。pilot 使用 batch 64、
`background_loss_weight=0.01`、1,000 steps；训练 wall **219.6 s**，峰值显存
allocated/reserved **2.23/5.11 GB**。在完整 validation guidance 数据（10,002 条）
上，selected-action guidance 与 forward reward 的 Pearson 相关为 **0.4437**，说明
forward reward 不是常数且可以被 adapter 学到；这只是可学习性门槛，不代表采样准确率
已经提升。

随后冻结 checkpoint、seed=42、ordinary Euler、`n_samples=3`、100 steps、batch 64 和
同一 validation reaction 200–399（200 个完整反应、12,000 条输出），只比较 guidance
强度。baseline 与两个 forward-guided 结果如下：

| 配置 | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Oracle | invalid@1/2/3 | mean final rank | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| guidance off | 51.0% | 66.5% | 72.0% | 77.0% | 83.5% | 86.5% | 11.675/11.225/12.425% | 2.474 | 253.0 s |
| forward guidance, β=1.00 | 50.0% | 67.0% | 71.5% | 77.0% | 80.5% | 83.0% | 11.850/11.500/11.775% | 2.271 | 379.2 s |
| forward guidance, β=0.25 | 52.5% | 65.5% | 71.5% | 78.5% | 82.5% | 85.5% | 11.325/11.825/12.525% | 2.380 | 380.1 s |
| forward guidance, β=0.10 (2,500-step) | 53.0% | 66.0% | 71.5% | 77.0% | 83.0% | 86.0% | 11.975/11.500/12.850% | 2.494 | 380.0 s |

为排除 adapter 欠训练，又保持上述所有采样条件不变，将同一架构训练到 2,500 steps（两遍
train data）。训练 wall **644.5 s**、峰值显存 allocated/reserved **2.24/5.11 GB**，
最后一次完整 validation loss **0.3206**，旧口径（混入无目标 action 行）的
reward–guidance Pearson **0.5210**；后续审计给出修正口径。
使用该 checkpoint、β=0.25 的结果为 Top-1/2/3/5/10 **53.5/66.5/69.5/75.0/81.5%**，
Oracle **85.0%**，invalid@1/2/3 **12.900/11.700/11.875%**，mean final rank **2.400**，
sampling wall **378.8 s**。相对于 pilot，Top-1 再升 1.0 个百分点，但 Top-3、Top-10 和
Oracle 分别再降 2.0、1.0、0.5 个百分点；因此“继续训练即可恢复覆盖”的假设也没有得到
支持。

这里 β=0.25 的 Top-1 比 baseline 高 1.5 个百分点，但 Top-3 低 0.5、Top-10 低 1.0、
Oracle 低 1.0 个百分点；β=1.0 的 Top-10/Oracle 分别下降 3.0/3.5 个百分点。2,500-step
adapter 的 β=0.10 将 Top-1 提高 2.0 个百分点，同时 Top-3/Top-10 各低 0.5 个百分点、
Oracle 低 0.5 个百分点，属于覆盖基本保住但没有长尾改善的结果。所有 guided 运行的 wall
都约为 baseline 的 **1.50 倍**，主要来自每个 Euler step 的 guidance forward。50-reaction
预筛中 β=0.25/0.5 的 Top-1 都下降 4 个百分点，也没有形成稳定的全局改善信号。

因此阶段 7 当前只通过了“checkpoint、方向、批量接口和 guidance 可学习性”门槛，尚未
通过“固定预算下 Top-1 不回退且 Top-3/10 或 Oracle 稳定提升”的准确率门槛。pilot、2,500-step
训练和低强度 β=0.10 三组结果都未通过，说明瓶颈不只是欠训练或 β 过强。该 forward reward 不能进入默认
Euler/Euler-Beam；后续若继续，应转向独立 reward 校准、终点/候选级的受约束使用，或严格
Z-space 研究，并保留 baseline 与本轮结果作为对照，不能用 test target 选择 β。

##### 阶段 7 全链路审计与 5,000-step 复核（2026-08-08）

这次复核把“baseline 是否错了”、“reward 是否真的启用”、“2,500 step 是否足够”和
“当前 approximate DGM 本身限制在哪里”分开检查。

**Baseline。** validation reaction 200–399 的输入严格对应 `src-val.txt` 行区间
`[4000, 8000)`；评分使用 `--length 200 --target_offset 200`。sampling metadata 会交叉
检查 augmentation=20、输出 beam=3、12,000 行、文件 SHA 和 target offset。基础 checkpoint
为 `new_checkpoints/checkpoint_step600000.pt`，有效 `use_origin_mask=False`；seed=42 已实际
施加到 CPU/CUDA RNG。`guidance_beta=0` 与 guidance-off 的 predictions 曾通过字节级一致性
检查。没有发现会使当前 ordinary-Euler baseline 指标算错的实现问题。需要注意它是为了
隔离 guidance 使用的机制基线，不是项目最强的 Euler-Beam R9K1M2 基线。

**Guidance 数据。** `train_forward_t1.pt` 有 80,006 条记录、40,003 个 unique product，
但配置为 `n_samples=1, time_samples=2`：每个 product 只有一个 Euler 终态，两条记录只是同一
终态的两个中间时间，reward 完全相同；40,003 个 product 中同组 reward range 非零的数量为
**0**。validation 也同样是 5,001 个 product、10,002 条记录。换言之，当前训练没有见过
同一 product 下“不同终态、不同 reward”的对比，这是比 epoch 数更直接的数据瓶颈。

validation action mask 审计为：含 insert/substitute/delete 目标的行分别为
**85.33% / 15.89% / 1.63%**，**13.17%** 的行已与采样终态一致而没有任何目标 action；
157 个 batch 中有 **87** 个完全没有 delete 标签。旧相关系数把无目标 action 的行以
`selected_guidance=0` 混入，2,500-step 的旧日志值为 0.521；修正后只在有效行计算，
同一 checkpoint 的全 validation Pearson 为 **0.665**，按 batch 汇总为约 **0.657**。
该修复只改变诊断，不改变 loss、模型权重或 sampler。

**Reward 质量。** 当前 Molecular Transformer reward 在 validation 的 5,001 个单终态中，
真实 target 命中的 1,781 条平均 raw score 为 **-2.115**，其余 3,220 条为 **-2.749**；
pairwise discrimination AUC 只有 **0.5639**。正确候选的正值 reward 均值 **0.191**，错误
候选为 **0.163**，分布重叠很大。这与已有 raw-rerank Top-1 从 51% 降到 36% 的结果一致：
reward 有方向信息，但远不是可靠的 retrosynthesis correctness 判别器。

**更长训练。** 保持数据、架构、LR、batch、seed 和 loss 全部不变，从头训练 5,000 steps
（约 4 epoch），wall **1,277.3s**，峰值 allocated/reserved **2.24/5.12GB**。完整
validation loss 在 step 2,500/4,000/4,500/5,000 分别为
**0.320633 / 0.318892 / 0.317587 / 0.321113**；修正后的有效行 Pearson 则为
**0.6571 / 0.6703 / 0.6793 / 0.6801**。模型仍在学习排序相关性，但 final calibration loss
并不优于 4,500，证明只保存 final checkpoint 不可靠；训练脚本现已额外保存按 validation
loss 选择的 `guidance_best.pt`。

固定原 validation-200 协议的最终采样结果如下：

| 配置 | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Oracle | invalid@1/2/3 | mean final rank | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| guidance off | 51.0% | 66.5% | 72.0% | 77.0% | 83.5% | 86.5% | 11.675/11.225/12.425% | 2.474 | 253.0s |
| 2,500-step, β=0.10 | 53.0% | 66.0% | 71.5% | 77.0% | 83.0% | 86.0% | 11.975/11.500/12.850% | 2.494 | 380.0s |
| 2,500-step, β=0.25 | 53.5% | 66.5% | 69.5% | 75.0% | 81.5% | 85.0% | 12.900/11.700/11.875% | 2.400 | 378.8s |
| 5,000-step final, β=0.10 | 52.5% | 66.5% | 71.0% | 76.5% | 82.0% | 87.5% | 11.200/11.950/11.825% | 2.829 | 380.4s |
| 5,000-step final, β=0.25 | 54.0% | 66.0% | 70.5% | 77.5% | 82.0% | 85.0% | 12.375/11.550/12.050% | 2.371 | 381.3s |

5,000-step `β=0.25` 的 Top-1 比 baseline 高 3 点，但 Top-3、Top-10、Oracle 都低 1.5
点，invalid@1 高 0.7 点；`β=0.10` 只提升 Top-1/Oracle，却降低 Top-3/10。结论不是
“DGM 代数失败”，而是**当前 reward + 单终态监督 + action-level 适配组合没有通过综合
门槛**。现有数据上不再启动 10,000-step 训练。

##### 阶段 7 forward-beam 重构奖励（2026-08-08）

为避免 teacher-forced likelihood 对错误候选仍给出偏高分，新增批量 Molecular Transformer
beam generation：把候选反应物正向生成 product beam，真实 product 出现在 beam 中时以倒数
名次 `1/rank` 作为 reward，未出现为 0。该 reward 不读取逆合成 target；validation target 只在
离线诊断中标注“候选是否正确”。20 个未参与实现选择的 validation 反应上，正向模型 beam=5
的 Hit@1/3/5 为 **65%/80%/85%**；扩展到 reaction 400–599 的 200 个已知反应后为
**71%/77%/79%**，MRR **0.7414**，生成 wall **12.0s**、吞吐 **16.65 reactions/s**，说明
checkpoint 方向和生成能力足以用于 reward 试验。

冻结 ordinary Euler（`n_samples=3, n_steps=100, seed=42`）在同一 validation-B
reaction 400–599 上得到 Top-1/3/10 **58.5%/77.5%/88.5%**、Oracle **91.0%**、sampling wall
约 **264s**。这组数据比 reaction 200–399 的 Top-1 **51.0%** 更容易，因此只允许在同一子集内
做配对比较，不能把两个绝对值当成方法提升。

对 validation-B 的 12,000 条已有候选做 forward-beam=5 重排：

| reward 输入 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | correctness AUC | 错误候选重构率 | reward wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 不使用 reward | 58.5% | 77.5% | 85.5% | 88.5% | 91.0% | N/A | N/A | 0s |
| 原始候选 SMILES | 59.5% | 79.0% | 84.0% | 89.5% | 91.0% | 0.6763 | 55.15% | 355.8s |
| canonical 候选 SMILES | 59.0% | 78.5% | 85.0% | 89.5% | 91.0% | **0.6904** | **50.56%** | **117.6s** |

canonical 模式把 8,131 个普通字符串归并为 2,332 个合法结构，reward 计算提速 **3.03 倍**；
正确候选重构率由 82.51% 轻微变为 81.86%，但 AUC 提升且错误候选误命中下降。Top-1/3 比
原始模式少 1 个反应，属于 200 样本上的 0.5 个百分点波动；综合判别质量、表示不变性和速度，
后续多终点数据默认使用 canonical 模式，并保留原始模式作为可复现实验开关。

这一步通过了“独立 reward 比 teacher-forced reward 更有判别力，且候选级受约束使用不降低
Top-1/3/10”的进入门槛，但还没有证明 learned guidance 有效。下一步严格按既定顺序生成每个
product 至少 4 个独立 Euler 终点、每终点 1 个随机中间时间，先审计 product 内 reward range、
高低 reward 与正确性的关系和生成成本；只有数据门槛通过才训练新的 guidance。

##### 多终点 guidance pilot（2026-08-08）

数据脚本保持 ordinary Euler 不变，使用 `n_samples=4, time_samples=1, n_steps=100`。train
取前 1,000 个 original products，validation 取独立 split 的 reaction 0–199；validation
seed=4242，且不与后续 accuracy A/B 的 reaction 200–399、400–599 重叠。RTX 3090 上先比较
product batch 32/64：前者为 **26.90 records/s、0.55/1.26GB allocated/reserved**，后者为
**24.72 records/s、0.86/2.27GB**，因此正式 pilot 选择 32。

| split | products/records | Euler wall | reward wall | 多终点组 | reward 可变组 | 平均 reward range | beam hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 1,000 / 4,000 | 121.7s | 113.7s | 85.2% | 453（45.3%） | 0.4085 | 56.75% |
| validation | 200 / 800 | 25.2s | 23.4s | 85.0% | 88（44.0%） | 0.3925 | 60.75% |

旧数据每个 product 只有一个终点，40,003 个组中 reward 可变组为 0；新数据在两个 split 上
都稳定达到约 44–45%，因此通过“同一条件下存在可学习 reward 对比”的数据门槛。reward 脚本
现会分别记录合法/非法输入、唯一 canonical source、真实生成数和去重复用数，不再把非法输入
误写为 cache hit；同时记录组大小、unique terminal 和 reward range。

随后在 pilot 上训练 500 steps（8 epochs，batch 64），wall **112.3s**、峰值
allocated/reserved **2.02/2.88GB**。完整 validation loss 在 step 100/200/300/400/500 为
**0.9495/0.9368/0.7825/0.7384/0.7728**，所以选择 step 400 的 `guidance_best.pt`。held-out
有效 action 行为 698/800，全局 reward–selected-H Pearson **0.6474**；reward>0/等于0 的
selected-H 均值为 **0.6308/0.2596**。同 product 的 216 个 unequal-reward pair 上，H 排序
正确率 **61.11%**（随机为 50%）。这通过了小规模可学习性门槛，但尚未证明采样 Top-k 提升；
下一步固定 ordinary Euler validation-A，先测保守 `β=0.10`，再决定是否扩大正式数据。

validation-A reaction 200–399 的固定预算采样结果为：Top-1/2/3/5/10
**49.5/67.0/72.5/79.5/84.0%**、Oracle **86.5%**、wall **372.2s**；对应 baseline 为
**51.0/66.5/72.0/77.0/83.5%**、Oracle **86.5%**、wall **253.0s**。因此 pilot 改善
Top-2/3/5/10（其中 Top-5 +2.5 点），但 Top-1 降 1.5 点，未通过默认方法的综合门槛。
它证明 action-level sampler 能利用部分新信号，却不足以支持在 200 个反应上继续扫描 β。

下一步扩大到 train 10,000 products / 40,000 records；这是 batch 64 下训练 5,000 steps 约
8 epochs 的中等规模，而不是直接生成 full-40k products。已有 validation-800 reward 对照中，
beam batch 8/16/32 wall 为 **23.40/19.01/19.36s**，rank/reward 逐项相同，因此后续固定
`batch_size=16`。按 pilot 吞吐估算 10k 的 Euler/reward 分别约 20/16 分钟；只有更大数据的
held-out calibration 与 validation-A/B 通过后，才考虑 full-40k 或 Euler-Beam。

##### 10k 多终点数据与 ordinary-Euler A/B（2026-08-08）

10,000 train products 生成 40,000 records：Euler wall **1,199.2s**、**33.35 records/s**、
峰值 allocated/reserved **0.68/4.60GB**；forward-beam reward wall **892.5s**。35,349 条合法
输入归并为 21,749 个唯一 canonical sources，复用 13,600 次，另有 4,651 条非法输入。
组内 reward 可变的 product 为 **4,647/10,000（46.47%）**，多终点组 **84.9%**，平均
reward range **0.4078**；这些值与 1k pilot 的 45.3%、85.2%、0.4085 一致。

40k records 使用 batch 64 训练 5,000 steps（8 epochs），wall **970.1s**、峰值
allocated/reserved **2.05/3.31GB**。完整 validation loss 在 step
500/1000/1500/2000/2500/3000/3500/4000/4500/5000 为
**0.6674/0.6728/0.7356/0.6854/0.6737/0.7252/0.6840/0.8022/0.8510/0.7469**，因此按预先
声明的最低 loss 选择 step500，而非 final。best 的 held-out Pearson **0.6707**、组内 unequal
reward pair 排序 **60.19%**；final 虽有 Pearson **0.6865**、pair 排序 **62.96%**，但不能在
看过结果后更换主 checkpoint 选择标准，仅保留为“校准 loss 与排序指标不完全一致”的诊断。

固定 `β=0.10`、ordinary Euler、`n_samples=3`、100 steps、seed42 的结果：

| split / 方法 | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Oracle | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation-A baseline（200–399） | 51.0% | 66.5% | 72.0% | 77.0% | 83.5% | 86.5% | 253.0s |
| validation-A 10k guidance | **53.0%** | 66.5% | 71.5% | **78.5%** | 83.5% | 85.5% | 371s |
| validation-B baseline（400–599） | **58.5%** | 73.0% | 77.5% | 85.5% | 88.5% | 91.0% | 264.4s |
| validation-B 10k guidance | 56.5% | **75.0%** | **80.0%** | **86.0%** | 88.5% | **91.5%** | 388s |

A 的 Top-1 +2 但 Oracle -1，B 的 Top-1 -2 但 Top-2/3 与 Oracle 提升；两块均约增加 47%
采样时间。结果证明信号能改变有意义的候选排序，却没有在两个独立 validation block 上稳定
保持 Top-1，因此阶段 7 综合门槛仍未通过：不生成 full-40k、不接 Euler-Beam，也不在 200
反应上继续扫描 β。下一步按既定方法限制，比较当前 per-position rate preservation 与
per-sample total-rate preservation；后者允许 guidance 把编辑强度在不同位置之间重新分配，
但必须先通过 β=0 identity、常数 H identity 和样本总 rate 守恒测试。

##### per-sample total-rate 适配实现（2026-08-08）

新增可选 `guidance_rate_normalization=per_sample`，旧 `per_position` 保持默认。两者都先按
`base action rate × H^β` 加权；旧模式在每个位置单独缩放回原总率，新模式只在每个样本的
所有可编辑位置上使用一个缩放因子，因此保持样本总 edit rate，却允许高 H 位置增加 rate、
低 H 位置降低 rate。BOS/PAD 不参与总率并保持原值。该模式仍是 action-level approximate
guidance，不是 exact Z-space DGM。

纯张量测试覆盖：per-position 回归、β=0 identity、两种模式下常数 H identity、per-sample
editable total-rate 守恒、BOS 非编辑位置不变、高 H 位置 rate 上升、token posterior 归一化和
非法参数拒绝。ordinary Euler sampler 级 β=0 也通过。真实 checkpoint 使用 validation 行
4000–4039、20 steps、3 samples 的 smoke：baseline 与 per-sample β=0 predictions SHA 均为
`a4671889...a6949ca` 且 byte-level `cmp` 通过；β=0.1 改变 **74/120** 行并无数值/CUDA 错误。
下一步在固定 validation-A 上比较 per-position 与 per-sample，其他参数完全不变。

validation-A 的 per-sample β=0.10 结果为 Top-1/2/3/5/10
**53.0/64.0/71.0/77.5/82.0%**、Oracle **84.5%**、sampling wall **约371s**。同 checkpoint、
同 β 的 per-position 为 **53.0/66.5/71.5/78.5/83.5%**、Oracle **85.5%**；baseline 为
**51.0/66.5/72.0/77.0/83.5%**、Oracle **86.5%**。per-sample 除 Top-1 持平外，在
Top-2/3/5/10 和 Oracle 上均被 per-position 支配，且无速度收益，因此没有必要继续运行
validation-B，也不将新模式设为默认。实现保留为隔离研究开关，默认仍是 per-position。

至此，当前 forward-beam reward + action-level adapter 的两种稳定归一化都未通过 A/B 综合
门槛。后续不再无依据扩大 full-40k、扫描 β 或接 Euler-Beam。可行研究路线分为：

1. 把已经在 validation-B 提升 Top-1/3/10 的 terminal forward-beam reranker 作为实用模块，
   研究更低成本的候选级混合/校准；
2. 重定义 guidance 的训练与 checkpoint 选择目标，例如显式 product 内 pairwise ranking，
   但这改变当前 Bregman/action-target 研究假设；
3. 进入严格 Z-space，保留 GAP 身份并追踪变长坐标，解决 89.735% ambiguous insert，属于较大
   sampler/状态表示改造。

三条路线的研究目标、改动范围和计算预算不同；在选定主目标前不应同时继续，以免在同一
validation 子集上反复试错造成选择偏差。

**方法限制。** `apply_action_guidance()` 当前在每个位置保持基础模型的总 edit rate，只在
该位置内部重分配 insert/substitute/delete/token。因此它不能把编辑概率从错误位置移到正确
位置，也不能增强低 rate 位置的纠错或抑制某位置的全部编辑；这保证数值稳定，但不是 exact
DGM 的完整 rate ratio。推理还会在 100 个 Euler step 中重复编码不变的 product；训练则
每个 epoch 都在 CPU 重算 alignment/action mask。这两处分别是推理和训练的明确效率空间。

**下一步固定顺序。** (1) 不再扫描当前数据的 step/β；先用 train 正例 + 多个 Euler 负例
校准 reward，或实现 Molecule Transformer forward beam 重建 product 的 reward，并在未参与
选择的 validation-B 验证判别和 rerank；(2) reward 通过后，以每个 product 至少 4 个独立
终态、每终态 1 个随机时间重新生成 guidance 数据，形成 product 内 reward 对比；(3) 使用
`guidance_best.pt` 做 5k 训练；(4) 对比当前 per-position rate preservation 与“保持每个样本
总 rate、允许位置间重分配”的稳定近似；(5) 只有 ordinary Euler 在 validation-A/B 同时保持
Top-1 且改善 Top-3/10 或 Oracle，才接 Euler-Beam R9K1M2。forward reward 的出处和训练集
重叠在得到可靠 provenance 前也必须作为最终 test 报告的限制项。

如果没有可靠的 forward checkpoint，不能为了“使用 DGM”临时把反向模型自身的
log-prob 当成独立 reward；那只是重复基础 proposal 的信息。

### 阶段 8：再考虑严格 Z-space DGM

#### DG-0 对齐映射审计（2026-08-08）

在不修改任何正式 sampler 的前提下，新增隔离模块
`edit_flows/guidance/zspace.py`。它把一条固定坐标 Z transition 明确分类为
`GAP→token=insert`、`token→GAP=delete`、`token→token=substitute`，并统一处理 BOS/PAD、
X-space 插入锚点和 Edit Flows 的 operation channel（`insert: token`、
`substitute: token+V`、`delete: 2V`）。反向函数返回所有可能的 Z 坐标，而不是假设插入
一定有唯一逆映射；连续 GAP run 会显式标记 `ambiguous=True`，`require_unique=True` 时
拒绝该 transition。

单元测试覆盖单坐标三类动作、BOS/多坐标拒绝、唯一插入和连续 GAP 非双射，共 **33 passed**
（forward/guidance 相关测试集合）。全量 `pytest` 为 **264 passed / 17 failed**；失败全部
来自既有 `tests/sampling/test_beam.py` 与当前 `EditCandidate(log_u)` API 不一致及其受影响
的旧 beam controlled-model 用例，本次隔离 DG-0 没有修改 beam 代码，避免把无关修复混入
DGM 结论。审计脚本为 `scripts/audit_zspace_mapping.py`，本次 JSON 保存在
`/root/autodl-tmp/dgm_guidance_runs/zspace_mapping_val.json`。在 validation 的全部 100,020 条预对齐 augmentation
行上做只读统计：共 **573,927** 个坐标变化，其中 insert **524,838**、substitute
**43,949**、delete **5,140**；insert 中 **470,963（89.735%）** 位于连续 GAP run，
因此静态对齐下没有唯一 X-space 插入逆映射。只有 **7,676/100,020（7.674%）** 的整行
source→target 对只改变一个坐标。这个结果不是 sampler 指标，而是形式化边界证据：当前
变长 Edit Flow 不能直接宣称论文的 exact posterior；必须先在固定 Z 坐标中保留 GAP 身份，
或把方法明确命名为 action-level approximate guidance。

随后完成 DG-1 的低风险接口：`compose_edit_action_log_weights()` 将基础模型的三类
rate/token 输出组合为统一的 `insert[V] + substitute[V] + delete` 通道，并用
`guided_log_probs()` 验证全 1 guidance 的逐元素 identity；它只做组合，不改变 rate
parameterization、归一化或现有 sampler。该接口与映射测试均通过，正式 guided sampler
仍未接入。一个固定坐标 toy 通过 `q(z)∝p(z)R(z)` 经验分布测试，说明 DGM 的密度比
代数在固定 Z-space 上是正确的；当前阻塞来自变长 X-space 的坐标映射，而不是该代数。

如果 action-level 近似显示收益且没有明显的概率不一致，再处理：

- aligned Z-space 的固定坐标；
- GAP 的 guidance；
- Z-space 到 insert/substitute/delete 的唯一映射；
- 与 Edit Flows conditional rate 的一致性证明和测试。

这一阶段才可以讨论是否使用“exact DGM”这个表述。

### 阶段通过门槛总表

| 阶段 | 任务 | 达到什么效果才算通过 | 当前状态 |
|---|---|---|---|
| 0 | 冻结 baseline | 同 seed predictions SHA 可复现；guidance off 与当前 Euler 完全一致；指标和 metadata 齐全 | 部分已有历史基线，DGM 专用快照待做 |
| 1 | synthetic DGM | 已知 `q ∝ pR` 的采样频率、后验重加权、ESS/evidence 与理论一致；常数 guidance 不改变 `p` | 代数工具和 19 个测试通过，多步 rollout 待做 |
| 2 | RDKit validity reward | reward 有限、非负、批量结果可复现且不读取 test target；invalid/ESS 变化可解释 | `[x]` 接口/格式转换/缓存/terminal smoke 通过；validity 仅保留为诊断 reward，未证明准确率提升 |
| 3 | guidance 数据生成 | 按 product 隔离 train/validation；样本近似基础 `pθ`；保存 product、终点、reward、`t`、alignment、seed | 正式 train/validation 生成、审计和 BOS 修复完成 |
| 4 | 训练 guidance model | loss 有效下降；`H>0`；held-out reward 与 `H` 有稳定相关/校准；训练和推理成本可接受 | balanced action-level 训练、held-out 校准和成本测量完成 |
| 5 | 普通 Euler 接入 | guidance off/constant 严格回归 baseline；guided log-prob 与采样分布一致；无非法概率 | 机制通过；validation-200 validity reward 未提升 Top-k，默认关闭 |
| 6 | Euler-Beam/SMC 接入 | 固定总预算下 Top-1 不明显下降，Top-3/10 或 Oracle 在不重叠 validation 稳定改善；ESS 不系统坍缩 | 暂缓，等待更有信息量的 forward reward |
| 7 | forward reward | Molecular Transformer 方向/tokenization/权重加载通过已知反应 smoke；validation forward 指标可接受；reward 可批量评分；guided Top-k 门槛通过 | 10k 多终点与两种 rate normalization 均完成；learned guidance 未通过 A/B 综合门槛，默认关闭；terminal reranker 单独通过候选级门槛 |
| 8 | 严格 Z-space DGM | GAP/变长动作映射明确；synthetic 和 identity-limit 测试通过；才可使用 exact DGM 表述 | DG-0 映射审计、DG-1 action-weight identity、固定坐标 toy 已通过；高比例非双射插入使完整 exact sampler 暂未开始 |

任何阶段只达到“代码能运行”而没有达到对应栏的正确性和对照门槛，都不记为通过。

### 各阶段的 Top-k 与效率指标

| 阶段 | 是否计算 Top-1～10 | 主要正确性指标 | 必须记录的效率指标 |
|---|---|---|---|
| 0 baseline | 是，作为统一参照 | Top-1～10、Oracle、invalid、unique、target rank | sampling wall、score wall、peak memory、GPU 设置 |
| 1 synthetic | 否，没有 SMILES target | 理论分布 vs empirical 分布、KL/TV、ESS、evidence | rollout wall、samples/s、内存 |
| 2 validity reward | 是，在 validation 上 | Top-1～10、Oracle、invalid、reward、ESS | reward wall、调用数、缓存命中率、总 wall |
| 3 guidance 数据 | 不以 Top-k 为主 | product 隔离、proposal 分布、reward 分布、数据完整性 | 生成 wall、吞吐、磁盘占用、显存 |
| 4 guidance 训练 | 训练阶段不直接算 Top-k | guidance loss、`H>0`、held-out reward/H 相关性和校准 | train wall、steps/s、samples/s、peak memory |
| 5 普通 Euler | 是 | guidance off/constant 回归 baseline；guided 分布和 log-prob 一致 | 基础/guidance forward 次数、wall、显存 |
| 6 Euler-Beam/SMC | 是 | Top-1～10、Oracle、invalid、unique、ESS、resampling、祖先多样性 | reward/guidance 调用数、wall、峰值显存 |
| 7 forward reward | 是；另测 forward 模型自身 validation 指标 | forward 方向/tokenization 正确；DGM 指标在不重叠 validation 改善 | forward batch wall、调用数、缓存、总 wall |
| 8 Z-space DGM | 是 | 变长映射、identity-limit、synthetic 和真实 Top-k 正确性 | 每步 forward 数、wall、显存、映射额外开销 |

因此，Top-1～10 不是只有最终 test 才计算：阶段 0、2、5、6、7、8 都会在 validation
上计算；阶段 1、3、4 的重点是 mechanics、数据和模型训练诊断，不用没有意义的 Top-k
数字替代它们的专门指标。所有真实 SMILES 阶段都同时记录 wall time、显存和模型调用数。

### `n_steps` 消融

第一版所有 guidance 正确性实验固定 `n_steps=100`。阶段 5 通过后，在相同 checkpoint、
seed、总候选预算、precision 和 batch 下运行：

```text
n_steps=50 / 100 / 200
```

分别比较 Top-1～10、Oracle、invalid、ESS、wall 和 peak memory。只有 50 步在不明显损害
准确率、覆盖和数值稳定性的情况下，才考虑把默认推理步数从 100 降低。

---

## 5. 需要提前准备的东西

### 5.1 已经具备

- 训练好的 Edit Flows checkpoint：`new_checkpoints/checkpoint_step600000.pt`；
- product/reactant 数据集及 train/validation/test 划分；
- vocabulary 和 tokenization；
- 当前 Euler、Euler-Beam、Euler-SMC 采样代码；
- `sample_retro.py`、`eval.py` 和 Top-1～10/Oracle/invalid 评分流程；
- 可复现 seed 和 metadata 记录方式。

### 5.2 必须先整理

1. **Guidance 数据划分**：按 product 划分 train-guidance、validation-guidance，不能把
   同一反应的 augmentation 随机拆到两侧。
2. **Reward 规范**：明确 reward 数值范围、非法输入处理、是否使用 `exp(βS)`、版本号。
3. **输出格式**：保存每个候选的 base log-prob、guided log-prob、reward、guidance 权重、
   ancestor/branch 信息。
4. **固定实验预算**：固定 `n_steps`、总候选数、batch size、seed 和 GPU 精度。
5. **合成测试**：在真实 SMILES 前完成已知分布的 DGM smoke test。

### 5.3 forward reward 需要的材料

如果选择 forward reaction model，需要准备：

- forward model 的权重文件；
- 模型架构和配置；
- tokenizer/vocabulary；
- 输入格式（reactants → product，是否需要 atom mapping）；
- 训练/validation 数据边界；
- 推理脚本和 batch 接口；
- forward model 本身的 validation 指标；
- 与当前数据集的 SMILES 预处理兼容性。

如果这些材料不完整，先不要把 forward reward 放进主实验。可以先用 RDKit reward 验证
接口，也可以先训练一个独立、可复现的 forward model，但这应作为单独的模型任务记录。

### 5.4 已上传的 Molecular Transformer checkpoint

你上传的文件是：

```text
new_checkpoints/MIT_mixed_augm_model_average_20.pt
大小约 150 MB
```

它对应官方
[`pschwllr/MolecularTransformer`](https://github.com/pschwllr/MolecularTransformer)
项目的 `MIT_mixed_augm` 模型。官方 README 说明该项目使用旧版 OpenNMT-py，典型模型是
reactants/reagents → product 的 Transformer；官方推理入口是 `translate.py`，预处理使用
SMILES 正则分词、RDKit canonicalization，并通过 `batch_size`、`max_length` 和 beam 生成
预测。具体配置和 tokenization 以官方
[`README`](https://github.com/pschwllr/MolecularTransformer#pre-processing)、
[`preprocess.py`](https://github.com/pschwllr/MolecularTransformer/blob/master/preprocess.py)
和 [`translate.py`](https://github.com/pschwllr/MolecularTransformer/blob/master/translate.py)
为准。

对本地 checkpoint 做了只读检查，发现它包含：

```text
vocab / optim / model / opt / generator
```

嵌入配置为：

```text
encoder/decoder：Transformer，各 4 层
hidden / word vector：256
heads：8
FFN：2048
dropout：0.1
shared source/target embeddings：是
词表输出维度：297
Noam warmup：8000
```

checkpoint 本身保存的是旧版 `torchtext.vocab.Vocab` 对象。当前 `/root/autodl-tmp/ef`
环境没有旧版 `torchtext`，因此直接 `torch.load` 会失败；官方仓库还要求非常老的
PyTorch 0.4.1 和 `torchtext==0.3.1`。这意味着它**不能直接当作当前项目的 PyTorch 模型
导入**，后续需要二选一：

1. 建立隔离的旧版 OpenNMT/torchtext 环境，调用官方 `translate.py`；或
2. 编写兼容加载器，将 checkpoint 的 state dict 和 vocabulary 移植到当前环境。

已完成现代 PyTorch 兼容加载和最小 forward smoke，当前仍不把未校准的 raw score 直接接入
DGM 主采样；需要先完成 validation reward 校准和 terminal guidance 对照。

还需要特别注意：

- Molecular Transformer 词表（约 297 个 token）与当前 Edit Flows 词表不同，不能混用；
- 当前任务是 product → reactants，而该 checkpoint 的常规方向是 reactants/reagents →
  product，理论上适合做 forward consistency，但必须用一条已知反应先确认输入方向和
  输出方向；
- 如果用 forward likelihood 作为 reward，官方 `translate.py` 主要负责生成，而不是
  直接返回给定 product 的 teacher-forced log-likelihood，需要额外实现评分接口；
- 如果使用 beam 生成后只判断是否重建 product，也必须记录 beam size、canonicalization、
  tokenization 和 reward 版本；
- Molecular Transformer 的 source tokenization 必须使用它自己的正则和 vocabulary，不能
  直接套当前 `example.vocab.src`。

因此，这个 checkpoint 很有价值，但它属于阶段 7 的 forward reward 资产，不是阶段 0～6
的阻塞条件。第一版仍然先用 RDKit reward 和 synthetic target 验证 DGM mechanics。

### 5.5 依赖安装策略

当前 `/root/autodl-tmp/ef` 环境已经具备第一阶段所需依赖：

```text
torch 2.13.0+cu130
rdkit 2026.03.5
numpy 1.26.4
einops 0.8.2
tensorboard 2.21.0
```

当前缺少 `torchtext` 和 `onmt`，但这不是 Edit Flows/DGM 第一阶段的缺包。由于官方
Molecular Transformer 要求旧版 PyTorch 0.4.1、`torchtext==0.3.1` 和旧 OpenNMT，禁止
直接在 `/root/autodl-tmp/ef` 中执行无版本约束的 `pip install torchtext` 或安装旧版
OpenNMT；这可能破坏当前 PyTorch/CUDA 环境。

阶段 7 当前已经采用第一优先级：只读 state-dict/vocabulary 移植到现代 PyTorch，主环境
没有安装旧依赖。如果未来需要复核官方 `translate.py` 的 beam 生成，再创建隔离
`/root/autodl-tmp/mt_legacy` 环境；这不是当前 teacher-forced reward 的前置条件。

安装任何包前都要记录环境、版本、安装命令和 `pip check`/import 结果；当前主环境不因
forward reward 的实验而改变。

---

## 6. reward 是先实现，还是 baseline 之后再加？

正确顺序是：

```text
先建立冻结 baseline
→ 同时可以先写 reward 接口和 synthetic reward
→ 再接真实 reward
```

具体来说：

- **baseline 必须先有**，否则无法知道 guidance 是否真正带来收益；
- **reward 接口可以提前写**，但先用 synthetic/RDKit 规则，不必一开始准备 forward model；
- **forward reward 应在 baseline 和 guidance mechanics 验证后再加入**；
- **不要先训练 forward model 和 guidance model，再回头寻找 baseline**。

原因是 forward model 本身可能带来新的误差、偏置和运行时间。如果一开始就加入它，最终
无法区分提升来自 DGM、forward model 质量，还是普通 reranking。

### 推荐的第一版 reward

第一版建议：

```text
R_validity(c,y) = 0          非法 SMILES/价态不通过
                   1          基本合法
```

它不一定会提升 Top-1，但适合验证：

- reward 接口；
- terminal twisting；
- DGM 权重和 ESS；
- invalid/coverage 的 trade-off。

第二版再加入 forward consistency，作为真正面向逆合成质量的主 reward。

---

## 7. 每一阶段的评估指标和停止条件

每次实验至少报告：

```text
Top-1 ... Top-10
Oracle-any
rank-1 invalid
valid unique / true unique
mean final target rank
mean / p10 ESS
resampling 次数
reward 分布
guidance forward 次数
wall time
peak memory
```

成功不应只定义为 Top-1 上升。一个可接受的 guidance 结果应该同时满足：

1. Top-1 不明显下降；
2. Top-3/Top-10 或 Oracle 在不重叠 validation 上有稳定改善；
3. invalid 下降不是以严重牺牲覆盖为代价；
4. ESS 没有系统性坍缩；
5. 运行时间和额外模型调用可接受；
6. guidance 关闭时完全回归 baseline。

如果 reward 只让 invalid 下降、但 Top-k 和 Oracle 都下降，应记录为 reward 偏置，而不是
称为方法成功。

---

## 8. 当前项目中的边界和非目标

- 不修改已有训练 checkpoint 和训练数据。
- 不把当前 `changed_state_bonus` 当成 DGM reward；它是 Euler-Beam 的启发式排序项。
- 不把基础模型自身 log-prob 当成独立 reward。
- 不在 test 上训练 guidance、选择 reward 或调 `β`。
- 不在第一版同时改变 `R/K/M`、temperature、bonus 和 reward。
- 不在 action-level 近似尚未验证时宣称已经实现论文中的 exact guidance。

当前 `edit_flows/sampling/euler_smc.py` 已有粒子权重、ESS 和 resampling mechanics，但
尚未接入独立 reward 或 learned guidance。它可以作为后续承载层，不能被当作已经完成的
DGM 实现。

---

## 9. 实施前的最终 checklist

- [ ] 固定 baseline checkpoint、seed、R9K1M2、`n_steps=100`。
- [ ] 在 validation-A 保存 baseline predictions 和完整指标。
- [ ] 完成 synthetic DGM 分布恢复测试。
- [x] 完成 teacher-forced 与 forward-beam reward 接口、缓存和 metadata；beam reward 默认 canonical 输入。
- [ ] 确认 guidance train/validation 按 product 隔离。
- [x] RDKit validity 仅保留诊断；主候选为 forward-beam=5 reciprocal-rank reward。
- [x] 确认 forward model 权重、官方 tokenizer、方向和现代兼容推理接口。
- [x] 完成 1k、2.5k 和 5k guidance adapter 对照，不修改基础 checkpoint；准确率门槛未通过。
- [x] 先在普通 Euler 上完成 guidance off/on A/B，并完成 validation-200 对照。
- [ ] 只有独立 reward 在普通 Euler 上通过准确率门槛后，再接 Euler-Beam/Euler-SMC。
- [ ] 最后才在 validation 上选择 forward reward 和 `β`，再运行 test。

截至 2026-08-08，阶段 7 已完成 pilot、2,500-step、5,000-step 和 β=0.10/0.25 复核；这些实验
都没有在固定预算下同时改善 Top-1 与 Top-3/10/Oracle。因此 forward reward 现阶段冻结为
可复现的诊断资产，不再继续无依据地扫描 β，也不接入默认 Euler/Euler-Beam。下一条安全的
研究路线是： (a) 另定义并独立验证一个 reward，再做 validation-A/B；或 (b) 采用明确标注
为 approximate 的 canonical-Z/action 方法。DG-0 已证明 exact Z-space 映射存在高比例非
双射插入，不能在没有新的坐标设计前宣称 exact DGM。在准确率目标和覆盖目标没有新的预先
取舍前，不再启动更长的 test/full-src-test GPU 实验。

---

## 10. 可自主推进的范围和必须请你决策的情况

在你休息期间，可以安全自主推进以下工作：

- 检查 Molecular Transformer checkpoint 和官方 tokenization；
- 编写只读 checkpoint 检查、reward 接口和 synthetic DGM 测试；
- 使用 validation 子集建立 baseline，不读取 test target；
- 实现 RDKit validity reward 和缓存；
- 训练小规模 guidance adapter，记录 loss、reward/H 相关性、ESS 和运行时间；
- 运行单元测试、smoke test、guidance off/on 对照；
- 更新本文档和任务规划，并为每个阶段创建范围明确的 Git commit。

以下情况必须暂停并请你决定，不会擅自选择研究目标：

1. validity reward 与 forward-consistency reward 得出相互冲突的主结论，需要决定论文主线；
2. 需要选择“提高 Top-1”还是“提高 Oracle/Top-10/validity”作为主要优化目标；
3. Molecular Transformer 旧环境移植失败，必须决定隔离旧环境、移植模型，还是暂时放弃
   forward reward；
4. 需要重新训练基础 Edit Flows、训练新的 forward model，或修改已有 checkpoint/数据集；
5. 某个 reward 造成明显的粒子坍缩，而不同的稳定化方案代表不同研究假设。

除此之外，我会继续按本文档推进，不等待逐步确认；实验结束后如果没有新的 GPU 任务，
会启动 `alive.py` 保持机器在线，并保留日志、metadata 和 Git 记录。

## 11. Product-internal pairwise guidance extension（2026-08-08）

在阶段 7 learned guidance 未通过 validation-A/B 综合准确率门槛后，按
`new_docs/dgm_pairwise_guidance_implementation_plan.md` 开始低风险的训练目标改造。当前
P0–P5 已完成，P5 的 pilot 正确运行但未通过收益门槛：

- P0：确认 1k/10k/validation guidance 数据每个 `source_index` 严格包含 4 条记录，现有
  guidance/forward 定向回归 **42 passed**；
- P1：新增 `ProductGroupBatchSampler`，只在显式 grouped 模式启用，保证 product group 不跨 batch；
- P2：新增 shared-anchor `mean(log H)` terminal score 和 pairwise softplus loss；同一 anchor
  的状态/时间下比较多个 terminal，equal reward 和 no-action candidate 均有明确处理；
- P3：`train_guidance.py` 已接入 grouped/pairwise 参数、全 anchor validation、1.15× Bregman
  guardrail、TensorBoard 指标和 `summary.json`；新增只读 `scripts/evaluate_guidance.py`；
- P3 定向回归为 **53 passed**。缩小模型的 1-step CPU pairwise、guarded checkpoint、旧默认
  路径和 checkpoint loader smoke 均通过。

P4 CUDA smoke 已完成：RTX 3090 上缩小模型 1-step 训练/validation、CUDA evaluator 和真实
`sample_retro.py` 调用均通过；同 seed、`n_steps=2`、20 条 augmentation 的 baseline 与
`guidance_beta=0` predictions SHA256 完全一致（`32c12c…f8b5d45`）。定向回归为 **53 passed**；
全量测试为 **285 passed, 17 failed**，17 个 failure 均为既有 beam 测试 API/controlled-model
问题。

P5 使用 seed42 在 1k train/200 validation 上完成了预注册的三组训练：grouped control 的
shared-anchor pair accuracy 为 **59.73%**，lambda=0.25 为 **58.63%**，lambda=1.0 为
**57.93%**；对应 wall 为 136.8/169.3/177.1s，peak allocated/reserved 约为
2.03/3.50、2.04/3.50、2.04/3.50GB。两种 pairwise 权重均低于 control，未达到 +3pp 门槛，
因此没有运行 seed43、10k 或 Top-k。P5 记为“实现正确、实验未通过”；随后完成 P5b 数据定义
审计。现有 1k/10k/validation guidance 记录虽然每个 product 有 4 条记录，但四条记录的时间
从未全部相同，且 validation-200 只有 23/200 组的 state 恰好全部相同（平均每组 2.655 个不同
state）。这是因为旧生成路径对每个独立 terminal 分别调用 `sample_intermediate_states(product,
terminal, t)`，`source_index` 只共享 product，不共享 `(x_t,t)`。因此当前 shared-anchor pairwise
指标是条件状态错配下的反事实诊断，不能作为可靠的同一 anchor 排序信号。

P5b 新增 `scripts/audit_guidance_anchors.py` 及单元测试，审计结果写入
`/root/autodl-tmp/dgm_guidance_runs/anchor_audit_val200.json`。下一步先隔离实现真正的
shared-anchor 数据生成（公共 `x_t` 后独立 continuation），并验证每组 state/time 完全一致；在
此之前不扩大到 10k、不做 Top-k 或 Euler 采样 A/B。

P5c 已完成最小实现：`sample_euler` 支持从中间 state/time continuation，新增
`scripts/generate_shared_anchor_guidance.py` 以 prefix 一次、批量 continuation 多个 terminal。
真实 checkpoint 的 CUDA smoke（4 products、2 children、4 steps）耗时 **0.799s**，峰值
allocated/reserved **234/247MB**；4/4 组 state/time 完全相同且 terminal 均有两个不同结果。
这只是数据结构和效率 smoke，下一步才在隔离的 1k shared-anchor 数据上重新比较 pairwise。另对
adaptive endpoint 做了 correctness 修正：第 50 个 state 的真实时间由 `get_euler_step_times()`
计算（full-step smoke 为 0.49999979），不再用轨迹长度近似；修正前生成的 1k 文件作废。

修正后完成新的 1k shared-anchor pilot（train 1,000/4,000，validation 200/800，所有组 state/time
完全共享）。control 的 pair accuracy 为 **55.15%**；lambda=.25 为 **59.66%**（+4.51pp），但
Pearson 从 0.1135 降到 -0.0118；lambda=1.0 为 **53.65%**。Bregman 分别为 0.69165、0.76427、
0.77967，pairwise 两组均在 1.15× guard 内，但只有 lambda=.25 的 rank 提升，且 Pearson gate
失败。因此这是“rank 改善、连续校准恶化”的部分结果，不启动 seed43、10k 或 sampling A/B；不能
宣称 end-to-end Top-k 提升。

随后加入只读的 `reward_score_pearson_within_group` 诊断，排除跨 product score offset：control
为 0.1490，lambda=.25 为 0.0667，lambda=1 为 0.0733。组内 Pearson 也下降，确认 lambda=.25
的连续校准损失是真现象；不修改旧 gate，不继续扫描 lambda。

下一步的唯一候选是一个默认关闭的 score-aware calibration loss：同一 anchor 内将 candidate
`mean(log H)` 与 reward 分别去中心化、归一化后做 MSE。预注册 `pairwise λ=0.25`、
`score_calibration_weight=0.10`，依据是现有 raw calibration loss≈1.64，使其训练贡献与 pairwise
项同量级；先 5-step smoke，再 500-step pilot，不扫描其他权重。

P5f pilot 已完成：best step 400，Bregman 0.78580，pair acc 56.65%，global Pearson 0.0966，
within-group Pearson 0.1165，wall 228.6s，peak allocated/reserved 2.06/3.44GB。它比未加校准的
lambda=.25（within Pearson 0.0667）有所修复，但仍低于 control 0.1490，联合门槛失败；因此不
进入 seed43、10k 或 sampling A/B。

随后按用户请求，对 P5f 的 `guidance_best.pt`（step 400）和 `guidance_final.pt`（step 500）做了
一次严格配对的 ordinary-Euler Top-k 诊断。数据为未参与 guidance 训练的 validation-A，原始
reaction 200--399（200 个完整反应）；两次均固定基础 checkpoint、seed=42、`n_steps=100`、
`n_samples=3`、batch=64、`guidance_beta=0.10`、`per_position`、augmentation=20、
`legacy_best_rank`、`beam_size=3` 和 `n_best=10`。

| checkpoint | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Oracle | invalid@1/2/3 | true unique | sampling wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `guidance_best.pt`（step 400） | 51.0% | 64.5% | 69.0% | 75.0% | 79.0% | **86.5%** | 12.200/11.775/11.825% | **12.070** | 369.4s |
| `guidance_final.pt`（step 500） | **52.0%** | **67.0%** | **71.0%** | **77.5%** | **80.5%** | 83.5% | **11.600/11.600/13.400%** | 11.935 | 377.8s |

final 在该 split 上的 Top-k 排序优于 best，但 Oracle 低 3 个百分点、真实候选覆盖略低，且采样
耗时高约 2.3%。因此不能简单地说 final 在所有方面更好：best 更偏向覆盖，final 更偏向已生成候选
的排序。由于这只是用户要求的 checkpoint 诊断，而非预注册的主模型选择实验，仍保留
`guidance_best.pt` 作为按验证规则选择的正式 checkpoint；未据此启动 test、10k 或 Euler-Beam。

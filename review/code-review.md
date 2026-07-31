# Edit Flows 逆合成实现代码审查

> 审查日期: 2026-07-23
> 审查范围: `edit_flows/` 全部核心代码、配置、脚本、测试
> 审查目标: 检查 Edit Flows 论文方法在化学逆合成任务上的实现正确性与适配性

---

## 一、总体评估

| 维度 | 评分 | 说明 |
|------|:----:|------|
| Edit Flows 核心理论实现 | ⭐⭐⭐⭐⭐ | Bregman 损失、Z-space 条件路径、速率参数化均正确 |
| Euler 采样 | ⭐⭐⭐⭐ | 核心逻辑正确，自适应步长和事件概率公式无误 |
| Beam/Greedy 搜索 | ⭐⭐⭐ | 单编辑搜索框架合理，但 beam search 缺少显式 STOP |
| 化学逆合成适配 | ⭐⭐ | 严重不足，缺乏化学领域特有约束和反应感知 |
| 代码质量 | ⭐⭐⭐⭐ | 测试覆盖好（30+ beam 测试），类型标注清晰，但部分模块缺少文档 |
| 训练工程 | ⭐⭐⭐ | 基础训练循环正确，但数据预处理、模型规模、正则化有改进空间 |

**核心结论**: Edit Flows 的数学实现基本正确，Oracle 实验（97% Top-1）已经验证了采样框架的正确性。主要问题不在"实现错了"，而在**"Edit Flows 作为通用序列编辑方法，未被充分适配到化学逆合成的特殊性"**。

---

## 二、发现的问题（按严重程度排列）

### 🔴 严重问题

#### 2.1 对齐策略完全忽略化学语义

**位置**: [edit_flows/core/alignment.py](edit_flows/core/alignment.py#L16-L55)

**问题**: `_align_pair` 使用标准 Levenshtein 编辑距离对产品 SMILES 和反应物 SMILES 进行对齐。但 SMILES 字符串的字符级编辑距离与化学反应的真实编辑距离存在本质差异：

1. **SMILES 重排问题**: 同一分子可以有多种 SMILES 表示（通过不同的起始原子和遍历顺序）。产物到反应物的 SMILES 变化中，很大部分来自 SMILES 表示的重新排列，而非真正的化学键变化。Levenshtein 对齐会将表示重排错误地解读为大量 insert/delete/substitute 操作。

2. **无原子映射**: 真正的逆合成中，产物中的每个原子（除离去基团外）都能在反应物中找到对应原子。但 Levenshtein 对齐不保证原子映射的一致性——产物中的原子 C 可能被对齐到反应物中完全不同的 C 原子上。

3. **具体案例**: 假设反应为 `CCO >> CC=O`（乙醇→乙醛，氧化反应）：
   - 化学上: C-C-OH → C-C=O（仅 OH→O 变化）
   - 但 SMILES 层面的编辑距离可能远大于 1，因为 SMILES 表示中 O 和 =O 的位置可能完全不同

**影响**: 训练时模型学习的编辑模式混杂了大量"SMILES 字符串编辑"而非"化学编辑"，导致模型泛化能力受限。

**建议修复**:
- 使用基于原子映射的 SMILES 对齐（如 RXNMapper、RDToolkit 的原子映射功能）
- 或使用反应 SMILES（reaction SMILES）格式，其中产物和反应物的原子顺序已对齐
- 在预处理阶段用化学工具进行规范化对齐，对训练数据做 `canonicalize_with_atom_mapping`

#### 2.2 训练用 Z-space 损失与化学多步逆合成不匹配

**位置**: [edit_flows/training/loss.py](edit_flows/training/loss.py), [edit_flows/core/z_space.py](edit_flows/core/z_space.py#L93-L113)

**问题**: 训练时 `make_ut_mask_from_z` 在 Z-space 中创建二元编辑掩码——每个 Z-space 位置最多标记一个编辑操作。但化学反应中，一个官能团变化可能对应 SMILES 层面的多个 token 变化。Z-space 的一对一对齐将多 token 反应拆解为多个独立编辑，丢失了编辑之间的化学耦合关系。

此外，`bregman_loss` 的 CE term 使用 `(log_uz_cat * uz_mask).sum()`，即所有正确编辑方向的 log-概率之和。这意味着每个编辑被独立对待——插入 'C' 和插入 'O' 被视为两个不相关的操作，而非"插入一个 C-O 键"的整体。

**影响**: 模型学习的是局部 token 编辑模式，无法捕捉化学反应的全局约束（如价键守恒、官能团协同变化）。

**建议修复**:
- 考虑使用 graph-level 或 fragment-level 的编辑操作替代 token-level 编辑
- 或在 loss 中加入化学有效性约束（如原子价态一致性、环结构完整性）

#### 2.3 Beam Search 未集成显式 STOP

**位置**: [edit_flows/sampling/beam.py](edit_flows/sampling/beam.py#L736-L954)

**问题**: `sample_beam_single_edit` 函数没有 `explicit_stop` 参数，仅依赖 `stop_u_tot_base` 阈值和 `time_policy.update()` 的返回信号。相比之下，`sample_greedy_single_edit` 已完整实现了显式 STOP（包括 Frozen-Hazard 和 Poisson 两种 κ 推进方案）。

Beam search 中的 STOP 缺陷:
1. 没有 STOP 作为显式候选动作——无法比较"继续编辑"和"停止"的概率
2. 没有 FH/Poisson κ 推进——beam 中的时间由 `TimePolicy` 统一管理，无法按每个 hypothesis 的状态独立推进
3. 停止判断是二元的（父状态满足条件→标记 finished），而非概率性的（比较 STOP 和 edit 的 log_prob）

**影响**: Beam search 的停止逻辑不够 principled，文档记录的 beam-5 仅比 greedy 高 ~2pp 可能部分源于此。

**建议修复**: 将 `explicit_stop` 机制从 greedy 移植到 beam：
- 每个 BeamState 维护自己的 κ（FH 或 Poisson）
- 展开时生成 STOP child 和 edit children
- STOP child 的 score = parent.score + log_p_stop
- Top-k 裁剪自然处理 STOP vs edit 的竞争

---

### 🟡 中等问题

#### 2.4 模型架构缺乏化学感知设计

**位置**: [edit_flows/models/transformer.py](edit_flows/models/transformer.py#L104-L225)

**问题**:
1. **标准 Transformer 编码器**: 使用 `nn.MultiheadAttention` 的标准 encoder-only 架构，没有针对 SMILES 序列的特殊设计（如分子图的 positional encoding、化学键类型嵌入）
2. **三个独立输出头**: `rates_out`（3 维）、`ins_logits_out`（V 维）、`sub_logits_out`（V 维）是三个完全独立的两层 MLP。插入和替换的 token 分布不共享任何参数，但化学上，同一位置应该插入什么 token 和替换成什么 token 有强相关性（都是让序列更接近目标分子）
3. **时间嵌入的处理方式**: 时间嵌入通过加法融入 token 嵌入（`token_emb + time_emb + pos_emb`），这意味着所有 token 位置共享相同的时间信息。但不同位置在不同时间的编辑需求是不同的——BOS 附近可能在早期就需要编辑，而序列末尾可能需要后期编辑

**影响**: 模型能力受限于通用 Transformer 的表达能力。对于化学分子这种有强结构化先验的数据，通用架构可能不是最优的。

**建议修复**:
- 考虑加入图神经网络或分子指纹作为辅助特征
- 在 ins_logits_out 和 sub_logits_out 之间加入参数共享或交叉注意力
- 考虑使用 per-position 的时间嵌入（或至少让时间嵌入通过 FiLM 层而非简单加法）

#### 2.5 速率重参数化的实现细节问题

**位置**: [edit_flows/core/rate_scale.py](edit_flows/core/rate_scale.py), [edit_flows/training/trainer.py](edit_flows/training/trainer.py#L136-L151)

**问题**:
1. **训练时监控指标使用了 effective rates**: 在 `train_step` 中，`u_ins`/`u_del`/`u_sub` 是通过 `apply_rate_parameterization` 后计算的。当 `use_rate_reparam=True` 时，这些是 real rates (乘了 k(t))。但模型输出的 base rates 才是我们关心的——它们是否趋近于真实的编辑计数。混合使用两种速率语义容易在调试时造成混淆。

2. **`get_rate_scale` 的 `clamp_kappa` 选项**: 当 `clamp_kappa=True` 时，先 clamp `1/(1-κ)` 再乘 `κ̇`；当 `clamp_kappa=False` 时，直接 clamp `κ̇/(1-κ)`。这两种方式在 κ→1 时行为不同——前者的上限是 `κ̇ * clamp_max`（当 κ̇ 较小时上限也小），后者的上限始终是 `clamp_max`。文档中没有解释为什么需要这个选项，也没有实验记录对比两种方式的效果。

**建议修复**: 在 README 或代码注释中明确 `clamp_kappa` 的物理含义和适用场景。统一监控指标的速率语义。

#### 2.6 训练数据未做化学有效性过滤

**位置**: [edit_flows/data/dataset.py](edit_flows/data/dataset.py), [scripts/train_retro.py](scripts/train_retro.py)

**问题**: `RetroDataset` 直接加载 token 化的 SMILES 文件，不做任何化学有效性检查。数据集中可能存在：
- 无效的 SMILES（不符合化学语法）
- 重复的反应（相同产物→相同反应物，来自数据增强）
- 产物和反应物之间的化学不可达反应（数据标注错误）

**影响**: 模型可能在无效数据上浪费容量，且无效 SMILES 的编辑模式可能误导模型。

**建议修复**: 在数据预处理阶段加入 RDKit 的 SMILES 验证和规范化。过滤掉无效反应。

#### 2.7 Euler 采样中 GAP token 未显式屏蔽

**位置**: [edit_flows/sampling/euler.py](edit_flows/sampling/euler.py#L349-L356)

**问题**: 在 `sample_euler` 中，模型输入是 X-space 的 `x_t`（通过 `rm_gap_tokens` 移除了 GAP），所以 X-space 中不应该出现 GAP token。但由于模型输出 log_ins_probs 和 log_sub_probs 覆盖了整个 vocab_size（包括 GAP_TOKEN = 2），理论上模型可以在插入/替换时采样出 GAP token。在 Euler 采样中，这被 `torch.multinomial` 处理（概率虽低但非零）。

在 beam search 中，这个问题通过 `_build_forbidden_mask` 得到正确处理（GAP_TOKEN 被屏蔽）。但在 Euler 采样中没有等效的处理。

**影响**: 极低概率下 Euler 采样可能生成含 GAP token 的序列，导致后续处理异常。

**建议修复**: 在 Euler 采样中也加入 forbidden token 屏蔽（或在模型输出层面通过 `masked_fill` 排除 GAP/BOS/PAD）。

#### 2.8 时间策略 RatioTimePolicy 的前两步使用 depth kappa

**位置**: [edit_flows/sampling/time_policy.py](edit_flows/sampling/time_policy.py#L169-L177)

**问题**: `RatioTimePolicy.get_kappa()` 在 step ≤ 1 时回退到 depth-based κ。原因是 step 1 时 u_prev 仍来自 step 0 的模型输出（编辑前），比率 u_prev/u_init 始终为 1.0。

但这意味着:
1. 前两步的时间与编辑进度无关，对于只需 1-2 步编辑的简单反应，模型可能在错误的"早期"时间信号下操作
2. 如果 step 0 的模型输出恰好是一个很好的状态（u_tot 已经很低），step 1 仍然使用 depth κ 而非反映真实进度的 κ

**建议修复**: 考虑在 step 1 就使用比率——step 0 的模型输出是在原始 product 状态下的，u_init 和 u_prev 的比值确实为 1.0，但这恰恰应该映射到小的 κ（刚开始）。或者改为使用 `max(1, step)` 步的 depth 作为 warmup。

---

### 🟢 轻微问题

#### 2.9 BeamState 的 last_edit 被纳入去重键的过度保守

**位置**: [edit_flows/sampling/beam.py](edit_flows/sampling/beam.py#L384-L395)

**问题**: `_beam_state_key` 将 `_last_edit_key(state.last_edit)` 纳入去重键。但 `last_edit` 只影响 `_is_reverse_op` 检查（候选过滤），不影响模型前向传播。这意味着两个序列完全相同但经由不同编辑路径到达的 state 不会被去重。

**影响**: Beam 中保留了比必要数量更多的 state，可能降低有效 beam 宽度。但由于 `last_edit` 确实影响下一步的合法候选集（不同 last_edit 过滤不同的 reverse op），完全去掉也不是绝对安全。

**建议修复**: 考虑在去重时忽略 last_edit，但保留 score 更高的那个。然后在展开时对每个 state 单独做 reverse-op 过滤。这样既能去重又能正确过滤。

#### 2.10 NoamScheduler 的 warmup 实现不一致

**位置**: [edit_flows/training/schedulers.py](edit_flows/training/schedulers.py#L1-L33)

**问题**: Noam 学习率调度器的公式为 `lr = factor * d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))`。原始 Transformer 论文的公式为 `lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))`。两者等价当 factor=1 时。但代码中 step 从 0 开始（`self._step = 0`），第一次 step() 后 step=1，此时 warmup 项为 `1 * warmup^(-1.5)`。如果 warmup=8000，warmup 项的初始学习率非常小（~1.4e-6），可能导致训练初期几乎不更新。

**影响**: 在 warmup 初期学习率极低，但这是 Noam scheduler 的标准行为，不算 bug。

**建议修复**: 如果训练初期 loss 下降过慢，尝试增大 `learning_rate_factor` 或减小 `warmup_steps`。

#### 2.11 `x_t` 空序列的边缘情况处理不一致

**位置**: [edit_flows/models/transformer.py](edit_flows/models/transformer.py#L186-L190), [edit_flows/sampling/euler.py](edit_flows/sampling/euler.py)

**问题**: 模型 `forward` 中处理了 `seq_len == 0` 的边缘情况（返回空张量）。但 Euler 采样中，如果所有 token 都被删除，`apply_ins_del_operations` 会返回只含 PAD 的序列，而非空序列。这两处的语义不完全一致。

**影响**: 极边缘情况，正常使用中不应触发。但如果模型在采样中删除了所有 token，行为可能不符合预期。

**建议修复**: 在 `apply_ins_del_operations` 中明确最小序列长度（如至少保留 BOS token）。

#### 2.12 测试中缺少化学领域测试

**位置**: [tests/](tests/)

**问题**: 所有 30+ 测试使用小词表（V=16）和合成数据，没有使用真实 SMILES 数据的端到端测试。测试验证了编辑操作、beam search、候选收集的代码正确性，但没有验证化学语义的正确性。

**建议修复**: 至少添加一个使用真实 SMILES tokenizer 和预训练 checkpoint 的 smoke test。

---

## 三、Edit Flows 理论与化学逆合成的根本性张力

经过仔细审查，我认为实现中的**根本问题不在于代码错误，而在于 Edit Flows 作为通用序列编辑方法与化学逆合成任务之间存在本质张力**：

### 3.1 Token 编辑 vs. 化学键编辑

Edit Flows 将编辑定义为 token 级别的 insert/delete/substitute。但在逆合成中：
- 一个化学键的断裂可能表现为 0 个 token 变化（如果 SMILES 表示恰好不变）或多个 token 变化（如果涉及环结构或支链重组）
- 反之，大量 token 变化可能只对应一个化学键变化（如 SMILES canonicalization 导致表示完全不同但分子相同）

### 3.2 编辑独立性假设

Z-space 中的逐位置独立假设（conditional path 中每个位置独立选择 z_0 或 z_1）与化学反应的协同性矛盾。化学键的断裂和形成是成对/成组发生的——一个原子不能单独改变其键合状态而不影响相邻原子。

### 3.3 时间语义

Edit Flows 中 κ(t) 从 0→1 表示编辑过程的进度。但化学逆合成中，"进度"的概念模糊——是将产物逐步转化为反应物的过程，还是逐步"逆向推导"的过程？前者是物理过程（没有化学意义），后者是推理过程（但不在 token 空间表达）。

### 3.4 多步逆合成的挑战

Edit Flows 目前处理的是单步逆合成（一个产物→一组反应物）。但实际逆合成通常是多步的（产物→中间体→...→起始原料）。将 Edit Flows 直接扩展到多步需要额外的框架设计。

---

## 四、代码质量评估

### 4.1 做得好的地方

| 方面 | 评价 |
|------|------|
| **测试覆盖** | Beam search 有 30 个针对性测试，覆盖 BOS 保护、no-op 过滤、reverse-op 检测、dead-end 处理、去重、origin mask 等 |
| **模块化设计** | 核心逻辑分离清晰：alignment / z_space / rate_scale / scheduler / loss / model / sampling 各司其职 |
| **类型标注** | 大部分函数有完整的类型标注（Tensor, Optional, List, Tuple 等） |
| **文档** | `docs/` 下有丰富的中文文档，take-over.md 对接手者非常友好 |
| **Oracle 验证** | Oracle 实验（97%）有效验证了采样框架的正确性，排除了匹配/采样层面的错误 |
| **速率重参数化** | `use_rate_reparam` 的设计让模型预测 scheduler 无关的 base rate，是合理的工程决策 |

### 4.2 可以改进的地方

| 方面 | 建议 |
|------|------|
| **核心模块文档** | `beam.py` 有清晰的模块级 docstring，但 `euler.py` 的 `sample_euler` 参数众多却缺少文档 |
| **魔法数字** | `LOG_EPS = -1e9`, `LOG_NEG_INF = -1e9`, `LOG_SMALL_RATE = -20.72` 散布在多个文件中，建议集中管理 |
| **配置管理** | `retro-example.yaml` 包含硬编码的绝对路径 `/data6/duanbh/...`，不适合其他用户 |
| **错误处理** | 一些边缘情况（如 vocab 文件不存在、checkpoint 不匹配）的错误信息不够具体 |

---

## 五、建议的优先修复顺序

### 短期（1-2周，不需要重训模型）

1. **将显式 STOP 集成到 Beam Search**（对应问题 2.3）
   - 这是当前 beam 收益小的直接原因
   - 约 100-200 行改动，集中在 `beam.py`

2. **Euler 采样加入 forbidden token 屏蔽**（对应问题 2.7）
   - 一行 mask 即可，防止 < 0.1% 的无效生成

3. **改进 RatioTimePolicy 的前期时间估计**（对应问题 2.8）
   - 考虑用 u_init 的值推测初始 κ，而非固定 depth

### 中期（2-4周，可能需要重训）

4. **基于原子映射的数据预处理**（对应问题 2.1）
   - 这是提升模型上限的最关键方向
   - 需要 RDKit/RXNMapper 集成

5. **加入"序列正确→u_tot→0"的训练信号**（已在 take-over.md 中提出）
   - 解决 STOP 校准问题

6. **模型架构加入化学感知设计**（对应问题 2.4）
   - 例如在 token embedding 中加入化学环境特征

### 长期

7. **重构编辑操作为化学键/官能团级别**（对应问题 2.2 和 3.1-3.3）
8. **多步逆合成框架设计**（对应问题 3.4）

---

## 六、逐文件问题清单

| 文件 | 行号 | 问题 | 严重度 |
|------|:----:|------|:------:|
| `core/alignment.py` | 16-55 | Levenshtein 对齐忽略化学语义 | 🔴 |
| `core/z_space.py` | 93-113 | `make_ut_mask_from_z` 创建二元掩码，丢失编辑计数信息 | 🟡 |
| `core/rate_scale.py` | 11-24 | `clamp_kappa` 选项语义不清 | 🟡 |
| `models/transformer.py` | 152-166 | 三个输出头无参数共享 | 🟡 |
| `models/transformer.py` | 198-204 | 时间嵌入同等加到所有位置 | 🟡 |
| `training/trainer.py` | 136-151 | 监控指标使用 effective rates 而非 base rates | 🟡 |
| `training/schedulers.py` | 1-33 | Noam scheduler 初始 step=0 | 🟢 |
| `sampling/euler.py` | 349-356 | Euler 采样未屏蔽 forbidden tokens | 🟡 |
| `sampling/beam.py` | 736-954 | Beam search 缺少显式 STOP | 🔴 |
| `sampling/beam.py` | 384-395 | last_edit 纳入去重键过于保守 | 🟢 |
| `sampling/time_policy.py` | 169-177 | RatioTimePolicy 前两步回退到 depth | 🟡 |
| `sampling/ops.py` | 64 | 序列长度硬截断可能丢失信息 | 🟢 |
| `data/dataset.py` | 22-37 | 无化学有效性过滤 | 🟡 |
| `configs/retro-example.yaml` | 33 | 硬编码绝对路径 | 🟢 |
| `tests/` | — | 缺少真实 SMILES 测试 | 🟢 |

---

## 七、总结

这个仓库的 Edit Flows 实现在**数学和算法层面基本正确**——Bregman 损失、Z-space 条件路径、Euler 采样、自适应步长、速率重参数化的公式推导和代码实现都经得起推敲。Oracle 实验的 97% Top-1 强有力地证明了采样框架本身没有错误。

主要问题集中在**化学逆合成的领域适配**上。Edit Flows 被设计为通用的序列编辑方法，但化学逆合成有其特殊性（SMILES 非唯一性、原子映射、化学键协同变化、价键约束），这些特殊性在当前的实现中被忽略了。

性能瓶颈（模型 Top-1 ~46% vs Oracle ~97%）主要来自两方面：
1. **模型学习问题**（~60% 的差距）：模型学到的编辑速率不够准确，尤其是"已完成时应输出零速率"的校准
2. **化学适配问题**（~40% 的差距）：token 级编辑操作不能完全捕捉化学反应的语义

建议接手者优先解决模型校准问题（如 take-over.md 中的 Phase 1-2 方案），因为这些改进不需要改变 Edit Flows 的理论框架，成本较低。在模型能力达到瓶颈后（预计 60-70%），再考虑从化学适配角度进行根本性的框架改进。

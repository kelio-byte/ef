# SPE 之后的下一阶段改进方案

更新日期：2026-08-24
状态：研究分析与实验规划；本轮没有修改模型、训练或采样代码，也没有启动实验。

## 1. 结论先行

SPE 的 tokenizer 选择已经冻结为 **global R-SMILES + SPE-M500**。下一步不应继续搜索 merge 数量，而应解决模型仍然存在的两个核心问题：

1. 九条轨迹仍可能在反应无关的位置做出早期编辑；
2. 当前模型在状态被修改后看不到一份不可变的原始产物，且被人为制造错误后几乎不会恢复。

建议按以下顺序推进：

| 优先级 | 方向 | 先做什么 | 是否需要重训基础模型 | 当前判断 |
|---:|---|---|---:|---|
| 1 | 反应中心感知的首编辑采样 | 先做真实中心上限，再训练 product-only 中心预测器 | 否 | 最值得优先验证 |
| 2 | 不可变产物条件 | 给动态状态提供始终可见的 product memory | 是 | 最值得进行的训练改造 |
| 3 | 自生成错误上的恢复训练 | 用错误中间状态训练模型恢复到目标 | 是 | 有直接诊断依据，但应接在 product memory 后 |
| 4 | DEL 稀有操作处理 | 先做 DEL 子集诊断，再决定辅助损失或 roll-in | 可能 | 保留 DEL，不能直接删除 |
| 5 | DGM 重启 | 只尝试中心/局部动作 guidance，不复用旧 reward 路线 | 先小后大 | 有条件重启，不是当前第一步 |

最推荐的第一个实质实验是：

> **训练一个只看产物图的反应中心预测器，把九条 M500 Euler 轨迹分配到 Top-3 中心假设，并且只在首个实质编辑时施加软偏置；首编辑以后全部恢复普通 Euler。**

它同时具有三点优势：直接针对早期有害编辑；不需要重训已有 M500 基础模型；不同中心天然提供比“随机选择 INS/SUB/DEL”更有化学意义的候选多样性。

## 2. 本方案依据的仓库事实

后续方案不是从一般直觉出发，而是由当前项目已经观测到的结果约束。

| 已知事实 | 当前证据 | 对下一步的含义 |
|---|---|---|
| M500 是当前 fragment-level baseline | `SPE/STATUS.md` | 不再搜索 M1000/M2000/Full |
| M500 自然轨迹中的有害首事件为 18.74%，Atom 为 25.86% | `revision/motivation_report.md` | 早期编辑质量仍是可改进对象 |
| 强制首 completion 出错后，Atom/M500 命中率都接近崩溃 | 同上 | 不能声称 M500 已具备后续纠错能力 |
| M500 编辑构成为 INS 67.34%、DEL 约 0.61%、SUB 32.05% | SPE 数据统计 | 操作分布不均衡，但 DEL 不能只凭总体占比判断 |
| 当前 Transformer 只输入动态状态 `x_t` 和时间 `t` | `edit_flows/models/transformer.py` | 原始 product 在后期可能被覆盖或丢失 |
| 主 M500 checkpoint 的 `use_origin_mask=false` | checkpoint/config 记录 | 不能在推理时临时打开 origin condition |
| 旧 DGM 端到端 Top-1 下降，时间增加约 47–49% | DGM 系列报告 | 旧 reward 和旧注入方式不应直接搬到 M500 |
| 原始 USPTO-50K CSV 保留 atom mapping | `datasets/USPTO_50K/raw_{train,val,test}.csv` | 可以从真实映射构建反应中心训练标签 |
| 原始训练数据的 `class` 全为 `UNK` | 本轮检查 40,008 条训练反应 | 当前不能依赖已知 reaction class 做条件引导 |

默认开发基线固定为：

```text
数据：datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500
checkpoint：new_checkpoints/spe_m500_checkpoints/checkpoint_step490000.pt
sampler：Euler
n_samples：9
n_steps：100
seed：42（开发通过后才补 7/123）
开发集：dev_unique1000_aug20，1000 个 reaction × 20 augmentation
```

完整 test 已经被用于多个 checkpoint 的报告，因此不能继续用它挑方法。新方法先用 dev；通过后，将 manifest 中尚未用于该分支的 `confirm_unique1000_aug20` 和 `final_unique2000_aug20` 投影到 M500，再按冻结协议使用。

## 3. 方向一：识别反应中心并利用起来

### 3.1 是否值得做

值得，而且是当前优先级最高的推理改进。

反应中心方法在逆合成中有明确先例。RetroXpert 先识别产物中的潜在断键中心，再从 synthons 生成反应物；LocalRetro 学习局部反应性并用全局注意力补充非局部信息；Graph2Edits/G²Retro 也把原子或键的变化作为生成入口。这些工作说明“先缩小可能发生变化的位置，再生成反应物”是成熟机制，而不是仅凭直觉提出的启发式。

它与我们的兼容点不是“把模型改成图生成器”，而是：

- global R-SMILES 已经尽量把产物与反应物的共同结构对齐；
- M500 将连续 token 合并，首个错误 fragment 的破坏更大；
- 现有诊断表明 M500 的优势主要是更少做出有害首事件，而不是事后恢复；
- 九条独立 Euler 轨迹正好可以分配给多个反应中心假设。

因此应先把反应中心用于**首编辑位置与轨迹分配**，而不是一开始重构整个基础模型。

### 3.2 反应中心应怎样定义

不能用“product 与 target 的字符串差异位置”直接充当化学反应中心。建议从原始 atom-mapped 反应构建图级标签：

1. 对 product 和 reactants 中 map number 相同的原子比较键集合；
2. product 中消失的键、键类型发生变化的键，其端点标为 bond center；
3. 原子电荷、氢数、手性等属性改变时，对应原子标为 atom center；
4. 对只存在于 reactants 的新片段，将其连接到 product 保留部分的 attachment atom 标为中心；
5. 同时保存中心本身以及半径 1、2 的邻域，允许模型使用软范围而非一个点。

这里必须允许一个反应有多个中心。G²Retro 等工作明确区分断键、键型变化和原子中心；把所有反应强行压成单中心会损失环开合或多位点变化。

### 3.3 如何映射到 SPE-M500

原始 CSV 有 atom mapping，但当前 M500 文本已经去掉 map number。不能靠行号直接拼接：原始 `raw_train.csv` 有 40,008 个反应，而当前训练集为 40,003 个 reaction block（每个 20 augmentation），中间存在过滤差异。

正确做法是：

1. 用去 map 后的 canonical product/reactants 和 reaction id 建立 raw→processed crosswalk；
2. 在生成每个 R-SMILES augmentation 时保留“字符串原子位置→原始 atom map”的映射；
3. SPE merge 时让每个 M500 token 同时合并其包含的 atom-map 集合；
4. 一个 M500 token 只要包含中心原子，或邻接中心键的端点，就得到对应中心标签；
5. 对 INS anchor，使用 anchor 两侧原子与中心的距离定义其位置分数。

SPE token 可能包含括号、环号或多个原子，因此不应把每个 SPE token 都称为独立的“化学片段”。内部文档中更准确的说法是 **fragment-level token**。

### 3.4 分阶段实验

#### RC-P0：标签和兼容性审计，不训练

先回答“化学中心是否真的覆盖当前字符串编辑位置”。

需要统计：

- 每个反应的中心数量及半径分布；
- oracle 首编辑位置落在中心半径 0/1/2 内的比例；
- INS/SUB/DEL 分别与中心的距离；
- 一个 M500 token 同时覆盖中心与大量非中心原子的比例；
- 同一反应 20 个 augmentation 的中心投影是否保持一致；
- 单中心、多中心、无可识别中心反应的比例。

随后做一个只用于诊断的 **true-center upper bound**：在 dev 上用真实中心偏置首编辑，但真实中心绝不进入正式推理。如果连真实中心都不能在同预算下改善首事件质量或 Top-k，则说明当前字符串编辑与图中心的映射不兼容，应停止该方向，不必训练中心预测器。

#### RC-P1：product-only 中心预测器

若 upper bound 有效，再训练一个小型产物图模型：

- 输入：去 atom map 的 product molecular graph；
- 输出：每条 product bond 和每个 atom 的中心概率，以及中心数量分布；
- 标签：RC-P0 构建的 atom/bond center；
- split：按原始 reaction 划分，20 augmentation 不能跨 split；
- 指标：bond/atom PR-AUC、Top-1/Top-3 center recall、多中心完整召回、校准误差；不能只报被大量负样本抬高的 accuracy。

图模型只对每个 product 运行一次，其结果可供 20 个 augmentation 和九条轨迹共享，额外推理成本应远低于每个 Euler step 再跑一个 guidance 网络。

#### RC-P2：中心分层的首编辑采样

主方案只测试一个配置，避免重新陷入大规模搜索：

```text
Top-3 预测中心 × 每个中心 3 条轨迹 = 9 条独立轨迹
中心只影响第一次实际发生的编辑
第一次编辑以后，九条轨迹全部恢复普通 Euler M=1
不同轨迹之间不竞争、不合并
```

中心偏置应是软偏置而不是 hard mask。可将位置—操作速率写成：

```text
lambda_center(i, mode) = lambda_base(i, mode) * exp(alpha * center_score(i))
```

然后对全序列所有合法动作统一缩放，使总 edit rate 与 base 保持一致。这样主要改变“第一次在哪里编辑”，而不是强迫模型编辑更多。INS/SUB 的 token completion 仍由基础 M500 模型给出；中心模型不负责猜 fragment token。

这与旧 structured diversification 的机制不同：旧方法按模型自身的 INS/SUB/DEL 和 token 排名强行分叉，缺少化学位置依据；中心分层按不同潜在反应位点保护谱系。

### 3.5 通过与停止条件

RC-P2 与普通 Euler 必须使用相同的九条轨迹、100 steps、augmentation、seed 和模型前向预算，报告：Top-1/3/5/10、Oracle、Invalid、unique candidates、首分叉重复率和 wall-clock。

进入多 seed/confirm 的条件：

- Top-1 点估计不低于普通 Euler；
- Top-3/Top-10/Oracle 至少一项提高；
- Invalid 增加不超过 0.5 pp；
- paired reaction bootstrap 不显示明确负向；
- 中心预测和映射耗时被完整计入。

若 true-center upper bound 无收益，停止；若 true center 有收益但 predicted center 无收益，问题在中心预测器；若 predicted center 能减少有害首事件但最终 Top-k 不升，说明“局部首步更合理”仍不足以解决完整生成，此时不继续调 `alpha`。

## 4. 方向二：删除操作很少，是否应删除或改名

### 4.1 结论：不应删除 DEL

本轮直接扫描 M500 训练对齐文件得到：

| DEL 统计 | 数值 |
|---|---:|
| 训练行数 | 800,060 |
| DEL 操作数 | 20,364 |
| DEL 占全部非恒等编辑 | 0.6167% |
| 至少包含一个 DEL 的 augmentation 行 | 17,683（2.2102%） |
| 至少一个 augmentation 的当前对齐中含 DEL 的 reaction block | 2,698 / 40,003（6.7445%） |
| target token 数短于 source 的 augmentation 行 | 13,829（1.7285%） |
| 至少一个 augmentation 出现 target 更短的 reaction block | 1,665 / 40,003（4.1622%） |

所以“DEL 只占 0.61%”不等于“DEL 只影响 0.61% 的反应”。这里的 6.74% 是当前 Levenshtein 对齐下的 reaction-level 覆盖率，不代表每个案例都不存在另一种无 DEL 对齐；但 target 确实短于 source 的具体序列在 INS+SUB 状态空间中必然不可达。如果完全删除 DEL：

- 对 target 比 source 短的具体序列，只有 INS+SUB 无法降低长度，目标在当前状态空间中不可达；
- 错误插入后也失去真正移除多余 token 的操作；
- 仅少输出一个 `lambda_del` 标量，几乎不会降低 Transformer 或 568 词表 softmax 的主要计算量。

把 DEL 改成 `SUB→EMPTY` 只是把删除重新编码成一个特殊替换，不会解决稀有监督，还会引入空 token、长度更新和采样语义的新复杂度。把 SUB 拆成 DEL+INS 也会增加步数，并损失 M500 已经证明有价值的 fragment rewrite。

插入和删除共同提供可变长度、可反复修订的序列支持，这也是 Edit Flows、Levenshtein Transformer 和 Insertion-Deletion Transformer 保留两类操作的根本原因。

### 4.2 是否可以换名字

可以在汇报层使用更直观的解释，但不要改代码中的操作语义：

| 代码名 | 推荐中文解释 | 不建议的说法 |
|---|---|---|
| INS | 插入一个 fragment-level token | 接上一个真实化学片段 |
| SUB | 改写一个 fragment-level token | 一定发生化学取代反应 |
| DEL | 移除一个 fragment-level token | 一定发生化学消除反应 |

SPE token 是字符串压缩单位，不总是独立、可解释的化学基团。因此不要把 INS/SUB/DEL 重命名成具体有机反应类型；那会造成语义错误。

### 4.3 更合理的处理方式

先做 `DEL-D0` 诊断，而不是立刻重训：

1. 在 teacher-forced 中间状态上统计 DEL 的 mode rank、概率校准和位置召回；
2. 在 dev 中单独报告“需要 DEL”“target 更短”“不需要 DEL”三类反应的 Top-k/Invalid；
3. 检查失败到底是模型不给 DEL rate，还是 DEL 位置正确但后续 token completion 失败；
4. 区分必要 DEL 与采样后为了清除错误插入而产生的恢复 DEL。

只有确认 DEL-required 子集显著更差后，才考虑下面两个训练方案：

- **小权重的 operation-type 辅助损失**：保留原 Bregman loss，只在确有编辑的对齐位置增加平衡后的 INS/SUB/DEL 分类损失；先固定一个很小的权重，不在 dev 扫描大量权重。
- **错误 roll-in 恢复训练**：给中间状态注入一个模型偏好的错误 INS，再训练模型使用 SUB/DEL 回到目标。这比简单复制稀有 DEL 样本更贴近我们已经观察到的“首错后无法恢复”。

不建议直接对 Bregman 主损失中的 DEL 粗暴乘大权重，因为这会改变 CTMC 的目标速率，而不只是改善分类曝光；它可能让采样过度删除并增加 invalid。

## 5. 方向三：DGM 是否应该继续

### 5.1 结论：可以有条件重启，但不能原样重跑

旧 DGM 失败不是单纯因为 atom-level 序列太长：

- E5/E6/E7 的 Top-1 均低于普通 Euler，最好仍约低 1.5 pp；
- 每步增加一个 guidance forward，端到端时间增加约 47–49%；
- forward reward 对正确候选有信号，但约 46.05% 的错误终点也得到正 reward；
- 大规模 ranker 中，raw forward reward 的 holdout Top-1 为 47.7%，learned residual/listwise 只有 38.1%/37.0%；
- candidate-level AUC 上升没有转化为 reaction-level Top-1；
- 旧实现按位置归一化，主要改变同一位置的 INS/SUB/DEL 偏好，不能有效回答“应该在哪个反应中心编辑”。

M500 使序列更短、平均编辑数更少、SUB 更多，这可能缩短 credit assignment 距离；但它没有自动修复 reward 错配，也没有消除 INS/DEL 带来的可变长度状态空间。因此，直接把旧 checkpoint、forward reward 和 `beta=0.1` 搬到 M500，成功依据不足。

此外，DGM 论文的“exact guidance”依赖可计算的目标/源密度比及其离散状态假设。当前 Edit Flows 的 insertion、deletion、GAP 和位置移动使旧 action-level 实现只能称为 DGM-inspired guidance，不能宣称拥有原论文的严格等价性。

### 5.2 更合理的重启方式：Reaction-Center Guidance

若 RC-P2 已证明预测中心有用，可以把反应中心作为比 forward reward 更局部、更干净的 guidance 信号：

1. 条件不是“forward model 喜欢这个终点”，而是“这个动作是否与预测中心假设一致”；
2. 使用 atom-mapped 训练反应产生中心和局部 action 标签；
3. `H(product, x_t, t, action)` 对位置与操作联合打分；
4. guidance 作用于跨位置的完整 action rate，并保持全局总 edit hazard，而不是在每个位置内部归一化；
5. 第一阶段只 guidance 首个实质编辑，验证后才考虑扩展到前 10–20% 时间。

这一版本仍应诚实称为 **reaction-center action guidance**。除非重新完成适用于可变长度 edit space 的推导，否则不要称为 exact DGM。

### 5.3 DGM 重启 gate

按以下顺序，不允许跳步：

1. **同候选动作离线 gate**：在相同 `(product, x_t, t)` 下，中心 guidance 是否提高 oracle-compatible action 的组内排序，同时不压低基础模型 Bregman 支持的动作；
2. **首事件 gate**：与更简单的 RC-P2 软偏置比较，是否进一步减少有害首事件；
3. **dev 端到端 gate**：同预算 Top-1 不下降，并至少改善一个深层指标；
4. **成本 gate**：若效果与 RC-P2 持平却明显更慢，直接选择 RC-P2，停止 DGM；
5. 只有前三项均通过，才训练多时间点 shared-anchor guidance，并进入 confirm。

旧 forward reward、旧 correctness ranker 和 atom-level guidance checkpoint 都只作为负结果存档，不作为新分支初始化目标。

## 6. 方向四：其他值得做的改进

### 6.1 不可变 product memory：最优先的训练改造

当前 `EditFlowsTransformer.forward()` 只接收 `tokens=x_t`、`time_step` 和 padding/origin mask。随着 INS/SUB/DEL 发生，原产物信息可能被改写；主 M500 模型又没有 origin embedding。这与逆合成的条件生成本质不完全匹配。

EditRetro 的 encoder 始终编码原 product，迭代 decoder 每轮都通过该 memory 获得条件；这为我们提供了直接的结构依据。建议先实现最小单变量对照：

```text
product encoder：只编码初始 M500 product，一次计算并缓存
state encoder：编码当前 x_t 和 t
cross-attention：state 查询不可变 product memory
输出 head：仍为现有 INS/SUB/DEL rate 与 INS/SUB token distribution
loss / scheduler / tokenizer：全部保持不变
```

M500 平均序列约 16，比 atom-level 约 51 短，因此增加 product memory 的成本在当前分支更可控。该实验需要从头训练，不能向旧 checkpoint 临时添加 cross-attention 后直接推理。

主要评估除 Top-k 外，还应比较：

- 首事件有害率；
- 后期状态与原 product 的信息保留；
- invalid；
- forced-wrong 后的恢复率；
- 每步速度和显存。

### 6.2 模型状态 roll-in：让模型真正见过自己的错误

当前训练状态来自 product/target 对齐路径，而真实推理状态包含模型自己产生的错误 fragment。我们的强制错误实验已经证明两者存在明显分布差异。

RetroXpert 使用不成功的预测 synthons 增强 reactant generator，但其 ablation 显示这种增强在已知 reaction type 时小幅改善、在未知 reaction type 时反而下降；EditRetro 则把模型生成/扰动状态用于后续 refinement 训练。由于本仓库的 reaction class 全为 `UNK`，这些工作只支持“应当严格验证 model-state roll-in”，并不保证它一定有效。对本项目最兼容、同时可证伪的版本是：

1. 先保留不可变 product memory；
2. 冻结一个 M500 模型生成中间状态；
3. 对 20–30% 训练 batch 使用一次高概率但 target 不支持的 INS/SUB 作为 roll-in；
4. 从该错误状态继续对真实 target 计算训练目标；
5. 其余 batch 保持原始 flow objective，避免模型只学纠错而损失正常生成。

第一轮只做“一次错误、固定混合比例”的方案，不同时搜索错误深度、多个概率和多个损失权重。若普通 Top-1 下降且 forced-wrong recovery 没有提高，立即停止。

### 6.3 利用 20 augmentation 的一致性

当前 20 个 R-SMILES augmentation 主要被当作独立训练行和推理候选来源。它们代表同一个分子图和同一个化学中心，却有不同 token 位置。

可在 reaction-center predictor 或 product memory 上增加轻量一致性约束：同一反应不同 augmentation 的 pooled product embedding、中心概率映射回分子图后应一致。不要强迫 token 位置 logits 逐点一致，因为不同 R-SMILES 的位置没有天然一一对应。

这个方向应在 center/product-memory 主实验有效后再加入，否则会同时改变架构与损失，难以归因。

### 6.4 语法约束只作为低优先级 invalid 改进

M500 仍有约 12% 的 rank-1 invalid。可以研究括号、环号和特殊 token 的 action mask，但不能简单要求每个中间状态都能被 RDKit 解析：Edit Flows 的合法中间编辑状态可能暂时不是完整 SMILES，hard prune 会误杀可恢复路径。

更安全的入口是只约束确定不可能合法的局部动作，例如：

- 结构 token/BOS/PAD 的既有硬屏蔽；
- 明显无法闭合的最终 ring/parenthesis 状态；
- 最后若干步的软语法 penalty，而不是全程 hard mask。

只有 invalid 明显下降且 Top-1 不受损时才保留。该方向不能替代反应中心或 product conditioning。

## 7. 推荐执行路线

### 第一阶段：不重训基础模型

1. 构建 atom-mapped reaction center 标签和 raw→M500 crosswalk；
2. 完成 RC-P0 locality audit；
3. 用 true-center 做首编辑 upper bound；
4. upper bound 通过后训练仅看产物的反应中心预测器；
5. 运行 Top-3 center × 3 trajectories 的 RC-P2；

这一阶段只回答：化学中心是否能为当前 M500 Euler 提供真实增量。操作集合调整不纳入本阶段。

### 第二阶段：只开一条完整训练线

若 RC-P2 有收益，优先训练 **M500 + immutable product memory**。中心预测可作为额外输入，但第一轮最好保持单变量：先只加 product memory。

如果 product memory 正向，再加入单错误 roll-in；不要同时修改 dropout、学习率、scheduler、operation weight 和 tokenizer。

### 第三阶段：有条件恢复 guidance

只有中心信号已通过上限、预测和端到端三个 gate，才做 reaction-center action guidance。若简单中心软偏置已经取得相同收益，就没有必要支付 DGM 每步额外前向的成本。

## 8. 统一实验表与止损规则

每次端到端实验统一报告：

| 质量 | 覆盖/多样性 | 稳定性 | 效率 | 机制指标 |
|---|---|---|---|---|
| Top-1/3/5/10 | Oracle、unique candidates | Invalid、3 seed 波动 | wall-clock、NFE、显存 | 中心命中、首事件有害率、恢复率 |

统一止损：

- 只改善离线 center/AUC，不改善 reaction-level Top-k：不进入 confirm；
- Top-1 下降且没有清晰 Oracle/Top-K 补偿：停止；
- 相同效果但比简单方案明显更慢：选择简单方案；
- 需要通过在已查看的完整 test 上挑 checkpoint/参数才能成立：不接受；
- DGM/reward 只提高 candidate-level AUC、再次降低 reaction-level Top-1：立即关闭。

## 9. 当前建议的明确答案

1. **反应中心：应该做。** 先做真实中心 upper bound，再训练仅看产物图的反应中心预测器，最后只偏置首个实质编辑并按 Top-3 中心分配九条轨迹。
2. **DGM：不要原样重跑。** 只有反应中心信号通过简单采样 gate 后，才尝试 center/action guidance；旧 forward reward、旧 ranker 和旧按位置归一化方案不继续。
3. **训练改造：优先 product memory，其次 roll-in。** 这是当前模型结构和纠错实验共同指出的缺口。

## 10. 主要文献依据

- [Edit Flows: Flow Matching with Edit Operations](https://arxiv.org/abs/2506.09018)：当前 INS/DEL/SUB 可变长度 CTMC 的基础，说明三种编辑共同定义序列状态空间。
- [Discrete Guidance Matching: Exact Guidance for Discrete Flow Matching](https://arxiv.org/abs/2509.21912)：DGM 的密度比与 exact guidance 来源；也用于界定当前可变长度 action guidance 不能直接宣称严格等价。
- [RetroXpert: Decompose Retrosynthesis Prediction Like A Chemist](https://proceedings.neurips.cc/paper/2020/hash/819f46e52c25763a55cc642422644317-Abstract.html)：先预测反应中心/断键再完成 reactants；其失败 synthon 增强只在 reaction type known 设置中略有收益，在 unknown 设置中反而下降，因此本项目不能把它当作已证实有效的 roll-in 方案。
- [Deep Retrosynthetic Reaction Prediction using Local Reactivity and Global Attention (LocalRetro)](https://pubs.acs.org/doi/10.1021/jacsau.1c00246)：以局部反应性和全局上下文识别可反应位点。
- [Graph2Edits](https://www.nature.com/articles/s41467-023-38851-5)：在分子图上连续预测反应相关编辑，支持中心/编辑联合建模的化学合理性。
- [G²Retro](https://www.nature.com/articles/s42004-023-00897-3)：显式区分断键、键型改变与原子中心，说明不能把所有反应压成单一中心类型。
- [EditRetro](https://www.nature.com/articles/s41467-024-50617-1)：保留不可变 product encoder memory、迭代编辑和自生成状态继续 refinement；同时指出 SPE 子结构不一定具有化学可解释性。
- [Levenshtein Transformer](https://arxiv.org/abs/1905.11006) 与 [Insertion-Deletion Transformer](https://arxiv.org/abs/2001.05540)：插入和删除的互补性以及用模型输出训练 deletion/recovery 的依据。
- [Root-aligned SMILES](https://pubs.rsc.org/en/content/articlelanding/2022/sc/d2sc02763a)：以共同 root 降低 product/reactants 的序列编辑距离，为当前 global R-SMILES 与中心映射提供表示层依据。

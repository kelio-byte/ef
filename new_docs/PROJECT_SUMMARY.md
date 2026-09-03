# Edit Flows 项目总结

> 本文按研究问题和证据强度重组截至 2026-08-12 的工作。早期 tiny、smoke、可视化和用于选参的开发实验只保留其提出假设的作用；除非另有说明，数字均不是完整测试集上的最终性能声明。

## 1. 项目目标、基础流程与评估对象

项目的目标是从产物 SMILES 生成反应物 SMILES，用于单步逆合成。Edit Flows 将这一过程表示为从产物序列到反应物序列的离散、可变长度连续时间编辑流：模型在每个时间点预测插入、替换、删除的速率及 token 分布，采样器用 100 个 Euler 步逐步编辑序列。

训练学到的是编辑过程的局部转移；推理时真正要解决的是如何在有限计算预算内生成、保留并排序最终反应物候选。评估会对同一反应使用 20 个产物 SMILES augmentation，再聚合候选得到最终 Top-k。因此必须区分两件事：

- **Oracle-any** 衡量正确反应物是否曾被生成，主要是覆盖率；
- **Top-k** 衡量它在聚合后的排序位置，受候选质量和排序共同影响。

augmentation 或多次采样带来的“至少命中一次”不能被当作单轨迹能力，更不能把每个 augmentation 当作独立反应。后续严格评估均以 reaction 为统计单位。

## 2. 基础模型是否可信：训练审计、修复与重训对照

对历史 checkpoint_step600000.pt 的审计结论是：核心 Bregman 损失、时间对齐和速率建模与预期方法一致，checkpoint 数值正常；已使用的 800,060 条预对齐训练样本都能和原始数据对应。因此，数据加载器的 fallback/PAD 风险和 zip() 截断风险是需要修复的工程缺陷，但**没有证据表明它们污染了这份历史 checkpoint**。

审计同时确认了会降低训练可复现性或训练规范性的事项：第一次更新在 Noam 学习率调度前使用了 Adam 默认学习率；fallback 对 PAD 的处理不稳健；resume 可能重复更新且未完整保存随机状态；训练没有验证集、最佳 checkpoint 或早停。新版训练路径修复了这些问题，并加入数据一致性检查、确定性 sampler/RNG 恢复及验证/TensorBoard 记录。

新旧 600k checkpoint 在相同的 R9K1M2、mini-1001 条件下几乎持平：旧模型 Top-1/3/10/Oracle 为 57.0/78.1/86.1/91.8%，新模型为 58.2/77.9/86.4/91.5%，invalid 从 13.7% 降至 12.8%。这说明修复后的训练基础可用于后续研究，且没有明显退化；它**不**证明新 checkpoint 本身带来了显著模型能力提升。

仍需保留两个模型层面的限制。第一，演化状态中没有不可变的显式产物条件，模型可以删除或替换本应固定的产物信息。第二，目标编辑极不均衡（插入约 91.2%，替换约 7.9%，删除约 0.9%），这解释了若干编辑类型弱点，但尚未证明某种简单重加权能解决它们。故当前基础模型是“可用且训练过程已可审计”的基线，而非已完全排除结构性瓶颈的终点。

## 3. 基础 Euler 采样暴露的真正问题

早期轨迹可视化和极小样本实验首先表明：不同 SMILES 表示、晚期编辑和 token 选择都会使单条轨迹高度不稳定；把 20 个 augmentation 与多次采样混在一起，能把很低的单次命中率放大为很高的“至少一次命中率”。这些实验只支持诊断和假设，不支持性能结论。

在修正早期数值/计分问题后，问题被收敛为四个相互关联的推理问题：

1. 正确候选是否被生成（覆盖率）；
2. 已生成的正确候选能否进入 Top-k（排序）；
3. 更大的采样预算是在增加互补轨迹，还是只在复制低价值分支；
4. invalid、模型前向次数和 augmentation 聚合如何限制可用预算。

这也是后续 Euler-Beam 研究的起点：它不是单纯追求更多候选，而是检验计算应当投入到独立轨迹、局部分支还是全局竞争。

## 4. Euler-Beam：从“宽 beam”直觉到受控实验结论

Euler-Beam 的三个参数代表不同资源分配方式：R 是彼此隔离的搜索岛/独立谱系数，K 是每个岛内长期保留的父状态数，M 是每个父状态本步采样并局部选择的子状态数。当前实现对相同状态合并概率质量，并以状态质量加一个“已发生编辑”的二元 bonus 排序；它不是化学 reward，也没有未来价值估计。

最初的困难是：严格累积完整轨迹概率会偏爱“不编辑”的路径。早期看似很强的 triggered-only 或反向计分，实际上是通过偏向低概率、更多编辑的路径获得结果，不能视为概率校准的 beam 证据。以子样本质量而不是重复乘样本动作概率、再用小的 changed-state bonus 抵消 no-op 偏置，才形成了可比较的 M>1 原型。

随后在旧 checkpoint、相同九条初始流和相同输出宽度的 mini-1001 受控对照中，隔离谱系明显优于把宽度集中到全局池：

| 配置 | Top-1 / Top-3 / Top-10 / Oracle | 判断 |
| --- | --- | --- |
| R9K1M2 | 57.0 / 78.1 / 86.1 / 91.8% | 该对照内最好的准确率/覆盖率配置 |
| R3K3M2 | 55.1 / 74.0 / 84.5 / 89.3% | 折中，但损失了深层 Top-k 与覆盖 |
| R1K9M2 | 54.1 / 72.9 / 83.5 / 87.0% | 约快 54%，但全局竞争过早淘汰有价值谱系 |

R9K1 对 R1K9 的 reaction-level 配对检验也支持 R9 的 Top-1、Top-3 和 Oracle 优势。R1K9 虽产生更多原始字符串，但 Oracle 更低，说明“表面多样性”不等于有用覆盖。在独立的 R10 局部选择对照中，M2 相比 M1 的 Oracle 仅约增加 0.1 个百分点，却使 Top-3/Top-5 分别提高约 2.2/2.3 个百分点，代价约为 13%；M3 进一步增加约 22% 时间且 Top-1 至 Top-10 与 Oracle 均回退。其收益主要是**局部候选的排序集中**，不是普遍扩大候选覆盖。

因此，当前 R9K1M2 应准确理解为“九条被保护的多尝试 Euler 谱系，每步在两个局部孩子间选择”，而不是成熟的大宽度 beam。它的支持证据是保护谱系和轻量局部选择优于当前全局大 K 竞争；并不支持“更大 K/M 一般更好”。与普通 Euler 的 200-reaction 对照方向上有利于 R9，但未达到足以作为全面优越性结论的统计强度。R9 的结果也主要来自已报告的 mini/开发条件，尚不是完整测试集定论。

两个早期方向已被收回。降低 token temperature 到 0.9 的 50-reaction 信号未能在 200-reaction 对照复现，默认仍为 1.0；“Euler-Beam 在 tiny 实验中普遍优于 Euler”的早期说法已被后续严格对照修正。另一个值得保留的工程事实是，相同状态共享模型前向可节省约 26% 小规模时间且不改变算法语义，但因数值可复现性细节仍应作为 opt-in 优化。

## 5. 为什么研究从启发式扩 beam 转向概率化偏好与 guidance

Euler-Beam 说明，单靠分支数和“是否已经编辑”的启发式分数没有能力判断一条中间状态是否会到达正确反应物。扩大 K 或 M 只会产生更多需立即比较的候选，反而会伤害低当前质量但未来有价值的谱系；R9 虽有效，计算代价也很高。Oracle 与 Top-k 之间仍有间隙，不能靠更大的全局池自动消除。

因此研究问题转为：能否为采样定义一个比当前质量/bonus 更有意义的目标偏好，并把它注入中间步骤？SMC/importance weighting 提供了“基模型 proposal 加目标 reward”的框架，DGM 则提供学习条件密度比 guidance 的动机。这里的转向是研究假设，不是 SMC 或 DGM 已成功的结论：Euler-SMC 的权重、ESS、重采样和祖先追踪已通过 target=proposal 的机制测试；在该测试中权重本应均匀，尚未验证独立 reward-twisted SMC 的准确率收益。

项目采用的终端 reward 来自冻结的 Molecular Transformer 正向模型：候选反应物能在 top-5 重建输入产物时按 rank 给分。它不使用评测目标反应物，因而可用于推理；但它只是有噪声的代理，而非真值。

## 6. DGM / action-level guidance：理论边界、实验与负结果

### 6.1 实际实现不是严格原始 DGM

原始 DGM 的严格推导依赖固定坐标上的后验对应。Edit Flows 有插入、删除和 GAP 的可变长度编辑，其中约 89.7% 的插入位于连续 GAP run，无法唯一映射到固定坐标；因此不能宣称与原始 DGM 严格等价。

当前已实现的是**动作级近似 guidance**：冻结的 Edit Flows 和正向 reward 模型之外，训练一个约 5.26M 参数、同时编码产物与当前状态的 guidance 网络，输出正的插入/替换/删除因子。在每个 Euler 步将基础动作率乘以 H^β（β=0.1），并在每个位置重新归一化，故它改变的是该位置动作类型的相对偏好，而非跨位置的总编辑强度。β=0 和常数 guidance 已逐字节复现基础采样，BOS masking 与数据/采样链路也已通过机制测试。代价是每步额外一次 guidance 前向，端到端时间约增加 47--49%；它不是加速器。

### 6.2 reward 与 credit assignment 如何被逐步收紧

终端 reward 确有信息：在保留的多时间点评估中，正确候选的全局 AUC 为 0.6971、同一 anchor 内 AUC 为 0.7308；但 46.05% 的错误终端仍得到正 reward。因此它可用于提出偏好，不能当作可靠的正确性标签。

guidance 数据经历了必要的修正。最初每个产物仅有两条终端记录，几乎没有产物内 reward 变化；扩大到多终端记录后，不同轨迹的中间状态和时间不同，pairwise 比较又不合法。共享 (product, x_t, t) anchor 的多 continuation 解决了可比性，五个时间点补足了只训练中段的盲区。即使如此，训练仍把终端 reward 分配给从中间状态到终点的编辑动作，只是一种粗粒度 endpoint credit，而不是可信的一步价值。

尝试用自然的单 Euler 步 transition 获得局部 credit 时，只有约 5.1% 样本发生变化，严格可区分组仅约 0.4%，数据不足。event-conditioned 单原子动作构造在 pilot 中把可区分组提高到 39.2%，解决了可用数据量，却没有通过候选 transition 的 Bregman/pairwise 门槛；其随后 dev-1000 端点采样 Top-1 为 55.7%，低于 58.2% 基线，且时间增加 49.1%。这否定的是该已测试的局部 credit 方案，不是否定所有 transition value 的可能性。

### 6.3 反应级协议下的当前开发集判断

为避免历史 A/B 开发数据被反复选参，评估被重设为按 reaction 聚合、20 个 augmentation 分组，并排除历史 [0,600)；当前使用新的 dev-1000，confirm-1000 和 final-2000 均未使用。以下均为新 checkpoint、普通 Euler、每 augmentation 3 个样本、100 步的开发集结果：

| 方法 | Top-1 | Top-3 | Top-10 | Oracle | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| E1 基线 | 58.2 | 75.5 | 83.5 | 86.6 | 比较基准 |
| E3 仅终端 rerank | 57.6 | 76.7 | 85.0 | 86.6 | 深层排序信号存在，Top-1 未过门槛 |
| E5 共享 anchor guidance | 56.2 | 76.1 | 84.9 | 87.5 | Top-1 下降 2.0pp；不通过 |
| E6 多时间点/pairwise guidance | 56.4 | 76.3 | 84.2 | 87.7 | 不通过 |
| E7 E6 加训至 2000 epoch | 56.7 | 75.9 | 83.8 | 86.6 | 不通过，且约慢 48% |

E5 的 Top-1 配对置信区间为 [-3.9, -0.2] 个百分点；E7 的深层改善置信区间则跨越零。所有 E3--E7 都未满足预先采用的 Top-1 不低于基线且有深层收益、invalid 不恶化的门槛，故没有进入 confirm 或 final，更没有作为默认采样器或与 Euler-Beam 叠加。

这组结果必须与离线训练指标一起解读：E7 的保留 pair accuracy 为 63.75%，高于控制组的 59.07%，Bregman 也受保护，说明 guidance 学到了更好的**局部 reward 排序**；但这没有转化为稳定的逆合成 Top-1，反而给出负向点估计。加训没有改变这一点，Oracle/invalid 也没有发生足以解释结果的崩塌。现有证据因而更指向 reward 与真实逆合成正确性的失配、终端到动作的 credit assignment 失配，以及动作级、固定位置总强度的 guidance 表达受限；这是一组有根据的诊断，而非已被单独因果证明的归因。外部 reward 校准的 P1/P2 分支同样未改善同池 Top-k，已关闭。

### 6.4 新 correctness reward 的一次冻结 gate

先做一个实现勘误：旧版 correctness-reward 脚本曾把真实 target 的 canonical component count 作为 product feature，故旧 P2/P3 的 AUC 和 rerank 数字不再是干净证据。修复后的实现只从序列化 product/candidate tokens 构造特征；target 仅用于候选池固定后的离线标签。旧结果保留在历史执行报告中，但不应继续被引用。

修复后先在 `1000/200/200` reaction 的 v2 split 上看到同样的 Top-1 下降；为排除小样本解释，又冻结 v3 大规模复核：ranker train/validation/holdout 分别为 `8000/1000/1000` 个独立 reaction（10000–17999、18000–18999、19000–19999）。validation 上 raw/residual/listwise Top-1 为 `47.4/39.5/40.1%`；一次性 holdout 上为 `47.7/38.1/37.0%`，Top-10 为 `80.4/80.3/80.4%`，三者 Oracle 均为 `80.4%`。residual 的 holdout global AUC 从 0.6922 提升到 0.7167，但 Top-1 差值的 1000-reaction paired bootstrap 95% CI 为 [-13.0, -6.3] pp；listwise CI 为 [-13.9, -7.5] pp。扩大数据后失败模式仍稳定，因此正式关闭当前 rerank 路线：不再扩大该 ranker、不构造 guidance data、不重训 DGM，也不运行改进后 `visualization_trajectory`。完整协议与报告见 [`dgm_reward_ranker_v3_large_protocol.md`](dgm_reward_ranker_v3_large_protocol.md) 和 [`dgm_reward_ranker_v3_large_report.md`](dgm_reward_ranker_v3_large_report.md)。

## 7. 当前项目状态

### 已经确定的

- 基础训练实现已完成关键审计与可复现性修复；新 checkpoint 可作为后续推理研究的可靠基线，但没有性能跃升的证据。
- 在当前质量函数下，保护独立谱系和 M2 局部选择有比全局大 K 更好的受控 mini 证据；它们分别改善有用覆盖和候选排序，代价是较高计算。
- 正向模型 reward 有真实但有限的判别力；它可以给深层终端排序提供信号，却尚不能稳定改善 Top-1。
- action-level guidance 的实现、数据管线、基线恒等性检查和 reaction-level 评估协议已建立；现有 learned guidance 在未触碰的确认/最终集合之前已被开发集门槛阻止。

### 已否定或暂不支持的

- 早期 tiny 结果不足以支持“宽 beam 普遍优于 Euler”；更大 K、更多 M 或全局竞争不是当前的准确率路线。
- Q temperature 小于 1 不是默认改进。
- 当前方法不是严格原始 DGM，不能以其理论保证解释结果。
- guidance 的离线 reward 排序提升不等于最终 Top-1 提升；现有 E3--E7、P1/P2 以及已测试的 event-conditioned local credit 均未通过采用门槛。
- 修复 target leakage 并扩大到 1000-reaction holdout 后，bounded residual 仍出现“candidate AUC 提高、reaction Top-1 明确下降”，listwise 更差；因此当前小型 endpoint reranker 已正式关闭，不能接到 DGM。
- Euler-SMC 目前只有机制正确性，不具备准确率成功证据。

### 仍未解决的核心问题

1. 如何获得与真实逆合成成功更一致、又不泄漏评测目标的 reward/偏好信号，而不是再叠加一个无法超过 Molecular Transformer 的小型 reranker；
2. 如何为可变长度编辑建立可信的中间动作价值或 credit assignment，而非把终端 reward 粗略复制到整段路径；
3. 是否应在固定训练基础设施上重新设计具有不可变显式产物条件、并缓解编辑不均衡的基础模型；
4. 如何把推理预算用于未来价值和谱系多样性，而不是盲目增大 K/M；任何新配置都应先冻结并通过未使用的 confirm/final reaction-level 集合，再报告完整测试结论。

## Source Map

核心训练证据见 training_code_audit.md、training_tensorboard_and_fixes.md、new_checkpoint_validation_parameter_sweep.md；Euler 采样与受控搜索证据见 sampling_overview.md、euler_beam_current_situation.md、euler_beam_next_stage_plan.md；DGM 的最新协议、数据和 reward 审计见 dgm_evaluation_v2.md、dgm_multitime_guidance_data.md、dgm_local_credit_assignment_plan.md、dgm_reward_quality_protocol.md，理论适配边界见 dgm_edit_flows_adaptation_status.md；本轮冻结执行、paired trajectory 与 endpoint-ranker 终止判定见 dgm_p0_protocol.md、dgm_p1_panel_v1.md、dgm_reward_ranker_v2_protocol.md、dgm_reward_ranker_v2_report.md、dgm_reward_ranker_v3_large_protocol.md、dgm_reward_ranker_v3_large_report.md。旧版执行记录及勘误见 dgm_execution_report.md。

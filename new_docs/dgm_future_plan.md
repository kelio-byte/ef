# DGM 后续研究规划

> 状态：这是下一轮研究的执行计划，不是当前方法已经有效的声明。当前 action-level guidance 在 dev-1000 上未超过普通 Euler；确认集、最终验证集和完整测试集保持未使用。
> 核心原则：先验证“reward 是否代表正确答案”，再验证“DGM 是否能利用该 reward”，最后才扩大推理或使用保留评估集。

## 1. 当前起点与不可改变的事实

当前已知结论决定了后续顺序：

- 最新 E7 guidance 在 dev-1000 的 Top-1 为 56.7%，低于普通 Euler 的 58.2%；它不能进入确认集。
- E7 在独立 guidance validation 的组内排序准确率为 63.75%，高于 Bregman-only control 的 59.07%。这说明模型能学会当前 forward reward 所定义的相对偏好，但不说明该偏好等于逆合成正确性。
- 当前 forward reward 对正确候选有方向性，却不够干净：held-out 同 anchor AUC 为 0.7308，但 46.05% 的错误终点仍得到正 reward。
- 当前方法是受 DGM 启发的动作级近似，不是严格 fixed-coordinate DGM；可变长度插入/删除仍是理论边界。
- 已关闭的分支包括：当前 E7 的训练时长/β/checkpoint 扫描、P1/P2 reward 校准、已测试的 event-conditioned local credit。它们不能作为下一轮的补救性调参方向。

本计划的主问题只有一个：

> 能否先得到一个更接近“候选是否为正确逆合成答案”的 reward，再证明逐步 guidance 能在固定预算下利用它提高普通 Euler 的反应级 Top-1？

## 2. 数据、评估与停止规则

### 2.1 数据角色

| 数据 | 允许用途 | 禁止用途 |
| --- | --- | --- |
| guidance train-1000 | 构造新 reward 的训练样本、生成 shared-anchor guidance 数据 | 作为独立方法结论 |
| reward holdout-200（训练反应 1,000–1,199） | reward 的一次独立离线评估与同候选池 rerank gate | 选择模型结构、特征、阈值、训练轮数后反复重跑 |
| dev_unique1000_aug20 | 冻结后的普通 Euler 端到端方法筛选 | 训练 reward、选择 checkpoint、扫描 β 或 seed |
| confirm_unique1000_aug20 | dev 通过后的唯一独立确认 | 当前任何调参或可视化选例 |
| final_unique2000_aug20 与完整 src-test | confirm 通过后的最终验证 | 在 dev 失败时提前消耗 |

所有划分都以原始反应为单位；同一反应的 20 条 SMILES augmentation 必须在同一划分内。训练、评估和可视化均不得把 augmentation 当作独立反应。

### 2.2 统一通过门槛

除专门注明的离线诊断外，一个候选方法要进入下一层，必须同时满足：

1. 与同预算普通 Euler 相比，reaction-level Top-1 点估计不低；
2. Top-3、Top-10 或 Oracle 至少一项提高；
3. invalid 增加不超过 0.5 个百分点；
4. 候选数、采样步数、随机种子规则、SMILES 规范化和评分代码均已核验；
5. 采样耗时、奖励耗时和显存完整记录。

任何阶段失败后，不在已使用集合上通过 β、随机种子、训练中间 checkpoint 或分支数继续搜索。失败结果应保留为该假设的结论。

## 3. 主路线总览

| 优先级 | 任务 | 任务类别 | 依赖 | 成功后的去向 |
| --- | --- | --- | --- | --- |
| P0 | 冻结当前基线与研究协议 | 实验治理 / 可复现性 | 无 | P1、P2 |
| P1 | 改进前：Euler 与当前 guidance 的成对编辑路径诊断 | 可解释性 / 机制诊断 | P0 | 形成一个可证伪的 reward 或 credit 假设 |
| P2 | 构造并训练逆合成正确性 reward | 数据构造 / 监督模型 | P0；可吸收 P1 的机制发现 | P3 |
| P3 | 新 reward 的独立离线与终点 rerank gate | 离线评估 / 推理对照 | P2 | P4 |
| P4 | 用通过的 reward 重训 DGM；完成 dev 与改进后轨迹复核，再进行 confirm、final 验证 | DGM 训练 / 端到端评估 | P3 | 可选的 Euler-Beam/SMC 扩展 |

P1 可以与 P2 的数据准备并行，但 P1 的结果只能提出新的可检验假设，不能用来调当前 E7 的 β 或在 dev 中挑选“最好看”的模型。P4 中的新候选通过 dev 后，必须按同一协议再做一次成对轨迹复核；两次复核都只承担机制解释和实现审计，不承担调参。

## 4. P0：冻结当前基线与研究协议

**任务类别：实验治理 / 可复现性**

### 要回答的问题

下一轮的任何变化，能否明确归因于新 reward 或新 credit 假设，而不是候选预算、随机性或测试集泄漏？

### 固定内容

1. 将普通 Euler E1 设为唯一端到端基线：新 600k base checkpoint、100 个采样步、20 augmentation、每条 augmentation 3 个候选、batch 64、seed 42、固定聚合与评分逻辑。
2. 将当前五时间点 action-level guidance 设为历史对照，不再作为可继续调参的候选。其已报告结果、checkpoint、β=0.10 与逐位置归一化应保存为冻结 provenance。
3. 写明每个集合的角色、进入条件和禁止操作，特别是 reward holdout 只能用于一次正式 gate，confirm/final/test 不得提前读取。
4. 为每次后续运行记录：代码 revision、基础 checkpoint、reward checkpoint、输入文件哈希、种子、候选预算、生成与评分命令、输出哈希。

### 交付物

- 一份冻结配置清单；
- 一份数据划分和允许用途清单；
- 普通 Euler 基线预测及其 metadata 的可定位路径；
- 失败分支清单，包含“为什么不能继续扫参”。

### 完成与止损

完成标准是不同实验能够在不猜测参数的前提下复现相同基线。若发现基线、seed 或输入布局不一致，先修复记录或重新建立可比较基线；在此之前不开始 reward 或 DGM 新实验。

## 5. P1：改进前的 Euler 与 guidance 成对编辑路径诊断

**任务类别：可解释性 / 机制诊断 / 成对对照实验**

### 要回答的问题

当前 guidance 让 Top-1 下降时，究竟改变了什么？它是在早期过度提交、放大某类错误编辑、提高 invalid，还是把正确答案从第一名推到更深位置？

### 可视化的唯一实现

本计划中所说的 `visualization_trajectory`，专指仓库现有的 `scripts/visualize_trajectory.py` 所生成的逐编辑事件 HTML：它应保留每个编辑事件的状态序列，以及 `ORACLE`、`MODEL`、`ACTUAL` 表格。它不是只画最终 SMILES、汇总曲线或人工挑选的示意图。

当前脚本只接收一个基础模型 checkpoint 并调用普通 `sample_euler`，因此 P1 开始前必须在**这个脚本**中加入可复现的成对模式，而不另起一套不一致的可视化实现：

1. 支持加载 action-level guidance checkpoint、`β` 和 rate normalization，并以 `sample_euler(..., guidance_model=...)` 生成 guidance 轨迹；普通 Euler 与 guidance 的输入、步数、样本数和 seed 必须相同。
2. 对 guidance 路径，事件记录必须同时保存 guidance 前的基础动作分布、guidance 输出、`H^β` 乘数，以及 guidance 后实际用于采样的动作分布；不能把三者覆盖成一个字段。
3. HTML 必须按同一反应、augmentation、seed 将 Euler 与 guidance 放在同一个可定位的成对视图中；输出目录、checkpoint 路径、配置和运行命令写入 metadata，避免不同运行相互覆盖。
4. 为新增的成对模式和事件字段补充小型测试：常数 guidance 必须与 Euler 一致；`β=0` 必须不改变采样分布；HTML 中能定位两条路径及第一次分叉。

### 比较对象与范围

- 比较冻结的普通 Euler E1 与最新 E7 guidance；两者使用相同基础 checkpoint、产品输入、100 步、候选预算和固定随机种子。
- 只从已经使用过的 dev-1000 选例，不使用 confirm、final 或 test。
- 预先按结果分为四层：Euler 正确而 guidance 错误、guidance 正确而 Euler 错误、两者均正确、两者均错误。
- 在每层固定随机抽取最多 8 个原始反应，并对每个反应使用 3 个预先列出的 seed；若某层不足 8 个，则使用该层全部反应并如实记录不足，不能从保留集合补样本。

这会形成最多 96 对可比较运行。这里的“成对”是同一输入、augmentation 与初始随机键；一旦两条轨迹的状态不同，后续随机事件不再逐事件一一对应，必须记录首次分叉而不能假装它们仍是同一路径。将反应编号、augmentation 与三个 seed 固化为 `P1-panel-v1` 清单；该清单在 P4-D 中必须原样复用。

### 每条轨迹必须保存的信息

1. 每个发生编辑的 Euler 步的中间 token 序列和可读 SMILES（若可解析），以及相邻编辑事件之间的 no-op 步数；
2. 每个编辑事件的插入/替换/删除、位置、token、guidance 前基础动作概率或速率，以及实际采样后的动作概率或速率；
3. guidance 的正权重、乘数 `H^β`、归一化前后动作分布，以及首次与 Euler 选择不同的步骤；
4. 最终反应物的 canonical 表示、有效性、forward reward、与数据集 target 是否一致的离线标签；
5. 运行 provenance：原始反应编号、augmentation、seed、checkpoint、β、归一化方式和代码版本。

真实 target 只可用于不反馈到采样决策的 `ORACLE` 诊断表、结束后的注释和分层统计；它不可作为基础模型或 guidance 的输入，也不可用于选择某一条“漂亮路径”。

### 产出形式

- `scripts/visualize_trajectory.py` 生成的成对 HTML：每个样本并排保留 Euler 与 guidance 的完整编辑事件表、状态、第一次分叉和终点；
- 一张聚合图或表：编辑类型比例、首次分叉时间、no-op 比例、invalid 率、终点 forward reward 与真实正确率；
- 一份按四层汇总的机制判断，明确每个判断由多少反应和多少 seed 支持。

### 如何解释结果

支持“当前 reward/credit 错配”假设的模式包括：guidance 经常压低 Euler 的正确关键编辑；更早锁定到 forward-reward 高但真实错误的终点；或系统性增加无效编辑。支持“仍有可利用机制”假设的模式包括：guidance 在可重复案例中修复同一类漏插入/错误替换，且这些修复在真实标签上比 Euler 更常正确。

P1 不能证明方法性能有效，也不能授权修改 β、挑选 checkpoint 或进入 confirm。它只负责把“DGM 为什么失败”收敛为一个可由 P2 或未来 credit 实验检验的假设。

## 6. P2：构造并训练逆合成正确性 reward

**任务类别：数据构造 / 监督学习 / reward 建模**

### 要回答的问题

能否学习一个在推理时只看产物和候选反应物、却比 forward reconstruction reward 更接近“该候选是否为数据集正确逆合成答案”的分数？

### 标签定义

对训练 split 的每个原始反应，已知输入产物 P 与记录中的真实反应物 R*。构造候选 C 后：

- 正例：canonical(C) 与 canonical(R*) 相同；
- 负例：C 为冻结普通 Euler 产生的、有效且 canonical(C) 不同于 canonical(R*) 的候选；
- 若冻结 Euler 恰好生成 R*，该候选保留为正例，绝不误标为负例；
- 无效候选在推理时固定得到最低分；它们应单独统计，不能让分类器从无效字符串的偶然格式中获得虚假的优势。

这一定义优化的是基准数据集的 exact retrosynthesis correctness。它不会识别所有化学上可能成立、但与数据集记录路线不同的替代反应；这是任务目标与 forward reconstruction reward 的关键区别。

### 数据构造步骤

1. 使用现有 guidance train-1000 的原始反应，并补入每个反应的真实反应物 R*；从冻结普通 Euler 的候选中取去重后的 hard negatives。
2. 对产物和候选使用一致的 map-free canonicalization、分子组分排序和 tokenization。保存 candidate 来源、Euler seed、是否来自真实 target、raw forward rank/reward、RDKit 有效性和长度/组分统计。
3. 按原始反应拆分训练与内部模型验证，任何一个反应的 20 augmentation 和全部候选只能出现在一侧。
4. 将已有的 reward holdout-200（原始训练反应 1,000–1,199）保留为唯一的独立 reward gate；它不得参与特征选择、正则、阈值或训练轮数选择。

### 模型与对照

新模型记为 g(P, C)，输出 0 到 1 的正确性分数。模型输入只能包含推理时可得的信息：产物、候选反应物、以及由二者计算的标签无关特征，例如 raw forward beam rank、有效性、长度差和组分数。

至少保留两类对照：

- raw forward-beam reciprocal-rank reward；
- 一个预先定义的简单可审计基线，例如逻辑回归或低容量模型。

新模型可以使用产物—候选的联合编码器，但不得把真实 R*、目标匹配标签、dev/confirm/final/test 的任何信息作为输入特征。训练使用二元交叉熵或等价的正确性损失，并以原始反应或共享 anchor 组进行平衡采样，避免同一产物的大量候选主导训练。

### 必须完成的审计

- 标签、canonicalization 和 product index 映射的单元测试；
- 正负例数量、重复率、无效率、每个反应候选数和类别平衡报告；
- 训练/内部验证/reward holdout 的反应编号零重叠报告；
- 对同一个候选重复记录，预测分数一致的检查；
- 保存模型配置、训练数据哈希、特征列表和 checkpoint 选择规则。

### P2 与 P3 的联合 holdout 规则

P2 的 AUC 审计和 P3 的终点 rerank 必须作为**同一次、预先定义的 reward holdout 报告**执行：先冻结模型、特征、阈值和候选池，再一次性生成全部指标。查看 P2 的结果后不得改动任何内容再重跑 P3；P3 不是第二轮调参。

### P2 的通过条件

在不触碰 reward holdout 结果前冻结模型后，g 在该 holdout 的 AUC 部分必须同时满足：

1. 全局 correctness AUC 比 raw forward reward 至少高 0.02；
2. 同一 shared-anchor 组内的 correctness AUC 比 raw forward reward 至少高 0.02；
3. 不能以明显增加无效候选得高分为代价；
4. 结果按原始反应和共享状态组 bootstrap，报告区间。

若任何条件不满足，关闭该具体 reward 假设；仍保留同次报告中的 rerank 结果以便解释，但不重建 guidance 数据、不运行 dev，也不通过在 holdout 上加特征或调阈值来补救。

## 7. P3：新 reward 的独立终点 rerank gate

**任务类别：离线评估 / 终点推理对照 / 成本控制**

### 要回答的问题

在同一个候选池中，新 reward 能否至少不损害第一名，并真正改善最终候选排序？若不能，它不值得作为昂贵的逐步 guidance 监督。

### 实验设计

使用 P2 前已经冻结的 reward holdout-200 候选池。候选由冻结普通 Euler 预先生成；所有方法读取完全相同的、去重后的合法候选。该 rerank 与 P2 的 AUC 审计在同一次冻结报告中完成，期间不得改变模型或参数。比较：

- 原始模型输出顺序；
- raw forward-beam reward 重排；
- 新 reward g(P, C) 重排。

新 reward 在打分时不读取真实 R*；真实 target 只在所有候选和分数固定后，用于计算 Top-k、Oracle、AUC 与 paired bootstrap。

### 必须报告

1. Top-1、Top-3、Top-5、Top-10 与 Oracle；
2. 每个反应的 paired delta 及 95% bootstrap 区间；
3. invalid、有效候选数、去重候选数和因无效候选而被跳过的比例；
4. reward 推理耗时、缓存/去重率和显存；
5. 典型 rerank 成功与失败案例，但案例只能解释汇总结果，不能替代汇总结果。

### P3 的通过条件

相对 raw forward reward，在同一 holdout 候选池上：

- Top-1 点估计不低；
- Top-3、Top-10 或 Oracle 至少一项提高；
- 没有通过删除候选、改变候选预算或读取 target 获得收益；
- 得到的收益在反应级 paired bootstrap 下不与明显负向结果冲突。

若 P3 失败，停止该 reward 配方。此时不训练新的 DGM，因为“终点排序都无收益”的 reward 没有理由在中间步骤产生可靠收益。

## 8. P4：用通过的 reward 重训 DGM，并进行端到端验证

**任务类别：DGM 训练 / 端到端评估 / 分阶段泛化验证**

P4 只在 P3 通过后启动。其目的不是重新发明多个 guidance 变化，而是在保持当前已审计的 action-level 管线不变时，隔离“换成更好 reward”是否能修复端到端 Top-1。

### P4-A：重建 shared-anchor guidance 数据

1. 保持基础 Edit Flows checkpoint 冻结，使用五个 anchor step（10、30、50、70、90）和每个 anchor 的四条独立 continuation。
2. 保持 train-1000 / guidance validation-200 的反应级隔离、共同中间状态规则、canonicalization 和 provenance 字段。
3. 用通过 P3 的 g(P, C) 替换 raw forward reward，输出范围固定在 0 到 1；保存原始 forward rank/reward 作为诊断字段，不覆盖它们。
4. 重新运行数据完整性、组内 state/time 一致性、reward 非常数比例、无效率、去重率和数据哈希审计。

### P4-B：训练与 checkpoint 选择

第一轮只更换 reward，固定其余当前设置：独立 product-conditioned guidance 网络、冻结基础模型、batch 64、学习率 1e-4、Bregman 动作目标、pairwise 权重 0.25、calibration 权重 0.10、2,000 个优化步、seed 42。

同时训练 Bregman-only control 和完整 candidate。checkpoint 只能由 guidance validation 选择：

- control 选择最低 Bregman；
- candidate 必须满足 Bregman 不高于 control 最优值的 1.15 倍；
- 在满足保护条件的 checkpoint 中选择最高的同 anchor 组内排序准确率；
- 不查看 dev-1000 Top-k 结果选择 checkpoint。

若新 reward 的数值尺度或正负例结构导致该损失不稳定，应先在 train/validation 诊断中提出一个新的、预先写明的损失假设；不得直接在 dev 上扫描损失权重。

### P4-C：冻结普通 Euler 的 dev-1000 对照

在 P4-B 选出一个 checkpoint 后，使用与 E1 完全相同的端到端协议：

- 新 600k 基础 checkpoint；
- ordinary Euler、100 步、20 augmentation、每条 augmentation 3 个候选；
- batch 64、seed 42、β=0.10、逐位置归一化；
- 20 条 augmentation 聚合为一个原始反应；
- 输出按反应聚合，计算 paired bootstrap。

必须并排报告普通 Euler、raw forward rerank、当前 E7 历史对照和新 reward guidance 的 Top-1 至 Top-10、Oracle、invalid、候选数、耗时与显存。

新 guidance 只有同时满足第 2.2 节的统一通过门槛，才进入 P4-D 的改进后轨迹复核；复核完成且未发现实现或 provenance 错误后，才可以进入 confirm。若 dev 失败，则该“新 reward + 当前动作级 guidance”组合关闭，不在 dev 上继续调 β、seed、训练时长或 checkpoint。

### P4-D：改进后必须复做的成对编辑路径复核

**任务类别：可解释性 / 改进有效性诊断 / 实现审计**

这不是“改进前看一次、改进后省略”的可选展示。P4-C 的候选一旦通过数值门槛并冻结 checkpoint、`β`、归一化和采样协议，就必须再次用 `scripts/visualize_trajectory.py` 的成对模式比较 Euler 与**新 guidance**；除了已冻结的 guidance 配置外，不能再修改任何采样条件。

#### 要回答的问题

新 reward / 新 guidance 的端到端正向信号是否对应于预期的编辑机制？它是否减少了 P1 中观察到的错误分叉、错误早期提交或无效编辑，而不是只偶然改变最终候选顺序？

#### 固定的两组比较

1. **直接前后对照组：**原样复用 `P1-panel-v1` 的全部反应、augmentation 和 seed，依次查看 `Euler vs 当前 E7` 与 `Euler vs 新 guidance`。这样每个先前的失败或成功轨迹都能直接回答：新方法是否修复、保持或恶化了同一机制。
2. **新候选差异组：**用普通 Euler 与新 guidance 的已冻结 dev 输出重新形成同样四层（Euler 正确而 guidance 错误、反向、均正确、均错误），并按 P1 的固定随机规则抽取每层最多 8 个反应、每反应 3 个 seed。该组防止旧面板遗漏新候选特有的成功或失败模式。

两组均只能使用 dev-1000。不得为了做“更好看”的轨迹提前读取 confirm、final 或完整 test，也不得在看完 HTML 后替换 reaction、seed、checkpoint、`β` 或 reward 配方。

#### 必须比较和保存的证据

1. 两轮 HTML 均保留相同的逐编辑事件表；改进后页面还必须显示基础分布、guidance、`H^β` 与 guidance 后分布，因而可以检查权重究竟改变了哪一个动作。
2. 对 `P1-panel-v1` 逐例标记：P1 已识别的首次错误分叉是否消失、推迟、转为正确，或仍然存在；同时汇总 no-op、编辑类型、invalid、首次分叉和最终正确性的变化。
3. 对新候选差异组按四层汇总新出现的正向与负向模式，并将样本清单、HTML 路径、运行命令、输入哈希、基础/guidance checkpoint 和 seed 一同保存。

#### 判断与止损

该复核是**描述性机制证据，不是第二个调参门槛**：路径“看起来更合理”不能替代 P4-C 的 reaction-level Top-k 结果，也不能据此挑选配置。若它发现 checkpoint、样本、seed、`β`、归一化或事件字段与冻结协议不一致，则先修正实现/记录并按冻结配置重做 P4-C 和本复核；若未发现此类机械错误，是否进入 confirm 只由已冻结的 P4-C 数值门槛决定。无论图像支持或否定何种机制，结论都必须与两轮可比较 HTML 一起报告。

### P4-E：confirm、final 与完整 test

只有 P4-C 通过且 P4-D 完成、未发现实现或 provenance 错误，才冻结 reward、数据构造、网络、损失、checkpoint、β、归一化、候选预算、采样步数和随机种子列表。

1. 在 confirm_unique1000_aug20 上分别以 seed 42、43、44 运行；三个 seed 独立比较，不能合并为更大的候选池。
2. confirm 仍满足统一通过门槛后，才在 final_unique2000_aug20 上进行一次较大规模验证。
3. final 通过后，才允许一次完整 src-test 评估。完整 test 不用于发现参数或选择方法。

## 9. P4 之后才允许讨论的扩展

以下不是当前主线任务，只有 P4 在 confirm 和 final 上均通过后才可立项：

| 扩展 | 任务类别 | 需要先证明什么 | 后续实验原则 |
| --- | --- | --- | --- |
| guidance 与 Euler-Beam 结合 | 推理算法组合 | ordinary Euler guidance 已稳定有效 | 固定总模型前向次数和候选预算，与无 guidance 的 Euler-Beam 成对比较 |
| reward-twisted SMC | 概率推理方法 | 独立 reward 在终点排序和普通 Euler guidance 上均有效 | 明确 target、proposal、importance weight 与 ESS；不能把 target=proposal 机制测试当作性能证据 |
| 更可信的动作价值 | credit assignment 研究 | reward 本身可靠但 action-level guidance 仍失败 | 构造受控 rollout 或可验证局部价值目标；已失败的 event-conditioned terminal-reward 方案不重复扫描 |
| 更接近 exact DGM 的表示 | 理论 / 模型重构 | 有清晰的收益目标与可验证映射 | 设计带持久位置身份的坐标表示，先验证插入/删除映射，再讨论严格 DGM 表述 |

## 10. 当前禁止项

在 P0–P4 完成前，以下操作不在计划内：

- 延长当前 E7 训练、扫描 β、扫描 seed 或事后挑 checkpoint；
- 重开已拒绝的 P1/P2 reward 校准分支；
- 在当前 raw forward reward 上直接接入 Euler-Beam；
- 使用 confirm、final 或完整 test 选择任何方法、参数或案例；
- 将路径可视化的少数个例表述为 DGM 已提升任务性能。

## 11. 每个阶段的最小交付清单

每一个通过或失败的阶段都必须留下同样四类资产：

1. 可复现的配置、命令、输入哈希、checkpoint 和输出目录；
2. 反应级汇总指标、paired bootstrap 与效率指标；
3. 数据完整性和泄漏检查结果；
4. 一段明确结论：假设是否支持、下一步是否获准、哪些实验被禁止继续。

这样即使某个假设失败，失败也会缩小问题范围，而不是成为无法解释的额外实验记录。

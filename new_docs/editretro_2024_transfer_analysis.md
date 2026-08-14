# EditRetro（Nat Commun 2024）可迁移设计筛选

日期：2026-08-13  
范围：仅分析；未改模型代码、未运行训练或采样。

## 结论

最值得研究的不是把 EditRetro 的三阶段编辑器重写到 Edit Flows 中，而是借它解决我们已确认的三个问题：**采样分支缺少有用的多样性约束、模型未在自身偏离状态上受训、以及动态编辑状态缺少不可变产物条件**。以下按“先小而独立、再结构改造”的顺序排列。每项都要先在固定的 reaction-level `dev_unique1000_aug20` gate 上比较同预算普通 Euler；通过后才进入 confirm/final/src-test。

| 优先级 | 候选 | 主要目标 | 成本 | 结论 |
| --- | --- | --- | --- | --- |
| 1 | 首个实质编辑的谱系分层 | Top-K / Oracle、多样性 | 低—中 | 最适合先做的推理实验 |
| 2 | 模型状态 roll-in 的鲁棒训练 | Top-1、invalid、后期编辑 | 中 | 与当前 flow 最兼容的训练迁移 |
| 3 | 不可变产物 memory conditioning | Top-1、invalid、编辑正确性 | 中—高 | 最直接击中当前结构性瓶颈 |
| 4 | SPE 片段词表（先可行性 gate） | Top-K、invalid、采样效率 | 高 | 有最强 tokenizer ablation，但不应直接开工 |

## 阅读与对照边界

仓库 `PDF/` 中没有用户指定的 `2024-Nat Commun-Retrosynthesis prediction with an iterative string editing model` 原 PDF；工作区及共享目录的 PDF 检索也未找到它。因此本文以开放获取的论文正文与 Supplementary Information 为准：[论文页面](https://www.nature.com/articles/s41467-024-50617-1)、[补充材料 PDF](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-024-50617-1/MediaObjects/41467_2024_50617_MOESM1_ESM.pdf)。这不影响机制判断，但不能表述为“已读取仓库新增 PDF”。

项目侧只先阅读了现状文档，而没有遍历代码库：[`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md)、[`euler_beam_current_situation.md`](euler_beam_current_situation.md)、[`dgm_evaluation_v2.md`](dgm_evaluation_v2.md)、[`dgm_future_plan.md`](dgm_future_plan.md)。关键前提是：Edit Flows 从 product SMILES 经 100-step variable-length Euler 编辑到 reactants；当前已有 20x SMILES augmentation 与 `legacy_best_rank` 聚合；R9K1M2 的收益主要是排序集中而非 Oracle；DGM/forward-reward rerank 已在 reaction-level dev 上失败；训练状态没有不可变显式 product 条件，目标编辑约 91.2% 是插入、删除仅约 0.9%。

## 1. 首个实质编辑的谱系分层（EditRetro 的 reposition sampling）

**论文怎么做。** EditRetro 先以 greedy reposition 产生一个基线，然后只在**首轮 refinement** 对一部分 reposition token 采样；后续轮次保持 greedy。每个保留的 reposition 产生 `k_t` 个 token-insertion 候选，候选先局部按 token probability 排名，再跨 SMILES augmentation 用 reciprocal-rank 聚合。其意图是把有限候选预算投入不同的反应中心/fragment 修改，而不是在完整序列解码过程中盲目扩束。Supplementary Fig. 1 给出了该候选生成与局部—全局排序流程；正文展示其候选在 USPTO-FULL 上具有实质差异的结构多样性（Supplementary Fig. 2），但**没有隔离“reposition sampling 单独带来多少 Top-k 增益”的 ablation**。

**为何与我们兼容。** 我们的 R9 已证明“保护独立谱系”优于过早全局竞争，但每条谱系内部 M2 的选择没有确保不同谱系真的探索不同的编辑假设；更多 K/M 又已被证实不是答案。因而可迁移的是“让有限分支在决定性编辑处去相关”的预算分配思想，而不是 EditRetro 的 reposition decoder。

**映射与最小实现。** 在 `sample_euler_beam` 的 R9 分支层引入一次、预先固定的 *first-state-change signature*：每条谱系第一次离开 product 状态时，记录编辑类型、位置（相对 product / GAP）和 token（插入可加局部邻域）。同一 augmentation 内，优先保留不同 signature 的 R 个谱系；后续 99 步仍完全使用现有 R9K1M2，不引入 reward、全局 K 或额外模型前向。先做一个只记录 signature/碰撞率/正确候选覆盖的诊断；只有确认明显的 action-family collapse，再做分层选择 A/B。

**预期收益。** 同等 9 输出、同等 100 forward-step 预算下，主要改善 Oracle、Top-3/10 与不同 reactive-site 覆盖；Top-1 只可能小幅提升，不能承诺。它尤其适合当前“R9 有用但 Oracle 仍有限、无 future value”的状态。

**成本与风险。** 低—中：仅修改采样期候选保留与 trace/统计，且能单独 A/B。最大风险是 Edit Flows 的一次 transition 可含多个位置编辑，signature 未必等同化学反应中心；并且项目轨迹有大量后期编辑，不能机械使用论文的“第一轮”。所以锚点必须是**第一次数值上非 no-op 的状态变化**，不是固定 Euler 时间步。若强行覆盖低概率 signature，Top-1 可能下降；gate 必须要求 Top-1 不低于普通 R9，且 Top-3/10 或 Oracle 至少一项提高。

**证据强度。** 机制和多样性现象有论文支持，但无独立消融；项目内 R9 隔离谱系的受控结果构成更直接的迁移前证据。故它排第一是因为实验便宜、可证伪，不是因为论文已证明该细节必然有效。

**当前实现状态（2026-08-13）。** 已在 `sample_euler_beam` 中实现为 opt-in 策略，命令行开关为 `--euler_beam_first_edit_diversity`。对于正式 R9K1M2 的 `n_runs=9, n_branches=1` 布局，它在同一 augmentation 的 9 个独立 run 之间分配不同首步；对于 `n_branches>1`，它在单个 beam pool 内分配。它只在首次真实状态变化时提取少量 action summary（不搬运完整 action tensor），按 `(操作类型, 位置, token)` 的首个编辑 signature 保留不同路线；最终输出槽位和 `n_children` 不变。默认关闭，因此历史 Euler/R9 基线仍可复现。建议与 `--euler_beam_share_identical_forwards` 一起做效率对照，并读取 metadata 中的 `first_edit_signature_*` 统计。

## 2. 模型状态 roll-in：用可恢复的偏离状态训练，而不是照搬随机字符串扰动

**论文怎么做。** EditRetro 对三个 decoder 使用 imitation-learning roll-in（Supplementary Fig. 6、Algorithm 1）：reposition decoder 有概率以 token decoder 的预测结果为输入；placeholder/token decoder 一部分使用 expert 的 reposition 输出，另一部分把目标 reactant 经随机删除和 shuffle 变成 noisy state。这样每个模块会在上游实际会给出的、而非只在 oracle state 中出现的输入上学习纠错。论文还以 masked reactant reconstruction 预训练 token decoder，并进行三轮自蒸馏以缓解 one-to-many target；但补充材料没有给出 roll-in 或 self-distillation 的独立数值 ablation。

**为何与我们兼容。** 当前训练可从 coupling 采样合法的 `x_t`，但这些 state 不一定包含模型采样时的系统性偏离；当前推理错误正是局部编辑累积后的分布外问题。与 DGM 不同，这项训练仍以真实反应物为监督，不依赖已失败的 forward reward，也不要求 variable-length state 有固定坐标对应。

**映射与最小实现。** 不迁移论文的字符 shuffle（它很容易破坏 SMILES 语法，且与连续时间率模型不等价）。先在现有训练 state 上做受控 roll-in：以低固定概率，从同一 product 的 oracle/coupled `x_t` 用冻结或 EMA base model 展开 1--3 个 Euler step，得到 `x'_t`；仅当 `x'_t` 可 tokenize、长度受限且与目标仍可构造监督时，用原 Bregman/edit loss 训练模型从该 state 回到同一 target。保留一半原始 coupling batch，并分桶报告 clean-vs-roll-in loss、edit type、invalid 与时间区间。第一轮只做短训练/held-out likelihood 与 one fixed dev gate，不扫 roll-in 深度或概率。

**预期收益。** 更稳的后期修复、较低 invalid，以及 Top-1 的小幅改善；对 insertion-heavy 模型可能主要减少“已偏离后继续错插”的级联错误。它不直接增加候选数，因此比扩大 K/M 更可能提升单轨迹质量。

**成本与风险。** 中等：训练数据和 batch 逻辑需要增加短采样，但模型结构、tokenizer、评价协议不变。核心风险是 on-policy state 的时间标签/目标 coupling 若定义不严谨，会破坏 Bregman 目标；另一个风险是模型 early checkpoint 产生过差 state，训练变成噪声。应先冻结 teacher、限制 1 step、只在同一 t-bin 内比较，并保留 clean-only 对照。若 invalid 或 clean validation 恶化即停止。

**证据强度。** 论文给出了完整 algorithm 和训练结构，但没有该项单独 ablation；支持属于“强机制证据、弱因果数字证据”。它仍优先于 self-distillation，因为后者需额外 forward-model 过滤，且项目已显示 forward reward 与 exact-match Top-1 存在失配。

## 3. 不可变 product memory conditioning（只借 encoder--decoder 分工）

**论文怎么做。** EditRetro 在每次 refinement 都将原始 product 输入 encoder，三个 decoder 对 encoder memory cross-attend；被迭代修改的是 decoder 的 current sequence，而非 product memory。reposition 显式保留/删除/重排 product token，之后再插入 placeholder 与 fragment token。这使“反应物当前草稿”可变而“目标产物证据”不可变。

**为何与我们兼容。** 这正对齐项目已审计出的首要基础模型缺口：当前演化 state 中不存在不可变显式产物条件，模型能够把本应依据的 product 信息删除或替换。该问题是模型表示瓶颈，不是 DGM 的 reward/credit 问题，因此不会重复已关闭的 guidance 路线。

**映射与最小实现。** 维持当前动态 `x_t`、GAP、ins/sub/del rate heads 与 Euler sampler；新增一个 product encoder memory，并让状态 Transformer 层以 cross-attention（或较低成本的 product-token cross-attention adapter）读取它，同时加入 time embedding。产品须以与当前 augmentation 同一条 SMILES 表示输入，避免表示不一致。第一版只训练新 adapter/condition projection 和输出层，必要时再全量微调；不要先重写成 EditRetro 的 reposition-placeholder-token 三 decoder。

**预期收益。** 理论上改善删除/替换的定位、reactant fragment 的 product-grounded copy，以及无效/漂移；这可能是四项中 Top-1 上限最大的，但需要重新训练才能验证。论文的 end-to-end 结果（USPTO-50K class-unknown Top-1 60.8%）支持整体架构强，但**没有“去掉 immutable product memory”的单独 ablation**，不能把其绝对数值外推到本项目。

**成本与风险。** 中—高：涉及 model interface、checkpoint 不兼容与重新训练；同时编码 product 与 state 会增加每个 Euler step 计算，必须缓存 product memory，不能每步重算。还有 product 与 current state 的 positional alignment 并非固定，cross-attention 可能学复制而压制必要的逆向修改。先以 adapter + frozen base 的小规模 gate 验证，再决定是否全量训练。

**证据强度。** 对项目缺口的直接性很强，但论文缺少单因子 ablation。因此它应是明确的结构研究候选，而非下一个低风险 patch。

## 4. SPE 片段 tokenizer：只做数据可行性与 edit-distance gate

**论文怎么做。** EditRetro 用 ChEMBL 预训练的 SMILES Pair Encoding（SPE）将常见子串合并为 fragment token，并按 R-SMILES 对齐 product/reactant。Supplementary Fig. 5 是最明确的单项 ablation：同为从头训练、USPTO-50K、10x augmentation，SPE 相比 atom-wise tokenization 将 Top-1 从 **52.4% 提至 57.3%**，Top-10 从 **74.3% 提至 82.2%**。补充说明将高 validity 部分归因于 fragment-level edit distance 低（训练集平均 5.4）以及该 tokenizer；其 Top-10 validity 为 99.99%。论文也明确承认 SPE fragment 不一定对应化学子结构，是局限而非保证。

**为何可能适用、又为何不能直接照搬。** 我们同样是 string editing，且插入占 91.2%；若 fragment token 缩短 product→reactant 的有效编辑距离，可能减少插入决策、缩短序列、提高有效性并降低 100-step 采样中的无效累积。可是 Edit Flows 的 variable-length coupling、GAP 对齐、rate vocabulary、max-length 和现有 checkpoint 都依赖当前 atom-wise token，因此这不是换 tokenizer 的局部替换；论文的增益也来自其三阶段编辑器，不能假定能在 flow 中复现。

**映射与最小实现。** 先不训练模型。只在训练/验证对上构建可逆 SPE 的 shadow preprocessing，并报告：长度分布、对齐失败率、产品到反应物的 insertion/substitution/deletion 比例、超长比例、片段跨 ring/branch 的错误率，以及 fragment 覆盖率。只有同时满足“对齐成功率不低于当前、median 有效编辑数显著下降、无灾难性 OOV/超长”才值得建立单独 tokenizer/vocab/coupling 训练分支。SPE 训练必须是全新 checkpoint，对照保持相同数据 split、augmentation、100-step/候选预算和 reaction-level 聚合。

**预期收益。** 若 shadow gate 成立，潜在提升包括 Top-K、invalid 和每步 token 决策难度；但推理 wall-clock 未必下降（词表更大、softmax 更贵）。它不是当前完整 src-test 前应插入的工作。

**成本与风险。** 高：数据预处理、对齐、vocab、训练、checkpoint 全部重建；片段 token 可能跨越反应中心，反而让细粒度 edit rate 更难学。由于项目的 deletion 稀少，若 SPE 只缩短复制部分而不降低关键插入不确定性，收益也可能消失。

**证据强度。** 四项中唯一有清晰 tokenizer 单因子数字 ablation，但跨架构迁移风险也最高；因此只值得先做 cheap feasibility gate，排在最后。

## 不建议迁移的做法

1. **SMILES augmentation + reciprocal-rank/global fusion：不做。** EditRetro 使用 test-time SMILES augmentation 和 local-rank 再 reciprocal-rank 聚合；项目已经有 20 augmentation 和 `legacy_best_rank`，重复实现不构成新方法。
2. **冻结 forward model 的 rerank/self-distillation filter：不做。** EditRetro 用 golden forward model 与序列相似度接受 self-distilled training targets。我们的 forward reward 虽有 AUC 信号，但 terminal rerank、calibration、action guidance 均使 dev Top-1 下降，不能把它作为“高质量 teacher”直接扩大使用。
3. **按“两次状态相同”提前停止 100-step Euler：不做。** EditRetro 正确 Top-1 中 80.18% 在一次 refinement 得到结果，且延迟为 177.4ms 对 292.1ms（与 R-SMILES 的单样本比较）。但我们的轨迹有显著晚期编辑，固定 100-step CTMC 的 state repeat 也不等同 EditRetro 的 deterministic refinement 收敛；早停很可能截断必要编辑。它最多是未来独立的 trace diagnostic，而非性能候选。
4. **完整的 reposition/placeholder/token 三 decoder 重构：不做。** 这是论文的主体，不是“小改动”；它与现有连续时间、GAP 和 Bregman rate parameterization 机制不同，会同时改变基础模型、训练目标和采样器，无法归因，也不符合当前优先级。

## 建议的验证顺序

1. 先做候选 1 的只读 trace 诊断，冻结 signature 定义与同预算 A/B；这是唯一可在不训练基础模型下获得直接答案的项。
2. 若候选 1 未显示 action-family collapse，停止该线；随后做候选 2 的 1-step roll-in 小训练 gate。
3. 若 roll-in 只改善 invalid 或无收益，不继续调概率/深度，转入候选 3 的 cached product-memory adapter 原型。
4. 候选 4 仅做 preprocessing feasibility report；在它通过前不建设训练分支。

这些候选均不应绕开既有 stop rule：任何 dev gate 失败，不消耗 confirm、final 或完整 `src-test`。完整 src-test 应只用于已通过独立 confirm 的冻结方法，或用户已明确要求的纯基线复跑。

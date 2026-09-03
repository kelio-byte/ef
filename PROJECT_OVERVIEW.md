# PROJECT_OVERVIEW：Edit Flows 项目全局总结

> 生成日期：2026-08-12。本文以 `PROJECT_FACTS_SUMMARY.md` 为主要信息入口，并对关键数字与状态回查了 `new_docs/` 原始文档。它是一份“项目级综合”，包含事实梳理与综合判断两类内容；两者用标记区分：普通文本=文档支持的实验事实；【判断】=基于这些事实的分析性判断，不代表文档中已声明的结论。
>
> 证据边界：所有性能数字均来自 tiny / validation / test-mini / dev 等中间集合；`confirm_unique1000_aug20`、`final_unique2000_aug20` 与完整 `src-test` 均未被使用。

---

# 1. 项目概述

**任务**：单步逆合成——给定产物 SMILES，生成反应物 SMILES。模型是 Edit Flows（离散、变长、连续时间编辑流）：每个时间步预测每个位置的 insert/substitute/delete rate 与 token 分布，采样器用 100 个 Euler 步逐步把产物编辑成候选反应物。评估以原始反应为单位（每反应 20 条 SMILES augmentation 聚合），核心指标是 Top-1～10、Oracle 覆盖率、invalid 率与耗时。

**三条研究主线及当前状态**：

| 主线 | 做什么 | 当前状态 |
|---|---|---|
| 采样/搜索工程 | 从单轨迹 Euler 到多分支 Euler-Beam，再到受控的 R/K/M 研究 | ✅ 形成成熟配置 R9K1M2；相对普通 Euler 的收益只有方向性证据，尚未在更大集合上确认 |
| 训练可信度 | 审计旧训练、修复基础设施、重训 600k 模型 | ✅ 基础设施可信、可复现；新模型与旧模型性能几乎持平 |
| 偏好注入（reward/guidance） | 用正向模型 reward 训练逐步 guidance，尝试 SMC/DGM 式推理 | ❌ 全部候选未通过反应级 Top-1 门槛；管线与评估协议已建立，方向已收敛到“信号与 credit”问题 |

**一句话现状**：项目把“如何生成更多候选”推进到了“如何为候选和中间状态定义正确信号”，并在这个过程中建立了一套严格的反应级评估与预注册门槛；目前采样端有一个尚未被严格确认的默认配置，引导端则被一个重复出现的负结果（离线指标改善、反应级 Top-1 下降）挡住。

---

# 2. 核心研究逻辑

项目的发展不是“换了几种方法”，而是一条证据驱动的因果链：

1. **起点：模型单次采样不可靠。** 20,000 次独立采样中 per-sample 命中率只有 26.3%；同一产物的不同 SMILES 写法命中率可差一个数量级；编辑事件 46% 挤在最后 20% 时间。因此“正确候选是否被生成”（覆盖率）与“正确候选能否排到第一”（排序）是两个不同的问题。

2. **第一步尝试：用更多轨迹和分支解决“运气”。** 从单条 Euler 到 K 分支 + M 后继 + 状态合并（Euler-Beam）。实验迅速否定了“更宽的 beam 更好”的直觉：全局大 K 池会提前淘汰低当前分数但未来正确的谱系；真正有效的是**保护相互隔离的谱系（R9）+ 每步少量局部二选一（M2）**。

3. **同时发现：启发式排序不是未来价值。** 完整单路径概率偏向“不编辑”；旧的激进评分（triggered-only + reverse）只是偏爱低概率多编辑路径。项目因此意识到：分支竞争需要一个“哪个中间状态更可能通向正确终点”的分数，而当时没有。

4. **第二步尝试：先确认基础模型本身可信。** 在引入任何新模型前完成训练审计：主 loss 与论文一致、数据未污染，但训练基础设施存在 Noam 首步错误、不可复现、无验证等问题。修复并重训后新旧模型几乎持平——这说明当时的问题不在“训练坏了”，而在模型结构或推理目标。

5. **第三步尝试：从外部信号构造“好候选”的偏好。** 引入冻结 Molecular Transformer 的正向重构 reward，训练随中间状态变化的 action-level guidance。数据与训练目标经过多轮修正（单终点 → 多终点 → shared-anchor → 五时间点；Bregman → pairwise → calibration），每一步都确实改善了**离线指标**（AUC、组内排序准确率），但所有候选在 1,000 反应开发集上的 Top-1 都低于普通 Euler 基线。

6. **第四步尝试：把失败定位到 credit assignment。** 既然 guidance 学会了“按 reward 排序”，却不能改善 Top-1，下一假设是“终点 reward 被错误地分配给了整条路径的编辑”。用真实一步后继（L1）失败于数据稀疏；改用 event-conditioned 原子编辑（E1）解决了数据可用性，但训练目标（E2）仍未通过；post-gate 的开发集诊断明确为负向。

7. **第五步尝试：把终点排序本身做成 gate。** 先后训练小型 endpoint ranker（v2、v3）与 reward 校准器（P1/P2），并在修复 target leakage、扩大 holdout 到 1,000 反应后确认：**candidate 级 AUC 提高与 reaction 级 Top-1 下降可以稳定共存**。这条“再加一个小 reranker”的路线被正式关闭。

8. **当前收敛点：** 瓶颈不在训练步数、数据量或 beam 宽度，而在三个相互关联的环节：reward 与真实逆合成正确性的失配；终点 reward 到单步动作的信用分配；以及 action-level guidance 的表达能力（固定位置总编辑率、非 exact 的坐标映射）。文档同时指出一个模型级方向——显式 product conditioning——尚未执行。

【判断】这条因果链最重要的方法论含义是：项目已经用“gate 化实验”把大量候选方案在低层指标处提前淘汰；当前剩下的选择不是在已失败空间里继续扫描参数，而是在三个被证据指向的环节上换定义（更好的 reward、可验证的局部价值、更强的模型条件）。

---

# 3. 方法演进

## 3.1 时间线总览

| 时间 | 阶段 | 关键产出 |
|---|---|---|
| 2026-07-28 | 基线诊断 | 可视化、20-aug×10-run 稳定性；确认覆盖率/排序分离、augmentation 敏感性、DELETE 弱点 |
| 2026-07-31 ～ 08-01 | Euler-Beam 建立 | seed/RNG 修复、完整路径概率、K×M、状态质量合并、TF32/批量化 |
| 2026-08-01 ～ 08-04 | 结构消融 | K=5 无增益、M=3/4 回退、no-op anchor；R3K3 vs R1K9；NNN+LL 混合候选池 |
| 2026-08-03 ～ 08-07 | 训练审计与重训 | 修复 Noam/RNG/validation；`retro_v2.yaml`；新 600k checkpoint；validation-200 冻结参数、mini-1001 确认 |
| 2026-08-04 ～ 08-05 | 前沿方法研究 | Q sharpening（无稳定收益）；Euler-SMC mechanics（机制通过，无 reward） |
| 2026-08-07 ～ 08-08 | DGM 管线建立 | forward-beam reward、guidance 数据与训练、ordinary Euler 接入、validation-A/B 探索 |
| 2026-08-08 | pairwise 修正 | 发现 shared-anchor 数据不共享、修正 continuation；P5d/P5f 联合 gate 未过 |
| 2026-08-11 | 评估协议 v2 与全面复核 | dev/confirm/final 三层协议；E1–E7 全部未过 dev；reward 校准 P1/P2、ranker v2/v3、local credit L1/E2 全部关闭 |
| 2026-08-12 | 事实层整理 | PROJECT_FACTS_SUMMARY.md；本项目综合 |

## 3.2 采样弧：从“多候选”到“受保护的谱系”

- 初始假设：多个分支/更多 child/更大 K 能提高覆盖。实验结果逐条修正：
  - 更大 M（3/4）增加 unique 候选却全面回退准确率 → 候选数量不是目标；
  - 全局单池 K9/K10 更快但 Top-2/Oracle 明显更差，且 rank2/3 invalid 飙升 → 跨谱系竞争会过早灭绝；
  - R9K1M2（九个互相隔离的谱系，每步两个 child 中选一个）在 mini-1001 上成为准确率最优。
- 关键机制结论：M2 的收益是**排序集中化**（Oracle 几乎不变，Top-3/5 显著提升）；R 的收益是**覆盖保护**（Oracle 显著提升）。
- 性能收口：TF32 约快 16–24%（Top-k 不变），batch64 为复现默认，相同状态 forward 共享为 opt-in（约 -26% 时间，TF32 下有 2/9000 行漂移）。

## 3.3 训练弧：从“可跑”到“可复现”

- 审计确认：主训练目标（Bregman loss + Z-space alignment）与论文一致；旧 checkpoint 未被数据缺陷污染。
- 修复：Noam 首步 lr 顺序、确定性 seed/RNG/resume、数据行数 fail-fast、PAD alignment、validation 与最佳 checkpoint。
- 重训：A6000 完成新的 600k checkpoint（结构/loss 不变）。mini-1001 上新旧几乎持平（Top-1 58.2 vs 57.0；Oracle 91.5 vs 91.8）。
- 记录偏差：训练文档（08-06）计划“先 10k–30k pilot，通过后再完整 600k 重训”，但随后直接完成了 600k 重训，文档未记录 pilot 结果或跳过理由。
- 遗留模型限制：无显式不可变 product 条件；目标编辑分布极不均衡（insert 91.2% / substitute 7.9% / delete 0.9%）。

## 3.4 偏好弧：从“终点分数”到“逐步引导”，再被 gate 关闭

- reward 选择：teacher-forced likelihood（AUC 0.564，弃用）→ forward-beam=5 重构（AUC 0.69–0.71）。
- 数据语义四轮修正：单终点（无组内对比）→ 每 product 4 终点（46.5% 组有 reward 变化）→ 真正 shared-anchor continuation（修正时间计算与原地改写 bug）→ 五时间点（step 10/30/50/70/90）。
- 训练目标：Bregman action loss → 增加 shared-anchor pairwise（rank 提升但校准破坏）→ 增加 score calibration（校准部分恢复但联合 gate 未过）。
- 推理形式：action-level 近似（`u=λ·Q·H^β`，逐位置保持总编辑率；β=0.10）；`per_position` 优于 `per_sample`；每步多一次 guidance forward，总时间约 +47–49%。
- 评估治理：旧 validation-A/B 因多次选参与统计不稳被降级；新协议以原始反应为统计单位，dev→confirm→final→完整 test 四层隔离。
- 关闭分支：E3–E7、P1/P2、ranker v2/v3、L1/E2 均按预注册规则正式关闭。

【判断】三条弧不是并行偶遇，而是同一问题的三次逼近：先扩大搜索（采样弧），再确认底层可信（训练弧），最后尝试注入外部知识（偏好弧）。每次逼近都留下“哪些变量被排除了”的明确记录，这是项目当前最大的资产。

---

# 4. 关键实验发现

本节按“实验回答了什么问题”组织，不按文档顺序。

## 4.1 单条轨迹为什么不可靠？（早期诊断）

- 事实：20,000 次独立采样 per-sample 命中率 26.3%，轮间稳定（25.0–27.6%）；9/10 产物在 200 次机会下 Top-1≈100%，产物 1143 仅约 70%。编辑事件 46% 集中在 t>0.8；事件判定不查 token（可 4/4 “correct” 却 MISMATCH）；纯 DELETE 操作几乎全败；augmentation 命中率 CV 最高 161%。
- 证明了什么：覆盖率与 Top-1 是不同指标；“多采样 + 多 augmentation”可以把很低的单次命中放大成高 Top-1。
- 没证明什么：任何采样器/模型优劣。这些实验只提出“编辑时间、token 选择、表示敏感性”三个假设。

## 4.2 分支搜索的有效形态是什么？（Euler-Beam 消融）

- 事实（mini-1001，旧 checkpoint，固定 100 步/seed42）：
  - R1K10M2：Top-1/10/Oracle = 52.5/81.6/87.2%，wall 1,687 s；
  - R3K3M2：55.1/84.5/89.3%，wall 2,072 s；
  - R9K1M2：57.0/86.1/91.8%，wall 3,060 s；
  - R1K9M2（同 9 条初始流并入全局池）：54.1/83.5/87.0%，wall 1,397 s；
  - R10K1M2：56.9/86.0/92.4%，wall 3,426 s。
- 配对统计：R9 相对 R3 的 Top-2/3/5 与 Oracle 显著（p<0.005）；R9 相对 R1 在所有主指标显著；R10M2 相对 R9M2 只新增 6 个 Oracle（p=0.031）但 Top-1/10 各少 1 命中。
- 证明了什么：在当前质量函数下，“保护谱系 + 局部二选一”优于“全局竞争 + 更多分支”；M2 改善排序、R 改善覆盖；更多 M/R/K 并不单调有益。
- 没证明什么：这些分数不是严格概率/SMC 权重；`changed_state_bonus` 是启发式；R9K1M2 是否优于普通 Euler 只有方向证据。

## 4.3 R9K1M2 相对普通 Euler 的真实收益有多大？

- 事实（test-mini 前 200 反应）：Euler N9 = 54.0/69.5/77.5/82.5/88.0/94.5（Top-1/2/3/5/10/Oracle）；R9K1M2 = 58.0/74.0/81.0/86.0/88.0/96.0。逐反应 McNemar：Top-1 p=0.096、Top-2 p=0.078、Top-3 p=0.210；Top-10 持平、Oracle 仅 +1.5pp。
- 证明了什么：R9 有“把正确候选前移”的正向方向；收益集中在 Top-1～5 排序。
- 没证明什么：200 反应不足以确认全面优越；且当时普通 Euler 的 CLI seed 尚未接入（后已修复），该对照不是最终可复现基线。

## 4.4 训练修复后模型变好了吗？

- 事实：新旧 checkpoint 在 mini-1001 上 Top-1 58.2 vs 57.0、Oracle 91.5 vs 91.8；invalid 12.8 vs 13.7%；差异 ≤1.2pp。
- 证明了什么：修复后的训练流程可复现、无退化；可作为后续基线。
- 没证明什么：重训没有带来能力跃升；模型结构限制（copy-product、编辑不均衡）仍然存在。

## 4.5 外部 reward 有信息吗？

- 事实：forward-beam=5 重构对正确/错误候选的 AUC 约 0.69（validation-B）到 0.71（holdout）；同 anchor 组内 AUC 0.68–0.73；但 46–50% 的错误候选仍获正 reward；正向模型自身 Hit@5 约 75–79%。
- 证明了什么：reward 有弱但可测的方向，可用于终端重排的深层排序（validation-B Top-3/10 小幅上升）。
- 没证明什么：它不足以作为正确性真值；在 dev-1000 上终端重排本身 Top-1 也从 58.2 降到 57.6。

## 4.6 learned guidance 到底学到了什么？

- 事实（dev_unique1000_aug20，普通 Euler 协议）：
  - E1 基线：58.2/75.5/83.5/86.6（Top-1/3/10/Oracle）；
  - E5 共享 anchor guidance：56.2/76.1/84.9/87.5（Top-1 CI [-3.9,-0.2]）；
  - E6 五时间点：56.4/76.3/84.2/87.7；
  - E7 长训练：56.7/75.9/83.8/86.6（Top-1 CI [-3.3,+0.3]；深层 CI 均跨 0）。
  - E7 训练级：held-out pair accuracy 63.75%（control 59.07%），Bregman 在 1.15× guard 内。
- 证明了什么：guidance 能学会“按当前 forward reward 排序动作”，且这个能力随训练延长真实提高；但没有转化为逆合成 Top-1，时间成本约 +48%。
- 没证明什么：reward 与正确性的一致性、终点 reward 到动作的 credit 正确性。E5 的 Top-10 正信号（+1.4pp，CI 不含 0）说明“排序方向”确实被改变了，但代价是第一名下降。

## 4.7 局部 credit 能否替代终点对齐？

- 事实：自然单步 Euler 数据稀疏（仅 5.1% 的步有编辑，严格可区分组 0.4%）；event-conditioned 原子编辑把可区分组提高到 39.2%；但 E2 的 transition-target 对照在共同评价上未过门槛（Bregman 差 -0.198%，门槛 -2%；pair acc 低 1.64pp；Pearson 低 0.007）；post-gate dev 诊断 Top-1 55.7%（CI [-4.5,-0.6]）。
- 证明了什么：数据可用性不是唯一障碍；把同一个终点 reward 赋给“第一步原子编辑”并不比赋给整条路径更好。
- 没证明什么：所有可能的局部价值估计都无效——文档只否定了“终点 reward 直接复制到 event-conditioned 第一步编辑”这一种方案。

## 4.8 endpoint reranker 能否成为 gateway？

- 事实（v3：8000/1000/1000 反应）：raw forward Top-1 47.7%；residual 38.1%（CI [-13.0,-6.3]）；listwise 37.0%（CI [-13.9,-7.5]）；Oracle 三者均为 80.4%。residual 全局 AUC 0.6922→0.7167。
- 证明了什么：“candidate AUC 提高、reaction Top-1 下降”不是小样本噪声；小型线性/特征 ranker 不能作为通往 DGM 的 gate。
- 没证明什么：更强的 reranker（联合编码器、更好的组损失）一定无效——文档明确限定“当前小型 reranker 设计”。

## 4.9 形式化边界在哪里？

- 事实：validation 预对齐数据中 89.735% 的 insert 位于连续 GAP run，无法唯一反演到 X-space 动作；仅 7.674% 的整行只改变一个坐标。
- 结论：任何“exact DGM 已实现”的说法都不成立；当前是 action-level approximate guidance。固定坐标 toy 证明 DGM 代数本身正确，阻塞在状态表示。

---

# 5. 当前最可靠的结论

以下结论有文档实验支持（除标注外均为中间集合结论）：

1. **评估协议已成熟**：反应为统计单位、augmentation 不独立、dev/confirm/final/test 分层、paired bootstrap；confirm/final/test 未被使用。
2. **训练基线可信**：主 loss 与论文一致；旧 checkpoint 未被数据缺陷污染；新训练流程可复现；新旧模型性能几乎持平。
3. **Euler-Beam 的有效部分明确**：隔离谱系（R）与局部二选一（M2）有受控证据；全局大 K、更大 M、更大 R 均无普适收益。
4. **R9K1M2 相对普通 Euler 的收益尚未确认**：200 反应上方向为正但不显著；需要更大、seed 修复后的严格对照。
5. **forward reward 是弱信号**：有方向（AUC≈0.69–0.73），但不是正确性真值（约一半错误候选得正分）。
6. **guidance 的离线排序能力 ≠ 逆合成 Top-1 能力**：E7 是这一负结论的最干净版本（训练级变好、端到端不变差但 Top-1 下降）。
7. **AUC/pair 指标与 reaction Top-1 系统性错位**：ranker、校准器、guidance 三条线重复出现同一模式。
8. **当前实现不是 exact DGM**：变长坐标映射是硬边界。
9. **性能工程已收口**：TF32/batch64/forward 共享（opt-in）是可用资产；其余低风险优化接近上限。

【判断】第 4 条是最容易被误读的结论：当前文档把 R9K1M2 称为“准确率默认”，但它的选择证据主要来自 test-mini-1001 的结构消融（旧 checkpoint）与 validation-200 的参数冻结（新 checkpoint），而不是一个未参与任何选择的独立确认集。因此“R9 更好”目前应表述为“在预注册规则约束下的工程默认”，不是已确认的最终结论。

---

# 6. 失败尝试与经验

按失败模式抽象，而不是逐个罗列：

## 6.1 候选级指标与反应级指标错位（最核心的模式）

- 实例：residual ranker（AUC +2.45pp，Top-1 -9.6pp）、P2 校准（同组 AUC +0.026，Top-1 -0.5pp）、E7（pair acc +4.68pp，Top-1 -1.5pp）、E3 终端重排（Top-3/10 改善，Top-1 -0.6pp）。
- 原因分析（文档证据指向）：这些指标在“候选对/候选排序”层面优化，而 reaction Top-1 是“每个反应只取第一名”的决策任务；候选级正确性排序与跨 augmentation 聚合后的第一名不一致。
- 对后续设计的影响：任何新信号必须先在**同一候选池的 reaction Top-1** 上通过 gate，再进入 guidance；AUC/pair accuracy 只能作为筛选器，不能作为通过证据。

## 6.2 数据语义错误被误当成方法失败

- 实例：shared-anchor 数据实际不共享 state/time（P5 结果无效）；correctness-reward target leakage（v1 数字失效）；adaptive endpoint 时间近似错误（旧 1k 文件作废）；continuation 原地改写 GPU tensor（2/1000 条污染）。
- 经验：先审计数据语义与来源，再解释方法结果；项目后期已把“结构审计”写成每个实验的前置门槛。

## 6.3 启发式分数被误当成概率

- 实例：triggered-only + reverse 评分在历史上“表现好”，但只是偏爱低概率多编辑路径；完整单路径概率又过度偏向 no-op；changed-state bonus 是二元启发式。
- 经验：区分“校准的概率目标”与“搜索启发式”；SMC/DGM 的理论接口只有在目标/proposal 明确定义后才可声称。

## 6.4 更多预算不等于更好

- 实例：M3/M4、K5/K9/K10、R10、antithetic child、多次 no-op、n_runs=4 均无正向收益或回退。
- 经验：固定总预算 + 配对检验 + 预注册替代规则，是避免“堆计算”假象的必要条件。

## 6.5 性能优化中的可复现性纪律

- 实例：TF32 改变 2/9000 行、batch128 改变 1392/10000 行、padding 分桶改变逐行输出；这些都不影响 Top-k 但破坏逐字节复现。
- 经验：默认路径保持逐行一致；数值路径变化必须显式 opt-in，不能静默替换。

## 6.6 过程层面的记录缺口

- 实例：计划“10k–30k pilot 后再完整重训”，实际直接完成了 600k 重训且无 pilot 记录；旧 A/B 被多次选参后降级；早期“Euler 完整 test 30–40 分钟”无法复现。
- 经验（记录层面）：实验治理（冻结协议、数据角色、provenance）本身就是本项目的产出之一；后续任何新训练都应恢复 pilot gate 纪律。

---

# 7. 当前瓶颈与未解决问题

按证据强度排序：

1. **Reward 与真实逆合成正确性的失配**（证据最强）：forward-beam reward 方向弱且约半数错误候选得正分；所有校准/ranker 尝试均无法同时改善 AUC 与 Top-1。
2. **终点 reward → 单步动作的 credit assignment**（证据强）：终点对齐是粗粒度近似；自然一步数据稀疏；event-conditioned 第一步编辑的训练目标也未通过。
3. **Action-level guidance 的表达限制**：`per_position` 只重排位置内动作；`per_sample` 更差；严格 Z-space/exact DGM 被 89.7% 非双射插入阻塞。
4. **模型级条件形式**：copy-product 没有不可变的完整 product 条件；编辑分布极不均衡（delete 0.92%）。文档认为显式 product conditioning 是最高价值的训练分支，未执行。
5. **当前默认方法缺少独立确认**：R9K1M2 vs 普通 Euler 需要更大的严格对照；confirm/final/完整 test 均未运行。
6. **推理效率**：model forward 占约 74%；低风险收益接近上限（理论约 1.34×）；n_steps 消融未做。
7. **SMC 尚无可验证的 target**：mechanics 正确，但没有独立 reward，无法评估准确率。

【判断】瓶颈的层级关系是：1 和 2 是 DGM 线能否继续的前提；4 是模型线能否突破的天花板；5 是当前成果能否被最终报告的前提。3 和 6 是次级工程问题，不应在 1/2/4 之前成为主线。

---

# 8. 下一阶段任务

以下任务全部来自文档中已有的计划或明确缺口；优先级排序是【判断】。每项同时说明它试图回答的问题。

## P0：冻结并确认当前默认基线的真实收益

- 内容：在未使用过的 validation 区间（或按 v2 协议新建的分层集合）上，用已修复 seed 的普通 Euler 与 R9K1M2 做同预算、同聚合、反应级配对的大规模对照；随后按协议一次运行完整 src-test 作为冻结评估。
- 回答：R9K1M2 相对普通 Euler 到底提升多少、是否显著、是否值得作为最终报告的主方法。
- 依据：任务 27 的 200-reaction 对照不显著；§23.4 明确要求确认后冻结并运行完整 test；confirm/final 仍未被使用。
- 成本：几次 GPU 长跑；无新方法开发。

## P1：先解决“什么是好候选”，再谈 guidance

- 内容：按文档 P2 计划构造逆合成正确性 reward——真实反应物为正例、冻结 Euler 有效候选为负例（invalid 固定最低分）、反应级隔离、一次性 holdout；但 gate 必须以**同一候选池的 reaction Top-1** 为主（吸收 v2/v3 与 P1/P2 的教训），AUC 只作辅助。
- 回答：是否存在一个比 forward-beam 重构更接近“数据集正确逆合成”且推理时可用的信号。
- 依据：所有 guidance 失败都回溯到 reward 失配；endpoint rerank 路线因 Top-1 下降被关闭，但更强判别器（联合编码、反应级组损失）未被否定。

## P2：可验证的中间动作价值（而非继续扫 E2 参数）

- 内容：仅在 P1 出现正向信号后，重新设计局部 credit：受控 rollout 估计单动作的长期回报，或定义可验证的局部价值目标；event-conditioned 数据生成器可复用，但“终点 reward 直接复制到第一步编辑”的方案不得重开。
- 回答：终点奖励能否以可验证方式归属到单步动作；credit 失配是否就是 guidance 无法提升 Top-1 的原因。

## P3：显式 product conditioning 训练分支

- 内容：独立 YAML/checkpoint；先在 10k–30k pilot 上验证曲线与可复现性（恢复此前被跳过的 pilot gate），再做与 copy-product 的单变量对照；成功后再考虑 CFG。
- 回答：模型级条件（不可变 product 记忆）是否比推理侧 guidance 更能缩小 Oracle–Top-k 差距。
- 依据：训练审计与多份规划文档均将其列为“最高价值训练方向”；重训结果持平说明纯基础设施修复不够。

## P4：严格 Z-space / GAP 身份表示

- 内容：设计带持久位置身份的固定坐标表示，先验证 insert/delete 双射，再做 posterior 推导与 exact-DGM 边界判定。
- 回答：exact DGM 在当前任务上是否可行；若不可行，action-level 近似的理论天花板在哪里。
- 条件：高风险、高成本；应在 P1/P2 出现正向证据后再启动，否则只保留 DG-0/DG-1 审计成果。

## P5：低风险工程收尾

- 内容：BOS/特殊 token 完整硬约束诊断；n_steps=50/100/200 消融；GPU 状态 key/合并原型（要求逐行一致）。
- 回答：在不改变方法语义的前提下，效率与数值正确性还能走多远。

## 不建议立即做的事（文档已明确禁止或已关闭）

- 继续扫描 guidance β、seed、训练中间 checkpoint；
- 重开 P1/P2 reward 校准、v2/v3 ranker、L1/E2 局部 credit；
- 在 dev 上继续调参后用 confirm 验证；
- 在 reward/credit 没有正向信号前接入 Euler-Beam guidance 或 SMC reward。

---

# 9. 一页式项目状态摘要

**任务**：单步逆合成（产物 → 反应物），Edit Flows + 100 步 Euler，反应级评估（20 augmentation 聚合）。

**已经做到的**：

- 建立了可信、可复现的训练与评估基础设施（旧模型可审计；新模型与旧模型持平；reaction 级协议 + 保留集分层）。
- 采样端找到有受控证据的配置 R9K1M2（隔离谱系 + M2 局部选择）：在 mini-1001 上 Top-1/10/Oracle = 57.0/86.1/91.8%（旧 checkpoint），优于全局大 K 与更大 M。
- 性能端获得 TF32（约 -20%）与 forward 共享（opt-in，约 -26%）等稳定优化。
- DGM 方向建立了完整闭环（reward → 数据 → guidance → 采样 → 反应级评估），并用预注册 gate 严谨地关闭了 E3–E7、P1/P2、ranker v2/v3、L1/E2。

**关键数字**（非最终 test）：

- dev 普通 Euler 基线：Top-1/3/10/Oracle = 58.2/75.5/83.5/86.6%，wall 1,271 s。
- 最佳 guidance（E7）：Top-1 56.7%（CI [-3.3,+0.3] 相对基线），wall +48%；离线 pair acc 63.75% 高于 control 59.07%。
- Reward 判别力：AUC 0.69–0.73；约一半错误候选得正分。
- 已确认模式：候选级 AUC/排序提高 ↔ 反应级 Top-1 下降可稳定共存。

**核心瓶颈**：reward 与逆合成正确性失配；终点 → 动作 credit 不可靠；action-level 表达受限；模型缺少显式 product 条件。

**下一步（按优先级）**：P0 严格确认 R9 vs Euler 并冻结完整 test；P1 训练真正的正确性判别器并以 Top-1 gate 验收；P2 只在 P1 通过后研究可验证局部 credit；P3 并行启动显式 product conditioning 小规模 pilot；P4/P5 为条件性工程任务。

**状态标记**：confirm（1,000 反应）、final（2,000 反应）、完整 src-test（5,007 反应）均未使用；任何“已在保留集上验证”的说法都无文档依据。

---

> 本文件为项目级综合，事实部分可回溯至 `PROJECT_FACTS_SUMMARY.md` 与 `new_docs/` 原始文档；【判断】部分为本文作者基于文档证据的分析，不改变原始记录。

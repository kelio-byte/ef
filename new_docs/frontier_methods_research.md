# 前沿方法对 Euler-Beam / Edit Flows 的可借鉴方向

> 日期：2026-08-03。本文基于 `PDF/` 下四篇本地论文，只提出研究路线和验证门槛，
> 不把论文在其它离散模型或多步规划上的结果直接当作本项目收益。

## 0. 从研究方向进入实施计划时的记录要求

本文负责比较论文方向；任何方向真正进入代码实施前，必须在
`new_docs/euler_beam_next_stage_plan.md`按统一模板建立任务条目，依次记录：

1. 方法/改进本身及其作用层次；
2. 为什么做，以及已有项目证据；
3. 对应当前什么问题、预期好处、风险和边界；
4. 适配到本项目的公式、代码位置、接口、seed、metadata与兼容方案；
5. 预注册实验，以及实现、测试、结果、分析、结论和Git commit占位符。

不得先实现后补动机，也不得只记录“引用某论文、尝试某参数”。Q sharpening和Euler-SMC
的首批标准化条目已经写入上述规划文档的任务29和任务30；后续CFG、reverse corrector、
discrete guidance或其它方法也必须使用相同格式。

## 1. 论文与本项目的关系

| 论文 | 核心思想 | 与当前项目的距离 |
|---|---|---|
| Edit Flows: Flow Matching with Edit Operations (2025) | edit CTMC、alignment flow matching、CFG/Q sharpening/reverse/localized edits | 基础方法，最直接 |
| Discrete Guidance Matching (ICLR 2026) | 学习 density-ratio 的 posterior guidance，避免离散一阶近似 | 需要额外 guidance 网络和理论适配 |
| Inference-Time Scaling via Importance Weighting and Optimal Proposal Design (ICLR 2026) | SMC、重要性权重、ESS/重采样、最优 proposal | 与 Euler-Beam 的“多粒子+合并”最相近 |
| RetroAgent (COLM 2026) | LLM 在 AND-OR graph 结构化记忆上做多步逆合成规划 | 当前项目是单步模型，属于上层新任务 |

## 2. 最值得立即尝试：Q sharpening

Edit Flows 原论文明确对 `Q_ins/Q_sub` 使用 temperature、top-p、top-k；这些操作只改变
事件发生后采哪个 token，不改变 rate head 和每步是否编辑。

适用原因：

- 当前 invalid 和低排名噪声有一部分来自错误 token；
- 只需在 sampling logits 上实现，可复用当前 checkpoint；
- 额外计算几乎为零，容易做严格单变量实验。

风险：分布过尖会降低 Top-10 覆盖，尤其是本就需要多解的 retrosynthesis。不能只看
Top-1。建议建立独立 `exp/q-sharpening` 分支，先在 validation-200 预注册：

```text
temperature ∈ {1.0, 0.9, 0.8}
其它 K/M/R/seed/score/policy 完全固定
```

第一轮不同时扫描 top-p/top-k；通过门槛是 Top-1 不降、Top-3/10 或 invalid 至少一项有
稳定收益，并在不重叠 validation 区间复核。失败就停止，不在 test 上继续调温度。

## 3. 最值得开新方法分支：Euler-SMC

### 3.1 为什么比继续堆 K/M 更有理论价值

当前 Euler-Beam 已经像粒子系统：K 个父状态、M 个 proposal、状态合并和重采样式剪枝。
但它不是严格 SMC：

- child mass 只是父 mass 等分后按碰撞 log-sum-exp；
- Top-K 是确定性截断，不按 normalized weight 随机/系统重采样；
- 没有 ESS 判断何时真的需要 resample；
- `stochastic_noop` 改 proposal 后没有 `target/proposal` 校正；
- changed-state bonus 不是来自可定义的目标分布。

Inference-Time Scaling 论文的关键启发不是“把 beam 改名 SMC”，而是明确三件事：目标
分布 `π`、proposal `q`、逐步重要性比值。如果这三者写不出来，就仍是启发式搜索。

### 3.2 对本项目可定义的目标

不能使用测试 Target 作为 reward。可行 reward 按可信度排序：

1. 独立 forward reaction model 对 `reactants -> given product` 的一致性分数；
2. 独立 reaction feasibility/classifier 分数；
3. RDKit validity、原子守恒、价态/片段约束等弱化学 reward；
4. 模型自身分数或共识，只能作 proposal 诊断，不能冒充外部正确性 reward。

仅优化 RDKit validity 的风险很高：本项目的 antithetic 实验已经显示 invalid 下降可以同时
让 Top-k 下降。最理想的 Euler-SMC 需要一个独立、只在 train/validation 构建的 forward
consistency reward。

### 3.3 分阶段实现

建议新建 `research/euler-smc`，而不是直接替换 `euler_beam.py`：

1. **SMC mechanics**：实现 log-weight、ESS、systematic resampling、ancestor id 和完整
   proposal log-prob；用 synthetic 离散 CTMC 验证无偏/一致性。此阶段不宣称准确率提升。
2. **Bootstrap baseline**：proposal 仍用模型 Euler transition。没有 reward twisting 时，
   target 与 proposal 相同，权重应近似相等；若算法此时凭空产生大收益，先检查是否引入
   了不透明偏置。
3. **Terminal reward**：只在 validation 接入独立化学 reward，比较固定总 child budget
   下的 Top-1～10、Oracle、invalid、ESS 和 wall。
4. **Twisted intermediate target**：把 reward 强度随 t 从 0 平滑增至 1，避免早期粒子
   坍缩；必要时才研究 learned/amortized proposal。

首个版本不应同时加入 guidance network、learned proposal 和新训练模型。先证明粒子权重
与重采样实现正确，再逐项增加。

## 4. Discrete Guidance Matching：高潜力，但不能直接粘贴

该论文假设已知或可学习 endpoint density ratio `r(x)=q(x)/p(x)`，训练 posterior-based
guidance network 估计条件期望，然后用 guidance 修正 pretrained posterior。优点是相对
逐候选 rate guidance 更高效，并避免把离散状态当连续向量做一阶 Taylor 近似。

对本项目的潜在形式：

- base distribution：当前 Edit Flow 的 `p(reactants | product)`；
- target distribution：由 forward consistency/feasibility reward 倾斜后的分布；
- guidance：对 insert/sub/delete 的 token posterior 进行修正。

主要障碍：论文推导针对固定维度离散坐标，而 Edit Flows 是可变长度、位置随 insert/delete
变化的 edit state。必须重新推导 guidance 如何作用到 edit operation posterior 或辅助 Z
alignment；不能直接对现有 `Q_ins/Q_sub` 相乘就称为“exact guidance”。此外仍需要 density
ratio/reward 数据和额外 guidance 网络训练。

建议优先级低于 Euler-SMC：先用 SMC 验证某个化学 reward 是否真的改善 retrosynthesis；
只有 reward 有效且额外前向成为瓶颈时，再开 `research/edit-guidance` 推导和 amortization。

## 5. Edit Flows 原论文中尚未使用的训练方向

### 5.1 显式 conditional flow + CFG

当前 checkpoint 没有独立 product conditioning，也没有 condition dropout，因此不能在
推理时补一个 unconditional forward 就声称 CFG。先训练显式 product-conditioned 模型，
并随机 drop condition，之后才能对 `λ` 和 `Q` 分别做 CFG。这与训练审计中的最高价值
模型分支一致。

可行性：中高；预期收益高；代价是新架构和完整重训。

### 5.2 Reverse rates / forward-backward corrector

论文用正向 rate 前进一步、反向 rate 回退一部分时间，引入保持边际分布的自校正。对
SMILES 很有吸引力，因为它可能撤销早期错误编辑；但需要学习反向 CTMC，且必须明确这里
“reverse”指时间反向 rate，不是当前 `legacy_triggered_reverse` 排序模式。

可行性：中；计算约增加一套模型前向；需要独立 checkpoint 和严格采样推导。

### 5.3 Localized edit operations

论文让一次编辑在 Z 空间促进邻域编辑，长序列任务上收益明显。SMILES 的局部括号、环标号
和官能团也可能受益，但化学反应中的远程结构关联又不完全局部。它会改变 auxiliary CTMC
和 loss，不是简单在 sampler 中让相邻位置一起编辑。

可行性：中低；实现/验证成本高。应排在显式 conditioning 和 Q sharpening 之后。

## 6. RetroAgent：可以接在上层，不能改善当前单步 Top-k 本身

RetroAgent 解决的是多步 route planning：single-step model 只负责给某个 molecule 提供
候选，LLM 在 AND-OR graph、building-block database 和 chemistry tools 上决定扩展哪个
frontier、采用几个 reaction，并自动传播 solved/open 状态。

当前 Euler-Beam 可以作为它的 template-free single-step proposer，但这不会自动提高
USPTO-50K 单步 Top-1～10。若未来目标转向“最终找到可购买原料的完整路线”，可单独建立
`research/retro-planner`：

- canonical molecule/reaction 去重与 cycle prevention；
- OR molecule / AND reaction 状态传播；
- building-block availability、SA score、route depth/cost；
- current model 输出 Top-k 作为 expansion candidates；
- 先用 Retro*/A* 或轻量 policy 建 baseline，再评估是否真的需要 LLM。

论文方案每条路线需要几十次 LLM 调用并使用 RL 训练，成本远高于本项目的单步采样。除非
研究目标明确改成多步规划，否则不建议现在占用 Euler-Beam 优化主线。

## 7. 方向优先级

| 优先级 | 方向 | 需重训当前模型 | 额外推理成本 | 主要目标 | 建议 |
|---:|---|---:|---:|---|---|
| 1 | Q temperature 小消融 | 否 | 近似 0 | invalid / Top-k 排序 | 立即做，validation-only |
| 2 | BOS/特殊 token 硬约束诊断 | 否 | 近似 0 | 正确性 / invalid | 先统计再决定修复 |
| 3 | Euler-SMC mechanics | 否 | 中 | 原理正确的多粒子扩展 | 独立分支 |
| 4 | 显式 product conditioning | 是 | 中 | 条件准确率 | 下一训练主分支 |
| 5 | forward reward + SMC | 需额外模型 | 高 | 化学一致性 / Top-k | reward 先验证 |
| 6 | reverse-rate corrector | 是 | 高 | 自校正 | conditioning 后 |
| 7 | exact discrete guidance adaptation | 需 guidance 网络 | 中高 | reward guidance | 需新理论推导 |
| 8 | localized Edit Flow | 是 | 中 | 局部序列一致性 | 后期研究 |
| 9 | RetroAgent-style planner | 新系统 | 很高 | 多步路线成功率 | 目标改变时再做 |

## 8. 推荐的下一阶段

若继续当前“单步准确率 + 效率”目标，建议顺序是：

1. 冻结现推荐 NNN 与 R1K9/R1K10 结果，不再从 test-mini 调参数；
2. 在 validation 上统计 BOS/特殊 token 事件以及错误 token 分布；
3. 实现 Q temperature 的最小改动并完成预注册消融；
4. 同时只搭建 Euler-SMC 的 synthetic correctness tests，不立刻跑长数据；
5. 决定重训时，先修训练基础设施，再做显式 product conditioning 单变量对照。

这条路线把低成本推理改进、概率正确的新 sampler、需要重训的模型升级分开，避免一次大
重写后无法判断收益来自哪里。

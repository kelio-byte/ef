# After-SPE 第一阶段详细计划：反应中心引导首次编辑

更新日期：2026-08-24
状态：已完成。S0–S4/RC1 已完成；RC1 的 oracle upper bound 没有给出足够的端到端收益，因此不启动 S5/RC2 中心预测器，也不运行 RC3。

## 0. 当前执行进度（2026-08-24）

| 阶段 | 状态 | 当前结论 |
|---|---|---|
| S0 baseline 与哈希冻结 | 已完成 | 历史指标可参考，但缺少完整本地预测，因此 RC1 必须同协议重跑 B0 |
| S1 图级中心标签与 crosswalk | 已完成 | train/val 全部 processed block 可匹配；原始 train 多 5 条被历史预处理过滤的反应 |
| S2 图中心到 M500 token 映射 | 已完成 | 20,000/20,000 个抽样视图逐 token 精确复现，映射可用 |
| S3/RC0 局部性审计 | 已完成 | radius-1 仅占 28.737% token，却覆盖 91.088% 已有-token 编辑和 96.343% INS 入口，RC0 通过 |
| S4/RC1 CPU 准备 | 已完成 | true/pseudo sidecar、首事件位置 bias、hazard 守恒和诊断均通过测试 |
| S4/RC1 GPU 实验 | 已完成 | smoke、pilot100、dev-1000 均完成；真实中心改善局部首编辑，但对 B0 的 Top-k 没有稳定上界收益 |
| S5/RC2 中心预测器 | 不启动 | B1 oracle 仅 Top-1 +1.1 pp（95% CI 含 0），Top-10 不变、Oracle -0.8 pp；预测器不可能优于 oracle |

最终结论见 `after_spe/results/stage1/stage1_report.md`；完整机器可读数值见
`after_spe/results/stage1/rc1_dev1000_summary.json`。

## 1. 第一阶段究竟要回答什么

SPE-M500 已经成为当前 fragment-level baseline，但轨迹诊断表明：

- M500 仍有 `18.74%` 的自然轨迹出现“首个事件使 token 距离变差”；
- 一旦人为制造一个有害首 completion，M500 和 Atom 都几乎无法恢复；
- 九条 Euler 轨迹虽相互独立，却没有利用“反应通常发生在哪个位置”的化学信息。

因此第一阶段只回答一个问题：

> **Q1：真实/预测反应中心能否帮助九条 M500 轨迹更合理地选择首个编辑位置，并提高 Top-k 或候选覆盖？**

第一阶段不做以下事情：

- 不重训 SPE-M500 基础模型；
- 不修改 tokenizer 或 merge 数量；
- 不恢复旧 forward-reward DGM；
- 不改变现有编辑操作空间；
- 不使用完整 test 选择中心模型、中心强度或 checkpoint；
- 不同时加入 product memory、roll-in 或训练损失重加权等变化。

这样，若实验成功，可以明确归因于“反应中心是否为首编辑提供了有价值的位置先验”；若失败，也能判断是中心与字符串编辑不兼容、中心预测不准，还是首编辑本身不是主要瓶颈。

## 2. 为什么这条路线有依据

### 2.1 项目内证据

| 证据 | 已有结果 | 对本阶段的约束 |
|---|---:|---|
| M500 有害首事件 | 18.74% | 首编辑仍有改善空间 |
| Atom 有害首事件 | 25.86% | M500 已有优势，但问题未消失 |
| 强制有害首 completion 后最终命中 | Atom 0.74%，M500 2.13% | 后续纠错非常弱，应优先减少早期错误 |
| M500 平均序列/编辑数 | aligned 16.05，edit 4.13 | 图中心映射到较短 token 序列的成本可控 |
| 当前模型输入 | 只有动态 `x_t` 与 `t` | 现有模型没有显式化学中心或不可变 product memory |
| 九轨迹采样 | Euler N=9，各轨迹独立 | 可自然分配给多个中心假设，不需要跨轨迹竞争 |

详细证据见：

- `SPE/STATUS.md`
- `revision/motivation_report.md`
- `revision/results/augmentation_robustness/summary/summary.md`
- `new_docs/structured_diversification_v2_dev1000_report.md`

### 2.2 文献证据及其适用边界

[RetroXpert](https://proceedings.neurips.cc/paper/2020/hash/819f46e52c25763a55cc642422644317-Abstract.html) 将反应中心定义为逆合成时需要断开的产物键，先预测中心/生成 synthons，再补全反应物；其中心网络还用“断键数量”作为辅助任务。它支持“中心预测可以缩小生成入口”，但不证明把中心直接乘到 Edit Flows rate 上一定有效。

[LocalRetro](https://pubs.acs.org/doi/10.1021/jacsau.1c00246) 用局部反应性与全局注意力预测局部变化，说明反应中心不能只依靠邻域规则，还需要完整分子上下文。

[Graph2Edits](https://www.nature.com/articles/s41467-023-38851-5) 和 [G²Retro](https://www.nature.com/articles/s42004-023-00897-3) 将断键、键型变化、原子属性变化等作为不同中心/图编辑类型，并考虑多中心反应。这支持本计划保留 atom center、bond center 和多个中心假设，而不是强行只选一个断键。

[Root-aligned SMILES](https://pubs.rsc.org/en/content/articlelanding/2022/sc/d2sc02763a) 的核心是利用 atom mapping 让 product/reactants 字符串更紧密对齐。本项目的 global R-SMILES 正是中心信息可能映射到字符串编辑位置的基础；但这种映射是否在 SPE-M500 后仍成立，必须由 RC0 审计验证。

当前原始训练 CSV 的 40,008 条反应全部标为 `class=UNK`。因此本阶段必须在 **reaction class unknown** 条件下训练中心预测器，不能使用类别 token 或从真实反应类型获得捷径。

## 3. 冻结的基础对象与数据边界

### 3.1 端到端 baseline

```text
数据：datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500
checkpoint：new_checkpoints/spe_m500_checkpoints/checkpoint_step490000.pt
sampler：Euler
n_samples：9
n_steps：100
scheduler：checkpoint 中保存的 cubic 配置
seed：42
augmentation：20
开发集：dev_unique1000_aug20（1000 reactions，20000 rows）
```

现有 dev baseline 结果只有在以下内容完全一致时才能复用：输入 SHA256、基础 checkpoint SHA256、代码 revision、sampler、seed、步数、候选数、augmentation 聚合和 action-support 修复状态。任一项不同就重新生成 baseline，不能把历史表格中的数字直接抄作新对照。

### 3.2 数据角色

| 数据 | 本阶段允许用途 | 禁止用途 |
|---|---|---|
| `raw_train.csv` | 构建中心标签、训练中心预测器、内部 train/validation | 端到端最终结论 |
| M500 train aligned | 中心—字符串映射审计 | 选择端到端 sampler 参数 |
| `dev_unique1000_aug20` | true-center upper bound、predicted-center 方法开发与一次主比较 | 训练中心预测器 |
| `confirm_unique1000_aug20` | dev 通过后的一次确认 | 当前调参和可视化挑例 |
| `final_unique2000_aug20` | confirm 通过后的冻结复核 | 提前查看 |
| 完整 test | 本阶段不用 | 选中心模型、bias 强度、checkpoint |

global manifest 中 dev/confirm/final 分别有 `1000/1000/2000` 个不重叠 reaction。当前 M500 目录只有 dev 投影；只有 dev 方法通过后，才使用现有 `scripts/project_reaction_split.py` 的同等逻辑生成 M500 confirm/final，并记录源 manifest 与输出哈希。

### 3.3 数据泄漏规则

本阶段会同时出现“真实中心”和“预测中心”，必须严格分开：

- **真实中心**读取 mapped reactants，只用于标签构建、离线兼容性统计和 dev upper bound；结果必须标记为 `ORACLE / NOT DEPLOYABLE`。
- **预测中心**在正式推理时只能读取当前 product；不能读取 target、真实中心、reaction class 或评估标签。
- 真实 target 只能在所有候选冻结后用于 Top-k 评分和 paired analysis。
- production sampler 的函数接口不得接收 target 路径；oracle center 应放在单独诊断入口，避免误调用。

## 4. 总体执行顺序

```text
S0  冻结输入、哈希与 baseline
 ↓
S1  从 atom-mapped raw reaction 构建图级反应中心标签
 ↓
S2  建立 raw reaction → global R-SMILES → M500 token 的可审计映射
 ↓
S3  RC0：中心与字符串编辑的兼容性统计
 ↓
S4  RC1：true-center 首编辑 upper bound
 ├─ 无足够上界收益（本次观察到）→ 停止中心路线，整理负结果
 └─ 有足够上界收益
      ↓
S5  RC2：训练仅看产物的反应中心预测器
      ↓
S6  RC3：predicted-center 九轨迹首编辑采样
      ↓
S7  汇总、决策和 Git 记录
```

S1–S3 可在 CPU 上完成。RC1/RC3 需要 GPU 推理；RC2 是一个只以产物分子图为输入的小型反应中心预测模型，不重训 15.8M 参数的 SPE-M500 模型。

## 5. S0：冻结输入、哈希与 baseline

### 5.1 目标

保证后面所有差异来自中心先验，而不是输入、checkpoint、评分或普通 Euler 行为改变。

### 5.2 必须记录

- Git commit 与 dirty worktree 列表；
- M500@490K checkpoint 路径、大小与 SHA256；
- M500 vocab、SPE rule 文件及 `merges=500` 的 SHA256；
- raw train/val CSV SHA256；
- dev src/tgt、global manifest SHA256；
- baseline predictions、metadata、score log 的路径和 SHA256；
- Python、PyTorch、RDKit、SmilesPE 版本；
- GPU、batch size、wall-clock 和峰值显存。

### 5.3 baseline 复用检查

先检索近期结果，不自动重跑。只有 sampling metadata 能证明它是：

```text
M500@490K + Euler N=9 + 100 steps + seed42 + dev1000 × 20 aug
```

且代码/action-support 一致时才复用。否则重新运行一次，并将它作为本阶段唯一 baseline。

### 5.4 交付物

```text
after_spe/results/stage1/s0_manifest.json
after_spe/results/stage1/s0_baseline_check.md
```

## 6. S1：构建图级反应中心标签

### 6.1 数据源

使用：

```text
datasets/USPTO_50K/raw_train.csv
datasets/USPTO_50K/raw_val.csv
```

CSV 字段是 `reactants>reagents>production`，原子带 map number。product 是 Edit Flows 的起始状态，reactants 是目标状态。解析时必须按 reaction SMILES 的三个字段处理，不能简单按字符串中任意 `>` 切割后猜测。

### 6.2 中心标签定义

在比较图属性前，先用固定 RDKit 版本完成 sanitize，并统一芳香键/Kekulé 表示。只由共振写法或 Kekulé 形式变化造成的 aromatic/conjugation 差异不计为反应中心；规范化步骤、RDKit 版本和最终比较字段写入 manifest。否则中心标签会混入“字符串/表示变化”，无法代表真实化学变化。

对 map number 相同的 product/reactant 原子建立对应，构造三类标签：

#### A. Bond center

若两个保留原子之间的键满足任一条件，其两个端点和该键标为中心：

- product 有键、reactants 无键；
- reactants 有键、product 无键；
- 两侧都有键但 bond type、aromatic/conjugation 或 stereo 不同。

#### B. Atom center

对保留原子比较：

- formal charge；
- total H / explicit H；
- chirality；
- aromaticity；
- radical/equivalent atom property。

属性发生变化时标为 atom center。具体字段必须在实现前冻结，避免看结果后增删标签定义。

#### C. Attachment center

若 reactants 中存在 product 没有的原子/片段，并与一个 product 保留原子相连，则把该保留原子标为 attachment center。新增原子本身不在 product 图上，不能作为推理时的中心候选。

另外记录：

- 只存在于 product 的 mapped atom；
- duplicated/missing map number；
- 无法解析反应；
- product/reactants 图完全相同但字符串目标不同；
- 多中心及相互距离。

这些异常不能静默修补；应分类型计数并保存样例。

### 6.3 中心假设的形成

将相距一条 product bond 以内的中心 atom/bond 合并为一个 center component。每个 component 保存：

```text
reaction_id
center_type(s)
atom_map_ids
bond_map_pairs
component_size
radius-1 / radius-2 atom sets
```

多中心反应保留多个 component。当前阶段最多使用三个中心假设，但数据报告必须显示 `>3` 中心的真实比例，不能预先截断标签。

### 6.4 映射训练集时不能直接依赖行号

本轮已确认：

- raw train：40,008 reactions；
- M500 train：800,060 rows = 40,003 reaction blocks × 20 augmentation。

两者相差 5 个反应。应使用以下 key 建立 multimap：

```text
(canonical map-free product,
 sorted canonical map-free reactant components)
```

若 key 重复，再用 reaction id、原始 occurrence 次序和完整 mapped reaction hash 消歧。最终必须报告：

- 唯一匹配数；
- 多义匹配数；
- raw-only / processed-only 数；
- 被过滤的 5 个反应的明确原因；
- train/val 是否存在跨 split 重复 canonical reaction。

匹配率达不到预期时停止后续工作，先修 crosswalk；不允许用 `zip()` 或最短长度截断。

### 6.5 单元测试

至少覆盖：

- 单断键；
- 键级变化；
- 原子电荷/手性变化；
- reactant-only attachment；
- 两个分离中心；
- duplicate map、map=0、非法 SMILES；
- reagent 字段非空；
- component 顺序变化但 canonical key 相同。

### 6.6 交付物

建议新增：

```text
edit_flows/chem/reaction_center.py
scripts/build_reaction_center_labels.py
tests/chem/test_reaction_center.py
after_spe/results/stage1/s1_center_label_report.md
after_spe/results/stage1/s1_crosswalk_report.json
```

大体积逐反应标签可以放在 `results/after_spe_stage1/cache/`，文档目录只保存 manifest、统计和哈希。

## 7. S2：把图中心映射到 global R-SMILES 与 M500 token

### 7.1 为什么这是独立 gate

图上正确的中心，不一定对应一个稳定的 SPE token 位置：

- global R-SMILES 会改变根和分支书写顺序；
- 同一反应有 20 个 augmentation；
- SPE token 可能同时包含原子、括号、环号和键符号；
- 新反应物片段的多个 INS token 可能都由同一个 attachment center 触发。

所以必须先证明“中心→首编辑位置”映射可靠，再训练 predictor。

### 7.2 推荐实现：带 provenance 的 SPE replay

不要对最终 M500 token 用字符串模糊匹配。推荐：

1. 对每条 product R-SMILES 做与现有预处理一致的 atom-wise tokenization；
2. 每个原子 token 同时携带其 RDKit atom index / 临时 atom map；语法 token 携带空集合；
3. 按 `SPE_ChEMBL.txt` 前 500 条规则原样 replay merge；
4. merge 后 token 的 atom 集合为两个子 token atom 集合的并集；
5. 最终 token surface 必须逐 token 等于现有 M500 文件；拼接后必须逐字符还原原始 product。

不能重新训练 SPE，也不能使用不同的正则 tokenizer。metadata 已冻结为 `SmilesPE==0.0.3`、`merges=500`、`dropout=0`。

### 7.3 token 与 insertion anchor 的中心分数

对当前 product 的 M500 token 位置 `i`：

- 作用于已有 token 的编辑位置分数：token 所含任一 atom 到指定 center component 的最短图距离；
- syntax-only token：使用左右最近有原子 token 的最小距离，但另行标记为 syntax projection；
- INS anchor `i`：取 anchor 两侧最近原子的最小中心距离；`INS(pos=0)` 使用 BOS 后第一个可定位原子；
- 无法定位的 token/anchor 得 0 分并计入 fallback，不猜测位置。

固定径向分数：

```text
distance = 0  → score = 1.0
distance = 1  → score = 0.5
distance >= 2 → score = 0.0
```

该定义在看端到端结果前冻结。

### 7.4 INS 的统计单位

不能把连续新增 fragment 的每个 token 都当作独立“反应中心命中”。在 aligned source/target 中，将连续 `<GAP> → token` 合并成一个 insertion run，只评估该 run 的 attachment anchor 到中心的距离。

否则长新增片段会人为制造大量“远离中心的 INS”，从而错误否定中心机制。

### 7.5 一致性检查

对全部 20 augmentation 报告：

- M500 token 逐字复现率，要求 100%；
- 能定位到 product atom 的 token/anchor 比例；
- 同一反应 20 个视图映射回图后 center component 是否一致；
- center token 数/总 token 数；
- 一个 token 覆盖多个中心 component 的比例；
- syntax-only 中心位置比例。

任一输入无法逐字复现现有 M500 tokenization 时，不能继续 RC0；先修 tokenizer provenance。

### 7.6 交付物

```text
edit_flows/chem/spe_provenance.py
tests/chem/test_spe_provenance.py
after_spe/results/stage1/s2_mapping_report.md
after_spe/results/stage1/s2_mapping_examples.jsonl
```

## 8. S3 / RC0：中心与当前编辑任务的兼容性审计

### 8.1 目标

在不训练中心模型、不运行完整采样前，判断化学中心是否真的覆盖 global-M500 的关键编辑入口。

### 8.2 必须报告的统计

按 reaction 聚合，而不是把 20 augmentation 当作 20 个独立样本：

| 指标 | 含义 |
|---|---|
| center component count | 单中心/多中心难度 |
| center token sparsity | 先验能缩小多少位置空间 |
| insertion-run anchor recall@r | 新片段插入入口是否靠近中心 |
| existing-token edit recall@r | 作用于已有 fragment 的编辑是否靠近中心 |
| any/all augmentation consistency | 结论是否依赖某一种 R-SMILES 写法 |
| token center collision | 一个大 SPE token 是否同时混合中心/非中心信息 |

`r=0/1/2` 全部报告，但后续 sampler 主方案只使用前述冻结的 `1.0/0.5/0` 径向分数。

### 8.3 预先定义的兼容性判断

不设一个脱离数据的绝对 90% 阈值。采用两层判断：

1. **结构层：**中心位置必须明显稀疏于全序列，同时对 insertion-run anchor 与其他已有 token 编辑位置有实质覆盖；若中心扩到 radius-2 后接近覆盖全序列，则它没有提供有效先验。
2. **行为层：**最终由 RC1 true-center upper bound 判断是否能改善同预算采样。静态 recall 高但 RC1 无收益，仍视为机制失败。

### 8.4 交付物

```text
after_spe/results/stage1/rc0_locality.json
after_spe/results/stage1/rc0_locality.md
```

## 9. S4 / RC1：真实中心首编辑 upper bound

### 9.1 它回答什么

真实中心 upper bound 不用于报告可部署准确率，只回答：

> 如果反应中心完全预测正确，仅将它用于第一次发生编辑的位置分布，当前 M500 模型是否有能力把这条信息转化为更好的终点？

这一步必须在训练中心 predictor 之前完成。若答案是否定的，就不值得花时间优化 predictor。

### 9.2 “首编辑”的准确语义

普通 Euler 在一个数值步可能并行采样多个位置。因此本计划中的“首编辑”严格指：

> 每条轨迹第一次出现任意实际编辑的 **first non-noop Euler step**。

中心 bias 从 `t=0` 开始作用；一旦该轨迹完成第一个非空 Euler step，后续 99 个或剩余步骤立即恢复普通 Euler。必须额外记录该步包含 1 个还是多个动作；若多动作比例很高，再决定是否需要单动作版本，本阶段不提前改变 Euler 语义。

### 9.3 只改变位置，不改变操作总量

对某条轨迹指定的 center component，位置权重为：

```text
w_{i,m} = exp(log(3) * center_score_{i,m})
```

即中心位置最多获得 3 倍位置权重，radius-1 获得约 `sqrt(3)` 倍。`log(3)` 是唯一预注册主值，不在 dev 扫描。

为了把因果变量限制为“位置”，对每一种现有编辑 mode 分别保持其全序列总 rate：

```text
lambda'_{i,m} = lambda_{i,m} * w_{i,m}
                  * sum_j lambda_{j,m}
                  / sum_j (lambda_{j,m} * w_{j,m})
```

只在合法 position mask 内求和。这样：

- 各编辑 mode 的总 hazard 分别不变；
- token completion `Q_ins/Q_sub` 完全不变；
- 唯一变化是第一次非空 Euler step 更倾向中心附近的位置。

若某个 mode 的合法总 rate 为零，则保持原值，不能产生 NaN 或均匀“复活”已屏蔽动作。

### 9.4 九条轨迹如何分配

将真实中心的 connected components 按确定性规则排序：

1. changed bond component 优先；
2. 再按 component 内 changed edge/atom 数；
3. 最后按 atom-map id 排序。

取前三个 center hypotheses，每个分配 3 条轨迹。若不足三个，则循环分配已有中心；若无可识别中心，九条轨迹退回普通 Euler并记录 fallback。若 `>3` 中心反应比例不可忽略，RC0 报告后再决定，不在当前 dev 临时改预算。

### 9.5 对照组

| 组 | 中心来源 | 用途 |
|---|---|---|
| B0 | 无中心，普通 Euler N=9 | 正式 baseline |
| B1 | 真实中心 | 机制 upper bound，不可部署 |
| B2 | 同一 product 上匹配中心数量/类型的确定性随机 pseudo-centers | 负对照，检查普通 rate 扰动是否也能“提升” |

B2 在同一个 product 图上按固定 seed 选择与真实中心数量、atom/bond 类型匹配、且尽量落在真实中心 radius-2 外的 pseudo-centers。它保持 product 尺寸和分支预算，只破坏化学中心对应关系；无法找到足够远位置时记录并放宽距离，不能改用其他 product 的坐标。若 B1 与 B2 同样变化，不能把收益归因于化学中心。

### 9.6 先 smoke，再 dev

1. 单元测试；
2. 10 reactions × 20 augmentation：检查输出行数、首事件、fallback、无 NaN；
3. 固定 100 reactions：只做机制 sanity，不下准确率结论；
4. dev-1000 主实验。

### 9.7 必须记录

- 每条轨迹指定的 center component；
- first non-noop step/t；
- 该步动作数量、mode、位置、token completion；
- 动作到指定中心的图距离；
- 各 mode 调整前后的总 hazard 误差；
- 九条轨迹首事件重复率；
- 最终 canonical unique 数、invalid 和命中；
- center lookup 与采样 wall-clock。

### 9.8 RC1 判断

继续训练 predictor 的必要条件：

- B1 相比 B0 的 Top-1 点估计不下降；
- Top-3/Top-10/Oracle 至少一项提高；
- invalid 增加不超过 0.5 pp；
- B1 明显优于 same-product pseudo-center B2；
- 首事件更靠近真实中心，且有害首事件没有增加；
- reaction-level paired bootstrap 不显示明确负向。

若 B1 无收益，停止 reaction-center sampler；仍保留标签和 RC0 作为负结果。不得通过扫描 10 个 bias 强度挽救。

### 9.9 RC1 最终结果（2026-08-24）

所有预注册组已在 `dev_unique1000_aug20` 完成。B0 与倍率为 1 的 B0-trace 的 180,000 条预测逐字节一致；B1/B2 的总 hazard 误差均低于 `2.48e-6`。

| 条件 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle-any |
|---|---:|---:|---:|---:|---:|
| B0 ordinary Euler | 60.1% | 76.6% | 80.5% | 83.7% | 90.0% |
| B1 true-center oracle | 61.2% | 77.0% | 80.1% | 83.7% | 89.2% |
| B2 pseudo-center | 59.0% | 76.0% | 79.5% | 83.8% | 89.0% |

B1 比 B2 的 Top-1 高 `2.2 pp`（paired bootstrap 95% CI `[+0.5, +3.9]`），且第一次编辑更接近目标、较少变远，说明真实中心方向不是随机扰动。然而 B1 相比实际 baseline B0 的 Top-1 仅 `+1.1 pp`（CI `[-0.5, +2.7]`），Top-10 不变、Oracle `-0.8 pp`；这不是足以训练一个信息更弱 predictor 的上界。

因此 RC1 判定为**机制局部有效、端到端不足**。不启动 RC2/RC3，不扫描额外倍率，也不使用 confirm/final/test 寻找偶然收益。完整解释见 `after_spe/results/stage1/stage1_report.md`。

## 10. S5 / RC2：训练仅看产物的反应中心预测器

RC2 只在 RC1 通过后开始。本轮 RC1 未达到足够的 oracle 上界，因此本节保留为历史设计，不执行。

这里的“仅看产物（product-only）”是指：模型输入只有待逆合成的 product 分子图，输出其中哪些原子或键最可能属于反应中心。它不读取真实 reactants、target、真实中心、reaction class 或 atom-map id，因此推理时可真正部署。它与 M500 生成模型彼此独立：前者只提供前三个中心位置假设，后者仍负责生成反应物。

### 10.1 为什么优先图模型

同一 product 有 20 个不同 R-SMILES，图中心却相同。图模型天然不依赖字符串遍历顺序，输出也更接近 RetroXpert/LocalRetro 等文献中的化学中心定义。

当前 `requirements.txt` 没有 PyTorch Geometric 或 DGL。第一版不应为了一个小模型引入复杂 CUDA 扩展；建议使用 RDKit 特征和纯 PyTorch 实现 4 层小型 message-passing network。

### 10.2 输入特征

Atom：

- atomic number；
- degree；
- formal charge；
- total/explicit H；
- aromatic、ring；
- hybridization；
- chirality。

Bond：

- bond type；
- conjugated；
- ring；
- stereo。

不输入 atom-map number、reaction id、真实 reaction class 或 target 派生统计。

### 10.3 输出

- 每个 atom 的 center logit；
- 每条 bond 的 center logit；
- 一个辅助 center-count head：`0 / 1 / 2 / >=3`。

center-count 辅助任务有 RetroXpert 的直接机制依据，但它只作为训练正则与诊断；九轨迹主方案仍固定取 Top-3 候选，避免 count 错误导致没有候选。

### 10.4 训练 split

- 只使用 raw train；
- 按原始 reaction 做确定性 90%/10% internal train/validation；
- 按 center type/count 分层；
- raw val、dev/confirm/final/test 不参与训练或 checkpoint 选择；
- 每个分子图只作为一个样本，不把 20 augmentation 重复为 20 个图样本。

### 10.5 第一版固定配置

建议冻结一个小配置，不做架构搜索：

```text
message-passing layers：4
hidden dim：128
dropout：0.1
optimizer：AdamW
loss：atom BCE + bond BCE + 0.2 × center-count CE
checkpoint：internal validation PR-AUC/Top-3 recall 的预注册组合
seed：42
```

正负样本高度不平衡，BCE 的 `pos_weight` 仅由 internal train 统计计算并设上限；不能根据 dev 结果调整。准确率不是主指标。

### 10.6 离线指标

按 reaction 报告：

- atom/bond PR-AUC、ROC-AUC；
- Top-1/Top-3 candidate center recall；
- 任一真实 component 被命中的比例；
- 所有真实 components 被覆盖的比例；
- center count accuracy；
- calibration/ECE；
- single-center 与 multi-center 分层结果；
- 反应类别未知条件下的表现。

Top-3 candidate 使用图距离 1 的 non-maximum suppression，避免三个候选只是同一中心的相邻原子。

### 10.7 predictor 停止条件

不设置脱离 upper bound 的任意绝对 recall 门槛。最终看 RC3 能保留多少 RC1 upper-bound 收益。但如果 predictor 的 Top-3 连真实 center component 都大面积无法覆盖，则不运行完整 RC3，先把它记录为 predictor 失败，而不是浪费端到端推理。

### 10.8 交付物

```text
edit_flows/reaction_center/model.py
edit_flows/reaction_center/data.py
scripts/train_reaction_center.py
scripts/evaluate_reaction_center.py
tests/reaction_center/
after_spe/results/stage1/rc2_training_report.md
after_spe/results/stage1/rc2_metrics.json
```

checkpoint 必须保存模型配置、feature schema、label manifest 和数据哈希。

## 11. S6 / RC3：预测中心的九轨迹正式实验

### 11.1 候选中心生成

每个 product 图只运行一次 predictor：

1. 合并 atom/bond 分数；
2. 用图距离 1 NMS 取 Top-3 center hypotheses；
3. 将三个中心映射到当前 augmentation 的 M500 token/INS anchor；
4. 每个中心分配三条轨迹；
5. center predictor 失败、RDKit 失败或映射失败时回退普通 Euler，并记录原因。

20 个 augmentation 对应同一 product graph 时，允许缓存图模型结果；但每个字符串视图仍需独立做 token-position projection。

### 11.2 sampler 实现边界

建议为普通 Euler 增加一个默认关闭的 first-event position-bias hook，或单独封装 `sample_center_first_edit_euler`；无论采用哪种方式，必须满足：

- `bias=None` 与当前 `sample_euler` 在固定 seed 下逐字节一致；
- 全零/常数 score 不改变采样；
- 每个 mode 的总 hazard 数值误差小于预注册容差；
- BOS 和其他特殊位置继续沿用现有 action-support 约束；
- special token/no-op substitution 的现有屏蔽完全保留；
- center bias 只持续到每条轨迹自己的 first non-noop step，不按整个 batch 同时关闭。

### 11.3 正式对照

| 组 | 方法 | 说明 |
|---|---|---|
| C0 | M500@490K + Euler N=9 | baseline |
| C1 | M500@490K + predicted-center N=9 | 主候选 |
| C2 | M500@490K + same-product random pseudo-centers N=9 | rate 扰动负对照 |
| C-oracle | RC1 true-center | 仅展示 upper bound，不与可部署方法混为一列 |

所有组保持 n_steps、n_samples、augmentation、seed、batch、token completion、最终聚合和评分一致。

### 11.4 主指标

- Top-1/3/5/10；
- Oracle-any；
- Invalid@1；
- 每个 reaction 的 unique canonical candidates；
- wall-clock、center predictor 时间、采样时间、峰值显存；
- 首事件中心距离；
- 首事件 mode/token 分布；
- 九轨迹首事件重复率和最终重复率；
- 有害首事件比例。

统计单位必须是原始 reaction。20 augmentation 只参与候选聚合，不当作 20 个独立样本计算显著性。

### 11.5 采用门槛

seed42 dev 上必须同时满足：

1. Top-1 不低于 C0；
2. Top-3、Top-10 或 Oracle 至少一项提高；
3. Invalid 增加不超过 0.5 pp；
4. C1 优于 same-product pseudo-center C2，说明不是任意扰动；
5. center 成本计入后仍有可接受效率；
6. reaction-level paired bootstrap 不显示明确负向。

通过后才补 seed 7/123。多 seed 方向一致后，冻结 predictor checkpoint、bias、NMS 和 sampler，再生成 M500 confirm；confirm 只运行一次。

### 11.6 失败如何定位

| 现象 | 解释 | 后续 |
|---|---|---|
| RC1 无收益 | 真实中心也不能帮助当前首编辑 | 关闭中心采样 |
| RC1 有效、RC2 recall 低 | predictor 问题 | 可改一次 predictor，不碰 sampler |
| RC2 好、RC3 首事件更中心但 Top-k 不升 | 首编辑不是主要终点瓶颈 | 不扫 bias，转 product memory |
| C1 与 same-product pseudo-center C2 相同 | 普通 rate 扰动而非化学信号 | 关闭方法 |
| Top-1 升、Oracle 降 | 中心过度集中，损失多路线覆盖 | 不作为默认；分析中心分层/NMS |
| Invalid 明显升 | 中心位置正确但 completion/语法失败 | 记录后转 product conditioning/roll-in |

## 12. 测试与正确性要求

### 12.1 数据测试

- raw CSV 三字段解析；
- atom map 唯一性；
- crosswalk 不截断；
- 20 augmentation block 完整；
- dev/confirm/final reaction index 零重叠；
- M500 token surface 100% 复现；
- target 不进入 formal center predictor/sampler。

### 12.2 sampler 测试

- `bias=None` 固定 seed 精确复现 Euler；
- constant bias 精确复现 Euler；
- 每 mode hazard 保持；
- bias 只在各自轨迹首个非空 step 前生效；
- 多动作首 step 正确记录；
- BOS/PAD/special/no-op support 不回退；
- predictor/mapping 失败安全回退 Euler；
- 固定 batch size 下输出行顺序和 reaction id 正确；若另做 batch benchmark，不假定不同 padding/RNG 组织会逐字节复现。

### 12.3 统计测试

- 所有 Top-k 以 reaction 为单位；
- paired bootstrap 按 reaction 重采样；
- 20 augmentation 不作为独立置信区间样本；
- C0/C1/C2 输入、候选预算和评分完全一致；
- oracle upper bound 在所有表中显式标注不可部署。

## 13. 目录和记录约定

建议最终形成：

```text
after_spe/
  next_stage_plan.md
  next_stage1.md
  stage1_report.md
  results/stage1/
    s0_manifest.json
    s0_baseline_check.md
    s1_center_label_report.md
    s1_crosswalk_report.json
    s2_mapping_report.md
    rc0_locality.md
    rc0_locality.json
    rc1_oracle_upper_bound.md
    rc2_training_report.md
    rc2_metrics.json
    rc3_dev_comparison.md
```

大预测文件、中心 label tensor 和 checkpoint 不直接塞进 Markdown 目录；只记录可定位路径、大小、SHA256 和生成命令。需要 Git LFS 的文件在上传前单独核实，不能再次提交 pointer 代替真实文件而不自知。

## 14. 预计时间与资源

在现有代码和数据可用的前提下：

| 工作 | 预计人工/运行时间 | GPU |
|---|---|---|
| S0 基线与哈希审计 | 1–2 小时 | 仅缺失 baseline 时约 25–35 分钟 |
| S1 标签/crosswalk | 0.5–1 天 | 否 |
| S2 SPE provenance 映射 | 0.5–1 天 | 否 |
| RC0 locality audit | 2–4 小时 | 否 |
| RC1 sampler + 测试 + dev | 0.5–1 天 | 约 1–2 小时 |
| RC2 predictor + 测试 | 0.5–1 天 | 训练预计 1–3 小时 |
| RC3 predicted-center dev | 0.5–1 天 | seed42 与负对照约 1–2 小时 |

整体约 `2.5–4.5` 个工作日，GPU 累计约 `3–6` 小时；若 RC1 失败，会提前停止 predictor 与 RC3，时间显著缩短。时间是工程估算，实际以首个 100-reaction benchmark 为准。

## 15. 第一阶段最终决策表

| RC1 true center | RC3 predicted center | 第一阶段结论 |
|---|---|---|
| 失败 | 不运行 | 反应中心不适合当前首编辑机制；转 product memory |
| 成功 | 失败（predictor recall 低） | 中心机制有潜力，允许只改一次 predictor |
| 成功 | 首事件改善但 Top-k 不升 | 首编辑不是充分条件；停止调 bias，转 product memory |
| 成功 | Top-k 通过门槛 | 固化 center-first Euler，进入多 seed/confirm |

## 16. 完成标准

第一阶段只有在以下内容全部完成后才算结束：

1. raw→processed→M500 的 center provenance 可复现且有哈希；
2. RC0 说明中心与现有 token 编辑位置的真实兼容程度；
3. RC1 给出 true-center upper bound 和 same-product pseudo-center 负对照；
4. 若 RC1 通过，RC2/RC3 给出 predicted-center 结果；
5. 所有结果使用 reaction-level 统计，明确 oracle 与可部署结果；
6. `stage1_report.md` 用一页结论回答：中心是否继续，以及下一阶段是否转 product memory；
7. 代码、测试、文档和必要小文件完成 Git commit；大文件是否上传单独记录。

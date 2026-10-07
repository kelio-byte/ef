# Reward Ranker v2：独立 endpoint-ranking 协议

状态：协议已冻结；本实验只研究终点候选排序，不训练 DGM。旧的 reward holdout（1000–1199）已经使用过，不再用于选择模型、特征或阈值。

## 研究问题

新 reward 是否能在同一个冻结 Euler 候选池中，把真实反应物稳定排到第一名？这里先隔离 endpoint ranking，不能把 AUC 提升直接解释为 DGM guidance 有效。

## 数据划分

所有划分以原始反应为单位；每个原始反应的 20 条 augmentation 保持在同一侧。以下区间均来自训练文件，且与已有 guidance/reward artifacts 的已用区间不重叠：

| 角色 | 原始 reaction index | 用途 |
| --- | --- | --- |
| ranker train | 2000–2999（1000 个） | 构造候选、训练 ranker |
| ranker validation | 3000–3199（200 个） | 只用于固定模型/损失选择规则，不看 holdout |
| ranker holdout | 3200–3399（200 个） | 一次性最终 endpoint gate |

候选生成统一为：冻结 `checkpoint_step600000.pt`，Euler 100 steps，anchor steps 10/30/50/70/90，每个 shared anchor 4 条 continuation，seed 42。随后使用冻结 Molecular Transformer `MIT_mixed_augm_model_average_20.pt` 的 beam=5 reconstruction reciprocal-rank reward。生成阶段不读取真实 target；target 只在 ranker 训练标签和最终离线统计阶段使用。

## 预注册对照

在同一个候选池上比较三种排序：

1. raw forward reciprocal-rank（不可学习的 baseline）；
2. 保序 residual ranker：`score = raw_forward_score + learned_residual`，只学习修正项，不允许任意推翻 raw score 的严格高低关系；
3. listwise/hard-negative ranker：以同一 reaction 的候选集合为单位，要求真实候选超过 raw reward 排名靠前的错误候选。

所有模型只使用推理时可得的产物、候选反应物、raw reward、有效性和候选/产物结构特征；不使用真实 target、reaction index 或 holdout 标签作为输入。invalid 候选固定最低分，并单独统计。

## 训练与选择规则

- train reaction 内 deduplicate `(product_index, canonical_candidate)`；保留 raw reward 最强的代表记录。
- 每个有真实候选的 reaction 构造正例集合与 hard-negative 集合；优先加入 raw reward 排名前列的错误候选。
- train/validation 按 reaction 分割，不能按 candidate 或 augmentation 分割。
- 只用 validation 选择预注册的损失/模型版本；holdout 在模型、特征和选择规则冻结后只评估一次。
- 不在旧 holdout、dev、confirm、final 或 test 上调参。

## Endpoint gate

ranker 只有同时满足以下条件，才允许进入 action-level guidance 研究：

1. holdout reaction-level Top-1 点估计不低于 raw forward baseline；
2. Top-3 或 Top-10 至少一项提高；
3. Oracle 不下降，invalid 比例不增加超过 0.5 个百分点；
4. paired bootstrap 不显示明显负向 Top-1；
5. 报告候选数、去重率、forward reward 推理成本和所有输入/模型哈希。

若 gate 失败，关闭该 ranker 配方，不重建 guidance 数据、不训练 DGM。只有 gate 通过，才另立 P4 protocol 验证 action-level credit；endpoint gate 本身不能证明 guidance 有效。


# Reward ranker v3：大规模复核协议

状态：已冻结并执行完成。该复核只判断 endpoint reranking 是否值得保留，不训练 DGM。结果见 [`dgm_reward_ranker_v3_large_report.md`](dgm_reward_ranker_v3_large_report.md)。

## 目的

v2 的 train/validation/holdout 只有 `1000/200/200` 个独立 reaction，且 ranker 是小型线性校正器。v3 不改变模型结构、特征、loss 或超参数，只扩大独立 reaction 数，回答一个限定问题：

> 在更大且未用于 v2 ranker 选择的候选池上，当前 residual/listwise reranker 是否能稳定超过冻结 Molecular Transformer 的 raw forward reward？

## 数据划分

数据来自训练集原始 reaction index。已审计既有 candidate artifacts，以下范围未被旧 reward/guidance candidate 数据使用：

| 角色 | reaction index | 独立 reaction | 原始记录数 |
|---|---:|---:|---:|
| ranker train | 10000–17999 | 8000 | 160000 |
| ranker validation | 18000–18999 | 1000 | 20000 |
| 一次性 holdout | 19000–19999 | 1000 | 20000 |

每个 product 使用同一冻结候选协议：100 Euler steps；第 10、30、50、70、90 步作为 shared anchors；每个 anchor 4 条独立 continuation；seed 42。每个 product 产生 20 条原始终点记录。候选生成不读取真实 target；候选池固定后才用 target 构造离线标签与 Top-k 统计。

基础 checkpoint：`new_checkpoints/checkpoint_step600000.pt`。

forward reward checkpoint：`new_checkpoints/MIT_mixed_augm_model_average_20.pt`，beam size 5，canonicalized source。raw reward 为原始产物在 forward beam 中的 reciprocal rank；未命中为 0。

## Ranker 与参数

保持 v2 完全一致：

- 无泄漏 `correctness_features_noleak_v2`，228 维推理可得特征；
- 单层线性 head；
- residual：`raw + 0.25*tanh(linear(features))`；
- listwise：reaction-level listwise target + raw-score top-3 hard-negative pair loss；
- Adam，learning rate `1e-3`，seed 42，2000 steps；
- temperature `0.25`，margin `0.05`，pair weight `0.5`；
- residual regularization `0.01`，hard-negative `k=3`；
- invalid 候选不参与训练，评估时固定为最低分；
- 按 `(product_index, canonical candidate)` 去重，重复候选保留 raw reward 最大代表。

训练只读取 train 数据；validation 仅在两个冻结模型训练完成后查看；holdout 在模型、特征、loss 和训练步数冻结后只读取一次。

## 判定规则

v3 holdout 上，learned ranker 必须同时满足：

1. reaction-level Top-1 不低于 raw forward baseline；
2. Top-3 或 Top-10 至少一项提高；
3. Oracle 不下降，invalid 候选比例不增加超过 0.5 个百分点；
4. 以原始 reaction 为 bootstrap 单位时，Top-1 差值不显示明确负向；
5. 候选池、输入哈希、checkpoint 哈希和 forward reward 成本完整记录。

若两个 learned ranker 均不满足，正式关闭当前 rerank 路线：不构造 guidance data、不训练 DGM、不进行改进后 trajectory visualization。该结论只针对当前 ranker 设计与 reward 特征，不外推为所有可能的强 reranker 都无效。

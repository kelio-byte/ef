# Reward ranker v3 大规模复核报告

## 结论先行

在扩大到 `8000/1000/1000` 个独立 reaction 后，当前两个 learned reranker 仍显著低于冻结 Molecular Transformer raw forward reward 的 reaction-level Top-1：

- raw forward：`47.7%`；
- bounded residual：`38.1%`；
- listwise/hard-negative：`37.0%`。

residual 的 Top-1 相对 raw 下降 `9.6 pp`，1000-reaction paired bootstrap 95% CI 为 `[-13.0, -6.3] pp`；listwise 下降 `10.7 pp`，CI 为 `[-13.9, -7.5] pp`。因此当前 rerank 设计在更大数据上仍然失败，正式关闭这条路线：不重建 guidance data、不训练新的 DGM、不做改进后的 trajectory visualization。

这次结果支持的是一个限定结论：**当前小型线性、手工特征、endpoint-label reranker 不值得接入 DGM。**它不是对所有更强 reranker 的普遍否定。

## 1. 为什么扩大规模

此前 v2 使用 `1000/200/200` 个独立 reaction，用户指出这可能不足以判断小幅 rerank 差异。因此 v3 保持模型、特征、loss 和超参数不变，只扩大独立 reaction 数，以检验失败是否只是小 holdout 噪声。

## 2. 冻结协议与数据

### Split

| 角色 | reaction index | 独立 reaction | 原始记录数 | 去重后候选数 |
|---|---:|---:|---:|---:|
| train | 10000–17999 | 8000 | 160000 | 42812 |
| validation | 18000–18999 | 1000 | 20000 | 5402 |
| holdout | 19000–19999 | 1000 | 20000 | 5399 |

这些区间与已审计的旧 reward/guidance candidate artifacts 不重叠。每个 product 使用 100-step Euler、step `10/30/50/70/90` 五个 shared anchors、每个 anchor 4 条 continuation、seed 42；每个 product 产生 20 条原始记录。候选池固定后才用 dataset target 做离线标签。

生成时按 product token length 排序进入 batch，以减少 padding；每条记录保留绝对 `product_index`，因此排序不改变 split 和 reaction-level 聚合。该优化只改变执行顺序和 RNG batch 组织，不改变候选数量、模型、步数或 reward 定义。

基础 Edit Flows checkpoint：`new_checkpoints/checkpoint_step600000.pt`。

forward reward checkpoint：`new_checkpoints/MIT_mixed_augm_model_average_20.pt`，beam=5，canonicalized source，未命中为 0，命中第 r 名为 `1/r`。

候选文件 SHA-256：

- train validity：`57930156ddd397f8bda88f418dc3a60a61bc46845d5c8d5333491bccab943d56`；beam：`e46536cb6ba10decc650f9f1f4ecb44795efd231606d6e81c10345a7a73dadd6`；
- validation validity：`cb5f7fefc60a7ff3521f7fd15d6e8d88006e32cf38ef99d2f32617a7e3d0f5ee`；beam：`bf15a91d3a034a988831f48936ea0ebc423457f15056bd2ebe2a1b6120c1397f`；
- holdout validity：`402b834ee74f00a36ef7b6a0520e8c54cf323114ba6f93ed4a04df2d27667154`；beam：`0ccbda2a8fe116d8ae4eceab634afe49e3f9baf9a8a3fa06e9faaf2e09f96c9f`。

## 3. Reward 与候选池审计

冻结 forward reward 的结果：

| split | 原始记录 | forward hit rate | mean raw reward | shared-anchor groups |
|---|---:|---:|---:|---:|
| train | 160000 | 59.77% | 0.5474 | 40000 |
| validation | 20000 | 58.49% | 0.5383 | 5000 |
| holdout | 20000 | 59.83% | 0.5547 | 5000 |

holdout 去重后有 5399 个候选、684 个 invalid、804 个 positive candidate、1000 个 reaction；Oracle 为 `80.4%`。Oracle 是候选覆盖上限，三种排序读取相同候选池，所以 Oracle 不会因 rerank 改变。

## 4. Ranker 配置

严格复用 v2 的无泄漏配置：

- 228 维 inference-available product/candidate 特征；
- 单层线性 head；
- residual：`raw + 0.25*tanh(linear(features))`；
- listwise reaction loss + raw-score 前 3 个 hard negatives；
- Adam，learning rate `1e-3`，seed 42，2000 steps；
- temperature `0.25`、margin `0.05`、pair weight `0.5`、residual regularization `0.01`；
- invalid 不参与训练，评估时固定最低分；
- `(product, canonical candidate)` 去重，重复候选保留 raw reward 最大代表。

训练模式没有读取 holdout；validation 结果固定后才进行一次 holdout evaluation。

## 5. Validation 结果

validation 的 raw baseline 为 Top-1/3/5/10 `47.4/75.1/79.1/80.8%`。

| 排序 | Global valid AUC | Shared-anchor AUC | Top-1 | Top-3 | Top-10 |
|---|---:|---:|---:|---:|---:|
| raw forward | 0.6835 | 0.6493 | 47.4% | 75.1% | 80.8% |
| residual | 0.7128 | 0.6883 | 39.5% | 72.0% | 80.7% |
| listwise | 0.5617 | 0.6965 | 40.1% | 73.1% | 80.6% |

residual 的局部 AUC 上升没有转化为 endpoint Top-1；listwise 的 global AUC 和 endpoint 指标同时变差。validation 已足以说明当前 ranker 不像是“只需要更多训练步数”的问题，但仍按协议完成独立 holdout。

## 6. 大规模 holdout 结果

| 排序 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle |
|---|---:|---:|---:|---:|---:|
| raw forward | 47.7% | 74.4% | 79.2% | 80.4% | 80.4% |
| residual | 38.1% | 72.6% | 79.5% | 80.3% | 80.4% |
| listwise | 37.0% | 72.0% | 79.3% | 80.4% | 80.4% |

holdout valid-candidate AUC：

- raw：`0.6922`；residual：`0.7167`（`+2.45 pp`）；listwise：`0.5598`（`-13.24 pp`）；
- shared-anchor AUC：raw `0.6493`；residual `0.6894`；listwise `0.6843`。

因此 residual 仍然出现“局部 AUC 变好、Top-1 变差”的同一模式。1000-reaction paired bootstrap（5000 次，以原始 reaction 为统计单位）为：

| 方法相对 raw | Top-1 95% CI | Top-3 95% CI | Top-10 95% CI |
|---|---:|---:|---:|
| residual | `[-13.0, -6.3] pp` | `[-3.6, 0.0] pp` | `[-0.3, 0.0] pp` |
| listwise | `[-13.9, -7.5] pp` | `[-4.2, -0.7] pp` | `[0.0, 0.0] pp` |

两个 learned ranker 的 Top-1 下降都已明确排除“只是 200 个 reaction 的偶然波动”。Oracle 完全不变，说明 reranker 没有增加候选覆盖，只是在相同候选池中把正确候选排到了后面。

## 7. 关闭决定

当前 rerank 分支同时满足以下失败特征：

1. train 从 1000 扩大到 8000 reaction 后，validation/holdout 仍然 Top-1 下降；
2. holdout 从 200 扩大到 1000 reaction 后，Top-1 下降区间仍明确为负；
3. residual 虽能提高 candidate-level AUC，但不能完成 reaction-level first-rank selection；
4. listwise 直接破坏 raw forward 的有效排序；
5. Oracle 不变，说明问题不是 rerank 覆盖不足，而是候选选择目标/特征错配。

因此：

- 不再继续扩大当前 ranker 的训练/验证/holdout；
- 不在 holdout 上继续调参、换特征或挑 checkpoint；
- 不使用当前 ranker 构造 guidance data；
- 不训练新的 DGM；
- 不运行改进后的 `visualization_trajectory`，因为没有新的改进方法。

这条结果把当前研究方向从“再加一个小 reranker”明确移回两个更基本的问题：如何让 forward reward 与真实 reaction-level correctness 更一致，以及如何直接定义可用于 action-level guidance 的未来价值，而不是先做一个无法超过 Molecular Transformer 的 endpoint 校正器。

报告产物：

- train/validation report：`/root/autodl-tmp/dgm_ranker_v3_large_runs/ranker_v3/train_validation_report.json`；
- holdout report：`/root/autodl-tmp/dgm_ranker_v3_large_runs/ranker_v3_holdout/holdout_report.json`。

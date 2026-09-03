# DGM reward-ranker v2 执行报告

## 结论先行

本轮按预注册的 reaction-level gate 完成了一个新的、无 target leakage 的 reward-ranker 分支。它比较了：

1. 冻结 forward beam 的 raw reciprocal-rank；
2. `raw + bounded learned residual`；
3. listwise + hard-negative learned ranker。

两个 learned ranker 在独立 holdout 上都没有通过终点 Top-1 gate：raw Euler reward 为 `46.5%`，residual 为 `40.0%`，listwise 为 `38.0%`。因此没有构造新的 guidance data，没有训练新的 DGM，也没有读取 confirm/final/test 集合。当前最可靠的判断是：候选级局部排序信号不足以支持 reaction-level rerank，更不能直接作为 action-level guidance 的训练目标。

## 1. 重要勘误：旧 P2 结果不再作为证据

此前的 correctness-reward 脚本把 `canonical_targets[product_index]` 误传为 product feature 的来源，泄漏了 target 的组分数。该实现会让模型通过一个看似无害的 product-component 特征间接看到答案，因此旧 P2 的 AUC、旧 P3 rerank 和据此作出的正向判断都不能作为干净证据。

本轮先修复为：product feature 只从候选记录中序列化的 `product_tokens` 解码，target 只用于离线构造训练/统计标签；修复提交为 `532a46e`。新模型使用 `correctness_features_noleak_v2`，不复用旧 checkpoint。

## 2. 冻结协议与数据

### 数据划分

使用训练集原始 reaction index 的未使用区间，避免从旧的 1000–1399 分支选择方法：

| 用途 | reaction index | 产品数 | 原始记录数 |
| --- | ---: | ---: | ---: |
| ranker train | 2000–2999 | 1000 | 20,000 |
| ranker validation | 3000–3199 | 200 | 4,000 |
| 一次性 holdout | 3200–3399 | 200 | 4,000 |

每个产品在第 10、30、50、70、90 个 Euler step 取 shared anchor；每个 anchor 独立续采样 4 条，故每个产品 20 条终点候选。基础采样固定为 `checkpoint_step600000.pt`、100 Euler steps、seed 42、cubic scheduler。forward reward 固定为冻结 Molecular Transformer 的 beam=5 reciprocal rank，且不读取 target。

候选池固定后，才用数据集真实反应物做离线标签：canonical(candidate) 等于 canonical(target) 为正例；invalid 不参与训练，评估时固定为最低分。每个 `(product, canonical candidate)` 只保留 raw forward reward 最强的一条代表。

新协议和哈希见 [`dgm_reward_ranker_v2_protocol.md`](dgm_reward_ranker_v2_protocol.md)。候选文件哈希为：

- train：`f7a0204b...f4de24`
- validation：`4e096324...0b435dc`
- holdout：`5eeefde8...306ddd`

附加 forward reward 后，raw 候选的 hit rate 为 train `57.58%`、validation `60.23%`、holdout `57.05%`（这是 beam 命中率，不是 reaction-level Top-1）。

## 3. 训练的两个 ranker

两者都是单层线性 head，输入是推理时可获得的 228 维特征：raw forward reciprocal rank、validity、产物/候选长度及长度差、token histogram、histogram 差分，以及从序列化 product/candidate 解码的组分统计。训练只使用 train split；validation 用于训练后比较，holdout 在训练模式中没有打开。

固定参数：

- Adam，learning rate `1e-3`，seed `42`，`2000` steps；
- listwise temperature `0.25`；pairwise margin `0.05`、pair weight `0.5`；
- residual cap `0.25`，residual regularization `0.01`；
- 每个 reaction 的 raw-score 前 `3` 个 hard negatives。

### residual ranker

打分为：

`score = raw_forward_reward + 0.25 * tanh(linear(features))`。

它被限制为只能对 raw 排序做小幅修正，避免一个低质量的 learned score 完全推翻已有 forward reward。

### listwise / hard-negative ranker

在每个 reaction 候选组内，对正例分配 listwise target，并额外要求正例胜过 raw reward 最强的若干负例；它不受 residual cap 保护，因此是更激进的对照。

训练产物：

- residual checkpoint：`/root/autodl-tmp/dgm_ranker_v2_runs/ranker_v2/ranker_residual.pt`，SHA-256 `4a4f2e99...4920864`；
- listwise checkpoint：`/root/autodl-tmp/dgm_ranker_v2_runs/ranker_v2/ranker_listwise.pt`，SHA-256 `d3f1e394...c7a7e8`。

## 4. validation：局部指标改善并没有转化为终点排序

validation 的 raw baseline 为 Top-1/3/10 `51.5% / 79.5% / 83.0%`。

| 排序 | valid-candidate global AUC | shared-anchor AUC | Top-1 | Top-3 | Top-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw forward | 0.7118 | 0.6557 | 51.5% | 79.5% | 83.0% |
| bounded residual | 0.7474 | 0.6766 | 41.0% | 76.5% | 83.0% |
| listwise/hard-negative | 0.6163 | 0.7036 | 40.5% | 74.0% | 82.5% |

residual 的全局 AUC 比 raw 高约 3.56 个百分点，组内 AUC 高约 2.10 个百分点，但 Top-1 下降 10.5 个百分点。listwise 的组内 AUC 虽更高，global AUC 和所有主要 Top-k 都更差。这已经说明“局部 pair/AUC 提高”与“每个 reaction 把正确候选放到第一名”不是同一个目标。

## 5. 一次性 holdout：两个 learned ranker 均未通过 gate

holdout 去重后有 1,025 个候选、200 个 reaction、133 个 invalid，Oracle 为 `78.5%`。三种排序使用完全相同的候选池，因此 Oracle 相同；任何差异都来自排序，而不是覆盖率。

| 排序 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw forward reward | 46.5% | 72.5% | 77.0% | 78.5% | 78.5% |
| bounded residual | 40.0% | 71.5% | 78.0% | 78.5% | 78.5% |
| listwise/hard-negative | 38.0% | 70.5% | 76.0% | 78.0% | 78.5% |

valid-candidate AUC 的 holdout 对照为：

- raw：`0.6955`；residual：`0.6985`（`+0.30 pp`）；listwise：`0.6336`（`-6.19 pp`）；
- shared-anchor AUC：raw `0.6667`，两者均为 `0.7222`。

但 reaction-level paired bootstrap（5,000 次、以原始 reaction 为统计单位）给出的 Top-1 差值 95% CI 为：residual `[-13.0, 0.0]` pp，listwise `[-15.5, -1.5]` pp。residual 的 point estimate 已下降 `6.5` pp，且没有达到“Top-1 不下降并有深层收益”的 gate；listwise 更明确失败。invalid 比例与候选覆盖没有发生可解释的改善。

holdout 报告：`/root/autodl-tmp/dgm_ranker_v2_runs/ranker_v2_holdout/holdout_report.json`，SHA-256 `c6765397...c78b1502`。

## 6. 研究判断与停止决定

这次结果不是“DGM 还没有训练，所以看不出效果”。在进入 DGM 之前，候选池内的 endpoint rerank 已经失败：

1. raw forward reward 本身仍是这组候选的最佳已测排序基线；
2. residual 学到了一点候选级局部信号，但信号不足以稳定识别 reaction-level 第一名；
3. 更激进的 listwise/hard-negative 目标反而破坏了 raw reward 已有的排序；
4. Oracle 不变，说明 learned ranker 没有增加候选覆盖，只是在重排已有候选。

因此当前最可能的瓶颈是 reward/特征与 reaction-level selection 的错配，而不是 DGM 优化器、guidance 强度或训练步数。这个实验不能单独区分 reward 表达、candidate-group loss、shared-anchor 样本量和 label 噪声各自的因果贡献；但它足以否定“把当前 ranker 直接接到 action-level DGM 就可能提升”的工程路径。

本轮明确未做：

- 没有用这两个 ranker 生成 guidance data；
- 没有训练或重训 DGM；
- 没有在本 holdout 上继续调特征、阈值、seed 或挑 checkpoint；
- 没有读取 confirm、final 或完整 test；
- 没有进行改进后 paired `visualization_trajectory`，因为改进分支未通过进入条件。

代码测试：`pytest -q tests/test_correctness_reward.py` 为 `3 passed`，`scripts/train_reward_ranker_v2.py` 通过 `py_compile`。

## 7. 下一步含义

如果继续研究，下一条路线不应是直接重训 DGM，而应先提出一个以 reaction-level Top-1/Top-k 为直接目标、并在新独立 holdout 上验证的候选组学习方案。只有新排序器同时满足候选覆盖不下降、Top-1 不下降且至少一个更深 Top-k 改善，才有理由把它转成 action-level guidance；否则应把失败定位在 reward/候选选择层，而不是继续扩大 DGM 实验。


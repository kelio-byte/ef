# SPE 总结

更新：2026-08-20 UTC。

这份文档回答三个实际问题：fragment-level edit 是否有效、应选哪种 tokenizer/checkpoint、后续训练或推理改进该从哪里开始。它统一已有记录的展示口径，不替代原始结果文件。

## 1. 先给结论

- 当前最值得继续的分支是：**改进后 global R-SMILES + SPE-M500**。
- 后续推理改进的主 baseline：**SPE-M500@490K + 普通 Euler N=9、100 steps、cubic**。它在三 seed 复评中拥有最好的 Top-1/3/5 和 Oracle 均值。
- `M500@500K` 保留为辅助 checkpoint：它的 Top-10 和 Invalid@1 略好，但不能替代 490K 作为主 baseline。
- Full-SPE 的主要价值是序列压缩/速度；已有多轮训练和 checkpoint sweep 均显示它弱于 M500，暂不继续作为主训练分支。
- 改进前 R-SMILES 的 M500 外机结果明显较低，但还缺少完整 artifacts 的本机核验；正在补做无裁剪的原始 R-SMILES Atom/Full 对照，不能把当前差距简单归因于 SPE。

## 2. 名称、数据集与统一评估口径

| 名称 | R-SMILES 表示 | Tokenizer | 数据目录 | 当前用途 |
|---|---|---|---|---|
| Atom | 改进后 global（`#global#`） | atom-level | `USPTO_50K_PtoR_aug20_#global#` | 历史 atom baseline |
| Full-SPE | 改进后 global | 全部 3,002 条 merge rule | `..._#global#_SPE` | 已完成，不再主推 |
| M500 | 改进后 global | 前 500 条 merge rule | `..._#global#_SPE_m500` | 当前主分支 |
| 原始 M500 | 改进前 | 前 500 条 merge rule | `USPTO_50K_PtoR_aug20_SPE_m500` | 表示对照，外机结果待核验 |

除特别注明外，所有正式 checkpoint 选择均使用对应 tokenizer 的 `evaluation_v2/dev_unique1000_aug20`：1,000 个 reaction block、每个 20 个 augmentation 输入（共 20,000 行），Euler N=9、100 steps、cubic、batch=32、seed=42、Top-10 打分。表内数值均为百分比；`Invalid@1` 越低越好。

`#global#` 是改进后的 R-SMILES 表示，不是额外的 sampler 或模型技巧。它先改善产物与反应物的编辑对齐；SPE 再在该表示上合并 token。

## 3. 为什么从 Full-SPE 转向 M500

先做 merge-depth 数据分析，是为了在“序列更短”和“单个 token 更难预测”之间找到平衡，而不是默认 merge 越多越好。

| 改进后 global 表示 | merge 数 | 真实词表 | 平均对齐长度 | 平均编辑距离 | 平均编辑密度 | SUB 占编辑比例 |
|---|---:|---:|---:|---:|---:|---:|
| Atom | — | 69 | 50.81 | 5.74 | 11.76 | 7.88 |
| SPE-M500 | 500 | 568 | 16.05 | 4.13 | 27.02 | 32.05 |
| SPE-M1000 | 1,000 | 1,066 | 14.26 | 4.05 | 29.62 | 34.11 |
| SPE-M2000 | 2,000 | 2,056 | 12.72 | 3.87 | 31.87 | 36.76 |
| Full-SPE | 3,002 | 3,035 | 12.06 | 3.86 | 33.49 | 37.12 |

**分析。** Full-SPE 只比 M500 再缩短约 4 个 token，却将词表扩大约 5.3 倍，并继续提高编辑密度和 fragment-SUB 比例。M1000/M2000 只完成数据构造与审计，没有训练结果。

**结论。** M500 是当前唯一同时具备显著序列压缩、可接受 token 分类难度、且已完成充分训练验证的中等 merge 点。

## 4. 已完成实验

### 4.1 Full-SPE：验证“更短”是否足够带来更好结果

**为什么做。** Full-SPE 是最直接的序列压缩方案。先验证它能否在保留准确率的同时降低采样成本；随后改变 batch/训练步数，排除“初始训练不足”这一解释。

| 表示与训练 | checkpoint | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid@1 | 采样时间 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Atom（B128） | 600K | 56.3 | 76.5 | 80.3 | 83.9 | 90.2 | 11.740 | 58.21 min |
| Full-SPE 初版（B128） | 600K final | 50.0 | 68.7 | 73.3 | 77.2 | 87.2 | 22.275 | 24.40 min |
| Full-SPE 重训（B256） | 470K | 56.4 | 73.3 | 78.0 | 81.0 | 88.0 | 16.490 | 23.85 min |
| Full-SPE 重训（B256） | 500K | 55.2 | **73.6** | **79.0** | **81.5** | 88.1 | 17.755 | 24.35 min |
| Full-SPE 重训（B256） | 600K | 55.5 | 72.4 | 77.3 | 81.4 | 88.8 | **15.400** | 24.17 min |
| Full-SPE 600K→800K continuation（B128） | 800K final | 51.5 | 71.3 | 74.8 | 78.4 | 86.8 | 20.470 | 24.68 min |

**分析。** 初版 Full-SPE 确实约快 2.4 倍，但 Top-K 和有效性明显变差。B256 重训改善了 Top-K/Invalid@1，却没有出现稳定的后期增益；继续到 800K 也未解决这个问题。

**结论。** Full-SPE 是有效的效率基线，但不是当前最好的准确率/覆盖率选择。后续不再扩大 Full-SPE 的 checkpoint sweep，也不把“再多训练一些步数”作为主假设。

### 4.2 改进后 global R-SMILES 的 SPE-M500：checkpoint 选择

**为什么做。** M500 在数据层面保留了大部分压缩收益，同时显著弱化了 Full-SPE 的高词表/高编辑密度代价。需要以同一 dev 集和普通 Euler 采样确定可复用 checkpoint，而不是用 validation loss 代替生成指标。

#### 单 seed 密集 sweep：Euler N=9，seed=42

| step | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid@1 |
|---:|---:|---:|---:|---:|---:|---:|
| 140K (`checkpoint_best`) | 55.4 | 73.4 | 78.0 | 81.2 | 89.6 | 15.615 |
| 150K | 55.8 | 73.3 | 78.0 | 82.0 | 89.7 | 15.300 |
| 200K | 57.7 | 74.4 | 78.6 | 81.6 | 89.5 | 14.240 |
| 250K | 57.7 | 73.9 | 78.4 | 83.0 | 89.8 | 14.255 |
| 300K | 57.4 | 74.7 | 80.4 | 83.7 | 89.2 | 13.170 |
| 460K | 58.3 | **76.7** | 80.0 | 83.6 | 89.4 | 13.030 |
| 470K | 58.9 | 75.6 | 80.0 | 84.0 | 89.6 | 13.415 |
| 480K | 58.6 | 75.1 | 79.2 | 83.6 | 89.6 | 12.460 |
| 490K | **60.1** | 76.6 | **80.5** | 83.7 | **90.0** | 12.850 |
| 500K | 59.9 | 75.9 | 80.1 | 84.3 | 89.7 | 12.245 |
| 510K | 58.2 | **76.8** | 80.4 | 83.2 | 89.1 | 14.265 |
| 520K | 58.3 | 76.5 | 79.8 | **84.7** | 89.9 | 12.840 |
| 530K | 58.6 | 76.5 | 80.2 | **84.7** | 89.7 | 12.950 |
| 540K | 58.1 | 75.9 | 80.0 | 83.8 | 89.4 | 13.995 |
| 600K | 58.1 | 77.2 | 80.4 | 83.7 | 89.0 | **11.985** |

**分析。** 生成质量不会随 step 或 validation loss 单调变化：490K 偏 Top-1/Top-5/Oracle，520–530K 偏 Top-10，600K 偏 validity。单 seed 不足以在 490K 和 500K 之间做最终选择。

#### 配对三 seed 复评：最终 checkpoint 决策证据

两点均在相同 dev block、Euler N=9、seed `42/7/123` 下复评；下表是均值 ± sample SD。

| checkpoint | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid@1 |
|---|---:|---:|---:|---:|---:|---:|
| **M500@490K** | **59.27 ± 0.74** | **76.40 ± 0.20** | **80.63 ± 0.81** | 83.67 ± 0.65 | **89.90 ± 0.66** | 12.65 ± 0.18 |
| M500@500K | 58.90 ± 0.87 | 76.13 ± 0.49 | 80.13 ± 0.15 | **83.97 ± 0.35** | 89.83 ± 0.15 | **12.47 ± 0.19** |

**分析。** 490K 的 Top-1/3/5/Oracle 均值更好；500K 仅在 Top-10 和 Invalid@1 略占优，且差距与 sampling 波动同量级。单样本 N=1 也支持把 460–490K 视为强区间，但其最优点与 N=9 不完全一致，不能替代 N=9 的 checkpoint 选择协议。

**结论。** 以 Top-1/Top-5 和覆盖为主要目标时，选 M500@490K；若任务只关注 Top-10/有效性，保留 M500@500K 作为次级对照。

### 4.3 M500@490K 的 sampler 对照：普通 Euler 还是 R9K1M2

**为什么做。** 表示方法和采样策略是两个变量。先固定最强 M500 checkpoint，判断结构化 Euler-Beam 是否值得作为后续 sampler 分支。下表为单 seed（42）结果。

| sampler | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid@1 |
|---|---:|---:|---:|---:|---:|---:|
| Euler N=9 | **60.1** | 76.6 | 80.5 | **83.7** | 90.0 | 12.850 |
| R9K1M2 | 60.0 | **77.3** | **80.7** | 83.6 | **90.4** | **12.395** |

**分析。** R9K1M2 改善 Top-3/5、Oracle 和有效性，但未提升 Top-1/Top-10；它不是表示方法优越性的证据。

**结论。** 普通 Euler N=9 保持为 M500 表示和未来方法改动的主对照；R9K1M2 是独立的 sampler 候选，后续只在固定 M500@490K 上比较。

### 4.4 改进前 R-SMILES + M500：为什么不能直接归因于 tokenizer

**为什么做。** 如果 M500 只在改进后 R-SMILES 上有效，收益可能来自 global alignment，而不是 fragment tokenization。因此在原始 R-SMILES 上训练 M500 做表示对照。

数据审计显示，原始 M500 虽更短，却明显更难编辑：平均编辑密度 `44.24%`（改进后为 `26.93%`），KEEP 比例 `55.13%`（改进后为 `74.28%`），SUB 占编辑 `47.1%`（改进后为 `32.1%`）。OOV、SPE round-trip、行错位和 `max_seq_len=96` 截断均已排除。

外机报告的 Euler N=9、seed=42 checkpoint sweep 如下：

| step | Top-1 | Top-3 | Top-5 | Top-10 | Oracle |
|---:|---:|---:|---:|---:|---:|
| 300K | 42.4 | 60.5 | 64.0 | 65.9 | 81.7 |
| 450K | 43.5 | **62.0** | **65.1** | 66.6 | 82.9 |
| 490K | 41.4 | 59.8 | 63.4 | 64.7 | 81.5 |
| 500K | 42.6 | 60.7 | 64.2 | 65.5 | 81.7 |
| 550K | **45.5** | 61.5 | 64.9 | **66.7** | **83.4** |
| 600K | 43.8 | 61.9 | 64.9 | 66.0 | 82.8 |

**分析。** 原始 M500 的最好点仍明显低于改进后 M500；Oracle 也低，说明问题不只是最终排序，而是目标覆盖不足。数据结构与该现象相符：高密度 fragment-SUB 使 Edit Flows 更难生成正确候选。

**结论与边界。** 这是一条强信号，不是最终因果结论：该 sweep 在另一台机器完成，完整 manifest、diagnostics、训练日志和 result artifacts 尚未全部迁回。当前正在补做无梯度裁剪的原始 R-SMILES Full-SPE 与 Atom-level 对照；完成同 protocol 比较前，不把“原始 R-SMILES 上 SPE 无效”写成定论。

### 4.5 探索性实验：不用于 checkpoint 决策

| 实验 | 为什么做 | 结果 | 结论 |
|---|---|---|---|
| Full-SPE 50K pilot | 验证预处理、训练、采样链路 | 链路正常；训练预算太小，且早期 sampler 有 action-support 混杂 | 仅作 smoke，不参与性能比较 |
| Full-SPE 60–200 Euler steps sweep | 判断 100 steps 是否明显不足 | 在 100 reaction block / 单 seed 上没有单调准确率或 invalid 改善 | 不把“增加推理步数”作为 Full-SPE 的主要修复方向 |
| M500 的 Euler N=1 sweep | 判断单轨迹好坏能否替代 N=9 选型 | N=1 的最佳 Top-1 在 480K，Oracle 最佳在 490K；与 N=9 不完全一致 | N=1 只作诊断，主协议仍是 N=9 |
| SPE-M1000/M2000 | 探索更深 merge | 仅有完整数据审计，无训练 | 在 M500 改进停滞前，不扩展新的 merge 深度 |

## 5. 总体结论与下一步

### 总体结论

1. **SPE 的价值依赖表示。** 改进后的 global alignment 先把原子/fragment 编辑变得稀疏、以保留和插入为主；M500 才能在此基础上取得较强结果。
2. **M500 是当前的平衡点。** 它把平均对齐长度从 50.81 缩到 16.05，同时避免了 Full-SPE 的超大词表和更高 fragment-SUB 比例。
3. **Full-SPE 的速度收益真实，但质量不足。** 多轮训练后仍未成为 M500 的竞争者，因此不继续消耗主要训练预算。
4. **生成指标而非 validation loss 决定 checkpoint。** M500@490K 与 @500K 的例子说明，必须在冻结 dev 协议下用 Top-K/Oracle/Invalid 选择。

### 后续训练后推理改进的 baseline

| 项目 | 固定选择 |
|---|---|
| 数据与表示 | 改进后 global R-SMILES，`USPTO_50K_PtoR_aug20_#global#_SPE_m500` |
| 模型 checkpoint | **SPE-M500@490K** |
| 主 sampler | 普通 Euler，N=9，100 steps，cubic |
| 主验证集 | `dev_unique1000_aug20`，1,000 reactions / 20,000 augmentation rows |
| 主指标 | Top-1、Top-3、Top-5、Top-10、Oracle、Invalid@1、unique candidates、运行时间 |
| 稳健性确认 | 先 seed=42；若改动接近或超过 baseline，再补 seed=7/123 |
| 次级对照 | M500@500K（Top-10/Invalid 取向）与 R9K1M2（sampler 改动） |

### 推荐顺序

1. 在 M500@490K 上做独立、低成本的推理/采样改进；与普通 Euler N=9 完全同协议比较。
2. 只有 dev 上的收益在额外 seed 下仍存在，才冻结改动和 checkpoint。
3. 随后做一次完整 `src-test`/`tgt-test` 评估；不要用完整 test 继续调参。
4. 并行完成原始 R-SMILES 的无裁剪 Atom/Full 控制，回答“global alignment 的贡献”和“fragment tokenization 的贡献”这两个不同问题。

## 6. 证据来源

- [SPE_experiment_summary.md](SPE_experiment_summary.md)：历史完整总览与原始结果索引。
- [spe_m500_checkpoint_comparison.md](spe_m500_checkpoint_comparison.md)：M500 的完整 N=9/N=1/MaxFrag/R9K1M2 明细。
- [ori_rsmiles_spe_m500_evaluation.md](ori_rsmiles_spe_m500_evaluation.md)：外机原始 R-SMILES M500 报告。
- [SPE_dev1000_checkpoint_evaluation.md](../new_docs/SPE_dev1000_checkpoint_evaluation.md)：本机 dev1000 的可复核结果、诊断和多 seed 复评。
- [RSMILES_SPE_M500_four_way_dataset_audit.md](../new_docs/RSMILES_SPE_M500_four_way_dataset_audit.md)：原始/改进 R-SMILES × Atom/M500 的数据机制审计。
- [SPE_prefix_rules_experiment.md](../new_docs/SPE_prefix_rules_experiment.md)：K=500/1000/2000/full 的构造与长度/编辑统计。

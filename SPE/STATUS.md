# SPE 阶段状态与后续交接

更新：2026-08-23

## 一句话结论

**SPE 的表示选择与基线建立阶段可以暂告一段落。** 后续改进默认以“改进后的 global R-SMILES + SPE-M500”作为 fragment-level 基线；Full-SPE 和改进前 R-SMILES 的 SPE 分支不再作为主线。
这不表示 SPE 已经没有可改进之处，而是表示：当前已有足够证据停止继续搜索 tokenizer 规则，转向改进采样、排序和训练/推理策略。

## 已冻结的默认设置

| 项目 | 当前默认选择 | 说明 |
|---|---|---|
| 表示 | 改进后的 global R-SMILES | 数据集名带 `#global#` 后缀 |
| Tokenizer | SPE-M500 | 只采用前 500 条 merge 规则 |
| 主要 checkpoint | `spe_m500_checkpoints/checkpoint_step490000.pt` | 在 dev 上预先选择的主 checkpoint |
| 辅助 checkpoint | `checkpoint_step500000.pt` | 仅在关注 Top-10 / invalid 时作为敏感性对照，不替代主 checkpoint |
| 默认推理 | Euler，N=9，100 steps，seed=42 | 使用与现有 dev 对比一致的协议；详细参数以实验命令为准 |
| 主开发集 | `datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/evaluation_v2/dev_unique1000_aug20/` | 1,000 个反应，20 个 augmentation/反应，共 20,000 行 |

`R9K1M2` 是值得保留的**采样器对照**，不是替代 M500 表示基线的另一种 tokenizer。后续若改采样或重排序，应首先和 M500 + Euler N=9 比较。

## 已得到的可靠结论

### 1. M500 是当前最合理的 SPE 粒度

Full-SPE 虽然大幅缩短序列，但词表明显变大、编辑密度上升，尤其 SUB 占比升高；训练更久后也没有恢复到 M500 的效果。M500 在压缩序列、控制词表规模和降低编辑难度之间取得了更好的平衡。

因此，**不要再把 Full-SPE 作为主要 tokenizer 方案继续调参**；除非未来有一个专门针对大词表/高替换率的新机制。

### 2. 改进后的 global R-SMILES 是 SPE 成功的重要前提

同为 M500，改进前 R-SMILES 的数据编辑难度显著更高：

| 数据表示 | 平均对齐长度 | 平均编辑数 | 编辑密度 | SUB 占比 |
|---|---:|---:|---:|---:|
| 改进前 R-SMILES + M500 | 15.24 | 6.84 | 44.24% | 47.1% |
| global R-SMILES + M500 | 16.05 | 4.13 | 26.93% | 32.1% |

改进前表示上的 atom-level、M500、Full-SPE 实验整体都明显较弱。现有证据支持：性能提升不能简单归因为“用了 SPE”，更合理的说法是 **global 对齐降低了编辑任务难度，M500 才能有效利用这种对齐**。

### 3. M500 的优势主要在 Top-1 与实用效率，不是所有指标全面超过 atom-level

在开发集的代表性结果中，M500@490K 的 Top-1 很强；但 atom-level 在 Oracle 或某些较大的 Top-K 指标上可略高。故正确表述是：

> M500 是当前更好的实用 fragment-level baseline，尤其适合作为提高首选预测质量和推理效率的起点；不能表述为它在所有指标上严格支配 atom-level。

### 4. M500 更少发生“明显有害的首个编辑”，但尚未证明它更会事后纠错

已在 global 数据的 1,000 个反应、全部 20 个 augmentation、seed=42 上进行轨迹诊断，并做了三 seed 的受控干预验证。

| 诊断 | Atom | M500 | 结论 |
|---|---:|---:|---|
| 自然轨迹中有害首事件比例 | 25.86% | 18.74% | M500 低 7.12 个百分点；20 个 augmentation 方向一致 |
| 最终 canonical 命中率（逐路径） | 35.52% | 36.55% | M500 略高，但置信区间跨 0，证据不足以称为稳定提升 |
| 强制首步为有害 completion 后的命中下降 | 52.83 pp | 50.39 pp | 两者都会严重崩溃；M500 没有表现出强的后续纠错能力 |

所以原始动机应修正为：**M500 的主要收益目前更像“降低早期走错概率”，而不是“走错后更容易修回来”。**

详细过程与原始统计见 `revision/motivation_report.md` 和 `revision/results/augmentation_robustness/summary/summary.md`。

## 已报告的全量测试：怎样解读

`测试实验数据.md` 记录了外部机器上全量测试的 M500 checkpoint 扫描。按其中数字，M500 在 Top-1/Top-3 上优于 Atom@600K，例如：

| 结果来源 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle-any | Invalid |
|---|---:|---:|---:|---:|---:|---:|
| Atom@600K | 58.798% | 78.590% | 83.084% | 86.739% | 92.510% | 12.375% |
| M500@510K（Top-1 最好） | 62.113% | 78.810% | 82.345% | 85.380% | 90.613% | 13.375% |
| M500@500K（Top-3 最好） | 61.674% | 79.049% | 82.425% | 85.700% | 90.713% | 11.749% |
| M500@600K（invalid 最低） | 61.354% | 78.770% | 82.485% | 85.780% | 90.853% | 11.463% |

这组数据支持“**M500 提升 Top-1，但 Atom 的候选覆盖/Oracle 仍有优势**”这一判断。

但要严格保留一个边界：这次全量测试已经扫过多个 checkpoint，因此它**不再是完全未参与模型选择的独立隐藏测试集**。今后不要再根据这张全量测试表挑 checkpoint 或继续调参。若需要正式的最终泛化结论，应冻结一个方案后，在未被反复查看的新 holdout 上只评一次。

此外，外部机器的原始结果文件未全部同步到本机；本段仅记录 `测试实验数据.md` 中已汇总的数字。任何论文级结论前，应补齐命令、sampler、seed、checkpoint SHA256 和原始预测文件的可追溯记录。

## 不应再混淆的边界

1. `dev_unique1000_aug20` 是从 **val** 的 20 倍 augmentation 行中按 global manifest 抽取的 1,000 个反应，不是 test 集。它适合开发、筛 checkpoint、做诊断。
2. SPE-M500 的主 checkpoint 是 **490K**，因为它是在 dev 选择流程中确定的；全量测试中看起来更好的 500K/510K 不能反过来改写这一选择。
3. 不要把“global R-SMILES 的提升”全部归因于 SPE；global 对齐本身显著改变了编辑数量和编辑类型分布。
4. 不要把自然轨迹里的“首事件距离变差”理解为严格的化学正确/错误标签。它只是可复现的代理指标；受控有害 completion 实验才更直接检验了纠错能力。
5. 原始 R-SMILES 的三组实验说明当前表示不利于 edit flow，但其训练/评估原始产物主要在外部机器；它们是方向性证据，不应单独承担严格的硬件或随机性归因。

## 后续改进应从哪里开始

后续 AI/实验应将下列方案视为**固定 baseline**：

```text
数据：USPTO_50K_PtoR_aug20_#global#_SPE_m500
模型：SPE-M500@490K
开发评估：dev_unique1000_aug20
采样：Euler N=9, 100 steps, seed=42
```

优先方向：

1. **推理阶段的候选重排序或多样化**：目标是补足 M500 的 Oracle/高 Top-K 劣势，同时不能牺牲 Top-1。必须完全 target-free。
2. **首步/早期编辑的风险控制**：因为诊断显示主要问题是早期有害编辑，而不是缺少事后纠错。可做小范围、可独立消融的 mode/token 选择或分支策略。
3. **纠错监督或训练目标的改造**：若要继续原动机，应明确构造“已发生可控错误后如何恢复”的监督；仅依赖自然轨迹不足以证明这一点。
4. **采样器比较**：R9K1M2 可以作为同 checkpoint 的对照。若外部全量 R9 评估完成，应先核对实际 checkpoint、seed 和命令，再纳入正式表格。

每个新想法先在 dev 上以与 baseline 完全相同的协议比较；只有在 seed=42 有清晰收益后，再补 seed=7/123。不要重新开启 Full-SPE 或改进前 R-SMILES 的大规模超参搜索，除非新方法明确针对它们的高编辑密度问题。

## 推荐阅读顺序

1. 本文档：当前边界与默认基线。
2. `0822.md`：面向汇报的动机和故事线。
3. `测试实验数据.md`：各 checkpoint 的汇总测试数字。
4. `spe_m500_checkpoint_comparison.md`、`SPE_experiment_summary.md`、`summary.md`：历史 checkpoint 与开发集实验细节。
5. `orig_rsmiles_dataset_statistics.md`、`ori_rsmiles_spe_m500_evaluation.md`：为什么改进前 R-SMILES 不宜作为主线。
6. `revision/motivation_report.md`：SPE 是否能纠错的诊断结论。

历史文档中可能保留了当时尚未获得全量结果时的表述；若与本文冲突，以本文的“边界”和“已冻结设置”为准。

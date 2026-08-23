# Fragment-level Tokenizer：从 Full-SPE 到 M500

更新：2026-08-21。本文用于汇报 fragment-level tokenization 在 Edit Flows 逆合成中的动机、实验路径、结论与边界；所有关键数值均已按当前记录更新。

## 1. 要解决什么问题？

当前 Edit Flows 直接在 atom-level SMILES token 上执行插入（INS）、删除（DEL）和替换（SUB）。在改进后的 global R-SMILES 数据上，Atom 表示的 product/source 平均长度为 45.57，reactant/target 平均长度为 50.76，平均对齐长度为 50.81。

更关键的是，Atom-level 编辑高度偏向插入：91.20% 的编辑是 INS，而 SUB 只有 7.88%。这意味着模型往往需要连续补出多个细粒度 atom token；一旦早期补错，后续能否纠正依赖训练中很少出现的 SUB/DEL。

| Tokenizer | Src mean | Tgt mean | Aligned mean | Mean edit | INS | DEL | SUB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Atom-level | 45.573 | 50.756 | 50.809 | 5.741 | 91.20% | 0.92% | 7.88% |

因此提出的假设是：

> 能否将连续的 atom token 合并为 fragment token，使序列更短，并把一部分“连续插入”改写为局部的 fragment SUB，从而提高采样效率和候选质量？

这只是一个待验证的假设。fragment 更大也意味着每次 INS/SUB 的分类更难，不能仅凭序列变短就推断效果会更好。

## 2. 统一实验口径

除特别说明外，下文的性能结果使用对应 tokenizer 的 `evaluation_v2/dev_unique1000_aug20`：1,000 个相同 reaction block、每个 block 20 个 augmentation 输入，共 20,000 行。采样固定为普通 Euler、`N=9`、100 steps、cubic scheduler、seed=42、推理 batch size=32，并按 reaction block 聚合 Top-K。

- `Invalid@1` 越低越好；`Oracle` 表示 9 条轨迹中是否至少覆盖到目标。
- Atom、Full 和 M500 各自使用同一批原始 reaction block 的 tokenizer 投影文件，因此可以比较生成结果。
- Atom 历史基线训练 batch size=128、`max_seq_len=256`；本轮 Full/M500 训练 batch size=256、`max_seq_len=96`。因此 Atom 是重要的实用基线，但 M500 vs Full（同为 B256）的比较才是更干净的 tokenizer 对照。

## 3. 初步尝试：Full-SPE 确实压缩了序列，但没有直接解决生成问题

首先使用 `SPE_ChEMBL.txt` 的全部 3,002 条 merge rule，得到 Full-SPE。它显著缩短序列并减少绝对编辑数：

| 指标 | Atom-level | Full-SPE | 变化 |
|---|---:|---:|---:|
| Src mean length | 45.573 | 9.655 | ↓ 78.8% |
| Tgt mean length | 50.756 | 12.034 | ↓ 76.3% |
| Mean edit | 5.741 | 3.857 | ↓ 32.8% |
| Real vocab | 69 | 3,035 | ↑ 44× |

初版 Full-SPE（B128、600K）在 dev-1000 上的采样时间为 24.40 min，而 Atom@600K 为 58.21 min，约快 2.4×；但 Top-1/Top-10 只有 50.0%/77.2%，Invalid@1 达 22.275%。

随后以 B256 重训并评估多个后期 checkpoint，排除“只训练不够久”这一解释：

| Full-SPE（B256） | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid@1 | 采样时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 400K | 56.3 | 73.2 | 77.4 | 80.4 | 87.6 | 19.885 | 23.65 min |
| 470K | **56.4** | 73.3 | 78.0 | 81.0 | 88.0 | 16.490 | 23.85 min |
| 500K | 55.2 | **73.6** | **79.0** | **81.5** | 88.1 | 17.755 | 24.35 min |
| 600K | 55.5 | 72.4 | 77.3 | 81.4 | **88.8** | **15.400** | 24.17 min |

Full-SPE 的不同 checkpoint 只是 Top-1、Top-K、Oracle 与 invalid 间存在取舍，并没有出现稳定的后期质量提升；此前继续到 800K 的尝试也没有解决这一问题。因此，Full-SPE 保留了真实的速度价值，但不再作为主训练分支。

## 4. 为什么“更短”没有自动变得更容易？

### 4.1 绝对编辑数下降，但编辑密度上升

`normalized edit` 定义为逐样本 `edit distance / raw target length` 后取平均。Full-SPE 的 edit 数从 5.741 降至 3.857，但由于序列缩短得更多，平均编辑密度从 11.756% 升至 33.486%。

| Tokenizer | Mean edit | Normalized edit |
|---|---:|---:|
| Atom-level | 5.741 | 11.756% |
| Full-SPE | 3.857 | 33.486% |

也就是说，Full-SPE 中每个 fragment 位置更频繁地需要被改变；“少编辑几次”不等于“每一步更容易预测”。

### 4.2 词表和 token completion 难度急剧增加

Full-SPE 的词表从 69 增至 3,035。INS/SUB 不再是在几十个 atom token 中选择，而是在数千个 fragment token 中选择；一个 fragment 选错还可能同时带来多个原子级字符错误。

### 4.3 编辑组成确实改变了，但这也是新的风险来源

| Tokenizer | INS | DEL | SUB |
|---|---:|---:|---:|
| Atom | 91.201% | 0.924% | 7.875% |
| Full-SPE | 62.274% | 0.611% | 37.115% |

Full-SPE 验证了最初的表示假设：它明显减少了 INS 的绝对主导，并让 SUB 成为重要操作。但在 3,035 词表下，fragment-SUB 本身成为高难度 token completion 问题。

> 因此，Full-SPE 的失败不说明 fragment-level edit 无效；更合理的解释是 merge 过深，压缩收益被高词表和高编辑密度抵消。

## 5. 控制 merge 深度：为什么选择 M500？

为验证“merge 过深”而不是“fragment 化本身”导致问题，保持 ChEMBL rule 顺序不变，只使用前 K 条规则。下表为改进后 global R-SMILES 的训练集统计：

| 表示 | Merge rules | Real vocab | Aligned mean | Mean edit | Normalized edit | SUB 占编辑 |
|---|---:|---:|---:|---:|---:|---:|
| Atom | — | 69 | 50.809 | 5.741 | 11.756% | 7.875% |
| SPE-M500 | 500 | 568 | 16.047 | 4.127 | 27.017% | 32.045% |
| SPE-M1000 | 1,000 | 1,066 | 14.259 | 4.053 | 29.621% | 34.109% |
| SPE-M2000 | 2,000 | 2,056 | 12.722 | 3.869 | 31.865% | 36.761% |
| Full-SPE | 3,002 | 3,035 | 12.057 | 3.857 | 33.486% | 37.115% |

M500 已把平均对齐长度从 50.81 缩至 16.05（约 -68%），并完成了大部分 INS→SUB 的结构变化；继续从 M500 合并到 Full 只再缩短约 4 个 token，却把词表扩大约 5.3 倍、编辑密度继续推高。

M1000/M2000 目前只有完整数据审计，没有模型训练结果。基于已有证据，先验证 M500 是成本最低且机制最明确的选择。

## 6. M500 的训练结果：fragment-level 的有效版本

### 6.1 与 Full-SPE 的严格 tokenizer 对照

在相同 B256、模型结构、训练超参数和 400K step 下，M500 相比 Full-SPE：

| 400K 对照 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid@1 | 采样时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-SPE | 56.3 | 73.2 | 77.4 | 80.4 | 87.6 | 19.885 | 23.65 min |
| SPE-M500 | **57.6** | **76.2** | **80.4** | **83.9** | **88.9** | **12.625** | 24.51 min |

M500 只慢 0.86 min（约 3.6%），但 Top-3/5/10 分别提高 3.0/3.0/3.5 pp，Invalid@1 降低 7.26 pp。对 1,000 个 reaction block 的 paired bootstrap 中，Top-3/5/10 改善的 95% CI 均不跨 0；这说明 M500 优于 Full 的证据不是单纯的采样偶然波动。

### 6.2 checkpoint 不是由 validation loss 决定的

M500 的单 seed 密集 sweep 显示后期指标并不单调：490K 的 seed=42 Top-1/Top-5/Oracle 分别为 60.1%/80.5%/90.0%，520K/530K 的 Top-10 达 84.7%，600K 的 Invalid@1 最低（11.985%）。因此不能以最低 validation loss 或最后一个 checkpoint 代替生成评估。

对 490K 与 500K 在 seed 42/7/123 的配对复评如下（均值 ± sample SD）：

| Checkpoint | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid@1 |
|---|---:|---:|---:|---:|---:|---:|
| **M500@490K** | **59.27 ± 0.74** | **76.40 ± 0.20** | **80.63 ± 0.81** | 83.67 ± 0.65 | **89.90 ± 0.66** | 12.65 ± 0.18 |
| M500@500K | 58.90 ± 0.87 | 76.13 ± 0.49 | 80.13 ± 0.15 | **83.97 ± 0.35** | 89.83 ± 0.15 | **12.47 ± 0.19** |

结论是“按目标选 checkpoint”，而不是一个点严格支配另一个：

- **M500@490K**：当前 Top-1/3/5 与 Oracle 的主 checkpoint，后续方法改进采用它作为 baseline。
- **M500@500K**：Top-10 与 Invalid@1 略优，保留为次级对照。
- 继续训练到 600K 没有带来稳定的 Top-1 增益；后期候选更有效但也更重复，不把“继续加训练步数”作为主改进方向。

### 6.3 与 Atom 实用基线相比

| 模型 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | Invalid@1 | 采样时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Atom@600K（B128） | 56.3 | **76.5** | 80.3 | 83.9 | **90.2** | **11.740** | 58.21 min |
| M500@500K（B256，seed=42） | **59.9** | 75.9 | 80.1 | **84.3** | 89.7 | 12.245 | 24.47 min |

M500 在该单 seed 对照中 Top-1 提高 3.6 pp，Top-10 提高 0.4 pp，采样约快 2.38×；但 Atom 的 Top-3/5、Oracle 和 Invalid 仍略好，且训练 batch/预算不完全相同。因此准确表述是：**M500 已是强且高效的 fragment-level baseline，但尚不能声称它在所有指标上全面替代 Atom。**

## 7. 表示本身很重要：global alignment 与 SPE 不能混为一个结论

SPE-M500 的成功发生在改进后的 global R-SMILES 上。为避免把 global alignment 的收益误记为 tokenizer 收益，构造了改进前/后 R-SMILES × Atom/M500 的数据审计。

| 数据表示 | Aligned mean | Mean edit | Edit density | KEEP | SUB 占编辑 |
|---|---:|---:|---:|---:|---:|
| 原始 R-SMILES Atom | 51.83 | 14.10 | 26.37% | 72.79% | 32.1% |
| 原始 R-SMILES M500 | 15.24 | 6.84 | **44.24%** | **55.13%** | **47.1%** |
| global R-SMILES Atom | 50.81 | 5.74 | 11.72% | 88.70% | 7.9% |
| global R-SMILES M500 | 16.05 | 4.13 | 26.93% | 74.28% | 32.1% |

改进后的 global 表示并没有主要靠缩短序列获益；它甚至略长，却显著降低 edit 数、提升 KEEP、降低 SUB。已排除 OOV、SPE round-trip、行错位和 M500 `max_seq_len=96` 截断等数据管线问题。

对应地，改进前 R-SMILES 的 M500 不裁剪梯度重训在 600K 仅达到 Top-1/Top-3/Top-5/Top-10/Oracle = 44.3/61.6/64.5/66.4/81.6%，Invalid@1=20.970%；从 600K 续训到 700K 没有进一步改善。改进前 Full-SPE 更低：最佳 Top-1 为 39.7%（550K），最佳 Top-10 为 61.4%，最低 Invalid@1 为 22.895%（600K）。

这说明两件事：

1. 即使在原始 R-SMILES 上，M500 也显著优于 Full-SPE，支持“Full merge 过深”这一判断。
2. 但原始表示下 M500 远低于 global M500，说明 global alignment 对任务难度有很大贡献。

最后一组原始 R-SMILES Atom-level 控制尚未完成，因此当前不能把“global 与 SPE 各贡献多少”写成严格的因果分解；这也是后续仍需补齐的唯一关键训练控制。

## 8. 轨迹可视化：M500 的收益与失败都不是均匀的

在同一 global dev 样本、Euler N=9、100 steps、seed=42 下，对 M500@490K 与 Atom@600K 做了逐事件轨迹可视化：

| 样例 | M500@490K | Atom@600K | 观察 |
|---|---:|---:|---|
| #2500 | 9/9 target match，9/9 valid | 6/9 target match，9/9 valid | 两者都需插入 `Cl` 和 `.`；Atom 会把卤素补成 `F`/`Br`，M500 在该局部 completion 上更稳定。 |
| #1 | 6/9，9/9 valid | 7/9，9/9 valid | Atom 的正确轨迹只需 2 次 INS；M500 的表示对应 3 INS + 1 SUB，更容易出现酸/酯/醛相关的 completion 错误。 |
| #8888 | 3/9，8/9 valid | 9/9，9/9 valid | M500 需要 4 INS + 1 SUB，出现不编辑、错误 SUB 与 invalid；Atom 的 4 次 INS 更稳定。 |

对 #1/#8888，M500@600K 的命中数仍为 6/9 与 3/9，没有显示出 600K 可以修复这种模式。这些样例是机制诊断而非总体统计：它们说明 M500 的优势主要可能出现在某些局部 token completion，而对需要 fragment 边界重排/断开的反应，fragment alignment 也可能增加 sequential decision 的难度。

## 9. 最终结论与下一步

### 已得到的结论

1. **Fragment-level editing 可以有效，但 merge 深度必须受控。** Full-SPE 的速度收益真实，却因 3,035 词表与 33.49% 编辑密度损失质量；M500 是当前已验证的平衡点。
2. **M500 是后续研究的正确 fragment-level baseline。** 它保留约 2.4× 的实际采样加速，并在主目标 Top-1 上优于历史 Atom baseline；490K 的三 seed 结果是当前最稳妥的 checkpoint 选择证据。
3. **不能只看 validation loss、训练步数或单一样例。** checkpoint 选择必须用冻结 dev 协议下的 Top-K、Oracle、Invalid 与多 seed 复评。
4. **global alignment 是 SPE 成功的重要前提。** 原始 R-SMILES 的高 edit density / 高 SUB 任务明显更难；原始 Atom 控制完成前，不能把所有提升都归因于 fragment tokenization。

### 后续推理/采样改进的固定 baseline

| 项目 | 选择 |
|---|---|
| 数据与表示 | 改进后 global R-SMILES，`datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500` |
| Checkpoint | **SPE-M500@490K** |
| 主采样器 | 普通 Euler，N=9，100 steps，cubic |
| 主验证集 | `evaluation_v2/dev_unique1000_aug20` |
| 主指标 | Top-1/3/5/10、Oracle、Invalid@1、unique candidates、运行时间 |
| 次级对照 | M500@500K（Top-10/invalid 取向）；R9K1M2 仅作为独立 sampler 对照 |

R9K1M2 在 M500@490K、seed=42 上将 Top-3/5/Oracle/Invalid 改善为 77.3%/80.7%/90.4%/12.395%，但 Top-1 为 60.0%（Euler 为 60.1%），因此它不替代普通 Euler 作为表示实验的默认基线。

下一步先在该固定 M500@490K baseline 上做低成本、可独立验证的推理/采样改进；只有 dev 收益在额外 seed 下仍存在，才进行一次未参与调参的完整 `src-test`/`tgt-test` 评估。同时，等待并评估最后的原始 R-SMILES Atom-level 控制，以完成 global alignment 与 fragment tokenization 的归因。

## 10. 证据记录

- [SPE 总结](summary.md)：完整 checkpoint、采样器与 baseline 决策记录。
- [M500 checkpoint 对比](spe_m500_checkpoint_comparison.md)：密集 checkpoint sweep、N=1/N=9 与三 seed 结果。
- [SPE 前缀规则实验](../new_docs/SPE_prefix_rules_experiment.md)：K=500/1000/2000/Full 的 tokenizer 审计。
- [四组数据审计](../new_docs/RSMILES_SPE_M500_four_way_dataset_audit.md)：原始/改进 R-SMILES × Atom/M500 的机制分析。
- [原始 R-SMILES M500 实验记录](../orig_rsmiles_spe_m500_experiments_summary.md) 与 [原始 R-SMILES Full 结果](../orig_rsmiles_spe_full.txt)。

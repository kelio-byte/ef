# 原始/改进 R-SMILES × Atom/SPE-M500：四组数据审计

记录日期：2026-08-20 UTC。此文档解释为什么“基于改进前 R-SMILES 的 SPE-M500”不能直接沿用改进后 R-SMILES 分支的直觉或训练判断。

## 目的与口径

将两个因素拆开：

| R-SMILES 表示 | Atom-level | SPE-M500（`SPE_ChEMBL.txt` 前 500 条 merge） |
|---|---|---|
| 改进前 | `USPTO_50K_PtoR_aug20` | `USPTO_50K_PtoR_aug20_SPE_m500` |
| 改进后（`#global#`） | `USPTO_50K_PtoR_aug20_#global#` | `USPTO_50K_PtoR_aug20_#global#_SPE_m500` |

- 所有训练集均为 800,060 条 augmentation pair；验证集均为 100,020 条。四组 raw/aligned 文件逐行一致，SPE 到原 SMILES 的 round-trip 校验通过。
- 编辑统计来自已有的 Levenshtein 对齐文件。`INS` 指 source 为 `<GAP>`、target 有 token；`DEL` 相反；`SUB` 指同一对齐列 token 不同。
- “平均编辑密度”是逐样本 `edit_distance / aligned_length` 的均值；“KEEP 比例”是全部对齐列中未编辑列的占比。两者口径不同，不能互相替代。
- 生成原始 JSON：
  - `results/dataset_audit/original_rsmiles_atom_vs_spe_m500.json`
  - `results/dataset_audit/global_rsmiles_atom_vs_spe_m500.json`
  - 命令为 `scripts/preprocessing/spe_stats.py`；该只读审计已增加 P95/P99、编辑密度与 KEEP 比例输出。

## 训练集结构统计

下表格式为 `均值 / P95 / 最大值`，长度单位均为 token。

| 数据表示 | 词表 | src 长度 | tgt 长度 | 对齐长度 | 编辑距离 | 平均编辑密度 / P95 | KEEP 比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始 R-SMILES Atom | 68 | 44.60 / 73 / 160 | 49.50 / 80 / 162 | 51.83 / 85 / 189 | 14.10 / 36 / 101 | 26.37% / 58.00% | 72.79% |
| 原始 R-SMILES M500 | 561 | 12.25 / 21 / 51 | 14.61 / 24 / 51 | 15.24 / 25 / 63 | 6.84 / 15 / 43 | **44.24% / 86.67%** | **55.13%** |
| 改进 R-SMILES Atom | 69 | 45.57 / 74 / 160 | 50.76 / 82 / 164 | 50.81 / 82 / 164 | 5.74 / 18 / 57 | 11.72% / 34.04% | 88.70% |
| 改进 R-SMILES M500 | 568 | 13.27 / 22 / 57 | 16.02 / 26 / 60 | 16.05 / 26 / 60 | 4.13 / 9 / 30 | 26.93% / 54.55% | 74.28% |

补充事实：

- M500 将总训练 token 数从原始 Atom 的 75.28M 降到 21.49M（-71.45%，3.50×）；在改进 R-SMILES 中从 77.07M 降到 23.43M（-69.59%，3.29×）。
- M500 的 `max_seq_len=96` 完全无损：原始 M500 对齐最大长度为 63，改进 M500 为 60。故原始 M500 的低分**不能**归因于截断。
- Atom 若也套用 96 会截去约 0.51%（src）/1.03%（tgt）的原始训练行；当前 Atom 历史训练并非按 96 截断。因此不能把“无截断”误解为 M500 的性能优势。

## 编辑类型：真正改变任务难度的部分

| 数据表示 | INS | DEL | SUB | 解读 |
|---|---:|---:|---:|---|
| 原始 Atom | 51.3% | 16.5% | 32.1% | 对齐仍有大量替换和删除 |
| 原始 M500 | 43.7% | 9.2% | **47.1%** | 短序列中近半 edit 是 fragment 替换 |
| 改进 Atom | **91.2%** | 0.9% | 7.9% | 改进 R-SMILES 将大部分差异转成可对齐的插入 |
| 改进 M500 | 67.3% | 0.6% | 32.1% | tokenization 后仍有 SUB，但明显低于原始 M500 |

关键对比：

1. **R-SMILES 改进的收益不是“更短”。** Atom 的平均 src/tgt 反而略长约 2%，M500 也长约 8–10%；但平均 edit 分别下降 59.3%（14.10 → 5.74）和 39.6%（6.84 → 4.13）。真正改善的是产物/反应物的编辑对齐结构。
2. **原始 M500 是四组中最稠密的编辑任务。** 它的绝对 edit 数小于 Atom（6.84 vs 14.10），但短得多的序列中平均 44.24% token 需要修改，P95 达 86.67%；这比改进 M500 的 26.93% / 54.55% 高很多。
3. **M500 并未消除 token 预测难度，只是减少位置数。** 原始 M500 的 SUB 占编辑的 47.1%，而改进 M500 为 32.1%。对 Edit Flows 而言，SUB/INS 都需要预测 Q token；一个 fragment token 选错往往携带多个原子级字符的错误。

## 词表与 token 选择难度

| 数据表示 | 词表大小 | 训练 token 一元熵 | 有效词表大小 `2^H` | Top-20 token 覆盖率 |
|---|---:|---:|---:|---:|
| 原始 Atom | 68 | 3.46 bit | 11.0 | 98.39% |
| 原始 M500 | 561 | 8.03 bit | 261.4 | 26.97% |
| 改进 Atom | 69 | 3.47 bit | 11.1 | 98.32% |
| 改进 M500 | 568 | 7.97 bit | 250.3 | 28.58% |

这不表示每一步都要在 250 个 token 中选择（只有 INS/SUB 有 Q head），但它说明 M500 的 Q 预测是显著更平坦、更高熵的分类问题。该压力在两种 R-SMILES 下都存在，因此它**不能单独解释**原始/改进 M500 的差异；但会使固定 Atom 超参数、固定训练 step 数的迁移更敏感。

## 验证集 OOV 与数据管线排除项

- 四组验证集均只有 20/100,020 行含 OOV（同一 augmentation block 的重复），src+tgt 合计 40 个 OOV token。
- Atom token-level OOV 约 0.00042%，M500 约 0.0014–0.0015%，均远低于足以解释 Top-K 差异的程度。
- SPE round-trip、raw/aligned 投影、src/tgt 行数全部通过；M500 也没有长度截断。

因此，当前没有证据指向 tokenizer 实现错误、数据错位、OOV 或 max length 截断。

## 对“原始 R-SMILES M500@490K 低分”的解释边界

数据本身已提供一个强而兼容的机制：原始 R-SMILES + M500 把任务变成**更短、但每个位置更常需要编辑且更常需要 fragment 替换**的序列编辑任务；而改进 R-SMILES 先显著降低了编辑量和 SUB，再叠加 M500，因而保留了短序列的效率优势。

但这仍不是对 490K checkpoint 的最终归因。还需要外机迁回的训练/评估产物来区分：

1. **优化不足或超参不匹配：** 查看 train/val 曲线、490K 附近 checkpoint 曲线，以及短序列/高熵 Q head 是否仍在改善；
2. **采样问题：** 用完全相同的 dev reaction blocks 比较 Euler N=1/9、invalid、Oracle、unique 与轨迹重合；
3. **表示瓶颈：** 若训练充分、采样充分后，原始 M500 仍稳定落后原始 Atom，则高密度 SUB + 高熵 Q 是更可信的主因。

## 下一步（在收到外机文件后）

优先迁移 `490K` 及相邻 `450K/500K/600K` checkpoint、对应 `config.yaml`、`training_monitor.jsonl`、`training_summary.json`、训练日志，以及 490K 的 `sampling_metadata.json`、`diagnostics.json`、评估日志和压缩后的 `predictions.txt`。不需要迁移完整数据集。

随后以同一冻结的 original-R-SMILES `dev_unique1000_aug20` 做：

1. checkpoint 粗筛（450K、490K、500K、600K）；
2. 只对最强的两个做第二个 sampling seed；
3. 与原始 R-SMILES Atom baseline 在同一 reaction blocks 上比较。

只有上述三步仍显示原始 M500 在 Top-1/Top-K、Oracle 与 invalid 上持续不具竞争力，才应把“原始 R-SMILES 上的 fragment-level edit”判为不值得继续；不能用单个 490K / 单 seed 结果直接下结论。

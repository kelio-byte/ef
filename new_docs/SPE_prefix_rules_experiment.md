# SPE 前缀规则实验（K=500/1000/2000）

## 目的

当前 full-SPE 使用 `SPE_ChEMBL.txt` 的全部 3,002 条 merge rule。它把序列压缩得很短，但同时带来更高的 fragment-level 编辑密度和更大的词表。本实验只改变使用的前缀规则数，先观察 tokenizer/编辑任务本身的变化，再决定是否训练新模型。

本轮没有启动训练，也没有覆盖现有 full-SPE 数据或 checkpoint。

## 实现

`scripts/preprocessing/preprocess_spe.py` 新增了显式的 `--merges K` 参数：

- `K >= 0`：只使用 `SPE_ChEMBL.txt` 前 K 条规则；
- `K=-1`：使用整个 codes 文件，保持原有 full-SPE 行为。

每个 K 都使用独立目录，重新生成未对齐数据、训练集 vocab 和 train/val/test alignment。metadata 记录了 `merges`、codes hash 和数据 hash。

本轮目录：

- `datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500`
- `datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m1000`
- `datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m2000`

## 数据完整性

三个目录均包含 800,060 train、100,020 val、100,140 test pairs。完整审计结果：

- SPE token round-trip failure：0；
- raw/aligned projection mismatch：0；
- aligned 两侧长度不一致：0；
- 未对齐数据中的 `<GAP>`：0；
- 所有序列均未超过训练 `max_seq_len=256`。

验证集 combined token OOV rate 为：K=500 **0.001364%**、K=1000 **0.001544%**、K=2000 **0.001735%**。三者都低，但随着 merge 数增加略有上升。

## 训练集统计

下面的 normalized edit distance 是逐样本计算 `edit distance / raw target length` 后再取平均。Atom 和 full-SPE 是已有正式数据；K=500/1000/2000 是本轮新生成数据。

| 表示 | src mean | tgt mean | aligned mean | src P95 | tgt P95 | max(src/tgt) | mean edit | normalized edit | real vocab |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Atom | 45.573 | 50.756 | 50.809 | 74 | 82 | 160/164 | 5.741 | 11.756% | 69 |
| SPE K=500 | 13.268 | 16.021 | 16.047 | 22 | 26 | 57/60 | 4.127 | 27.017% | 568 |
| SPE K=1000 | 11.610 | 14.238 | 14.259 | 20 | 23 | 49/52 | 4.053 | 29.621% | 1,066 |
| SPE K=2000 | 10.297 | 12.700 | 12.722 | 17 | 21 | 46/48 | 3.869 | 31.865% | 2,056 |
| SPE full K=3002 | 9.655 | 12.034 | 12.057 | 16 | 20 | 43/45 | 3.857 | 33.486% | 3,035 |

### Aligned 文件的详细统计

以下数据直接来自每个目录下的 `spe_statistics_vs_atom.json`。`aligned length` 是 DP alignment 后的列数；`edit distance` 是逐样本 alignment 操作数。JSON 当前提供 P90（不是 P95），因此这里明确标注为 P90。

#### Aligned length

| 表示 | split | mean | median | P90 | max |
|---|---|---:|---:|---:|---:|
| Atom | train | 50.808943 | 50 | 74 | 164 |
| Atom | val | 50.734293 | 50 | 73 | 132 |
| Atom | test | 50.670232 | 49 | 74 | 157 |
| SPE K=500 | train | 16.046751 | 15 | 23 | 60 |
| SPE K=500 | val | 16.061308 | 16 | 23 | 55 |
| SPE K=500 | test | 16.041282 | 15 | 24 | 60 |
| SPE K=1000 | train | 14.259054 | 14 | 21 | 52 |
| SPE K=1000 | val | 14.281324 | 14 | 21 | 50 |
| SPE K=1000 | test | 14.250619 | 14 | 21 | 55 |
| SPE K=2000 | train | 12.722051 | 12 | 19 | 48 |
| SPE K=2000 | val | 12.742452 | 12 | 19 | 47 |
| SPE K=2000 | test | 12.714350 | 12 | 19 | 50 |
| SPE full K=3002 | train | 12.057239 | 12 | 18 | 45 |
| SPE full K=3002 | val | 12.067856 | 12 | 18 | 44 |
| SPE full K=3002 | test | 12.050969 | 12 | 18 | 46 |

#### Edit distance

| 表示 | split | mean | median | P90 | max | total edits |
|---|---|---:|---:|---:|---:|---:|
| Atom | train | 5.740897 | 4 | 15 | 57 | 4,593,062 |
| Atom | val | 5.738122 | 4 | 15 | 55 | 573,927 |
| Atom | test | 5.842331 | 4 | 16 | 68 | 585,051 |
| SPE K=500 | train | 4.127404 | 4 | 7 | 30 | 3,302,171 |
| SPE K=500 | val | 4.108338 | 4 | 7 | 23 | 410,916 |
| SPE K=500 | test | 4.144568 | 4 | 7 | 32 | 415,037 |
| SPE K=1000 | train | 4.053486 | 4 | 7 | 28 | 3,243,032 |
| SPE K=1000 | val | 4.042501 | 4 | 7 | 23 | 404,331 |
| SPE K=1000 | test | 4.068334 | 4 | 7 | 30 | 407,403 |
| SPE K=2000 | train | 3.869311 | 4 | 7 | 24 | 3,095,681 |
| SPE K=2000 | val | 3.848170 | 4 | 7 | 20 | 384,894 |
| SPE K=2000 | test | 3.881096 | 4 | 7 | 28 | 388,653 |
| SPE full K=3002 | train | 3.856764 | 4 | 6 | 23 | 3,085,643 |
| SPE full K=3002 | val | 3.834613 | 4 | 6 | 20 | 383,538 |
| SPE full K=3002 | test | 3.868384 | 4 | 7 | 28 | 387,380 |

### 编辑操作分布

| 表示 | INS | DEL | SUB |
|---|---:|---:|---:|
| Atom | 91.201% | 0.924% | 7.875% |
| SPE K=500 | 67.341% | 0.614% | 32.045% |
| SPE K=1000 | 65.362% | 0.529% | 34.109% |
| SPE K=2000 | 62.677% | 0.562% | 36.761% |
| SPE full K=3002 | 62.274% | 0.611% | 37.115% |

## 初步结论

1. **K=500 明显降低了 full-SPE 的 fragment 编辑密度和词表规模**：normalized edit 从 33.486% 降到 27.017%，词表从 3,035 降到 568；代价是 aligned 序列从 12.057 增至 16.047。
2. **K=1000 是较平衡的候选**：aligned 长度 14.259，词表 1,066，normalized edit 29.621%。它仍比 Atom 短很多，但没有 full-SPE 那么大的词表和编辑密度。
3. **K=2000 已经接近 full-SPE 的编辑分布**：词表仍为 2,056，normalized edit 为 31.865%，但序列只比 K=1000 短约 10.8%。因此它的额外收益目前不明显。
4. 这组统计支持“full-SPE 的质量下降可能与 token 粒度/编辑密度有关”的假设，但**不能直接推出准确率提升**；必须训练匹配的新 checkpoint 后才可验证 Top-K、Oracle 和 invalid rate。

## 下一步建议

优先训练 **K=1000**，因为它在序列压缩、词表规模和编辑密度之间最均衡。K=500 可作为质量/invalid 优先的备选；K=2000 暂不优先，除非 K=1000 的推理速度不足。

训练时必须为每个 K 使用新的 data directory、vocab 和 alignment，并从头初始化模型；现有 full-SPE checkpoint 不能直接迁移到这些数据上。

统计原始 JSON：各实验目录下的 `spe_statistics_vs_atom.json`。

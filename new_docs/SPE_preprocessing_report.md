# SPE 数据预处理与统计报告

日期：2026-08-13

## 结论

SPE shadow dataset 已完成生成、重新 alignment 和全量审计，路径为：

```text
datasets/USPTO_50K_PtoR_aug20_#global#_SPE/
```

数据完整性通过，可以作为独立 tokenizer 分支进入后续训练准备；本阶段没有训练模型，也没有修改原始
`datasets/USPTO_50K_PtoR_aug20_#global#/`。

SPE 的主要效果是显著缩短序列和 token-level 编辑距离：全量三 split 的平均 src/tgt 长度约从
`45.5/50.7` 降到 `9.7/12.0`，平均 Levenshtein distance 从 `5.75` 降到 `3.86`，总编辑数下降
约 `33.0%`。代价是词表从 `69` 个真实 token 增加到 `3,035` 个，模型词表从 `73` 增加到 `3,039`；
val/test 的 OOV 非零但很低，combined token rate 约为 `4.47e-5` / `4.01e-5`。

这说明 SPE 预处理层面的可行性较好，但还不能直接推出模型收益。进入重新训练前，应单独建立 SPE 配置和
checkpoint，并保持原 tokenizer 的训练/评估 protocol 作为 baseline。

## 1. 处理流程与实现

新增：

- [`scripts/preprocessing/preprocess_spe.py`](../scripts/preprocessing/preprocess_spe.py)：只读取原始未 aligned 的
  `src-{split}.txt` / `tgt-{split}.txt`，去除旧 tokenizer 的显示空格，调用 `SmilesPE` dropout=0，写入新的
  `_SPE` 目录，并按训练集 src+tgt 的频率降序生成 `example.vocab.src`。
- [`scripts/preprocessing/spe_stats.py`](../scripts/preprocessing/spe_stats.py)：复用已生成的 alignment 做统计，
  同时检查源/目标行数、aligned 两侧长度、SPE 相对原始 SMILES 的无损恢复、`<GAP>` 位置和 val/test OOV。
- [`scripts/preprocessing/SPE_ChEMBL.txt`](../scripts/preprocessing/SPE_ChEMBL.txt)：预训练 SPE merge code，
  3002 行，SHA256 为
  `ee408e30e0aae598770f233013b312b206df424f6e593b410b0107d7d2237a43`。

SPE 依赖固定为 `SmilesPE==0.0.3`。alignment 没有重新实现，而是直接调用现有：

```bash
python scripts/precompute_alignments.py \
  --data_dir datasets/USPTO_50K_PtoR_aug20_#global#_SPE \
  --splits train val test --num_workers 32
```

## 2. Sanity check

先对每个 split 的前 10 对数据执行了完整链路：SPE 预处理、现有 alignment、统计审计。结果：

- 每个 split 10 对，src/tgt 行数和配对关系保持不变；
- `''.join(SPE_tokens) == original_smiles` 全部通过；
- aligned src/tgt 每行 token 长度一致；
- 未对齐 SPE 文件中 `<GAP>` 数量为 0；
- `<GAP>` 仅由 alignment 脚本写入 aligned 文件；
- 预处理输出目录与源目录不同，覆盖源目录会被脚本拒绝。

## 3. 全量数据完整性

| split | pairs | SPE round-trip failures | raw/aligned count mismatch | aligned length mismatch | unaligned `<GAP>` |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 800,060 | 0 | 0 | 0 | 0 |
| val | 100,020 | 0 | 0 | 0 | 0 |
| test | 100,140 | 0 | 0 | 0 | 0 |

训练集 vocab 只使用新的 SPE `train/src-train.txt` 和 `train/tgt-train.txt`，没有使用 val/test。

| 项目 | 原 tokenizer | SPE |
| --- | ---: | ---: |
| 真实 vocab size | 69 | 3,035 |
| model vocab size（含 PAD/BOS/GAP/UNK） | 73 | 3,039 |

## 4. Token 长度统计

下面的 `max>256` 是长度严格大于 `max_seq_len=256` 的比例；本数据两种 tokenizer 均为 0。

### train

| side | tokenizer | mean | median | P90 | max | >256 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| src | 原 tokenizer | 45.573 | 45 | 67 | 160 | 0 |
| src | SPE | 9.655 | 9 | 15 | 43 | 0 |
| tgt | 原 tokenizer | 50.756 | 50 | 74 | 164 | 0 |
| tgt | SPE | 12.034 | 12 | 18 | 45 | 0 |

### val

| side | tokenizer | mean | median | P90 | max | >256 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| src | 原 tokenizer | 45.487 | 45 | 67 | 130 | 0 |
| src | SPE | 9.674 | 9 | 15 | 36 | 0 |
| tgt | 原 tokenizer | 50.683 | 50 | 73 | 132 | 0 |
| tgt | SPE | 12.046 | 12 | 18 | 44 | 0 |

### test

| side | tokenizer | mean | median | P90 | max | >256 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| src | 原 tokenizer | 45.334 | 44 | 68 | 153 | 0 |
| src | SPE | 9.649 | 9 | 15 | 40 | 0 |
| tgt | 原 tokenizer | 50.616 | 49 | 74 | 157 | 0 |
| tgt | SPE | 12.025 | 12 | 18 | 46 | 0 |

## 5. 编辑距离、aligned length 与操作类型

### 平均 / 中位数 / P90 Levenshtein distance

| split | 原 mean / median / P90 | SPE mean / median / P90 | mean change |
| --- | ---: | ---: | ---: |
| train | 5.741 / 4 / 15 | 3.857 / 4 / 6 | −32.82% |
| val | 5.738 / 4 / 15 | 3.835 / 4 / 6 | −33.17% |
| test | 5.842 / 4 / 16 | 3.868 / 4 / 7 | −33.79% |

### 平均 aligned length

| split | 原 tokenizer | SPE | change |
| --- | ---: | ---: | ---: |
| train | 50.809 | 12.057 | −76.27% |
| val | 50.734 | 12.068 | −76.21% |
| test | 50.670 | 12.051 | −76.22% |

### 全量三 split 编辑操作比例

| 操作 | 原 count / ratio | SPE count / ratio |
| --- | ---: | ---: |
| INS | 5,248,098 / 91.239% | 2,401,643 / 62.274% |
| DEL | 52,980 / 0.921% | 23,569 / 0.611% |
| SUB | 450,962 / 7.840% | 1,431,349 / 37.115% |
| total | 5,752,040 | 3,856,561 |

SPE 主要把一部分原本由多个 atom-level insertion 表达的变化压缩成 fragment-level substitution；因此
INS 比例下降、SUB 比例上升是预期现象，不能只用 INS 比例判断任务是否变简单。

## 6. val/test OOV

OOV 词表是仅由 SPE 训练集 src+tgt 建立的 `example.vocab.src`，特殊 token 不计入 OOV。

| split | side | OOV tokens | OOV token rate | lines with OOV | OOV line rate |
| --- | --- | ---: | ---: | ---: | ---: |
| val | src | 55 | 5.684e-5 | 55 | 5.499e-4 |
| val | tgt | 42 | 3.486e-5 | 42 | 4.199e-4 |
| val | combined | 97 | 4.465e-5 | 55 | 5.499e-4 |
| test | src | 45 | 4.657e-5 | 44 | 4.394e-4 |
| test | tgt | 42 | 3.488e-5 | 41 | 4.094e-4 |
| test | combined | 87 | 4.008e-5 | 44 | 4.394e-4 |

OOV 很低但不是 0。按当前 `RetroDataset` 规则，后续训练时这些 token 会映射到 `<UNK>`；正式训练前应确认
这不会影响目标端 supervision，或另行决定是否将少量 holdout fragment 纳入词表（本阶段没有这样做）。

## 7. 结果文件与复现

- 预处理 metadata：
  [`spe_preprocessing_metadata.json`](../datasets/USPTO_50K_PtoR_aug20_#global#_SPE/spe_preprocessing_metadata.json)
- 全量统计：
  [`spe_statistics.json`](../datasets/USPTO_50K_PtoR_aug20_#global#_SPE/spe_statistics.json)
- 新 vocab：
  [`example.vocab.src`](../datasets/USPTO_50K_PtoR_aug20_#global#_SPE/example.vocab.src)

重新生成未对齐 SPE 数据：

```bash
conda activate ef
python scripts/preprocessing/preprocess_spe.py \
  --source-dir datasets/USPTO_50K_PtoR_aug20_#global# \
  --output-dir datasets/USPTO_50K_PtoR_aug20_#global#_SPE \
  --codes scripts/preprocessing/SPE_ChEMBL.txt \
  --cache-reset-interval 50000
```

## 8. 是否进入重新训练

建议进入“单独 SPE checkpoint 的训练准备/小规模训练验证”，但不要替换当前 baseline，也不要直接从旧
checkpoint resume。原因是预处理指标满足进入门槛：序列和编辑距离显著下降、长度无超限、round-trip 0 失败、
OOV 极低；但 vocab 和 operation distribution 都发生了实质变化，必须重新初始化匹配 SPE vocab 的模型，
再与原 tokenizer 在同一 protocol 下比较 Top-K、Oracle、invalid rate 和运行效率。训练 pilot 的独立配置为
[`configs/retro_spe_pilot.yaml`](../configs/retro_spe_pilot.yaml)，训练完成后应把 loss/u_tot、三类编辑率、
吞吐、显存和小规模 Euler sampling 结果记录在单独的
[`SPE_training_pilot_report.md`](SPE_training_pilot_report.md)，不应覆盖原 tokenizer 的 checkpoint。

# SPE 数据预处理与评估计划

## 目标

为当前 Edit Flows 化学逆合成项目构建 SPE tokenization 数据，用于验证 fragment-level token 是否能够缩短序列并降低 Levenshtein 编辑距离。

## 输入数据

现有目录：

```text
datasets/USPTO_50K_PtoR_aug20_#global#/
```

其中：

- `train/val/test` 下的 `src-*.txt`、`tgt-*.txt` 是未做 Levenshtein 对齐的 `x0/x1`；
- `*_aligned_src.txt`、`*_aligned_tgt.txt` 是旧 tokenizer 下预计算得到的 `z0/z1`，只能作为格式参考；
- `example.vocab.src` 是旧 tokenizer 的词表。

已准备：

```text
scripts/preprocessing/SPE_ChEMBL.txt
```

目标输出：

```text
datasets/USPTO_50K_PtoR_aug20_#global#_SPE/
```

## 正确处理顺序

```text
原 #global# src/tgt
    ↓
去除旧 tokenizer 的 token 分隔空格，
重建完整 SMILES 字符串
    ↓
SPE tokenization
    ↓
生成新的 SPE src/tgt
    ↓
复用 scripts/precompute_alignments.py
重新进行 token-level Levenshtein DP
    ↓
生成新的 *_aligned_src/tgt
    ↓
按当前项目规则构建 SPE vocab
    ↓
统计 OOV、长度、编辑距离等指标
```

注意：`scripts/precompute_alignments.py` 直接在字符串 token 上做 Levenshtein，对 vocab 无依赖，因此不需要先构建 vocab 再做 alignment。

## 任务要求

1. 先检查当前仓库的数据读取、vocab 构建/读取、特殊 token 规则，以及 `scripts/precompute_alignments.py` 的使用方式，不要凭假设重写已有逻辑。

2. SPE 数据唯一来源为未 aligned 的：

```text
train/src-train.txt
train/tgt-train.txt
val/src-val.txt
val/tgt-val.txt
test/src-test.txt
test/tgt-test.txt
```

不要对旧的 `*_aligned_*` 文件再次做 SPE。

3. 使用 `SmilesPE` 和：

```text
scripts/preprocessing/SPE_ChEMBL.txt
```

进行 SPE tokenization，`dropout=0`。

当前 src/tgt 已经过旧 tokenizer 分词，因此需要先去除旧 tokenizer 加入的 token 分隔空格，将每行重新拼接为完整 SMILES 字符串，再做 SPE。这里不是重新从 raw CSV 生成数据，也不是重新执行 #global# 对齐或数据增强。

对每条数据检查：

```text
''.join(SPE_tokens) == original_smiles
```

确保 tokenization 无损。

4. 将 SPE 后的未 aligned 数据写入：

```text
datasets/USPTO_50K_PtoR_aug20_#global#_SPE/
```

保持与原数据集相同的 train/val/test 目录结构和文件命名。

5. 直接复用现有：

```text
scripts/precompute_alignments.py
```

在新的 SPE `src/tgt` 上重新执行 Levenshtein DP，生成：

```text
train/train_aligned_src.txt
train/train_aligned_tgt.txt
val/val_aligned_src.txt
val/val_aligned_tgt.txt
test/test_aligned_src.txt
test/test_aligned_tgt.txt
```

不要重新实现另一套 alignment 算法。

注意该脚本检测到输出文件已存在时会跳过；调试重算时需明确删除对应旧输出，且只能删除新 SPE 目录中的文件，不能影响原始 `#global#` 数据。

6. 检查当前项目 `example.vocab.src` 的实际生成和读取规则，并按完全相同的规则构建新的 SPE vocab。

原则：

- 只使用训练集；
- 不使用 val/test 构建 vocab；
- 不改变现有特殊 token 规则；
- 不为了 SPE 实验顺手改变其他 vocab 逻辑。

7. 基于新 vocab 统计 val/test OOV 情况。

8. 在训练前完成当前 tokenizer 与 SPE tokenizer 的数据统计比较，至少包括：

- vocab size
- src/tgt mean / median / P90 token length
- max sequence length
- 长度超过 `max_seq_len=256` 的比例
- mean / median / P90 Levenshtein edit distance
- INS / DEL / SUB 数量或比例（如果能从现有 alignment 可靠统计）
- aligned sequence length
- val/test OOV rate

9. 先做小样本 sanity check，确认：
- SMILES 恢复正确；
- SPE token 拼接可无损恢复；
- src/tgt 行数和配对关系不变；
- aligned src/tgt 长度逐行一致；
- `<GAP>` 使用符合现有训练代码预期。

确认通过后再处理全量数据。

10. 本阶段只完成：
- SPE 预处理脚本；
- alignment；
- vocab；
- 数据完整性检查；
- 数据统计；
- 文档更新。

不要开始模型训练，不要覆盖原始 `#global#` 数据。

## 最终报告

完成后给出简洁报告：

- 新增/修改了哪些代码；
- 新数据集路径；
- SPE 与当前 tokenizer 的关键统计差异；
- 数据完整性检查结果；
- 是否建议进入重新训练阶段。

## 执行原则

- 优先复用现有代码；
- 尽量小改动；
- 不改变与 SPE 无关的实验变量；
- 完成代码、测试、全量处理、统计、文档更新和 Git commit 后再停止，不需要逐步等待确认。

# First-Step Analysis 实现说明

## 1. 概述

本轮实现围绕“第一步”分析新增了两条独立实验链路：

1. **静态初始预测分析**
   - 固定输入 `x_t = x_0`
   - 不执行采样
   - 只看模型在指定 `t` 下的 forward 输出是否能命中 oracle 编辑位置 / 类型 / token

2. **第一次真实编辑影响分析**
   - 运行 Euler 采样
   - 记录轨迹中第一次实际发生的 edit event
   - 分析第一次真实编辑正确性与最终结果的相关性
   - 支持首步干预：`normal` / `force_correct_first` / `force_wrong_first`

所有新实验入口都是独立脚本，不调用 `scripts/eval_retro.py`，默认也不写入 checkpoint 的 `eval/` 目录。建议统一输出到：

```text
analysis_outputs/first_step/...
```

这样不会覆盖已有的：

```text
checkpoints/*/*/eval/predictions.txt
```

---

## 2. 新增文件

| 文件 | 作用 |
|------|------|
| `scripts/first_step_forward_analysis.py` | 实验 1：静态初始预测分析 |
| `scripts/first_event_impact_analysis.py` | 实验 2：第一次真实编辑相关性与首步干预分析 |
| `edit_flows/analysis/first_step.py` | 两个脚本复用的分析工具函数 |
| `edit_flows/analysis/__init__.py` | 导出分析工具函数 |
| `edit_flows/sampling/euler.py` | 扩展 Euler 采样，支持第一次事件记录与首步干预 |
| `tests/sampling/test_euler.py` | 补充事件概率与 first-event 记录测试 |

已做的代码级验证：

```bash
python -m py_compile \
    scripts/first_step_forward_analysis.py \
    scripts/first_event_impact_analysis.py \
    edit_flows/sampling/euler.py \
    edit_flows/analysis/first_step.py

pytest tests/sampling/test_euler.py -q
```

当前测试结果：`13 passed`。

---

## 3. 公共实现

### 3.1 `edit_flows/analysis/first_step.py`

该文件提供两个脚本共用的轻量工具：

| 函数 | 作用 |
|------|------|
| `load_parallel_texts()` | 加载 products / targets，支持 `deduplicate` 和 `max_lines` |
| `tokenize_smiles()` | 将 tokenized SMILES 转为 token id |
| `build_model_batch()` | 构造带 `BOS` 的 `x_0, x_1` batch |
| `decode_sequence()` | 将模型输出 token id 转回 tokenized string |
| `extract_oracle_event_set()` | 从 oracle 输出中抽取正确编辑位置 / 类型 / token |
| `extract_position_labels()` | 从 oracle rate 中生成位置级标签 |
| `compute_average_precision()` | 计算 position AP |
| `compute_reaction_edit_distance()` | 计算预测与 target 的 Levenshtein edit distance |

### 3.2 `edit_flows/sampling/euler.py`

新增了几类内部辅助逻辑：

| 函数 | 作用 |
|------|------|
| `_sample_edit_actions()` | 从 `log_rates, log_ins_probs, log_sub_probs` 中采样本轮 edit action |
| `_extract_first_event_summary()` | 将第一次事件整理为可保存的字典 |
| `_select_oracle_anchor()` | 为 `force_correct_first` 选一个 oracle 正确首编辑 |
| `_select_wrong_anchor()` | 为 `force_wrong_first` 选一个模型高分但 oracle 错误的编辑 |
| `_override_with_anchor_event()` | 将某个样本当前 step 的编辑事件替换为指定 anchor edit |

同时扩展了 `sample_euler()`：

```python
sample_euler(
    ...,
    event_prob_mode="poisson",
    record_first_events=False,
    x_1=None,
    vocab_size=None,
)
```

当 `record_first_events=True` 且提供 `x_1, vocab_size` 时，函数会额外返回 `first_events`，用于相关性分析。

新增了首步干预接口：

```python
sample_euler_with_first_step_intervention(
    model,
    x_0,
    x_1,
    scheduler,
    vocab_size,
    mode="normal",
    ...
)
```

支持：

| mode | 含义 |
|------|------|
| `normal` | 不干预，按模型正常采样 |
| `force_correct_first` | 第一次发生事件的 step，将该样本事件替换为 oracle 正确 anchor edit |
| `force_wrong_first` | 第一次发生事件的 step，将该样本事件替换为模型高分但 oracle 错误的 anchor edit |

注意：当前干预实现是“首个 anchor edit”口径，不是强制执行 oracle 的完整 event set。这个口径更稳定，也更适合作为第一轮因果诊断。

---

## 4. 实验 1：静态初始预测分析

### 4.1 脚本

```text
scripts/first_step_forward_analysis.py
```

### 4.2 主要功能

该脚本固定：

```text
x_t = x_0
```

然后在多个指定时间点 `t` 上执行模型 forward，比较模型输出与 oracle 输出。

它同时保存两套分数：

1. **base**
   - `model.forward()` 的原始输出
   - 代表模型自身输出偏好

2. **effective**
   - 对 `log_rates` 应用 `apply_rate_parameterization()` 后的实际采样速率
   - 代表真实 Euler 采样会使用的速率

当前统计指标包括：

| 指标 | 含义 |
|------|------|
| `Center Hit@1` | 模型最高编辑位置是否命中 oracle 正位置 |
| `Center Hit@3` | 模型 top-3 编辑位置是否命中 oracle 正位置 |
| `Center Hit@5` | 模型 top-5 编辑位置是否命中 oracle 正位置 |
| `Center MRR` | 第一个 oracle 正位置在模型排序中的倒数排名 |
| `Position AP` | 将位置编辑需求作为多标签检索的 AP |
| `Type Acc@oracle-pos` | anchor 位置上的编辑类型是否正确 |
| `Ins Token Acc@1/5` | insert token 的 top-1 / top-5 准确率 |
| `Sub Token Acc@1/5` | substitute token 的 top-1 / top-5 准确率 |
| `Full First-Edit Acc` | 位置、类型、token 全部正确的比例 |

### 4.3 输入参数

核心参数：

| 参数 | 说明 |
|------|------|
| `--checkpoint` | 模型 checkpoint |
| `--products_file` | product 文件 |
| `--targets_file` | target 文件 |
| `--vocab_file` | vocab 文件；不传时从 checkpoint config 的 `data_dir/vocab_file` 读取 |
| `--output_dir` | 输出目录 |
| `--scheduler` | 采样 scheduler，可选 `cubic` / `linear`；不传时使用 checkpoint config |
| `--time_grid` | 逗号分隔的时间点 |
| `--deduplicate` | 每 N 行取一行，适配 aug20 数据 |
| `--max_lines` | 限制处理样本数 |
| `--batch_size` | forward batch size |
| `--device` | `cpu` 或 `cuda` |

### 4.4 使用示例

以 `#global#` 的 1000 条 deduplicated 子集为例：

```bash
PYTHONPATH=. python scripts/first_step_forward_analysis.py \
    --checkpoint checkpoints/USPTO_50K_PtoR_aug20_#global#/2026-06-08_17-20-39/checkpoint_stepXXXX.pt \
    --products_file train_subsets/USPTO_50K_PtoR_aug20_#global#/test/src-test.txt \
    --targets_file train_subsets/USPTO_50K_PtoR_aug20_#global#/test/tgt-test.txt \
    --vocab_file /data6/duanbh/desktop/retrosynthesis/dataset/USPTO_50K_PtoR_aug20_#global#/example.vocab.src \
    --output_dir analysis_outputs/first_step/global_train_subset/forward \
    --deduplicate 20 \
    --max_lines 1000 \
    --time_grid 0,1e-3,1e-2,5e-2,0.1,0.2,0.3 \
    --scheduler linear \
    --batch_size 64 \
    --device cuda
```

注意：示例里的 `checkpoint_stepXXXX.pt` 需要替换成实际 checkpoint 文件名。

### 4.5 输出文件

| 文件 | 内容 |
|------|------|
| `summary.json` | 每个 `t` 下的聚合指标，区分 `base` 与 `effective` |
| `per_example.pt` | 每条样本的 anchor 位置、oracle 位置、类型、是否 full correct 等明细 |
| `report.md` | 简短人读摘要 |

推荐先看：

```text
summary.json
```

重点比较：

- `t=0` 的 `Center Hit@1/3/5`
- `t=0` 的 `Full First-Edit Acc`
- 指标从 `t=0` 到 `t=0.3` 是否稳定
- `base` 与 `effective` 是否差异很大

---

## 5. 实验 2：第一次真实编辑影响分析

### 5.1 脚本

```text
scripts/first_event_impact_analysis.py
```

### 5.2 主要功能

该脚本运行实际 Euler 采样，并围绕第一次真实编辑做两类分析：

1. `--mode correlation`
   - 不干预采样
   - 记录每条 sample 的第一次真实 event
   - 统计第一次 event 正确性与最终 exact match 的关系

2. `--mode intervention`
   - 分别运行：
     - `normal`
     - `force_correct_first`
     - `force_wrong_first`
   - 对比首步干预对最终结果的影响

### 5.3 第一次真实编辑的记录内容

每个 sample 的 `first_event` 中保存：

| 字段 | 含义 |
|------|------|
| `first_event_step_idx` | 第一次事件所在 step 近似索引 |
| `first_event_t` | 第一次事件发生前的时间 `t` |
| `n_first_events` | 该 step 内同时发生的事件数 |
| `event_positions` | 该 step 内发生编辑的位置集合 |
| `anchor_pos` | 事件集合中的第一个 anchor 位置 |
| `anchor_type` | anchor 编辑类型 |
| `anchor_token` | anchor token，delete 为 `-1` |
| `center_hit` | anchor 位置是否命中 oracle 正位置 |
| `type_correct` | anchor 类型是否正确 |
| `token_correct` | anchor token 是否正确 |
| `event_set_correct` | 当前事件集合是否等于 oracle 正位置集合 |

注意：`event_set_correct` 是比较严格的指标。对于第一轮诊断，通常还应同时看 `center_hit`、`type_correct` 和 `token_correct`。

### 5.4 correlation 模式用法

```bash
PYTHONPATH=. python scripts/first_event_impact_analysis.py \
    --checkpoint checkpoints/USPTO_50K_PtoR_aug20_#global#/2026-06-08_17-20-39/checkpoint_stepXXXX.pt \
    --products_file train_subsets/USPTO_50K_PtoR_aug20_#global#/test/src-test.txt \
    --targets_file train_subsets/USPTO_50K_PtoR_aug20_#global#/test/tgt-test.txt \
    --vocab_file /data6/duanbh/desktop/retrosynthesis/dataset/USPTO_50K_PtoR_aug20_#global#/example.vocab.src \
    --output_dir analysis_outputs/first_step/global_train_subset/event_correlation \
    --mode correlation \
    --deduplicate 20 \
    --max_lines 1000 \
    --n_samples 10 \
    --n_steps 100 \
    --scheduler linear \
    --batch_size 32 \
    --device cuda
```

输出：

| 文件 | 内容 |
|------|------|
| `correlation_summary.json` | 条件概率与 top-1 exact match 汇总 |
| `per_sample_events.pt` | 每个 sample 的 first event、最终预测、target、final edit distance |
| `report.md` | 人读摘要 |

重点看：

- `P(final correct | first event set correct)`
- `P(final correct | first event set wrong)`
- `n_first_event_correct`
- `n_first_event_wrong`
- `top1_acc`

### 5.5 intervention 模式用法

```bash
PYTHONPATH=. python scripts/first_event_impact_analysis.py \
    --checkpoint checkpoints/USPTO_50K_PtoR_aug20_#global#/2026-06-08_17-20-39/checkpoint_stepXXXX.pt \
    --products_file train_subsets/USPTO_50K_PtoR_aug20_#global#/test/src-test.txt \
    --targets_file train_subsets/USPTO_50K_PtoR_aug20_#global#/test/tgt-test.txt \
    --vocab_file /data6/duanbh/desktop/retrosynthesis/dataset/USPTO_50K_PtoR_aug20_#global#/example.vocab.src \
    --output_dir analysis_outputs/first_step/global_train_subset/event_intervention \
    --mode intervention \
    --deduplicate 20 \
    --max_lines 1000 \
    --n_samples 10 \
    --n_steps 100 \
    --scheduler linear \
    --batch_size 32 \
    --device cuda
```

输出：

| 文件 | 内容 |
|------|------|
| `intervention_summary.json` | `normal` / `force_correct_first` / `force_wrong_first` 三组结果 |
| `per_sample_events.pt` | 三组模式下的 sample 明细 |
| `report.md` | 人读摘要 |

重点比较：

- `normal.top1_acc`
- `force_correct_first.top1_acc`
- `force_wrong_first.top1_acc`
- `force_correct_first` 相比 `normal` 的提升
- `force_wrong_first` 相比 `normal` 的下降

解释口径：

- 如果 `force_correct_first` 提升大，说明首步编辑是主要瓶颈之一
- 如果 `force_correct_first` 提升有限，说明后续多步速率预测也有明显问题
- 如果 `force_wrong_first` 明显拉低结果，说明模型对开局错误缺乏纠错能力

---

## 6. 推荐整体实验流程

建议按下面顺序使用这些脚本。

### Step 1：先做静态初始预测

先运行：

```text
scripts/first_step_forward_analysis.py
```

推荐先在已有 1000 条 deduplicated `#global#` 子集上跑：

- `t = 0`
- `t = 1e-3`
- `t = 1e-2`
- `t = 5e-2`
- `t = 0.1`
- `t = 0.2`
- `t = 0.3`

目的：

- 判断模型在固定 `x_t=x_0` 时是否能找到初始应编辑位置
- 判断 `t=0` 是否稳定，还是只有 `0.1~0.3` 较好
- 比较 `base` 和 `effective` 差异，确认 rate reparam 的影响

如果 `Center Hit@k` 和 `Full First-Edit Acc` 明显偏低，说明模型初始反应中心感知不足。

### Step 2：再做第一次真实编辑相关性

运行：

```text
scripts/first_event_impact_analysis.py --mode correlation
```

目的：

- 看第一次真实编辑通常发生在什么 `t`
- 看第一次真实编辑正确时，最终 exact match 是否显著更高
- 看第一次真实编辑错误时，模型后续是否还有补救能力

建议先用 `n_samples=10`，与现有生成评测口径更接近。

### Step 3：做首步干预

运行：

```text
scripts/first_event_impact_analysis.py --mode intervention
```

目的：

- 直接比较 `normal` vs `force_correct_first`
- 估计“只修正第一步”能带来多少收益
- 用 `force_wrong_first` 评估错误开局的破坏性

这一步是判断“第一步是否是主要瓶颈”的核心因果证据。

### Step 4：迁移到 test 随机子集

如果 train-subset 诊断信号清晰，建议新建正式 test 随机子集，例如：

```text
analysis_subsets/USPTO_50K_PtoR_aug20_#global#/test_dedup_seed42_1000/
```

包含：

- `src-test.txt`
- `tgt-test.txt`
- `meta.json`

然后完整复跑：

1. 静态初始预测
2. 第一次真实编辑相关性
3. 首步干预

test 子集结果更适合作为正式结论。

### Step 5：扩展对照实验

后续可以扩展到：

- 完整 test 集
- standard 数据集
- 不同 checkpoint
- `use_rate_reparam: false` 对照模型
- 不同 scheduler

重点比较：

- `t=0` 的 `Center Hit@k`
- `t=0` 的 `Full First-Edit Acc`
- 第一次真实编辑正确率
- `force_correct_first` 的边际收益

如果 `use_rate_reparam: true` 确实改善初始反应中心感知，预期会看到：

- 实验 1 中 `t=0` 指标更好
- 实验 2 中第一次真实编辑更准确
- `force_correct_first` 的额外提升变小

---

## 7. 注意事项

1. 不要把这些分析输出写到 checkpoint 的 `eval/` 目录。
   - 推荐使用 `analysis_outputs/first_step/...`

2. 不要用 `scripts/eval_retro.py` 跑这些实验。
   - `eval_retro.py` 会生成并覆盖 `predictions.txt`

3. 当前脚本统计的是 token-level Edit Flows 位置。
   - 文档中称作 `center`，但它不是严格的化学图反应中心原子标签
   - 它是当前模型空间下“应编辑 token 位置”的反应中心近似

4. 实验 2 同时受模型输出、scheduler、离散化和随机采样影响。
   - 因此实验 1 和实验 2 应分开解释

5. `force_correct_first` 当前强制的是 oracle anchor edit。
   - 不是强制完整 oracle event set
   - 后续如果需要，可以扩展为“完整 first event set 干预”

6. `force_wrong_first` 使用的是模型高分但 oracle 错误的编辑。
   - 这比随机错误更贴近真实失败模式

---

## 8. 当前实现状态

当前已完成：

- 静态初始预测分析脚本
- 第一次真实编辑记录
- 首步正确 / 错误干预
- 聚合 summary 输出
- per-sample 明细保存
- 基础单测

当前尚未运行实际实验。下一步建议先在已有 1000 条 `#global#` deduplicated 子集上运行实验 1，确认指标和输出格式符合预期，再进入实验 2。

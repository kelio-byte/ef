# Beam Search for Edit Flows：实现总结

## 1. 做了什么

在 Edit Flows 逆合成项目中新增了两种确定性/半确定性采样方式，作为随机 Euler 采样的替代：

- **Greedy single-edit**：每步选择条件概率最高的一个编辑，贪心推进
- **Beam single-edit**：每步保留 `beam_size` 条候选路径，按累积分数去重裁剪

核心近似：将原始 CTMC 的“每步多位置同时可能编辑”离散化为“每步只执行一个编辑”，在此离散化空间上做确定性搜索。

### 1.1 编辑分数

采用方案 B（条件概率评分）：

$$
\text{score}(e \mid x_t, t) = \log u^{\text{real}}(e \mid x_t, t) - \log u_{\text{tot}}^{\text{real}}(x_t, t)
$$

其中 `u_tot_real = Σ_i (λ_ins[i] + λ_sub[i] + λ_del[i])`，利用 `Σ_a Q(a|i) = 1` 精确计算，无需枚举所有 O(L*V) 候选。

### 1.2 时间推进

第一版实现两种模式：

| time_mode | 行为 | 用途 |
|-----------|------|------|
| `depth` | `t_k = k / max_edits` | 默认，线性对应编辑进度 |
| `fixed` | `t_k = const` | Ablation：检验时间是否重要 |

`utot_ratio` 等更复杂的模式留待后续实验。

### 1.3 停止条件

1. 达到 `max_edits` 硬上限
2. （可选）`u_tot_base < stop_u_tot_base`，用 base rate 而非 real rate 判断（避免 `k(t)` 在 t→1 时的放大效应）

**Greedy 模式**：stop 检查在候选选择之后、编辑执行之前，满足条件则直接标记 inactive，不执行额外编辑。

**Beam 模式**：stop 检查在**展开之前**进行。若 parent state 满足 stop 条件，parent 直接标记 finished 并保留到候选池，不再生成 child。当 parent 无可扩展候选（全被 reverse-op 过滤或候选集为空）时，parent 同样作为 finished 保留，不会丢失。已标记 finished 的 state 会在后续轮次中 carry over，与新建 child 一起参与 top-k 竞争。

### 1.4 约束与裁剪

- **BOS 保护**：禁止 `sub(pos=0, *)` 和 `del(pos=0)`；`ins(pos=0, *)` 解释为"在 BOS 后插入"，允许
- **no-op 过滤**：`sub(pos, token)` 若 `token == x_t[pos]`（替换为自身），直接屏蔽，不进入候选
- **逆操作禁止**（精确化）：
  - `sub(i, a→b)` 后禁止 `sub(i, b→a)`；但 `sub(i, a→b→c)` 合法多步修正**允许**
  - `ins(i, *)` 后禁止同位置 `del(i)`
  - `del(i, a)` 后禁止同位置同 token `ins(i, a)`
- **状态去重**：相同序列（strip PAD 后）只保留最高分
- **候选裁剪**：每位置 token top-k（`k_ins_token`/`k_sub_token`）→ 全局 edit top-k（`k_edit_expand`）
- **特殊 token 过滤**：PAD/BOS/GAP/UNK 不在 insert/substitute 的目标 token 中

---

## 2. 代码位置

| 文件 | 作用 |
|------|------|
| `edit_flows/sampling/beam.py` | 核心实现：数据结构、候选收集、单编辑应用、greedy/beam 采样入口 |
| `scripts/sample_retro.py` | CLI 入口：新增 `--sampler` 等参数，分发到不同采样器 |

未修改的复用模块：

| 文件 | 复用内容 |
|------|----------|
| `edit_flows/sampling/euler.py` | `_compute_model_time` |
| `edit_flows/sampling/ops.py` | `apply_ins_del_operations` |
| `edit_flows/core/rate_scale.py` | `apply_rate_parameterization`、`get_rate_scale` |
| `edit_flows/core/scheduler.py` | `KappaScheduler` 及其子类 |

---

## 3. 并行化策略

### Greedy 模式

所有 sample 在同一 batch 中同步推进。每步一次 model forward 处理 `(B_active, L_max)`，然后逐 sample 选最优编辑、batch 执行序列更新。已完成的 sample 通过 `active` mask 排除。

### Beam 模式

所有 sample × 所有活跃 beam 状态扁平化为 `(total_active, L_max)` 的大 batch，每步一次 forward。结果按 sample 拆分后独立展开、去重、top-k 裁剪。

GPU 利用率与原始 Euler 采样（`B * n_samples` 条序列并行）在同一量级。

---

## 4. 脚本使用方式

### 4.1 基本用法

```bash
# Greedy single-edit（默认参数）
python scripts/sample_retro.py \
  --checkpoint <path.pt> \
  --products_file <products.txt> \
  --device cuda \
  --sampler greedy_edit \
  --output_dir <dir>

# Beam single-edit
python scripts/sample_retro.py \
  --checkpoint <path.pt> \
  --products_file <products.txt> \
  --device cuda \
  --sampler beam_edit \
  --beam_size 5 \
  --output_dir <dir>

# 原始 Euler（默认，向后兼容）
python scripts/sample_retro.py \
  --checkpoint <path.pt> \
  --products_file <products.txt> \
  --device cuda \
  --sampler euler \
  --n_steps 100 \
  --output_dir <dir>
```

### 4.2 完整参数列表

| 参数 | 适用采样器 | 默认值 | 说明 |
|------|-----------|--------|------|
| `--sampler` | 全部 | `euler` | `euler` / `greedy_edit` / `beam_edit` |
| `--beam_size` | beam_edit | `5` | Beam 宽度 |
| `--max_edits` | greedy/beam | `20` | 最大编辑步数 |
| `--time_mode` | greedy/beam | `depth` | 时间推进模式：`depth` / `fixed` |
| `--time_const` | greedy/beam | `0.5` | `fixed` 模式下的固定 t 值 |
| `--k_ins_token` | greedy/beam | `4` | 每位置 insert token top-k |
| `--k_sub_token` | greedy/beam | `4` | 每位置 substitute token top-k |
| `--k_edit_expand` | greedy/beam | `16` | 全局 edit 候选 top-k |
| `--stop_u_tot_base` | greedy/beam | `-1.0` | `<0` 禁用；`>0` 为 base-rate 停止阈值 |
| `--n_steps` | euler | `100` | Euler 离散步数 |
| `--n_samples` | 全部 | `1` | 每个产物的独立采样数（greedy/beam 下为输出复制次数） |

### 4.3 典型实验命令

```bash
# Phase 1: Greedy vs Euler 快速对比
python scripts/sample_retro.py \
  --checkpoint <ckpt> --products_file <test.txt> \
  --device cuda --sampler greedy_edit --max_edits 15 \
  --output_dir eval/greedy

python scripts/sample_retro.py \
  --checkpoint <ckpt> --products_file <test.txt> \
  --device cuda --sampler euler --n_steps 100 \
  --output_dir eval/euler

# Phase 2: 时间模式 ablation
python scripts/sample_retro.py \
  --checkpoint <ckpt> --products_file <test.txt> \
  --device cuda --sampler greedy_edit \
  --time_mode fixed --time_const 0.5 \
  --output_dir eval/greedy_fixed

# Phase 3: Beam vs Greedy
python scripts/sample_retro.py \
  --checkpoint <ckpt> --products_file <test.txt> \
  --device cuda --sampler beam_edit --beam_size 5 --max_edits 15 \
  --output_dir eval/beam5

# 启用 base-rate 提前停止
python scripts/sample_retro.py \
  --checkpoint <ckpt> --products_file <test.txt> \
  --device cuda --sampler greedy_edit \
  --stop_u_tot_base 0.1 \
  --output_dir eval/greedy_stop
```

### 4.4 评测

采样结果文件（`predictions.txt`）格式与 Euler 采样一致，可直接用现有评测脚本：

```bash
python scripts/score.py --predictions eval/greedy/predictions.txt ...
python scripts/score_#global#.py --predictions eval/greedy/predictions.txt ...
```

### 4.5 注意事项

1. Greedy/beam 是确定性的，`--n_samples 5` 会产生 5 行相同输出（仅用于与 Euler 评测格式兼容）
2. 当前 `time_mode` 只有 `depth` 和 `fixed`，更复杂的 `utot_ratio` / `chosen_rate` / `expected_wait` 尚未实现
3. stop 阈值 `stop_u_tot_base` 需要根据模型实际输出的 `u_tot_base` 量级来调；Beam 模式下 stop/dead-end parent 会自动保留，不会丢失
4. beam_size 越大越慢，建议从 3~5 开始实验
5. 模型 checkpoint 的 `use_origin_mask` 设置需与训练时一致；若 config 与权重不一致会自动检测并回退

---

## 7. 建议的实验顺序

1. **Greedy + depth vs Euler**：先看单编辑贪心是否优于随机 Euler
2. **Greedy + depth vs Greedy + fixed**：检验时间信号是否重要
3. **Greedy vs Beam（beam_size=3,5,10）**：检验 beam 是否带来额外收益
4. **调 stop_u_tot_base**：看提前停止能否减少无效编辑
5. **后续**：`utot_ratio` 时间模式、`edit_score_mode` ablation

---

## 5. 与 beam-search/todo1.md 的对应关系

| todo 文档方案 | 实现状态 |
|--------------|---------|
| 方案 A（log_u 评分） | 未实现（`score` 字段固定为 cond_prob 评分） |
| **方案 B（cond_prob 评分）** | **已实现，作为唯一评分方式** |
| 方案 C（CTMC + dt） | 未实现 |
| 方案 D（时间折叠） | `time_mode=fixed` 已覆盖 |
| 方案 E（A*） | 未实现 |
| time_mode: depth | 已实现 |
| time_mode: fixed | 已实现 |
| time_mode: utot_ratio | 未实现 |
| time_mode: chosen_rate | 未实现 |
| time_mode: expected_wait | 未实现 |
| time_mode: euler | 未实现 |
| 逆操作禁止 | 已实现（精确化：sub 只禁真正回退，ins/del 精确位置匹配） |
| 重复状态去重 | 已实现（beam 模式） |
| 候选两级裁剪 | 已实现 |
| BOS sub/del 保护 | **已实现**（审查后补充） |
| no-op substitution 过滤 | **已实现**（审查后补充） |
| Beam stop 语义修正 | **已实现**（展开前判定，保留 parent） |
| Beam state 保活 | **已实现**（dead-end/finished 不丢失） |
| STOP 规则停止 | 已实现（max_edits + u_tot_base 阈值） |
| 显式 STOP 动作 | 未实现 |
| CLI 参数化 | 已实现（11 个新参数） |

---

## 6. 实现审查与修复记录

首次实现后（`docs/beam-search/exp1.md`）进行了代码审查（`docs/beam-search/todo2.md`），发现并修复了以下问题：

### 6.1 Beam stop 语义错误

**问题**：stop 判定在展开 candidate 并执行编辑**之后**才标记 child.is_finished。正确语义应是 parent 满足 stop → 保留 parent，不展开。

**修复**：stop 检查移到 for cand 循环之前。parent 满足条件时直接标记 finished 并加入候选池，不执行额外编辑。

### 6.2 Beam state 丢失

**问题**：parent state 在以下情况会直接消失：(a) 所有候选被 reverse-op 过滤；(b) 候选集为空。上一轮已 finished 的 state 也不会 carry over。最终可能回退输出 `x_0`。

**修复**：每轮开始时 carry over 已 finished 的 states；dead-end parent 转为 finished 保留；不再有回退到 `x_0` 的 fallback 路径。

### 6.3 BOS sub/del 未禁止

**问题**：候选收集仅基于 `non_pad_mask` 过滤位置，BOS（pos=0）可被 sub/del 改写。在单编辑搜索中一步即可毁掉整条序列，是实验 #1 高 Invalid SMILES（51-60%）的极可能原因。

**修复**：`_collect_edit_candidates_single` 中显式过滤 `sub(pos=0, *)` 和 `del(pos=0)`。`ins(pos=0)` 保留（插入到 BOS 之后）。

### 6.4 no-op substitution 未过滤

**问题**：`sub(pos, current_token)` 可进入候选，消耗编辑预算并干扰排序。

**修复**：在 `log_u_sub` 中 mask 掉 `token == x_t[pos]` 的候选。`_collect_edit_candidates_single` 新增 `x_t` 参数。

### 6.5 reverse-op 规则过宽

**问题**：(a) 同位置连续 substitute 全禁，误杀合法多步修正 a→b→c；(b) ins/del 使用 `abs(pos ± 1)` 邻域近似，位置漂移后不精确。

**修复**：`EditCandidate` 新增 `old_token` 字段（sub 候选自动填入）。`_is_reverse_op` 改为精确判定：
- sub→sub 检查 `edit.token == last_edit.old_token`（只禁真正回退）
- ins→del / del→ins 改为精确位置匹配

### 6.6 测试补充

新增 `tests/sampling/test_beam.py`，30 个测试覆盖以上所有修复点及核心函数 (`_is_reverse_op`, `_collect_edit_candidates_single`, `_apply_single_edit_to_sequence`, `sample_greedy_single_edit`, `sample_beam_single_edit`)。

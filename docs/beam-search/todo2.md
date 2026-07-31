# Beam Search 实现审查：当前问题与测试待补清单

## 1. 目的

本文档总结当前 `edit_flows/sampling/beam.py` 实现中，审查时发现的主要问题，以及下一步应优先补充的测试。

重点不是讨论“单编辑 greedy / beam 这个方向是否最终有效”，而是先把**明显的实现风险**和**可验证的行为约束**列清楚。当前 `0%` 的结果不能直接视为方法失败，因为实现层面仍存在多处足以严重拉坏结果的逻辑问题。

---

## 2. 当前主要问题

## 2.1 Beam 的 stop 逻辑语义错误

当前 `sample_beam_single_edit` 中，`stop_u_tot_base` 的判定发生在：

1. 已经从 parent state 取出 candidate
2. 已经对 candidate 执行 `_apply_single_edit_to_sequence`
3. 生成了 child state

之后才把 `child.is_finished = True`。

这意味着：

- 正确语义应是：**当 parent state 已满足 stop 条件时，不再展开，直接保留 parent state**
- 当前实现却变成：**先额外做一次 edit，再把改坏后的 child 标记为 finished**

这会系统性破坏 stop 规则，一旦启用 `stop_u_tot_base`，结果会被明显污染。

### 应改成

- stop 判定应在展开 candidate 之前进行
- 若 parent state 满足 stop 条件，应直接把 parent state 作为 finished state 保留下来
- 不应再额外生成任何 child

---

## 2.2 Beam 无法展开时会把 state 直接丢失

当前 beam 每轮只保留新生成的 child states。

如果某个 parent state 出现以下任一种情况：

- 没有可用候选
- 候选全被 reverse-op 规则过滤
- 命中 stop 条件而不应继续展开

那么该 parent state 当前实现下不会自动保留下来，而是直接从下一轮 beam 中消失。

进一步地，如果一个 sample 的所有 beam state 都消失，最终代码会回退输出 `x_0`。

这会把：

- “搜索无法继续”

错误地变成：

- “输出原始输入”

这是严重的 beam 语义错误。

### 应改成

- 无法展开时，parent state 应转为 finished state 并保留
- 下一轮 beam 应同时包含：
  - 新展开出的 child states
  - 本轮保留下来的 finished / dead-end parent states
- 只有在 top-k 裁剪后自然被更高分状态淘汰时，才允许其离开 beam

---

## 2.3 当前候选集没有禁止对 BOS 做 sub / del

当前候选位置过滤仅基于 `non_pad_mask`，没有显式排除 `pos=0`（BOS 位置）。

虽然 token 分布里禁止生成 `BOS/PAD/GAP/UNK`，但这并不能阻止：

- `sub` 作用在 BOS 上
- `del` 删除 BOS

对于该项目当前的序列表示，BOS 是结构性前缀，不应被搜索器改写或删除。

这类错误在 Euler 里可能只是低概率噪声，但在单编辑 greedy / beam 中，一步走错就可能整条路径报废，并显著提升 invalid SMILES。

### 应改成

- 明确禁止 `sub(pos=0, *)`
- 明确禁止 `del(pos=0)`
- `ins(pos=0, a)` 是否允许，需要统一语义：
  - 如果 `pos=0` 表示“在 BOS 后插入”，则允许
  - 否则也应禁掉

建议先沿用当前 `apply_ins_del_operations` 的语义，把 `ins(pos=0)` 解释为“在 BOS 后插入”，但仍禁止对 BOS 本身替换和删除。

---

## 2.4 当前候选集没有过滤 no-op substitution

当前 substitute 候选生成时，没有排除：

- `sub(pos, current_token)`

即“把当前位置替换成它自己”。

在随机 Euler 中，这类无效事件只是并行随机过程中的一部分；但在单编辑搜索中，这会：

- 消耗一整步编辑预算
- 干扰 greedy 排序
- 污染 beam 扩展空间

如果模型在 unchanged 位置对当前 token 给出较高 `Q_sub`，这类 no-op 很容易进入 top-k。

### 应改成

- 生成 substitute 候选时，显式过滤 `token == x_t[pos]`

---

## 2.5 reverse-op 规则过宽，误杀合法路径

当前 `_is_reverse_op` 的规则比“防止立刻撤销上一步”更强，存在误杀：

### 现状问题 1

任意“同一位置上的连续 substitute”都被当成 reverse：

- 现在是：`sub(i, *)` 后，下一步任何 `sub(i, *)` 都禁
- 正确应是：只禁真正的 `a -> b` 后立刻 `b -> a`

否则会错误禁止：

- `a -> b -> c`

这类本来合法的多步修正路径。

### 现状问题 2

`ins` 后相邻 `del` 一律禁掉，位置规则过于粗糙：

- 当前使用 `abs(edit.pos - last_edit.pos) <= 1`

但插入前后的坐标会变化，相邻位置不一定表示“撤销上一操作”。

### 应改成

- substitute 只禁止真正的反向替换
- insert/delete 的逆操作约束应更精确，至少不要用当前这么宽的邻域近似
- 如果短期内不想精确追踪 token 历史，宁可先把规则放松，也不要过度误杀

---

## 2.6 Greedy/Beam 新增逻辑缺少单元测试保护

当前仓库已有对 Euler 和 origin mask 的测试，但没有看到针对 `beam.py` 的系统测试。

这导致以下关键行为目前没有自动回归保护：

- BOS 不应被改写
- no-op substitute 不应进入候选
- stop 语义是否正确
- beam 在无候选时是否保留 parent state
- reverse-op 规则是否只禁真正回退

在当前阶段，这些测试是必要的，不然实验结果很难区分“模型问题”和“采样器 bug”。

---

## 3. 建议优先修复顺序

建议按下面顺序处理：

1. 修 `beam` 的 state 保活语义
2. 修 `beam` 的 stop 语义
3. 禁止对 BOS 做 `sub/del`
4. 过滤 no-op substitution
5. 收紧 reverse-op 规则
6. 补测试后再做小规模复测

原因：

- 前两项会直接改变 beam 的正确性
- 第三、四项最像会显著导致 invalid 和 0% 命中
- 第五项更多是“剪枝过猛”问题，优先级略低于前四项

---

## 4. 必须补的测试

以下测试建议新增到 `tests/sampling/test_beam.py`。

## 4.1 候选生成：禁止对 BOS 做 substitute / delete

### 目的

验证候选收集函数不会返回：

- `sub(pos=0, *)`
- `del(pos=0)`

### 建议形式

- 构造一个很短的假序列：`[BOS, A, B]`
- 人工构造 `log_rates/log_ins_probs/log_sub_probs`
- 让 `pos=0` 的 `sub/del` 分数最高
- 验证最终候选列表中仍不包含这两类编辑

---

## 4.2 候选生成：过滤 no-op substitution

### 目的

验证 `sub(pos, current_token)` 不会作为候选返回。

### 建议形式

- 构造序列 `[BOS, A, B]`
- 令 `sub(pos=1, token=A)` 分数最高
- 验证该候选被过滤，返回的是下一个合法候选

---

## 4.3 `_is_reverse_op`：允许合法的连续 substitute

### 目的

避免当前“同位置连续 substitute 全禁”的误杀。

### 建议形式

- 构造 `last_edit = sub(pos=3, token=B)`
- 当前 token 已变为 `B`
- 新候选为：
  - `sub(pos=3, token=A)` 应视为 reverse
  - `sub(pos=3, token=C)` 不应视为 reverse

这要求 reverse-op 判断不仅看位置，还要能区分“回退到上一个 token”和“继续改成第三个 token”。

---

## 4.4 Beam stop：命中 stop 时保留 parent，不生成 child

### 目的

验证 stop 语义修正后，beam 会保留原状态。

### 建议形式

- 构造一个 dummy model，使 `u_tot_base < stop_u_tot_base`
- 同时让某个 edit 候选分数很高
- 调用 `sample_beam_single_edit`
- 验证输出仍等于输入 `x_0`，而不是多做了一步 edit

---

## 4.5 Beam dead-end：无候选时保留 parent state

### 目的

验证 beam 在无可扩展候选时不会把状态丢失，更不会回退成错误默认值。

### 建议形式

- 构造 dummy model，使所有候选都非法或被过滤掉
- 初始状态设为一个非平凡序列
- 调用 `sample_beam_single_edit`
- 验证输出仍为该初始状态，而不是空 beam 后回退逻辑造成的异常结果

---

## 4.6 Greedy：禁止选中 BOS 上的 edit

### 目的

验证 greedy 采样的最终执行 edit 不会落在 BOS 的 `sub/del` 上。

### 建议形式

- 构造 dummy model，让 BOS 位置的 `sub/del` 打分最高
- 让另一个普通位置存在次优合法编辑
- 调用 `sample_greedy_single_edit`
- 验证实际执行的是普通位置的合法编辑

---

## 4.7 Greedy：no-op substitute 不消耗一步

### 目的

验证 greedy 不会把一步预算浪费在 `sub(pos, same_token)` 上。

### 建议形式

- dummy model 令 no-op substitute 分数最高
- 另一个真实 edit 分数次高
- 调用 greedy 一步
- 验证实际输出发生了真实变化，而不是原地不动

---

## 4.8 Origin mask：单编辑 apply 与 Euler 语义一致

### 目的

虽然 beam 里复用了 Euler 的三值 marker 逻辑，但仍建议单独验证 `_apply_single_edit_to_sequence` 的语义。

### 建议形式

分别覆盖：

- substitute 后对应位置变 `False`
- insert 的新 token 为 `False`
- delete 后对应 token 与 mask 一起移除

这类测试可直接对 `_apply_single_edit_to_sequence` 做单测，不必每次都走完整 beam/greedy 主流程。

---

## 5. 可选补充测试

这些不是当前最急，但后面建议加上。

## 5.1 Beam 去重：同序列保留最高分

验证两个不同 parent 扩展出同一 `x_t` 时，beam 只保留高分版本。

## 5.2 Beam top-k 裁剪稳定性

验证同一 sample 下超过 `beam_size` 个候选时，最终保留的是分数最高的 `beam_size` 个。

## 5.3 输出格式兼容性

验证 `scripts/sample_retro.py` 在 `greedy_edit/beam_edit` 下仍按 `n_samples` 正确复制输出行数，兼容现有 `score.py` / `score_#global#.py`。

---

## 6. 结论

当前 beam / greedy 实现里，至少有几类问题不能简单归因为“方法本身不适合”：

- beam 停止与状态保活语义不正确
- 候选集约束不完整（BOS / no-op substitute）
- reverse-op 剪枝过猛
- 缺少回归测试

因此，下一步更合理的路径是：

1. 先修实现语义
2. 先补最小必要测试
3. 再用小子集复测 greedy / beam
4. 最后再判断结果差是否主要来自模型速率不够尖锐

在这些修复完成前，不建议直接把当前 `0%` 结果当作该方向的最终结论。

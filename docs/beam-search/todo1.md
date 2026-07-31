# Beam Search for Edit Flows：思路细化与实现 TODO

## 1. 背景与目标

当前项目的采样方式是 [sample_euler](/data1/duanbh/desktop/edit-flows/edit_flows/sampling/euler.py) 中的随机 Euler 采样：

- 给定当前序列 `x_t`
- 模型输出每个位置的三类编辑总速率 `lambda_ins / lambda_sub / lambda_del`
- 再输出插入/替换 token 分布 `Q_ins / Q_sub`
- 在一个 Euler 步内，多个位置可同时触发编辑，最终通过 [apply_ins_del_operations](/data1/duanbh/desktop/edit-flows/edit_flows/sampling/ops.py) 一次性更新序列

这种方式的优点是忠实于原始 CTMC + Euler 近似，但缺点也很明显：

1. **完全随机**，没有类似自回归 beam search 的“保留多条高分候选路径”的机制
2. **一步可发生多个 edit**，使得“路径评分”不容易定义
3. 实际逆合成数据上，product/reactant 的编辑距离通常较小，很多 Euler 小步仅仅是在浪费预算

本文目标是把“Edit Flows 能否做 beam-like search”这件事拆成几个可实现方案，并与当前代码实现对齐，供下一阶段开发。

---

## 2. 问题本质：Edit Flows 没有 AR 概率，但有事件强度

自回归模型天然有：

$$
\log p(y) = \sum_k \log p(y_k \mid y_{<k}, x)
$$

因此 beam search 可以直接按 token 条件概率累积分数。

Edit Flows 不直接给出 token 条件概率，而是给出**连续时间编辑过程中的瞬时速率**。对当前状态 `x_t`，模型定义所有 possible edit 的速率：

- 插入：`u_ins(i, a)`
- 替换：`u_sub(i, a)`
- 删除：`u_del(i)`

其中：

$$
u_{\text{ins}}(i,a) = \lambda_{\text{ins}}(i) \cdot Q_{\text{ins}}(a \mid i)
$$

$$
u_{\text{sub}}(i,a) = \lambda_{\text{sub}}(i) \cdot Q_{\text{sub}}(a \mid i)
$$

$$
u_{\text{del}}(i) = \lambda_{\text{del}}(i)
$$

因此，Edit Flows 的“局部偏好”不是 token 概率，而是**下一次编辑事件更倾向于发生在哪里、发生什么**。

这意味着 beam search 若要成立，需要先回答：

1. 如何从速率 `u` 定义单步分数？
2. 如何把连续时间 `t` 纳入路径评分？
3. 如何定义停止条件？

---

## 2.5 符号约定：base rate、real rate 与总速率

后文会同时讨论模型原始输出、rate reparam 后的真实速率、以及基于它们定义的 `u_tot`。这些符号容易混淆，因此先约定。

### 2.5.1 模型原始输出对应的速率

记：

$$
u^{\text{base}}(e \mid x,t)
$$

表示模型原始输出对应的 edit 强度。

对具体三类 edit：

$$
u_{\text{ins}}^{\text{base}}(i,a) = \lambda_{\text{ins}}^{\text{base}}(i) \cdot Q_{\text{ins}}(a \mid i)
$$

$$
u_{\text{sub}}^{\text{base}}(i,a) = \lambda_{\text{sub}}^{\text{base}}(i) \cdot Q_{\text{sub}}(a \mid i)
$$

$$
u_{\text{del}}^{\text{base}}(i) = \lambda_{\text{del}}^{\text{base}}(i)
$$

若 `use_rate_reparam = false`，则 `base` 速率就是最终真实速率。

若 `use_rate_reparam = true`，则 `base` 速率还没有乘上 scheduler 的 rate scale。

### 2.5.2 rate scale

记：

$$
k(t) = \frac{\kappa'(t)}{1-\kappa(t)}
$$

这是理想形式。当前代码里若启用了 `clamp_kappa` / `clamp_max`，则真正使用的是相应的 clamped 版本，见 [rate_scale.py](/data1/duanbh/desktop/edit-flows/edit_flows/core/rate_scale.py)。

后文为简洁起见，统一记为 `k(t)`，默认指“当前配置下真正生效的 rate scale”。

### 2.5.3 真实 CTMC 速率

记：

$$
u^{\text{real}}(e \mid x,t)
$$

表示真正用于 CTMC 事件采样的 edit 强度。

定义为：

$$
u^{\text{real}}(e \mid x,t)=
\begin{cases}
u^{\text{base}}(e \mid x,t), & \text{if use\_rate\_reparam = false} \\
k(t)\,u^{\text{base}}(e \mid x,t), & \text{if use\_rate\_reparam = true}
\end{cases}
$$

这与当前采样实现一致：先做 `apply_rate_parameterization(...)`，再进入真正的 Euler 事件采样，见 [euler.py#L116](/data1/duanbh/desktop/edit-flows/edit_flows/sampling/euler.py#L116)。

### 2.5.4 总速率

后文若不特别说明，`u_tot` 一律指：

$$
u_{\text{tot}}^{\text{real}}(x,t)
= \sum_{e \in \mathcal{E}(x)} u^{\text{real}}(e \mid x,t)
$$

即真实总速率。

若需要表示模型原始输出对应的总速率，则显式写为：

$$
u_{\text{tot}}^{\text{base}}(x,t)
= \sum_{e \in \mathcal{E}(x)} u^{\text{base}}(e \mid x,t)
$$

两者关系为：

$$
u_{\text{tot}}^{\text{real}}(x,t)=
\begin{cases}
u_{\text{tot}}^{\text{base}}(x,t), & \text{if use\_rate\_reparam = false} \\
k(t)\,u_{\text{tot}}^{\text{base}}(x,t), & \text{if use\_rate\_reparam = true}
\end{cases}
$$

### 2.5.5 为什么必须区分这两个量

以下概念都应基于 **真实速率**，而不是 base rate：

1. 下一事件条件概率 `u_real(e) / u_tot_real`
2. waiting time / expected wait
3. 任何试图模拟真实 CTMC 时间推进的规则
4. stop 阈值中基于“系统是否接近静止”的判断

否则在 `use_rate_reparam = true` 时，会错误忽略 `k(t)` 对真实事件尺度的影响。

---

## 3. 与当前实现对齐：现有代码已经提供了哪些积木

目前实现里，做 beam 搜索并不需要重写所有东西，关键积木已经存在。

### 3.1 模型输出已经足够构造所有候选 edit

[EditFlowsTransformer.forward](/data1/duanbh/desktop/edit-flows/edit_flows/models/transformer.py) 返回：

- `log_rates`: `(B, L, 3)`
- `log_ins_probs`: `(B, L, V)`
- `log_sub_probs`: `(B, L, V)`

可直接组合成：

$$
\log u_{\text{ins}}(i,a) = \log \lambda_{\text{ins}}(i) + \log Q_{\text{ins}}(a\mid i)
$$

$$
\log u_{\text{sub}}(i,a) = \log \lambda_{\text{sub}}(i) + \log Q_{\text{sub}}(a\mid i)
$$

$$
\log u_{\text{del}}(i) = \log \lambda_{\text{del}}(i)
$$

这和训练里 [train_step](/data1/duanbh/desktop/edit-flows/edit_flows/training/trainer.py) 构造 `log_u_tia_*` 的方式完全一致。

### 3.2 rate reparam / cross-scheduler 逻辑可直接复用

当前 [sample_euler](/data1/duanbh/desktop/edit-flows/edit_flows/sampling/euler.py) 在真正采样前，已经完成了：

- `time_input` 映射
- 跨 scheduler 时的 `t -> t_model` 修正
- `use_rate_reparam` 下的 `apply_rate_parameterization`
- 必要时的 raw-rate correction

因此 beam 版本不应重新发明这些逻辑，最稳妥的方式是复用：

- `_compute_model_time`
- `apply_rate_parameterization`
- `get_rate_scale`

换句话说，beam sampler 的前半段应尽量与现有 `sample_euler` 保持一致，只替换“如何从 `log_u_real` 里选编辑”的后半段。

### 3.3 单个 edit 的应用也已有底层实现

[apply_ins_del_operations](/data1/duanbh/desktop/edit-flows/edit_flows/sampling/ops.py) 支持：

- insert
- delete
- replace（通过 `ins_mask & del_mask`）

虽然现有实现是为“多位置并行更新”设计的，但单编辑完全可以退化成：

- `ins_mask` 只有一个位置为 True
- `del_mask` 只有一个位置为 True
- `ins_tokens` 只有一个位置有效

因此 beam search 的状态转移可直接复用该函数。

### 3.4 origin mask 的状态演化逻辑也已存在

若 checkpoint 使用 `use_origin_mask: true`，beam sampler 也必须同步追踪 `origin_mask`，否则采样语义与训练不一致。

当前 [sample_euler](/data1/duanbh/desktop/edit-flows/edit_flows/sampling/euler.py) 已实现：

- substitute 后该位置变 `False`
- insert 新 token 标记为 `False`
- delete 时 token 连同 mask 一起删掉

beam sampler 应直接复用这套三值 marker 逻辑，而不是重新写一套简化版。

---

## 4. 方案 A：高强度编辑路径搜索（最简单、最易落地）

### 4.1 基本近似

将连续时间问题先离散化为“每一步只允许一个 edit”，记第 `k` 步状态为 `x^{(k)}`。当前状态下所有可选动作：

- `insert(i, a)`
- `substitute(i, a)`
- `delete(i)`

定义该动作的局部分数为：

$$
s(e \mid x, t) = \log u^{\text{real}}(e \mid x, t)
$$

一条路径 `\pi = (e_1, e_2, ..., e_K)` 的总分为：

$$
S(\pi) = \sum_{k=1}^{K} \log u^{\text{real}}(e_k \mid x^{(k-1)}, t_k)
$$

然后做标准 beam search：

1. 初始 beam 只有 `x_0`
2. 每一步对每个 beam state 取 top-M edits
3. 展开后保留总分 top-B 的候选
4. 达到 `max_edits` 或触发 stop 时结束

### 4.2 优点

- 最容易实现
- 不需要先把 CTMC 的路径概率严格推完
- 非常适合“小编辑距离”数据

### 4.3 缺点

- `u_real(e)` 是瞬时强度，不是严格条件概率
- 会偏好“总速率大”的状态，即便该状态并不稳定

### 4.4 适合用作什么

最适合做第一版 sanity check：

- 看纯 greedy 是否已经优于随机 Euler
- 看 beam 是否能显著提升 Top-1 / Top-K

### 4.5 与当前实现的对应开发项

建议新建：

- `edit_flows/sampling/beam.py`

其中先实现一个最小版本：

- `collect_edit_candidates(...)`
- `apply_single_edit(...)`
- `sample_greedy_single_edit(...)`
- `sample_beam_single_edit(...)`

第一版完全可以不碰 [sample_retro.py](/data1/duanbh/desktop/edit-flows/scripts/sample_retro.py) 现有默认逻辑，而是通过新的 CLI 开关切换。

---

## 5. 方案 B：基于“下一事件条件概率”的 beam search（最推荐）

### 5.1 从 CTMC 出发的更合理定义

对当前状态 `x_t`，记所有 edit 的总速率为：

$$
u_{\text{tot}}^{\text{real}}(x_t, t) = \sum_{e \in \mathcal{E}(x_t)} u^{\text{real}}(e \mid x_t, t)
$$

连续时间马尔可夫链中，**下一次事件类型**的条件分布为：

$$
p(e \mid \text{next event happens}, x_t, t)
= \frac{u^{\text{real}}(e \mid x_t, t)}{u_{\text{tot}}^{\text{real}}(x_t, t)}
$$

因此最自然的 beam 单步分数是：

$$
s(e \mid x_t, t) = \log u^{\text{real}}(e \mid x_t, t) - \log u_{\text{tot}}^{\text{real}}(x_t, t)
$$

这相当于把局部强度归一化成“下一次唯一发生的编辑是谁”的条件概率。

### 5.2 路径分数

若路径 `\pi = (e_1, ..., e_K)`，则：

$$
S(\pi) = \sum_{k=1}^{K}
\left[
\log u^{\text{real}}(e_k \mid x^{(k-1)}, t_k)
- \log u_{\text{tot}}^{\text{real}}(x^{(k-1)}, t_k)
\right]
$$

也可写成：

$$
S(\pi) = \sum_{k=1}^{K}
\log \frac{u^{\text{real}}(e_k \mid x^{(k-1)}, t_k)}{u_{\text{tot}}^{\text{real}}(x^{(k-1)}, t_k)}
$$

### 5.3 为什么它更像自回归 beam

自回归 beam 用的是：

$$
\sum_k \log p(y_k \mid y_{<k}, x)
$$

这里则是：

$$
\sum_k \log p(e_k \mid \text{next event}, x^{(k-1)}, t_k)
$$

即把“生成下一个 token”换成了“生成下一个 edit”。

这比方案 A 更接近概率搜索，也更不容易偏向“总速率大但编辑分散”的状态。

### 5.4 工程意义

这是我目前最推荐的主线方案。理由：

1. 数学上最干净
2. 与当前模型输出完全兼容
3. 不需要显式模拟 waiting time
4. 很适合你提出的“每一步只进行一次编辑”

### 5.5 候选 edit 的具体构造

对每个位置 `i`：

- insertion 候选：
  - 先取 `log_lambda_ins[i] + topk(log_Q_ins[i])`
- substitution 候选：
  - 先取 `log_lambda_sub[i] + topk(log_Q_sub[i])`
- deletion 候选：
  - 直接取 `log_lambda_del[i]`

然后拼成全局候选集合 `E(x_t)`，并计算：

$$
\log u_{\text{tot}}^{\text{real}} = \log \sum_{e \in E(x_t)} \exp(\log u^{\text{real}}(e))
$$

最后每个候选用：

$$
\text{score}(e) = \log u^{\text{real}}(e) - \log u_{\text{tot}}^{\text{real}}
$$

### 5.6 重要实现细节

为了避免候选空间过大，必须做截断：

- 对每个位置，insert 只保留 top-`k_ins_token`
- 对每个位置，substitute 只保留 top-`k_sub_token`
- 对全局 edit，再保留 top-`k_edit_expand`

否则复杂度会是 `O(L * V)`，对词表较大时不现实。

---

## 6. 方案 C：把 waiting time 一并纳入分数（更严格，但更复杂）

### 6.1 小时间步近似

若仍保留一个显式离散时间步 `dt`，则当前状态下某个具体 edit 在该小步内发生的概率近似为：

$$
p(e \text{ in } [t, t+dt]) \approx u^{\text{real}}(e \mid x_t, t) \cdot dt
$$

与此同时，“这一小步内没有事件发生”的概率为：

$$
p(\text{no event}) \approx \exp(-u_{\text{tot}}^{\text{real}} dt)
$$

因此某个 edit 的更严格单步对数分数可写成：

$$
s(e \mid x_t, t)
\approx \log u^{\text{real}}(e \mid x_t, t) - u_{\text{tot}}^{\text{real}}(x_t, t) \cdot dt + \log dt
$$

若不同候选共用同一个 `dt`，`log dt` 是常数，可忽略，于是：

$$
s(e \mid x_t, t)
\approx \log u^{\text{real}}(e \mid x_t, t) - u_{\text{tot}}^{\text{real}}(x_t, t) \cdot dt
$$

### 6.2 解释

这个分数比方案 B 多考虑了“等待代价”：

- `u_real(e)` 大，鼓励该 edit
- `u_tot_real` 大，则系统更容易在该步发生任意事件，也更容易扰动，因而会带来额外惩罚

### 6.3 难点

关键问题是 `dt` 如何选：

1. 固定 `dt = 1 / n_steps`
2. 用当前 [get_adaptive_h](/data1/duanbh/desktop/edit-flows/edit_flows/sampling/euler.py) 的 `adapt_h`
3. 由 beam depth 映射成一个 schedule

不同 `dt` 会显著影响分数尺度，因此实现上比方案 B 更难调。

### 6.4 判断

这个方案更“理论正确”，但不适合第一版直接上。建议作为方案 B 之后的 ablation：

- B: `log u_real - log u_tot_real`
- C: `log u_real - u_tot_real * dt`

比较两者在 Top-1 / invalid / unique 上的差异。

---

## 7. 方案 D：时间折叠 / 弱时间依赖搜索

### 7.1 动机

如果实验上发现：对同一个 `x`，模型输出的编辑偏好与 `t` 关系不大，那么可以进一步近似为：

$$
u(e \mid x_t, t) \approx u(e \mid x)
$$

此时搜索问题可简化为一个离散状态图搜索：

- 节点：当前序列 `x`
- 边：单个 edit
- 边权：由 `u(e \mid x)` 定义

### 7.2 三种具体做法

#### 做法 D1：固定时间

每一步都用固定时间送入模型，例如：

- `t = 0.5`
- 或固定 `kappa = 0.5`

优点：

- 实现最简单
- 直接验证“t 是否真的不重要”

缺点：

- 偏离训练分布

#### 做法 D2：按编辑深度映射时间

如果最大编辑步数为 `K`，第 `k` 步用：

$$
t_k = \frac{k}{K}
$$

或用 scheduler 映射后定义 `kappa_k`。

这种方式保留了“越往后越接近 target”的粗略语义，但不再需要 Euler 的密集小步。

#### 做法 D3：事件驱动时间推进

每发生一次 edit，令 `t` 增加一个固定粗步长：

$$
t_{k+1} = \min(1, t_k + \Delta)
$$

例如：

- `\Delta = 1 / max_edits`

这是一种折中的离散化方式，通常比固定 `t` 更自然。

### 7.3 判断

这是一个很值得做的实验轴，因为它直接测试你的观察：

- “若输入 token 不变，编辑速率与 t 的关系不是特别大”

建议第一阶段就做：

1. 固定 `t=0.5`
2. 深度线性推进 `t_k = k / K`
3. 真实 Euler 风格推进

看三者性能差异。

---

## 8. 方案 E：A* / best-first search（长期方向）

### 8.1 形式

把问题写成：

- `g(x)`: 已执行路径的累计得分
- `h(x)`: 从当前状态到“好终点”的启发式估计

按：

$$
f(x) = g(x) + h(x)
$$

做 best-first / A* 搜索。

### 8.2 当前难点

逆合成采样时不知道 target，因此不能使用 oracle edit distance 作为 `h(x)`。

可选 heuristic 只能来自模型内部或先验，例如：

1. `-log u_tot_real(x,t)` 作为“趋于静止”的偏好
2. 一个额外训练的 value head，预测“离可接受 reactant 还有多远”
3. 基于 SMILES 规则的合法性/稳定性打分

### 8.3 判断

这是一个可能很强的长期方向，但不适合现在立即实现。当前阶段先把 beam search 跑通更重要。

---

## 9. STOP 动作：必须单独设计

自回归模型天然有 EOS；编辑搜索没有这个机制，因此 beam search 必须显式定义 `STOP`。

### 9.1 最简单的规则停止

当下面任一条件成立时停止：

1. `u_tot_real < threshold`
2. 达到 `max_edits`
3. 连续若干步最优 edit 分数很低

优点：

- 简单
- 不需要修改模型

缺点：

- 不是搜索空间中的显式动作，路径分数不完全统一

### 9.2 显式 STOP 动作

给 stop 定义一个分数，例如：

$$
s(\text{stop} \mid x_t) = \alpha - \log(1 + u_{\text{tot}}^{\text{real}}(x_t, t))
$$

或者更简单地：

$$
s(\text{stop} \mid x_t) = -\beta \cdot u_{\text{tot}}^{\text{real}}(x_t, t)
$$

然后把 stop 当成一个普通候选加入 beam 扩展。

### 9.3 推荐策略

第一版建议用规则停止，不要过早引入可学习 stop：

1. `max_edits` 作为硬上限
2. `u_tot_real < threshold` 作为提前结束

等 beam 框架稳定后，再考虑把 stop 显式并入搜索。

---

## 10. 防止循环与抖动：单编辑搜索的必要约束

单编辑 beam search 很容易出现局部往返：

- `A -> B -> A`
- 插入后马上删除
- 某位置连续替换成不同 token

因此建议第一版就加入以下保护。

### 10.1 最近一步逆操作禁止

若上一步是：

- `sub(i, a -> b)`，下一步禁止 `sub(i, b -> a)`
- `ins(i, a)`，下一步禁止对应位置的立刻删除
- `del(i, a)`，下一步禁止立刻把同 token 插回原位

### 10.2 重复状态去重

beam 合并时，对生成出的序列做哈希去重：

- 若多个路径到达同一 `x`
- 只保留分数最高的那条

这是非常关键的，因为编辑搜索的状态空间不是树，而是图。

### 10.3 长度约束

可显式限制：

- 最短长度
- 最长长度（已有 `max_seq_len`）
- 相对输入产物的最大长度偏移

这样能减少明显不合理的扩展。

---

## 11. 候选空间裁剪：必须做，不然会太慢

### 11.1 原始复杂度

假设当前长度为 `L`，词表大小为 `V`，则所有候选规模约为：

$$
L \cdot V \text{ (insert)} + L \cdot V \text{ (sub)} + L \text{ (del)}
$$

即：

$$
O(LV)
$$

这在 beam 每一步都完整枚举时会非常贵。

### 11.2 推荐裁剪策略

第一版建议做两级截断：

1. **位置内 token top-k**
   - insert 只保留每个位置 top-`k_ins_token`
   - sub 只保留每个位置 top-`k_sub_token`

2. **全局 edit top-k**
   - 对所有候选 edit 再取 top-`k_edit_expand`

推荐初值：

- `k_ins_token = 4`
- `k_sub_token = 4`
- `k_edit_expand = 16`

### 11.3 特殊 token 过滤

建议在候选层直接过滤不该生成的 token：

- `PAD`
- `BOS`
- `GAP`

必要时也可过滤：

- `UNK`

特别是 insert/sub 的 token top-k 阶段，就应先做 mask。

---

## 12. 时间推进策略：不要直接把步数当时间

`single-edit beam` 中最容易写的做法是：

$$
t_k = \frac{k}{K}
$$

其中 `K` 是最大编辑步数。

但这个定义存在一个重要问题：它默认“编辑步数进度”和“训练时的扩散进度”近似线性对应。这个假设在当前任务里并不稳，因为：

1. 实际逆合成样本的编辑距离通常很小
2. 前几步 edit 往往已经完成了大部分语义变化
3. 后续很多步只是微调或冗余搜索

因此，若仍然令 `t_k = k/K`，就可能出现：

- 状态实际上已经接近终态，但 `t_k` 还很小
- 网络持续看到“早期时间”，形成新的 train-test mismatch

即便模型对 `t` 不特别敏感，**过于 OOD 的时间输入仍可能影响 edit 排序**。因此更合理的做法不是“根据步数推时间”，而是：

1. 先根据当前状态估计“还剩多少编辑压力”
2. 将其映射为更有语义的 `kappa_k`
3. 再通过训练 scheduler 的 `inverse()` 得到 `t_k`

也就是说，beam state 中建议维护的是 **`kappa_state`**，而不是直接维护 `t_state`。

### 12.1 为什么优先设计 `kappa` 而不是 `t`

在当前项目中，真正有语义的是：

$$
\kappa(t)
$$

它控制了 `z_t` 处于 source/target 混合路径上的“进度”。而 `t` 只是 scheduler 下的参数化。

因此更干净的设计是：

$$
\kappa_k \rightarrow t_k = \kappa_{\text{train}}^{-1}(\kappa_k)
$$

例如：

- cubic scheduler: `t_k = kappa_k^(1/3)`
- linear scheduler: `t_k = kappa_k`

这样做的好处是：

1. `kappa` 具有明确的“接近 target”语义
2. 不会直接拍脑袋向网络输入 OOD 的 `t`
3. 和当前 [scheduler.inverse()](/data1/duanbh/desktop/edit-flows/edit_flows/core/scheduler.py) 实现天然兼容

### 12.2 风险：闭环时间更新

如果 `kappa_k` 由模型当前输出的速率决定，就形成了一个闭环：

- 当前模型输出 `u`
- `u` 决定下一步 `kappa`
- 下一步 `kappa` 又影响模型输出

这不是不能做，但第一版必须加保护：

1. `kappa_k` 单调不减
2. 每步最大增量受限
3. 必要时做平滑

例如：

$$
\kappa_k \leftarrow \max(\kappa_{k-1}, \kappa_k^{\text{raw}})
$$

$$
\kappa_k \leftarrow \min(\kappa_k, \kappa_{k-1} + \Delta_{\max})
$$

必要时也可加低通滤波：

$$
\kappa_k \leftarrow (1-\lambda)\kappa_{k-1} + \lambda \kappa_k^{\text{raw}}
$$

### 12.3 Beam 版本至少有四种可比实现

为了方便实验，beam sampler 建议把时间推进显式参数化。

### 12.3.1 `time_mode = "fixed"`

每步都用固定 `t_const`。

用途：

- 检验时间是否重要

问题：

- 最简单，但可能偏离训练分布
- 更适合作为 ablation，不适合作为主方案

### 12.3.2 `time_mode = "depth"`

若最大深度为 `K`，第 `k` 层用：

$$
t_k = \frac{k}{K}
$$

用途：

- 最自然的离散编辑版时间表

问题：

- 容易实现
- 但“编辑步数进度 ≠ 扩散进度”

判断：

- 应保留为 baseline
- 不建议作为主推进方式

### 12.3.3 `time_mode = "utot_ratio"`（推荐）

核心思想：当前状态若仍有大量编辑需求，则 `kappa` 应较小；若总编辑压力已经显著下降，则 `kappa` 应靠近 1。

记初始状态和当前状态的真实总速率分别为：

$$
u_{\text{tot},0}^{\text{real}} = u_{\text{tot}}^{\text{real}}(x^{(0)}, t_0)
$$

$$
u_{\text{tot},k}^{\text{real}} = u_{\text{tot}}^{\text{real}}(x^{(k)}, t_k)
$$

定义相对剩余编辑压力：

$$
r_k = \frac{u_{\text{tot},k}^{\text{real}}}{u_{\text{tot},0}^{\text{real}} + \epsilon}
$$

然后令：

$$
\kappa_k = 1 - \min(1, r_k^\alpha)
$$

或更一般地：

$$
\kappa_k = 1 - \min(1, c \cdot r_k^\alpha)
$$

其中：

- `alpha > 0` 控制曲线形状
- `c` 控制整体缩放

直觉：

- 早期真实总速率接近初始值，`r_k \approx 1`，故 `kappa_k` 小
- 后期真实总速率显著下降，`r_k \ll 1`，故 `kappa_k` 接近 1

优点：

1. 直接利用模型当前状态判断“还剩多少编辑工作”
2. 自动适应不同样本难度
3. 比 `k/K` 更贴近状态而非步数

缺点：

1. `u_tot_real` 可能不单调
2. 若模型速率校准较差，会把 `kappa` 带偏

因此实现时建议：

- `kappa_k = max(kappa_{k-1}, kappa_k_raw)`
- 限制 `kappa` 每步最大增量

这是目前最推荐优先尝试的时间推进方式。

### 12.3.4 `time_mode = "chosen_rate"`

除了看当前总速率，也可以看**被选中 edit 的强度或条件概率**，用来衡量“本步推进有多扎实”。

设第 `k` 步被选中的 edit 为 `e_k`，其条件概率为：

$$
p_k = \frac{u^{\text{real}}(e_k \mid x^{(k)}, t_k)}{u_{\text{tot},k}^{\text{real}}}
$$

可定义：

$$
\Delta \kappa_k = \eta \cdot p_k^\alpha
$$

$$
\kappa_{k+1} = \min(1, \kappa_k + \Delta \kappa_k)
$$

也可直接用选中 edit 的原始强度：

$$
\Delta \kappa_k = \eta \cdot \left(u^{\text{real}}(e_k)\right)^\alpha
$$

但更推荐用条件概率 `p_k`，因为它已经被局部归一化，更稳。

直觉：

- 若当前最优 edit 很尖锐、模型很有把握，则 `kappa` 推进更快
- 若当前 edit 分布平、模型不确定，则 `kappa` 推进更慢

优点：

1. 利用 beam 实际选中的动作信息
2. 比单纯看 `u_tot_real` 更关注“决策置信度”

缺点：

1. 更依赖 beam/scoring 本身
2. 闭环更强，可能不稳定

判断：

- 值得作为第二阶段实验
- 不建议第一版就作为唯一方案

### 12.3.5 `time_mode = "expected_wait"`

这里必须特别注意：**expected wait 应基于真实总速率，而不是 base-rate 总速率**。

CTMC 中下一事件等待时间期望：

$$
\mathbb{E}[\tau_k] = \frac{1}{u_{\text{tot},k}^{\text{real}}}
$$

于是可令：

$$
t_{k+1} = \min\left(1,\ t_k + \frac{c}{u_{\text{tot},k}^{\text{real}} + \epsilon}\right)
$$

或先定义 `kappa` 再映射。

若 `use_rate_reparam = true`，则：

$$
u_{\text{tot},k}^{\text{real}} = k(t_k)\,u_{\text{tot},k}^{\text{base}}
$$

因此，若实现时手头拿到的是模型原始输出对应的 `u_tot_base`，则必须先乘上 `k(t_k)` 再用于 expected wait；否则会错误低估真实事件尺度。

直觉：

- 早期真实总速率大，等待时间短，时间前进小
- 后期真实总速率小，等待时间长，时间前进大

用途：

- 更贴近 CTMC 事件驱动解释

问题：

- 即便使用了真实总速率，后期仍可能推进过快
- 对 `u_tot_real` 的尺度非常敏感
- 因此仍建议额外限制 `Delta_t_max` 或 `Delta_kappa_max`

判断：

- 理论上自然
- 工程上不一定最稳，建议后做

### 12.3.6 `time_mode = "euler"`

模仿当前 [sample_euler](/data1/duanbh/desktop/edit-flows/edit_flows/sampling/euler.py)，用：

- `default_h = 1 / n_steps`
- `adapt_h = get_adaptive_h(default_h, t, scheduler)`

但每个离散步只执行一个 edit。

用途：

- 与现有 Euler 保持最大一致性

问题：

- 仍然继承了“很多时间步实际上没有必要”的问题
- 对单编辑 beam 来说不一定是最自然的离散化

### 12.4 推荐优先级与建议

建议先做：

1. `utot_ratio`
2. `depth`
3. `fixed`
4. `chosen_rate`
5. `expected_wait`
6. `euler`

原因：

- `utot_ratio` 最符合“根据当前状态还剩多少编辑压力来定义进度”的思路
- `depth` 仍然值得保留作为简单 baseline
- `chosen_rate` 与 `expected_wait` 更适合第二阶段 ablation

### 12.5 推荐默认实现

第一阶段若只选一个默认时间推进方案，建议：

$$
\kappa_k^{\text{raw}} = 1 - \min\left(1,\left(\frac{u_{\text{tot},k}^{\text{real}}}{u_{\text{tot},0}^{\text{real}}+\epsilon}\right)^\alpha\right)
$$

$$
\kappa_k = \max(\kappa_{k-1}, \kappa_k^{\text{raw}})
$$

$$
\kappa_k = \min(\kappa_k, \kappa_{k-1} + \Delta_{\max})
$$

再由训练 scheduler 得到：

$$
t_k = \kappa_{\text{train}}^{-1}(\kappa_k)
$$

推荐初值：

- `alpha = 1.0`
- `epsilon = 1e-8`
- `Delta_max = 0.2`

后续可在此基础上做 ablation。

这里公式中的 `u_tot` 默认都指 `u_tot_real`。若实现时复用的是模型原始输出，则应先通过当前采样配置恢复成真实速率后再计算。

---

## 13. 一个具体推荐版本：第一阶段主推实现

综合理论合理性、实现复杂度和当前代码结构，建议第一阶段主推如下版本。

### 13.1 搜索定义

- 每一步只允许一个 edit
- beam score 用：

$$  
s(e \mid x_t, t) = \log u^{\text{real}}(e \mid x_t, t) - \log u_{\text{tot}}^{\text{real}}(x_t, t)
$$

- 总分为累加和

### 13.2 时间定义

- 默认用 `time_mode = "utot_ratio"`
- beam state 中保存 `kappa_state`
- 每步根据当前 `u_tot_real` 更新 `kappa_state`
- 再通过训练 scheduler 的 `inverse()` 得到 `t_k`

### 13.3 停止规则

- `max_edits` 硬上限
- `u_tot_real < threshold` 提前停止

### 13.4 约束

- 禁止一步逆操作
- 重复状态去重
- token top-k 裁剪

### 13.5 为什么先做这个

因为它同时满足：

1. 比随机 Euler 更接近“最优路径搜索”
2. 比严格 CTMC path probability 简单很多
3. 避免直接用 `t_k = k/K` 带来的时间 OOD 风险
4. 和现有模型/采样代码兼容性最好

---

## 14. 建议的实现拆分

### 14.1 新文件

建议新增：

- `edit_flows/sampling/beam.py`

### 14.2 建议函数

#### `collect_edit_candidates`

输入：

- `x_t`
- `origin_mask`
- `t`
- `scheduler`
- `train_scheduler`
- `time_input`
- `use_rate_reparam`

输出：

- 候选 edit 列表
- 每个候选的 `log_u_real`
- `log_u_tot_real`

职责：

- 复用当前 `sample_euler` 的前半段时间/速率逻辑
- 从模型输出构造单编辑候选

#### `apply_single_edit`

输入：

- `x_t`
- `origin_mask`
- 一个 edit 结构体

输出：

- `x_next`
- `origin_mask_next`

职责：

- 调用 [apply_ins_del_operations](/data1/duanbh/desktop/edit-flows/edit_flows/sampling/ops.py)
- 单编辑更新 token 和 origin mask

#### `sample_greedy_single_edit`

用途：

- 作为 beam 前的最小可行版本
- 先验证单编辑贪心是否有效

#### `sample_beam_single_edit`

用途：

- 真正的 beam search 主入口

### 14.3 脚本侧改动

建议在 [scripts/sample_retro.py](/data1/duanbh/desktop/edit-flows/scripts/sample_retro.py) 增加：

- `--sampler euler|greedy_edit|beam_edit`
- `--beam_size`
- `--max_edits`
- `--time_mode`
- `--time_alpha`
- `--time_delta_max`
- `--edit_score_mode`

其中：

- `edit_score_mode = log_u | cond_prob | ctmc_dt`
- `log_u` 表示使用 `log u_real(edit)` 评分
- `cond_prob` 表示使用 `log u_real(edit) - log u_tot_real`
- `ctmc_dt` 表示使用 `log u_real(edit) - u_tot_real * dt`
- `time_mode = depth | fixed | utot_ratio | chosen_rate | expected_wait | euler`

分别对应：

- 方案 A
- 方案 B
- 方案 C

---

## 15. 建议的实验顺序

### Phase 1：先验证“单编辑搜索”本身是否成立

1. `greedy_edit` + `log_u_real`
2. `greedy_edit` + `log_u_real - log_u_tot_real`
3. `beam_edit` + `log_u_real - log_u_tot_real`

看是否优于现有随机 Euler。

### Phase 2：验证时间是否真的不重要

固定其他设置，只比较：

1. `time_mode=fixed, t=0.5`
2. `time_mode=depth`
3. `time_mode=utot_ratio`
4. `time_mode=euler`

### Phase 3：验证更严格 CTMC 分数是否有增益

比较：

1. `cond_prob`: `log_u_real - log_u_tot_real`
2. `ctmc_dt`: `log_u_real - u_tot_real * dt`

### Phase 4：再考虑 stop 与 rerank

若 beam 已有收益，再讨论：

1. 显式 stop 动作
2. sample 多条后再 rerank
3. 加 SMILES 合法性约束

---

## 16. 风险与注意事项

### 16.1 最大风险：模型的速率不够校准

Oracle 结果说明“理论上存在好路径”，但模型当前的 `u_real` 未必足够尖锐。因此 beam search 也可能出现：

- 把错误但高频的 edit 越搜越坚定

所以 beam 不一定自动提升 Top-1，但它至少能帮助诊断：

- 问题是“随机采样太差”
- 还是“模型本身对 edit 排序就错”

### 16.2 第二个风险：搜索空间与长度爆炸

即使编辑距离小，插入操作也会迅速放大分支数，因此必须做：

- token top-k 裁剪
- beam 去重
- 最大深度约束

### 16.3 第三个风险：stop 不好定义

若 stop 太激进，会过早结束；若 stop 太保守，会继续乱改。第一版最好先用简单阈值，不要过早引入复杂设计。

---

## 17. 结论

基于当前代码实现，Edit Flows 完全可以定义一种“编辑事件级别的 beam search”。

最可行的切入点不是直接追求严格的连续时间最优路径，而是先做以下近似：

1. **每一步只允许一个 edit**
2. **把 `u_real(e)/u_tot_real` 视为下一编辑事件的条件概率**
3. **对 edit 序列做 beam search**

在目前阶段，最推荐优先实现的是：

- `single-edit`
- `beam score = log u_real - log u_tot_real`
- `time_mode = utot_ratio`
- `max_edits` 很小

这会是最接近“Edit Flows 版 beam search”的第一版工程落地方案。

---

## 18. TODO 清单

1. 新增 `edit_flows/sampling/beam.py`
2. 抽取/复用 `sample_euler` 中的时间映射与 rate parameterization 逻辑
3. 实现 `collect_edit_candidates`
4. 实现 `apply_single_edit`
5. 实现 `sample_greedy_single_edit`
6. 实现 `sample_beam_single_edit`
7. 在 `scripts/sample_retro.py` 增加 sampler/beam 相关 CLI
8. 增加基础测试：
   - 单编辑 insert/sub/del/replace 的状态转移
   - origin mask 在 beam sampler 下的演化
   - 候选分数 `log_u_real` / `log_u_real-log_u_tot_real` 计算正确
   - `utot_ratio` / `chosen_rate` 的 `kappa` 更新正确
   - `kappa` 单调性与 `Delta_max` 限制生效
   - 重复状态去重
9. 先在 `train_subsets` 上比较 greedy vs beam vs 随机 Euler
10. 再做时间模式与 stop 规则 ablation

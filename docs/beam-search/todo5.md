# Beam Search 路线图 v5：显式 STOP 动作与 First-Event 概率化搜索

## 1. 背景与目标

在 `docs/beam-search/exp4.md` 中，`RatioTimePolicy` 已将 greedy single-edit 的 Top-1 从 35.5% 提升到 44.5%，说明：

1. **时间 mismatch 的确是重要瓶颈**
2. **单编辑搜索本身不是死路**
3. 当前采样的主要问题，已从“时间给错”进一步收缩为：**停止信号仍然不够 principled**

目前 greedy/beam 的停止方式有两类：

- 外部阈值：`u_tot_base < stop_u_tot_base`
- policy 内部 stop：如 `KappaTimePolicy` 的 `u_tot_base < 1`

这两种 stop 都有共同问题：

- 它们是**外部规则**，不在当前候选编辑的同一概率空间内
- 它们依赖 `u_tot_base` 的**绝对量级**，容易出现样本间尺度不一致
- 它们无法回答一个更本质的问题：**当前应该继续执行某个 edit，还是应该显式选择 STOP？**

因此下一阶段的目标是：

> 将 STOP 视为与 insert / substitute / delete 并列的**显式动作**，把单编辑搜索从“局部速率贪心 + 外部阈值 early stop”改为“基于 first-event 概率的显式动作搜索”。

理想结果是：

- 在每个状态 `(kappa, seq)` 上，同时得到
  - `p_stop`
  - 各候选编辑的 `p_e`
- beam / greedy 都统一用

$$
\text{score}(\pi)=\sum_k \log p(e_k \mid \text{history}) + \log p_{\text{stop}}
$$

来排序，而不再依赖外部阈值 stop。

---

## 2. 当前状态与核心转向

### 2.1 当前 greedy/beam 的语义

当前 `edit_flows/sampling/beam.py` 的行为是：

1. 在当前 `(x_t, t)` 上前向一次，得到 base rate `u_e`
2. 用当前步的条件分数

$$
\log u_e - \log u_{\text{tot}}
$$

排序候选 edit
3. 再用 `stop_u_tot_base` 或 policy 内部规则决定是否停止

这等价于：

- edit 的排序依据来自模型
- stop 的依据来自外部 heuristic

二者不统一。

### 2.2 下一阶段的核心转向

下一阶段不再把“停止”视为 beam 外部逻辑，而是改为：

- `STOP` 是一个候选 child
- `INSERT / SUB / DEL / STOP` 共同竞争
- beam score 统一累计这些动作的 log-prob

因此新问题变成两个：

1. 如何从 `(kappa, seq, u_e)` 定义 `p_stop` 与 `p_e`
2. 当选择某个 edit `e` 后，如何定义下一时刻 `kappa'`

本文就是对这两个问题给出可实现方案，并制定实验顺序。

---

## 3. 基本建模假设

为避免与训练中的 real rate / base rate / scheduler scaling 混淆，这里统一约定：

- 当前状态记为 `(kappa, seq)`
- 模型输出的 `u_e` 指 **base rate 形式** 的 edit 强度
- 也即：**未乘 `k(t)`**，直接表示在 `kappa ~ 1` 剩余区间内的编辑需求

这一定义与 `RatioTimePolicy` 的经验成功是相容的：它本质上也是把 `u_tot_base` 当作“剩余编辑量”的 proxy。

在这个语义下，定义：

$$
U = \sum_{e \in \mathcal{E}(seq)} u_e
$$

其中 `e` 遍历当前所有合法 edit 候选。

工作假设：

- `U` 近似表示从当前 `(kappa, seq)` 到终点所需的**剩余编辑总量**
- `u_e / U` 近似表示在“下一次有意义编辑”发生时，该编辑类型/位置/token 被选中的相对偏好

这不是严格 CTMC 推导，但与当前模型最贴近，也是最容易落地的工作解释。

---

## 4. 显式 STOP 的三种候选算法

## 4.1 方案 A：Poisson Remaining Mass（最小可行版本，推荐优先实现）

### 4.1.1 定义 `p_stop` 与 `p_e`

将剩余编辑数近似为一个 Poisson 随机变量：

$$
N \sim \text{Poisson}(U)
$$

则：

$$
p_{\text{stop}} = P(N=0)=e^{-U}
$$

表示“从现在到终点无需任何编辑”的概率。

若至少还会发生一次编辑，则条件在“有编辑发生”下，第一个编辑是谁，用当前的相对 edit mass 分配：

$$
p_e = (1-e^{-U}) \cdot \frac{u_e}{U}
$$

于是有：

$$
p_{\text{stop}} + \sum_e p_e = 1
$$

这给出了一个**完整归一化**的动作分布。

### 4.1.2 定义下一时间 `kappa'`

直觉上，若当前总剩余编辑量为 `U`，执行一次编辑后，剩余量应约减少 1：

$$
U' \approx U - 1
$$

于是定义剩余区间 `(1-kappa)` 的收缩：

$$
1-kappa' = (1-kappa)\cdot \max\left(0,\frac{U-1}{U}\right)
$$

等价写为：

$$
kappa' = 1 - (1-kappa)\cdot \max\left(0,\frac{U-1}{U}\right)
$$

当 `U` 很大时，`kappa'` 仅前进一点；当 `U` 接近 1 时，`kappa'` 快速靠近 1。

### 4.1.3 优缺点

优点：

- 单次 forward 即可得到 `STOP + edits` 的完整分布
- 与当前 `u_tot_base` 经验用法最一致
- 最容易接入现有 greedy/beam 框架

缺点：

- `U` 被直接当成剩余编辑计数，解释仍偏 heuristic
- `kappa'` 只依赖 `U`，与具体选了哪个 `e` 无关

### 4.1.4 适用定位

这是**第一优先级原型**。如果它已经能明显优于 `ratio + threshold`，说明“显式 STOP + 概率化路径分数”是值得继续深挖的。

---

## 4.2 方案 B：Frozen-Hazard First-Event（推荐作为 v1 主实现）

方案 A 的概率公式，其实可以视为一个更有连续时间意味的 frozen-hazard 近似。

### 4.2.1 基本想法

设当前剩余区间长度为：

$$
H = 1-kappa
$$

假设从当前到终点，在“第一个事件发生前”，每个 edit 的 hazard 保持常数：

$$
\lambda_e = \frac{u_e}{H}, \qquad \lambda = \sum_e \lambda_e = \frac{U}{H}
$$

则“直到终点都没有事件发生”的概率为：

$$
p_{\text{stop}} = e^{-\lambda H}=e^{-U}
$$

而第一个事件是 `e` 的概率为：

$$
p_e = (1-e^{-U})\cdot\frac{u_e}{U}
$$

因此，方案 B 与方案 A 在 `p_stop / p_e` 上完全一致，但 `kappa'` 可以得到更严格的定义。

### 4.2.2 定义下一时间 `kappa'`

设事件等待时间 `T ~ Exp(\lambda)`。在条件 `T < H` 下，第一个事件发生时间的条件期望为：

$$
\mathbb{E}[T \mid T < H]
= \frac{1}{\lambda} - \frac{H e^{-\lambda H}}{1-e^{-\lambda H}}
$$

代回 `U = \lambda H`，得到：

$$
kappa' =
kappa + (1-kappa)\left(\frac{1}{U} - \frac{e^{-U}}{1-e^{-U}}\right)
$$

该公式具有更好的数值语义：

- `U` 大时，近似前进 `(1-kappa)/U`
- `U` 小时，不会像简单 `1/U` 那样过于激进

### 4.2.3 优缺点

优点：

- 仍只需一次 forward
- `p_stop / p_e` 形式简单
- `kappa'` 有明确 first-event waiting-time 解释

缺点：

- 仍然把未来 hazard 冻结为当前值
- `kappa'` 仍不依赖具体 edit `e`

### 4.2.4 适用定位

这是我目前最推荐的 **v1 主实现方案**：

- 概率定义足够 clean
- 与当前实现兼容性高
- 工程复杂度可控

---

## 4.3 方案 C：Grid-Integrated First-Event（更 principled，作为 v2）

### 4.3.1 核心思想

在第一个事件发生前，序列 `seq` 保持不变，只有 `kappa` 从当前值走向 1。  
因此，可以沿未来若干 `kappa` 网格点，对**同一个 `seq`** 重复前向，构造 first-event 分布。

设网格：

$$
kappa = \sigma_0 < \sigma_1 < \cdots < \sigma_m = 1
$$

在每个 `\sigma_j` 上，对固定 `seq` 前向，得到：

$$
u_e(\sigma_j, seq), \qquad U(\sigma_j)=\sum_e u_e(\sigma_j, seq)
$$

### 4.3.2 survival 与 first-event 概率

定义某种 hazard 近似 `h_e(\sigma)`，最自然的选择是：

$$
h_e(\sigma)=\frac{u_e(\sigma)}{1-\sigma}, \qquad
h_{\text{tot}}(\sigma)=\frac{U(\sigma)}{1-\sigma}
$$

则生存函数为：

$$
S(\sigma)=\exp\left(-\int_{kappa}^{\sigma} h_{\text{tot}}(r)\,dr\right)
$$

于是：

$$
p_{\text{stop}} = S(1)
$$

$$
p_e = \int_{kappa}^{1} S(\sigma)\,h_e(\sigma)\,d\sigma
$$

### 4.3.3 定义下一时间 `kappa'_e`

对选中的 edit `e`，下一时间可定义为该 first-event time 的条件均值：

$$
kappa'_e =
\frac{\int_{kappa}^{1}\sigma\,S(\sigma)\,h_e(\sigma)\,d\sigma}{p_e}
$$

也可选用 MAP：

$$
kappa'_e = \arg\max_{\sigma} S(\sigma) h_e(\sigma)
$$

这时 `kappa'` 将**依赖具体 edit `e`**，语义比方案 A/B 更完整。

### 4.3.4 优缺点

优点：

- 最接近“从当前状态到终点，第一个事件是谁”的原始想法
- `STOP` 和各个 edit 完全在同一积分框架里
- `kappa'` 可依赖具体 action

缺点：

- 每个 state 需要多次 model forward
- beam 场景下计算代价明显更高

### 4.3.5 适用定位

作为 **v2 / 验证性增强方案**。  
不建议一上来就全量实现，但很适合在 v1 成功后验证：

- 当前 `U` 的单点近似是否已经足够
- 未来 `kappa` 曲线形状是否真的重要

---

## 5. 推荐路线：先 B，后 C

结合目前项目状态，建议顺序如下：

### 第一阶段：实现方案 B

原因：

1. `STOP` 能显式进入动作空间
2. `beam score` 能从 heuristic 累积分数改成真正概率累积
3. 只需一次 forward，能快速验证方向
4. 和当前 `RatioTimePolicy` 的经验基础最接近

### 第二阶段：如果 B 有收益，再实现方案 C 的小网格版

建议仅对：

- 当前 top-M edits
- `STOP`

做积分近似，而不是对所有 edit 全量积分。

这样可以将成本控制在可接受范围内。

---

## 6. 与现有代码的对接方案

## 6.1 核心设计转变

当前实现中，`TimePolicy` 同时承担两件事：

1. 给当前步提供 `kappa`
2. 提供内部 stop 信号

引入显式 `STOP` 后，这两件事应拆开：

- `TimePolicy`：只负责给出当前 `kappa`
- `StopPolicy`：取消，或仅保留为 fallback/debug
- 新增 `ActionModel` / `FirstEventModel` 风格逻辑：从 `(kappa, seq, u_e)` 计算
  - `p_stop`
  - 各候选 `p_e`
  - 非 stop child 的 `kappa'`

### 6.2 推荐的最小改动原则

不建议一开始大规模重构接口。  
第一版保持：

- `TimePolicy` 仍存在，但只用于**初始时间 / 外层时间先验**
- 真正的后续 `kappa'` 由显式 first-event 规则决定

也即：

- greedy/beam 进入 state 后，先从 state 中读 `kappa`
- forward 一次，得到 `u_e`
- 用方案 B 计算 `p_stop / p_e`
- 选中某个 edit 后，直接写回 child state 的 `kappa'`

此时 `TimePolicy` 可以先退化为：

- `InitialKappaPolicy`
- 或仅在 step 0 提供初始 `kappa`

换言之，**state 上需要显式携带当前 `kappa`**，而不是每步再从 `step -> policy -> kappa` 映射。

### 6.3 建议的数据结构调整

当前 `BeamState`：

```python
@dataclass
class BeamState:
    x_t: Tensor
    origin_mask: Optional[Tensor]
    score: float
    last_edit: Optional[EditCandidate] = None
    is_finished: bool = False
    time_policy: Optional[TimePolicy] = None
```

建议逐步演化为：

```python
@dataclass
class BeamState:
    x_t: Tensor
    origin_mask: Optional[Tensor]
    kappa: float
    score: float                 # sum log p(action)
    last_edit: Optional[EditCandidate] = None
    is_finished: bool = False
```

其中：

- `kappa` 成为 hypothesis 级显式状态
- `time_policy` 可在过渡期保留，但最终应退出核心循环

### 6.4 候选动作结构

当前只有 `EditCandidate`。  
建议新增统一动作结构：

```python
@dataclass
class ActionCandidate:
    kind: str                   # "stop" | "ins" | "sub" | "del"
    edit: Optional[EditCandidate]
    prob: float
    log_prob: float
    next_kappa: Optional[float] # stop 时为 None 或 1.0
```

这样 greedy / beam 的展开逻辑就能统一处理 `STOP` 与普通 edit。

---

## 7. 方案 B 的具体实现草案

## 7.1 当前步 forward

对每个 active state：

1. 取其当前 `kappa`
2. `t = scheduler.inverse(kappa)`
3. `t_model = _compute_model_time(...)`
4. model forward，得到
   - `log_rates`
   - `log_ins_probs`
   - `log_sub_probs`
5. 组合出候选 edit 对应的 base mass `u_e`

这里推荐继续沿用现有候选裁剪：

- 每位置 token top-k
- 全局 top `k_edit_expand`

因为显式 `STOP` 并不要求全量枚举所有 edit。

## 7.2 由候选 edit 计算 `U`

这是一个关键实现选择：

### 选项 1：用“可执行候选集”的总质量

即只对当前 legal edit 集合求和。

优点：

- 与当前 beam 实际可选动作完全一致
- `STOP + candidate_edits` 自动归一

缺点：

- `k_edit_expand` 裁剪会让 `U` 受候选池大小影响

### 选项 2：用“全局可执行总量”

也即像当前 `_compute_executable_u_tot(...)` 那样，对所有合法 insert/sub/del 的总量求和。

优点：

- `U` 更稳定，更接近“真实剩余编辑总量”

缺点：

- 进入 beam 的候选只是其中一小部分时，`sum p_e < 1-p_stop`

### 推荐

第一版建议：

- 用 **全局可执行总量** 作为 `U`
- 在进入 beam 的 top-K edits 上做截断
- 并记录一个额外的

$$
p_{\text{other}} = (1-p_{\text{stop}}) - \sum_{e \in \text{topK}} p_e
$$

诊断候选裁剪损失。

若 `p_other` 长期很大，说明当前 `k_edit_expand` 太小，不适合此框架。

## 7.3 动作概率

对当前 state：

$$
U = \sum_e u_e
$$

$$
p_{\text{stop}} = e^{-U}
$$

$$
p_e = (1-e^{-U}) \cdot \frac{u_e}{U}
$$

对 top-K 候选 edit 构造 child：

- `STOP` child：`log_prob = log(p_stop)`
- edit child：`log_prob = log(p_e)`

## 7.4 下一时间

对所有 edit child，共用：

$$
kappa' =
kappa + (1-kappa)\left(\frac{1}{U} - \frac{e^{-U}}{1-e^{-U}}\right)
$$

再做：

- `clip(kappa', kappa + eps, 1 - eps)`

防止数值边界问题。

`STOP` child 则直接标记 finished，不需要后续 `kappa'`。

## 7.5 Beam score

统一改为：

$$
\text{score}_{\text{child}} =
\text{score}_{\text{parent}} + \log p(\text{action})
$$

最终路径分数自然包含末尾的 `STOP`。

这比当前“局部 edit score + 外部 stop”更完整。

---

## 8. 方案 C 的小网格增强版

若方案 B 有效果，再继续：

### 8.1 网格设计

对每个 active state 的当前 `kappa`，取未来 `m=4~8` 个网格点：

$$
\sigma_j = kappa + \frac{j}{m}(1-kappa)
$$

或者用偏向前期更密的非线性网格。

### 8.2 计算内容

对固定 `seq` 在每个 `\sigma_j` 前向一次，得到：

- `U(\sigma_j)`
- top-K edit 的 `u_e(\sigma_j)`

用离散积分近似 `p_stop` 与 `p_e`。

### 8.3 只对 top-K edit 做积分

不做全量 `L*V` 级别积分。  
当前步先用单点 forward 拿到 top-K，之后只对这 K 个 edit 跟踪其时间曲线。

这样计算量更接近：

- `1 + m` 次 forward / state

虽然仍贵，但可作为小规模验证实验。

---

## 9. 预实验与决策门槛

## 9.1 P1：诊断 `p_other`

目的：判断 top-K 候选裁剪是否会破坏概率化 beam。

做法：

- 在现有 greedy/beam 轨迹上记录
  - `U_global`
  - `sum_topK_u`
  - `p_other`

判据：

- 若 `p_other < 0.1` 且大多数步骤稳定较小 → 当前候选裁剪足够
- 若 `p_other` 经常 > 0.3 → 需增大 `k_edit_expand` 或改候选策略

## 9.2 P2：显式 STOP 的 greedy 对照

对比：

- `ratio + stop_u_tot_base`
- `explicit_stop_v1`（方案 B）

看：

- Top-1
- Invalid
- 平均编辑步数
- stop 时机分布

若 `explicit_stop_v1` 至少持平且 stop 更稳定，说明方向成立。

## 9.3 P3：beam 收益是否回升

若 greedy 版本成立，再比较：

- greedy + explicit stop
- beam-3 + explicit stop
- beam-5 + explicit stop

核心问题：

> 当 stop 被纳入显式动作概率后，beam 是否终于能比 greedy 带来更明显收益？

若 beam 收益从此前的 ~2pp 扩大到 4-8pp，将强烈支持这一路线。

## 9.4 P4：方案 B vs 方案 C

仅在 100~200 条样本小规模比较：

- 单点 frozen-hazard
- 小网格积分版

若 C 只带来极小增益，不值得承担高算力成本。

---

## 10. 详细执行顺序

### Step 1：做诊断，不改主逻辑

在现有 `beam.py` / 诊断脚本中新增日志：

- `U_global`
- `sum_topK_u`
- `p_stop = exp(-U)`
- `p_other`
- 当前 threshold stop 与显式 stop 倾向是否一致

目的：先确认这个概率解释在量级上是否合理。

### Step 2：实现显式 STOP 的 greedy v1（方案 B）

改动原则：

- 不先改 beam
- 不先废弃 `TimePolicy`
- 先做 greedy 原型，最容易排错

实现后：

- `STOP` 成为候选 child
- 禁用 `stop_u_tot_base` 主判定，仅保留 debug fallback
- `score` 改为 `sum log p(action)`

### Step 3：小规模 greedy 实验

在 200 条 test 子集上与 `ratio_stop0.05/0.1/0.5` 对照。

重点观察：

- 是否减少 Type A 过度编辑
- 是否减少 stop 阈值敏感性
- 是否把平均步数拉到更合理范围

### Step 4：将显式 STOP 接入 beam

此时再改 beam 扩展逻辑：

- parent 直接可展开出一个 `STOP` child
- 与 edit child 一起进入候选池
- `finished` 不再主要依赖外部 stop，而由 `STOP` child 自然产生

### Step 5：若有必要，再推进方案 C

仅在方案 B 已显著优于 baseline 的前提下继续。

---

## 11. 风险点与注意事项

### 11.1 `u_e` 的语义并非严格 calibrated probability mass

整个方案的根基是：

- 把 base rate `u_e` 解释为剩余编辑需求

这是工作假设，不是论文中已证明的结论。  
因此必须依靠实验验证，而不能先验认定其概率严格正确。

### 11.2 `k_edit_expand` 会影响概率质量截断

当前 beam 为了效率只保留 top-K 候选。  
在显式 STOP 框架下，这不再只是“搜索效率问题”，而会直接影响分布质量。

因此 `p_other` 诊断是必须的。

### 11.3 下一时间 `kappa'` 与具体 edit 无关

方案 B 中所有 edit child 共用同一个 `kappa'`。  
这意味着：

- 位置/token 的差异只影响 `seq'`
- 不影响事件时间

这在理论上不够细，但作为第一版是可接受的。

### 11.4 显式 STOP 不能完全替代所有 safeguard

即使显式 STOP 生效，以下规则仍应保留：

- BOS 保护
- no-op sub 过滤
- reverse-op 过滤
- dead-end 保活

这些是 correctness 约束，不应因为 stop 改造而删除。

---

## 12. 当前建议结论

基于目前项目进展，下一步最值得做的不是继续调 `stop_u_tot_base`，也不是再发明新的 `TimePolicy`，而是：

> 将 `STOP` 正式纳入动作空间，并以 first-event probability 的近似形式，统一 edit 与 stop 的排序和路径分数。

具体推荐：

1. **先做方案 B（Frozen-Hazard First-Event）**
2. **先从 greedy 开始**
3. **先保留 `TimePolicy` 作为初始/过渡时间机制**
4. **用 `p_other` 诊断候选裁剪是否破坏分布**
5. **若 greedy 有明显收益，再接 beam**
6. **若还需要更细致时间语义，再上方案 C**

这条路线的最大价值在于：它第一次把“是否停止”从外部 heuristic 拉回到了与 edit 同一套概率化决策框架中。

---

## 13. 预期文件改动（第一阶段）

| 文件 | 改动 |
|------|------|
| `edit_flows/sampling/beam.py` | 新增显式 `STOP` child 逻辑；greedy 路径分数改为 `sum log p(action)`；beam 后续再接入 |
| `edit_flows/sampling/time_policy.py` | 过渡期保留；后续可能弱化为初始 `kappa` 提供器 |
| `scripts/sample_retro.py` | 新增开关，如 `--explicit_stop` / `--event_model frozen_hazard` |
| `tests/sampling/test_beam.py` | 新增 `STOP` child、分数归一、`kappa'` 更新、beam carry-over 相关测试 |
| `experiments/*` | 新增显式 STOP 对照实验脚本 |

---

## 14. 下一步

建议按以下顺序开始：

1. 在现有 greedy/ratio 轨迹上加日志，验证 `U / p_stop / p_other` 的量级
2. 先实现 `explicit_stop_greedy_v1`（方案 B）
3. 在 200 条 test 子集上与 `ratio + threshold` 做 head-to-head 对照
4. 若成立，再把同一动作模型接入 beam

这将是 beam-search 阶段从“时间修正”进入“动作概率化”的下一次关键转向。

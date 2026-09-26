# FastFlow 对 EFRetro 推理加速的启发与实现方案

## 1. 核心结论

FastFlow 对我们**确实有值得借鉴的推理加速思路**，但不适合直接照搬其连续空间中的 velocity 外推。

更适合 EFRetro 的思路是：

> **在离散编辑流中，对编辑强度场进行自适应复用或外推，从而减少 Transformer 的实际 forward 次数。**

FastFlow 的关键并不是简单地把 50 步减成 25 步，而是：

1. 识别模型输出变化较小、轨迹较平滑的区间；
2. 用前面已经计算出的模型输出近似后续若干步；
3. 跳过这些中间步骤中的昂贵神经网络计算；
4. 在变化较大的区域恢复精确模型计算；
5. 用 bandit 动态决定一次可以安全跳过多少步。

FastFlow 在论文中采用历史 velocity 的有限差分进行外推，并使用多臂老虎机（MAB）在“速度”和“误差”之间做自适应权衡。

对我们的离散编辑流而言，最自然的对应不是直接外推离散状态 \(x_t\)，而是外推决定编辑过程的：

\[
u=\lambda Q
\]

即**编辑强度场（edit intensity field）**。

---

# 2. 当前 EFRetro 推理流程

当前正式推理配置为：

- 100 个采样时间步；
- \(R=9\)：每个增强输入进行 9 条独立完整轨迹采样；
- \(M=2\)：每一步从当前状态采样 2 个子候选；
- \(K=1\)：每一步最终只保留 1 个子状态；
- batch size = 32；
- cubic scheduler：\(\kappa(t)=t^3\)。

对于一个 batch：

\[
32\times 9=288
\]

条轨迹会同时推进。

当前 `sample_branches()` 每一步大致执行：

```text
x_t
 ↓
Transformer(x_t, t)
 ↓
log_rates, log_ins_probs, log_sub_probs
 ↓
复制给 M=2 个 child
 ↓
分别采样编辑操作
 ↓
得到两个 x_{t+h}
 ↓
K=1：选择一个 child
 ↓
进入下一步
```

### 一个重要结论

当前的 \(M=2\) **并不会导致 Transformer forward 两次**。

两个 child 共用同一次模型输出，只在模型输出之后进行两次随机采样。因此当前最主要的计算瓶颈不是 \(M\)，而是：

\[
\boxed{100\text{ 次串行 Transformer forward}}
\]

所以最有价值的加速方向应该是：

> **减少每条轨迹真正调用 Transformer 的次数，即降低 NFE（Number of Function Evaluations）。**

---

# 3. 当前所谓 adaptive \(h\) 实际上接近固定 100 步

当前代码使用：

\[
h_{\text{adapt}}
=
\min\left(
\frac1{100},
\frac{1-\kappa(t)}{\kappa'(t)}
\right)
\]

而 cubic scheduler 为：

\[
\kappa(t)=t^3,\qquad
\kappa'(t)=3t^2
\]

在当前大部分采样区间内：

\[
h_{\text{adapt}}=0.01
\]

因此虽然代码函数名为 `get_adaptive_h()`，当前实际推理基本仍然是：

\[
t=0,\ 0.01,\ 0.02,\ldots,0.99
\]

即固定 100 步。

这给我们留下了一个非常自然的优化空间：

> **真正实现自适应时间步长。**

---

# 4. 方案一：Hazard-aware Adaptive Step Size

这是最建议优先验证的方法。

当前每个位置的编辑事件概率为：

\[
P(\text{event})
=
1-\exp(-h\lambda)
\]

因此从数学形式上看，采样器天然支持不同的时间步长 \(h\)。

当前固定为：

\[
h=0.01
\]

我们可以改成动态选择：

\[
h\in\{0.01,0.02,0.04,0.06\}
\]

即：

```text
稳定 / 低编辑活动区域
        ↓
h = 0.04 或 0.06

高编辑活动 / 状态复杂区域
        ↓
h = 0.01
```

---

## 4.1 用什么判断当前能不能走大步？

最自然的量是当前总编辑 hazard：

\[
\Lambda(x_t,t)
=
\sum_i
\left(
\lambda^{ins}_i+
\lambda^{sub}_i+
\lambda^{del}_i
\right)
\]

其中：

- \(\lambda^{ins}\)：插入速率；
- \(\lambda^{sub}\)：替换速率；
- \(\lambda^{del}\)：删除速率。

如果：

\[
\Lambda(x_t,t)\ll 1
\]

说明接下来一小段时间内整体发生编辑的概率较低，可以安全增大步长。

反之：

\[
\Lambda(x_t,t)\text{ 较大}
\]

则维持细粒度步长。

第一版甚至不需要 bandit，可以直接设计简单规则，例如：

\[
\Delta t
=
\begin{cases}
0.06,& h\Lambda<\tau_1\\
0.04,& \tau_1\le h\Lambda<\tau_2\\
0.02,& \tau_2\le h\Lambda<\tau_3\\
0.01,& \text{otherwise}
\end{cases}
\]

然后在 dev1000 上搜索合理阈值。

---

## 4.2 优点

这一方案：

- 不修改模型；
- 不重新训练；
- 不改变已有 \(\lambda,Q\) 建模方式；
- 与当前 Poisson event sampling 完全兼容；
- 实现成本最低；
- 可以直接观察 NFE 和 wall-clock speedup。

因此它最适合作为第一轮实验。

---

# 5. 方案二：把 FastFlow 的 velocity 外推改造成 Edit Intensity 外推

FastFlow 在连续 flow 中使用：

\[
v(x_t,t)
\]

描述当前状态的连续运动方向。

我们的状态 \(x_t\) 是离散 token sequence，包含：

- INS；
- SUB；
- DEL。

因此不能直接对 \(x_t\) 本身做 Taylor 展开。

特别是发生 INS / DEL 后：

- 序列长度会变化；
- token 位置会移动；
- 后续位置和前一步不再严格对应。

所以不应该直接照搬：

\[
x_{t+\Delta t}
\approx
x_t+\Delta t\,v_t
\]

---

## 5.1 我们真正对应的“velocity”是什么？

在我们的离散编辑流中，真正决定状态转移的是：

\[
u=\lambda Q
\]

例如插入：

\[
u^{ins}_{i,v}
=
\lambda^{ins}_iQ^{ins}_{i,v}
\]

替换：

\[
u^{sub}_{i,v}
=
\lambda^{sub}_iQ^{sub}_{i,v}
\]

删除：

\[
u^{del}_i
=
\lambda^{del}_i
\]

因此可以把 \(u\) 看成离散编辑空间中的“velocity field”。

于是可以借鉴 FastFlow，使用前两次 exact model evaluation：

\[
u_p,\quad u_k
\]

估计：

\[
\frac{du}{dt}
\approx
\frac{u_k-u_p}{t_k-t_p}
\]

进一步外推：

\[
\hat u_{k+j}
=
u_k+
(t_{k+j}-t_k)
\frac{u_k-u_p}{t_k-t_p}
\]

这样在部分中间时间点可以不调用 Transformer，而直接使用：

\[
\hat u
\]

进行采样。

---

# 6. 一个重要限制：只在状态没有发生编辑时继续外推

这是离散编辑场景和连续图像 flow 最大的区别之一。

如果：

\[
x_{k+1}=x_k
\]

那么输入状态完全没变，只是时间：

\[
t_k\rightarrow t_{k+1}
\]

发生变化。

这时：

\[
u(x,t)
\]

随时间平滑变化是一个非常合理的假设。

因此可以：

```text
Exact model
   ↓
u_k
   ↓
Approx u_{k+1}
   ↓
没有发生编辑
   ↓
Approx u_{k+2}
   ↓
仍然没有编辑
   ↓
继续 approximate
```

但只要发生：

\[
x_{k+1}\neq x_k
\]

例如进行了 INS / DEL / SUB，就马上：

```text
发生编辑
   ↓
停止复用旧的 u
   ↓
重新调用 Transformer
   ↓
得到新的 exact u
```

因此比较稳妥的规则是：

\[
\boxed{
x_t\text{ 一旦变化，就立即重新计算模型输出}
}
\]

这能够避免 token 插入、删除后的位置错位问题。

---

# 7. 方案三：No-event Jump / State-preserving Event Skipping

这一方案尤其适合我们的 Poisson sampler。

当前代码中：

\[
P_{ins}
=
1-\exp(-h\lambda_{ins})
\]

以及：

\[
P_{sub/del}
=
1-\exp[-h(\lambda_{sub}+\lambda_{del})]
\]

因此整个状态在一个小时间段内**完全不发生编辑**的概率，可以近似由总 hazard 得到：

\[
P_0
=
\exp(-h\Lambda)
\]

---

## 7.1 结合当前 \(M=2\) 的 K1M 策略

当前每一步生成 \(M=2\) 个 child。

如果：

- 两个 child 都没有发生编辑，则父状态保持；
- 一个发生编辑、一个不发生编辑，由于 `changed_state_bonus=0.5`，变化状态通常更容易被保留。

因此粗略看，最终状态保持不变需要两个 child 都不编辑：

\[
P_{\text{stay}}
\approx
P_0^M
\]

即：

\[
P_{\text{stay}}
\approx
\exp(-Mh\Lambda)
\]

对于当前：

\[
M=2
\]

就是：

\[
P_{\text{stay}}
\approx
\exp(-2h\Lambda)
\]

---

## 7.2 如果未来一段时间几乎不会发生编辑

当前做法是：

```text
Transformer
   ↓
no edit

Transformer
   ↓
no edit

Transformer
   ↓
no edit

Transformer
   ↓
no edit
```

如果我们判断未来：

\[
\Delta t=0.04
\]

都很可能没有事件，则可以直接：

```text
Transformer
   ↓
estimate integrated hazard
   ↓
判断 [t, t+0.04] 内没有事件
   ↓
t ← t+0.04
   ↓
Transformer
```

这就直接省掉了中间 3 次甚至更多 model forward。

---

## 7.3 更完整的形式

不必假设 \(\Lambda\) 在整个区间完全不变。

可以利用前两个 exact evaluation：

\[
\Lambda_p,\quad \Lambda_k
\]

做线性外推：

\[
\hat\Lambda(s)
\]

然后计算：

\[
P_{\text{no-event}}
=
\exp
\left(
-
\int_t^{t+\Delta t}
M\hat\Lambda(s)\,ds
\right)
\]

如果：

\[
P_{\text{no-event}}>\tau
\]

例如足够接近 1，就直接进行 jump。

这可以称作：

> **Adaptive Event-Skipping Edit Flow**

相比直接把 FastFlow 的连续 velocity 外推照搬过来，这一方法与我们的 Poisson 编辑机制更加自然。

---

# 8. 方案四：Bandit-guided Adaptive Skipping

Bandit 不建议作为第一步，但可以作为最终完整方法。

FastFlow 中每个 arm 对应：

\[
m=\text{skip length}
\]

例如：

\[
m\in\{0,2,4,6\}
\]

reward 为：

\[
r(m)
=
\mu m
-
\ell(\hat v,v)
\]

其中：

- \(\mu m\)：奖励跳过更多模型计算；
- \(\ell\)：惩罚近似 velocity 与真实 velocity 的偏差。

---

## 8.1 对我们的改造

我们可以把 velocity discrepancy 改成 edit intensity discrepancy：

\[
r(m)
=
\mu m
-
D(\hat u,u)
\]

其中：

\[
u=\lambda Q
\]

candidate arms 可以从：

\[
m\in\{0,1,2,4,6\}
\]

开始。

误差可以使用：

\[
D(\hat u,u)
=
\frac{
\|\hat u-u\|_1
}{
\|u\|_1+\epsilon
}
\]

也可以测试：

- log-intensity MSE；
- KL divergence；
- JS divergence；
- rate 与 conditional-Q 分开计算误差。

---

# 9. 为什么不建议一开始就上 Bandit？

Bandit 并不是 FastFlow 能加速的最核心原因。

真正核心的是：

> **相邻时间步的模型输出存在冗余，因此没有必要每一步都进行完整 neural evaluation。**

Bandit只是负责决定：

\[
\text{“这一次可以跳多远？”}
\]

所以首先应该验证：

\[
u_t
\]

在我们的实际采样轨迹中是否真的存在稳定区间。

---

# 10. 第一轮应该先做什么分析？

建议在当前 sampler 上记录：

\[
D(u_t,u_{t+1})
\]

以及：

\[
\mathbf 1[x_t=x_{t+1}]
\]

和：

\[
\Lambda_t
\]

---

## 10.1 建议画三类图

### 图 1：Edit intensity change vs timestep

例如：

\[
D_t
=
\frac{
\|u_{t+1}-u_t\|_1
}{
\|u_t\|_1+\epsilon
}
\]

观察是否存在：

```text
Early        Middle                         Late
██████       ▂▂▂▂▂▂▂▂▂▂▂▂                  █████
变化较大       大片稳定区域                     再次变化
```

---

### 图 2：State-change ratio vs timestep

统计：

\[
P(x_{t+1}\neq x_t)
\]

如果某些区间：

\[
P(x_{t+1}=x_t)
\]

非常高，那么 No-event Jump 会很有潜力。

---

### 图 3：Total hazard vs timestep

记录：

\[
\Lambda_t
\]

观察：

- 哪些 timestep hazard 很低；
- 哪些 timestep 编辑活动集中；
- 是否不同样本具有明显不同的困难程度。

如果 easy / hard reaction 的曲线有明显差异，就进一步支持 adaptive inference。

---

# 11. 如果分析结果不同，对应采取不同方案

## 情况 A：大量 timestep 中状态不变化

即：

\[
x_t=x_{t+1}
\]

很常见。

则优先做：

\[
\boxed{\text{No-event Jump}}
\]

---

## 情况 B：状态会变化，但 \(u_t\) 很平滑

则优先做：

\[
\boxed{\text{Edit-intensity extrapolation}}
\]

---

## 情况 C：\(u_t\) 和状态都频繁变化

则 FastFlow 式 output reuse 的收益可能有限。

此时应优先考虑：

\[
\boxed{\text{Adaptive large-step Poisson sampling}}
\]

即直接让时间步长根据 hazard 动态变化。

---

# 12. 当前代码中还存在的 Exact Acceleration 空间

在进行算法层面的 FastFlow-style 加速之前，建议先清理 sampler 中现有的 CPU/GPU 同步开销。

这些优化不会改变算法结果，因此应该单独作为 exact acceleration baseline。

---

## 12.1 `_token_keys_batch()` 每步都会 GPU → CPU

当前：

```python
rows = x_t.detach().cpu().tolist()
```

每个 timestep 都会触发 GPU → CPU synchronization。

然后还会在 Python 中进行：

- tuple 构造；
- dict grouping；
- child state comparison。

建议将状态比较尽量 tensorize，在 GPU 上完成。

---

## 12.2 `h_values = hc.squeeze(-1).cpu().tolist()`

同样会导致每一步 device synchronization。

当前所有 trajectory 的 \(t\) 在正常情况下基本同步，因此其实没必要：

- 每条 Branch 保存独立 Python float；
- 每一步将 h 从 GPU 搬回 CPU。

可以直接使用 tensorized time。

---

## 12.3 `_apply_edits_batch()` 中存在 `.item()` 和 `if tensor.any()`

例如：

```python
max_new_len = int(new_lengths.max().item())
```

以及：

```python
if keep_mask.any():
```

这些也可能造成同步。

可以考虑重写为更 GPU-friendly 的版本。

---

## 12.4 Python `Branch` / list / dict 开销

当前每一步：

- 创建大量 `Branch` 对象；
- 建 Python list；
- 建 dict；
- 对每个 branch 分别执行 selection。

尤其当前：

\[
M=2
\]

其实可以写一个专门的 tensorized K1M2 selection。

---

# 13. Exact sampler 优化后的理想形式

当前：

```text
Python Branch list
        ↓
GPU forward
        ↓
GPU → CPU token keys
        ↓
Python dict grouping
        ↓
Python child selection
        ↓
下一步
```

建议改成：

```text
Tensorized batch state
        ↓
GPU forward
        ↓
GPU tensor sampling
        ↓
GPU tensor state comparison
        ↓
GPU K1M selection
        ↓
下一步
```

之后再进行：

- `torch.compile`
- BF16 / FP16 inference benchmark
- fused attention / SDPA benchmark

这样后续算法带来的 wall-clock speedup 才更可信。

---

# 14. 推荐实验路线

建议按照下面的顺序推进。

---

## Stage 0：Exact engineering baseline

保持：

- 100 steps；
- R=9；
- M=2；
- 所有随机逻辑一致。

只做：

- sampler tensorization；
- 减少 CPU/GPU synchronization；
- compile / mixed precision benchmark。

目标：

\[
\text{Top-k 完全不变}
\]

只提升 wall-clock speed。

---

## Stage 1：Static Step Reduction

测试：

\[
N=100,\ 50,\ 25
\]

作为 baseline。

这一步用于证明：

> 简单减少推理步数虽然加速，但可能导致 Top-k / coverage 下降。

---

## Stage 2：Hazard-aware Adaptive Step

允许：

\[
\Delta t\in
\{0.01,0.02,0.04,0.06\}
\]

根据：

\[
\Lambda_t
\]

动态决定步长。

---

## Stage 3：State-preserving Event Skip

当：

\[
P_{\text{no-event}}
\]

足够高时，直接跳过多个时间步。

---

## Stage 4：Edit-intensity Extrapolation

使用：

\[
\hat u_{t+\Delta t}
\]

替代部分 Transformer forward。

但：

\[
x_t\text{ 一旦变化}
\]

立即恢复 exact evaluation。

---

## Stage 5：Bandit-guided Adaptive Inference

最终让 bandit 动态选择：

\[
m\in\{0,1,2,4,6\}
\]

并使用：

\[
r(m)
=
\mu m-D(\hat u,u)
\]

学习不同 timestep / 不同 reaction 的合适 skip length。

---

# 15. 最终实验表建议

| Method | Top-1 | Top-3 | Top-10 | Oracle | Validity | Avg. NFE ↓ | Time ↓ | Speedup ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-100 |  |  |  |  |  | 100 |  | 1.00× |
| Static-50 |  |  |  |  |  | 50 |  |  |
| Static-25 |  |  |  |  |  | 25 |  |  |
| Adaptive Step |  |  |  |  |  |  |  |  |
| Event Skip |  |  |  |  |  |  |  |  |
| Intensity Extrapolation |  |  |  |  |  |  |  |  |
| Bandit Adaptive |  |  |  |  |  |  |  |  |

建议同时报告：

- average NFE；
- median NFE；
- easy / hard reactions 分组 NFE；
- GPU wall-clock latency；
- speedup；
- Top-k；
- candidate coverage / oracle；
- validity。

---

# 16. 为什么当前阶段不建议减少 R 或 M？

当前推理中：

\[
R=9
\]

负责提供多条独立采样轨迹。

而：

\[
M=2
\]

每步只增加采样分支，并不会增加 Transformer forward 次数。

因此如果当前目标是：

> 在尽量不损失 candidate coverage 的情况下提升速度，

那么更合理的是：

\[
\boxed{\text{保持 }R=9,\ M=2,\ \text{优先降低每条轨迹的 NFE}}
\]

而不是直接减少采样轨迹数量。

---

# 17. 最值得当前立即尝试的方案

如果现在只选择一个方向，优先建议：

\[
\boxed{
\text{Hazard-aware Adaptive }\Delta t
+
\text{State-preserving Event Skipping}
}
\]

原因：

1. 与当前 `1-exp(-hλ)` 的 Poisson sampler 完全兼容；
2. 不需要重新训练；
3. 不需要修改模型结构；
4. 实现复杂度低；
5. 可以直接减少 Transformer forward；
6. 不需要一开始就引入额外网络；
7. 可以自然进一步升级到 FastFlow-style intensity extrapolation；
8. 若有效，最后再加入 bandit，可以形成完整的 adaptive inference contribution。

---

# 18. 一句话概括潜在 Contribution

可以把这一方向概括为：

> **现有离散编辑流在推理时采用固定时间离散化，对所有反应和所有生成阶段执行等量模型计算。我们观察到编辑强度在部分生成区间具有明显冗余，因此提出基于编辑 hazard 与局部强度变化的自适应推理策略，在稳定区间动态增大时间步长或跳过冗余神经网络评估，在编辑活跃区间恢复细粒度计算，从而在尽量保持候选质量与覆盖率的同时降低推理成本。**

如果进一步加入 bandit：

> **进一步将 skip length 选择建模为在线决策问题，根据局部 edit-intensity approximation error 自适应分配推理计算量，使简单反应使用更少 NFE，而复杂反应保留更多精确模型评估。**

---

# 19. 当前最值得做的下一步

在真正实现加速算法前，先在现有 dev1000 推理中增加日志，记录：

```text
timestep
x_t 是否变化
总 hazard Λ_t
u_t 与 u_{t-1} 的距离
每个 timestep 的实际编辑数量
```

然后回答三个问题：

1. **哪些时间段 state 几乎不变？**
2. **哪些时间段 edit intensity 比较平滑？**
3. **不同 reaction 的轨迹复杂度是否明显不同？**

这三个结果基本可以决定：

- Adaptive Step 是否可行；
- Event Skip 是否可行；
- FastFlow-style extrapolation 是否值得做；
- Bandit 是否真正有必要。

因此第一轮最关键的不是直接实现完整 FastFlow，而是先证明：

\[
\boxed{
\text{我们的离散编辑轨迹中确实存在可利用的推理冗余}
}
\]

这会让后续整个 acceleration contribution 更有逻辑，也更有实验依据。

# 前沿推理方法汇报：从 Inference-Time Scaling 到 Euler-Beam 的适配路线

> 用途：师兄汇报材料  
> 日期：2026-08-05  
> 主题：我们阅读了哪些前沿方法、哪些方法值得借鉴、它们分别解决什么问题、目前适配和验证到了哪一步。

本文的中心论文是 **Inference-Time Scaling of Discrete Diffusion Models via Importance
Weighting and Optimal Proposal Design**。它与当前 Euler-Beam 的关系最直接：两者都试图在
推理阶段使用多条粒子/分支来提高离散生成质量。但是，论文的关键贡献不是简单地增加分支，
而是把“目标分布、proposal、重要性权重和重采样”定义清楚。当前 Euler-Beam 还没有完全
达到这个严格意义上的 SMC（Sequential Monte Carlo）定义。

本文同时整理了项目中 PDF 目录下阅读过的其它方法，并把“论文已经证明的内容”“我们已经
实现并验证的内容”和“下一步待验证的假设”分开记录，避免把其它任务或其它模型上的结果
直接当作本项目收益。

---

## 1. 汇报先给结论

### 1.1 最值得借鉴的主线

第一优先级是把 Euler-Beam 中的“多分支 + 启发式排序”逐步改造成有明确概率意义的
Euler-SMC：

1. 每个粒子有明确的 proposal transition q；
2. 用 target/proposal 的逐步 log importance ratio 更新权重；
3. 用 ESS 判断是否需要重采样，而不是每一步无条件 Top-K；
4. 保存 ancestor lineage，分析粒子是否过早坍缩；
5. 只有引入独立的化学 reward 后，SMC 才有机会改变 proposal 的排序并带来准确率收益。

这条路线已经完成了 mechanics、Euler transition adapter 和 target=proposal bootstrap
的接口验证，但还没有接入独立 reward，因此目前不能声称 Euler-SMC 已经提升了 Top-k。

### 1.2 已经做过、但不建议直接替换默认值的方向

Edit Flows 原论文中的 Q sharpening 已经适配到当前采样器。它的优点是几乎不增加计算，
但在不重叠 validation 区间上没有稳定收益：

- T=0.9 在 validation-A/B 的局部 Top-3 有改善；
- 更大的 validation-C 没有复现该改善；
- T<1 会使 proposal 更集中，true unique 和部分 Oracle/Top-10 可能下降；
- 因此当前默认仍是 T=1.0。

这说明“让概率分布更尖”不是普适的准确率修复，尤其不能只看 Top-1。

### 1.3 需要新训练或新模型的方向

- **Discrete Guidance Matching**：有潜力避免把离散 token 当作连续向量做一阶 guidance，
  但需要额外 guidance/density-ratio 模型，并且要重新推导可变长度 insert/delete edit
  的 posterior，尚未实现。
- **显式 conditional flow + CFG**：可能提高 product 条件建模能力，但当前 checkpoint
  没有可用于 CFG 的 unconditional/condition-dropout 训练结构，不能只在推理脚本中补。
- **reverse-rate corrector、localized edit**：需要新的训练或 CTMC 推导，暂不进入当前
  主线。
- **RetroAgent**：解决的是多步逆合成路线规划，不是当前单步 USPTO-50K Top-k 采样器的
  直接替换。

---

## 2. 阅读范围与文献位置

### 2.1 本项目实际阅读的 PDF

| 方法 | 主要层次 | 与本项目的关系 | 当前状态 |
|---|---|---|---|
| Edit Flows: Flow Matching with Edit Operations（2025） | 训练目标、edit CTMC、推理策略 | 当前 checkpoint 和 Euler sampler 的基础 | 已用于 Q sharpening 分析 |
| Discrete Guidance Matching（ICLR 2026） | 离散 posterior guidance | 可能解决离散 guidance 近似问题 | 理论适配待做 |
| Inference-Time Scaling of Discrete Diffusion Models via Importance Weighting and Optimal Proposal Design（ICLR 2026） | SMC、importance weighting、optimal proposal | 与多分支 Euler-Beam 最相近 | mechanics 已实现，reward 待接入 |
| RetroAgent（COLM 2026） | LLM + AND-OR graph 多步路线规划 | 可作为上层 planner | 不属于当前单步优化主线 |

本地原文位置：

~~~text
PDF/2025--Edit Flows Flow Matching with Edit Operations.pdf
PDF/2026--Discrete Guidance Matching Exact Guidance for Discrete Flow Matching.pdf
PDF/2026--Inference-Time Scaling of Discrete Diffusion Models via Importance Weighting and Optimal Proposal De.pdf
PDF/2026--RetroAgent Harnessing LLMs to Search Over Structured Memory for Agentic Retrosynthesis Planning.pdf
~~~

主论文的公开版本为 [arXiv:2505.22524](https://arxiv.org/abs/2505.22524)，作者为
Zijing Ou、Chinmay Pani 和 Yingzhen Li；项目文件标注为 ICLR 2026。报告中的论文方法和
论文实验结论以原文为准，项目实验以本仓库的代码、metadata 和结果目录为准。

### 2.2 需要区分的三个概念

1. **采样 proposal**：模型实际用来生成下一状态的分布 q。
2. **目标 target**：我们希望粒子近似的分布 π，例如基础模型分布加独立 reward 的
   reward-tilted 分布。
3. **最终评分**：本项目的 Top-1～10、Oracle、invalid 等是离线评价指标。测试集 target
   只能用于最终评价，不能在采样时充当 reward。

如果 target 没有独立定义，或者 target 与 proposal 完全相同，那么 SMC 只能验证采样
mechanics，不能凭空提高化学准确率。

---

## 3. 主论文：Inference-Time Scaling 的核心思想

### 3.1 论文要解决什么问题

离散 diffusion 模型通常已经训练完成，但实际生成还要满足额外约束，例如毒性控制、
DNA 活性、图像文本对齐或其它 reward。重新训练模型成本高，而且直接 guidance 容易
under-optimise reward，直接 fine-tuning 又可能 reward over-optimisation、损伤质量和
多样性。

论文的目标是：**不改变或少改变预训练模型，通过增加 inference-time compute，并用有
概率意义的 SMC 进行 reward alignment。**

### 3.2 从 importance sampling 到 SMC

设当前模型给出的离散反向 transition 是 proposal：

$$
q(x_{t-1}\mid x_t).
$$

我们想从某个 target transition/path measure 采样。论文的逐步权重递推可写成：

$$
w_{t-1}
=w_t\,
\frac{\pi(x_{t-1})}{\pi(x_t)}
\frac{\gamma(x_t\mid x_{t-1})}
     {q(x_{t-1}\mid x_t)},
$$

其中：

- π 是中间目标分布；
- q 是实际 proposal；
- γ 是用于构造可计算比值的 forward/noising kernel；
- 权重在 log 空间累加，避免数值下溢。

每个时间步的基本流程是：

1. 按当前权重选择或重采样 ancestor；
2. 用 q 生成新的粒子；
3. 计算该粒子的增量 log importance weight；
4. 归一化权重并计算 ESS；
5. 只有在粒子退化明显时才重采样；
6. 保留粒子的祖先关系和诊断信息。

有效样本数为：

$$
\mathrm{ESS}=\frac{1}{\sum_i \tilde w_i^2},
$$

其中 \(\tilde w_i\) 是归一化权重。ESS 小表示权重集中在少数粒子上，继续不加控制地
推进会导致 particle degeneracy。

### 3.3 论文中的 target 设计

论文重点研究两类 target：

1. **Product target**：多个模型分布的乘积，包含 CFG 的一般形式；
2. **Reward-tilting target**：

   $$
   \pi_t(x)\propto p_\theta(x)\exp\left(\frac{\lambda_t}{\alpha}r(x)\right).
   $$

   其中 \(r(x)\) 是 reward，\(\alpha\) 控制 reward 与原模型分布之间的 KL 权衡，
   \(\lambda_t\) 随 denoising 时间逐渐打开，避免早期粒子被 reward 过早拉到错误模式。

如果 reward 只在干净终点可计算，论文使用对终点预测的 Monte Carlo 估计：

$$
\hat r(x_t)=\frac{1}{M}\sum_{m=1}^M r(x_0^{(m)}),
\quad x_0^{(m)}\sim p_\theta(x_0\mid x_t).
$$

这里的 M 是 reward estimate 的内部样本数，不应与当前 Euler-Beam 的每个分支
n_children 混为一谈。

### 3.4 论文为什么强调 proposal

论文的关键观点是：SMC 的效果不只取决于粒子数，还取决于 proposal 是否接近局部最优
proposal。对 reward-tilting，局部最优形式近似为：

$$
q^*(x_{t-1}\mid x_t)
\propto \exp(r(x_{t-1}))p_\theta(x_{t-1}\mid x_t).
$$

直接计算其归一化常数通常太慢，因此论文研究两种近似：

- **first-order/gradient proposal**：在当前状态附近对 reward 做一阶展开，减少枚举代价，
  但需要可微 reward，离散采样通常还要 Gumbel-Softmax 等近似；
- **amortised proposal**：训练一个 proposal network，直接最小化 importance weight 的
  log-variance，使重要性权重更稳定、ESS 更高。

这解释了为什么“多生成一些 child”不一定有效：如果 q 本身没有覆盖正确模式，增加
粒子只是更快地重复同一个错误 proposal。

### 3.5 论文实验给出的可迁移结论

论文在 synthetic discrete tasks、毒性文本、DNA design 和 text-to-image 上验证了：

- SMC 可以在固定预训练模型的基础上做 inference-time control；
- 粒子数增加通常提高 reward/对齐质量，但可能牺牲多样性；
- 更接近 optimal 的 proposal 通常比固定 pretrained proposal 更有效；
- amortised proposal 的 reward/ESS 表现好，但可能出现 mode-seeking 和 reward
  over-optimisation；
- denoising steps 已经足够多时，额外粒子的收益会减弱；
- reward schedule 太早或太快打开可能造成高方差和不可逆错误。

这些结果支持“有 target 的粒子方法值得研究”，但没有证明当前 Edit Flows checkpoint
在没有独立化学 reward 的情况下必然收益。

---

## 4. 当前 Euler-Beam 与论文 SMC 的差异

| 对比项 | 当前 Euler-Beam | 论文式 SMC |
|---|---|---|
| 状态 | Euler 编辑后的 SMILES 分支 | 带权粒子及其 path/ancestor |
| proposal | Euler 模型 + event/token sampling | 明确定义的 q |
| child 数量 | 每个 parent 生成 M 个 child | 可按粒子数和 proposal 采样 |
| 选择方式 | 按 log mass、collision 和 bonus 做确定性 Top-K | importance weight + ESS + resampling |
| 目标分布 | 没有独立 π；bonus 是启发式 | π 明确，可以是模型分布加 reward |
| 概率校正 | 历史上偏向“采样后排序” | 使用 log target/log proposal 比值 |
| 多样性诊断 | true unique、重复状态统计 | ESS、祖先数、权重方差、genealogy |
| seed | 每条分支稳定派生，保证可复现 | 每个 product/particle/step 的独立稳定 seed |
| 结果解释 | “保留 K 条高分路径” | “近似 target 的加权粒子族” |

因此，当前 Euler-Beam 的“合并与复制”确实在外观上接近粒子系统，但不能把
changed-state bonus 或 Top-K 剪枝直接称为 importance weighting。论文真正值得借鉴的
是概率接口和诊断方式，而不是简单把变量名改成 particle。

---

## 5. 适配到当前项目的方案与进度

### 5.1 阶段 0：冻结现有基线

保持现有 Euler-Beam、Euler、checkpoint、数据集和评分器不变。所有新方法都使用独立
入口或 opt-in 参数，确保可以与当前 Top-1～10、invalid、true unique 和 wall 做公平
比较。

当前工程约定：

- 默认高准确率配置：R9K1M2；
- 速度平衡配置：R3K3M2；
- proposal 的 seed 必须与 batch 切分无关；
- total child budget、n_steps、TF32/FP32、batch size 和输出数量要明确记录；
- tiny/test-mini 只能用于回归和描述，不能用测试 target 选择超参数。

### 5.2 阶段 1：SMC mechanics（已完成）

新增文件：edit_flows/sampling/euler_smc.py。已实现：

- log-weight normalization；
- log-space ESS；
- systematic resampling；
- product/step 稳定派生 seed；
- ancestor id 传播；
- 输入合法性和有限值检查。

已新增 tests/sampling/test_euler_smc.py，覆盖 11 项 mechanics 测试，包括归一化、ESS、
importance ratio、resampling 确定性和 batch/layout invariance。

这一阶段只证明“SMC 的数学机械没有明显实现错误”，不证明准确率提升。

### 5.3 阶段 2：Euler transition adapter 与 bootstrap（已完成）

euler_transition_step() 复用 Euler-Beam 的无状态 action sampling、Poisson event semantics、
编辑应用和完整 step log-prob，一次执行一个 batched Euler proposal。当前明确拒绝
event_prob_mode=linear，因为还没有与其匹配的精确 log-prob scorer。

bootstrap rollout 设：

$$
\text{target}=\text{proposal}.
$$

因此理论上：

- log importance increment = 0；
- 权重保持均匀；
- ESS = 粒子数；
- 不应因为“换成 SMC”凭空产生准确率收益。

真实 checkpoint smoke 已完成：

- 2 行、1 step CUDA transition：next state shape 和 proposal log-prob 有限；
- 1 个 product、3 个粒子、4 steps：ESS=[3,3,3,3]，无 resampling，累计 evidence=0；
- GPU smoke 约 0.99s，仅用于接口量级，不是正式效率对比。

### 5.4 阶段 3：独立 terminal reward（下一阶段）

要让 SMC 真正改变排序，需要独立于反向 Edit Flows 模型、且不读取测试 target 的 reward。
推荐优先顺序：

1. 独立 forward reaction model 的 product/reaction consistency；
2. 独立 feasibility/classifier；
3. RDKit validity、原子守恒、价态和片段约束等弱化学 reward；
4. 反向模型自身 log-prob 只能做 proposal 诊断，不能冒充独立 reward。

第一版只做 terminal twisting：

$$
\log \pi(x_1)-\log q(x_1)
=\beta R(x_1).
$$

中间 100 步仍由 Euler proposal 推进，最后用 reward 更新粒子权重。只有 terminal reward
在不重叠 validation 区间上稳定改善 Top-3/10 或 Oracle，才研究 intermediate twisting。

### 5.5 阶段 4：intermediate target 与 optimal proposal（后续）

如果 terminal reward 有效，再按论文思想逐步引入 \(\lambda_t\) schedule：

- 早期保持 base model 影响；
- 中后期逐渐增加 reward；
- 监测 ESS、ancestor diversity 和 reward variance；
- 先比较固定 proposal，再比较 gradient proposal；
- 只有 reward 模型稳定且计算成本可接受，才考虑 amortised proposal。

当前不同时加入 Q temperature、changed-state bonus、learned proposal 和新的 child policy，
否则无法判断收益来源。

---

## 6. 其它阅读方法：价值、问题、适配与结论

下面每种方法都按照“方法本身 → 为什么读 → 可以解决什么问题 → 如何适配 → 实验占位
与结论”记录。

### 6.1 Edit Flows：Q sharpening、CFG、reverse 和 localized edit

#### 方法与动机

Edit Flows 是当前项目的基础工作，使用 insert/substitute/delete 等 edit operation 构造
离散 flow/CTMC。论文还讨论了对 token 条件分布做 temperature、top-k、top-p，及
conditional guidance、reverse rates、localized operations 等推理或训练方向。

我们读它的原因是：当前 checkpoint 来自该路线，任何 sampler 改动都必须尊重 rate head、
event semantics 和变量长度编辑。

#### 可以解决的问题与风险

- Q sharpening 可能降低错误 token 和 invalid；
- CFG 需要可靠的 conditional/unconditional 训练结构；
- reverse rates 可能撤销早期错误编辑；
- localized edit 可能改善局部括号、环标号和官能团一致性。

风险是 proposal 过尖导致多样性下降；CFG 若没有 condition dropout 就只是伪造 guidance；
reverse/localized edit 会改变 CTMC 语义，不能仅靠脚本层拼接。

#### 本项目适配

Q sharpening 已在 euler_beam.py 中实现 q_temperature，T=1.0 保持兼容，T<1 只作用于
insert/substitute token 分布，不改变 event rate。当前 checkpoint 没有 origin_embedding
权重，且训练配置 use_origin_mask=False，因此不应在推理时强行打开 origin condition。

显式 CFG 需要新训练：product conditioning、condition dropout、unconditional 分支和
单独 checkpoint。reverse-rate/localized edit 需要新的 rate/loss 推导及专门实验。

#### 实验占位与已知结论

Q sharpening 的正式实验已完成，摘要如下：

| split | T | Top-1 | Top-3 | Top-10 | Oracle | invalid | true unique | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation-A（50反应） | 1.0 | 74 | 92 | 96 | 98 | 18.8% | 27.40 | 105.338s |
| validation-A（50反应） | 0.9 | 72 | 94 | 96 | 98 | 19.3% | 26.94 | 105.858s |
| validation-B（50反应） | 1.0 | 62 | 84 | 96 | 98 | 9.4% | 18.84 | 100.408s |
| validation-B（50反应） | 0.9 | 62 | 86 | 96 | 98 | 8.8% | 18.58 | 99.469s |
| validation-C（200反应） | 1.0 | 50.0 | 74.5 | 86.0 | 91.5 | 13.675% | 26.315 | 395.965s |
| validation-C（200反应） | 0.9 | 51.0 | 74.0 | 86.0 | 91.0 | 13.425% | 25.775 | 386.366s |

tiny 的 post-hoc 对照也做过，但 tiny 是历史 test 开发子集，不能用于选温度：
T=0.9 只提高了 Top-2，Top-3/Top-10/Oracle 没有同步提高；T=0.8 的 Top-1 和 Oracle
进一步下降。因此当前默认保持 T=1.0。

**结论：** Q sharpening 是低成本可用的诊断开关，但不是已经证明的主改进；CFG、
reverse、localized edit 暂不进入 sampler 主线。

### 6.2 Discrete Guidance Matching

#### 方法与动机

该方向学习 endpoint density ratio 或 posterior-based guidance，在离散状态空间直接修正
后验，而不是把离散 token 简单当作连续向量求 score。它的目标是得到更准确、更高效的
discrete guidance。

#### 可以解决的问题与风险

它可能解决当前“用启发式 bonus 或温度近似 reward guidance”的问题，降低离散一阶
近似的偏差，并在每一步减少枚举所有候选状态的成本。

主要障碍是当前 Edit Flows 的状态不是固定维度 token 替换：insert/delete 会改变序列长度
和位置，edit operation posterior 还涉及合法 SMILES 结构。若直接把 guidance ratio 乘到
Q_ins/Q_sub 上，不能称为 exact guidance，也可能破坏 event rate 与 log-prob 一致性。

#### 本项目适配

需要先定义：

1. base：p(reactants | product) 的 Euler edit transition；
2. target：由独立 forward consistency 或 feasibility reward 倾斜后的分布；
3. guidance：作用在 insert/substitute/delete operation posterior 还是 token posterior；
4. variable-length alignment：如何处理 insert/delete 后的位置映射和状态空间支持集。

建议先完成 Euler-SMC terminal reward，确认 reward 有效且 reward forward 成为主要瓶颈，
再推导 guidance network；不要在没有 reward 的情况下先训练一个“看起来像 guidance”的
额外网络。

#### 实验占位与结论

当前没有实现和准确率实验。未来实验必须同时报告：

- 与 Euler-SMC 固定 particle/child budget 的 Top-1～10、Oracle；
- importance weight variance、ESS 和 invalid；
- guidance forward 次数与 wall；
- 是否保持 proposal support，是否出现状态非法或概率不闭合。

**结论：** 理论潜力高，但适配成本和正确性风险都高，优先级低于已完成的 SMC mechanics。

### 6.3 Inference-Time Scaling 主论文

#### 方法与动机

论文用 SMC 把“额外推理计算”变成可解释的粒子推断：用 proposal 探索，用 importance
weight 修正 target，用 ESS/resampling 控制粒子退化，再用 first-order 或 amortised
proposal 降低权重方差。

#### 可以解决的问题与风险

它直接对应当前 Euler-Beam 的三个问题：

1. M 个 child 是否只是重复 proposal，还是提供新的 target 覆盖？
2. K 个分支何时该复制、何时该淘汰？
3. bonus/Top-K 排序是否具有概率意义？

潜在收益是固定预算下更合理地保护正确模式，并通过 ESS 解释多样性。风险是 reward 错误
或权重比值错误会造成更严重坍缩；reward 也可能成为推理瓶颈。

#### 本项目适配

已经按独立新方法实现：

- edit_flows/sampling/euler_smc.py：粒子、weight、ESS、systematic resampling；
- euler_transition_step()：复用 Euler proposal，并返回 step log-prob；
- run_euler_smc_bootstrap()：target=proposal 的正确性基线；
- tests/sampling/test_euler_smc.py：11 项新测试。

下一步是单独实现 terminal_reward，不读取 test target；然后在固定总 child budget 下与
R9K1M2 对照。若 reward 使 ESS 快速降到 1 附近，或者 Top-10/Oracle 下降，就停止该 reward
方向，不能只因为 Top-1 偶尔上升而保留。

#### 实验占位与已知结论

| 阶段 | 已做实验 | 结果 | 解释 |
|---|---|---|---|
| synthetic mechanics | 11 项单元测试 | 全部通过 | weight/ESS/resampling 接口正确 |
| real transition smoke | 2 行、1 step CUDA | shape、finite log-prob 通过 | proposal adapter 可调用 |
| real bootstrap | 1 product、N=3、4 steps | ESS=3、无 resampling、evidence=0 | target=proposal 没有凭空制造偏置 |
| validation accuracy | 尚未运行 | 无 Top-k 结论 | 尚未有独立 target/reward |

**结论：** SMC 方向值得继续，但目前完成的是“正确性基础设施”，不是已经验证的准确率
方法。

### 6.4 RetroAgent

#### 方法与动机

RetroAgent 把逆合成看作多步路线规划：用 LLM 在 AND-OR graph、结构化 memory、building
block 数据库和化学工具上决定下一步扩展哪个 molecule、使用哪个反应模板，并传播
solved/open 状态。它解决的是完整路线是否能落到可购买原料，而不是单步反应候选排序。

#### 可以解决的问题与风险

如果我们的研究目标以后变成 route success、route length 或 building-block availability，
RetroAgent 的上层搜索和记忆机制有借鉴价值。它不能自动修复当前单步 Euler-Beam 的
Top-1～10，而且每条路线需要多次 LLM/tool 调用，成本和实验变量都会显著增加。

#### 本项目适配

可以把当前 Euler-Beam 当作 single-step candidate proposer，再新增上层 planner：

1. molecule OR node；
2. reaction AND node；
3. canonicalization、去重和 cycle prevention；
4. building-block availability、SA score、route depth/cost；
5. 先用 Retro*/A* 或轻量 policy 建立非 LLM baseline；
6. 最后才判断是否需要 LLM memory/search。

该方向应单独建立 research/retro-planner，不与当前单步 sampler 的 Top-k 主线混合。

#### 实验占位与结论

当前没有多步路线实验。未来至少要报告 route success、平均深度、搜索节点数、LLM/tool
调用数、时间和单步候选质量。

**结论：** 这是潜在的上层产品方向，不是当前 Euler-Beam 推理改进的直接候选。

---

## 7. 训练层面的边界

本次阅读也暴露出一个重要边界：部分前沿方法不是采样脚本能补出来的。

### 7.1 当前 checkpoint 能做什么

- 可以复用当前反向 rate/model 做 Euler、Euler-Beam、Q temperature 和 proposal adapter；
- 可以做严格 seed、batch/layout invariance；
- 可以建立独立 SMC mechanics；
- 可以在有外部 reward 时做 terminal/intermediate weighting。

### 7.2 当前 checkpoint 不能假设具备什么

- 没有 origin_embedding 权重，不能凭空打开 origin mask condition；
- 训练配置 use_origin_mask=False，不能在推理时伪造 conditional/unconditional pair；
- 没有经过 condition dropout 的明确 CFG 训练，就不能声称 CFG 已适配；
- 没有独立 forward consistency/feasibility reward，就不能声称 reward-guided SMC 已经成立；
- 没有 reverse-rate 或 localized-edit 专用训练，就不能只改 sampler 名称实现这些方法。

因此，训练审计仍然有价值，但训练代码、数据集和 checkpoint 不属于本次文献汇报的
无授权大范围修改范围。若确定开训练分支，必须独立记录数据边界、loss、checkpoint 和
对照实验。

---

## 8. 已完成实验的统一解读

### 8.1 Q temperature 的证据

当前 validation-A/B/C 的实验说明：

- T=0.9 的局部收益不能在较大不重叠 validation-C 稳定复现；
- T=0.8 更容易牺牲 unique/Oracle；
- wall 基本不变，说明这是 proposal 形状变化，不是效率优化；
- 因此不把 T<1 设为默认，也不继续在 test-mini 上调 top-k/top-p。

### 8.2 Euler-SMC 的证据

当前 11 项新测试、针对性回归和真实 checkpoint smoke 说明：

- log-space weight、ESS、systematic resampling 和 genealogy 接口可用；
- batch 切分不改变派生 seed；
- target=proposal bootstrap 保持均匀权重；
- 还没有独立 target/reward，因此没有 Top-k 提升或正式 wall 结论。

### 8.3 当前结果不能怎样解读

不能把以下现象直接解释成“新方法有效”：

- 只在 tiny 上 Top-1 偶尔上升；
- 只增加输出数后 Top-10 上升；
- invalid 下降但 true unique/Oracle 同时下降；
- target=proposal bootstrap 的 ESS=N；
- 改变 R/K/M 后的总输出预算不同，却直接比较百分比。

正式比较必须固定总 child budget、输出预算、n_steps、seed、batch、TF32/FP32、评分器和
augmentation 聚合方式。

---

## 9. 下一步实验计划（适合向师兄汇报）

### 9.1 首个真正的 Euler-SMC accuracy experiment

先不改现有 Euler-Beam 默认入口，创建独立 SMC 实验：

1. 选择一个独立 reward。首选 forward reaction consistency；如果暂时没有，则先用
   RDKit/化学约束做诊断，不宣称最终收益。
2. 仅做 terminal twisting，固定 R9K1M2 的总 child budget。
3. 在不重叠 validation-A/B/C 上预注册 beta 小集合，不能使用 test target 调参。
4. 记录 Top-1～10、Oracle、invalid、true unique、mean/p10 ESS、resampling 次数、
   ancestor 数、reward 分布、forward 次数、显存和 wall。
5. 通过门槛：Top-1 不明显回退，且 Top-3/10 或 Oracle 在不重叠 validation 上稳定提高，
   同时 ESS 不系统性坍缩、invalid 下降不以覆盖率为代价。

### 9.2 如果 terminal reward 有效

按以下顺序增加复杂度：

1. intermediate λ_t schedule；
2. 相同 reward 下比较 base proposal 与 first-order proposal；
3. 只有额外 forward 成为瓶颈且数据足够时，训练 amortised proposal；
4. 再评估是否需要 Discrete Guidance Matching 的 operation-level guidance。

每一步都要有单变量消融和 prediction metadata，不能把多个论文想法一次性叠加。

### 9.3 如果 terminal reward 无效

记录失败原因并停止该 reward，不用更多 R/K/M 掩盖问题。下一选择是：

- 训练显式 product-conditioned + condition-dropout 模型，研究 CFG；
- 或研究 reverse-rate corrector；
- 或转向多步路线规划的 RetroAgent-style 独立分支。

这些方向需要师兄确认研究目标，因为它们已经超出“当前单步 Euler-Beam sampler 改进”的
范围。

---

## 10. 一页式方法决策表

| 方法 | 解决的主要问题 | 是否需重训 | 当前证据 | 推荐度 |
|---|---|---:|---|---:|
| Q sharpening | token proposal 过平、错误 token | 否 | 已验证但无稳定收益 | 低成本诊断 |
| Euler-SMC mechanics | 启发式 Top-K、缺少权重/ESS | 否 | mechanics 和 bootstrap 通过 | 最高优先级 |
| Terminal reward + SMC | 没有独立的未来质量信号 | 需外部 reward | 尚未实验 | 下一步 |
| Intermediate twisting | reward 过晚/过早造成不稳定 | 否/外部 reward | 尚未实验 | terminal 成功后 |
| First-order optimal proposal | reward proposal 方差高 | 需可微 reward | 尚未实验 | 中期 |
| Amortised proposal | 多次 reward/forward 太慢 | 是 | 论文有跨任务证据 | 后期 |
| Discrete Guidance Matching | 离散 guidance 近似偏差 | 是 | 尚未适配变量长度 edit | 高风险后期 |
| Explicit conditioning + CFG | product 条件建模不足 | 是 | 当前 checkpoint 不支持 | 训练主线 |
| Reverse/localized Edit Flow | 早期错误、局部结构一致性 | 是 | 尚未适配 | 后期 |
| RetroAgent planner | 多步路线搜索 | 新系统 | 单步未验证 | 研究目标改变时 |

---

## 11. 代码、文档和 Git 记录

### 11.1 已修改或新增的相关代码

- edit_flows/sampling/euler_beam.py：Q temperature 和共享 forward 等已有 opt-in 改动；
- edit_flows/sampling/euler_smc.py：独立 SMC mechanics、Euler transition adapter、bootstrap；
- tests/sampling/test_euler_smc.py：SMC 单元测试；
- scripts/sample_retro.py、scripts/eval.py：采样/评价入口和 metadata 兼容；
- new_docs/frontier_methods_research.md：前沿方向总览；
- new_docs/euler_beam_next_stage_plan.md：任务29、30、31 的执行记录。

### 11.2 已完成的相关 commit

| 内容 | commit |
|---|---|
| Q temperature 实现 | 67ae17d |
| Q temperature validation 记录 | 7adf233 |
| Euler-SMC mechanics | fd149f1 |
| Euler transition adapter | 3553b1b |
| Euler-SMC bootstrap | c251843 |
| bootstrap smoke 记录 | d41c729 |
| Task29 tiny 对照记录 | 7206878、d682036 |

本报告应单独提交，避免把 PDF、历史恢复脚本和无关可视化修改带入本次 commit。

---

## 12. 汇报时可以直接使用的表述

> 我阅读了四类方向。Edit Flows 本身给了我们 Q sharpening、CFG、reverse 和 localized
> edit 的基础，但当前 checkpoint 只适合直接验证 Q temperature；该实验没有得到稳定的
> accuracy 提升。Discrete Guidance Matching 解决的是离散 guidance 的近似偏差，不过
> 我们的 insert/delete 会改变序列长度，必须重新推导 operation-level posterior，暂时不
> 能直接套用。
>
> 与 Euler-Beam 最接近的是 Inference-Time Scaling 论文。它的核心不是简单增加分支，而
> 是用 SMC 明确定义 proposal、target、importance weight、ESS 和 resampling。我们已经
> 在不影响现有 sampler 的独立文件中完成了 mechanics、Euler transition adapter 和
> target=proposal bootstrap；测试和 checkpoint smoke 都通过，但因为还没有独立化学 reward，
> 目前没有把它报告成准确率改进。下一步是先选一个不读取测试 target 的 forward consistency
> reward，做 terminal twisting 的固定预算 validation 实验。
>
> RetroAgent 则是更上层的多步路线规划方法，未来可以把我们的单步候选作为它的扩展器，
> 但它不直接解决当前单步 Top-k。因此目前最严谨的主线是：先完成有独立 reward 的
> Euler-SMC 正确性和收益验证，再决定是否进入 guidance 或重新训练。

---

## 13. 参考资料

1. [Ou, Pani, Li. Inference-Time Scaling of Discrete Diffusion Models via Importance
   Weighting and Optimal Proposal Design, arXiv:2505.22524](https://arxiv.org/abs/2505.22524)
2. 本地 PDF：PDF/2025--Edit Flows Flow Matching with Edit Operations.pdf
3. 本地 PDF：PDF/2026--Discrete Guidance Matching Exact Guidance for Discrete Flow Matching.pdf
4. 本地 PDF：PDF/2026--RetroAgent Harnessing LLMs to Search Over Structured Memory for Agentic
   Retrosynthesis Planning.pdf
5. 项目规划：new_docs/euler_beam_next_stage_plan.md
6. 前沿方法总览：new_docs/frontier_methods_research.md

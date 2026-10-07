# Euler-Beam 当前局势与机制复盘

日期：2026-08-04

本文只总结当前checkpoint和现有推理实现，不把test-mini工程选型结果解释为未见测试集上的
最终论文结论。权威实验记录仍在`euler_beam_next_stage_plan.md`；早期
`euler_beam_report.md`中的“所有Top-k一致优于Euler”等表述已经被更严格实验推翻，不能
继续作为当前结论。

## 1. 结论先行

当前创新是**部分有效，而不是完整成立**：

1. 当前`M=2`多后继、局部选择、changed-state bonus和单次no-op anchor组成的整套策略，
   相对Euler式单轨迹能把更多正确候选提前到Top-1～Top-5；这是目前最可信的正向收益。
2. 该收益主要是**排序集中化**，不是大幅增加覆盖。相同200反应上R9相对纯Euler的
   Top-1/2/3提高4.0/4.5/3.5pp，但Top-10相同，Oracle只提高1.5pp；R10K1M2相对
   R10K1M1也显著提高Top-3/5，但Oracle几乎不变。
3. “长期维护大K、合并并全局Top-K剪枝”目前没有准确率收益证据。相反，同9条初始流的
   R1K9比R9K1快54.4%，但Top-1/3/10和Oracle低2.90/5.20/2.60/4.80pp。
4. 因此当前最高准确率方法`R9K1M2`本质上更接近**9条隔离的 multi-try Euler轨迹**，
   不是成熟的宽beam search。每个run内部K=1，每步只从两个随机child中保留一个；不同
   run之间不合并、不竞争。
5. 当前没有单一配置同时支配准确率和速度。R9K1M2是准确率配置，R3K3M2是平衡配置，
   R1K9M2是同9输出预算的速度配置。

## 2. 把R、K、M拆开

对每条augmentation：

```text
R = 隔离搜索池数量
K = 每个搜索池长期保留的父状态上限
M = 每个父状态在一个Euler step生成的child数
输出数 = R × K
单步最大候选数 = R × K × M
```

Transformer只对保留的父状态做forward；M只复制较轻的动作采样、编辑与候选处理，不把
Transformer forward放大M倍。候选产生后，同一池内相同token state用`logsumexp`合并经验
质量，再按以下主键保留Top-K：

```text
empirical log_mass
+ changed_state_bonus × I[state != product]
+ deterministic seed tie-break
```

`path_log_p`虽然按完整Euler事件概率计算，但在M>1正式模式下只作诊断，不参与Top-K主排序；
否则会对已经按模型分布采样的child再次乘概率，产生概率平方偏置。这意味着当前方法不是
精确action枚举beam，而是带启发式先验的Monte Carlo状态频次筛选。

在K1M2中，如果两个child不同且都已偏离product，它们通常具有相同经验质量，最终主要由
seed tie-break决定；bonus只区分“仍为product”和“已改变”状态。`stochastic_noop`也只在
约`t=0.9`的一个step把第二个child设为no-op，不是在每一步都固定一个no-op child。

## 3. 相对纯Euler是否有效

### 3.1 当前同预算方向实验

固定test-mini前200反应、20倍augmentation、每条输入9个输出、100 steps、TF32 high和
legacy aggregation。纯Euler通过同进程`torch.manual_seed(42)`固定全局RNG，batch16；
R9复用完整mini已有结果的相同前缀。

| 方法 | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Oracle | valid/slots | unique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Euler N9 | 54.0 | 69.5 | 77.5 | 82.5 | 88.0 | 94.5 | 85.856% | 25.485 |
| R9K1M2 | 58.0 | 74.0 | 81.0 | 86.0 | 88.0 | 96.0 | 85.125% | 26.670 |
| R9-Euler (pp) | +4.0 | +4.5 | +3.5 | +3.5 | 0.0 | +1.5 | -0.731 | +1.185 |

逐反应R9-only/Euler-only为Top-1 `13/5`（双侧exact McNemar `p=0.0963`）、Top-2
`15/6`（`p=0.0784`）、Top-3 `15/8`（`p=0.210`）、Top-10 `6/6`、Oracle `5/2`。
方向支持R9把正确候选前移，但200反应还不足以给出统计显著结论；Top-10和Oracle更说明
新增收益主要来自排序而非覆盖。

纯Euler采样4000条augmentation输入耗时523.35秒，吞吐7.643 input/s；完整mini R9耗时
3059.83秒，吞吐6.543 input/s。两者运行长度和batch不同，不能作严格wall配对，但当前
测量表明R9约慢17%的量级，而不是数量级回归。

### 3.2 这个对照的限制

当前`sample_retro.py --sampler euler --seed`尚未把seed传进普通Euler sampler，metadata
正确记录为`seed_applied_to_sampler=false`。本实验用外部同进程seed固定了这一次运行，
但还没有做到Euler-Beam已有的per-product、与batch切分无关的稳定随机流。因此：

- 可以说“当前R9有提高前段Top-k的正向证据”；
- 不能说“已在完整mini/full test严格证明全面优于纯Euler”；
- 在修复Euler seed并完成独立validation前，不能恢复早期报告的全面优越结论。

R10K1M1是另一个有用代理：K1/M1没有分支竞争，分布语义接近10条Euler轨迹，但RNG、
categorical实现和bookkeeping不同，不能与`sample_euler()`逐字节等同。

## 4. 多child策略是否有效

### 4.1 当前整套M2策略相对M1

完整test-mini-1001结果：

| 配置 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | wall |
|---|---:|---:|---:|---:|---:|---:|
| R10K1M1, stochastic, bonus0 | 56.344 | 76.224 | 80.120 | 84.915 | 92.308 | 3022.81s |
| R10K1M2, noop, bonus0.5 | 56.943 | 78.422 | 82.418 | 86.014 | 92.408 | 3425.59s |
| M2-M1 | +0.599 | +2.198 | +2.298 | +1.099 | +0.100 | +13.3% |

M2-only/M1-only在Top-3为`42/20`（`p=0.00715`），Top-5为`41/18`
（`p=0.00379`），Oracle为`21/20`。因此当前M2整套策略确实改善中段Top-k，但没有增加
总体目标覆盖。由于M1和M2还同时改变了policy和有效bonus，这证明的是“多child选择方案”
整体有效，不能把全部收益严格归因于child数这个单变量。

### 4.2 M不是越大越好

R1K10的M2/M3固定为纯`stochastic`、bonus0.5，是更严格的M单变量实验：

| 配置 | Top-1 | Top-3 | Top-10 | Oracle | wall | unique | shortfall |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2 | 52.547 | 71.628 | 81.618 | 87.213 | 1687.14s | 30.301 | 29.009% |
| M3 | 51.249 | 68.831 | 79.421 | 83.916 | 2065.51s | 31.887 | 14.763% |

M3虽然增加unique并减少补齐，却慢22.4%，Top-1～10和Oracle全面回退。更多child探索到了
更多字符串，但新增区域的化学有效性和目标相关性更差。M=4历史短筛也出现明显no-event
塌缩。因此目前只支持局部最优`M=2`，不支持“扩大M会持续提高性能”。

## 5. R9K1M2与R1K9M2

这是本次最严格的搜索结构实验：完整mini-1001、总父宽度9、输出9、M2、noop、bonus0.5、
100 steps、TF32、batch64、seed42全部相同。R1使用`initial_seed_groups=9`，把R9的9个
初始run流原样放进单一K9池。

| 配置 | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Oracle | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1K9M2 | 54.146 | 67.333 | 72.927 | 77.123 | 83.516 | 87.013 | 1396.58s |
| R9K1M2 | 57.043 | 71.528 | 78.122 | 82.617 | 86.114 | 91.808 | 3059.83s |
| R9-R1 (pp) | +2.897 | +4.195 | +5.195 | +5.494 | +2.598 | +4.795 | +119.1% |

R9-only/R1-only为Top-1 `71/42`（`p=0.00815`）、Top-2 `74/32`
（`p=5.5e-5`）、Top-3 `78/26`（`p=3.28e-7`）、Top-10 `55/29`
（`p=0.00604`）、Oracle `62/14`（`p=2.32e-8`）。R9的准确率优势不是随机波动。

效率与候选质量：

| 指标 | R1K9 | R9K1 |
|---|---:|---:|
| parent evaluations | 7,726,148 | 18,018,000 |
| child evaluations | 15,452,296 | 36,036,000 |
| valid/slots | 74.614% | 86.192% |
| mean true unique | 28.110 | 25.807 |
| final-slot shortfall | 27.421% | 0% |
| peak CUDA allocated | 1,749,637,632 B | 2,110,356,992 B |

R1为什么快：相同状态在全局池中合并后只继续保留一个逻辑状态，高分模式又会淘汰其它
模式，所以实际父/child评估减少约57%；wall降低54.4%，约2.19倍加速。

R1为什么不准：当前`log_mass + bonus`只衡量当前状态的经验频次/进度，不是“这个状态
最终能形成正确反应物”的value。全局竞争会让暂时低质量但未来有用的轨迹提前灭绝；rank2～8
invalid明显升高，并产生大量final-slot补齐。R1的unique更多却Oracle更低，再次证明表面
字符串多样性不是有效搜索多样性。

R9为什么更准：每个run的生存权受到隔离保护；一个run中的高频模式不能淘汰其它run，
因此低概率但正确的反应方向能继续走完100步，且所有9个输出槽都来自真实独立轨迹。
代价是相同/相近状态不能跨run共享计算，9条轨迹始终都要继续forward。

## 6. 哪些变量会改变性能

| 变量 | 主要作用 | 当前证据 | 当前选择 |
|---|---|---|---|
| `R` | 隔离与覆盖；成本近似随活跃run增加 | R9比R1/R3准但慢；R10边际饱和 | 准确率R9 |
| `K` | 池内竞争、合并和计算共享 | K9最快但丢Top-k；K5早期无收益 | 准确率K1，平衡K3 |
| `M` | 每父状态proposal数；不重复Transformer | M2有中段Top-k收益；M3/M4回退 | M2 |
| `changed_state_bonus` | 对抗no-event偏置 | 0.5局部平衡；1.0只换取少量Top-1 | 0.5，不视为跨M常数 |
| child policy | proposal语义 | noop在R3有收益，在K10无稳定收益 | 当前R9用单次noop anchor |
| score mode | 状态保守性/激进性 | legacy exploration覆盖高但invalid高且不校准 | full_probability |
| aggregation | 20个augmentation的最终排序 | Oracle与Top-k仍有5～6pp间隙；模式无普适赢家 | legacy_best_rank主报告 |
| seed | 随机轨迹与结果波动 | Beam已稳定到product/run；Euler CLI尚未接入 | Beam seed42；Euler待修 |
| steps/scheduler | 离散化和forward次数 | 100/cubic与训练匹配，但未做充分最优性扫描 | 100/cubic |
| batch size | GPU利用、padding和TF32数值路径 | 32≈64，128更慢且显存高 | 64 |
| matmul precision | 3090 Tensor Core吞吐 | TF32约快16～24%，主Top-k保持 | high/TF32 |

bonus不是独立于M的理论常数。每个parent质量先除以M，M变化会改变经验质量与no-event状态
的相对竞争；历史M3需要更高bonus才能压住原样状态，但bonus0.8仍不能恢复M2准确率。
因此不能用`bonus ≈ log(M)`机械自动缩放并期待性能保持。

## 7. 当前最佳配置与效率定位

最高准确率配置：

```text
R9K1M2
n_runs=9, n_branches=1, n_children=2
child_policy=stochastic_noop
changed_state_bonus=0.5
score_mode=full_probability
n_steps=100, cubic
batch_size=64, TF32 high, seed=42
legacy_best_rank aggregation
```

完整mini-1001为Top-1/3/10 `57.043/78.122/86.114%`，Oracle `91.808%`，wall
3059.83秒。按输入规模线性外推完整5007反应约4.25小时，但必须以正式实测为准。

速度—准确率三档：

| 档位 | 配置 | Top-1/3/10 | Oracle | mini wall |
|---|---|---:|---:|---:|
| 速度 | R1K9M2 | 54.146/72.927/83.516 | 87.013 | 1396.58s |
| 平衡 | R3K3M2 | 55.145/74.026/84.515 | 89.311 | 2071.54s |
| 准确率 | R9K1M2 | 57.043/78.122/86.114 | 91.808 | 3059.83s |

三档形成清晰Pareto前沿。不能只说“R9最好”而忽略它比R1慢2.19倍，也不能因R1更快就把
全局beam称为更优方法。

## 8. 下一步优先级

### P0：先把基线做正确

给普通Euler接入与product/sample、batch切分无关的稳定seed，补单元测试；随后只在未用于
本轮test-mini选择的validation区间做Euler N9与R9严格对照。没有这一步，项目不能在论文
中声称相对原始Euler的最终增益。

### P1：保留R9搜索隔离，但共享重复forward

R1证明“合并并删除lineage”会伤准确率，但不代表重复状态必须重复计算。checkpoint不使用
origin mask；同一product、同一step下token state相同的多个run具有相同模型输入。可以：

1. 只对唯一父状态做一次Transformer forward；
2. 把模型输出映射回所有原run/seed；
3. 每条lineage仍独立生成child并保留，不做跨run淘汰。

该方案把“计算去重”和“搜索合并”解耦，理论上可以保持R9逐行输出不变，同时回收早期和
合流状态的重复forward成本。先加每step唯一父状态率profile；只有存在足够重复才实现，
并以prediction SHA完全一致作为硬门槛。

### P2：诊断M2到底靠什么选择

先记录每step：两个child相同率、bonus决定率、seed tie-break决定率、选择no-op/changed
比例和被丢child的path probability。若大多数选择只是seed tie-break，说明当前M2收益来自
多抽一次样，而不是成熟的value排序；下一方法应研究validation上训练/校准的future-value、
lineage保护或ESS触发重采样，而不是继续增大M。

### P3：把Oracle转成Top-k

R9完整mini的Oracle与Top-10仍相差5.694pp，R10新增6个Oracle覆盖也没有提高Top-10。
在不碰test target调参的前提下，应在validation研究候选频次、augmentation support、
合法性和模型诊断分数组成的校准ranker；Q sharpening也应放在同一validation协议下作为
独立proposal变量。其优先级高于R11/R12或继续扫描bonus。

总体方向应从“继续堆更多run/branch/child”改为：**保留独立lineage的覆盖优势，消除它的
重复计算，再为child和最终候选增加真正有预测力的排序信号。**

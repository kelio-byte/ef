# PROJECT_FACTS_SUMMARY：Edit Flows / Euler-Beam / DGM 项目事实层总结

> 生成日期：2026-08-12。依据 `new_docs/` 下全部正文 Markdown（截至 2026-08-12）；`.ipynb_checkpoints/` 中的同名文件是旧快照，仅用于识别文档更新时间，不构成独立证据源。本文件按“研究问题与方法演进”重组，不逐文档摘要；只忠实重建文档中已出现的事实，不新增实验事实，不把推测写成验证结论。
>
> 状态图例：✅ = 已获实验支持；🟡 = 初步观察、未充分验证；❌ = 已失败或被放弃；🔄 = 被后续实验/审计修正或取代；⏳ = 已计划但未执行。

---

# 1. 项目目标与核心问题

## 1.1 任务定义

- 任务：单步逆合成，输入产物 SMILES，输出反应物 SMILES。
- 基础方法：Edit Flows——把 `product → reactants` 建模为离散、变长、连续时间编辑流（CTMC）。模型在每个时间点输出每个位置的 insert/substitute/delete rate 及 token 条件分布 `Q_ins/Q_sub`；采样器用 100 个 Euler 步逐步编辑序列。
- 耦合方向：`x0=product → x1=reactants`；训练使用预对齐 Z 空间（`<GAP>` token），模型只观察去 GAP 后的 `x_t`。
- 数据集：`datasets/USPTO_50K_PtoR_aug20_#global#`。train 800,060 行（含 20× augmentation）；val 100,020 行 = 5,001 个完整反应块；test 100,140 行 = 5,007 个完整反应块；tiny 1,000 行 = 50 个反应；mini-1001 20,020 行 = 1,001 个反应。
- 基础 checkpoint：旧 `checkpoint_step600000.pt`；新 `new_checkpoints/checkpoint_step600000.pt`（A6000 重训，2026-08-07）。
- 外部奖励模型（仅 DGM 线）：冻结 Molecular Transformer `new_checkpoints/MIT_mixed_augm_model_average_20.pt`（旧 OpenNMT 格式，约 4 层、hidden 256、FFN 2048、8 heads、共享词表 297）。

## 1.2 评估口径与指标

- 一个原始反应有 20 条 SMILES augmentation，统计单位只能是原始反应；把 augmentation 当独立样本是错误的（早期 `val200` 目录名歧义曾导致“200 行=10 个反应”的误用，已被 v2 协议修正）。
- 核心指标：Top-1～Top-10（聚合后正确反应物排名）、Oracle-any（覆盖率）、invalid rate、true unique、wall time、峰值显存；方法比较用 reaction-level paired bootstrap 与 exact McNemar。
- 默认聚合：`score_#global#.py` 的 `legacy_best_rank`（跨 augmentation 按最好局部位置 + reciprocal-rank 累积）；`rrf`/`frequency_first`/`hybrid` 为 opt-in 消融模式。
- 保留集规则（v2 协议）：`dev_unique1000_aug20`（可筛选）、`confirm_unique1000_aug20`（冻结后唯一确认）、`final_unique2000_aug20`（确认后最终验证）、完整 `src-test`（最后一次性评估）。除 dev 外均未使用。

## 1.3 贯穿项目的研究问题

1. 如何把计算预算用于“生成正确候选”（覆盖率）而不是只增加重复/低价值分支；
2. 已生成的正确候选如何进入 Top-k（排序），以及 R/K/M、child 选择、聚合方式各自贡献什么；
3. 是否存在比“模型自身 log-prob / 启发式 bonus”更接近逆合成正确性、又不泄漏评测目标的 reward，并能注入中间步骤（SMC / DGM / action-level guidance）；
4. 训练基础设施是否可信、可复现；是否需要显式 product conditioning 等模型级改动。

---

# 2. 方法演进主线

按“问题 → 尝试 → 结果 → 判断 → 下一步”组织。

## 2.1 基线 Euler 与可视化诊断（2026-07-28 前后）

- 问题：单条 Euler 轨迹高度随机，结果与 SMILES 排列、编辑时间、token 选择强相关。
- 尝试：first-step vs trajectory 可视化、20-aug × 10-run 稳定性实验。
- 结果：per-sample 命中率仅 26.3%（20,000 次采样），但 200 次机会下 9/10 产物 Top-1≈100%（产物 1143 例外，仅 ~70%）；编辑事件 46% 集中在 t>0.8；事件正确性只检查“位置+类型”不检查 token（可 4/4 “correct” 却 MISMATCH）；模型对纯 DELETE 最弱；不同 augmentation 命中率差异可超 10 倍（4837 的 CV=161%）。
- 判断：这些是诊断与假设来源，不是性能结论；“200 次机会几乎必中”说明了 per-sample 命中率与 Top-1 ACC 的差距。
- 下一步：修复可视化判定、扩展 time_grid、实现 Euler-Beam。

## 2.2 Euler-Beam：从“宽 beam 直觉”到受控结论（2026-07-31 ～ 08-08）

- 动机：同一初始状态可发散到不同结果（同源异果）、不同路径可汇聚到同一结果（殊途同归）、单轨迹可能死锁；由此提出 K 分支 + M 后继 + 状态合并 + 排序剪枝。
- 早期困难与修正：
  - 分支 seed 曾共用全局 RNG（任务 0 修复）；
  - 完整单路径概率排序偏向 no-event（Top-1 30%），旧 “triggered-only + reverse” 是激进启发式而非概率分数（🔄）；
  - M>1 后改用“子样本质量合并 + changed-state bonus”得到可比较原型；
  - 性能从 seed 修复后的 72 分钟回到 479 秒，再经批量化到 231 秒（任务 2.5/3/4）。
- 参数语义：R=隔离搜索池数、K=池内父状态上限、M=每父状态 child 数、输出数=R×K；M 不放大 Transformer forward。
- 关键实验结论（详见 §3.2）：
  - 全局大 K（R1K9/R1K10）更快但 Top-k/Oracle 更差；隔离谱系（R9K1）最好；
  - M=2 有中段排序收益，M=3/M=4 回退；
  - `stochastic_noop`（t≈0.9 单次 no-op anchor）在 R3/R9 上有收益，K10 上无稳定收益；
  - Q temperature <1 无稳定收益（默认 1.0）；
  - forward 共享为 opt-in 效率优化（TF32 下非逐字节一致）；
  - Euler-SMC 只完成 mechanics/bootstrap，无准确率证据。
- 当前默认（准确率）：R9K1M2、`stochastic_noop`、bonus=0.5、`full_probability`、100 steps、cubic、batch64、TF32 high、seed42、legacy_best_rank。R3K3M2=平衡档，R1K9/R1K10=速度档。

## 2.3 训练审计、基础设施修复与重训（2026-08-03 ～ 08-07）

- 审计结论：`bregman_loss()` 与论文 Eq.23 主体一致；checkpoint 数值有限、配置可 strict load；预对齐数据完整，无证据表明 fallback/PAD/zip 缺陷污染旧 checkpoint。
- 发现并修复（新训练路径 `configs/retro_v2.yaml` + `train_retro.py`）：
  - Noam 第一次 update 顺序错误（旧脚本先 `optimizer.step()` 再用 Adam 默认 lr=1e-3，之后才进 Noam；修复为每步先设置 Noam lr 再 update）；
  - 无统一 seed/RNG/checkpoint resume 语义；fallback alignment 把 PAD 当真 token；`zip()` 静默截断；无 validation/最佳 checkpoint。
- 重训：A6000 上新 600k checkpoint，模型结构/loss/数据不变，实际改变的是 Noam 顺序和确定性数据顺序/RNG。
- 对比结论：新旧 checkpoint 在同协议 mini-1001 上几乎持平（旧 57.0/78.1/86.1/91.8%，新 58.2/77.9/86.4/91.5%）；修复没有带来模型级跃升，但提供了可复现基线。
- 模型级遗留限制：没有不可编辑的显式 product 条件（copy-product）；目标编辑极不均衡（insert 91.20%、substitute 7.88%、delete 0.92%）。

## 2.4 从启发式扩 beam 转向 reward/guidance（2026-08-05 之后）

- 问题：R/K/M 只能扩大采样与启发式排序，无法判断中间状态是否通向正确反应物；Oracle 与 Top-k 的差距不能靠大池消除。
- 理论借鉴：Inference-Time Scaling 论文（SMC：proposal/target/importance weight/ESS）、Discrete Guidance Matching（ICLR 2026）、Edit Flows 原论文（Q sharpening/CFG/reverse/localized）、RetroAgent。
- 已落地：Euler-SMC mechanics（`euler_smc.py`，11 项测试，bootstrap target=proposal 通过，无 reward 因此无准确率结论）；DGM action-level guidance 完整管线（reward → guidance 数据 → 训练 → ordinary Euler 推理 → 开发集评估）。
- 边界：由于变长编辑 + 连续 GAP 插入不可唯一反演（89.735%），当前实现是 **action-level approximate DGM**，不能称 exact DGM。
- 结果：所有 learned guidance 候选（E3–E7）与局部 credit 方案（L1/E2）均未通过开发集门槛；endpoint ranker v2/v3、reward 校准 P1/P2 正式关闭；确认集/最终验证集/完整 test 未使用。

## 2.5 当前状态总览

- 推理主线：R9K1M2 Euler-Beam 是当前最成熟的采样配置；完整 test 尚未作为最终评估运行。
- DGM 主线：管线可运行、reward 有弱判别力，但没有任何候选通过“Top-1 不降 + 深层改善”门槛；默认采样关闭 guidance。
- 训练主线：基础设施修复完成、新 checkpoint 可用；显式 product conditioning 是已记录的下一训练方向，未执行。
- 保留集：confirm/final/test 未被读取。

---

# 3. 关键实验与结果

## 3.1 基础模型与训练

### 3.1.1 历史 checkpoint 配置（checkpoint_step600000.pt）

| 项目 | 值 |
|---|---|
| step / total | 600,000 |
| 真实词表 / 模型词表 | 69 / 73（+4 特殊 token） |
| hidden / layers / heads / FFN / max_len | 256 / 10 / 8 / 2048 / 256 |
| dropout / attention dropout | 0.3 / 0.3 |
| batch / optimizer | 128 / Adam β=(0.9,0.998) ε=1e-8 |
| lr schedule | Noam factor=1.0 warmup=8000（旧脚本首步误用 1e-3） |
| scheduler / time input / rate reparam / origin mask | cubic κ=t³ / raw t / off / off（无 origin_embedding） |
| 参数量 | 13,537,173 |

### 3.1.2 训练数据编辑分布（800,060 对预对齐序列）

- insert 4,188,917（91.201%）、substitute 361,694（7.875%）、delete 42,451（0.924%）；平均每对 5.741 个编辑；0-edit pair 为 0；≥10 编辑占 15.95%，≥20 占 3.88%。
- 含义：delete head 监督极稀疏；推理 invalid 常与过插入/错误插入有关；文档明确不建议未经实验直接加 class weight（会改变 CTMC 目标 rate）。

### 3.1.3 新旧 checkpoint 对比（同协议，R9K1M2）

tiny（50 反应）：

| checkpoint | Top-1..10 | Oracle | rank-1 invalid | wall |
|---|---:|---:|---:|---:|
| 旧 | 60/70/80/84/84/84/84/84/84/84 | 98% | 22.3% | 114.72 s |
| 新 | 58/74/78/80/84/86/86/88/90/90 | 92% | 22.3% | 113.49 s |

mini-1001（1,001 反应，20020×9=180,180 行）：

| checkpoint | Top-1 | Top-3 | Top-10 | Oracle | rank-1 invalid | wall |
|---|---:|---:|---:|---:|---:|---:|
| 旧 | 57.043 | 78.122 | 86.114 | 91.808 | 13.666% | 2046.70 s |
| 新 | 58.242 | 77.922 | 86.414 | 91.508 | 12.767% | 2029.18 s |

判断：差异在 ±1.2pp 以内，新模型覆盖目标平均排名更靠前（2.783 vs 3.047）、invalid 略低；不能宣称显著提升或退化。✅

新 checkpoint validation-200 消融（固定 9 输出预算；用于冻结 mini 配置）：

| 配置 | Top-1 | Top-3 | Top-10 | Oracle | rank-1 invalid | wall |
|---|---:|---:|---:|---:|---:|---:|
| R9K1M2 noop T=1.0 bonus=.5（冻结） | 70.0 | 90.5 | 93.5 | 95.5 | 12.10% | 411.64 s |
| R3K3M2 noop T=1.0 bonus=.5 | 62.0 | 83.5 | 89.0 | 94.0 | 5.675% | 335.57 s |
| R9K1M1 stochastic T=1.0 bonus=.5 | 69.5 | 88.5 | 92.5 | 96.0 | 11.325% | 319.30 s |
| R9K1M2 noop T=0.9 bonus=.5 | 69.5 | 91.5 | 94.0 | 95.5 | 11.75% | 410.08 s |
| R1K9M2 stochastic T=1.0 bonus=.5 | 61.0 | 80.5 | 87.0 | 90.0 | 2.575% | 288.18 s |

注意：bonus=0.5 与 0.8 的预测 SHA 完全一致（本轮无排序作用）；R/K 组织影响最大；M=2 收益主要在 Top-3～10 排序而非 Oracle。

## 3.2 Euler-Beam 采样

### 3.2.1 早期基准与口径修正

- 早期 `euler_beam_report.md` 声称“所有 Top-k 一致优于 Euler”（tiny：58 vs 54 Top-1 等）；该表述被后续严格实验推翻（🔄），只能作为研究线索。
- 修复完整路径概率后，固定基准一度为 479.062 s、Top-1/2/3=30/54/60%（原样率 26.1%），vs 普通 Euler 81.144 s、56/68/74%——确认完整单轨迹概率偏向少编辑路径。
- 旧 58/68/76% 恢复版本使用不同 seed/评分语义，不能作为当前实现的严格对照。
- 评分输入曾有文件混用风险（`bench_euler` 实际为 54/66/74，旧文档记录 56/68/74），已通过 metadata/输入校验解决。

### 3.2.2 R/K/M 受控实验（mini-1001，旧 checkpoint；输出宽度见各配置）

| 配置 | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Oracle | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1K10M2（stochastic） | 52.547 | 67.033 | 71.628 | 76.523 | 81.618 | 87.213 | 1687.14 s |
| R1K10M3（stochastic） | 51.249 | 63.636 | 68.831 | 74.326 | 79.421 | 83.916 | 2065.51 s |
| R3K3M2（noop） | 55.145 | 68.332 | 74.026 | 78.821 | 84.515 | 89.311 | 2071.54 s |
| R9K1M2（noop） | 57.043 | 71.528 | 78.122 | 82.617 | 86.114 | 91.808 | 3059.83 s |
| R1K9M2（noop，9 初始流复用 R9） | 54.146 | 67.333 | 72.927 | 77.123 | 83.516 | 87.013 | 1396.58 s |
| R10K1M1（stochastic bonus0） | 56.344 | 70.529 | 76.224 | 80.120 | 84.915 | 92.308 | 3022.81 s |
| R10K1M2（noop bonus0.5） | 56.943 | 71.828 | 78.422 | 82.418 | 86.014 | 92.408 | 3425.59 s |

关键配对统计：
- R9 vs R3（预注册替代规则）：Top-1 +1.9（p=0.073）、Top-2 +3.2（p=0.0016）、Top-3 +4.1（p=3.8e-5）、Top-5 +3.8（p=0.0001）、Top-10 +1.6（p=0.081）、Oracle +2.5（p=0.0010）→ R9 胜出并冻结为准确率默认。
- R9 vs R1（同 9 条初始流）：R9 在 Top-1/2/3/10/Oracle 全面显著更好；R1 wall 降 54.4%（父评估 7.73M vs 18.02M），但 valid/slots 74.6% vs 86.2%、shortfall 27.4% vs 0% → 全局竞争会淘汰未来有用谱系。
- R10M2 vs R9M2：第 10 个 run 新增 6 个 Oracle（p=0.031）但 Top-1/10 各少 1 命中，wall +11.95% → 不升级，关闭 R 搜索。
- R10M2 vs R10M1：M2 的 Top-3/5 显著改善（p=0.007/0.004），Oracle 仅 +0.1 → M2 收益是“排序集中化”而非覆盖。
- M3 vs M2（R1K10 纯 stochastic）：Top-2～10 与 Oracle 全面回退、wall +22.4%，尽管 true unique 更高 → “更多 child”不是普适改进。

### 3.2.3 与普通 Euler 的公平对照（test-mini 前 200 反应）

| 方法 | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| Euler N9（同进程 seed42） | 54.0 | 69.5 | 77.5 | 82.5 | 88.0 | 94.5 |
| R9K1M2 | 58.0 | 74.0 | 81.0 | 86.0 | 88.0 | 96.0 |

逐反应 McNemar 均未达显著（Top-1 p=0.096、Top-2 p=0.078、Top-3 p=0.210）；Top-10 与 Oracle 增益极小 → 只能说“有把正确候选前移的正向证据”，不能说已严格全面优于 Euler。限制：当时普通 Euler 的 CLI seed 尚未接入（用外部同进程 seed），不能作为最终可复现基线；后已修复。

### 3.2.4 validation-200 上的 R3K3 vs R1K9（最公平的搜索结构实验）

| seed | 配置 | Top-1 | Top-2 | Top-3 | Top-10 | Oracle | wall |
|---|---|---:|---:|---:|---:|---:|---:|
| 42 | R3K3 | 64.5 | 80.5 | 85.0 | 90.5 | 94.5 | 482.18 s |
| 42 | R1K9 | 62.5 | 75.5 | 79.5 | 89.0 | 92.5 | 324.39 s |
| 43 | R3K3 | 61.5 | 78.0 | 82.5 | 90.0 | 92.5 | 484.47 s |
| 43 | R1K9 | 58.0 | 74.0 | 82.0 | 87.0 | 89.5 | 324.01 s |

两 seed 均复现：R1 快 ~32–33%、rank1 更干净，但 rank2/3 invalid 明显升高、Top-2/Oracle/尾部更差 → R3K3 是准确率模式，R1K9 是低延迟模式，形成 Pareto 而非一方支配。

### 3.2.5 性能工程结论

- TF32（`matmul_precision=high`，RTX 3090）：完整 tiny 配对中约快 24%，2979/3000 行与 FP32 一致，Top-1/2/3 相同 → 日常实验推荐，最终严格复现用 `highest`；CLI 默认仍为 `highest`。
- batch64：R3K3 完整 tiny 上 32/64/128=109.78/110.12/117.13 s；batch128 更慢且改变部分行 → 默认 64。
- 相同状态 forward 共享（opt-in）：tiny R9K1M2 wall 154.0→114.1 s（-25.95%），TF32 下 2/9000 行漂移，FP32 下逐字节一致 → 作为 opt-in 提交，不静默改默认。
- 已否决：torch.compile（首编译过慢）、BF16（模型 `_log_softplus` dtype 不兼容，需改共享模型）、padding 分桶/外层长度排序/efficient attention 开关（短测均未满足 10% 加速或逐行一致门槛）。
- 普通 Euler 性能复核：旧入口与当前入口同机同输入仅差 2.0%（189.59 vs 185.77 s）；历史“30–40 分钟跑完整 test”无法用当前代码复现（tiny 线性外推约 5 小时量级），原因未唯一还原，不能据此认定旧指标被刻意压低（🟡）。

### 3.2.6 Q sharpening（temperature）与 Euler-SMC

Q temperature（只改变 token 分布、不改变 rate）validation 结果：

| split | T=1.0 | T=0.9 | T=0.8 |
|---|---:|---:|---:|
| validation-A(50) Top-1/3/10/Oracle | 74/92/96/98 | 72/94/96/98 | 74/94/96/98 |
| validation-B(50) Top-1/3/10/Oracle | 62/84/96/98 | 62/86/96/98 | 60/86/98/98 |
| validation-C(200) Top-1/3/10/Oracle | 50.0/74.5/86.0/91.5 | 51.0/74.0/86.0/91.0 | — |

T=0.9 的 Top-3 收益在 A/B 出现但 C200 不复现；T<1 普遍降低 true unique；tiny post-hoc 也不支持 → 默认保持 T=1.0（❌ 不作为改进）。

Euler-SMC：`euler_smc.py` 实现 log-weight、ESS、systematic resampling、ancestor 传播；11 项 mechanics 测试通过；真实 checkpoint bootstrap（target=proposal）ESS=N、无重采样、evidence=0；terminal twisting 只有数学/小样本 smoke。尚无独立 reward，因此没有任何 Top-k 收益声明。✅ 机制正确 / ⏳ 准确率未验证。

## 3.3 DGM / action-level guidance

### 3.3.1 Reward 及其质量

- 最终采用：Molecular Transformer 正向 beam=5 重构，候选被重建为输入 product 的第 k 名 → reward=1/k，未命中=0；候选先 canonicalize + 缓存（同批 8,131 个原始串 → 2,332 个唯一结构，提速 3.03×）。
- 旧 teacher-forced likelihood reward 判别力弱：正确性 AUC 0.5639，raw rerank 显著损害 Top-1 → 弃用。
- forward-beam reward 判别力（validation-B 12,000 条 Euler 候选）：correctness AUC 0.6904；正确候选重构命中 81.86%，错误候选仍有 50.56% 被重构 → “有方向但不完美”。
- 正向模型自检：validation-B 200 反应 Hit@1/3/5=71/77/79%，MRR 0.7414；dev-1000 上 Hit@1/3/5=68.0/73.4/74.7%，MRR 0.7073。
- 多时间点数据审计（train-1000/val-200）：全局 correctness AUC 0.6798/0.6971，同 anchor 组内 AUC 0.6965/0.7308；held-out 中 46.05% 错误终点仍得正 reward，42.40% 的组同时含正确/错误终点。

### 3.3.2 Guidance 数据与训练

- 数据演化：单终点（无 product 内 reward 对比，无效）→ 每 product 4 终点/1 时间（46.47% 组有 reward 变化）→ shared-anchor continuation（先公共 `x_t,t` 再独立续采样；修正 adaptive endpoint 时间后所有组 state/time 完全一致）→ 五时间点（step 10/30/50/70/90，每 anchor 4 条 continuation；train 20,000 records/5,000 组，val 4,000/1,000 组）。
- 五时间点各步“4 条后续至少两种不同 reward”的组比例：train 45.3/42.3/40.3/35.3/23.1%，val 47.5/44.0/41.0/38.5/22.0%（越接近终点越少变化，step 90 仍保留）。
- Guidance 模型：`ProductConditionedGuidance` 约 5.26M 参数（product encoder 2 层 + state encoder 4 层、hidden 256、8 heads、FFN 1024、dropout 0.1、三正输出头 H_ins/H_sub/H_del）。
- 训练：Bregman action target（selected action 目标≈reward+1e-4，背景 1e-4），AdamW lr 1e-4、batch 64；后续加入 shared-anchor pairwise（λ=0.25）与 score-calibration（weight 0.10）候选，checkpoint 按“Bregman ≤1.15× control + 最高 pair accuracy”选择。
- 推理：`apply_action_guidance()` 把 `u=λ·Q·H^β` 在每个位置内部重排，保持每位置总编辑率；β=0/常数 H 与 baseline 逐字节一致；默认 `per_position`（`per_sample` 在 validation-A 上更差）。
- 效率代价：每个 Euler step 多一次 guidance forward，端到端约慢 47–49%（如 dev-1000 1,270.9→1,884.9 s）。

### 3.3.3 历史 validation-A/B（已降级为探索记录，🔄）

| split / 方法 | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Oracle | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| A baseline | 51.0 | 66.5 | 72.0 | 77.0 | 83.5 | 86.5 | 253.0 s |
| A 10k guidance β=0.10 | 53.0 | 66.5 | 71.5 | 78.5 | 83.5 | 85.5 | 371 s |
| B baseline | 58.5 | 73.0 | 77.5 | 85.5 | 88.5 | 91.0 | 264.4 s |
| B 10k guidance β=0.10 | 56.5 | 75.0 | 80.0 | 86.0 | 88.5 | 91.5 | 388 s |

结论：方向不一致（A Top-1 +2/Oracle -1；B Top-1 -2/Top-2/3/Oracle 升），未通过综合门槛；两个 200-reaction 区间被多次用于选参，此后不再承担方法结论。

终端重排（validation-B，固定候选池，非逐步 guidance）：

| 排序 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle |
|---|---:|---:|---:|---:|---:|
| 原始 Euler 顺序 | 58.5 | 77.5 | 85.5 | 88.5 | 91.0 |
| canonical forward-beam rerank | 59.0 | 78.5 | 85.0 | 89.5 | 91.0 |

✅ 说明 reward 有候选级排序价值；但这是终点重排，不等于 learned guidance 有效。

### 3.3.4 Shared-anchor pairwise 系列（P5 → P5f，均未通过）

- P5（数据未真正共享 anchor 时的 pilot）：lambda=0.25/1.0 的 pair accuracy 58.63%/57.93%，均低于 control 59.73% → 不通过；P5b 审计发现旧数据 state/time 并不共享（val-200 仅 23/200 组 state 全同、0 组 time 全同），指标无效（🔄）。
- P5c 实现真实 continuation；P5d（修正数据）：control pair acc 55.15%，λ=0.25 59.66%（+4.51pp）但 Pearson 0.1135→-0.0118 → “排序改善、校准失败”，联合 gate 未通过。
- P5e：组内 Pearson 诊断确认校准退化真实（control 0.1490，λ=0.25 0.0667）。
- P5f（λ=0.25 + score_calibration_weight=0.10）：pair acc 56.65%、组内 Pearson 0.1165（仍未达 control 0.1490）→ 不通过；不进入 10k/采样 A/B。

### 3.3.5 反应级开发集评估 v2（E1–E7；dev_unique1000_aug20）

统一协议：新 600k checkpoint、普通 Euler、100 steps、每 augmentation 3 候选、batch64、seed42、β=0.10、per_position、20 augmentation 聚合为 1,000 反应。

| 方法 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle | invalid | 采样 wall | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| E1 普通 Euler 基线 | 58.2 | 75.5 | 79.8 | 83.5 | 86.6 | 11.805% | 1,270.9 s | 参照 |
| E3 终点 forward rerank | 57.6 | 76.7 | 81.3 | 85.0 | 86.6 | 11.805% | 复用 E1 | Top-1 未过门槛（-0.6pp，CI [-1.6,+0.4]） |
| E4 历史大规模 guidance | 55.8 | 75.7 | 80.0 | 83.1 | 86.2 | 11.918% | 1,871.4 s | Top-1 -2.4pp（CI [-4.2,-0.6]） |
| E5 共享 anchor guidance | 56.2 | 76.1 | 80.9 | 84.9 | 87.5 | 11.718% | 1,865.9 s | Top-1 -2.0pp（CI [-3.9,-0.2]），Top-10 +1.4pp（CI [+0.1,+2.7]） |
| E6 五时间点+排序/校准（500-step） | 56.4 | 76.3 | 80.0 | 84.2 | 87.7 | 11.803% | 1,887.6 s | Top-1 -1.8pp（CI [-3.7,0.0]） |
| E7 E6 的 2,000-step 长训练 | 56.7 | 75.9 | 80.5 | 83.8 | 86.6 | 11.983% | 1,884.9 s | Top-1 -1.5pp（CI [-3.3,+0.3]）；深层 CI 均跨 0 |

训练级对照：E7 候选 held-out pair accuracy 63.75%（control 59.07%）、Bregman 0.5573 在 guard 内 → 更长训练确实提升离线排序，但未转化为 Top-1。

**结论：E3–E7 全部不进入 confirm；confirm/final/test 未使用。**

### 3.3.6 Reward 校准支线 P1/P2（正式关闭）

隔离 200-reaction holdout（train 反应 1,000–1,199）上，同候选池比较：

| 排序 | 全局 AUC | 同组 AUC | Top-1 | Top-3 | Top-10/Oracle |
|---|---:|---:|---:|---:|---:|
| raw forward | 0.7073 | 0.6836 | 41.5 | 76.5 | 83.0/83.0 |
| P1 线性校准 | 0.7655 | 0.6843 | 40.0 | 73.0 | 83.0/83.0 |
| P2 +teacher-forced likelihood | 0.7598 | 0.7093 | 41.0 | 73.0 | 83.0/83.0 |

P2 首次让同组 AUC 点估计超 +0.02，但 Top-1/Top-3 仍下降 → 两支线均按 gate 拒绝；不再重训 guidance、不加特征、不重跑 dev。

### 3.3.7 Endpoint ranker v2 / v3（正式关闭）

重要勘误：旧 correctness-reward 脚本曾把真实 target 的 canonical component count 当作 product 特征（target leakage），旧 P2/P3 的 AUC 与 rerank 数字不再干净（🔄）；修复提交 `532a46e`，v2 起只从序列化 product/candidate tokens 构造特征。

v2（train/val/holdout = 1000/200/200，2000–2999/3000–3199/3200–3399）holdout：

| 排序 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle |
|---|---:|---:|---:|---:|---:|
| raw forward | 46.5 | 72.5 | 77.0 | 78.5 | 78.5 |
| bounded residual（raw+0.25·tanh） | 40.0 | 71.5 | 78.0 | 78.5 | 78.5 |
| listwise/hard-negative | 38.0 | 70.5 | 76.0 | 78.0 | 78.5 |

paired CI：residual Top-1 [-13.0, 0.0] pp；listwise [-15.5, -1.5] pp。

v3（8000/1000/1000；10000–17999/18000–18999/19000–19999）holdout：

| 排序 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle |
|---|---:|---:|---:|---:|---:|
| raw forward | 47.7 | 74.4 | 79.2 | 80.4 | 80.4 |
| residual | 38.1 | 72.6 | 79.5 | 80.3 | 80.4 |
| listwise | 37.0 | 72.0 | 79.3 | 80.4 | 80.4 |

residual 全局 AUC 0.6922→0.7167（+2.45pp）、同组 AUC 0.6493→0.6894，但 Top-1 CI [-13.0,-6.3] pp；listwise CI [-13.9,-7.5] pp。Oracle 均不变 → “candidate AUC 提高、reaction Top-1 明确下降”的模式在大样本下稳定；当前小型 endpoint reranker 路线正式关闭（不构造 guidance data、不重训 DGM、不跑改进后可视化）。

### 3.3.8 局部 credit assignment（L0/L1/E0–E2）

- L0：记录真实一步后继不改变采样；通过。
- L1（自然一步 Euler）：1,000 条 record 中仅 51 条（5.1%）发生编辑；严格“两种不同 action 且 reward 有差异”组仅 1/250（0.4%）< 20% 门槛 → 数据不足，L2/L3 禁止。
- E0/E1（event-conditioned 原子 action proposal）：条件于发生一个有效编辑抽取单原子动作；E1 中严格可区分组 98/250=39.2%（按 step 48/44/44/36/24%）→ 数据门槛通过。
- E2（正式 2,000-step 对照，train-1000/val-200）：transition candidate 相对 terminal control 的共同 transition-target Bregman 仅 -0.198%（门槛 -2%），pair accuracy -1.639pp，Pearson -0.0069 → 未通过；不运行 E3。
- 用户要求的 post-gate dev-1000 诊断（不改变 E2 结论）：transition guidance Top-1 55.7% vs 基线 58.2%（CI [-4.5,-0.6]），Top-2/3/5/10 全部微降，Oracle 86.2% vs 86.6%，wall +49.1% → 负向，不进入 confirm。

### 3.3.9 P0 冻结协议与 P1 机制面板

- P0：普通 Euler dev-1000 基线复跑成功（SHA 与 E1 一致到报告精度；Top-1/3/5/10=58.2/75.5/79.8/83.5%，Oracle 86.6%，invalid@rank1/2/3=11.875/11.425/12.115%）。
- P1 面板（96 条 paired path，Euler vs E7）：有效终点 82/96 vs 83/96；canonical target 匹配 23/96 vs 24/96；insert/sub/delete 490/53/8 vs 490/51/7；首次分叉仅 14/96（中位 step 74.5）；层分解 72 条两者都错、23 条两者都对、1 条 guidance 单独对、0 条 Euler 单独对 → 没有稳定“减少 invalid/修复同类错误”的机制证据（🟡，机制诊断非性能结论）。

## 3.4 Z-space 形式化边界

- 对 validation 100,020 条预对齐行：573,927 个坐标变化 = insert 524,838 / substitute 43,949 / delete 5,140；其中 89.735% 的 insert 位于连续 GAP run，静态对齐下无唯一 X-space 逆映射；仅 7.674% 的整行只改变一个坐标。
- 结论：当前变长 Edit Flow 不能宣称 exact DGM；固定坐标 toy 证明 DGM 代数本身正确，阻塞来自坐标映射。✅ 形式化边界 / ⏳ 完整 exact 方案未实现。

---

# 4. 当前已经确认的结论

以下结论在文档中具有实验支持（注意：除特别说明外均为 dev/validation/mini 级，不是完整 test 最终结论）。

1. 训练基础可信：历史 checkpoint 的 loss/对齐/数据完整；Bregman loss 与论文 Eq.23 主体一致；已发现并修复 Noam 首步、RNG/resume、数据 fail-fast、validation 缺失等基础设施问题（✅）。
2. 新旧 600k checkpoint 性能几乎持平；新 checkpoint 可作为后续推理研究的可靠基线，但没有性能跃升证据（✅）。
3. 模型级限制明确：无显式不可变 product 条件；目标编辑极不均衡（insert 91.2% / delete 0.92%）（✅）。
4. Euler-Beam 中“保护独立谱系（R9）+ M2 局部选择”在当前质量函数下有受控证据：R9K1M2 优于 R3K3 与 R1K9/R1K10（mini-1001）；R9 相对普通 Euler 只有排序前移的正向方向证据，200 反应上未达统计显著（✅/🟡）。
5. M=2 的收益主要是“局部候选排序集中”，不是覆盖扩大；M3/M4 回退；K 扩大无收益（✅）。
6. 全局大 K 池更快但会竞争性灭绝有用谱系（R1K9 两 seed 复现）；R3K3=平衡档、R1K9=速度档是 Pareto 关系（✅）。
7. Q temperature <1、antithetic child、forced-edit、多次 no-op、最终 path 权重、内部多分支返回等均无稳定收益（✅ 已消融）。
8. TF32 在 3090 上是日常实验推荐（约快 16–24%，Top-k 保持一致）；batch64 是复现默认；相同状态 forward 共享是 opt-in 提速（✅）。
9. forward-beam=5 重构 reward 有弱但可测的候选级判别力（AUC 0.69–0.71、同组 AUC 0.68–0.73），但约 46–50% 错误终点仍得正 reward，不是正确性真值（✅）。
10. action-level guidance 的实现/数据/恒等性/评估框架完整；但 E3–E7、P1/P2、endpoint ranker v2/v3、L1/E2 均未通过采用门槛；默认采样关闭 guidance（✅ 负结论）。
11. guidance 离线排序改善（pair accuracy 63.75%）≠ 最终 Top-1 改善（dev Top-1 56.7% < 58.2%）——这是当前最核心的已验证负结论（✅）。
12. 当前 rerank/guidance 失败的模式稳定：candidate-level AUC 提高与 reaction-level Top-1 下降可同时发生（✅）。
13. Euler-SMC 机制正确（target=proposal 无偏），但没有任何独立 reward 下的准确率证据（✅ 边界）。
14. 严格评估协议 v2 已建立：reaction 为统计单位、dev/confirm/final 分层、配对 bootstrap；confirm/final/test 未使用（✅）。

---

# 5. 已失败、废弃或被后续取代的方案

| 方案/结论 | 状态与证据 | 取代者/说明 |
|---|---|---|
| “Euler-Beam 在 tiny 上全面优于 Euler” | ❌🔄 早期报告，被严格对照推翻 | 当前表述：排序前移的正向证据，无全面优越声明 |
| 完整单路径概率作为排序主键 | ❌ 偏向 no-event，Top-1 30% | `log_mass` 合并 + changed-state bonus |
| legacy triggered-only + reverse 评分 | ❌ 是激进启发式，invalid 高且不校准 | 仅保留为消融模式 |
| M=3/M=4、K=5 扩大 | ❌ 准确率回退或无增益 | M=2、K=1/3 由实验选择 |
| antithetic child | ❌ invalid 降但 Top-k 全降 | 独立 stochastic child |
| forced-edit / 多次 no-op / no-op mass 0.75 | ❌ | 固定 t≈0.9 单次 no-op |
| Q temperature <1 | ❌ 无稳定收益，默认 1.0 | — |
| R1K9/R1K10 作为准确率默认 | ❌（作为准确率）/✅（作为速度） | R9K1M2 准确率默认 |
| R10K1M1/M2 升级 | ❌ 覆盖增量不支付成本 | R9K1M2 保持 |
| 旧 correctness-reward P2/P3（v1） | ❌🔄 target leakage，数字失效 | `532a46e` 修复后 v2/v3 重新评估 |
| endpoint reranker（residual/listwise）v2/v3 | ❌ Top-1 明确下降且大样本稳定 | 路线关闭 |
| reward 校准 P1/P2（线性 + likelihood） | ❌ 全局 AUC 升但同池 Top-1/Top-3 降 | 支线关闭 |
| learned guidance（10k / shared-anchor / 五时间点 / 2,000-step） | ❌ dev Top-1 均低于基线 | 冻结为历史对照 |
| per-sample rate normalization | ❌ validation-A 上被 per-position 支配 | per_position 默认 |
| 自然单 Euler 步 transition credit（L1） | ❌ 数据稀疏（0.4% 可区分组） | event-conditioned proposal（E 线） |
| event-conditioned transition target（E2） | ❌ 共同 Bregman/排序门槛未过 | 支线关闭 |
| torch.compile / BF16 / padding 分桶等性能尝试 | ❌ | TF32 + forward 共享（opt-in） |
| 早期“validation-A/B 各 200 反应可作方法结论” | 🔄 已被多次选参、统计不稳 | v2 dev/confirm/final 协议 |
| “score.py 1/(position+1) 权重”适用独立采样 | 🔄 默认保留 legacy_best_rank；frequency-first 仅在高互补候选池有效 | 已实现 opt-in 聚合模式 |
| 早期“Euler 完整 test 30–40 分钟”的说法 | 🟡 当前代码无法复现，原因未还原 | 以 metadata 实测为准 |
| 旧 5 种可视化/路径事件判定（不查 token、time_grid 截断） | 🔄 已修复/扩展 | 修复后诊断工具 |

---

# 6. 尚未解决的问题

1. **Reward 与真实逆合成正确性的失配**：forward-beam reward 有方向但 46–50% 错误终点得正分；没有找到同时满足“AUC 提升”与“reaction Top-1 不降”的 reward/校准（P1/P2、v2/v3 ranker 均失败）。
2. **Terminal-to-action credit assignment**：把终点 reward 沿对齐路径分配给动作是粗粒度近似；自然一步数据稀疏（L1），event-conditioned 原子编辑的局部监督也未通过（E2）；尚未有可信的中间动作价值。
3. **固定位置总强度的 guidance 表达受限**：当前 `per_position` 只重排位置内动作/token，不能跨位置转移编辑强度；`per_sample` 更差；严格 Z-space/exact DGM 被非双射插入阻塞（89.735% ambiguous）。
4. **显式 product conditioning**：当前 copy-product 条件较弱，中间状态碰撞会平均不同 product 的目标；文档认为这是最有价值的下一训练分支，但未执行。
5. **编辑不均衡**：delete 仅 0.92% 目标编辑；如何在不破坏 CTMC 目标 rate 的前提下处理未解决。
6. **R9K1M2 相对普通 Euler 是否真正全面更优**：200-reaction 方向证据不显著；需要更大、冻结的对照（Euler CLI seed 已修复，可做严格复现对照）。
7. **推理效率**：model forward 占 ~74% 时间；在不改变预测的前提下低风险收益已接近上限（理论 ~1.34×）；n_steps=50/100/200 消融未做。
8. **Euler-SMC 的独立 reward 验证**：没有独立化学 reward 前不能声称 SMC 提升准确率。
9. **完整 test 最终评估**：confirm/final/src-test 均未运行（受 gate 保护）。
10. **环境/复现细节冲突**：执行报告记录 conda `ef`/PyTorch 2.7.1+cu126/RTX 3090，dgm.md 依赖清单记录 torch 2.13.0+cu130；两者未在文档中消解（🟡 需以实际环境为准）。

---

# 7. 当前已有的后续任务与研究计划

以下均为文档中已记录、尚未完成或被 gate 阻断的计划；未列入文档的新研究方向不在此列出。

## 7.1 DGM 主线的 gate 化顺序（dgm_future_plan / dgm_local_credit_assignment_plan）

- P0：冻结普通 Euler 基线 + 协议 —— ✅ 已完成。
- P1：Euler vs guidance 成对路径诊断（`P1-panel-v1`，96 path）—— ✅ 已完成（机制诊断，非调参）。
- P2：构造并训练“逆合成正确性 reward”（正例=target 匹配，负例=Euler 有效候选，invalid 最低分；反应级隔离；holdout 一次性 gate）—— ✅ 已实现并执行 v2（但 P3 失败，故 P2 的通过仅限 AUC 点估计，不能进入 DGM）。
- P3：新 reward 的独立终点 rerank gate —— ❌ 失败（correctness reward Top-1 42.0% < raw 48.5%）。
- P4：用通过 P3 的 reward 重训 DGM → dev-1000 → 复用 P1 面板复核 → confirm → final → 完整 test —— ⏳ 未启动（P3 阻塞）。
- 局部 credit 支线：L0 ✅ → L1 ❌ → E0/E1 ✅（event-conditioned 数据门槛）→ E2 ❌ → E3 ⏳ 未启动（E2 阻塞）。
- 严格 Z-space / exact DGM：需要保留 GAP 身份、固定坐标映射、guidance posterior 推导 —— ⏳ 未实现；DG-0/DG-1 已审计接口。

## 7.2 推理与采样侧计划

- 特殊 token/BOS 完整硬约束：位置 0 屏蔽已在 guidance 数据阶段实现并统一到 Euler/Euler-Beam；文档仍将“BOS/特殊 token 动作约束未完全硬编码”列为当前限制，后续应先诊断再决定是否完整硬屏蔽（⏳）。
- n_steps 消融（50/100/200）在 DGM 阶段 5 通过后才做 —— ⏳ 未执行。
- GPU 状态 key/合并、child 维广播等性能候选（预期 4–8%）—— ⏳ 未实现（需先保证逐行一致）。
- 混合候选池 NNN+LL + frequency-first：holdout-200 上 Top-3 净增 11/200（p=0.0074）、Oracle 91%，但 Top-1/2 不显著、LL 为未校准启发式 → 保留为高覆盖实验模式，未设为默认；文档未列正式后续任务，仅在任务 13 中冻结为参考档（🟡）。

## 7.3 训练侧计划

- 显式 product conditioning（独立 encoder/cross-attention 或受保护 product memory）+ condition dropout 后再做 CFG —— 已审计为最有价值的训练分支，⏳ 未执行；必须用独立 YAML/checkpoint，不改 `configs/retro.yaml`。
- 正式 10k–30k pilot → 完整重训（retro_v2）—— pilot 未按文档最终执行；A6000 已完成一次 600k 重训并评估（⏳ 后续按需）。
- reverse-rate corrector / localized edit / RetroAgent 式 planner —— 均为“论文方向”，需先满足各自前提（如独立 reward、明确研究目标），⏳ 未执行。

## 7.4 评估侧计划

- confirm_unique1000_aug20（seed 42/43/44 各一次）→ final_unique2000_aug20 → 完整 src-test 一次性评估 —— ⏳ 全部等待通过 dev gate 的方法。

---

# 8. 关键事实索引

## 8.1 后续模型做判断时必须知道的最重要事实

1. **当前最高准确率采样配置**：R9K1M2（n_runs=9, n_branches=1, n_children=2）、`stochastic_noop`、bonus=0.5、`full_probability`、100 steps/cubic、batch64、TF32 high、seed42、legacy_best_rank；mini-1001（旧 ckpt）Top-1/3/10/Oracle = 57.0/78.1/86.1/91.8%，wall 3059.83 s。
2. **当前 dev 基线（普通 Euler）**：dev_unique1000_aug20 Top-1/3/5/10=58.2/75.5/79.8/83.5%，Oracle 86.6%，invalid 11.805%，wall 1,270.9 s。
3. **所有 guidance 候选均未通过 dev gate**：Top-1 在 55.8–57.6% 之间（低于 58.2%）；E3 终端重排 Top-1 57.6% 也失败。
4. **“AUC/排序改善 ≠ Top-1 改善”是重复出现的模式**：residual ranker（AUC +2.45pp，Top-1 -9.6pp）、P2 校准（同组 AUC +0.0257，Top-1 -0.5pp）、E7（pair acc +4.68pp，Top-1 -1.5pp）。
5. **Reward 上限**：forward-beam=5 对正确/错误候选的判别 AUC ≈0.69–0.71，同组 ≈0.68–0.73；错误候选得正分率 ≈46–51%。
6. **DGM 不是 exact**：89.735% 的 insert 位于连续 GAP run，无法唯一反演；任何“exact DGM 已实现”的说法都违反文档事实。
7. **R 保护谱系 > 全局 K 竞争**：R9K1M2 vs R1K9M2（同 9 条初始流）在 mini-1001 上 Top-1/3/10/Oracle 全面显著更好；R1 只是更快。
8. **M2 的收益是排序集中化**：Oracle 几乎不变（R10 上 +0.1pp），Top-3/5 提升明显；M3/M4 回退。
9. **新 checkpoint 与旧 checkpoint 几乎持平**：mini-1001 差异 ≤1.2pp；不要用旧 ckpt 的 tiny 数字直接推断新模型。
10. **保留集状态**：confirm（1,000 反应）、final（2,000 反应）、完整 src-test（5,007 反应）均未使用；任何声称已在这些集合上验证的说法都无文档依据。

## 8.2 影响结果可信度的历史缺陷与勘误（必须保留的 Bug 表）

| 缺陷 | 影响 | 修复/处理 |
|---|---|---|
| Rate correction 在 t=0 时 clamp 1e-12 → rate 放大 1e12 倍，生成随机垃圾 | 早期可视化/采样结果失真 | `clamp_min(1e-2)`（两处 sampler + first-step 可视化） |
| Euler-Beam 分支共享全局 RNG、seed 不生效且依赖 batch 排列 | 旧 58% 基线的随机语义不成立 | 任务 0 私有/无状态 branch RNG；历史结果不能与现实现严格比较 |
| 评分文件/beam_size 混用风险 | `bench_euler` 曾误读 beam 文件；早期 56/68/74 vs 54/66/74 | metadata + 严格行数/哈希校验（任务 12） |
| correctness-reward 脚本把 target component count 当 product 特征（leakage） | 旧 P2/P3 的 AUC/rerank 全部失效 | `532a46e` 修复；v2/v3 用无泄漏特征重新评估 |
| 早期“shared-anchor”数据实际不共享 state/time | P5 pairwise 指标无效 | P5b 审计发现；P5c 实现真实 continuation 并重做 P5d |
| adaptive endpoint 时间近似错误 | 修正前 1k shared-anchor 文件无效 | `get_euler_step_times()` 修正，旧文件作废 |
| continuation 原地改写 GPU 输入 tensor | 2/1000 条 transition_tokens 被污染 | `68f89de` clone 修复；污染文件改名保留 |
| 普通 Euler 的 `--seed` 未接入采样器 | 历史 Euler 对照不可严格配对 | 已修复（metadata 记录 `seed_applied_to_sampler=true`） |
| PyTorch 2.6+ `torch.load(weights_only=True)` 拒绝旧 checkpoint | 采样加载失败 | `88a0f2e` 显式 `weights_only=False` |
| `visualize_trajectory --n_branches` 伪支持（未实现 recording 却解包） | 运行时崩溃/误导 | 明确报错；完整 branch 树记录未实现 |
| 旧 `val200` 目录名歧义（200 行 vs 200 反应） | 早期 smoke 曾被误认为 200 反应 | v2 协议以反应为统计单位并排除历史 [0,600) |
| 早期“Euler 完整 test 30–40 分钟”无 metadata | 无法复现，不能作为当前效率证据 | 以当前实测与 metadata 为准 |

## 8.3 文档冲突速查

| 冲突点 | 早期/旧说法 | 当前说法 |
|---|---|---|
| Euler-Beam vs Euler | tiny 上全面更优（58/68/76 等） | 仅排序前移方向证据；R9 vs Euler N9 在 200 反应上 McNemar 不显著 |
| bench_euler 基线 | 56/68/74 | 文件实测 54/66/74（来源差异，非评分重构造成） |
| 58% 恢复版本 | 可作对照 | 不同 seed/评分语义，仅研究线索 |
| correctness reward v1 | AUC 0.7306 等 | 有 target leakage，失效；v2/v3 为权威 |
| shared-anchor pairwise（P5） | 排序 59.66% 等 | 数据不共享 anchor，指标无效；P5d 修正后联合 gate 仍未过 |
| validation-A/B | 曾用于方法结论 | 降级为历史探索；方法结论以 v2 dev/confirm/final 为准 |
| guidance checkpoint best vs final | — | 协议选 best（Bregman/pairwise gate）；final 在个别 Top-k 上更好但 Oracle 更低，不能事后换 |
| 环境版本 | execution report: PyTorch 2.7.1+cu126 | dgm.md 依赖清单: torch 2.13.0+cu130（未消解，复现以实际环境为准） |

---

> 本文件由 `new_docs/` 全部正文文档忠实重建；原始 Markdown 未被修改或删除。如需核对单条事实，请以各文档的完整上下文和实验目录 metadata 为准。

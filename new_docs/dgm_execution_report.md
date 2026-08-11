# DGM 后续计划执行报告

执行环境：conda `ef`，Python 3.10.20，PyTorch 2.7.1+cu126，NVIDIA RTX 3090（23.56 GiB）。

结论先行：P0、P1、P2 已完成；P2 的新 correctness reward 在离线排序指标上有正向信号，但 P3 的同候选池终点 rerank 失败。因此按照计划的 stopping rule，P4 不启动，也没有读取 confirm、final 或完整 test。当前不能声称“新 reward + DGM”已经改善端到端逆合成。

## 1. 执行范围与提交

本轮只使用 train-1000、reward holdout-200（原始训练反应 1000–1199）和 dev-1000。没有修改模型训练代码以外的实验分支，也没有重新扫描已关闭的 E7 的 β、seed、训练步数或 checkpoint。

关键提交：

- `c2e1e83`：paired `scripts/visualize_trajectory.py` 与 Euler 事件字段
- `429ae0f`：冻结 P0 基线协议
- `70db4b6`：冻结 `P1-panel-v1` 分层清单
- `1305c49`：新增 correctness reward 训练/评估管线
- `34e789d`：paired trajectory 的结构化 metadata
- `68540dd`：correctness reward 单元测试

测试：`50 passed`（含新增 reward、paired visualization 和 Euler sampling 测试）。

## 2. P0：基线冻结

普通 Euler 的正式复跑使用基础 checkpoint `new_checkpoints/checkpoint_step600000.pt`、100 steps、20 augmentation、每条 augmentation 3 个候选、batch 64、seed 42。完整命令、输入哈希和资源记录见 [`dgm_p0_protocol.md`](dgm_p0_protocol.md)。

产物：

- `results/dgm_execution/p0_euler_baseline_seed42/predictions.txt`
- 预测 SHA-256：`9e645acca9718a55ff8ef7c63fafead82f66e2f4e7f126ca2817ed33e59b5b38`
- 输入 `src.txt` SHA-256：`c20e337496f52bbeacc7e5870e3eefbb27a653800fa3d3b0fd8dbca8cfd098f6`

| 指标 | 普通 Euler |
| --- | ---: |
| Top-1 | 58.2% |
| Top-3 | 75.5% |
| Top-5 | 79.8% |
| Top-10 | 83.5% |
| Oracle-any | 86.6% |
| invalid（输入 rank 1/2/3） | 11.875% / 11.425% / 12.115% |

与既有 E1 结果一致到报告精度，P0 通过。和冻结 E7 的 reaction-level paired comparison 为：Top-1 `56.7% - 58.2% = -1.5 pp`（95% CI `-3.3～+0.3`），Top-3 `+0.4 pp`，Top-10 `+0.3 pp`，Oracle `0.0 pp`。E7 保留为历史对照，不再调参。

## 3. P1：改进前的成对 `visualization_trajectory`

### 协议

使用 `scripts/visualize_trajectory.py` 的 paired 模式，对冻结 Euler 与 E7（`beta=0.10`、`per_position`）使用同一产品行、同一 seed、100 steps。依据 P0/E7 reaction-level Top-1 分成四层，每层固定抽 8 个原始反应，seed 为 42/43/44；清单见 [`dgm_p1_panel_v1.md`](dgm_p1_panel_v1.md)。实际生成 12 份 HTML（96 对 path），每条 HTML 仍保留完整编辑事件表、ORACLE、BASE MODEL、`GUIDANCE H^β` 和 GUIDED MODEL。

结构化汇总及 HTML/metadata 哈希保存在：
`/root/autodl-tmp/dgm_reward_runs/correctness_reward_v1/p1_panel_summary.json`（SHA-256 `004cb88e0433a403363b1a2cf3d0c8520928e3ef071cdd9cbc16b7a42de816e8`）。

### 结果

| P1 面板汇总（96 paths） | Euler | E7 guidance |
| --- | ---: | ---: |
| 有效终点 | 82/96 | 83/96 |
| canonical target 匹配 | 23/96 | 24/96 |
| 总 insert / substitute / delete 操作 | 490 / 53 / 8 | 490 / 51 / 7 |
| forward beam 命中（beam=5） | 45/96 | 46/96 |

两条路径在 14/96 个 path 上出现首次精确 state 分叉；分叉 step 的中位数为 74.5。结果分解为：72 条两者都错、23 条两者都对、1 条 guidance 单独对、0 条 Euler 单独对。由于这是按第一 augmentation、每个 seed 一条 path 的机制面板，不能把 `23/96` 或 `24/96` 当作 dev 性能估计；它回答的是“实际编辑路径改变了什么”。

判断：E7 在少数样本上改变了末端，但没有出现稳定的“减少 substitution/invalid 或提前修复同一类错误”的机制证据；改进前诊断支持继续检查 reward 与 correctness 的错配，而不支持继续扩大当前 beam/guidance 分支。

## 4. P2：新的逆合成 correctness reward

这不是已关闭的 forward-reward calibration。新标签为：canonical(candidate) 等于数据集真实反应物则为正例；冻结 Euler 产生的有效且不等于 target 的候选为负例；invalid 不参与训练，推理时固定最低分。

### 训练方法与参数

- 数据：`train_shared_anchor1000_t10_30_50_70_90_beam.pt`，20,000 条记录，原始反应 0–999
- 去重：每个 `(product_index, canonical candidate)` 一个候选，重复候选保留最高 raw reciprocal-rank 代表
- 去重后：5,452 candidates；668 invalid；812 positives；产品级内部验证为反应 0–799 / 800–999，零重叠
- 模型：单层 logistic correctness head，228 维低容量特征
  - raw forward reciprocal-rank、validity、长度/长度差、组分数
  - product/candidate token histogram 及差分
- 训练：BCE，反应级平衡抽样，batch 64，Adam，learning rate `1e-3`，weight decay `1e-4`，2,000 steps，seed 42
- 冻结 checkpoint：`/root/autodl-tmp/dgm_reward_runs/correctness_reward_v1/reward_model.pt`
  - SHA-256：`93be56fd303f639116708c2ed5387dbbc023ee058f1b1544868c0baaf023edeb`

### P2/P3 一次性 holdout 报告

holdout 为原始反应 1000–1199 的 4,000 条 shared-anchor records；去重后 981 candidates（124 invalid、166 positives）。模型冻结后一次性生成 AUC 与终点 rerank，报告在：
`/root/autodl-tmp/dgm_reward_runs/correctness_reward_v1/holdout_v1/holdout_report.json`。

| 指标（valid candidates） | raw forward reward | correctness reward | 差值 |
| --- | ---: | ---: | ---: |
| 全局 AUC | 0.6864 | 0.7306 | +4.42 pp |
| shared-anchor 组内 AUC | 0.6114 | 0.6777 | +6.63 pp |

全局 AUC 差值的 reaction-level bootstrap 95% CI 为 `+0.85～+7.75 pp`；组内 AUC 差值 CI 为 `-4.75～+17.33 pp`，说明 holdout 只有 200 个反应时组内证据仍较宽。invalid 没有被高分：124 个 invalid 全部被固定到最低分。

P2 的点估计达到计划中 `+0.02` 的两个 AUC 门槛，但组内区间不充分精确；这不是端到端有效性的证明，只说明它比 forward reconstruction reward 更接近当前 holdout 的局部 correctness 标签。

## 5. P3：终点 rerank gate（失败）

三种排序读取完全相同的去重 holdout 候选池：原始首次出现顺序、raw forward reciprocal-rank、冻结 correctness reward。真实 target 只在候选池固定后用于统计。

| 排序 | Top-1 | Top-3 | Top-5 | Top-10 | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| 首次出现顺序 | 47.5% | 75.0% | 81.0% | 83.0% | 83.0% |
| raw forward reward | 48.5% | 79.5% | 82.5% | 83.0% | 83.0% |
| correctness reward | 42.0% | 76.0% | 82.0% | 83.0% | 83.0% |

P3 gate 要求 Top-1 不下降；实际 correctness reward 比 raw forward reward 低 6.5 pp，Top-3 低 3.5 pp。Top-10 和 Oracle 没有补偿性提高，候选数与 invalid 比例相同。因此 P3 明确失败。

这给出比“模型没学到 reward”更具体的判断：模型确实提高了离线局部排序 AUC，但它学到的局部 correctness 分数在同一终点候选池上不能保住第一名。问题更可能位于候选级特征/排序目标与 reaction-level Top-1 的错配（以及 shared-anchor 组内样本量不足），而不是 DGM 优化器尚未运行够久。根据停止规则，不能在该 holdout 上继续加特征、调阈值或挑 checkpoint 来挽救结果。

## 6. P4 状态与保留集保护

P4-A（重建 guidance 数据）、P4-B（训练新 DGM）、P4-C（新 guidance dev）、P4-D（改进后复用 P1 面板）均**未启动**。原因不是资源或实现错误，而是 P3 的硬依赖 gate 失败：终点排序不改善的 reward 没有理由进入逐步 guidance。confirm、final 和完整 test 也保持未读取。

因此当前新增 reward 的正确结论是：

> 它在 holdout 上提高了局部 correctness AUC，但在相同候选池的 reaction-level Top-1 rerank 上反而下降；“AUC 提高 ⇒ DGM/逆合成性能提高”的假设被本次 P3 对照否定。

## 7. 当前下一步（按证据而非继续调参）

1. 保留本次 reward 与 holdout 报告作为失败分支，不训练其 DGM。
2. 如果继续研究 reward，先设计能直接优化 reaction-level candidate selection 的候选组损失/特征，并预先规定新的独立 holdout；不能在本 holdout 上反复修补。
3. 进一步分析 P1 中“首次分叉晚、编辑类型几乎不变”的路径，重点检查 credit assignment 与 terminal-to-action 映射，而不是简单增加 guidance 强度。
4. 只有新的 reward 通过离线 AUC **和**终点 rerank gate，才允许按原计划执行 P4，并在 dev 通过后原样复用 `P1-panel-v1` 做改进后成对可视化。


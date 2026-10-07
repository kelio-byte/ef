# 采样 action 语义审计与修复（2026-08-14）

## 结论

本次修复的是**采样 action support 与训练 target 不一致**的 bug，不改变模型、loss、训练数据、训练配置或 checkpoint。

- `INS(pos=0)` 是合法的：它在 BOS **之后**插入 token，不会改写 BOS；
- `SUB(pos=0)` 和 `DEL(pos=0)` 非法：它们会改写/删除 BOS；
- `PAD/BOS/GAP/UNK` 不可作为 INS/SUB 的输出 token；
- `SUB(pos, current_token)` 是 no-op，不应被采样为一次编辑；
- ordinary Euler、Euler-Beam、Euler-SMC、Structured v1/v2 现在遵守同一语义；单编辑 `beam.py` 原本已经遵守该语义。

修复前的 A/B 源码基线已固定为 commit `97efd65`。本次**不**用仅训练 50k steps 的 SPE checkpoint 重新比较 Top-K/Oracle：该比较不足以判断 SPE 质量，待有训练充分且协议匹配的 checkpoint 后再做。

## 为什么 pos=0 的 INS 必须存在

训练 loader 读取 pre-aligned Z-space pair，随后在 `prepare_batch()` 前面补 BOS。若对齐 target 在原序列首位有 GAP，`fill_gap_tokens_with_repeats_log()` 会把该 GAP 对应的 INSERT action 投影到 X-space 的位置 0；`apply_ins_del_operations()` 的 `INS(pos=0)` 定义正是“BOS 后插入”。

对当前 SPE 训练文件的只读流式统计：

| 项目 | 数量 |
|---|---:|
| aligned train pairs | 800,060 |
| 有至少一个 leading insert 的 pair | 96,187（12.02%） |
| leading insert action 总数 | 279,240 |
| target 中的 PAD/BOS/UNK 输出 action | 0 |

因此，旧的“位置 0 所有操作都屏蔽”会排除训练中真实出现的目标；正确约束是“BOS 不可被编辑，但可作为首位插入锚点”。

## 生产路径对照

| 路径 | 修复前 | 修复后 |
|---|---|---|
| `sample_euler` / intervention / oracle diagnostic | 所有 pos=0 action 被拦截；Q 可抽 special/no-op | 仅 INS 可用；Q 仅在合法 token 上条件化；非法 rate 在抽样前归零 |
| `sample_euler_beam` | 所有 pos=0 action 被拦截；Q 可抽 special/no-op；proposal 与 score support 不一致 | 与 Euler 相同；path score 使用同一条件 Q 与有效 rate |
| `euler_smc` | 继承 Euler-Beam 的 proposal/score 不一致 | 继承修复后的 proposal，并向 score 传入 state support |
| Structured v1/v2 | 枚举从 position 1 开始 | 可枚举 `INS(pos=0)`，仍禁止 BOS SUB/DEL |
| single-edit greedy/beam (`beam.py`) | 已允许 BOS 后 INS、禁止 BOS SUB/DEL、过滤 special/no-op | 无生产行为改动；已回归确认 |

其中 Q 的“条件化”意味着：在固定 INS/SUB rate 下，只在合法输出 token 中归一化；若某位置没有任何合法 Q token，该 INS/SUB rate 直接视为 0。这样不会把非法 event 先抽出再静默丢弃，也使 Euler-Beam 的 proposal 和 score 使用同一分布。

## 实现边界

- 新的共用 helper 位于 `edit_flows/sampling/ops.py`；训练模块没有导入它。
- 本次 diff 只覆盖 `edit_flows/sampling/` 与测试文件；未修改 `edit_flows/training/`、`scripts/train_retro.py`、SPE YAML、数据文件或 checkpoint。
- `LOG_ZERO_CUTOFF` 把模型 padding 路径使用的有限负哨兵（`-1e9`）当作零概率，避免“全 masked Q”在重归一化后被错误复活为均匀分布。
- 历史 `old_euler.py`、`recover_euler_beam.py` 不属于正式 `scripts/sample_retro.py` sampler 入口，未改写其历史行为。

## 验证

- 新增覆盖：BOS 后 INS 的实际序列效果、leading GAP 到 pos=0 的 target 投影、Euler/Euler-Beam special/no-op 过滤、score 对 BOS 非法 rate 的忽略、Structured v1/v2 的 BOS 插入锚点。
- 定向采样/对齐测试：162 passed。
- 全量 CPU 回归（排除需加载外部 legacy Molecular Transformer checkpoint 的一项）：391 passed，1 deselected。
- 训练与集成回归：26 passed。`git diff --check` 和 sampling/training `compileall` 通过。

该 legacy checkpoint 测试在当前无卡容器的 2 GiB cgroup 下会被系统终止；它与本次采样代码无关，且不影响上述训练/采样回归结论。

## 后续评估约定

本次只证明语义正确性与回归安全性，不声称准确率提升。待 SPE 有训练充分的 checkpoint 且 GPU 可用后，再在固定 100-reaction、20 augmentation、N=10、100-step/cubic、seed 42 协议下，对 `97efd65` 与本修复提交分别采样并正式报告 Top-1/3/5/10、Oracle、invalid、unique candidates 和 wall time。

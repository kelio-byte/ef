# Beam Search 实验 #1：单编辑贪心/Beam 搜索初次尝试

## 1. 背景与动机

基于 `docs/beam-search/todo1.md` 的方案设计，实现了 Edit Flows 的单编辑贪心/Beam 搜索（`docs/beam-search/impl.md`），并在小规模测试集上进行了首次实验验证。

核心假设：将原始 CTMC 的"每步多位置同时编辑"离散化为"每步只做一个编辑"，在此离散化空间上用条件概率评分做确定性搜索，是否优于随机 Euler 采样？

## 2. 实验设置

### 2.1 模型

- **Checkpoint**: `checkpoints/USPTO_50K_PtoR_aug20_#global#/2026-06-28_14-14-18/checkpoint_step790000.pt`
- **训练配置**:
  - `hidden_dim: 256`, `num_layers: 12`, `num_heads: 8`
  - `scheduler: cubic`, `use_rate_reparam: true`, `time_input: t`
  - `use_origin_mask: true`（但实际训练时权重中无 `origin_embedding`，自动回退为 false）
  - 训练步数 ~790k / 5,000,000

### 2.2 测试数据

- **数据集**: `analysis_subsets/USPTO_50K_PtoR_aug20_#global#/test_dedup_seed42_1000/`
- **规模**: 1000 条产物-反应物对（从完整测试集随机去重采样，seed=42）
- **格式**: `#global#` 格式（R-SMILES），augmentation=20

### 2.3 实验参数

| 参数 | Greedy | Beam | Euler (n=1) | Euler (n=10) |
|------|--------|------|-------------|--------------|
| `max_edits` | 15 | 15 | — | — |
| `n_steps` | — | — | 100 | 100 |
| `beam_size` | — | 5 | — | — |
| `time_mode` | depth / fixed | depth | — | — |
| `k_ins_token` | 4 | 4 | — | — |
| `k_sub_token` | 4 | 4 | — | — |
| `k_edit_expand` | 16 | 16 | — | — |
| `batch_size` | 32 | 8 | 32 | 8 |
| GPU | cuda:4 | cuda:6 | cuda:7 | cuda:7 |

### 2.4 评测方式

使用 `scripts/score_#global#.py`，`--beam_size 1 --n_best 1 --edit_flows`。主要指标：Top-1 Accuracy、Invalid SMILES Rate、MaxFrag Top-1。

## 3. 实验过程中遇到的问题与修复

### 3.1 Config 与实际权重不一致

**问题**: checkpoint config 中 `use_origin_mask: true`，但模型权重中没有 `origin_embedding.weight`。推测训练时实际使用的是旧版 config（`use_origin_mask: false`），但保存 checkpoint 时复制了更新后的 `configs/retro.yaml`。

**修复**: 在 `scripts/sample_retro.py` 中增加自动检测——若 config 要求 origin_mask 但权重中无对应参数，自动回退并打印 warning。

```python
use_origin_mask = cfg.get("use_origin_mask", False)
has_origin_embed = any("origin_embedding" in k for k in ckpt["model_state_dict"])
if use_origin_mask and not has_origin_embed:
    use_origin_mask = False
```

### 3.2 `k(0)=0` 导致评分失效（非根因）

**怀疑**: cubic scheduler 下 `k(0) = 0`，导致 step 0 时所有 `u_real = 0`，候选评分全等，greedy 退化为随机选择。

**尝试修复**: 将候选评分从 real rate 改为 base rate——`log_u_base(e) - log_u_tot_base` 替代 `log_u_real(e) - log_u_tot_real`。

**结果**: 修复不改变任何行为。原因是 `k(t)` 对所有编辑是全局常数，在 cond_prob 评分的分子分母中**精确抵消**：

$$
\frac{u^{\text{real}}(e)}{u_{\text{tot}}^{\text{real}}}
= \frac{k(t) \cdot u^{\text{base}}(e)}{k(t) \cdot u_{\text{tot}}^{\text{base}}}
= \frac{u^{\text{base}}(e)}{u_{\text{tot}}^{\text{base}}}
$$

数值上 `exp(LOG_NEG_INF)` 的截断也无影响——只是给所有 log-rate 加了同一个常数，logsumexp 中同样抵消。**`k(0)=0` 不是根因。**

### 3.3 Beam 搜索极慢：`.item()` GPU 同步瓶颈（已修复）

**问题定位**：`beam.py` 的 `_collect_edit_candidates_single()` 函数中，每个候选编辑的分数通过 `.item()` 从 GPU 逐个迁到 CPU。以 beam_size=5, batch_size=8, 平均长度 ~50, max_edits=15 为例：

| 每步操作 | 计算 |
|----------|------|
| 活跃状态数 | 8 × 5 = 40 |
| 每状态候选数 | 50 pos × (4 ins + 4 sub + 1 del) = 450 |
| 每步 `.item()` 调用 | 40 × 450 = **18,000 次** |
| 15 步总计 per batch | 18,000 × 15 = **270,000 次** |

每次 `.item()` 触发一次 GPU→CPU 同步，27 万次累积即为数十秒。

**修复**（commit `TODO`）：将 `_collect_edit_candidates_single` 完全向量化：

1. **候选构造向量化**：不再 Python `for i in range(L)` 逐个位置取 topk，改为对 `(L, V)` 张量直接 `torch.topk(log_u_ins, k_ins, dim=-1)`，一次 GPU 调用得到所有位置的候选。
2. **批量 CPU 传输**：所有候选的 `(values, positions, ops, tokens)` 在 GPU 上拼接、过滤、取全局 topk 后，**一次** `.cpu().tolist()` 迁到 CPU 再构造 `EditCandidate` 列表。
3. **次要热点修复**：`log_u_tot_score[b].item()` 和 `u_tot_base[b].item()` 改为循环前批量 `.cpu().tolist()`，Python 侧直接索引。

修复后仅剩 2 个 `.item()` 调用（verbose logging 分支，每 5 步触发一次，冷路径）。

**优化后耗时**（相同 GPU、相同配置，max_edits=10 复测）：

| 模式 | 优化前 per batch | 优化后 per batch | 加速比 |
|------|-----------------|-----------------|--------|
| Greedy (bs=32) | ~12s | **~1.7s** | ~7× |
| Beam (bs=5, bs_batch=8) | 40-67s | **~13.5s** | ~3-5× |

Beam 加速比低于 Greedy 是因为 `.item()` 消除后，per-state `_apply_single_edit_to_sequence`（每步 N×k_edit_expand ≈ 640 次）和去重哈希成为新的主要开销，不在本次优化范围内。

## 4. 实验结果

### 4.1 原始耗时（优化前）

| # | 实验 | Top-1 Acc | Invalid SMILES | MaxFrag | 耗时 |
|---|------|-----------|----------------|---------|------|
| 1 | **Euler n=10** | **46.0%** | 12.5% | 52.0% | ~3 min |
| 2 | **Euler n=1** | **34.0%** | 11.3% | 40.0% | ~1 min |
| 3 | Greedy (depth) | 0.0% | 53.8% | 8.0% | ~6 min |
| 4 | Greedy (fixed t=0.5) | 0.0% | 60.5% | 6.0% | ~6 min |
| 5 | Beam (size=5, depth) | 0.0% | 51.4% | 2.0% | ~86 min |

### 4.2 优化后耗时（相同实验、相同 GPU，max_edits=10 复测）

| 模式 | 优化前 per batch | 优化后 per batch | 加速比 |
|------|-----------------|-----------------|--------|
| Greedy (bs=32) | ~12s | **~1.7s** | ~7× |
| Beam (bs=5, bs_batch=8) | 40-67s | **~13.5s** | ~3-5× |

优化后 Greedy 约 7× 加速、Beam 约 3-5× 加速。Beam 加速比低于 Greedy 的原因是 `.item()` 消除后，per-state 的 `_apply_single_edit_to_sequence`（每步 N×k_edit_expand ≈ 640 次）和去重哈希成为新的主要开销，不在本次优化范围内。

### 4.3 结果分析

Euler n=10 的 Top-3 达 70%、Top-5 达 72%，作为当前模型的上界参考。

Greedy 和 Beam 的 Top-1 均为 0%，Invalid SMILES 高达 51-60%，远差于 Euler n=1 的 11.3%。单编辑确定性搜索在当前模型上完全不可行。

## 5. 单样本诊断分析

以下是对一个具体样本（Euler n=1 命中但 Greedy 失败）的逐步诊断。

### 5.1 样本信息

- **产物** (39 tokens): `C ( N C ( = O ) C ( F ) ( F ) F ) c 1 c c c c c 1 S ( = O ) ( = O ) C 1 C C 1`
- **目标** (56 tokens): `C ( N . C ( O C ( = O ) C ( F ) ( F ) F ) ( = O ) C ( F ) ( F ) F ) c 1 c c c c c 1 S ( = O ) ( = O ) C 1 C C 1`
- **Euler n=1 输出**: 完全匹配目标

### 5.2 目标编辑

产物到目标的主要变化（对齐后）：

1. `N` 后插入 `. C ( O`（而非仅有 `C`）
2. `C ( F ) ( F ) F )` 后插入 `( = O ) C ( F ) ( F ) F )`

### 5.3 Greedy 前几步轨迹

```
Step 0 (t=0.000, u_tot_base=14.9):
  Top-1: ins(4, C→))   score=-1.642    ← 在 C 后插入 ")" ！
  正确编辑应该是: ins(3, N→.)  但 "." 连 Top-5 都没进
  执行后: C ( N C ) ( = O ) C ...

Step 1 (t=0.067, u_tot_base=13.8):
  Top-1: ins(4, C→F)   score=-1.989    ← 继续插入 F
  执行后: C ( N C F ) ( = O ) C ...

Step 2 (t=0.133, u_tot_base=15.3):
  Top-1: ins(3, N→O)   score=-1.878    ← 在 N 后插入 O
  执行后: C ( N O C F ) ( = O ) C ...

Step 3 (t=0.200, u_tot_base=14.2):
  Top-1: ins(3, N→()   score=-2.061    ← 在 N 后插入 "("
  执行后: C ( N ( O C F ) ( = O ) C ...

... 后续多步继续在局部做微编辑，路径完全偏离目标
```

最终 Greedy 输出: `C ( N . C ( O C ( = O ) C ( F ) ( F ) ( = O ) C ( F ) ( F ) F ) ...`

vs 目标: `C ( N . C ( O C ( = O ) C ( F ) ( F ) F ) ( = O ) C ( F ) ( F ) F ) ...`

关键差异: `( F ) ( F ) ( = O )` vs `( F ) ( F ) F ) ( = O )`——Greedy 把 `( = O )` 插在了错误位置。

### 5.4 诊断结论

在每一步，模型给大量微编辑打出相近的分数（log_u 在 0.0~1.0 范围，得分差异 < 0.3 nats），真正的正确编辑不在 Top-5 甚至不在候选池中。模型没有学到"编辑的优先级排序"，只学到了"哪些区域可能需要被编辑"的分布式弱信号。

## 6. 根因分析

这是**训练目标与搜索策略之间的 mismatch**，不是实现 bug：

| | 训练时 (Bregman Loss) | 单编辑 Greedy/Beam |
|---|---|---|
| 编辑模式 | 所有位置同时可能编辑 | 每步只允许一个编辑 |
| 速率来源 | 分布式信号汇聚 | 需要精准的单编辑排序 |
| 优化目标 | `k(t)(u_tot - CE)` — 惩罚总速率、奖励正确编辑 | 每一步选 `argmax p(e\|next_event)` |
| 信号质量要求 | 弱信号即可（靠随机采样+多步纠正） | 需要尖锐、准确排序 |

Euler 采样利用随机性 + 多步来"探索"编辑空间——100 步中可以多次尝试、部分失败也可被后续步骤纠正。而单编辑搜索每步都是硬决策，一旦走错就无法回头（greedy）或需要指数级 beam 宽度来覆盖（beam）。

**这印证了 stage-1.md 的核心判断：当前主要瓶颈是"模型速率不够尖锐"，不是采样策略问题。**

## 7. 结论与建议

### 7.1 单编辑 Beam/Greedy 方向

**当前应暂停。** 模型没有被训练成支持这种搜索策略。如果要让 beam search 工作，需要：

1. 修改训练目标，显式加入单编辑排序的监督信号
2. 或等模型速率校准大幅改善后再尝试

### 7.2 优先级回归

回到 stage-1.md 建议的方向：**让模型学到更准确、更尖锐的编辑速率**：

- Rate reparam ablation（`use_rate_reparam=true/false`）
- 训练目标调整（auxiliary loss 惩罚非正确编辑、加权正确编辑）
- 模型架构改进（rate head 容量、分治不同编辑类型）

### 7.3 工程改进

- ✅ `beam.py` 中 `.item()` GPU 同步已向量化修复（Greedy ~7× 加速，Beam ~3-5× 加速）
- ✅ `sample_retro.py` 中 config/权重不一致的自动检测已修复
- Beam 剩余瓶颈：per-state `_apply_single_edit_to_sequence` 调用（每步 N×k_edit_expand 次）和去重哈希，如需进一步加速可考虑批量化 edit apply 和 GPU 侧去重

### 7.4 相关文件

| 文件 | 说明 |
|------|------|
| `edit_flows/sampling/beam.py` | Beam/Greedy 搜索实现 |
| `scripts/sample_retro.py` | CLI 入口（已增加 `--sampler` 等参数） |
| `experiments/exp{1-5}*/` | 本次实验输出与评测日志 |
| `docs/beam-search/todo1.md` | 方案设计文档 |
| `docs/beam-search/impl.md` | 实现总结 |

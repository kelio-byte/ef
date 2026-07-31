# Euler-Beam 设计与实现文档

## 1. 动机

纯 Euler 采样的问题（来自之前分析）：

| 问题 | 表现 |
|------|------|
| 方差大 | 同一产物 10 次采样，命中率从 0/10 到 3/10 不等 |
| 可能静默 | 100 步 0 编辑，完全浪费 |
| 无法利用共识 | 多条成功轨迹无法互相加强 |

Beam 思想：同时维护 K 条 Euler 轨迹，去重加权，剪枝保留高质量分支。

## 2. 核心设计

### 2.1 数据结构

```python
class _BranchState:
    x_t: Tensor       # (1, L) 当前序列
    weight: float     # 共识权重（多分支汇聚 = 高权重）
    path_log_p: float # 编辑路径累计 log-prob（≤0）
    t: float          # 当前时间
    seed: int         # 随机种子（保证分支间独立性）
```

### 2.2 主循环

每步执行：

```
1. Gather: 收集所有 t < 1.0 的活跃分支
2. Batch:  所有分支的 x_t 拼成 (N, max_L) tensor
3. Forward: 一次 model(x_batch, t_batch, pad) → log_rates, log_ins_probs, log_sub_probs
4. Sample:  每个分支独立 torch.manual_seed(s.seed+step) → _sample_edit_actions()
5. Score:   计算本步 log_p（基于实际触发的编辑的概率）
6. Merge:   相同 token 序列的分支合并权重
7. Prune:   按 (-path_log_p, weight) 排序，保留 Top-K
8. Split:   不足 K 条时从最高 rank 分支分裂（新 seed）
```

### 2.3 编辑评分

```python
# 每个触发的编辑的 log-prob
# INS: log(1-exp(-h·λ_ins)) + log(token_prob)
# SUB: log(1-exp(-h·λ_sub)) + log(token_prob)
# DEL: log(1-exp(-h·λ_del))
# 累积到 path_log_p
```

### 2.4 去重与加权

```python
# 相同 token 序列 → 合并
merged[key].weight += br.weight        # 共识度累加
merged[key].path_log_p = max(lp_a, lp_b)  # 保留更好的路径

# 但：如果所有分支都相同（尚未发散），不合并
# → 保留所有分支的种子多样性
if len(merged) == 1:
    ranked = candidates  # 保持原样
```

### 2.5 排序与剪枝

```python
# 优先保留有编辑的分支（path_log_p < 0）
# path_log_p = 0 表示什么都没做 → 排最后
sort_key = (-path_log_p, weight)

# 保留 Top-K
all_branches[b] = ranked[:beam_size]
```

### 2.6 分裂

```python
# 不足 K 条时从最高 rank 分支克隆
while len(branches) < beam_size:
    parent = branches[len % len(branches)]
    branches.append(BranchState(
        x_t=parent.x_t.clone(),
        weight=parent.weight * 0.5,    # 分裂降权
        path_log_p=parent.path_log_p,
        t=parent.t,
        seed=parent.seed + 10000 + len,  # 新种子
    ))
```

## 3. 效率设计

| 方面 | 策略 |
|------|------|
| 模型 forward | K 条分支拼成一个 batch，**单次 GPU forward** |
| 采样 | CPU 循环，每分支一次 `torch.manual_seed` + `_sample_edit_actions`（轻量）|
| 去重 | 纯 Python dict，key 为 token tuple（无 RDKit，快速）|
| 排序 | 分支数 ≤ beam_size（默认 5），O(K log K) 可忽略 |

**单步耗时 = 1 次 model forward + K 次轻量采样**，与 beam_size 几乎线性无关。

## 4. 与现有采样器的对比

| | Euler | Euler-Beam | Greedy (beam.py) | Beam (beam.py) |
|:---|:---|:---|:---|:---|
| 分支数 | 1 | K | 1 | K |
| 编辑选择 | 随机采样 | 随机采样 | argmax | argmax |
| 每步编辑数 | 0~N | 0~N | 1 | 1 |
| 去重加权 | 无 | 有 | 无 | 有 |
| 确定性 | 否 | 可复现（固定 seed）| 是 | 是 |
| 多样性 | 仅随机 | 随机 + 多分支 | 仅编辑选择 | 仅编辑选择 |

## 5. 已验证的行为

- [x] 多分支在早期步（t<0.7）保持同步，t>0.7 后逐步发散
- [x] 不同 seed 的分支在不同步数触发编辑（已验证 #96740 在 step 73-99 间 5 分支共产生 ~18 次编辑）
- [x] 去重正确：相同序列合并权重，不同序列保留
- [x] 最终输出 ≠ 输入（序列长度从 52→56），确认编辑生效

## 6. 待解决问题

### 6.1 Event Recording（优先）

当前 beam 返回的 `all_events` 为空，导致可视化报告 "0 edit events"。需要参考 `sample_euler` 中的 event recording 逻辑，在 beam 模式中记录编辑事件。

### 6.2 排序策略验证

当前 `(-path_log_p, weight)` 的排序策略需要在大规模测试中验证：
- 是否会过早丢弃有潜力的分支？
- path_log_p 的累积是否合理（不同步数的分支如何公平比较）？

### 6.3 命中率对比

需要系统对比 Euler-Beam vs Euler 在多个产物上的命中率（如 10 产物 × 20 augs 全套测试）。

### 6.4 参数调优

| 参数 | 当前值 | 待探索 |
|------|:---:|------|
| beam_size | 5 | 3, 10, 20 |
| 分裂权重衰减 | 0.5 | 0.3, 0.7 |
| 去重阈值 | 精确 token match | canonical SMILES match |
| n_steps | 100 | 与 Euler 保持一致 |

## 7. 下一步计划

1. **实现 event recording**，让 beam 模式也能生成完整 HTML 可视化
2. **小规模对比**：产物 4837 上 beam_size=5 vs Euler 50 samples
3. **排序策略调优**：如果 beam 命中率持续低于 Euler，调整 sort key
4. **集成到 sample_retro.py**：添加 `--sampler euler_beam` 选项

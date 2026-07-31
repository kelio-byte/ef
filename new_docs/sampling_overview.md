# 采样方式总览

## 1. 采样器对比

项目在 `edit_flows/sampling/` 下有四种采样器：

| 采样器 | 文件 | 入口函数 | 类型 |
|:---|:---|:---|:---|
| Euler | `euler.py` | `sample_euler()` | 随机 |
| Euler-Beam | `euler_beam.py` | `sample_euler_beam()` | 随机+分支 |
| Greedy | `beam.py` | `sample_greedy_single_edit()` | 确定性 |
| Beam | `beam.py` | `sample_beam_single_edit()` | 确定性+分支 |

### 1.1 Euler (`sample_euler`)

```
机制: 连续时间随机编辑过程

每步:
  model(x_t, t) → log_rates (L,3), log_ins_probs (L,V), log_sub_probs (L,V)
  rate → prob = 1-exp(-h·λ)          ← 速率转概率
  rand() < prob → 触发编辑            ← 随机掷骰子(0~N个位置同时触发)
  multinomial(probs) → 选token        ← 按概率分布选token
  apply edits → x_{t+h}
  t += adapt_h

特点:
  - 每步可以编辑 0~N 个位置
  - 随机性强, 同一产物每次结果不同
  - 时间通过 kappa scheduler 自适应推进
  - n_samples 条轨迹完全独立, 全部输出
```

### 1.2 Euler-Beam (`sample_euler_beam`)

```
机制: K 条 Euler 轨迹的并行竞速

每步:
  K 条分支拼成一个 batch → 一次 model forward
  每条分支独立随机采样 (同一 batch, 不同 seed)
  逐分支应用编辑 + 计分

每步后:
  去重: 相同 token 序列 → 合并权重
  排序: (-path_log_p, weight) → Top-K 剪枝
  分裂: 不足 K 条时从最优分支克隆

特点:
  - 继承 Euler 的 0~N 编辑/步灵活性
  - 去重加权利用共识
  - 最终只输出 1 条最优分支
  - n_runs=3 表示独立跑 3 次, 输出 3 条
```

### 1.3 Greedy (`sample_greedy_single_edit`)

```
机制: 每步选全局最优的 1 个编辑

每步:
  model(x_t, κ) → log_rates, log_ins_probs, log_sub_probs
  合并评分: log_u = log(λ_ins) + log(token_prob)   ← 位置+token 联合打分
  全局 top-1: 跨所有位置×所有token选最高分
  应用这 1 个编辑
  κ 推进 (TimePolicy 或 Frozen-Hazard)

特点:
  - 每步恰好 1 个编辑 (确定性)
  - 不需要 n_samples (结果可复现)
  - 用 TimePolicy 决定何时 STOP
  - 输出 1 条/产物
```

### 1.4 Beam (`sample_beam_single_edit`)

```
机制: Greedy 的 beam 版本, K 条并行

每步:
  同 Greedy, 但取 top-K 个编辑 → K 个子状态
  去重, 按 log_p 排序, 保留 top-K
  STOP 是显式动作 (Frozen-Hazard 模式)

特点:
  - 每步恰好 1 个编辑 (确定性)
  - K 条 beam 并行
  - κ 每条独立
  - 输出 1 条/产物 (top-1 beam)
```

---

## 2. 关键区别

| | Euler | Euler-Beam | Greedy | Beam |
|:---|:---|:---|:---|:---|
| 编辑选择 | 随机采样 | 随机采样 | argmax | argmax |
| 每步编辑数 | 0~N | 0~N | 1 | 1 |
| 并行度 | 1 条 | K 条分支 | 1 条 | K 条 beam |
| 确定性 | 否 | 可复现(seed) | 是 | 是 |
| 时间推进 | t+=adapt_h | t+=adapt_h | κ=TimePolicy | κ=FH |
| STOP 机制 | t≥1.0 | t≥1.0 | 策略/U阈值 | 显式STOP |
| 输出/产物 | n_samples 条 | n_runs 条 | 1 条 | 1 条 |

---

## 3. 与各脚本的适配关系

### 3.1 `sample_retro.py` — 批量采样入口

```
--sampler 决定调哪个函数:

  euler       → sample_euler()          使用 --n_samples
  euler_beam  → sample_euler_beam()     使用 --n_branches --n_runs
  greedy_edit → sample_greedy_single_edit()  使用 --max_edits --time_policy
  beam_edit   → sample_beam_single_edit()    使用 --beam_size --max_edits

输出格式:
  每产物每轮一行, 空格分隔的 SMILES token
  Euler:  n_products × n_samples 行
  Beam:   n_products × n_runs 行
  Greedy: n_products × 1 行
  Beam:   n_products × 1 行

适配状态:
  ✅ euler        — 完全适配
  ✅ euler_beam   — 完全适配 (输出格式与 Euler 对齐)
  ✅ greedy_edit  — 完全适配 (原有)
  ✅ beam_edit    — 完全适配 (原有)
```

### 3.2 `score_#global#.py` — 打分

```
输入: predictions.txt (每行一个 SMILES)
参数: --augmentation N  --beam_size K  --n_best M

假设数据结构:
  predictions.txt 按 (产物, augmentation, beam位置) 排列
  总行数 = n_products × augmentation × beam_size

排序逻辑:
  1. 每个 augmentation 内去重 (保持顺序)
  2. 跨 augmentation 计分: Σ 1/(position+1)
  3. 用 best_position 打破平局
  4. Top-N ACC: target 在 top-N 命中 → 正确

适配状态:
  ✅ euler (--n_samples=K)         → score.py --beam_size K
  ✅ euler_beam (--n_runs=K)       → score.py --beam_size K
  ✅ greedy_edit                   → score.py --beam_size 1
  ✅ beam_edit (--beam_size=K)     → score.py --beam_size K

⚠️ 已知问题:
  1/(position+1) 权重假设 beam search (位置0最优)，对 Euler/Euler-Beam
  的独立采样不完全适用 (位置任意). 但跨 augmentation 频次仍主导排序.
```

### 3.3 `visualize_first_step.py` — 首步静态分析

```
机制:
  取 x₀ (产物序列, 未编辑)
  对每个 t ∈ time_grid:
    model(x₀, t) → 预测的 rates 和 probs
    oracle(x₀, target) → 最优编辑
    deterministic top-1 → ACTUAL 行

表格:
  x₀ row: 产物 token 序列 (上边框标记 oracle 编辑类型)
  ORACLE: λ_ins/sub/del + top5 token
  MODEL:  λ_ins/sub/del + top5 token
  ACTUAL: 确定性 top-1 编辑 (>1e-2 阈值)

适配:
  所有采样器 — 因为 first_step 独立于采样器, 只依赖 model.forward()
  ⚠️ 只能看到单步静态预测, 无法反映多步编辑过程
```

### 3.4 `visualize_trajectory.py` — 完整轨迹可视化

```
机制:
  选择采样器 → 跑完整个采样过程 → 记录每步编辑事件
  只展示 best sample 的事件序列

表格 (每个 event):
  x_t row: 当前 token 序列 (标记 origin/oracle/actual)
  ORACLE: 当前状态下 oracle 认为该怎么编辑
  MODEL:  当前状态下模型预测的 rates 和 probs
  ACTUAL: 实际触发的编辑 (随机采样结果)

支持的采样器:
  ✅ euler (--n_samples)        — sample_euler() + event recording
  ⚠️ euler_beam (--n_branches)  — sample_euler_beam() 但 event recording 未实现
  ❌ greedy_edit                — 未集成
  ❌ beam_edit                  — 未集成
```

---

## 4. 完整调用链

```
                 ┌──────────────────────────────────────────┐
                 │           sample_retro.py                │
                 │  --sampler euler/euler_beam/greedy/beam  │
                 └──────────────┬───────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┬─────────────────┐
              ▼                 ▼                  ▼                 ▼
        sample_euler()   sample_euler_beam()  sample_greedy()  sample_beam()
        (euler.py)       (euler_beam.py)      (beam.py)        (beam.py)
              │                 │                  │                 │
              │                 │                  │                 │
    ┌─────────┴─────────┐      │           ┌──────┴──────┐   ┌──────┴──────┐
    │ 共享底层:          │      │           │ 共享底层:    │   │ 共享底层:    │
    │ model.forward()   │      │           │ model.forward│   │ model.forward│
    │ _sample_edit_     │      │           │ _build_log_  │   │ _build_log_  │
    │   actions()       │      │           │   u_edit()   │   │   u_edit()   │
    │ apply_ins_del_    │      │           │ _select_top_ │   │ _select_top_ │
    │   operations()    │      │           │   edits()    │   │   edits()    │
    │ get_adaptive_h()  │      │           │ TimePolicy   │   │ TimePolicy   │
    └───────────────────┘      │           └──────────────┘   └──────────────┘
                               │
                               │ (额外: 去重/排序/剪枝/分裂)
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
      predictions.txt                   score_#global#.py
      (SMILES, 每行一个)                 (Top-N ACC)
              │
              ▼
      visualize_trajectory.py
      (HTML 可视化, 仅 euler/euler_beam)
```

---

## 5. 时间推进方式对比

```
Euler / Euler-Beam:
  t=0 ──adapt_h──→ t₁ ──adapt_h──→ t₂ ──...──→ t≥1.0 停止
  adapt_h = min(1/n_steps, (1-κ)/κ')
  每个分支基于自己的 x_t 和 t 调用 model

Greedy / Beam:
  κ₀ ──TimePolicy──→ κ₁ ──TimePolicy──→ κ₂ ──...──→ STOP
  模型看到的是 scheduler.inverse(κ)
  TimePolicy 基于 U (总编辑量) 反馈决定下一步 κ
```

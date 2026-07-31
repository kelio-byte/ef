# TimePolicy：Beam Search 时间调度抽象

## 1. 动机

在 `docs/beam-search/exp4.md` 确认时间 mismatch 假说之后，`todo4.md` 提出了自适应时间的方向 A（`utot_progress`），后续又讨论了 κ-based 迭代方案。原有的 `time_mode` 通过 if-else 分支部署在采样循环中，新增策略会导致逻辑散落。因此将时间调度抽象为 `TimePolicy` 接口，与 `KappaScheduler` 的设计模式一致——新增策略只需写一个子类，不触碰采样循环。

## 2. 接口设计

```python
class TimePolicy(ABC):
    def reset(self, batch_size: int, device: torch.device, max_edits: int) -> None: ...
    def get_kappa(self, step: int) -> Tensor:                             # → (B, 1) κ
    def update(self, kappa: Tensor, u_tot_base: Tensor) -> Tensor:        # → (B,) stop bool
    def clone(self) -> "TimePolicy": ...
    def state_key(self) -> tuple: ...
```

接口在 κ-空间统一：`get_kappa` 返回 κ，`update` 消费同一个 κ。调用侧负责 κ → t 的转换。

调用时序（与采样循环的交错关系）：

```
reset(B)                                # 采样开始前一次
for step in range(max_edits):
    kappa = policy.get_kappa(step)      # ① 取 κ（在 model forward 之前）
    t = scheduler.inverse(kappa)        # κ → t
    t_model = compute_model_time(t)
    log_rates, ... = model(x_t, t_model)
    u_tot_base = compute_u_tot(...)
    policy_stop = policy.update(        # ② 同一 κ 喂回，拿到停止信号
        kappa.squeeze(-1), u_tot_base)
    # ... stop check + edit selection ...
```

关键设计决定：

- **`get_kappa` 和 `update` 分开** — `u_tot_base` 只有 model forward 之后才能拿到，所以 `get_kappa(step)` 使用的是上一步 `update()` 累积的状态
- **接口统一用 κ** — 所有策略的 `get_kappa` 返回 κ、`update` 接收 κ。调用侧用 `scheduler.inverse(κ)` 转 t 给模型，不经过 `scheduler(t)` 的 round-trip
- **State-aware 策略在 κ-空间直接运算** — Ratio/Kappa 的 `get_kappa(≥1)` 直接返回内部 κ，无需 `scheduler.inverse`。`scheduler` 仅在初始化时用于将 depth_t 转为初始 κ
- **停止信号** — `update()` 返回 `(B,)` bool。不想管的策略返回全 False，由外部 `stop_u_tot_base` 阈值兜底；有理论停止条件的策略（如 Kappa）直接返回
- **支持克隆与去重** — `clone()` 用于 beam 扩展时复制 policy 内部状态；`state_key()` 提供可哈希状态快照，供 beam dedup 区分“token 相同但时间状态不同”的 hypothesis

## 3. 四个实现

### 3.1 DepthTimePolicy

对应原 `time_mode="depth"`。所有样本相同 κ。

```
构造参数: scheduler  (用于 t → κ)
get_kappa(step) → t = (step + 1) / (max_edits + 1)
                   return scheduler(t)
update()        → return all False
```

### 3.2 FixedTimePolicy

对应原 `time_mode="fixed"`。所有样本相同 κ。

```
构造参数: scheduler, time_const (default 0.5)
get_kappa(step) → return scheduler(time_const)
update()        → return all False
```

### 3.3 RatioTimePolicy

对应 `todo4.md` 方向 A（`utot_progress`）。用模型速率总量的下降比例估计编辑进度。

```
构造参数: scheduler  (仅用于 step 0-1 的初始 κ)
reset()   → u_init = u_prev = zeros(B,)

get_kappa(0)  → depth κ (= scheduler(depth_t))
get_kappa(1)  → depth κ (step 1 时 u_prev == u_init，ratio 仍为 1，避免 κ=0)
get_kappa(≥2) → κ = clamp(1 - u_prev / u_init, 1e-8, 1)
                 (直接在 κ-空间返回，无需 scheduler)

update()    → 首次调用: u_init = u_tot_base; 每次: u_prev = u_tot_base
              return all False  (停止由外部 stop_u_tot_base 控制)
```

**为什么前两步用 depth κ**：step 0 的 `u_tot_base` 是第一个 edit 之前的状态，step 1 的 `get_kappa` 消费的是 step 0 的 `u_prev`，此时 ratio = 1.0（编辑尚未体现在速率下降中，κ = 0 对模型是 OOD）。到 step 2 时，step 1 的 model forward 已经看到了 post-edit 状态，`u_prev` 开始反映真实进度。

### 3.4 KappaTimePolicy

用户的 κ-based 迭代方案。利用 flow 数学结构：`(1-κ)` 代表剩余概率质量，`u_tot_base` 代表剩余编辑需求，假设两者以相同速率衰减。

```
构造参数: scheduler  (仅用于 step 0 的初始 κ)
reset()   → kappa_cur = zeros(B,)

get_kappa(0)  → return scheduler(depth_t)  (同时存 kappa_cur)
get_kappa(≥1) → return kappa_cur            (直接在 κ-空间返回)

update()    → kappa_next = 1 - (1 - kappa) * (u_tot - 1) / u_tot
              kappa_cur = clamp(kappa_next, 0, 1)
              return u_tot_base < 1.0  ← 理论停止条件
```

**迭代公式推导**（详见对话记录）：

```
(1 - κ') / (1 - κ) = (u_tot - 1) / u_tot
→ κ' = 1 - (1 - κ) × (u_tot - 1) / u_tot
```

其中 `u_tot - 1` 假设了一次编辑恰好贡献 1 个单位的速率下降。在比值形式下 `(u_tot-1)/u_tot` 是一个相对度量，u_tot 的绝对偏差被部分缓解。

**停止条件**：`u_tot < 1` 时模型估计剩余编辑不足一次，自然停止。

**Type B 错误的自动处理**：若错误编辑导致 u_tot 飙升（如 5→16），则 `(u_tot-1)/u_tot ≈ 0.94`，κ 几乎原地踏步，相当于自动"倒车"给模型更多早期信号去修复。

## 4. 与 Beam Search 的交互

当前实现中，**adaptive TimePolicy 在 beam 中是 hypothesis 级状态**，而不是 sample 级共享状态。

原因：`ratio` / `kappa` 这类策略会随着 `u_tot_base` 演化；同一个 sample 的不同 beam state 可能已经走到完全不同的编辑阶段。如果仍按 sample 共享一份 policy，就会出现：

- 一个接近完成的分支把 κ 推到很晚，拖着其他未完成分支一起“变晚”
- 或一个分支触发停止，导致同 sample 的其他 beam 也被一起停掉

现在的做法是：

1. 初始 beam state 各自持有一份 `time_policy.clone()`
2. 每份 policy 在创建时 `reset(batch_size=1, ...)`
3. `get_kappa(step)` / `update(...)` 都按 active state 单独调用
4. 子节点继承父节点 `update()` 之后的 policy 状态（再 `clone()` 一份）

调用示意：

```python
for state in active_flat:
    kappa = state.time_policy.get_kappa(step)      # (1, 1)
    t = scheduler.inverse(kappa)
    ...
    stop = state.time_policy.update(
        kappa.squeeze(-1), u_tot_base_for_state
    )                                              # (1,)

    for cand in candidates:
        child = BeamState(
            ...,
            time_policy=state.time_policy.clone(),
        )
```

这样 beam 内每条 hypothesis 的时间轨迹都能独立分叉，和其自身的编辑轨迹保持一致。

### 4.1 Beam 去重语义

由于 `origin_mask` 和 `time_policy` 都会影响后续 model forward，beam dedup 不能只看 token 序列。当前 dedup key 包含：

1. token 序列
2. `origin_mask`
3. `time_policy.state_key()`
4. `last_edit`（影响 reverse-op 过滤）
5. `is_finished`

这避免了“同一 `x_t` 但不同编辑历史 / 不同时间状态”的 hypothesis 被错误合并。

## 5. 文件变更

| 文件 | 操作 | 说明 |
|------|:----:|------|
| `edit_flows/sampling/time_policy.py` | **新建** | ABC + 4 个实现 |
| `edit_flows/sampling/beam.py` | 修改 | 删除 `_depth_time_value`，`time_mode`/`time_const` → `time_policy`，`get_t` → `get_kappa` + `scheduler.inverse`，beam 中改为 hypothesis 级 policy 状态，并将 dedup key 扩展到 `origin_mask` / `time_policy` / `last_edit` |
| `edit_flows/sampling/__init__.py` | 修改 | 导出 TimePolicy 类 |
| `scripts/sample_retro.py` | 修改 | `--time_mode` → `--time_policy`（choices: depth/fixed/ratio/kappa） |
| `scripts/oracle_greedy.py` | 修改 | 使用 `FixedTimePolicy` |
| `experiments/trajectory_diag/run_trajectory_diag.py` | 修改 | 内联 depth_t 替代已删除的 `_depth_time_value` |
| `experiments/exp3_beam_d/run_exp_d.py` | 修改 | `--time_mode` → `--time_policy` |
| `tests/sampling/test_beam.py` | 修改 | 使用 `FixedTimePolicy`；新增 adaptive policy 分叉与 origin-mask-aware dedup 回归测试（35/35 通过） |

## 6. 使用方式

### CLI（sample_retro.py）

```bash
# depth 模式（默认，行为不变）
python scripts/sample_retro.py ... --time_policy depth

# fixed 模式
python scripts/sample_retro.py ... --time_policy fixed --time_const 0.5

# 方向 A：ratio 自适应
python scripts/sample_retro.py ... --time_policy ratio

# κ-based 迭代自适应
python scripts/sample_retro.py ... --time_policy kappa
```

### Python API

```python
from edit_flows.sampling.time_policy import KappaTimePolicy
from edit_flows.core.scheduler import CubicScheduler

scheduler = CubicScheduler()
policy = KappaTimePolicy(scheduler=scheduler)
result = sample_greedy_single_edit(
    model, x_0, scheduler, time_policy=policy, ...
)
```

### 新增策略

继承 `TimePolicy`，实现 `reset` / `get_kappa` / `update` / `state_key` 四个方法即可。若策略需要在 beam 中使用，内部状态必须能被 `clone()` 正确复制。

## 7. 设计注意事项

1. **`time_policy` 是必选参数**，位于 `scheduler` 之后、`max_edits` 之前，无默认值。调用方必须显式传入。
2. **所有策略构造函数均需 `scheduler`**。Depth/Fixed 用于 `get_kappa` 中 t → κ 的转换；Ratio/Kappa 仅用于 step 0（或 step 0-1）的初始 κ 计算，后续步直接在 κ-空间运算。
3. **beam 中 state-aware policy 是 per-hypothesis，不是 per-sample**。只有 `depth/fixed` 这类无状态策略，sample 级 / hypothesis 级才等价。
4. **`policy_stop` 和 `stop_u_tot_base` 是 OR 关系**：任一触发即停止。对于 Kappa 策略推荐 `stop_u_tot_base=-1`（完全由内部 `u_tot<1` 控制）。
5. **κ 语义独立于 scheduler 类型**：cubic/linear 的 t 尺度不同，但 κ 的含义一致（= scheduler(t)）。`update` 接收的 κ 就是 `get_kappa` 返回的同一个值，无 round-trip 误差。
6. 诊断脚本（`edit_ranking_diag*.py`、`p1_time_mismatch/run_p1.py`）保留了自己的 depth_t 计算，未接入 TimePolicy，不影响功能。

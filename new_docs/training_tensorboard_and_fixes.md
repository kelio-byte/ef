# 训练基础修复与 TensorBoard 监控

日期：2026-08-06

本文记录训练基础设施阶段的实现、验证结果和使用方式。目标是修复已审计的训练
代码问题，并让下一次重训可以观察 loss、三类编辑速率（lambda）和验证曲线。当前
`checkpoint_step600000.pt` 与 `configs/retro.yaml` 不会被改写；新的配置放在
`configs/retro_v2.yaml`，因此历史推理实验仍可复现。

## 1. 已完成的修复

| 问题 | 修改 | 结果 |
|---|---|---|
| Noam 第一次 update 顺序错误 | `scripts/train_retro.py` 将优化器初始 lr 设为 0，并在每个 `optimizer.step()` 前调用 `NoamScheduler.step()` | update (n) 使用 Noam 的 step-(n) 学习率，不再先使用 Adam 默认 `1e-3` |
| checkpoint 的 step 语义不清 | 同时保存兼容字段 `step` 和明确字段 `completed_steps`，resume 从下一次 update 开始 | 不重复 checkpoint 已完成的 update |
| epoch 中途 resume 会重新生成 shuffle permutation | 新增可按 `seed + epoch` 生成 permutation、按 batch offset 恢复的 `EpochRandomSampler` | 不因 DataLoader 重新迭代而跳过/重复剩余样本；支持 worker prefetch |
| 用不同 batch size 误 resume | checkpoint 记录每 epoch 的有效 batch 数，恢复时校验 | 防止悄悄改变数据顺序；需要新 batch size 时应新建 run |
| RNG 未保存 | checkpoint 保存 Python、NumPy、PyTorch CPU/CUDA 和 DataLoader generator 状态 | 固定 seed 的训练可以进行可重复的断点恢复 |
| raw 文件 `zip()` 静默截断 | `dataset.py`、`precompute_alignments.py` 使用 `zip_longest` 并在行数不一致时报错 | 数据损坏尽早暴露 |
| fallback alignment 将 PAD 当真实 token | DP 前剥离每条序列的 PAD，再将不同样本的 alignment 重新 PAD | batch 内不同长度不再污染 alignment，也不会因 `stack` 长度不同崩溃 |
| 没有 validation/最佳模型 | 增加可配置 validation loop、保存 `checkpoint_best.pt` | 能观察 train/validation gap，并按 validation loss 选模型 |

### Noam 的具体语义

Noam 学习率为

```text
lr(n) = factor * d_model^(-1/2)
        * min(n^(-1/2), n * warmup_steps^(-3/2))
```

新循环的顺序是：

```text
lr_scheduler.step()       # 设置 update n 的 lr
prepare_batch / forward
backward
optimizer.step()          # 使用刚设置的 lr(n)
completed_steps = n
```

旧 checkpoint 仍然保留原来的权重；这项修复只影响使用新脚本重新训练或继续训练时的
后续优化轨迹，不能把旧 checkpoint 的第一次 update 追溯改正。

## 2. TensorBoard 记录内容

`train_retro.py` 从配置读取 `retro.tensorboard`。每次达到 `log_interval` 时写入：

- `train/loss`、`validation/loss`：与训练目标相同的 Bregman loss；
- `*/u_tot`、`*/u_ins`、`*/u_del`、`*/u_sub`：总编辑率和三类操作率；
- `*/lambda/total`、`*/lambda/insert`、`*/lambda/delete`、`*/lambda/substitute`：明确标记为 lambda 的同一组 batch-mean rate；
- `*/lambda_fraction/{insert,delete,substitute}`：三类 lambda 占总 lambda 的比例；
- `*/schedule/t_mean`、`*/schedule/kappa_mean`：本批次采样的时间和 scheduler 后的 κ；
- `*/schedule/rate_scale_mean`、`*/schedule/rate_scale_max`：loss 使用的 κ 相关 rate scale，可观察末端截断是否频繁命中；
- `train/learning_rate`：每个记录点实际使用的 Noam 学习率；
- `run/config`：本次运行写入的完整 YAML。

这里的 lambda 是模型输出的每个位置 insert/substitute/delete rate 在序列维度求和
后的 batch 均值；它不是额外的可学习参数。`u_*` 是兼容旧日志的命名，`lambda/*`
是更直观的 TensorBoard 别名。

## 3. 配置文件

`configs/retro.yaml` 作为历史基线保持不变。`configs/retro_v2.yaml` 保持相同模型、
数据、scheduler 和 loss 设置，新增：

```yaml
seed: 42
num_workers: 2
checkpoint_interval: 10000
keep_checkpoints: 10
save_best_checkpoint: true

tensorboard:
  enabled: true
  log_dir: tensorboard
  log_interval: 100
  validation_start_step: 100000
  validation_interval: 20000
  validation_batches: null   # 完整 validation split
  flush_interval: 500
  flush_secs: 30
```

当前推荐的正式训练策略是从 100,000 steps 开始、每 20,000 steps 验证一次。这样在
600,000-step 训练中约验证 26 次；3090 上完整 validation 每次约 39 秒，总开销约
17 分钟。短 pilot 可以把 `validation_start_step` 改小，并把 `validation_batches` 改为
100，以便更快检查曲线。

正式重训前，可以先将 `total_steps` 改成很小的值做 pilot；确认曲线、显存和吞吐正常
后再恢复 `600000`。这种 pilot 应使用新 checkpoint 目录，不覆盖已有实验结果。

## 4. 推荐指令

在仓库根目录执行：

```bash
# 如果环境尚未安装项目依赖
python -m pip install -e .

# 使用修复后的训练脚本和 TensorBoard 配置
PYTHONPATH=. python scripts/train_retro.py \
  --config configs/retro_v2.yaml \
  --device cuda \
  --save_dir checkpoints/retro_v2

# 另开终端查看曲线
tensorboard --logdir checkpoints/retro_v2 --port 6006
```

脚本实际保存目录为 `checkpoints/retro_v2/<dataset-name>/<timestamp>/`，终端会打印
确切的 TensorBoard 目录和 checkpoint 路径。恢复某个 checkpoint：

```bash
PYTHONPATH=. python scripts/train_retro.py \
  --config configs/retro_v2.yaml \
  --checkpoint checkpoints/retro_v2/<dataset>/<timestamp>/checkpoint_stepN.pt \
  --device cuda
```

如果只想做快速 pilot，将 `tensorboard.validation_batches` 改为 10～100；正式比较时
建议保持 `null`，使用完整 validation。`validation_start_step` 可以控制何时开始验证，
设为 0 表示从第一个满足 interval 的 update 开始。

## 5. 预期效果与边界

这些改动主要修复优化和可观测性，不改变 Transformer 结构、Bregman loss、训练数据或
历史 checkpoint，因此不能直接承诺 Top-k 准确率提升。预期能观察到：

1. warmup 前几步的 loss/参数轨迹不再受到一次 `1e-3` 异常更新污染；
2. `train/learning_rate` 从 Noam step-1 开始平滑 warmup，再进入衰减；
3. lambda 三个分项和占比可以揭示插入偏置、delete/substitute head 是否退化；
4. validation loss 与 train loss 的分离可以判断过拟合和最佳停止点；
5. `rate_scale_max` 可以确认 cubic scheduler 末端的数值截断是否频繁发生。

在同一设备/软件栈下，固定 seed 可以恢复相同的采样顺序和 RNG 状态；GPU 底层若使用
非确定性 kernel，仍可能存在极小的 bitwise 差异。TensorBoard 本身不会改变模型参数。
新训练是否比旧 checkpoint 更好，仍需要在固定的
tiny/mini/full test protocol 上比较 Top-1～10、invalid rate、运行时间和显存。

## 6. 验证记录

- 针对性单测：28 项通过，覆盖 Noam、训练指标、alignment、数据 fail-fast 和可恢复 sampler；
- 2～3 步真实 CPU smoke test：TensorBoard event、`checkpoint_best.pt`、`completed_steps`、RNG 和 scheduler state 均生成正确；
- 固定 seed 的 3-step 对照：完整连续训练与“2-step 保存后 resume”得到逐项相同的模型和 Adam 状态；
- TensorBoard tag smoke test：确认 train/validation 的 loss、lambda 和 schedule 标签存在；
- 全量 pytest 目前有 17 个与本轮无关的既有 `sampling/beam.py` 测试失败，原因是测试仍使用 `EditCandidate(log_u_real=...)`，而当前类字段为 `log_u`；训练相关和其他 218 项通过。本轮没有修改 beam 采样模块。

## 7. 阶段状态

- [x] Noam 第一次 update 顺序修复
- [x] RNG、completed-step、epoch 内 sampler 的 resume 修复
- [x] 数据行数与 alignment 边界检查
- [x] validation 与最佳 checkpoint
- [x] TensorBoard 配置、lambda/loss/schedule 监控
- [x] validation 延迟到 100k steps，并将 interval 调整为 20k
- [ ] 用 `retro_v2.yaml` 做正式 10k～30k pilot，并与旧 checkpoint 在固定 protocol 上比较
- [ ] pilot 通过后再决定是否启动完整 600k 重训

## 8. 新 600k checkpoint 的 provenance（2026-08-07）

A6000 上完成的 `new_checkpoints/checkpoint_step600000.pt` 使用 `configs/retro_v2.yaml`
对应的结构和目标设置。与历史 checkpoint 相比，真正改变训练轨迹的是：

- Noam 学习率在每个 optimizer update 前设置，修复第一次 update 误用 Adam 默认 `1e-3`
  的顺序问题；
- seed=42、按 epoch 派生的 deterministic sampler，以及 Python/NumPy/PyTorch/DataLoader
  RNG 的 checkpoint 保存和恢复。

validation、TensorBoard、时间戳日志、best checkpoint 和数据 fail-fast 是可观测性/恢复性
改动；本次训练使用预对齐数据，fallback alignment 修复没有证据作用于这次训练。模型结构、
dropout=0.3、cubic scheduler、Bregman loss、`use_origin_mask=False` 和 600k steps 均未改变。
因此新模型应作为独立 checkpoint 评估，不能把旧模型参数或旧 tiny 最优采样参数视作自动
适配。

validation-200 的参数消融和 mini-1001 冻结规则记录在
[`new_checkpoint_validation_parameter_sweep.md`](new_checkpoint_validation_parameter_sweep.md)。

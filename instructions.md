# Edit-Flows 项目使用说明

本文档集中记录本项目当前推荐的命令行用法。新实验优先使用这里的
`train_retro.py`、`sample_retro.py`、`eval.py` 和 `score_#global#.py`；`old_*`、
`recover_*` 等脚本只用于复现历史结果，不能与当前实现混用。

## 0. 环境和路径

项目当前使用 Conda 环境 `ef`。建议每次打开终端后先激活并设置变量：

```bash
cd /root/autodl-tmp/edit_flows
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ef
export PY=python
export DATA='datasets/USPTO_50K_PtoR_aug20_#global#'
```

新机器的 Conda 安装路径如果不同，用 `conda info --base` 找到对应的 `etc/profile.d/conda.sh`。

常用数据文件：

|用途|文件|规模|
|---|---|---:|
|训练集|`$DATA/train/src-train.txt`、`tgt-train.txt`|约 800k 行（含 augmentation）|
|验证集|`$DATA/val/src-val.txt`、`tgt-val.txt`|100020 行|
|完整测试集|`$DATA/test/src-test.txt`、`tgt-test.txt`|100140 行 = 5007 个反应 × 20|
|tiny 测试|`$DATA/test/src-test-tiny.txt`、`tgt-test-tiny.txt`|1000 行|
|mini 测试|`$DATA/test/src-test-mini-1001.txt`、`tgt-test-mini-1001.txt`|20020 行 = 1001 个反应 × 20|

如果 mini 文件名在本地不同，以 `ls "$DATA/test"` 查到的实际文件名为准。

## 1. AutoDL GPU 保活脚本

机器在 GPU 连续 60 分钟没有使用时可能自动关机。项目根目录的 `alive.py` 会构造一个随机的
多层感知机，在 GPU 上反复执行 forward/backward/optimizer step：

```python
batch_size = 1024
```

它会真实占用显存和算力，并不是无负载的心跳。因此：

```bash
# 不跑实验时启动
$PY alive.py

# 需要跑实验时，在该终端按 Ctrl+C 停止
nvidia-smi                 # 另一个终端确认 GPU 已释放
```

如果出现显存不足，可先停止实验进程，再把 `alive.py` 中的 `batch_size` 临时调小；不要在正式
采样或训练时同时运行它，否则会争抢显存和计算资源，造成速度下降甚至 OOM。脚本没有命令行
参数，batch size 目前需要直接编辑文件。`alive.py` 是本地保活脚本；`requirements.txt` 是仓库中的
依赖清单，可用于在另一台机器重建 Python 环境。

## 2. 训练

### 2.1 当前训练配置

推荐配置文件是 `configs/retro_v2.yaml`。当前 TensorBoard/validation 设置为：

- TensorBoard 开启，默认每 100 step 记录一次训练指标；
- 从完成 100000 step 后才开始 validation；
- validation 间隔 20000 step；
- `validation_batches: null` 表示完整验证集；
- 600k step 训练期间大约会进行 26 次完整 validation。

validation 的数据来自配置中指定的 validation split（当前为 `$DATA/val`），不是 test 集。若只想
通过训练 loss 判断，可在 yaml 中把 `tensorboard.validation_start_step` 或
`tensorboard.validation_interval` 设为 `null`/禁用对应 validation 配置；训练仍会记录 loss。

### 2.2 从头训练

```bash
$PY scripts/train_retro.py \
  --config configs/retro_v2.yaml \
  --device cuda \
  --save_dir checkpoints/retro_v2
```

### 2.3 20k 训练 smoke test（推荐先运行）

`configs/retro_v2_smoke_20k.yaml` 保留历史模型/目标函数设置，但将总步数设为 20000，并在
5000、10000、15000、20000 step 做 20 个 validation batches。它用于先确认数据加载、Noam
scheduler、loss、checkpoint、TensorBoard 和 validation 链路都正常，不代表最终模型质量。

```bash
$PY scripts/train_retro.py \
  --config configs/retro_v2_smoke_20k.yaml \
  --device cuda \
  --save_dir checkpoints/retro_v2_smoke_20k
```

由于 `--save_dir` 会再拼接数据集名和时间戳，实际输出位于
`checkpoints/retro_v2_smoke_20k/<dataset-name>/<timestamp>/`。完成后应检查 `train.log`、
`checkpoint_step20000.pt`、`checkpoint_best.pt`（若 validation loss 有改善）以及 TensorBoard
event 文件。终端和 `train.log` 中的每条逻辑日志行都会带本地时间戳，格式为
`[MM/DD/HH/MM]`，例如 `[08/06/14/09] step ...`。

### 2.4 从 checkpoint 继续训练

```bash
$PY scripts/train_retro.py \
  --config configs/retro_v2.yaml \
  --checkpoint checkpoints/retro_v2/checkpoint_step100000.pt \
  --device cuda \
  --save_dir checkpoints/retro_v2_resume
```

`--checkpoint` 会恢复模型、optimizer、scheduler、global step 及随机状态；不要把只保存模型权重
的旧文件误当成可无缝 resume 的 checkpoint。训练代码还会保存 `checkpoint_latest.pt` 和按 step
命名的 checkpoint（具体保留数量由 `--keep_checkpoints`/yaml 决定）。

### 2.5 预计算 alignment（通常只需做一次）

```bash
$PY scripts/precompute_alignments.py \
  --data_dir "$DATA" \
  --splits train val \
  --num_workers 32
```

运行前先检查 alignment 文件是否已经存在；已有文件无需重复计算。

## 3. TensorBoard

训练时日志目录由 yaml 的 `tensorboard.log_dir` 决定（通常为 `tensorboard/`）。启动：

```bash
$PY -m tensorboard.main --logdir tensorboard --port 6006 --host 0.0.0.0
```

浏览器打开转发后的 6006 端口。重点关注：

- `train/loss`、`train/loss_*`：训练损失及分项；
- `train/lambda`、时间/噪声相关统计：模型对离散编辑强度的学习；
- `validation/*`：validation loss 和分项（只在 validation 到期时出现）；
- step、学习率和耗时：检查 scheduler 顺序、吞吐和异常停顿。

## 4. 统一推理和打分（推荐）

`scripts/eval.py` 将采样和 `score_#global#.py` 合并到一次调用中，自动读取采样 metadata，避免手动
填写错误的 `beam_size`。必须提供 checkpoint、products、targets 和输出目录。

### 4.1 Euler-Beam：R3K3M2（当前常用配置）

```bash
$PY scripts/eval.py \
  --checkpoint checkpoint_step600000.pt \
  --products_file "$DATA/test/src-test-tiny.txt" \
  --targets "$DATA/test/tgt-test-tiny.txt" \
  --augmentation 20 \
  --output_dir results/eval_tiny_r3k3m2 \
  --sampler euler_beam \
  --n_branches 3 --n_children 2 --n_runs 3 \
  --n_steps 100 --batch_size 64 --device cuda --seed 42 \
  --n_best 10
```

Euler-Beam 最终每个原始产物输出 `n_runs × n_branches` 条候选；`n_children` 是每个父分支在每个
step 内生成的后继数，只影响内部扩展和剪枝，不直接决定最终输出条数。因此 R3K3M2 最终是 9
条候选，最多只能报告 Top-9；若要真正报告 Top-10，应使用例如 R1K10 或 R2K5。

### 4.2 Euler：10 条独立样本

```bash
$PY scripts/eval.py \
  --checkpoint checkpoint_step600000.pt \
  --products_file "$DATA/test/src-test-tiny.txt" \
  --targets "$DATA/test/tgt-test-tiny.txt" \
  --augmentation 20 \
  --output_dir results/eval_tiny_euler10 \
  --sampler euler --n_samples 10 \
  --n_steps 100 --batch_size 16 --device cuda --seed 42 \
  --n_best 10
```

### 4.3 mini 或完整 test

把上面命令中的两个输入文件和输出目录换成目标 split 即可。例如 mini：

```bash
$PY scripts/eval.py \
  --checkpoint checkpoint_step600000.pt \
  --products_file "$DATA/test/src-test-mini-1001.txt" \
  --targets "$DATA/test/tgt-test-mini-1001.txt" \
  --augmentation 20 \
  --output_dir results/eval_mini_r3k3m2 \
  --sampler euler_beam --n_branches 3 --n_children 2 --n_runs 3 \
  --n_steps 100 --batch_size 64 --device cuda --seed 42 --n_best 10
```

完整 test 使用 `src-test.txt`/`tgt-test.txt`；正式长实验前建议先在 tiny 或 mini 上确认参数、显存和
输出格式。

### 4.4 常用 eval 选项

- `--start_product`、`--max_products`：只处理指定的原始产物区间，便于短实验；
- `--overwrite`：允许覆盖已有输出目录，默认会避免误覆盖；
- `--score_only`：跳过采样，读取输出目录中的 `predictions.txt`/metadata 直接打分；
- `--dry_run`：只打印内部采样和打分命令，不真正运行；
- `--n_best`：报告的最高 K（候选不足时只能报告实际候选数）；
- `--aggregation_mode`：`legacy_best_rank`（历史口径）、`rrf`、`frequency_first` 或 `hybrid`；
- `--diagnostics`、`--diagnostics_json`：输出 coverage、invalid rate、多样性和跨 run 重叠等诊断；
- `--process_number`：打分阶段的 CPU 进程数，过大可能与采样抢 CPU；
- `--save_file`、`--save_accurate_indices`：保存详细结果及命中样本索引。

### 4.5 DGM / Guidance 对照（当前仅 ordinary Euler）

`eval.py` 可以把独立 guidance checkpoint 透传给 `sample_retro.py`：

```bash
$PY scripts/eval.py \
  --checkpoint new_checkpoints/checkpoint_step600000.pt \
  --products_file "$DATA/val/src-val.txt" \
  --targets "$DATA/val/tgt-val.txt" \
  --output_dir results/dgm_val200_beta010 \
  --sampler euler --n_samples 3 --n_steps 100 \
  --batch_size 64 --device cuda --seed 42 \
  --start_product 4000 --max_products 4000 --augmentation 20 \
  --guidance_checkpoint /path/to/guidance_best.pt \
  --guidance_beta 0.10 --n_best 10
```

先加 `--dry_run` 检查内部命令；上例会自动推导为 validation reaction 200–399、
`target_offset=200`。当前 guidance 只支持 `--sampler euler`，尚未通过综合准确率门槛，默认
推理不要添加这两个参数。guidance 训练使用 `scripts/train_guidance.py`；有 validation 时会
同时保存 `guidance_final.pt` 和按最低 validation loss 选择的 `guidance_best.pt`，正式采样
优先使用后者。训练/数据生成的完整研究协议见 `new_docs/dgm.md`。

## 5. 分离运行采样和打分

需要调试中间文件时使用 `sample_retro.py`：

```bash
$PY scripts/sample_retro.py \
  --checkpoint checkpoint_step600000.pt \
  --products_file "$DATA/test/src-test-tiny.txt" \
  --sampler euler_beam \
  --n_branches 3 --n_children 2 --n_runs 3 \
  --n_steps 100 --batch_size 64 --device cuda --seed 42 \
  --output_dir results/manual_beam_tiny
```

它会生成 `predictions.txt` 和 `sampling_metadata.json`。随后手动打分：

```bash
$PY scripts/score_#global#.py \
  --predictions results/manual_beam_tiny/predictions.txt \
  --targets "$DATA/test/tgt-test-tiny.txt" \
  --augmentation 20 --beam_size 9 --n_best 10 \
  --diagnostics \
  --diagnostics_json results/manual_beam_tiny/diagnostics.json
```

这里 `beam_size` 必须等于每个原始反应最终保留的候选数（R×K），不是 `n_children`。Euler 的
`n_samples=10` 则使用 `--beam_size 10`。

`score_#global#.py` 的主要选项：`--score_alpha`、`--aggregation_mode`、`--length`、
`--target_offset`、`--process_number`、`--synthon`、`--detailed`、`--raw`、`--save_file`、
`--save_accurate_indices`、`--diagnostics`、`--diagnostics_json`。同一组实验必须固定 aggregation
mode、augmentation、beam size 和 n_best，避免指标口径变化。

## 6. Euler-Beam 关键参数和公平比较

`sample_retro.py` 支持的采样器包括 `euler`、`euler_beam`、`greedy_edit`、`beam_edit`。

- `--n_steps`：Euler 时间离散步数；当前常用 100；
- `--batch_size`：模型前向 batch；显存允许时可测试 64、128 等，但比较时必须固定；
- `--n_branches`（K）：每个 run 的 beam 分支数；
- `--n_runs`（R）：独立随机 run 数；最终候选数为 R×K；
- `--n_children`（M）：每个父分支、每一步生成的后继数；M>1 才真正产生扩展后剪枝；
- `--euler_beam_child_policy`：`stochastic` 或 `stochastic_noop`；必须在对比实验中固定；
- `--euler_beam_score_mode`：`full_probability`（当前推荐）或历史模式；
- `--euler_beam_changed_state_bonus`：改变状态的启发式 bonus，不是 reward model，也不是训练得到的 λ；
- `--euler_beam_q_temperature`：Q 分布温度；1.0 表示不改变 checkpoint 的分布；小于 1 会使分布更尖锐；
- `--euler_beam_matmul_precision`：3090 上 `high` 可启用支持的 TF32 matmul，速度更快；`highest` 更接近
  严格 FP32。准确率对比应固定这一项；
- `--euler_beam_share_identical_forwards`：复用相同状态的前向结果，属于效率优化；比较时固定开关；
- `--euler_beam_profile`：同步 CUDA 并记录 profile，仅用于短 profiling，不要用于正式计时。

例如 R1K10M2（10 个最终分支、单次 run）：

```bash
$PY scripts/eval.py \
  --checkpoint checkpoint_step600000.pt \
  --products_file "$DATA/test/src-test-tiny.txt" \
  --targets "$DATA/test/tgt-test-tiny.txt" \
  --augmentation 20 --output_dir results/eval_tiny_r1k10m2 \
  --sampler euler_beam --n_runs 1 --n_branches 10 --n_children 2 \
  --n_steps 100 --batch_size 64 --device cuda --seed 42 --n_best 10
```

每次改变 R/K/M、temperature、bonus 或 child policy 时，保留 seed、checkpoint、数据 split、n_steps、
batch size 和 scoring 口径，并记录运行时间和 GPU 状态。

## 7. trajectory 可视化

### 7.1 Euler 多路径 trajectory

```bash
$PY scripts/visualize_trajectory.py \
  --checkpoint checkpoint_step600000.pt \
  --products_file "$DATA/test/src-test-tiny.txt" \
  --targets_file "$DATA/test/tgt-test-tiny.txt" \
  --n_examples 5 --n_samples 3 --n_steps 100 \
  --seed 42 --device cuda --table False --html True
```

脚本当前是 Euler trajectory 可视化：`--n_samples` 表示每个 example 的独立路径数，非 Euler-Beam
的 R×K；当前实现拒绝非零 `--n_branches`。`--table False` 只展示开头的完整路径变化概览，不展示
逐步表格；`--table True` 会追加每条路径的逐步 Product → +Edit → … → Target 表格。可用
`--example_ids 0,7,25` 固定样本，`--deduplicate` 去重，`--max_lines` 限制输出行数。

输出目录通常为 `visualizations/trajectory-<example_ids>/`，HTML 文件名带时间戳。

### 7.2 第一时间步分析

```bash
$PY scripts/visualize_first_step.py \
  --checkpoint checkpoint_step600000.pt \
  --products_file "$DATA/test/src-test-tiny.txt" \
  --targets_file "$DATA/test/tgt-test-tiny.txt" \
  --output_dir visualizations/first_step_tiny \
  --n_examples 5 --time_grid 0.1,0.3,0.5,0.7,0.9 \
  --seed 42 --device cuda
```

### 7.3 不生成路径的 forward 诊断

```bash
$PY scripts/first_step_forward_analysis.py \
  --checkpoint checkpoint_step600000.pt \
  --products_file "$DATA/test/src-test-tiny.txt" \
  --targets_file "$DATA/test/tgt-test-tiny.txt" \
  --output_dir results/first_step_forward_tiny \
  --max_lines 1000 --batch_size 64 --device cuda
```

## 8. 数据统计和其他诊断

统计反应物中多个 component（例如含 `.` 的输入）：

```bash
$PY scripts/stats_reactant_components.py \
  --targets_file "$DATA/test/tgt-test.txt" \
  --augmentation 20 \
  --json_out results/reactant_component_stats.json
```

编辑排序诊断：

```bash
$PY scripts/edit_ranking_diag_v2.py \
  --checkpoint checkpoint_step600000.pt \
  --data_dir "$DATA" --n_samples 10 --device cuda \
  --output_dir results/edit_ranking_diag
```

调度器比较脚本：

```bash
$PY scripts/compare_schedulers.py
```

`oracle_sample.py`、`oracle_loss_profile.py`、`oracle_trajectory_analysis.py` 是研究用 oracle 工具，
使用前先查看各自的 `--help`，不要把 oracle 结果当作正常 checkpoint 推理指标。

## 9. 运行前检查清单

1. `nvidia-smi` 确认没有 `alive.py` 或其他旧实验占用 GPU；
2. 确认 checkpoint、products、targets 的路径和行数对齐；
3. 先用 `--max_products` 在 tiny/少量样本上做 smoke test；
4. 固定 seed、checkpoint、split、augmentation、R/K/M、n_steps、batch size、TF32 设置和 scoring mode；
5. 记录 `sampling_metadata.json`、diagnostics JSON、总耗时和显存；
6. 长实验结束后再重新启动 `alive.py` 保持机器在线。

## 10. 历史脚本说明

`scripts/old_sample_retro.py`、`scripts/old_score_#global#.py`、`scripts/recover_sample_retro.py` 和
旧的通用 `train.py`/`sample.py`/`score.py` 仅用于历史版本复现。它们的 seed、输出格式、采样逻辑
或打分口径可能与当前实现不同。`scripts/eval_retro.py` 是较早的 Euler-only 端到端封装，不建议用于
Euler-Beam 对比。若必须复现旧指标，应单独建立输出目录并在实验记录中标注“legacy”。

## 11. 常见故障

- **CUDA out of memory**：停止 `alive.py`，降低 `--batch_size`、`--n_branches` 或 `--n_children`；
- **结果数量不对**：检查 Euler 的 `n_samples` 与 Euler-Beam 的 R×K；`n_children` 不等于最终输出数；
- **Top-K 看起来异常**：确认 `--beam_size` 等于最终候选数，且 `--augmentation 20` 与数据文件一致；
- **速度很慢**：确认没有开启 `--euler_beam_profile`，检查 batch size 和 `--euler_beam_share_identical_forwards`，
  并用 `nvidia-smi` 查看 GPU 利用率；
- **想只重新打分**：使用 `eval.py --score_only` 或直接运行 `score_#global#.py`；
- **想检查命令而不运行**：使用 `eval.py --dry_run` 或各脚本的 `--help`。

## 12. 跨机器迁移（包括 A6000）

当前分支已经跟踪项目代码、配置以及 `datasets/USPTO_50K_PtoR_aug20_#global#` 下的训练/验证/测试
文件和 pre-aligned 文件；clone 后不需要再次复制这些数据。模型 checkpoint（`.pt`）、`results/`、
`visualizations/` 和本地 `alive.py` 不属于仓库，需要按需单独传输。

```bash
git clone -b task18-all-branches-eval \
  https://github.com/kelio-byte/ef.git
cd ef
export PY=/path/to/python
$PY -m pip install -r requirements.txt
$PY -m pip install -e .
$PY -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

A6000 使用通用 CUDA 路径，与 3090 可运行同一代码。为了复现实验，首次仍固定
`batch_size=128`、`num_workers=2` 和 `seed=42`；A6000 的额外显存不应直接改变 batch size。跨机器
确认流程后，再用短 smoke test 比较 `num_workers=2/4/8` 的实际 step throughput；worker 数量主要消耗
CPU/RAM，不能仅凭 GPU 显存大小决定。

最后更新：2026-08-08

# SPE 训练的单卡 / 双卡 DDP 说明

## 结论

`scripts/train_retro.py` 现已支持 PyTorch **DistributedDataParallel
(DDP)**。两张 3090 应使用 DDP，不使用 `DataParallel`：每张卡各一个 Python
进程，梯度在每次 update 后同步；训练数据、验证数据、日志和 checkpoint 都按 rank
正确处理。

`DataParallel` 会在一个主进程中复制模型、集中 scatter/gather，官方也明确推荐单机多卡
使用 DDP。这里的模型能完整放进单张 3090，因此不需要 FSDP 或模型并行。

## 选哪个配置

| 可用 GPU | 配置 | 每卡 batch | 有效全局 batch | 启动方式 |
|---|---|---:|---:|---|
| 1 × 3090 | [`retro_spe_600k.yaml`](../configs/retro_spe_600k.yaml) | 128 | 128 | `python` |
| 2 × 3090 | [`retro_spe_600k_ddp2.yaml`](../configs/retro_spe_600k_ddp2.yaml) | 64 | 128 | `torchrun` |

两份配置的模型、数据、loss、Noam factor、warmup 和 **有效全局 batch 都是 128**。因此
双卡版本不是把 batch 盲目翻倍，而是把已有单卡 batch 128 均分为 `64 + 64`，保持本次
600k 的优化协议可比。

现有 SPE-50k pilot 在单张 3090 上已经验证过 batch 128：峰值 CUDA allocated 为
3.44 GB、reserved 为 5.36 GB。因此每卡 64 是保守且安全的起点。若以后只追求吞吐，可
另开实验使用 `128 + 128 = global 256`，但那会改变优化 batch，必须单独验证学习率与
warmup，不能与 global-128 结果混为同一 baseline。

## 启动命令

一张 3090：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
python scripts/train_retro.py \
  --config configs/retro_spe_600k.yaml \
  --device cuda \
  --save_dir checkpoints/retro_spe_600k
```

两张 3090：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
torchrun --standalone --nproc_per_node=2 scripts/train_retro.py \
  --config configs/retro_spe_600k_ddp2.yaml \
  --device cuda \
  --save_dir checkpoints/retro_spe_600k_ddp2
```

`torchrun` 自动提供 `RANK/WORLD_SIZE/LOCAL_RANK`。脚本在 CUDA 下选择 NCCL，并把每个
rank 绑定到对应 GPU；不需要在 YAML 中再写 GPU id。`OMP_NUM_THREADS=4` 也修复了当前
shell 将它设为无效值 `0` 时出现的 `libgomp` 警告。

## 保证的训练语义

1. 训练集用同一 `seed + epoch` permutation，rank 0/1 分别取得交错且不重叠的 shard；
   每个 global update 的样本并集与 global-batch-128 的顺序一致。
2. validation 不 padding、不重复样本；各 rank 的加权总和通过 all-reduce 汇总为一个
   全局 validation 指标。
3. 只有 rank 0 创建 run 目录、写 TensorBoard/log/monitor 和 checkpoint；所有 rank
   在 checkpoint 边界同步。
4. checkpoint 保存未包裹的 `model_state_dict`，所以采样脚本和单卡恢复不会看到
   `module.` 前缀；同时保存每个 rank 的 RNG 状态和 world-size/batch 拓扑。
5. 从单卡 checkpoint 切换到 `2 × 64` 时，脚本允许有效 global batch 相同的继续训练，
   但会明确提示：由于 rank-local 随机数流改变，它不是 bitwise-identical continuation。

## 验证记录（2026-08-14）

- 当前机器确认：1 × RTX 3090 24 GB，PyTorch 2.7.1+cu126，CUDA 与 NCCL 可用。
- 新增 2 进程 CPU/Gloo 端到端测试：两个 rank 完成 2 个 train update、sharded validation、
  best/periodic checkpoint，随后从该 checkpoint 恢复至 step 4；checkpoint 的 topology 为
  `world_size=2`、`2 × batch=2`，且无 `module.` 前缀。
- 新增单张实际 3090 端到端测试：2 个 CUDA update、validation 与 portable checkpoint
  均通过。
- 训练相关回归：26 passed；完整仓库回归（排除需要外部 legacy checkpoint 的 1 项）为
  395 passed、1 deselected。CPU DDP + 单卡 CUDA entrypoint 测试均通过。

当前只有一张卡，因此还没有实际执行双 GPU NCCL collective 的吞吐/显存测量。第二张卡
可用后，应先按双卡命令跑 5 个 update（约 2–5 分钟，检查 `train.log` 中
`world_size=2`、`batch/rank=64`、`effective global batch=128`），再启动 600k。

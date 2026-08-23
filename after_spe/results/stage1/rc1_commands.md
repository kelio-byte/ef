# Stage1 RC1 运行命令

日期：2026-08-24

## 推荐入口

GPU 开机并进入 `ef` 环境后，先运行：

```bash
cd /root/autodl-tmp/edit_flows
conda activate ef
bash scripts/run_stage1_rc1.sh smoke
```

smoke 通过后才运行：

```bash
bash scripts/run_stage1_rc1.sh pilot100
```

确认 100-reaction 结果无明显退化后，运行完整 dev-1000：

```bash
bash scripts/run_stage1_rc1.sh dev1000
```

脚本每次建立带 UTC 时间戳的新目录，不覆盖旧结果。也可用 `OUTPUT_ROOT=/指定目录` 固定输出位置。默认使用当前环境的 `python`、CUDA 第 0 张卡、batch size 32。

## 每个规模会跑什么

| 规模 | reaction | 输入行 | 组别 |
|---|---:|---:|---|
| `smoke` | 10 | 200 | B0、B0-trace、B1、B2 |
| `pilot100` | 100 | 2,000 | B0、B0-trace、B1、B2 |
| `dev1000` | 1,000 | 20,000 | B0、B0-trace、B1、B2 |

B0-trace 用 `max_multiplier=1.0` 收集普通 Euler 的首事件。脚本会逐字节比较 B0 和 B0-trace 的预测；不同则立即失败。正式运行时间以没有诊断开销的 B0 为准，B1/B2 的时间包含 sidecar 与首事件记录开销。

所有组固定为 M500@490K、Euler N=9、100 steps、seed=42、20 augmentation 和相同评分协议。

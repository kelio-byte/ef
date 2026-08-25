# Stage1 RC1 运行命令

日期：2026-08-24

正式 smoke、pilot100、dev-1000 和独立 confirm-1000 已完成；当前结论见 [stage1_report.md](stage1_report.md) 与 [rc1_confirm1000_report.md](rc1_confirm1000_report.md)。以下命令只保留用于复现，不应再以 dev 扫描新的 bias 倍率。

RC1.5 的 dev-1000 与独立 confirm-1000 也已完成，报告见 [rc15_dev1000_report.md](rc15_dev1000_report.md) 和 [rc15_confirm1000_report.md](rc15_confirm1000_report.md)。confirm 未满足端到端覆盖门槛，因此不再运行新的中心采样实验，也不训练 product-only predictor。

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

若要按冻结协议复现独立确认集：

```bash
bash scripts/run_stage1_rc1.sh confirm1000
```

`confirm1000` 使用 `confirm_unique1000_aug20`，而不是 dev；它要求相应的 M500 evaluation split 和 center sidecar 已准备好，脚本会检查 split 名称是否匹配。

脚本每次建立带 UTC 时间戳的新目录，不覆盖旧结果。也可用 `OUTPUT_ROOT=/指定目录` 固定输出位置。脚本会切到项目根目录并设置 `PYTHONPATH`；优先使用 `/root/miniconda3/envs/ef/bin/python`，其次使用 `/root/autodl-tmp/ef/bin/python`，也可用 `PYTHON_BIN=/路径/python` 显式覆盖。默认 CUDA 第 0 张卡、batch size 32。

## 每个规模会跑什么

| 规模 | reaction | 输入行 | 组别 |
|---|---:|---:|---|
| `smoke` | 10 | 200 | B0、B0-trace、B1、B2 |
| `pilot100` | 100 | 2,000 | B0、B0-trace、B1、B2 |
| `dev1000` | 1,000 | 20,000 | B0、B0-trace、B1、B2 |
| `confirm1000` | 1,000 | 20,000 | B0、B0-trace、B1、B2 |

B0-trace 用 `max_multiplier=1.0` 收集普通 Euler 的首事件。脚本会逐字节比较 B0 和 B0-trace 的预测；不同则立即失败。正式运行时间以没有诊断开销的 B0 为准，B1/B2 的时间包含 sidecar 与首事件记录开销。

所有组固定为 M500@490K、Euler N=9、100 steps、seed=42、20 augmentation 和相同评分协议。

## RC1.5 混合轨迹入口

先做新接口的 smoke：

```bash
bash scripts/run_stage1_rc15.sh smoke
```

已完成的 dev 复现命令：

```bash
bash scripts/run_stage1_rc15.sh dev1000
```

该命令只重新生成 RC1.5；它会验证并复用已完成的同协议 B0、B0-trace、B1 输出，避免无意义地重复三遍完整推理。本次提交前的 smoke 已确认新代码的 B0、B0-trace、B1 与旧 RC1 smoke 逐字节一致。

如需逐字复现已完成的独立复核，命令为：

```bash
bash scripts/run_stage1_rc15.sh confirm1000
```

`confirm1000` 使用 `confirm_unique1000_aug20`、其独立 center sidecar 和既有的 confirm B0/B1 输出；脚本会核验输入哈希、采样协议和 B0/B0-trace 的逐字节一致性。不要把该命令用于继续调 checkpoint、倍率、trajectory 数、seed 或步数。

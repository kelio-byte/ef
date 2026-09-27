# 随机 10 个反应的推理轨迹案例

## 选取与文件

从 dev1000 中未参与调参的 800 个反应里，排除先前展示的反应 `0`、`4`、`5`，对其余反应编号排序，再用 `random.Random(20260927).sample(..., 10)` 抽取。选中的是 `99`、`107`、`164`、`316`、`410`、`468`、`524`、`637`、`646`、`882`。反应编号和文件名均**从 0 开始**；每个反应取第 1 个增强视图（dev1000 产品行号为反应编号 × 20）和第 1 次独立采样。

每个独立 HTML 同时显示加速前、加速后的模型调用时间轴、逐步 token 状态和最终结果，文件名为 `反应编号-trajectory.html`。[单页汇总分析与全部 10 例轨迹](index.html)把对照表、观察和完整案例放在同一个 HTML 中；[原始逐步数据](index.json.gz)供核对。两种方法使用同一 checkpoint 和 GPU 子候选选择实现。重跑完整的 32 输入 batch、每输入 9 次采样后，10 例共 20 条最终预测均与已保存的 dev1000 预测逐条一致。

## 逐例对比

按模型调用开始时间 `t` 分为四段：第 1～4 段依次是 `[0,0.25)`、`[0.25,0.50)`、`[0.50,0.75)`、`[0.75,1)`。每格数字都是**加速前 → 加速后**；四段相加等于全程 NFE。“命中”只指这一条轨迹的最终分子与目标一致，不是反应级 Top-k。

| 反应行号（点开案例） | 第1段调用 | 第2段调用 | 第3段调用 | 第4段调用 | 全程 NFE | 单条轨迹结果 |
|---:|---:|---:|---:|---:|---:|---|
| [99](99-trajectory.html) | 26 → 7 | 25 → 6 | 25 → 7 | 24 → 9 | 100 → 29 | 命中 → 命中 |
| [107](107-trajectory.html) | 26 → 7 | 25 → 6 | 25 → 6 | 24 → 9 | 100 → 28 | 命中 → 命中 |
| [164](164-trajectory.html) | 26 → 7 | 25 → 6 | 25 → 6 | 24 → 7 | 100 → 26 | 命中 → 命中 |
| [316](316-trajectory.html) | 26 → 7 | 25 → 6 | 25 → 6 | 24 → 12 | 100 → 31 | 命中 → 命中 |
| [410](410-trajectory.html) | 26 → 7 | 25 → 6 | 25 → 10 | 24 → 37 | 100 → 60 | 未命中 → 未命中 |
| [468](468-trajectory.html) | 26 → 7 | 25 → 6 | 25 → 16 | 24 → 25 | 100 → 54 | 未命中 → 未命中 |
| [524](524-trajectory.html) | 26 → 7 | 25 → 6 | 25 → 6 | 24 → 7 | 100 → 26 | 命中 → 命中 |
| [637](637-trajectory.html) | 26 → 7 | 25 → 6 | 25 → 6 | 24 → 7 | 100 → 26 | 命中 → 命中 |
| [646](646-trajectory.html) | 26 → 7 | 25 → 6 | 25 → 17 | 24 → 20 | 100 → 50 | 未命中 → 未命中 |
| [882](882-trajectory.html) | 26 → 7 | 25 → 6 | 25 → 13 | 24 → 28 | 100 → 54 | 命中 → 未命中（有效分子） |

## 从案例能看到什么

- 10 例加速后都减少了全程模型评估次数：从每例 100 次降至 26～60 次，10 例平均为 38.4 次。这是 **NFE**，不是逐例墙钟加速倍数。
- 其中 9 例加速前后的最终 token 输出完全相同：6 例两边都命中目标，3 例两边都未命中。反应 882 是唯一最终输出不同的案例；加速后产物有效，但这条轨迹未命中目标。
- 第 1、2 段的调用数在 10 例中分别都从 `26→7`、`25→6`。后段则按轨迹变化：例如反应 410 的第 4 段从 24 增到 37 次，反应 882 从 24 增到 28 次。即使后段更密集，单条轨迹仍可能出错。

这 10 例是在查看输出前按固定随机种子抽出的轨迹样本，便于检查机制和失败案例；样本太小，不能用其中的 `6/10` 命中率估计总体质量。总体质量与耗时以 [完整 test 和 dev1000 记录](../../RESULTS.md)为准。轨迹记录会增加同步开销，故这些案例只比较 NFE 和采样路径。

## 复现与校验

本轮**只读取已有的 `index.json.gz` 重绘合并分析页面，没有重新运行模型**：

```bash
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/render_trajectory_analysis.py \
  --input experiments/fastflow_acceleration/case_study/random10/index.json.gz \
  --output experiments/fastflow_acceleration/case_study/random10/index.html --force
```

下列命令是原始逐步数据的完整复现方式，本轮未执行：

```bash
PYTHONPATH=/root/autodl-tmp/efretro /root/autodl-tmp/ef/bin/python scripts/visualize_trajectory.py \
  --indices 1980,2140,3280,6320,8200,9360,10480,12740,12920,17640 --run-index 0 \
  --baseline-predictions outputs/fastflow_accel/baseline_dev1000/predictions.txt \
  --hazard-predictions outputs/fastflow_accel/hazard04_l015_dev1000/predictions.txt \
  --output experiments/fastflow_acceleration/case_study/random10/index.html --per-case-html --force
```

若从头复现原始逐步数据，最后再运行上面的纯重绘命令，生成带汇总分析的 `index.html`。

合并分析页面 `index.html` 的 SHA-256：`7a12d9ad787860353401c5827194b4bca949648fc13e53457e5c2d9e4761a69b`；压缩逐步数据 `index.json.gz` 的 SHA-256：`403ea159060edac43685db87391f19385e8332a61f5deef61fa1d2360fd4a3b7`。

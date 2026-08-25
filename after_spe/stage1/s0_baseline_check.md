# Stage1 S0：输入与基线冻结结果

日期：2026-08-24

## 结论

Stage1 的数据、checkpoint、tokenizer 规则和开发集已经冻结。当前无 GPU，但不影响 S1–S3 的标签构建、映射和静态诊断。

历史文档记录了 `SPE-M500@490K + Euler N=9 + 100 steps + seed=42` 在 dev-1000 上的结果：Top-1/3/5/10 为 `60.1/76.6/80.5/83.7%`，Oracle-any 为 `90.0%`，Invalid@1 为 `12.850%`，耗时约 `24.34 min`。

但是，本机没有找到这组 **seed=42 普通 Euler** 的完整预测文件和 sampling metadata。因此，这些数字只作为历史参照，不能直接用于 Stage1 的逐反应配对统计。进入 RC1 时必须在同一代码版本下重新跑：

1. B0：普通 Euler；
2. B1：真实反应中心引导；
3. B2：同一产物上的伪中心负对照。

这样可以避免把代码版本、随机流或历史结果文件的差异误判为中心先验收益。

## 冻结协议

| 项目 | 冻结值 |
|---|---|
| 数据表示 | 改进后 global R-SMILES + SPE-M500 |
| 数据目录 | `datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500` |
| checkpoint | `new_checkpoints/spe_m500_checkpoints/checkpoint_step490000.pt` |
| 采样器 | Euler |
| 轨迹/反应 | 9 |
| 推理步数 | 100 |
| scheduler | cubic |
| seed | 42 |
| 开发集 | `dev_unique1000_aug20` |
| 统计单位 | 1000 个 reaction；每个保留完整 20 augmentation |
| 输入行数 | 20000 |

## 关键哈希

| 文件 | SHA256 |
|---|---|
| M500@490K checkpoint | `225156c3e0120bba2f019285670f43a5bbbada1f6e29b75b208255e0abbfefc5` |
| M500 vocab | `efb4e3ef2b62c57799287c53089f69abeaadfce1644679907f0fa241003d6483` |
| SPE_ChEMBL rules | `ee408e30e0aae598770f233013b312b206df424f6e593b410b0107d7d2237a43` |
| raw train | `69661b12baa44d5a0be6cfc7698af8b518341fcb4427780c60358e0d9dcd8e7f` |
| raw val | `a52eb4cfd889820cf5172f65ac0e1ac124f3a36051674d5ca8bd63d037e149ee` |
| dev src | `54384b145933d85ce707f2eb5b7551e4ba3d3ab9197686ea5f1664e608fd8439` |
| dev tgt | `1862b520154415e10af85bcdc2d466e741ede4643f58750f9c36f6d224fffdaf` |
| global evaluation manifest | `1d030df8a52d3fdc2c9fd8bf82b1a247ad70b87784b2a843ee4fc87ac96e1f35` |

完整机器可读记录见 `s0_manifest.json`。

## 当前环境

CPU 环境为 Python `3.10.20`、PyTorch `2.7.1+cu126`、RDKit `2026.03.4`、SmilesPE `0.0.3`。当前 `cuda_available=false`，所以 GPU 型号、显存和正式推理 wall-clock 尚未填写；RC1 开始时补充。

## Git 边界

S0 起点 commit 为 `d4b99a6568e333517911bee8ede1f7b8744a73f5`。开始前已有三个用户删除项：`PROJECT_FACTS_SUMMARY.md`、`PROJECT_OVERVIEW.md`、`session-handoff.md`；Stage1 不修改、不暂存这些文件。

# `checkpoint_step600000.pt` 训练配置对齐记录

## 结论

当前 `configs/retro.yaml` 是历史 baseline 配置，不应为了后续实验直接改写。它经
YAML 解析后与 `checkpoint_step600000.pt` 内保存的 `config` 字典逐字段相等；本次只
在文件头增加了冻结说明。今后的新训练（例如显式 product conditioning、origin mask
或新的优化器）应另建 YAML，避免改变历史 baseline 的复现含义。

## 从 checkpoint 能确定的内容

| 项目 | 实际值 |
|---|---:|
| checkpoint step | 600,000 |
| 真实词表大小 | 69 |
| 模型词表大小（含 4 个特殊 token） | 73 |
| hidden dim / layers / heads | 256 / 10 / 8 |
| FFN dim / max length | 2048 / 256 |
| dropout / attention dropout | 0.3 / 0.3 |
| activation / positional scale | ReLU / `true` |
| batch size | 128 |
| total steps | 600,000 |
| optimizer | Adam, β=(0.9, 0.998), ε=1e-8 |
| learning-rate schedule | Noam, factor=1.0, warmup=8,000 |
| gradient clipping | 关闭 (`max_grad_norm=0.0`) |
| flow scheduler | cubic (`κ(t)=t³`) |
| model time input | raw `t` |
| rate reparameterization | 关闭 |
| origin mask | 关闭；权重中没有 `origin_embedding` |

checkpoint 的 optimizer 学习率为
`8.068715304598785e-05`，与 Noam 在 step 600,000 的理论值一致；scheduler state
为 `_step=600000`。模型按这些参数构造后 strict load 无 missing/unexpected keys，
参数量为 13,537,173，权重均为有限值。

## 推断出的历史训练流程

checkpoint 不包含 `train.log`、随机数状态或完整命令，因此设备、seed 和每一批的随机
顺序无法被绝对恢复。下面的部分由 checkpoint 配置、当前训练脚本和本地数据共同支持：

1. 数据目录为 `datasets/USPTO_50K_PtoR_aug20_#global#`，词表有 69 个真实 token。
2. 训练集的 `src-train.txt`、`tgt-train.txt`、`train_aligned_src.txt` 和
   `train_aligned_tgt.txt` 都是 800,060 行；预对齐文件存在时，`train_retro.py` 优先
   使用 `PreAlignedDataset` 和 `identity_align_xs_to_zs`，而不是在线 DP 对齐。
3. 因此每个 epoch 有 `floor(800060 / 128)=6250` 个 batch，600,000 个 update 约为
   96 个 epoch。
4. 耦合方向是 `x₀=product → x₁=reactants`；训练时从预对齐 Z 空间采样 `z_t`，模型
   输入去 GAP 后的状态，loss 使用 cubic scheduler。
5. DataLoader 的未显式写入 YAML 的运行时默认是 `shuffle=True`、`drop_last=True`、
   `num_workers=2`、`pin_memory=True`。

有一个历史实现细节需要保留在复现说明中：`train_retro.py` 在第一次 optimizer update
之后才调用 `NoamScheduler.step()`，所以第一次 update 使用了 Adam 的默认初始学习率
`1e-3`，后续才进入 Noam 曲线。这是旧训练脚本的行为，不是本次为了新实验修正的内容。

## 本次对齐修改

- `configs/retro.yaml`：只增加“冻结历史 baseline”的注释，数值不变。
- `scripts/train.py`：这是旧的通用合成数据入口，并非该 checkpoint 的训练入口。修复
  了它向当前 `prepare_batch` 传入旧参数名 `vocab_size` 的问题，并保持其原有的三特殊
  token 词表约定；做了 CPU 单步兼容性冒烟测试。
- `pyproject.toml`：补充 `einops>=0.6`。`edit_flows.utils.helpers` 会无条件导入
  `einops`，否则新机器即使按项目安装也无法导入训练脚本。

没有修改模型权重、训练/验证/测试数据、历史结果或 checkpoint 内的配置。

## 验证记录

- YAML 与 checkpoint config：逐字段相等。
- checkpoint 模型加载：strict load 通过；forward 输出形状为
  `(B,L,3)`, `(B,L,73)`, `(B,L,73)`，数值有限。
- 从 checkpoint 恢复 optimizer/Noam state：学习率和 scheduler step 均匹配。
- `tests/training` 与 `tests/models/test_transformer.py`：15 passed。
- 通用 `scripts/train.py` 兼容性单步 smoke：通过。

## 后续规则

历史 baseline 固定由 `configs/retro.yaml` + `checkpoint_step600000.pt` 表示。任何新
训练方案都应创建独立 YAML，并在文档中记录模型结构、数据分支、随机性、验证指标和
checkpoint provenance；不要覆盖这个文件来表达新方案。

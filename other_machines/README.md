# 外机训练与评估脚本

这里放需要拿到另一台 GPU 机器上运行的启动脚本。脚本会自动切换到仓库根目录，因此可以从仓库根目录或本目录调用，例如：

```bash
cd /root/autodl-tmp/edit_flows
bash other_machines/train_and_eval_retro_spe_m500.sh
```

脚本分类：

- `train_and_eval_retro_atom_rsmiles.sh`：原始 R-SMILES Atom-level 训练与评估；
- `train_and_eval_retro_spe_full.sh`：原始 R-SMILES Full-SPE 训练与评估；
- `train_and_eval_retro_spe_m500.sh`：原始 R-SMILES SPE-M500 训练与评估；
- `train_and_eval_retro_spe_m500_product_memory.sh`：global R-SMILES SPE-M500 Product-Memory 训练与评估；
- `train_and_eval_retro_atom_product_memory.sh`：改进后的 global R-SMILES Atom-level Product-Memory 训练与完整 test R9K1M2 评估；Atom 长序列使用 `batch_size=128`，用于和 Atom baseline 做单变量对照。默认评估 `450K/490K/500K/550K/600K`，可用 `MAX_PRODUCTS=20000` 只跑前 1,000 个 test 反应；
- `eval_atom_product_memory_r9k1m2_full_checkpoints.sh`：已完成 Atom-level Product-Memory 训练后的完整 test R9K1M2 checkpoint sweep；默认评估 `450K/490K/500K/550K/600K`，可用 `RUN_DIR` 和 `STEPS` 指定具体训练目录与 checkpoint；
- `eval_product_memory_next.sh`：已完成 Product-Memory 训练后的分阶段评估。依次运行 `smoke`、`dev_euler`、`dev_r9`；只有 dev 确认后才用 `ALLOW_PRODUCT_MEMORY_FULL_REFERENCE=YES` 解锁 `full_euler` 或 `full_r9`。它固定 M500@500K、100 steps、cubic，并在 R9 模式冻结 `R9K1M2`（9 runs、K=1、M=2、full-probability、`stochastic_noop`）协议；
- `eval_product_memory_r9k1m2_full_checkpoints.sh`：Product-Memory 的直接完整 test checkpoint sweep。默认依次评估 `450K/490K/500K/550K/600K`，固定 R9K1M2，适用于当前跳过 dev、直接比较多个 checkpoint 的决定；
- `eval_full_test_r9k1m2_spe_m500.sh`：global R-SMILES Atom/M500 的 R9K1M2 全量 test；
- `run_r9k1m2_b1_oracle_checkpoints.sh`：真实反应中心引导 B1 的多个 checkpoint 全量 test；
- `run_r9k1m2_first_event_distance_full.sh`：以 B0-trace（倍率 1）和 oracle B1（倍率 3）在完整 test 上比较首个非空编辑前后的 token 距离；输出流式汇总，不写百万条事件 JSON。先运行 smoke：

  ```bash
  bash other_machines/run_r9k1m2_first_event_distance_full.sh smoke
  ```

  smoke 正常后才可运行完整诊断：

  ```bash
  ALLOW_FULL_R9_FIRST_EVENT_DISTANCE=YES \
    bash other_machines/run_r9k1m2_first_event_distance_full.sh full
  ```

  该脚本要求已有 `SPE-M500@490K` checkpoint、M500 full-test 数据，以及 Git-LFS 下载的 `after_spe/center_sidecars/test_all_aug20/scores.jsonl`；单张 3090 顺序运行约 5–6 小时。结果写入 `results/after_spe_stage1/r9k1m2_first_event_distance/`；
- `run10tra.sh`：历史可视化批处理脚本。

长时间任务前仍需单独准备对应 checkpoint、数据集和配置；模型 checkpoint 不由 Git 自动下载。

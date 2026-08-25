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

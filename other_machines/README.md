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
- `run10tra.sh`：历史可视化批处理脚本。

长时间任务前仍需单独准备对应 checkpoint、数据集和配置；模型 checkpoint 不由 Git 自动下载。

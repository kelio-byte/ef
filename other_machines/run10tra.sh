#!/bin/bash
# 文件名: run10tra.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 生成带时间戳的日志文件名（每轮10次共用一个文件）
LOGFILE="run_$(date +'%Y-%m-%d_%H-%M-%S').log"

# 写入批次开始标记
echo "==================================================" | tee -a "$LOGFILE"
echo "===== Batch of 10 runs started at $(date) =====" | tee -a "$LOGFILE"
echo "==================================================" | tee -a "$LOGFILE"

for i in {1..10}; do
    echo "===== Run $i started at $(date) =====" | tee -a "$LOGFILE"
    python scripts/visualize_trajectory.py \
    --checkpoint checkpoint_step600000.pt \
    --products_file "datasets/USPTO_50K_PtoR_aug20_#global#/test/src-test.txt" \
    --targets_file "datasets/USPTO_50K_PtoR_aug20_#global#/test/tgt-test.txt" \
    --output_dir "visualizations/trajectory-1/" \
    --scheduler cubic \
    --deduplicate 1 \
    --example_ids "1" \
    --n_steps 100 \
    --n_samples 3 \
    --device cuda
    echo "===== Run $i finished at $(date) =====" | tee -a "$LOGFILE"
done

echo "===== Batch finished at $(date) =====" | tee -a "$LOGFILE"
echo "==================================================" | tee -a "$LOGFILE"

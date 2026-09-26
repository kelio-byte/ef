#!/usr/bin/env bash
# 用途：评测已有的 550K checkpoint，不重新训练。
# 输入：checkpoint_step550000.pt 的路径。
# 输出：同目录下 test_step550000/ 中的预测、元数据和全量 test 指标。

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "用法：bash scripts/evaluate_550k.sh <checkpoint_step550000.pt>" >&2
  exit 2
fi

checkpoint_arg="$1"
if [[ ! -f "$checkpoint_arg" ]]; then
  echo "找不到 checkpoint：$checkpoint_arg" >&2
  exit 2
fi
checkpoint="$(realpath "$checkpoint_arg")"
if [[ "${checkpoint##*/}" != "checkpoint_step550000.pt" ]]; then
  echo "需要 checkpoint_step550000.pt，收到：${checkpoint##*/}" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
output_dir="$(dirname -- "$checkpoint")/test_step550000"
if [[ -e "$output_dir/predictions.txt" || -e "$output_dir/metrics.json" ]]; then
  echo "评测结果已存在，拒绝覆盖：$output_dir" >&2
  exit 1
fi

cd "$repo_root"
"${PYTHON:-python}" "$repo_root/scripts/evaluate.py" \
  --split test \
  --checkpoint "$checkpoint" \
  --output "$output_dir" \
  --n-runs 9 \
  --n-children 2 \
  --device "${DEVICE:-cuda}" \
  --workers "${WORKERS:-8}"

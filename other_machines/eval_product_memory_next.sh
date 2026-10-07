#!/usr/bin/env bash
# Evaluate the trained Global R-SMILES + SPE-M500 product-memory checkpoint.
#
# This script deliberately separates four stages:
#   smoke       one reaction, R9K1M2: verify the new cache-aware beam path
#   dev_euler   missing ordinary-Euler seeds (7, 123) for the trained PM@500K
#   dev_r9      R9K1M2 dev1000 evaluation (42, 7, 123)
#   full_*      final reference evaluation only after dev selection; gated
#
# Examples (run from either repository root or this directory):
#   conda activate ef
#   bash other_machines/eval_product_memory_next.sh smoke
#   bash other_machines/eval_product_memory_next.sh dev_euler
#   bash other_machines/eval_product_memory_next.sh dev_r9
#
# After reviewing dev results only:
#   ALLOW_PRODUCT_MEMORY_FULL_REFERENCE=YES \
#     bash other_machines/eval_product_memory_next.sh full_r9
#
# Optional overrides:
#   CHECKPOINT=/path/to/checkpoint_step500000.pt \
#   SEEDS="42 7 123" BATCH_SIZE=32 \
#     bash other_machines/eval_product_memory_next.sh dev_r9

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

SCOPE="${1:-${SCOPE:-smoke}}"
DATA="${DATA:-datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500}"
SAVE_ROOT="${SAVE_ROOT:-checkpoints/retro_spe_m500_product_memory_600k}"
DATA_NAME="${DATA_NAME:-$(basename "$DATA")}"
CHECKPOINT="${CHECKPOINT:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PROCESS_NUMBER="${PROCESS_NUMBER:-12}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

if [ -n "${PYTHON_BIN:-}" ]; then
  : # Explicit override.
elif [ -x /root/autodl-tmp/ef/bin/python ]; then
  PYTHON_BIN=/root/autodl-tmp/ef/bin/python
elif [ -x /root/miniconda3/envs/ef/bin/python ]; then
  PYTHON_BIN=/root/miniconda3/envs/ef/bin/python
else
  PYTHON_BIN=python
fi

require_hydrated_file() {
  local required_path="$1"
  if [ ! -f "$required_path" ]; then
    echo "Required file not found: $required_path" >&2
    exit 1
  fi
  if head -n1 "$required_path" | grep -q '^version https://git-lfs.github.com/spec/v1$'; then
    echo "Git-LFS pointer is not hydrated: $required_path" >&2
    echo "Run: git lfs pull --include=\"datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/**\"" >&2
    exit 1
  fi
}

resolve_checkpoint() {
  if [ -n "$CHECKPOINT" ]; then
    return
  fi
  local candidates=()
  mapfile -t candidates < <(
    find "$SAVE_ROOT/$DATA_NAME" -mindepth 2 -maxdepth 2 \
      -type f -name 'checkpoint_step500000.pt' -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | awk '{print $2}'
  )
  if [ "${#candidates[@]}" -eq 0 ]; then
    echo "Could not find checkpoint_step500000.pt under $SAVE_ROOT/$DATA_NAME" >&2
    echo "Set CHECKPOINT=/absolute/or/relative/path/checkpoint_step500000.pt" >&2
    exit 1
  fi
  CHECKPOINT="${candidates[0]}"
  if [ "${#candidates[@]}" -gt 1 ]; then
    echo "Multiple PM@500K checkpoints found; using newest: $CHECKPOINT" >&2
    printf 'Other candidates:\n%s\n' "${candidates[*]:1}" >&2
    echo "Set CHECKPOINT explicitly if this is not the intended run." >&2
  fi
}

case "$SCOPE" in
  smoke)
    SAMPLER_KIND=r9
    DEFAULT_SEEDS="42"
    PRODUCTS_FILE="$DATA/evaluation_v2/dev_unique1000_aug20/src.txt"
    TARGETS_FILE="$DATA/evaluation_v2/dev_unique1000_aug20/tgt.txt"
    MAX_PRODUCTS=20
    ;;
  dev_euler)
    SAMPLER_KIND=euler
    # seed 42 was already recorded by the training script; these complete
    # the planned three-seed dev confirmation without repeating it.
    DEFAULT_SEEDS="7 123"
    PRODUCTS_FILE="$DATA/evaluation_v2/dev_unique1000_aug20/src.txt"
    TARGETS_FILE="$DATA/evaluation_v2/dev_unique1000_aug20/tgt.txt"
    MAX_PRODUCTS=20000
    ;;
  dev_r9)
    SAMPLER_KIND=r9
    DEFAULT_SEEDS="42 7 123"
    PRODUCTS_FILE="$DATA/evaluation_v2/dev_unique1000_aug20/src.txt"
    TARGETS_FILE="$DATA/evaluation_v2/dev_unique1000_aug20/tgt.txt"
    MAX_PRODUCTS=20000
    ;;
  full_euler)
    SAMPLER_KIND=euler
    DEFAULT_SEEDS="42"
    PRODUCTS_FILE="$DATA/test/src-test.txt"
    TARGETS_FILE="$DATA/test/tgt-test.txt"
    MAX_PRODUCTS=""
    ;;
  full_r9)
    SAMPLER_KIND=r9
    DEFAULT_SEEDS="42"
    PRODUCTS_FILE="$DATA/test/src-test.txt"
    TARGETS_FILE="$DATA/test/tgt-test.txt"
    MAX_PRODUCTS=""
    ;;
  *)
    echo "Unknown scope: $SCOPE" >&2
    echo "Use one of: smoke, dev_euler, dev_r9, full_euler, full_r9" >&2
    exit 1
    ;;
esac

if [[ "$SCOPE" == full_* ]] \
  && [ "${ALLOW_PRODUCT_MEMORY_FULL_REFERENCE:-}" != "YES" ]; then
  echo "Refusing full test without explicit acknowledgement." >&2
  echo "Run: ALLOW_PRODUCT_MEMORY_FULL_REFERENCE=YES bash $0 $SCOPE" >&2
  exit 1
fi

SEEDS="${SEEDS:-$DEFAULT_SEEDS}"
mkdir -p logs results

for required in "$DATA/example.vocab.src" "$PRODUCTS_FILE" "$TARGETS_FILE"; do
  require_hydrated_file "$required"
done
resolve_checkpoint
if [ ! -f "$CHECKPOINT" ]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

SOURCE_ROWS="$(wc -l < "$PRODUCTS_FILE")"
TARGET_ROWS="$(wc -l < "$TARGETS_FILE")"
if [ "$SOURCE_ROWS" -ne "$TARGET_ROWS" ] || [ $((SOURCE_ROWS % 20)) -ne 0 ]; then
  echo "Invalid augmentation-20 layout: src=$SOURCE_ROWS, tgt=$TARGET_ROWS" >&2
  exit 1
fi
if [ -n "$MAX_PRODUCTS" ] && { [ "$MAX_PRODUCTS" -le 0 ] || [ "$MAX_PRODUCTS" -gt "$SOURCE_ROWS" ] || [ $((MAX_PRODUCTS % 20)) -ne 0 ]; }; then
  echo "Invalid MAX_PRODUCTS=$MAX_PRODUCTS for $SOURCE_ROWS input rows" >&2
  exit 1
fi

"$PYTHON_BIN" - "$CHECKPOINT" <<'PY'
from pathlib import Path
import sys
import torch

path = Path(sys.argv[1])
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")
config = checkpoint.get("config", {})
if not bool(config.get("use_product_memory", False)):
    raise SystemExit("Checkpoint is not configured for product memory")
state = checkpoint.get("model_state_dict", {})
if not any(key.startswith("product_memory_encoder_layers.") for key in state):
    raise SystemExit("Checkpoint lacks product-memory encoder weights")
print(
    "Checkpoint product memory: encoder_layers=",
    config.get("product_memory_encoder_layers"),
    "; fusion_after=",
    config.get("product_memory_fusion_after_layers"),
    sep="",
)
PY

echo "Project:    $PROJECT_ROOT"
echo "Python:     $PYTHON_BIN"
echo "Scope:      $SCOPE"
echo "Checkpoint: $CHECKPOINT"
echo "Data:       $DATA"
echo "Inputs:     $SOURCE_ROWS rows ($(($SOURCE_ROWS / 20)) reactions)"
echo "Sampler:    $SAMPLER_KIND"
echo "Seeds:      $SEEDS"
echo "Batch size: $BATCH_SIZE"

for seed in $SEEDS; do
  case "$seed" in
    ''|*[!0-9]*)
      echo "Invalid seed: $seed" >&2
      exit 1
      ;;
  esac

  if [ "$SAMPLER_KIND" = euler ]; then
    sampler_args=(
      --sampler euler
      --n_samples 9
    )
    sampler_tag=euler_n9
  else
    # Frozen R9K1M2 protocol.  Do not enable first-edit diversity, center
    # bias, or other search changes in this model-vs-sampler comparison.
    sampler_args=(
      --sampler euler_beam
      --n_runs 9
      --n_branches 1
      --n_children 2
      --euler_beam_score_mode full_probability
      --euler_beam_changed_state_bonus 0.5
      --euler_beam_q_temperature 1.0
      --euler_beam_matmul_precision high
      --euler_beam_child_policy stochastic_noop
    )
    sampler_tag=r9k1m2
  fi

  output_dir="results/product_memory_step500k_${SCOPE}_${sampler_tag}_seed${seed}_${RUN_TAG}"
  log_file="logs/product_memory_step500k_${SCOPE}_${sampler_tag}_seed${seed}_${RUN_TAG}.log"
  command=(
    "$PYTHON_BIN" scripts/eval.py
    --checkpoint "$CHECKPOINT"
    --products_file "$PRODUCTS_FILE"
    --targets "$TARGETS_FILE"
    --data_dir "$DATA"
    --vocab_file "$DATA/example.vocab.src"
    --output_dir "$output_dir"
    "${sampler_args[@]}"
    --n_steps 100
    --scheduler cubic
    --batch_size "$BATCH_SIZE"
    --device cuda
    --seed "$seed"
    --augmentation 20
    --n_best 10
    --process_number "$PROCESS_NUMBER"
  )
  if [ -n "$MAX_PRODUCTS" ]; then
    command+=(--max_products "$MAX_PRODUCTS")
  fi

  printf '\n============================================================\n'
  echo "Starting $SCOPE / $sampler_tag / seed=$seed: $(date -Is)"
  echo "Output: $output_dir"
  printf '============================================================\n'
  "${command[@]}" 2>&1 | tee "$log_file"
done

echo "Complete: scope=$SCOPE; checkpoint=$CHECKPOINT"

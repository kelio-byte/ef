#!/usr/bin/env bash
# Evaluate an already-trained Atom-level product-memory run on the complete
# improved/global R-SMILES test set with the fixed R9K1M2 protocol.
#
# Run after training (or after selecting an existing run directory):
#   bash other_machines/eval_atom_product_memory_r9k1m2_full_checkpoints.sh
#
# Optional:
#   RUN_DIR=checkpoints/retro_atom_product_memory_600k/USPTO_50K_PtoR_aug20_#global#/TIMESTAMP \
#   STEPS="490000 500000 550000 600000" \
#   bash other_machines/eval_atom_product_memory_r9k1m2_full_checkpoints.sh
#
# Use MAX_PRODUCTS=20000 for a 1,000-reaction smoke-sized prefix; leave it
# empty (the default) for all 100,140 augmentation rows.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

DATA="${DATA:-datasets/USPTO_50K_PtoR_aug20_#global#}"
SAVE_ROOT="${SAVE_ROOT:-checkpoints/retro_atom_product_memory_600k}"
DATA_NAME="${DATA_NAME:-$(basename "$DATA")}"
RUN_DIR="${RUN_DIR:-}"
STEPS="${STEPS:-450000 490000 500000 550000 600000}"
EVAL_SEED="${EVAL_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PROCESS_NUMBER="${PROCESS_NUMBER:-12}"
MAX_PRODUCTS="${MAX_PRODUCTS:-}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

if [ -n "${PYTHON_BIN:-}" ]; then
  :
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
    echo "Run: git lfs pull --include=\"datasets/USPTO_50K_PtoR_aug20_#global#/**\"" >&2
    exit 1
  fi
}

has_all_requested_checkpoints() {
  local directory="$1"
  local step
  for step in $STEPS; do
    if [ ! -f "$directory/checkpoint_step${step}.pt" ]; then
      return 1
    fi
  done
  return 0
}

resolve_run_dir() {
  if [ -n "$RUN_DIR" ]; then
    return
  fi
  local candidates=()
  mapfile -t candidates < <(
    find "$SAVE_ROOT/$DATA_NAME" -mindepth 1 -maxdepth 1 -type d \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr | cut -d' ' -f2-
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if has_all_requested_checkpoints "$candidate"; then
      RUN_DIR="$candidate"
      return
    fi
  done
  echo "No run directory under $SAVE_ROOT/$DATA_NAME contains all requested checkpoints: $STEPS" >&2
  echo "Set RUN_DIR explicitly or reduce STEPS." >&2
  exit 1
}

for required in \
  "$DATA/example.vocab.src" \
  "$DATA/test/src-test.txt" \
  "$DATA/test/tgt-test.txt"; do
  require_hydrated_file "$required"
done

SOURCE_ROWS="$(wc -l < "$DATA/test/src-test.txt")"
TARGET_ROWS="$(wc -l < "$DATA/test/tgt-test.txt")"
if [ "$SOURCE_ROWS" -ne 100140 ] || [ "$TARGET_ROWS" -ne 100140 ]; then
  echo "Expected 100140 augmentation rows; got src=$SOURCE_ROWS, tgt=$TARGET_ROWS" >&2
  exit 1
fi
if [ -n "$MAX_PRODUCTS" ]; then
  case "$MAX_PRODUCTS" in
    ''|*[!0-9]*)
      echo "MAX_PRODUCTS must be a positive integer, got: $MAX_PRODUCTS" >&2
      exit 1
      ;;
  esac
  if [ "$MAX_PRODUCTS" -le 0 ] || [ "$MAX_PRODUCTS" -gt "$SOURCE_ROWS" ] || [ $((MAX_PRODUCTS % 20)) -ne 0 ]; then
    echo "MAX_PRODUCTS must be positive, <=$SOURCE_ROWS, and divisible by 20" >&2
    exit 1
  fi
fi

resolve_run_dir
if [ ! -d "$RUN_DIR" ]; then
  echo "RUN_DIR does not exist: $RUN_DIR" >&2
  exit 1
fi

CHECKPOINTS=()
for step in $STEPS; do
  case "$step" in
    ''|*[!0-9]*)
      echo "Invalid checkpoint step: $step" >&2
      exit 1
      ;;
  esac
  checkpoint="$RUN_DIR/checkpoint_step${step}.pt"
  if [ ! -f "$checkpoint" ]; then
    echo "Checkpoint not found: $checkpoint" >&2
    exit 1
  fi
  CHECKPOINTS+=("$checkpoint")
done

"$PYTHON_BIN" - "${CHECKPOINTS[@]}" <<'PY'
from pathlib import Path
import sys

import torch

for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config", {})
    if not bool(config.get("use_product_memory", False)):
        raise SystemExit(f"{path}: checkpoint is not product-memory enabled")
    if config.get("data_dir") != "datasets/USPTO_50K_PtoR_aug20_#global#":
        raise SystemExit(f"{path}: unexpected data_dir={config.get('data_dir')!r}")
    state = checkpoint.get("model_state_dict", {})
    if not any(key.startswith("product_memory_encoder_layers.") for key in state):
        raise SystemExit(f"{path}: product-memory encoder weights are missing")
    print(
        f"verified {path.name}: product_memory_encoder_layers="
        f"{config.get('product_memory_encoder_layers')}, "
        f"fusion_after={config.get('product_memory_fusion_after_layers')}"
    )
PY

mkdir -p logs results
echo "Project:     $PROJECT_ROOT"
echo "Python:      $PYTHON_BIN"
echo "Run dir:     $RUN_DIR"
echo "Data:        $DATA"
echo "Full test:   $SOURCE_ROWS rows / $((SOURCE_ROWS / 20)) reactions"
echo "Checkpoints: $STEPS"
echo "Protocol:    R9K1M2, 100 Euler steps, seed=$EVAL_SEED, batch=$BATCH_SIZE"

for step in $STEPS; do
  checkpoint="$RUN_DIR/checkpoint_step${step}.pt"
  output_dir="results/atom_product_memory_step${step}_fulltest_r9k1m2_seed${EVAL_SEED}_${RUN_TAG}"
  log_file="logs/atom_product_memory_step${step}_fulltest_r9k1m2_seed${EVAL_SEED}_${RUN_TAG}.log"

  printf '\n============================================================\n'
  echo "Starting Atom product-memory R9K1M2 full test: step=$((step / 1000))K | $(date -Is)"
  echo "Checkpoint: $checkpoint"
  echo "Output:     $output_dir"
  printf '============================================================\n'

  command=(
    "$PYTHON_BIN" scripts/eval.py
    --checkpoint "$checkpoint"
    --products_file "$DATA/test/src-test.txt"
    --targets "$DATA/test/tgt-test.txt"
    --data_dir "$DATA"
    --vocab_file "$DATA/example.vocab.src"
    --output_dir "$output_dir"
    --sampler euler_beam
    --n_samples 3
    --n_runs 9
    --n_branches 1
    --n_children 2
    --n_steps 100
    --scheduler cubic
    --batch_size "$BATCH_SIZE"
    --device cuda
    --seed "$EVAL_SEED"
    --augmentation 20
    --n_best 10
    --process_number "$PROCESS_NUMBER"
    --euler_beam_score_mode full_probability
    --euler_beam_changed_state_bonus 0.5
    --euler_beam_q_temperature 1.0
    --euler_beam_matmul_precision high
    --euler_beam_child_policy stochastic_noop
  )
  if [ -n "$MAX_PRODUCTS" ]; then
    command+=(--max_products "$MAX_PRODUCTS")
  fi
  "${command[@]}" 2>&1 | tee "$log_file"
done

echo "Complete: Atom product-memory R9K1M2 full-test checkpoint sweep"

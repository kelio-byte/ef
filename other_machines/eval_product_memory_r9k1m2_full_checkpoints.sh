#!/usr/bin/env bash
# Full-test checkpoint sweep for the trained product-memory model.
#
# Fixed protocol for every checkpoint:
#   Global R-SMILES + SPE-M500; R9K1M2; 100 Euler steps; cubic scheduler;
#   full_probability; stochastic_noop; changed-state bonus 0.5; seed 42.
#
# Default checkpoints are the five already evaluated on dev1000.  They cover
# different observed strengths without turning this into a hyperparameter grid:
#   450K (Top-3), 490K, 500K (Top-1 / Top-10 / Oracle / Invalid),
#   550K (Top-5), 600K (late checkpoint).
#
# Run:
#   conda activate ef
#   cd /root/autodl-tmp/edit_flows
#   bash other_machines/eval_product_memory_r9k1m2_full_checkpoints.sh
#
# If automatic run-directory detection is ambiguous, pin it explicitly:
#   RUN_DIR=checkpoints/retro_spe_m500_product_memory_600k/\
#USPTO_50K_PtoR_aug20_#global#_SPE_m500/<timestamp> \
#   bash other_machines/eval_product_memory_r9k1m2_full_checkpoints.sh
#
# Useful overrides:
#   STEPS="490000 500000" BATCH_SIZE=32 \
#     bash other_machines/eval_product_memory_r9k1m2_full_checkpoints.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

DATA="${DATA:-datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500}"
SAVE_ROOT="${SAVE_ROOT:-checkpoints/retro_spe_m500_product_memory_600k}"
DATA_NAME="${DATA_NAME:-$(basename "$DATA")}"
RUN_DIR="${RUN_DIR:-}"
STEPS="${STEPS:-450000 490000 500000 550000 600000}"
EVAL_SEED="${EVAL_SEED:-42}"
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
  echo "Could not find one run directory containing all requested steps:" >&2
  echo "  $STEPS" >&2
  echo "under: $SAVE_ROOT/$DATA_NAME" >&2
  echo "Set RUN_DIR explicitly, or reduce STEPS to files that are present." >&2
  exit 1
}

validate_integer() {
  local name="$1"
  local value="$2"
  case "$value" in
    ''|*[!0-9]*)
      echo "$name must be a positive integer, got: $value" >&2
      exit 1
      ;;
  esac
  if [ "$value" -le 0 ]; then
    echo "$name must be positive, got: $value" >&2
    exit 1
  fi
}

validate_integer EVAL_SEED "$EVAL_SEED"
validate_integer BATCH_SIZE "$BATCH_SIZE"
validate_integer PROCESS_NUMBER "$PROCESS_NUMBER"

for required in \
  "$DATA/example.vocab.src" \
  "$DATA/test/src-test.txt" \
  "$DATA/test/tgt-test.txt"; do
  require_hydrated_file "$required"
done

SOURCE_ROWS="$(wc -l < "$DATA/test/src-test.txt")"
TARGET_ROWS="$(wc -l < "$DATA/test/tgt-test.txt")"
if [ "$SOURCE_ROWS" -ne 100140 ] || [ "$TARGET_ROWS" -ne 100140 ]; then
  echo "Expected full test layout of 100140 augmentation rows in both files; got src=$SOURCE_ROWS, tgt=$TARGET_ROWS" >&2
  exit 1
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
        raise SystemExit(f"{path}: not a product-memory checkpoint")
    state = checkpoint.get("model_state_dict", {})
    if not any(key.startswith("product_memory_encoder_layers.") for key in state):
        raise SystemExit(f"{path}: product-memory encoder weights missing")
    print(
        f"verified {path.name}: "
        f"memory_encoder_layers={config.get('product_memory_encoder_layers')}, "
        f"fusion_after={config.get('product_memory_fusion_after_layers')}"
    )
PY

mkdir -p logs results

"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; activate the ef environment first")
print(f"Torch: {torch.__version__}; GPU: {torch.cuda.get_device_name(0)}")
PY

echo "Project:     $PROJECT_ROOT"
echo "Python:      $PYTHON_BIN"
echo "Run dir:     $RUN_DIR"
echo "Data:        $DATA"
echo "Full test:   $SOURCE_ROWS rows / $(($SOURCE_ROWS / 20)) reactions"
echo "Checkpoints: $STEPS"
echo "Sampler:     R9K1M2 (n_runs=9, n_branches=1, n_children=2)"
echo "Protocol:    100 steps, cubic, full_probability, stochastic_noop, bonus=0.5, seed=$EVAL_SEED"
echo "Batch size:  $BATCH_SIZE"

for step in $STEPS; do
  checkpoint="$RUN_DIR/checkpoint_step${step}.pt"
  output_dir="results/product_memory_step${step}_fulltest_r9k1m2_seed${EVAL_SEED}_${RUN_TAG}"
  log_file="logs/product_memory_step${step}_fulltest_r9k1m2_seed${EVAL_SEED}_${RUN_TAG}.log"

  printf '\n============================================================\n'
  echo "Starting PM R9K1M2 full test: checkpoint=$((step / 1000))K | $(date -Is)"
  echo "Checkpoint: $checkpoint"
  echo "Output:     $output_dir"
  printf '============================================================\n'

  "$PYTHON_BIN" scripts/eval.py \
    --checkpoint "$checkpoint" \
    --products_file "$DATA/test/src-test.txt" \
    --targets "$DATA/test/tgt-test.txt" \
    --data_dir "$DATA" \
    --vocab_file "$DATA/example.vocab.src" \
    --output_dir "$output_dir" \
    --sampler euler_beam \
    --n_samples 3 \
    --n_runs 9 \
    --n_branches 1 \
    --n_children 2 \
    --n_steps 100 \
    --scheduler cubic \
    --batch_size "$BATCH_SIZE" \
    --device cuda \
    --seed "$EVAL_SEED" \
    --augmentation 20 \
    --n_best 10 \
    --process_number "$PROCESS_NUMBER" \
    --euler_beam_score_mode full_probability \
    --euler_beam_changed_state_bonus 0.5 \
    --euler_beam_q_temperature 1.0 \
    --euler_beam_matmul_precision high \
    --euler_beam_child_policy stochastic_noop \
    2>&1 | tee "$log_file"
done

echo "Complete: product-memory R9K1M2 full-test checkpoint sweep."

#!/usr/bin/env bash
# Product-memory P1 on the established Global R-SMILES + SPE-M500 dataset.
#
# Normal run:
#   bash train_and_eval_retro_spe_m500_product_memory.sh
#
# Safe preflight / GPU smoke check (trains exactly five optimizer steps and
# does not launch the long evaluation suite):
#   SMOKE_STEPS=5 bash train_and_eval_retro_spe_m500_product_memory.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CONFIG="${CONFIG:-configs/retro_spe_m500_product_memory_600k.yaml}"
SAVE_ROOT="${SAVE_ROOT:-checkpoints/retro_spe_m500_product_memory_600k}"
DATA="${DATA:-datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500}"
DATA_NAME="${DATA_NAME:-$(basename "$DATA")}"
EVAL_STEPS="${EVAL_STEPS:-450000 490000 500000 550000 600000}"
EVAL_SEED="${EVAL_SEED:-42}"
MAX_PRODUCTS="${MAX_PRODUCTS:-20000}"
PROCESS_NUMBER="${PROCESS_NUMBER:-12}"
if [ -n "${PYTHON_BIN:-}" ]; then
  : # Explicit override, e.g. PYTHON_BIN=/root/miniconda3/envs/ef/bin/python.
elif [ -x /root/autodl-tmp/ef/bin/python ]; then
  PYTHON_BIN=/root/autodl-tmp/ef/bin/python
elif [ -x /root/miniconda3/envs/ef/bin/python ]; then
  PYTHON_BIN=/root/miniconda3/envs/ef/bin/python
else
  PYTHON_BIN=python
fi
SMOKE_STEPS="${SMOKE_STEPS:-}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs results

if [ ! -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
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

for required in \
  "$DATA/example.vocab.src" \
  "$DATA/train/train_aligned_src.txt" \
  "$DATA/train/train_aligned_tgt.txt" \
  "$DATA/val/val_aligned_src.txt" \
  "$DATA/val/val_aligned_tgt.txt"; do
  require_hydrated_file "$required"
done

SMOKE_MODE=0
CONFIG_TO_RUN="$CONFIG"
if [ -n "$SMOKE_STEPS" ]; then
  case "$SMOKE_STEPS" in
    ''|*[!0-9]*)
      echo "SMOKE_STEPS must be a positive integer, got: $SMOKE_STEPS" >&2
      exit 1
      ;;
  esac
  if [ "$SMOKE_STEPS" -lt 1 ]; then
    echo "SMOKE_STEPS must be >= 1, got: $SMOKE_STEPS" >&2
    exit 1
  fi
  SMOKE_MODE=1
  CONFIG_TO_RUN="$(mktemp /tmp/retro_spe_m500_product_memory_smoke.XXXXXX.yaml)"
  trap 'rm -f "$CONFIG_TO_RUN"' EXIT
  "$PYTHON_BIN" - "$CONFIG" "$CONFIG_TO_RUN" "$SMOKE_STEPS" <<'PY'
from pathlib import Path
import sys

import yaml

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
steps = int(sys.argv[3])
payload = yaml.safe_load(source.read_text())
config = payload.get("retro", payload)
config["total_steps"] = steps
config["checkpoint_interval"] = steps
config["keep_checkpoints"] = 1
tensorboard = config.setdefault("tensorboard", {})
tensorboard["log_interval"] = 1
tensorboard["validation_start_step"] = steps + 1
tensorboard["validation_interval"] = 0
monitoring = config.setdefault("monitoring", {})
monitoring["interval"] = 1
destination.write_text(yaml.safe_dump(payload, sort_keys=False))
PY
fi

"$PYTHON_BIN" - "$CONFIG_TO_RUN" "$DATA" "$EVAL_STEPS" "$SMOKE_MODE" <<'PY'
from pathlib import Path
import sys

import torch
import yaml

if sys.version_info < (3, 10):
    raise SystemExit(f"Python >= 3.10 required, got {sys.version}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; activate the ef environment first")

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text())
config = config.get("retro", config)
requested_data = sys.argv[2]
if config.get("data_dir") != requested_data:
    raise SystemExit(
        "Config/data mismatch: "
        f"config data_dir={config.get('data_dir')!r}, DATA={requested_data!r}"
    )
if not bool(config.get("use_product_memory", False)):
    raise SystemExit("Product-memory config must set use_product_memory: true")
if int(config.get("product_memory_encoder_layers", 0)) != 2:
    raise SystemExit(
        "P1 requires product_memory_encoder_layers: 2; got "
        f"{config.get('product_memory_encoder_layers')!r}"
    )
if list(config.get("product_memory_fusion_after_layers") or []) != [5, 10]:
    raise SystemExit(
        "P1 requires product_memory_fusion_after_layers: [5, 10]; got "
        f"{config.get('product_memory_fusion_after_layers')!r}"
    )
if bool(config.get("use_origin_mask", False)):
    raise SystemExit("P1 must keep use_origin_mask: false for a clean ablation")
if float(config.get("max_grad_norm", -1.0)) != 0.0:
    raise SystemExit(
        "P1 is an unclipped baseline comparison and requires "
        f"max_grad_norm: 0.0; got {config.get('max_grad_norm')!r}"
    )

smoke_mode = bool(int(sys.argv[4]))
if not smoke_mode:
    try:
        eval_steps = [int(step) for step in sys.argv[3].split()]
    except ValueError as exc:
        raise SystemExit(f"Invalid EVAL_STEPS: {sys.argv[3]!r}") from exc
    total_steps = int(config.get("total_steps", 0))
    interval = int(config.get("checkpoint_interval", 0))
    if not eval_steps or any(step <= 0 or step > total_steps for step in eval_steps):
        raise SystemExit(
            f"EVAL_STEPS must be within 1..{total_steps}, got {eval_steps}"
        )
    if interval <= 0 or any(step % interval for step in eval_steps):
        raise SystemExit(
            "Every EVAL_STEPS entry must be a multiple of "
            f"checkpoint_interval={interval}, got {eval_steps}"
        )

print(f"Python: {sys.executable} ({sys.version.split()[0]})")
print(f"Torch: {torch.__version__}; GPU: {torch.cuda.get_device_name(0)}")
print(f"Config/data consistency: {config['data_dir']}")
print(
    "Product memory: x0 encoder layers="
    f"{config['product_memory_encoder_layers']}; fusion after state layers="
    f"{config['product_memory_fusion_after_layers']}"
)
print(f"Unclipped control: max_grad_norm={config['max_grad_norm']}")
if smoke_mode:
    print(f"Smoke mode: total_steps={config['total_steps']} (evaluation disabled)")
else:
    print(f"Evaluation steps: {eval_steps}")
PY

if [ "$SMOKE_MODE" -eq 0 ]; then
  for required in \
    "$DATA/evaluation_v2/dev_unique1000_aug20/src.txt" \
    "$DATA/evaluation_v2/dev_unique1000_aug20/tgt.txt"; do
    require_hydrated_file "$required"
  done

  case "$MAX_PRODUCTS" in
    ''|*[!0-9]*)
      echo "MAX_PRODUCTS must be a positive integer, got: $MAX_PRODUCTS" >&2
      exit 1
      ;;
  esac
  if [ "$MAX_PRODUCTS" -le 0 ] || [ $((MAX_PRODUCTS % 20)) -ne 0 ]; then
    echo "MAX_PRODUCTS must be positive and divisible by augmentation=20, got: $MAX_PRODUCTS" >&2
    exit 1
  fi

  SOURCE_ROWS="$(wc -l < "$DATA/evaluation_v2/dev_unique1000_aug20/src.txt")"
  TARGET_ROWS="$(wc -l < "$DATA/evaluation_v2/dev_unique1000_aug20/tgt.txt")"
  if [ "$SOURCE_ROWS" -ne 20000 ] || [ "$TARGET_ROWS" -ne 20000 ] || [ "$MAX_PRODUCTS" -gt "$SOURCE_ROWS" ]; then
    echo "Invalid dev_unique1000_aug20 layout: src=$SOURCE_ROWS, tgt=$TARGET_ROWS, requested=$MAX_PRODUCTS" >&2
    exit 1
  fi
fi

if [ "$SMOKE_MODE" -eq 1 ]; then
  SAVE_ROOT="${SMOKE_SAVE_ROOT:-${SAVE_ROOT}_smoke}"
  TRAIN_LOG="logs/train_spe_m500_product_memory_smoke_${RUN_TAG}.log"
else
  TRAIN_LOG="logs/train_spe_m500_product_memory_${RUN_TAG}.log"
fi

echo "Starting training with ${CONFIG_TO_RUN}"
"$PYTHON_BIN" scripts/train_retro.py \
  --config "$CONFIG_TO_RUN" \
  --device cuda \
  --save_dir "$SAVE_ROOT" \
  2>&1 | tee "$TRAIN_LOG"

RUN_DIR="$(find "$SAVE_ROOT/$DATA_NAME" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr \
  | head -n1 \
  | cut -d' ' -f2-)"
if [ -z "$RUN_DIR" ]; then
  echo "No run directory found under $SAVE_ROOT/$DATA_NAME" >&2
  exit 1
fi

if [ "$SMOKE_MODE" -eq 1 ]; then
  SMOKE_CKPT="$RUN_DIR/checkpoint_step${SMOKE_STEPS}.pt"
  if [ ! -f "$SMOKE_CKPT" ]; then
    echo "Smoke checkpoint not found: $SMOKE_CKPT" >&2
    exit 1
  fi
  echo "Smoke passed: ${SMOKE_CKPT}"
  exit 0
fi

CHECKPOINTS=()
for step in $EVAL_STEPS; do
  CHECKPOINTS+=("$RUN_DIR/checkpoint_step${step}.pt")
done

EVAL_TAG="$(date +%Y%m%d_%H%M%S)"
for eval_ckpt in "${CHECKPOINTS[@]}"; do
  if [ ! -f "$eval_ckpt" ]; then
    echo "Checkpoint not found: $eval_ckpt" >&2
    exit 1
  fi

  STEP_NAME="$(basename "$eval_ckpt" .pt)"
  OUT="results/spe_m500_product_memory_${STEP_NAME}_dev_unique1000_euler_n9_seed${EVAL_SEED}_${EVAL_TAG}"
  EVAL_LOG="logs/spe_m500_product_memory_${STEP_NAME}_dev1000_euler_n9_seed${EVAL_SEED}_${EVAL_TAG}.log"

  echo "Evaluating ${eval_ckpt}"
  "$PYTHON_BIN" scripts/eval.py \
    --checkpoint "$eval_ckpt" \
    --products_file "$DATA/evaluation_v2/dev_unique1000_aug20/src.txt" \
    --targets "$DATA/evaluation_v2/dev_unique1000_aug20/tgt.txt" \
    --data_dir "$DATA" \
    --vocab_file "$DATA/example.vocab.src" \
    --output_dir "$OUT" \
    --sampler euler \
    --n_samples 9 \
    --n_steps 100 \
    --scheduler cubic \
    --batch_size 32 \
    --device cuda \
    --seed "$EVAL_SEED" \
    --augmentation 20 \
    --max_products "$MAX_PRODUCTS" \
    --n_best 10 \
    --process_number "$PROCESS_NUMBER" \
    2>&1 | tee "$EVAL_LOG"
done

echo "Done"
echo "Evaluated checkpoints: ${CHECKPOINTS[*]}"

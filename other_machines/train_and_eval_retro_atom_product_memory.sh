#!/usr/bin/env bash
# Train Atom-level + immutable product memory on improved/global R-SMILES,
# then evaluate selected checkpoints on the complete test set with R9K1M2.
#
# Normal run:
#   bash other_machines/train_and_eval_retro_atom_product_memory.sh
#
# Safe GPU smoke test (exactly five optimizer steps; no generation evaluation):
#   SMOKE_STEPS=5 bash other_machines/train_and_eval_retro_atom_product_memory.sh
#
# Useful overrides:
#   EVAL_STEPS="490000 500000 550000 600000" \
#     bash other_machines/train_and_eval_retro_atom_product_memory.sh
#   MAX_PRODUCTS=20000 bash ...  # first 1,000 full-test reactions only
#   CONFIG=... SAVE_ROOT=... DATA=... bash ...

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CONFIG="${CONFIG:-configs/retro_atom_product_memory_600k.yaml}"
SAVE_ROOT="${SAVE_ROOT:-checkpoints/retro_atom_product_memory_600k}"
DATA="${DATA:-datasets/USPTO_50K_PtoR_aug20_#global#}"
DATA_NAME="${DATA_NAME:-$(basename "$DATA")}"
EVAL_STEPS="${EVAL_STEPS:-450000 490000 500000 550000 600000}"
EVAL_SEED="${EVAL_SEED:-42}"
MAX_PRODUCTS="${MAX_PRODUCTS:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
PROCESS_NUMBER="${PROCESS_NUMBER:-12}"
SMOKE_STEPS="${SMOKE_STEPS:-}"

if [ -n "${PYTHON_BIN:-}" ]; then
  : # Explicit override, e.g. PYTHON_BIN=/root/autodl-tmp/ef/bin/python.
elif [ -x /root/autodl-tmp/ef/bin/python ]; then
  PYTHON_BIN=/root/autodl-tmp/ef/bin/python
elif [ -x /root/miniconda3/envs/ef/bin/python ]; then
  PYTHON_BIN=/root/miniconda3/envs/ef/bin/python
else
  PYTHON_BIN=python
fi

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
    echo "Run: git lfs pull --include=\"datasets/USPTO_50K_PtoR_aug20_#global#/**\"" >&2
    exit 1
  fi
}

for required in \
  "$DATA/example.vocab.src" \
  "$DATA/train/train_aligned_src.txt" \
  "$DATA/train/train_aligned_tgt.txt" \
  "$DATA/val/val_aligned_src.txt" \
  "$DATA/val/val_aligned_tgt.txt" \
  "$DATA/val/src-val.txt" \
  "$DATA/val/tgt-val.txt" \
  "$DATA/test/src-test.txt" \
  "$DATA/test/tgt-test.txt"; do
  require_hydrated_file "$required"
done

if [ -n "$MAX_PRODUCTS" ]; then
  case "$MAX_PRODUCTS" in
    ''|*[!0-9]*)
      echo "MAX_PRODUCTS must be a positive integer, got: $MAX_PRODUCTS" >&2
      exit 1
      ;;
  esac
  if [ "$MAX_PRODUCTS" -le 0 ] || [ $((MAX_PRODUCTS % 20)) -ne 0 ]; then
    echo "MAX_PRODUCTS must be positive and divisible by 20, got: $MAX_PRODUCTS" >&2
    exit 1
  fi
fi

SOURCE_ROWS="$(wc -l < "$DATA/test/src-test.txt")"
TARGET_ROWS="$(wc -l < "$DATA/test/tgt-test.txt")"
if [ "$SOURCE_ROWS" -ne 100140 ] || [ "$TARGET_ROWS" -ne 100140 ]; then
  echo "Expected full test layout of 100140 rows; got src=$SOURCE_ROWS, tgt=$TARGET_ROWS" >&2
  exit 1
fi
if [ -n "$MAX_PRODUCTS" ] && [ "$MAX_PRODUCTS" -gt "$SOURCE_ROWS" ]; then
  echo "MAX_PRODUCTS=$MAX_PRODUCTS exceeds full test rows=$SOURCE_ROWS" >&2
  exit 1
fi

"$PYTHON_BIN" - "$CONFIG" "$DATA" "$EVAL_STEPS" "$SMOKE_STEPS" <<'PY'
from pathlib import Path
import sys

import torch
import yaml

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text())
cfg = config.get("retro", config)
requested_data = sys.argv[2]
if cfg.get("data_dir") != requested_data:
    raise SystemExit(
        "Config/data mismatch: "
        f"config data_dir={cfg.get('data_dir')!r}, DATA={requested_data!r}"
    )
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; activate the ef environment first")
if not bool(cfg.get("use_product_memory", False)):
    raise SystemExit("Atom product-memory config must set use_product_memory: true")
if int(cfg.get("product_memory_encoder_layers", 0)) != 2:
    raise SystemExit(
        "This control requires product_memory_encoder_layers: 2; got "
        f"{cfg.get('product_memory_encoder_layers')!r}"
    )
if list(cfg.get("product_memory_fusion_after_layers") or []) != [5, 10]:
    raise SystemExit(
        "This control requires product_memory_fusion_after_layers: [5, 10]; got "
        f"{cfg.get('product_memory_fusion_after_layers')!r}"
    )
if bool(cfg.get("use_origin_mask", False)):
    raise SystemExit("Keep use_origin_mask: false for a clean product-memory ablation")
if float(cfg.get("max_grad_norm", -1.0)) != 0.0:
    raise SystemExit(
        "This Atom control requires max_grad_norm: 0.0; got "
        f"{cfg.get('max_grad_norm')!r}"
    )
if int(cfg.get("max_seq_len", 0)) < 189:
    raise SystemExit(
        "Global atom pre-aligned pairs require max_seq_len >= 189; got "
        f"{cfg.get('max_seq_len')!r}"
    )

smoke_steps = sys.argv[4]
if smoke_steps:
    if not smoke_steps.isdigit() or int(smoke_steps) <= 0:
        raise SystemExit(f"SMOKE_STEPS must be a positive integer, got {smoke_steps!r}")
    print(f"Smoke mode requested: {int(smoke_steps)} optimizer steps")
else:
    try:
        eval_steps = [int(step) for step in sys.argv[3].split()]
    except ValueError as exc:
        raise SystemExit(f"Invalid EVAL_STEPS: {sys.argv[3]!r}") from exc
    total_steps = int(cfg.get("total_steps", 0))
    interval = int(cfg.get("checkpoint_interval", 0))
    if not eval_steps or any(step <= 0 or step > total_steps for step in eval_steps):
        raise SystemExit(
            f"EVAL_STEPS must be within 1..{total_steps}, got {eval_steps}"
        )
    if interval <= 0 or any(step % interval for step in eval_steps):
        raise SystemExit(
            "Every EVAL_STEPS entry must be a multiple of "
            f"checkpoint_interval={interval}, got {eval_steps}"
        )
    keep = int(cfg.get("keep_checkpoints", 0))
    all_steps = list(range(interval, total_steps + 1, interval))
    retained = all_steps[-keep:] if keep > 0 else []
    missing = sorted(set(eval_steps).difference(retained))
    if missing:
        raise SystemExit(
            "keep_checkpoints would prune requested evaluation steps: "
            f"{missing}; increase keep_checkpoints or reduce EVAL_STEPS"
        )
    print(f"Evaluation steps: {eval_steps}")
    print(f"Checkpoint retention covers: {retained[0]}..{retained[-1]}")

print(f"Python: {sys.executable} ({sys.version.split()[0]})")
print(f"Torch: {torch.__version__}; GPU: {torch.cuda.get_device_name(0)}")
print(f"Config/data consistency: {cfg['data_dir']}")
print(
    "Atom product memory: encoder_layers="
    f"{cfg['product_memory_encoder_layers']}; fusion_after="
    f"{cfg['product_memory_fusion_after_layers']}"
)
print(
    "Training shape: "
    f"batch_size={cfg.get('batch_size')}; max_seq_len={cfg.get('max_seq_len')}; "
    f"dropout={cfg.get('dropout')}"
)
PY

if [ -n "$SMOKE_STEPS" ]; then
  SMOKE_CONFIG="$(mktemp /tmp/retro_atom_product_memory_smoke.XXXXXX.yaml)"
  SMOKE_SAVE_ROOT="${SMOKE_SAVE_ROOT:-${SAVE_ROOT}_smoke}"
  SMOKE_LOG="logs/train_atom_product_memory_smoke_${RUN_TAG}.log"
  trap 'rm -f "$SMOKE_CONFIG"' EXIT

  "$PYTHON_BIN" - "$CONFIG" "$SMOKE_CONFIG" "$SMOKE_STEPS" <<'PY'
from pathlib import Path
import sys

import yaml

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
steps = int(sys.argv[3])
payload = yaml.safe_load(source.read_text())
cfg = payload.get("retro")
if not isinstance(cfg, dict):
    raise SystemExit(f"Missing retro block in {source}")
cfg["total_steps"] = steps
cfg["checkpoint_interval"] = steps
cfg["keep_checkpoints"] = 1
cfg["save_best_checkpoint"] = False
tensorboard = cfg.setdefault("tensorboard", {})
tensorboard["enabled"] = False
tensorboard["log_interval"] = 1
tensorboard["validation_interval"] = 0
monitoring = cfg.setdefault("monitoring", {})
monitoring["enabled"] = False
destination.write_text(yaml.safe_dump(payload, sort_keys=False))
print(f"Created {destination} for a {steps}-step smoke test")
PY

  echo "Starting ${SMOKE_STEPS}-step Atom product-memory smoke test"
  "$PYTHON_BIN" scripts/train_retro.py \
    --config "$SMOKE_CONFIG" \
    --device cuda \
    --save_dir "$SMOKE_SAVE_ROOT" \
    2>&1 | tee "$SMOKE_LOG"

  RUN_DIR="$(sed -n 's/.*Checkpoint dir: //p' "$SMOKE_LOG" | tail -n1)"
  SMOKE_CKPT="$RUN_DIR/checkpoint_step${SMOKE_STEPS}.pt"
  if [ -z "$RUN_DIR" ] || [ ! -f "$SMOKE_CKPT" ]; then
    echo "Smoke checkpoint not found: ${SMOKE_CKPT:-<empty>}" >&2
    exit 1
  fi
  echo "Smoke passed: $SMOKE_CKPT"
  echo "The formal 600K run was not started."
  exit 0
fi

TRAIN_LOG="logs/train_atom_product_memory_${RUN_TAG}.log"
echo "Starting training with $CONFIG"
"$PYTHON_BIN" scripts/train_retro.py \
  --config "$CONFIG" \
  --device cuda \
  --save_dir "$SAVE_ROOT" \
  2>&1 | tee "$TRAIN_LOG"

RUN_DIR="$(sed -n 's/.*Checkpoint dir: //p' "$TRAIN_LOG" | tail -n1)"
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
  echo "Could not recover the checkpoint directory from $TRAIN_LOG" >&2
  exit 1
fi

echo "Training complete; starting full-test R9K1M2 checkpoint sweep"
RUN_DIR="$RUN_DIR" \
  DATA="$DATA" \
  SAVE_ROOT="$SAVE_ROOT" \
  STEPS="$EVAL_STEPS" \
  EVAL_SEED="$EVAL_SEED" \
  BATCH_SIZE="$EVAL_BATCH_SIZE" \
  PROCESS_NUMBER="$PROCESS_NUMBER" \
  MAX_PRODUCTS="$MAX_PRODUCTS" \
  bash other_machines/eval_atom_product_memory_r9k1m2_full_checkpoints.sh

echo "Complete: Atom product-memory training and full-test R9K1M2 evaluation"
echo "Run directory: $RUN_DIR"

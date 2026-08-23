#!/usr/bin/env bash
# Original-R-SMILES Full-SPE unclipped control.
# Requires: conda activate /root/autodl-tmp/ef
set -euo pipefail

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export PATH=/root/autodl-tmp/ef/bin:$PATH
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CONFIG="${CONFIG:-configs/retro_spe_full_rsmiles_600k_bs256.yaml}"
SAVE_ROOT="${SAVE_ROOT:-checkpoints/retro_ori_rsmiles_spe_full_600k_unclipped}"
DATA="${DATA:-datasets/USPTO_50K_PtoR_aug20_SPE_full}"
DATA_NAME="${DATA_NAME:-$(basename "$DATA")}"
MANIFEST="${MANIFEST:-datasets/USPTO_50K_PtoR_aug20_#global#/evaluation_v2/manifest.json}"
EVAL_STEPS="${EVAL_STEPS:-450000 490000 500000 550000 600000}"
EVAL_SEED="${EVAL_SEED:-42}"
MAX_PRODUCTS="${MAX_PRODUCTS:-20000}"
PROCESS_NUMBER="${PROCESS_NUMBER:-12}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="logs/train_ori_rsmiles_spe_full_unclipped_${RUN_TAG}.log"
EVAL_DIR="$DATA/evaluation_v2/dev_unique1000_aug20"

mkdir -p logs results

if [ ! -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi
if [ ! -f "$MANIFEST" ]; then
  echo "Global evaluation manifest not found: $MANIFEST" >&2
  exit 1
fi

for required in \
  "$DATA/example.vocab.src" \
  "$DATA/train/train_aligned_src.txt" \
  "$DATA/train/train_aligned_tgt.txt" \
  "$DATA/val/val_aligned_src.txt" \
  "$DATA/val/val_aligned_tgt.txt" \
  "$DATA/val/src-val.txt" \
  "$DATA/val/tgt-val.txt"; do
  if [ ! -f "$required" ]; then
    echo "Required file not found: $required" >&2
    exit 1
  fi
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

"$PYTHON_BIN" - "$CONFIG" "$DATA" "$EVAL_STEPS" <<'PY'
from pathlib import Path
import sys

import torch
import yaml

if sys.version_info < (3, 9):
    raise SystemExit(f"Python >= 3.9 required, got {sys.version}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; activate /root/autodl-tmp/ef before running")

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text())
config = config.get("retro", config)
configured_data = config.get("data_dir")
requested_data = sys.argv[2]
if configured_data != requested_data:
    raise SystemExit(
        "Config/evaluation data mismatch: "
        f"config data_dir={configured_data!r}, DATA={requested_data!r}"
    )
if float(config.get("max_grad_norm", -1.0)) != 0.0:
    raise SystemExit(
        "This unclipped control requires max_grad_norm: 0.0; "
        f"got {config.get('max_grad_norm')!r}"
    )

try:
    eval_steps = [int(step) for step in sys.argv[3].split()]
except ValueError as exc:
    raise SystemExit(f"Invalid EVAL_STEPS: {sys.argv[3]!r}") from exc
total_steps = int(config.get("total_steps", 0))
checkpoint_interval = int(config.get("checkpoint_interval", 0))
if not eval_steps or any(step <= 0 or step > total_steps for step in eval_steps):
    raise SystemExit(
        f"EVAL_STEPS must be within 1..{total_steps}, got {eval_steps}"
    )
if checkpoint_interval <= 0 or any(step % checkpoint_interval for step in eval_steps):
    raise SystemExit(
        "Each EVAL_STEPS value must be a multiple of checkpoint_interval="
        f"{checkpoint_interval}, got {eval_steps}"
    )

print(f"Python: {sys.executable} ({sys.version.split()[0]})")
print(f"Torch: {torch.__version__}; GPU: {torch.cuda.get_device_name(0)}")
print(f"Config/data consistency: {configured_data}")
print(f"Unclipped control: max_grad_norm={config['max_grad_norm']}")
print(f"Evaluation steps: {eval_steps}")
PY

"$PYTHON_BIN" - "$DATA" "$MANIFEST" "$EVAL_DIR" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

data = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text())
out = Path(sys.argv[3])
augmentation = manifest["augmentation"]
indices = manifest["splits"]["dev_unique1000_aug20"]["original_reaction_indices"]

if augmentation != 20 or len(indices) != 1000 or len(set(indices)) != 1000:
    raise SystemExit(
        "Unexpected manifest layout: "
        f"augmentation={augmentation}, reactions={len(indices)}, "
        f"unique={len(set(indices))}"
    )

out.mkdir(parents=True, exist_ok=True)
for side in ("src", "tgt"):
    rows = (data / "val" / f"{side}-val.txt").read_text().splitlines()
    if len(rows) % augmentation:
        raise SystemExit(f"{side}-val.txt does not contain complete augmentation blocks")
    if min(indices) < 0 or max(indices) >= len(rows) // augmentation:
        raise SystemExit(f"Manifest indices are outside {side}-val.txt")
    selected = [
        item
        for index in indices
        for item in rows[index * augmentation:(index + 1) * augmentation]
    ]
    if len(selected) != 20000:
        raise SystemExit(f"Expected 20000 {side} rows, got {len(selected)}")
    expected = ("\n".join(selected) + "\n").encode()
    destination = out / f"{side}.txt"
    if destination.exists():
        actual = destination.read_bytes()
        if actual != expected:
            raise SystemExit(
                f"Refusing to overwrite non-matching existing evaluation file: "
                f"{destination}"
            )
        status = "verified existing"
    else:
        destination.write_bytes(expected)
        status = "created"
    print(
        f"{status}: {destination} | rows=20000 | "
        f"sha256={hashlib.sha256(expected).hexdigest()}"
    )
PY

SOURCE_ROWS="$(wc -l < "$EVAL_DIR/src.txt")"
TARGET_ROWS="$(wc -l < "$EVAL_DIR/tgt.txt")"
if [ "$SOURCE_ROWS" -ne "$TARGET_ROWS" ] || [ "$MAX_PRODUCTS" -gt "$SOURCE_ROWS" ]; then
  echo "Invalid evaluation layout: src=$SOURCE_ROWS, tgt=$TARGET_ROWS, requested=$MAX_PRODUCTS" >&2
  exit 1
fi

echo "Starting training with ${CONFIG}"
"$PYTHON_BIN" scripts/train_retro.py \
  --config "$CONFIG" \
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
  OUT="results/ori_rsmiles_spe_full_unclipped_${STEP_NAME}_dev_unique1000_euler_n9_seed${EVAL_SEED}_${EVAL_TAG}"
  EVAL_LOG="logs/ori_rsmiles_spe_full_unclipped_${STEP_NAME}_dev1000_euler_n9_seed${EVAL_SEED}_${EVAL_TAG}.log"

  echo "Evaluating ${eval_ckpt}"
  "$PYTHON_BIN" scripts/eval.py \
    --checkpoint "$eval_ckpt" \
    --products_file "$EVAL_DIR/src.txt" \
    --targets "$EVAL_DIR/tgt.txt" \
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

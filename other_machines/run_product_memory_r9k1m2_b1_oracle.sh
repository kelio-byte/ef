#!/usr/bin/env bash
# Full-test upper-bound diagnostic: product-memory + oracle reaction-center
# first-event bias under the frozen R9K1M2 sampler.
#
# This is deliberately a single pre-specified primary test.  STEP defaults to
# PM@500K because that checkpoint was selected from dev results before this
# combined full-test evaluation.  The true center is derived from the answer,
# so this remains an oracle upper bound, not a deployable method.
#
# Run a cheap compatibility check first:
#   conda activate ef
#   cd /root/autodl-tmp/edit_flows
#   CHECKPOINT=/path/to/checkpoint_step500000.pt \
#     bash other_machines/run_product_memory_r9k1m2_b1_oracle.sh smoke
#
# Then run the one full primary test:
#   ALLOW_FULL_PM_CENTER_ORACLE_TEST=YES \
#   CHECKPOINT=/path/to/checkpoint_step500000.pt \
#     bash other_machines/run_product_memory_r9k1m2_b1_oracle.sh full
#
# Optional overrides:
#   RUN_DIR=checkpoints/retro_spe_m500_product_memory_600k/\
#USPTO_50K_PtoR_aug20_#global#_SPE_m500/<timestamp> STEP=500000 \
#   ALLOW_FULL_PM_CENTER_ORACLE_TEST=YES \
#   bash other_machines/run_product_memory_r9k1m2_b1_oracle.sh full

set -euo pipefail

MODE="${1:-smoke}"
case "$MODE" in
  smoke|full) ;;
  *)
    echo "Usage: $0 {smoke|full}" >&2
    exit 2
    ;;
esac

if [[ "$MODE" == "full" && "${ALLOW_FULL_PM_CENTER_ORACLE_TEST:-}" != "YES" ]]; then
  echo "Refusing full PM + oracle-center evaluation without acknowledgement." >&2
  echo "Re-run with:" >&2
  echo "  ALLOW_FULL_PM_CENTER_ORACLE_TEST=YES bash $0 full" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

DATA_DIR="${DATA_DIR:-datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500}"
SIDECAR="${SIDECAR:-after_spe/center_sidecars/test_all_aug20}"
SAVE_ROOT="${SAVE_ROOT:-checkpoints/retro_spe_m500_product_memory_600k}"
DATA_NAME="${DATA_NAME:-$(basename "$DATA_DIR")}"
RUN_DIR="${RUN_DIR:-}"
CHECKPOINT="${CHECKPOINT:-}"
STEP="${STEP:-500000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/after_spe_product_memory/r9k1m2_b1_oracle}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PROCESS_NUMBER="${PROCESS_NUMBER:-12}"
SEED="${SEED:-42}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  : # Explicit override.
elif [[ -x /root/autodl-tmp/ef/bin/python ]]; then
  PYTHON_BIN=/root/autodl-tmp/ef/bin/python
elif [[ -x /root/miniconda3/envs/ef/bin/python ]]; then
  PYTHON_BIN=/root/miniconda3/envs/ef/bin/python
else
  PYTHON_BIN=python
fi

case "$STEP" in
  ''|*[!0-9]*)
    echo "STEP must be a positive integer, got: $STEP" >&2
    exit 1
    ;;
esac
if [[ "$STEP" -le 0 ]]; then
  echo "STEP must be positive, got: $STEP" >&2
  exit 1
fi

for name in BATCH_SIZE PROCESS_NUMBER SEED; do
  value="${!name}"
  case "$value" in
    ''|*[!0-9]*)
      echo "$name must be a positive integer, got: $value" >&2
      exit 1
      ;;
  esac
  if [[ "$value" -le 0 ]]; then
    echo "$name must be positive, got: $value" >&2
    exit 1
  fi
done

resolve_checkpoint() {
  if [[ -n "$CHECKPOINT" ]]; then
    return
  fi
  if [[ -n "$RUN_DIR" ]]; then
    CHECKPOINT="$RUN_DIR/checkpoint_step${STEP}.pt"
    return
  fi

  local candidates=()
  mapfile -t candidates < <(
    find "$SAVE_ROOT/$DATA_NAME" -mindepth 2 -maxdepth 2 \
      -type f -name "checkpoint_step${STEP}.pt" -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | cut -d' ' -f2-
  )
  if [[ "${#candidates[@]}" -eq 0 ]]; then
    echo "Could not find PM checkpoint_step${STEP}.pt under $SAVE_ROOT/$DATA_NAME" >&2
    echo "Set CHECKPOINT=/path/to/checkpoint_step${STEP}.pt explicitly." >&2
    exit 1
  fi
  CHECKPOINT="${candidates[0]}"
  if [[ "${#candidates[@]}" -gt 1 ]]; then
    echo "Multiple PM checkpoints found; using newest: $CHECKPOINT" >&2
    echo "Set CHECKPOINT explicitly to avoid ambiguity." >&2
  fi
}

PRODUCTS="$DATA_DIR/test/src-test.txt"
TARGETS="$DATA_DIR/test/tgt-test.txt"
for required in \
  "$PRODUCTS" \
  "$TARGETS" \
  "$DATA_DIR/example.vocab.src" \
  "$SIDECAR/metadata.json" \
  "$SIDECAR/scores.jsonl"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

SOURCE_ROWS="$(wc -l < "$PRODUCTS")"
TARGET_ROWS="$(wc -l < "$TARGETS")"
if [[ "$SOURCE_ROWS" -ne 100140 || "$TARGET_ROWS" -ne 100140 ]]; then
  echo "Expected full test layout of 100140 augmentation rows, got src=$SOURCE_ROWS tgt=$TARGET_ROWS" >&2
  exit 1
fi

resolve_checkpoint
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

# Refuse accidental use of an ordinary M500 checkpoint: the sampling code will
# automatically activate product memory only when this checkpoint records it.
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
state = checkpoint.get("model_state_dict", {})
if not bool(config.get("use_product_memory", False)):
    raise SystemExit(f"{path}: config does not enable product memory")
if not any(key.startswith("product_memory_encoder_layers.") for key in state):
    raise SystemExit(f"{path}: product-memory encoder weights are missing")
print(
    f"Verified product-memory checkpoint: {path.name} | "
    f"encoder_layers={config.get('product_memory_encoder_layers')} | "
    f"fusion_after={config.get('product_memory_fusion_after_layers')}"
)
PY

# The sidecar is valid only for the exact augmented input rows.  This catches
# accidental mixing of a dev sidecar, an old dataset, or different R-SMILES.
"$PYTHON_BIN" - "$SIDECAR/metadata.json" "$PRODUCTS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

metadata_path, products_path = map(Path, sys.argv[1:])
metadata = json.loads(metadata_path.read_text())
expected = metadata["files"]["m500_products"]["sha256"]
actual = hashlib.sha256(products_path.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"sidecar/input SHA256 mismatch: {actual} != {expected}")
rows = len(products_path.read_text().splitlines())
if metadata["input_row_count"] != rows:
    raise SystemExit(
        f"sidecar/input row-count mismatch: {metadata['input_row_count']} != {rows}"
    )
print(f"Sidecar preflight: rows={rows}, input SHA256={actual}")
PY

"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; activate the ef environment first")
print(f"Torch: {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")
PY

if [[ "$MODE" == "smoke" ]]; then
  MAX_PRODUCTS=20
  DIAGNOSTIC_DETAIL=full
else
  MAX_PRODUCTS=""
  DIAGNOSTIC_DETAIL=summary
fi

OUTPUT_DIR="$OUTPUT_ROOT/pm_step${STEP}_${MODE}_${RUN_TAG}"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to reuse output directory: $OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUTPUT_DIR")"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --products_file "$PRODUCTS"
  --targets "$TARGETS"
  --data_dir "$DATA_DIR"
  --vocab_file "$DATA_DIR/example.vocab.src"
  --output_dir "$OUTPUT_DIR"
  --sampler euler_beam
  --n_samples 3
  --n_runs 9
  --n_branches 1
  --n_children 2
  --n_steps 100
  --scheduler cubic
  --batch_size "$BATCH_SIZE"
  --device cuda
  --seed "$SEED"
  --augmentation 20
  --n_best 10
  --process_number "$PROCESS_NUMBER"
  --euler_beam_score_mode full_probability
  --euler_beam_changed_state_bonus 0.5
  --euler_beam_q_temperature 1.0
  --euler_beam_matmul_precision high
  --euler_beam_child_policy stochastic_noop
  --first_event_center_sidecar "$SIDECAR"
  --first_event_center_source oracle
  --first_event_center_max_multiplier 3.0
  --first_event_center_diagnostics "$DIAGNOSTIC_DETAIL"
)
if [[ -n "$MAX_PRODUCTS" ]]; then
  ARGS+=(--max_products "$MAX_PRODUCTS")
fi

echo "Project:     $PROJECT_ROOT"
echo "Mode:        $MODE"
echo "Python:      $PYTHON_BIN"
echo "Checkpoint:  $CHECKPOINT"
echo "Data:        $DATA_DIR"
echo "Sidecar:     $SIDECAR"
echo "Inputs:      $SOURCE_ROWS rows / $((SOURCE_ROWS / 20)) reactions"
echo "Sampler:     R9K1M2; 9 independent runs, 2 children/run"
echo "Center:      oracle first-event position bias only; multiplier=3"
echo "Memory:      cached immutable product x0 per input view"
echo "Output:      $OUTPUT_DIR"

echo "Starting PM + oracle-center ${MODE} evaluation: $(date -u --iso-8601=seconds)"
"$PYTHON_BIN" scripts/eval.py "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}.log"
echo "Finished: $(date -u --iso-8601=seconds)"
echo "Output: $OUTPUT_DIR"

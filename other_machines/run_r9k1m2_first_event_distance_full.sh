#!/usr/bin/env bash
# Full-test first-event distance diagnostic for SPE-M500@490K + R9K1M2.
#
# This runs two otherwise identical samplers:
#   B0-trace: oracle sidecar present but multiplier=1.0 (bitwise-neutral)
#   B1:       true-center oracle position bias, multiplier=3.0
#
# It reports token-distance progress after each selected lineage's first
# non-empty Euler step.  It does NOT produce a new Top-k evaluation.
set -euo pipefail

MODE="${1:-smoke}"
case "$MODE" in
  smoke|full) ;;
  *)
    echo "Usage: $0 {smoke|full}" >&2
    exit 2
    ;;
esac

if [[ "$MODE" == "full" && "${ALLOW_FULL_R9_FIRST_EVENT_DISTANCE:-}" != "YES" ]]; then
  cat >&2 <<'EOF'
Refusing the full diagnostic without an explicit acknowledgement.
It runs B0-trace and B1 across 100,140 augmentation views (about 5–6 hours
sequentially on one 3090).  Re-run with:

  ALLOW_FULL_R9_FIRST_EVENT_DISTANCE=YES bash other_machines/run_r9k1m2_first_event_distance_full.sh full
EOF
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x /root/autodl-tmp/ef/bin/python ]]; then
  PYTHON_BIN=/root/autodl-tmp/ef/bin/python
elif [[ -x /root/miniconda3/envs/ef/bin/python ]]; then
  PYTHON_BIN=/root/miniconda3/envs/ef/bin/python
else
  PYTHON_BIN=python
fi

CHECKPOINT="${CHECKPOINT:-new_checkpoints/spe_m500_checkpoints/checkpoint_step490000.pt}"
DATA_DIR="${DATA_DIR:-datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500}"
SIDECAR="${SIDECAR:-after_spe/center_sidecars/test_all_aug20}"
PRODUCTS="$DATA_DIR/test/src-test.txt"
TARGETS="$DATA_DIR/test/tgt-test.txt"
VOCAB_FILE="$DATA_DIR/example.vocab.src"
N_STEPS="${N_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_PARENT="${OUTPUT_PARENT:-results/after_spe_stage1/r9k1m2_first_event_distance}"
OUTPUT_DIR="$OUTPUT_PARENT/${RUN_ID}_${MODE}"
RESUME="${RESUME:-NO}"

if [[ "$MODE" == "smoke" ]]; then
  MAX_PRODUCTS=20  # exactly one reaction × 20 augmentation views
else
  MAX_PRODUCTS=""
fi

for required in \
  "$CHECKPOINT" "$PRODUCTS" "$TARGETS" "$VOCAB_FILE" \
  "$SIDECAR/metadata.json" "$SIDECAR/scores.jsonl"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

PRODUCT_ROWS="$(wc -l < "$PRODUCTS")"
TARGET_ROWS="$(wc -l < "$TARGETS")"
SIDECAR_ROWS="$(wc -l < "$SIDECAR/scores.jsonl")"
if [[ "$PRODUCT_ROWS" != "$TARGET_ROWS" || "$PRODUCT_ROWS" != "$SIDECAR_ROWS" ]]; then
  echo "Line-count mismatch: products=$PRODUCT_ROWS targets=$TARGET_ROWS sidecar=$SIDECAR_ROWS" >&2
  echo "A Git-LFS pointer typically has only a few lines; run git lfs pull for the sidecar." >&2
  exit 1
fi
if [[ "$PRODUCT_ROWS" != "100140" ]]; then
  echo "Expected full M500 test to contain 100140 views; found $PRODUCT_ROWS" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print("Torch:", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
PY

mkdir -p "$OUTPUT_PARENT"
if [[ -e "$OUTPUT_DIR" && "$RESUME" != "YES" ]]; then
  echo "Output directory already exists: $OUTPUT_DIR" >&2
  echo "Choose a new RUN_ID, or use RESUME=YES only to reuse completed condition summaries." >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
df -h "$OUTPUT_PARENT"

echo "Project:    $PROJECT_ROOT"
echo "Python:     $PYTHON_BIN"
echo "Checkpoint: $CHECKPOINT"
echo "Data:       $DATA_DIR"
echo "Sidecar:    $SIDECAR"
echo "Mode:       $MODE"
echo "Views:      ${MAX_PRODUCTS:-$PRODUCT_ROWS}"
echo "Output:     $OUTPUT_DIR"
echo "Protocol:   R9K1M2, n_steps=$N_STEPS, batch_size=$BATCH_SIZE, seed=$SEED"

MAX_ARGS=()
if [[ -n "$MAX_PRODUCTS" ]]; then
  MAX_ARGS=(--max_products "$MAX_PRODUCTS")
fi

run_condition() {
  local condition="$1"
  local multiplier="$2"
  local neutral_flag="$3"
  local condition_dir="$OUTPUT_DIR/$condition"
  local summary="$condition_dir/first_event_distance.json"
  local log="$OUTPUT_DIR/${condition}.log"
  mkdir -p "$condition_dir"

  if [[ -f "$summary" ]]; then
    if [[ "$RESUME" == "YES" ]]; then
      echo "Reusing completed $condition summary: $summary"
      return 0
    fi
    echo "Refusing to overwrite completed condition: $summary" >&2
    exit 1
  fi

  local neutral_args=()
  if [[ "$neutral_flag" == "YES" ]]; then
    neutral_args=(--assert_neutral)
  fi
  echo
  echo "============================================================"
  echo "Starting $condition (multiplier=$multiplier): $(date -u --iso-8601=seconds)"
  echo "============================================================"
  set -o pipefail
  "$PYTHON_BIN" scripts/diagnose_r9k1m2_first_event_distance.py \
    --checkpoint "$CHECKPOINT" \
    --products_file "$PRODUCTS" \
    --targets "$TARGETS" \
    --data_dir "$DATA_DIR" \
    --vocab_file "$VOCAB_FILE" \
    --sidecar "$SIDECAR" \
    --output_json "$summary" \
    --condition "$condition" \
    --max_multiplier "$multiplier" \
    --n_steps "$N_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --device cuda \
    --seed "$SEED" \
    --scheduler cubic \
    "${MAX_ARGS[@]}" \
    "${neutral_args[@]}" \
    2>&1 | tee "$log"
  echo "Finished $condition: $(date -u --iso-8601=seconds)"
}

run_condition b0_trace 1.0 YES
run_condition b1_oracle 3.0 NO

COMPARISON_JSON="$OUTPUT_DIR/comparison.json"
COMPARISON_MD="$OUTPUT_DIR/summary.md"
if [[ -f "$COMPARISON_JSON" || -f "$COMPARISON_MD" ]]; then
  if [[ "$RESUME" != "YES" ]]; then
    echo "Refusing to overwrite existing comparison output in $OUTPUT_DIR" >&2
    exit 1
  fi
  echo "Comparison already exists; leaving it unchanged."
else
  "$PYTHON_BIN" scripts/compare_first_event_distance.py \
    --baseline "$OUTPUT_DIR/b0_trace/first_event_distance.json" \
    --candidate "$OUTPUT_DIR/b1_oracle/first_event_distance.json" \
    --output_json "$COMPARISON_JSON" \
    --output_markdown "$COMPARISON_MD"
fi

echo
echo "Complete: $OUTPUT_DIR"
echo "Read the concise result: $COMPARISON_MD"

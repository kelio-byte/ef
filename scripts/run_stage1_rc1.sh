#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
case "$MODE" in
  smoke)
    MAX_PRODUCTS=200
    ;;
  pilot100)
    MAX_PRODUCTS=2000
    ;;
  dev1000)
    MAX_PRODUCTS=20000
    ;;
  *)
    echo "Usage: $0 {smoke|pilot100|dev1000}" >&2
    exit 2
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# The local AutoDL setup keeps the project environment here.  Prefer it when
# the caller did not explicitly provide a Python executable, but retain the
# override for a different machine or conda environment.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -x /root/miniconda3/envs/ef/bin/python ]]; then
  PYTHON_BIN="/root/miniconda3/envs/ef/bin/python"
elif [[ -x /root/autodl-tmp/ef/bin/python ]]; then
  PYTHON_BIN="/root/autodl-tmp/ef/bin/python"
else
  PYTHON_BIN="python"
fi
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/after_spe_stage1/rc1_runs/${RUN_ID}_${MODE}}"
CHECKPOINT="new_checkpoints/spe_m500_checkpoints/checkpoint_step490000.pt"
DATA_DIR="datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500"
PRODUCTS="${DATA_DIR}/evaluation_v2/dev_unique1000_aug20/src.txt"
TARGETS="${DATA_DIR}/evaluation_v2/dev_unique1000_aug20/tgt.txt"
VOCAB="${DATA_DIR}/example.vocab.src"
SIDECAR="results/after_spe_stage1/center_sidecars/dev_unique1000_aug20"

for required in \
  "$CHECKPOINT" "$PRODUCTS" "$TARGETS" "$VOCAB" \
  "$SIDECAR/metadata.json" "$SIDECAR/scores.jsonl"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; RC1 must run in GPU mode")
print("Torch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PY

mkdir -p "$OUTPUT_ROOT"

COMMON_ARGS=(
  --checkpoint "$CHECKPOINT"
  --products_file "$PRODUCTS"
  --targets "$TARGETS"
  --data_dir "$DATA_DIR"
  --vocab_file "$VOCAB"
  --sampler euler
  --n_samples 9
  --n_steps 100
  --batch_size 32
  --device cuda
  --seed 42
  --augmentation 20
  --max_products "$MAX_PRODUCTS"
  --n_best 10
  --process_number 12
  --overwrite
)

run_group() {
  local name="$1"
  shift
  echo "Starting ${name}: $(date -u --iso-8601=seconds)"
  "$PYTHON_BIN" scripts/eval.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "$OUTPUT_ROOT/$name" \
    "$@" 2>&1 | tee "$OUTPUT_ROOT/${name}.log"
  echo "Finished ${name}: $(date -u --iso-8601=seconds)"
}

run_group b0_plain
run_group b0_trace \
  --first_event_center_sidecar "$SIDECAR" \
  --first_event_center_source oracle \
  --first_event_center_max_multiplier 1.0

if ! cmp -s \
  "$OUTPUT_ROOT/b0_plain/predictions.txt" \
  "$OUTPUT_ROOT/b0_trace/predictions.txt"; then
  echo "B0 and neutral B0-trace predictions differ; stopping RC1." >&2
  exit 1
fi
echo "B0 neutrality check: predictions are byte-identical"

run_group b1_oracle \
  --first_event_center_sidecar "$SIDECAR" \
  --first_event_center_source oracle \
  --first_event_center_max_multiplier 3.0

run_group b2_pseudo \
  --first_event_center_sidecar "$SIDECAR" \
  --first_event_center_source pseudo \
  --first_event_center_max_multiplier 3.0

echo "RC1 ${MODE} complete: $OUTPUT_ROOT"

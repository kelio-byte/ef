#!/usr/bin/env bash
# Oracle-only B1 reaction-center test for the frozen Global R-SMILES SPE-M500
# R9K1M2 baseline.  The true center is derived from test targets, so this is
# an upper-bound diagnostic, not a deployable inference method.
set -euo pipefail

MODE="${1:-smoke}"
case "$MODE" in
  smoke)
    MAX_PRODUCTS=20       # exactly one aug20 reaction block
    PRODUCTS="datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/evaluation_v2/dev_unique1000_aug20/src.txt"
    TARGETS="datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/evaluation_v2/dev_unique1000_aug20/tgt.txt"
    SIDECAR="results/after_spe_stage1/center_sidecars/dev_unique1000_aug20"
    DIAGNOSTIC_DETAIL=full
    ;;
  full)
    if [[ "${ALLOW_FULL_ORACLE_TEST:-}" != "YES" ]]; then
      echo "Refusing full oracle test. Re-run only after approval with:" >&2
      echo "  ALLOW_FULL_ORACLE_TEST=YES $0 full" >&2
      exit 2
    fi
    MAX_PRODUCTS=""
    PRODUCTS="datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/test/src-test.txt"
    TARGETS="datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/test/tgt-test.txt"
    SIDECAR="results/after_spe_stage1/center_sidecars/test_all_aug20"
    # 100,140 product views × 9 runs would otherwise generate a very large
    # first-event JSON.  Compact counts retain the safety checks cheaply.
    DIAGNOSTIC_DETAIL=summary
    ;;
  *)
    echo "Usage: $0 {smoke|full}" >&2
    exit 2
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -x /root/miniconda3/envs/ef/bin/python ]]; then
  PYTHON_BIN=/root/miniconda3/envs/ef/bin/python
elif [[ -x /root/autodl-tmp/ef/bin/python ]]; then
  PYTHON_BIN=/root/autodl-tmp/ef/bin/python
else
  PYTHON_BIN=python
fi

CHECKPOINT="${CHECKPOINT:-new_checkpoints/spe_m500_checkpoints/checkpoint_step490000.pt}"
DATA_DIR="${DATA_DIR:-datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-results/after_spe_stage1/r9k1m2_b1/${RUN_ID}_${MODE}}"

for required in \
  "$CHECKPOINT" "$PRODUCTS" "$TARGETS" "$DATA_DIR/example.vocab.src" \
  "$SIDECAR/metadata.json" "$SIDECAR/scores.jsonl"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    if [[ "$MODE" == full && "$required" == "$SIDECAR/metadata.json" ]]; then
      printf '%s\n' \
        'Prepare the oracle sidecar first (CPU only; do not use it to tune the model):' \
        '  python scripts/build_reaction_center_labels.py \' \
        '    --processed_dir datasets/USPTO_50K_PtoR_aug20_#global# \' \
        '    --splits test --workers 8' \
        '  python scripts/build_center_bias_sidecar.py --all_processed_blocks \' \
        '    --global_products datasets/USPTO_50K_PtoR_aug20_#global#/test/src-test.txt \' \
        '    --m500_products datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/test/src-test.txt \' \
        '    --raw_csv datasets/USPTO_50K/raw_test.csv \' \
        '    --labels results/after_spe_stage1/cache/reaction_centers_test.jsonl \' \
        '    --crosswalk results/after_spe_stage1/cache/raw_to_processed_test.jsonl \' \
        '    --output_dir results/after_spe_stage1/center_sidecars/test_all_aug20 \' \
        '    --workers 8' >&2
    fi
    exit 1
  fi
done

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
        "sidecar/input row-count mismatch: "
        f"{metadata['input_row_count']} != {rows}"
    )
print(
    "Sidecar preflight:", metadata["evaluation_split"],
    "| rows=", rows,
    "| selection=", metadata.get("selection_kind", "manifest_split"),
)
PY

"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print("Torch:", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
PY

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
  --n_runs 9 --n_branches 1 --n_children 2
  --n_steps 100 --batch_size 64 --device cuda --seed 42 --scheduler cubic
  --euler_beam_score_mode full_probability
  --euler_beam_changed_state_bonus 0.5
  --euler_beam_q_temperature 1.0
  --euler_beam_matmul_precision high
  --euler_beam_child_policy stochastic_noop
  --augmentation 20 --n_best 10 --process_number 12
  --first_event_center_sidecar "$SIDECAR"
  --first_event_center_source oracle
  --first_event_center_max_multiplier 3.0
  --first_event_center_diagnostics "$DIAGNOSTIC_DETAIL"
)
if [[ -n "$MAX_PRODUCTS" ]]; then
  ARGS+=(--max_products "$MAX_PRODUCTS")
fi

echo "Starting ${MODE} B1 oracle R9K1M2 evaluation: $(date -u --iso-8601=seconds)"
"$PYTHON_BIN" scripts/eval.py "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}.log"
echo "Finished: $(date -u --iso-8601=seconds)"
echo "Output: $OUTPUT_DIR"

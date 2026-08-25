#!/usr/bin/env bash
# Run the frozen oracle B1 + R9K1M2 protocol for several SPE-M500 checkpoints.
#
# This is an upper-bound diagnostic: the center sidecar is built from the
# ground-truth reaction, so it is not a deployable inference method.
# The actual sampling protocol lives in scripts/run_r9k1m2_b1_oracle.sh.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ "${ALLOW_FULL_ORACLE_TEST:-}" != "YES" ]]; then
  echo "Refusing to start full oracle B1 evaluations." >&2
  echo "Re-run with:" >&2
  echo "  ALLOW_FULL_ORACLE_TEST=YES bash $0" >&2
  exit 2
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-new_checkpoints/spe_m500_checkpoints}"
DATA_DIR="${DATA_DIR:-datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500}"
SIDECAR="${SIDECAR:-results/after_spe_stage1/center_sidecars/test_all_aug20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/after_spe_stage1/r9k1m2_b1_checkpoints}"

# Override with a space-separated list if needed, e.g. CHECKPOINT_STEPS="490000 550000".
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-480000 500000 510000 520000}"

PRODUCTS="$DATA_DIR/test/src-test.txt"
TARGETS="$DATA_DIR/test/tgt-test.txt"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -x /root/miniconda3/envs/ef/bin/python ]]; then
  PYTHON_BIN=/root/miniconda3/envs/ef/bin/python
elif [[ -x /root/autodl-tmp/ef/bin/python ]]; then
  PYTHON_BIN=/root/autodl-tmp/ef/bin/python
else
  PYTHON_BIN=python
fi

echo "Project: $PROJECT_ROOT"
echo "Python:  $PYTHON_BIN"
echo "Data:    $DATA_DIR"
echo "Sidecar: $SIDECAR"
echo "Steps:   $CHECKPOINT_STEPS"

# Preflight everything before launching the first expensive evaluation.
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

for step in $CHECKPOINT_STEPS; do
  checkpoint="$CHECKPOINT_DIR/checkpoint_step${step}.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
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
        f"sidecar/input row-count mismatch: {metadata['input_row_count']} != {rows}"
    )
print(f"Sidecar preflight: rows={rows}, input SHA256={actual}")
PY

"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(f"Torch: {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")
PY

mkdir -p "$OUTPUT_ROOT"

for step in $CHECKPOINT_STEPS; do
  checkpoint="$CHECKPOINT_DIR/checkpoint_step${step}.pt"
  output_dir="$OUTPUT_ROOT/step${step}_full"

  if [[ -e "$output_dir" ]]; then
    if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
      echo "Skipping existing output: $output_dir"
      continue
    fi
    echo "Refusing to reuse output directory: $output_dir" >&2
    echo "Set SKIP_EXISTING=1 to skip it, or move the incomplete directory." >&2
    exit 1
  fi

  echo
  echo "============================================================"
  echo "Starting B1 oracle R9K1M2 full test for step ${step}K"
  echo "Checkpoint: $checkpoint"
  echo "Output:    $output_dir"
  echo "Started:   $(date -u --iso-8601=seconds)"
  echo "============================================================"

  CHECKPOINT="$checkpoint" \
  DATA_DIR="$DATA_DIR" \
  SIDECAR="$SIDECAR" \
  PRODUCTS="$PRODUCTS" \
  TARGETS="$TARGETS" \
  OUTPUT_DIR="$output_dir" \
  PYTHON_BIN="$PYTHON_BIN" \
  ALLOW_FULL_ORACLE_TEST=YES \
    bash scripts/run_r9k1m2_b1_oracle.sh full

  echo "Finished step ${step}K: $(date -u --iso-8601=seconds)"
done

echo
echo "All requested B1 oracle checkpoint evaluations completed."
echo "Results: $OUTPUT_ROOT"

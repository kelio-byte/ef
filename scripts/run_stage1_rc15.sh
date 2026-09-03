#!/usr/bin/env bash
set -euo pipefail

# Fixed RC1.5 oracle experiment.  It is deliberately not a hyperparameter
# sweep: three of nine trajectories receive the already-frozen true-center
# first-event position bias; the other six are ordinary Euler trajectories.

MODE="${1:-smoke}"
case "$MODE" in
  smoke)
    MAX_PRODUCTS=200
    REUSE_REFERENCE=0
    DEFAULT_EVALUATION_SPLIT="dev_unique1000_aug20"
    DEFAULT_REFERENCE_RC1_ROOT=""
    ;;
  dev1000)
    MAX_PRODUCTS=20000
    REUSE_REFERENCE=1
    DEFAULT_EVALUATION_SPLIT="dev_unique1000_aug20"
    DEFAULT_REFERENCE_RC1_ROOT="results/after_spe_stage1/rc1_runs/20260823T225828Z_dev1000"
    ;;
  confirm1000)
    MAX_PRODUCTS=20000
    REUSE_REFERENCE=1
    DEFAULT_EVALUATION_SPLIT="confirm_unique1000_aug20"
    DEFAULT_REFERENCE_RC1_ROOT="results/after_spe_stage1/rc1_runs/20260824T055400Z_confirm1000"
    ;;
  *)
    echo "Usage: $0 {smoke|dev1000|confirm1000}" >&2
    exit 2
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

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
OUTPUT_ROOT="${OUTPUT_ROOT:-results/after_spe_stage1/rc15_runs/${RUN_ID}_${MODE}}"
REFERENCE_RC1_ROOT="${REFERENCE_RC1_ROOT:-$DEFAULT_REFERENCE_RC1_ROOT}"
CHECKPOINT="new_checkpoints/spe_m500_checkpoints/checkpoint_step490000.pt"
DATA_DIR="datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500"
EVALUATION_SPLIT="${EVALUATION_SPLIT:-$DEFAULT_EVALUATION_SPLIT}"
PRODUCTS="${DATA_DIR}/evaluation_v2/${EVALUATION_SPLIT}/src.txt"
TARGETS="${DATA_DIR}/evaluation_v2/${EVALUATION_SPLIT}/tgt.txt"
VOCAB="${DATA_DIR}/example.vocab.src"
SIDECAR="results/after_spe_stage1/center_sidecars/${EVALUATION_SPLIT}"

for required in \
  "$CHECKPOINT" "$PRODUCTS" "$TARGETS" "$VOCAB" \
  "$SIDECAR/metadata.json" "$SIDECAR/scores.jsonl"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

"$PYTHON_BIN" - "$SIDECAR/metadata.json" "$EVALUATION_SPLIT" <<'PY'
import json
import sys

metadata_path, expected_split = sys.argv[1:]
metadata = json.load(open(metadata_path))
if metadata.get("evaluation_split") != expected_split:
    raise SystemExit(
        "sidecar split mismatch: "
        f"{metadata.get('evaluation_split')!r} != {expected_split!r}"
    )
if metadata.get("input_row_count") < 1:
    raise SystemExit("center sidecar has no input rows")
print(
    "Evaluation split:", expected_split,
    "| sidecar rows:", metadata["input_row_count"],
)
PY

"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; RC1.5 must run in GPU mode")
print("Torch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PY

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to reuse existing output root: $OUTPUT_ROOT" >&2
  exit 1
fi
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

validate_reference() {
  "$PYTHON_BIN" - "$REFERENCE_RC1_ROOT" "$PRODUCTS" "$CHECKPOINT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, products_path, checkpoint_path = map(Path, sys.argv[1:])
groups = ("b0_plain", "b0_trace", "b1_oracle")
for group in groups:
    directory = root / group
    for name in ("predictions.txt", "sampling_metadata.json", "diagnostics.json"):
        if not (directory / name).is_file():
            raise SystemExit(f"reference is missing {directory / name}")
for group in ("b0_trace", "b1_oracle"):
    if not (root / group / "center_bias_diagnostics.json").is_file():
        raise SystemExit(f"reference is missing {group} first-event diagnostics")

source_sha = hashlib.sha256(products_path.read_bytes()).hexdigest()
checkpoint_size = checkpoint_path.stat().st_size
for group in groups:
    metadata = json.loads((root / group / "sampling_metadata.json").read_text())
    sampling = metadata["sampling"]
    assert metadata["sampler"] == "euler", group
    assert metadata["product_count"] == 20000, group
    assert metadata["output_beam_size"] == 9, group
    assert metadata["output_line_count"] == 180000, group
    assert metadata["input"]["sha256"] == source_sha, group
    assert metadata["checkpoint"]["size_bytes"] == checkpoint_size, group
    assert sampling["n_samples"] == 9, group
    assert sampling["n_steps"] == 100, group
    assert sampling["seed"] == 42, group

b1_sampling = json.loads(
    (root / "b1_oracle" / "sampling_metadata.json").read_text()
)["sampling"]
assert b1_sampling["first_event_center_source"] == "oracle"
assert b1_sampling["first_event_center_max_multiplier"] == 3.0
assert (
    root / "b0_plain" / "predictions.txt"
).read_bytes() == (
    root / "b0_trace" / "predictions.txt"
).read_bytes()
print("Reference RC1 protocol validated:", root)
print("Reference product SHA256:", source_sha)
PY
}

verify_mixed_sanity() {
  "$PYTHON_BIN" - "$OUTPUT_ROOT/rc15_mixed" "$MAX_PRODUCTS" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
max_products = int(sys.argv[2])
metadata = json.loads((directory / "sampling_metadata.json").read_text())
sampling = metadata["sampling"]
assert sampling["first_event_center_guided_trajectories"] == 3
assert sampling["first_event_center_ordinary_trajectories"] == 6
assert metadata["output_line_count"] == max_products * 9

diagnostics = json.loads((directory / "center_bias_diagnostics.json").read_text())
assert diagnostics["guided_trajectories_per_product"] == 3
assert diagnostics["ordinary_euler_trajectories_per_product"] == 6
records = diagnostics["records"]
summary = diagnostics["summary"]
roles = summary["first_event_trajectory_role_counts"]
no_events = diagnostics.get("no_event_trajectory_role_counts", {})
for role, expected_per_product in (
    ("center_guided", 3), ("ordinary_euler", 6),
):
    observed = int(roles.get(role, 0)) + int(no_events.get(role, 0))
    assert observed == max_products * expected_per_product, (
        role, observed, max_products * expected_per_product
    )
for record in records:
    role = record["row_metadata"]["trajectory_role"]
    enabled = record["position_bias_enabled"]
    assert enabled == (role == "center_guided")
    if role == "ordinary_euler":
        assert record["position_bias_reweighted"] is False
assert diagnostics["max_hazard_relative_error"] < 1e-5
print(
    "RC1.5 sanity passed: 3 guided + 6 ordinary trajectories per input; "
    f"first events={diagnostics['first_event_count']}; "
    f"no events={diagnostics['no_event_count']}"
)
print("First-event roles:", roles, "| no-event roles:", no_events)
PY
}

if [[ "$REUSE_REFERENCE" == "1" ]]; then
  validate_reference
  for group in b0_plain b0_trace b1_oracle; do
    ln -s "$(realpath "$REFERENCE_RC1_ROOT/$group")" "$OUTPUT_ROOT/$group"
  done
  echo "RC1.5 reuses validated B0/B0-trace/B1 outputs from: $REFERENCE_RC1_ROOT"
else
  run_group b0_plain
  run_group b0_trace \
    --first_event_center_sidecar "$SIDECAR" \
    --first_event_center_source oracle \
    --first_event_center_max_multiplier 1.0
  if ! cmp -s \
    "$OUTPUT_ROOT/b0_plain/predictions.txt" \
    "$OUTPUT_ROOT/b0_trace/predictions.txt"; then
    echo "B0 and neutral B0-trace predictions differ; stopping RC1.5." >&2
    exit 1
  fi
  echo "B0 neutrality check: predictions are byte-identical"
  run_group b1_oracle \
    --first_event_center_sidecar "$SIDECAR" \
    --first_event_center_source oracle \
    --first_event_center_max_multiplier 3.0
fi

run_group rc15_mixed \
  --first_event_center_sidecar "$SIDECAR" \
  --first_event_center_source oracle \
  --first_event_center_max_multiplier 3.0 \
  --first_event_center_guided_trajectories 3

verify_mixed_sanity

if [[ "$MODE" == "dev1000" || "$MODE" == "confirm1000" ]]; then
  SUMMARY_JSON="after_spe/results/stage1/rc15_${MODE}_summary.json"
else
  SUMMARY_JSON="$OUTPUT_ROOT/summary.json"
fi
"$PYTHON_BIN" scripts/analyze_stage1_rc1.py \
  --experiment rc15 \
  --run-root "$OUTPUT_ROOT" \
  --products-file "$PRODUCTS" \
  --targets-file "$TARGETS" \
  --vocab-file "$VOCAB" \
  --max-seq-len 96 \
  --bootstrap-draws 10000 \
  --seed 42 \
  --output-json "$SUMMARY_JSON"

echo "RC1.5 ${MODE} complete: $OUTPUT_ROOT"
echo "Summary: $SUMMARY_JSON"

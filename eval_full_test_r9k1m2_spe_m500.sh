#!/usr/bin/env bash
# R9K1M2 full-test evaluation for:
#   1) SPE m500 checkpoints on datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500/test
#   2) Global atom-level checkpoint on datasets/USPTO_50K_PtoR_aug20_#global#/test
set -euo pipefail

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export PATH=/root/autodl-tmp/ef/bin:$PATH
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

SPE_CKPT_DIR="${SPE_CKPT_DIR:-new_checkpoints/spe_m500_checkpoints}"
SPE_DATA="${SPE_DATA:-datasets/USPTO_50K_PtoR_aug20_#global#_SPE_m500}"
SPE_STEPS="${SPE_STEPS:-470000 480000 490000 500000 510000 520000 600000}"

ATOM_CKPT="${ATOM_CKPT:-new_checkpoints/checkpoint_step600000.pt}"
ATOM_DATA="${ATOM_DATA:-datasets/USPTO_50K_PtoR_aug20_#global#}"
ATOM_STEPS="${ATOM_STEPS:-600000}"

EVAL_SEED="${EVAL_SEED:-42}"
PROCESS_NUMBER="${PROCESS_NUMBER:-12}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p logs results

run_eval() {
  tag="$1"
  ckpt="$2"
  data="$3"
  step="$4"
  out="results/${tag}_step${step}_fulltest_r9k1m2_seed${EVAL_SEED}"
  log="logs/${tag}_step${step}_fulltest_r9k1m2_seed${EVAL_SEED}.log"

  "$PYTHON_BIN" scripts/eval.py \
    --checkpoint "$ckpt" \
    --products_file "$data/test/src-test.txt" \
    --targets "$data/test/tgt-test.txt" \
    --data_dir "$data" \
    --vocab_file "$data/example.vocab.src" \
    --output_dir "$out" \
    --sampler euler_beam \
    --n_runs 9 \
    --n_branches 1 \
    --n_children 2 \
    --n_steps 100 \
    --batch_size 64 \
    --device cuda \
    --seed "$EVAL_SEED" \
    --scheduler cubic \
    --euler_beam_score_mode full_probability \
    --euler_beam_changed_state_bonus 0.5 \
    --euler_beam_matmul_precision high \
    --euler_beam_child_policy stochastic_noop \
    --augmentation 20 \
    --n_best 10 \
    --process_number "$PROCESS_NUMBER" \
    2>&1 | tee "$log"
}

# Preflight: SPE m500
for step in $SPE_STEPS; do
  ckpt="$SPE_CKPT_DIR/checkpoint_step${step}.pt"
  if [ ! -f "$ckpt" ]; then
    echo "SPE m500 checkpoint not found: $ckpt" >&2
    exit 1
  fi
done
for required in \
  "$SPE_DATA/test/src-test.txt" \
  "$SPE_DATA/test/tgt-test.txt" \
  "$SPE_DATA/example.vocab.src"; do
  if [ ! -f "$required" ]; then
    echo "SPE m500 required file not found: $required" >&2
    exit 1
  fi
done

# Preflight: Global atom-level
for step in $ATOM_STEPS; do
  if [ ! -f "$ATOM_CKPT" ]; then
    echo "Atom-level checkpoint not found: $ATOM_CKPT" >&2
    exit 1
  fi
done
for required in \
  "$ATOM_DATA/test/src-test.txt" \
  "$ATOM_DATA/test/tgt-test.txt" \
  "$ATOM_DATA/example.vocab.src"; do
  if [ ! -f "$required" ]; then
    echo "Atom-level required file not found: $required" >&2
    exit 1
  fi
done

echo "SPE m500 full test: $SPE_DATA/test"
echo "SPE m500 checkpoints: $SPE_STEPS"
echo "Atom-level full test: $ATOM_DATA/test"
echo "Atom-level checkpoints: $ATOM_STEPS"
echo "Sampler: R9K1M2 (euler_beam, n_runs=9, n_branches=1, n_children=2)"

for step in $SPE_STEPS; do
  run_eval spe_m500 "$SPE_CKPT_DIR/checkpoint_step${step}.pt" "$SPE_DATA" "$step"
done

for step in $ATOM_STEPS; do
  run_eval atom_global "$ATOM_CKPT" "$ATOM_DATA" "$step"
done

echo "Done"

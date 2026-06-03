#!/usr/bin/env bash
set -euo pipefail

cd /workspace

LOG_DIR="/home/repro/logs"
NOFLAG_OUT="/home/repro/spiq_outputs_noflag"
FLAG_OUT="/home/repro/spiq_outputs_flag"

mkdir -p "$LOG_DIR" "$NOFLAG_OUT" "$FLAG_OUT"

MASTER_LOG="$LOG_DIR/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1

INPUTS=(0 1)
REPS=2
N_GENS=4000
OPTIMIZER=1

echo "===== START ALL EXPERIMENTS ====="
echo "Master log: $MASTER_LOG"
echo "Start time: $(date)"

for INPUT_IDX in "${INPUTS[@]}"; do
  echo
  echo "=================================================="
  echo "INPUT ${INPUT_IDX}: NO FEATURE FLAG"
  echo "=================================================="

  SPIQ_NOFLAG_LOG="$LOG_DIR/spiq_noflag_input${INPUT_IDX}_reps${REPS}.log"

  python3 base/spiq_initialization.py \
    --input_idx "$INPUT_IDX" \
    --reps "$REPS" \
    --n_gens "$N_GENS" \
    --out_dir "$NOFLAG_OUT" \
    2>&1 | tee "$SPIQ_NOFLAG_LOG"

  JSON_NOFLAG_FILE=$(ls "${NOFLAG_OUT}/spiq_initial_point_input${INPUT_IDX}_reps${REPS}_"*.json | head -1)

  TRIAL_NOFLAG=$((10 + INPUT_IDX))
  IBMQ_NOFLAG_LOG="$LOG_DIR/ibmq_noflag_input${INPUT_IDX}_trial${TRIAL_NOFLAG}.log"

  cd /workspace/base

  python3 IBMQExperiments.py \
    --input_idx "$INPUT_IDX" \
    --trial "$TRIAL_NOFLAG" \
    --reps "$REPS" \
    --optimizer "$OPTIMIZER" \
    --spiq_json "$JSON_NOFLAG_FILE" \
    2>&1 | tee "$IBMQ_NOFLAG_LOG"

  cd /workspace

  echo
  echo "=================================================="
  echo "INPUT ${INPUT_IDX}: WITH FEATURE FLAG"
  echo "=================================================="

  SPIQ_FLAG_LOG="$LOG_DIR/spiq_flag_input${INPUT_IDX}_reps${REPS}.log"

  python3 base/spiq_initialization.py \
    --input_idx "$INPUT_IDX" \
    --reps "$REPS" \
    --n_gens "$N_GENS" \
    --selection-strategy clustering \
    --num-select 5 \
    --selection-seed 42 \
    --out_dir "$FLAG_OUT" \
    2>&1 | tee "$SPIQ_FLAG_LOG"

  JSON_FLAG_FILE=$(ls "${FLAG_OUT}/spiq_initial_point_input${INPUT_IDX}_reps${REPS}_"*.json | head -1)

  TRIAL_FLAG=$((20 + INPUT_IDX))
  IBMQ_FLAG_LOG="$LOG_DIR/ibmq_flag_input${INPUT_IDX}_trial${TRIAL_FLAG}.log"

  cd /workspace/base

  python3 IBMQExperiments.py \
    --input_idx "$INPUT_IDX" \
    --trial "$TRIAL_FLAG" \
    --reps "$REPS" \
    --optimizer "$OPTIMIZER" \
    --spiq_json "$JSON_FLAG_FILE" \
    2>&1 | tee "$IBMQ_FLAG_LOG"

  cd /workspace
done

echo
echo "===== ALL EXPERIMENTS DONE ====="
echo "End time: $(date)"
echo "Master log saved to: $MASTER_LOG"
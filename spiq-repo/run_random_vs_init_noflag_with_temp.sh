#!/usr/bin/env bash
set -euo pipefail

cd /workspace

RUN_ID=$(date +%Y%m%d_%H%M%S)

LOG_DIR="/home/repro/logs/random_vs_init_noflag_${RUN_ID}"
INIT_OUT="/home/repro/spiq_outputs_noflag_${RUN_ID}"

mkdir -p "$LOG_DIR" "$INIT_OUT"

MASTER_LOG="$LOG_DIR/run_all_random_vs_init_noflag_with_temp.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

INPUTS=(0)
REPS=2
N_GENS=100
OPTIMIZER_ID=1
OPTIMIZER_NAME="COBYLA"
ITERATIONS=10000

BASE_DIR="/workspace/Week88/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data"
PICKLE_BASE="/workspace/base/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data"

echo "===== START RANDOM VS INIT-NO-FLAG EXPERIMENTS ====="
echo "Run ID: $RUN_ID"
echo "Master log: $MASTER_LOG"
echo "SPIQ no-flag output dir: $INIT_OUT"
echo "Base dir: $BASE_DIR"
echo "Pickle base: $PICKLE_BASE"
echo "Start time: $(date)"

for INPUT_IDX in "${INPUTS[@]}"; do
  echo
  echo "=================================================="
  echo "INPUT ${INPUT_IDX}: VANILLA / NO INITIALIZATION"
  echo "This uses QAOA initial_point = [0.5, 0.5, 0.5, 0.5] when reps=2"
  echo "=================================================="

  TRIAL_RANDOM=$((30 + INPUT_IDX))
  RANDOM_LOG="$LOG_DIR/ibmq_random_input${INPUT_IDX}_trial${TRIAL_RANDOM}.log"

  cd /workspace/base

  python3 IBMQExperiments.py \
    --input_idx "$INPUT_IDX" \
    --trial "$TRIAL_RANDOM" \
    --reps "$REPS" \
    --optimizer "$OPTIMIZER_ID" \
    2>&1 | tee "$RANDOM_LOG"

  cd /workspace

  echo
  echo "--------------------------------------------------"
  echo "POSTPROCESS: INPUT ${INPUT_IDX}, RANDOM, TRIAL ${TRIAL_RANDOM}"
  echo "--------------------------------------------------"

  TEMP_RANDOM_LOG="$LOG_DIR/temp_random_input${INPUT_IDX}_trial${TRIAL_RANDOM}.log"

  python3 base/Temp.py \
    --input_idx "$INPUT_IDX" \
    --trial "$TRIAL_RANDOM" \
    --reps "$REPS" \
    --optimizer_name "$OPTIMIZER_NAME" \
    --iterations "$ITERATIONS" \
    --base_dir "$BASE_DIR" \
    --pickle_base "$PICKLE_BASE" \
    2>&1 | tee "$TEMP_RANDOM_LOG"

  echo
  echo "=================================================="
  echo "INPUT ${INPUT_IDX}: SPIQ INITIALIZATION WITHOUT FEATURE FLAG"
  echo "=================================================="

  SPIQ_NOFLAG_LOG="$LOG_DIR/spiq_noflag_input${INPUT_IDX}_reps${REPS}.log"

  python3 base/spiq_initialization.py \
    --input_idx "$INPUT_IDX" \
    --reps "$REPS" \
    --n_gens "$N_GENS" \
    --out_dir "$INIT_OUT" \
    2>&1 | tee "$SPIQ_NOFLAG_LOG"

  JSON_NOFLAG_FILE=$(ls "${INIT_OUT}/spiq_initial_point_input${INPUT_IDX}_reps${REPS}_"*.json | head -1)

  echo "Using SPIQ JSON: $JSON_NOFLAG_FILE"

  TRIAL_INIT=$((40 + INPUT_IDX))
  INIT_LOG="$LOG_DIR/ibmq_init_noflag_input${INPUT_IDX}_trial${TRIAL_INIT}.log"

  cd /workspace/base

  python3 IBMQExperiments.py \
    --input_idx "$INPUT_IDX" \
    --trial "$TRIAL_INIT" \
    --reps "$REPS" \
    --optimizer "$OPTIMIZER_ID" \
    --spiq_json "$JSON_NOFLAG_FILE" \
    2>&1 | tee "$INIT_LOG"

  cd /workspace

  echo
  echo "--------------------------------------------------"
  echo "POSTPROCESS: INPUT ${INPUT_IDX}, INIT-NO-FLAG, TRIAL ${TRIAL_INIT}"
  echo "--------------------------------------------------"

  TEMP_INIT_LOG="$LOG_DIR/temp_init_noflag_input${INPUT_IDX}_trial${TRIAL_INIT}.log"

  python3 base/Temp.py \
    --input_idx "$INPUT_IDX" \
    --trial "$TRIAL_INIT" \
    --reps "$REPS" \
    --optimizer_name "$OPTIMIZER_NAME" \
    --iterations "$ITERATIONS" \
    --base_dir "$BASE_DIR" \
    --pickle_base "$PICKLE_BASE" \
    2>&1 | tee "$TEMP_INIT_LOG"

  echo
  echo "Finished input ${INPUT_IDX}"
  echo "  random trial      : $TRIAL_RANDOM"
  echo "  init-no-flag trial: $TRIAL_INIT"
done

echo
echo "===== ALL RANDOM VS INIT-NO-FLAG EXPERIMENTS + TEMP DONE ====="
echo "End time: $(date)"
echo "Master log saved to: $MASTER_LOG"
echo "Logs saved under: $LOG_DIR"
echo "SPIQ no-flag outputs saved under: $INIT_OUT"
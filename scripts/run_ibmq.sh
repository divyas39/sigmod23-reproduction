#!/bin/bash

# in case the script is not started from within sigmod-repro directory

# if [ ! "${PWD}" = "/home/repro/sigmod-repro" ]; then
#     cd /home/repro/sigmod-repro/
# fi

cd /home/repro/sigmod-repro/base

# IBMQExperiments.py currently hardcodes `for i in range(0, 1)` in
# `conduct_IBMQ_QPU_experiments`, so only input_idx=0 is processed. If that
# loop is widened, extend SPIQ_INPUT_IDX below accordingly.
SPIQ_INPUT_IDX=0
SPIQ_DIR=/home/repro/sigmod-repro/spiq_init_outputs

echo "Started running IBMQ experiments..."

# # AQGD
# echo "Running trial=1, reps=2, optimizer=0 (AQGD)"
# python3 IBMQExperiments.py \
#   --trial 1 \
#   --reps 2 \
#   --optimizer 0

# # SPSA: run once, reps=2, trial=1
# echo "Running trial=1, reps=2, optimizer=2 (SPSA)"
# python3 IBMQExperiments.py \
#   --trial 1 \
#   --reps 2 \
#   --optimizer 2

# # COBYLA: run 5 trials, reps=2
# for trial in 1 2 3 4 5; do
#   echo "Running trial=${trial}, reps=2, optimizer=1 (COBYLA)"
#   python3 IBMQExperiments.py \
#     --trial "${trial}" \
#     --reps 2 \
#     --optimizer 1
# done

for opt in 1; do
  for reps in 2; do
    spiq_json="${SPIQ_DIR}/spiq_initial_point_input${SPIQ_INPUT_IDX}_reps${reps}.json"
    if [ ! -f "${spiq_json}" ]; then
      echo "SPIQ init file missing: ${spiq_json}" >&2
      echo "Regenerate with: python3 spiq_initialization.py --input_idx ${SPIQ_INPUT_IDX} --reps ${reps} --n_gens 200" >&2
      exit 1
    fi

    for trial in 1 2 3; do
      echo "Running trial=${trial}, reps=${reps}, optimizer=${opt}, spiq_json=${spiq_json}"
      python3 IBMQExperiments.py \
        --trial "${trial}" \
        --reps "${reps}" \
        --optimizer "${opt}" \
        --spiq_json "${spiq_json}" \
        >> "ibmq_experiment_opt${opt}_reps${reps}_trial${trial}.log" 2>&1
    done
  done
done



# python3 IBMQExperiments.py --trial 1 --reps 1 --optimizer 1
# python3 IBMQExperiments.py --trial 2 --reps 1 --optimizer 1
# python3 IBMQExperiments.py --trial 3 --reps 1 --optimizer 1
# python3 IBMQExperiments.py --trial 4 --reps 1 --optimizer 1
# python3 IBMQExperiments.py --trial 5 --reps 1 --optimizer 1

# python3 IBMQExperiments.py --trial 1 --reps 2 --optimizer 1
# python3 IBMQExperiments.py --trial 2 --reps 2 --optimizer 1
# python3 IBMQExperiments.py --trial 3 --reps 2 --optimizer 1
# python3 IBMQExperiments.py --trial 4 --reps 2 --optimizer 1
# python3 IBMQExperiments.py --trial 5 --reps 2 --optimizer 1

# python3 IBMQExperiments.py --trial 1 --reps 3 --optimizer 1
# python3 IBMQExperiments.py --trial 2 --reps 3 --optimizer 1
# python3 IBMQExperiments.py --trial 3 --reps 3 --optimizer 1
# python3 IBMQExperiments.py --trial 4 --reps 3 --optimizer 1
# python3 IBMQExperiments.py --trial 5 --reps 3 --optimizer 1


# python3 IBMQExperiments.py --trial 1 --reps 1 --optimizer 2
# python3 IBMQExperiments.py --trial 2 --reps 1 --optimizer 2
# python3 IBMQExperiments.py --trial 3 --reps 1 --optimizer 2
# python3 IBMQExperiments.py --trial 4 --reps 1 --optimizer 2
# python3 IBMQExperiments.py --trial 5 --reps 1 --optimizer 2

# python3 IBMQExperiments.py --trial 1 --reps 2 --optimizer 2
# python3 IBMQExperiments.py --trial 2 --reps 2 --optimizer 2
# python3 IBMQExperiments.py --trial 3 --reps 2 --optimizer 2
# python3 IBMQExperiments.py --trial 4 --reps 2 --optimizer 2
# python3 IBMQExperiments.py --trial 5 --reps 2 --optimizer 2

# python3 IBMQExperiments.py --trial 1 --reps 3 --optimizer 2
# python3 IBMQExperiments.py --trial 2 --reps 3 --optimizer 2
# python3 IBMQExperiments.py --trial 3 --reps 3 --optimizer 2
# python3 IBMQExperiments.py --trial 4 --reps 3 --optimizer 2
# python3 IBMQExperiments.py --trial 5 --reps 3 --optimizer 2


# python3 IBMQExperiments.py --trial 1 --reps 1 --optimizer 0
# python3 IBMQExperiments.py --trial 1 --reps 2 --optimizer 0
# python3 IBMQExperiments.py --trial 1 --reps 3 --optimizer 0

# python3 IBMQExperiments.py --trial 1 --reps 1 --optimizer 2
# python3 IBMQExperiments.py --trial 2 --reps 1
# python3 IBMQExperiments.py --trial 3 --reps 1
# python3 IBMQExperiments.py --trial 4 --reps 1
# python3 IBMQExperiments.py --trial 5 --reps 1
# python3 IBMQExperiments.py --trial 1 --reps 2 --optimizer 0
# python3 IBMQExperiments.py --trial 2 --reps 2 --optimizer 1
# python3 IBMQExperiments.py --trial 3 --reps 2 --optimizer 2
# python3 IBMQExperiments.py --trial 4 --reps 2
# python3 IBMQExperiments.py --trial 5 --reps 2
# python3 IBMQExperiments.py --trial 1 --reps 3 --optimizer 0
# python3 IBMQExperiments.py --trial 2 --reps 3 --optimizer 1
# python3 IBMQExperiments.py --trial 3 --reps 3 --optimizer 2
# python3 IBMQExperiments.py --trial 4 --reps 3
# python3 IBMQExperiments.py --trial 5 --reps 3
echo "IBMQ experiments done."

# cd /home/repro/sigmod-repro/scripts/plotting
# echo "Plotting IBMQ results..."
# Rscript ibmq_plotting.r
# echo "Plotting done."

cd /home/repro
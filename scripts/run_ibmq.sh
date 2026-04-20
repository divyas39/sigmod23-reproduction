#!/bin/bash

# in case the script is not started from within sigmod-repro directory

# if [ ! "${PWD}" = "/home/repro/sigmod-repro" ]; then
#     cd /home/repro/sigmod-repro/
# fi

cd /home/repro/sigmod-repro/base

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
    for trial in 1 2 3; do 
      echo "Running trial=${trial}, reps=${reps}, optimizer=${opt}"
      python3 IBMQExperiments.py \
        --trial "${trial}" \
        --reps "${reps}" \
        --optimizer "${opt}" \
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
import json
import os
import pathlib 
import Scripts.ProblemGenerator as ProblemGenerator
import Scripts.QUBOGenerator as QUBOGenerator
import Scripts.CircuitGeneration as CircuitGenerator
import Scripts.TopologyGenerator as TopologyGenerator
import Scripts.Postprocessing as Postprocessing
from multiprocessing import Pool
import csv
import IBMQExperiments
import numpy as np
import config as config
import pickle
from decimal import *
from pathlib import Path
import Scripts.Postprocessing1 as Postprocessing1


def parse_QPU_data(include_header=True,currentInput=0):
    if include_header:
        IBMQExperiments.save_to_csv(['num_qaoa_iterations', 'num_predicates', 'valid_ratio', 'opt_ratio',f'Trial: {IBMQExperiments.TRIAL_ID}',f'Reps:{IBMQExperiments.TAG}',f'Optimizer:{IBMQExperiments.current_optim}'], 'ExperimentalAnalysis/IBMQ/QPUPerformance/Results', 'results.txt')
 
    processing = config.configuration["ibmq-processing"]
    if processing == "qpu":
        result_path_prefix = 'ExperimentalAnalysis/IBMQ/QPUPerformance/Results/QPU_Data'
    elif processing == "cpu":
        result_path_prefix = 'base/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/'
    else:
        result_path_prefix = 'ExperimentalAnalysis/IBMQ/QPUPerformance/Results/Collected_Data/'
        
    iterations_categories = [10000]
    thres_vals = {0:range(0, 301),1:range(0, 301),2:range(0, 301), 3: [10]}
    
    for iterations in iterations_categories:
        for i in range(1):

            card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem('base/ExperimentalAnalysis/IBMQ/QPUPerformance/Problems/JSON/' + str(i) + '_predicates', generated_problems=False)
            response = IBMQExperiments.load_pickled_result(result_path_prefix + '/' + str(iterations) + '_Iterations/' + str(i) + '_predicates')
            
            best_join_order, best_join_order_costs, valid_ratio, optimal_ratio = Postprocessing.postprocess_IBMQ_response(response, card, pred, pred_sel, thres_vals[i],trial_id1=IBMQExperiments.TRIAL_ID,tag1=IBMQExperiments.TAG,current_optim1=IBMQExperiments.current_optim,iterations1=iterations,inputNumber=i)
            IBMQExperiments.save_to_csv([iterations, i, IBMQExperiments.get_rounded_val(valid_ratio), IBMQExperiments.get_rounded_val(optimal_ratio)], 'ExperimentalAnalysis/IBMQ/QPUPerformance/Results', 'results.txt')

if __name__ == '__main__':
    processing = config.configuration["ibmq-processing"]
    thre=[[150, 200, 300],[160, 200, 240, 280],[120, 150, 180, 220, 260, 300]]
    result_path_prefix = 'base/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/'
    response = IBMQExperiments.load_pickled_result(result_path_prefix + '/' + str(10000) + '_Iterations/' + str(0) + '_predicates-newQUBO')
    card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem('base/ExperimentalAnalysis/IBMQ/QPUPerformance/Problems/JSON/' + str(0) + '_predicates', generated_problems=False)
    
    out_path = Path("postprocess_output.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(str(response))

    print(f"Saved postprocess output to {out_path.resolve()}")

    out_path = "readout_summary_bitstring_energy_prob.csv"
    # Postprocessing.postprocess_IBMQ_response(response, card, pred, pred_sel,[150])

    # Postprocessing.write_bitstring_energy_prob(response, out_path)

    # Postprocessing1.readout(response, card, pred, pred_sel, card_dict=None)


    # Postprocessing.write_bitstring_energy_prob(response,card, pred, pred_sel)
    Postprocessing.postprocess_qiskit_with_readout(response, card, pred, pred_sel)


    # out_path = "ising_hamiltonian.txt"
    # with open(out_path, "w", encoding="utf-8") as f:
    #     for i in thre:
    #         qubo, weight_a = QUBOGenerator.generate_QUBO_for_IBMQ([10,15,20], i, 0,[], [])
    #         op, offset = qubo.to_ising()
    #         f.write(f"=== op (Ising Hamiltonian of thres {i}) ===\n")
    #         f.write(str(op))
    #         f.write("\n\n=== offset ===\n")
    #         f.write(str(offset))
    #         f.write("\n\n=== weight_a ===\n")
    #         f.write(str(weight_a))
    #         f.write("\n")

    # print(f"Saved to {out_path}")
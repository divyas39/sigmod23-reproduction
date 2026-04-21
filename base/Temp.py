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
from Scripts import QUBOGenerator1
import numpy as np
import config as config
import pickle
from decimal import *
from pathlib import Path
import Scripts.Postprocessing1 as Postprocessing1
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
import Scripts.Postprocessing1 as PS1
from qiskit.providers.aer import QasmSimulator
from qiskit.utils import QuantumInstance
import re
from qiskit.algorithms import NumPyMinimumEigensolver

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




def run_callback_parameter_simulation_and_postprocess(
    parameters,
    qubo,
    reps,
    card,
    pred,
    pred_sel,
    PS1,
    card_dict=None,
    quantum_instance=None,
    shots=10240,
    opt_time_ms=0.0,
    base_dir="./Week27/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data",
    trial_id=1,
    tag=4,
    current_optim="COBYLA",
    iterations=10000,
    input_id=0,
    eval_count=None
):
    if card_dict is None:
        card_dict = {}
    if quantum_instance is None:
        backend = QasmSimulator()
        quantum_instance = QuantumInstance(backend=backend, shots=shots)
    op, _ = qubo.to_ising()
    ansatz = QAOAAnsatz(op, reps=reps).decompose()
    param_map = {p: v for p, v in zip(ansatz.parameters, parameters)}
    qc = ansatz.assign_parameters(param_map, inplace=False)
    qc = qc.copy()
    qc.measure_all()
    execute_result = quantum_instance.execute(qc)

    best_for_time, all_solutions, solutions_for_readout = postprocess_callback_execute_with_readout(
        execute_result=execute_result,
        qubo=qubo,
        card=card,
        pred=pred,
        pred_sel=pred_sel,
        PS1=PS1,
        card_dict=card_dict,
        opt_time_ms=opt_time_ms,
        base_dir=base_dir,
        trial_id=trial_id,
        tag=tag,
        current_optim=current_optim,
        iterations=iterations,
        input_id=input_id,
        eval_count=eval_count
    )

    return {
        "execute_result": execute_result,
        "best_for_time": best_for_time,
        "all_solutions": all_solutions,
        "solutions_for_readout": solutions_for_readout,
    }


def batch_run_callback_history_and_postprocess(
    callback_history,
    qubo,
    reps,
    card,
    pred,
    pred_sel,
    PS1,
    card_dict=None,
    quantum_instance=None,
    shots=10240,
    opt_time_ms=0.0,
    base_dir="./Week27/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data",
    trial_id=1,
    tag=2,
    current_optim="COBYLA",
    iterations=10000,
    input_id=0
):
   
    results = []

    for item in callback_history:
        eval_count = int(item["eval_count"])
        parameters = item["parameters"]

        one_result = run_callback_parameter_simulation_and_postprocess(
            parameters=parameters,
            qubo=qubo,
            reps=reps,
            card=card,
            pred=pred,
            pred_sel=pred_sel,
            PS1=PS1,
            card_dict=card_dict,
            quantum_instance=quantum_instance,
            shots=shots,
            opt_time_ms=opt_time_ms,
            base_dir=base_dir,
            trial_id=trial_id,
            tag=tag,
            current_optim=current_optim,
            iterations=iterations,
            input_id=input_id,
            eval_count=eval_count
        )

        results.append({
            "eval_count": eval_count,
            "parameters": parameters,
            "best_for_time": one_result["best_for_time"],
            "all_solutions": one_result["all_solutions"],
            "solutions_for_readout": one_result["solutions_for_readout"],
        })

    return results


def postprocess_callback_execute_with_readout(
    execute_result,
    qubo,
    card,
    pred,
    pred_sel,
    PS1,
    card_dict=None,
    opt_time_ms=0.0,
    base_dir="./Week27/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data",
    trial_id=1,
    tag=4,
    current_optim="COBYLA",
    iterations=10000,
    input_id=0,
    eval_count=None
):
    if card_dict is None:
        card_dict = {}

    result_dir = os.path.join(
        base_dir,
        f"iterations_{iterations}",
        f"reps_{tag}",
        f"{current_optim}",
        f"input0",
        f"trial{trial_id}"
    )
    os.makedirs(result_dir, exist_ok=True)
    print('Save to'+result_dir)


    suffix = f"_eval{eval_count}" if eval_count is not None else ""
    csv_path = os.path.join(result_dir, f"readout_summary{suffix}.csv")

    counts = execute_result.get_counts()
    if not counts:
        raise ValueError("execute_result.get_counts()empty")

    total = sum(counts.values())
    solutions = []

    for bitstring, cnt in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        prob = cnt / total
        b = bitstring.replace(" ", "")
        x = np.array([int(ch) for ch in b[::-1]], dtype=int)
        try:
            energy = float(qubo.objective.evaluate(x))
        except Exception:
            energy = float(qubo.objective.evaluate(list(x)))

        bitlist = list(x)               
        occ = int(cnt)                 
        stringbit = "".join(str(int(v)) for v in bitlist) 
        probability = float(prob)        

        solutions.append([bitlist, occ, energy, stringbit, probability])

    response_like = [solutions, float(opt_time_ms)]
    best_for_time, all_solutions = PS1.readout(
        response_like, card, pred, pred_sel, card_dict
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        w.writerow(["# best_solutions_for_time"])
        w.writerow([
            "bitstring", "rank", "join_order", "cost",
            "time_ms", "used_fallback", "energy", "count", "probability"
        ])
        for idx, sol in enumerate(best_for_time):
            bitstring, join_order, cost, t_ms, used_fallback, energy, occ, probability = sol
            w.writerow([bitstring, idx, join_order, cost, t_ms, used_fallback, energy, occ, probability])

        w.writerow([])

        w.writerow(["# all_solutions"])
        w.writerow([
            "bitstring", "index", "join_order", "cost",
            "time_ms", "used_fallback", "energy", "count", "probability"
        ])
        for idx, sol in enumerate(all_solutions):
            bitstring, join_order, cost, t_ms, used_fallback, energy, occ, probability = sol
            w.writerow([bitstring, idx, join_order, cost, t_ms, used_fallback, energy, occ, probability])

    print(f"Saved to {csv_path}")
    return best_for_time, all_solutions, solutions

    
    


def _parse_parameter_string(param_str):
    s = param_str.strip()

    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]

    parts = re.split(r"\s+", s.strip())
    parts = [p for p in parts if p]

    return [float(x) for x in parts]


def convert_callback_csv_to_history(csv_path, encoding="utf-8"):
    callback_history = []

    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            param_col = "paramter" if "paramter" in row else "parameter"

            callback_history.append({
                "eval_count": int(row["iteration"]),
                "parameters": _parse_parameter_string(row[param_col]),
                "callback_value": float(row["energy"]),
                "stddev": float(row["std"]),
            })

    return callback_history

if __name__ == '__main__':
    processing = config.configuration["ibmq-processing"]
    thre=[[150, 200, 300],[160, 200, 240, 280],[120, 150, 180, 220, 260, 300]]
    result_path_prefix = 'base/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/'
    card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem('base/ExperimentalAnalysis/IBMQ/QPUPerformance/Problems/JSON/' + str(0) + '_predicates', generated_problems=False)
    qubo, penalty_weight=QUBOGenerator1.generate_IBMQ_QUBO_for_left_deep_trees_v2(card, pred, pred_sel)
    

    for trial in range(1, 4):
        response = IBMQExperiments.load_pickled_result(result_path_prefix + '/' + str(10000) + '_Iterations/' + str(0) + '_predicates-newQUBO/trial' + str(trial))
        res=convert_callback_csv_to_history(f'base/Week27/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/iterations_10000/reps_2/COBYLA/input0/trial{trial}/energy_per_iteration_10000_COBYLA_2_{trial}.csv')

        ## get res for each iteration

        batch_run_callback_history_and_postprocess(res,qubo=qubo,card=card,pred=pred,pred_sel=pred_sel,PS1=PS1,reps=2,trial_id=trial)
        

        # out_path = Path("postprocess_output.txt")
        # with open(out_path, "w", encoding="utf-8") as f:
        #     f.write(str(response))
        

        # print(f"Saved postprocess output to {out_path.resolve()}")

        # out_path = "readout_summary_bitstring_energy_prob.csv"

        # Postprocessing.postprocess_IBMQ_response(response, card, pred, pred_sel,[150])

        # Postprocessing.write_bitstring_energy_prob(response, out_path)

        # Postprocessing1.readout(response, card, pred, pred_sel, card_dict=None)


        # Postprocessing.write_bitstring_energy_prob(response,card, pred, pred_sel)

        ## for readout of final outputs

        Postprocessing.postprocess_qiskit_with_readout(
            response,
            card,
            pred,
            pred_sel,
            trial_id=trial,
            tag=2,
            current_optim="COBYLA",
            iterations=10000,
        )

        op, offset = qubo.to_ising()

        print(f"ising Hamiltonian is: {op}; with offset of {offset}")

        solver = NumPyMinimumEigensolver()
        result = solver.compute_minimum_eigenvalue(op)

        print("minimum eigenvalue:", result.eigenvalue)
        print("minimum eigenstate:", result.eigenstate)
        print("minimum energy with offset:", result.eigenvalue.real + offset)


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
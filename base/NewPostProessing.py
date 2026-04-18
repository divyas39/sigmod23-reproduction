import json
import os
import pathlib
import csv
import re
import argparse
from decimal import *
from pathlib import Path
from itertools import product

import numpy as np
import pickle
import config as config
import IBMQExperiments

import Scripts.ProblemGenerator as ProblemGenerator
import Scripts.QUBOGenerator as QUBOGenerator
import Scripts.CircuitGeneration as CircuitGenerator
import Scripts.TopologyGenerator as TopologyGenerator
import Scripts.Postprocessing as Postprocessing
import Scripts.Postprocessing1 as Postprocessing1
import Scripts.Postprocessing1 as PS1
from Scripts import QUBOGenerator1

from multiprocessing import Pool
from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
from qiskit.providers.aer import QasmSimulator
from qiskit.utils import QuantumInstance
from qiskit.algorithms import NumPyMinimumEigensolver


OPTIMIZER_MAP = {
    0: "AQGD",
    1: "COBYLA",
    2: "SPSA",
}


def parse_QPU_data(include_header=True, currentInput=0):
    if include_header:
        IBMQExperiments.save_to_csv(
            [
                'num_qaoa_iterations',
                'num_predicates',
                'valid_ratio',
                'opt_ratio',
                f'Trial: {IBMQExperiments.TRIAL_ID}',
                f'Reps:{IBMQExperiments.TAG}',
                f'Optimizer:{IBMQExperiments.current_optim}'
            ],
            'ExperimentalAnalysis/IBMQ/QPUPerformance/Results',
            'results.txt'
        )

    processing = config.configuration["ibmq-processing"]
    if processing == "qpu":
        result_path_prefix = 'ExperimentalAnalysis/IBMQ/QPUPerformance/Results/QPU_Data'
    elif processing == "cpu":
        result_path_prefix = 'base/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/'
    else:
        result_path_prefix = 'ExperimentalAnalysis/IBMQ/QPUPerformance/Results/Collected_Data/'

    iterations_categories = [10000]
    thres_vals = {0: range(0, 301), 1: range(0, 301), 2: range(0, 301), 3: [10]}

    for iterations in iterations_categories:
        for i in range(1):
            card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(
                'base/ExperimentalAnalysis/IBMQ/QPUPerformance/Problems/JSON/' + str(i) + '_predicates',
                generated_problems=False
            )
            response = IBMQExperiments.load_pickled_result(
                result_path_prefix + '/' + str(iterations) + '_Iterations/' + str(i) + '_predicates'
            )

            best_join_order, best_join_order_costs, valid_ratio, optimal_ratio = Postprocessing.postprocess_IBMQ_response(
                response,
                card,
                pred,
                pred_sel,
                thres_vals[i],
                trial_id1=IBMQExperiments.TRIAL_ID,
                tag1=IBMQExperiments.TAG,
                current_optim1=IBMQExperiments.current_optim,
                iterations1=iterations,
                inputNumber=i
            )
            IBMQExperiments.save_to_csv(
                [iterations, i, IBMQExperiments.get_rounded_val(valid_ratio), IBMQExperiments.get_rounded_val(optimal_ratio)],
                'ExperimentalAnalysis/IBMQ/QPUPerformance/Results',
                'results.txt'
            )


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
    base_dir="./Week23/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data",
    trial_id=4,
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
    base_dir="./Week23/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data",
    trial_id=4,
    tag=4,
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
    base_dir="./Week23/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data",
    trial_id=4,
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
        f"input{input_id}"
    )
    os.makedirs(result_dir, exist_ok=True)
    print("Save to " + result_dir)

    suffix = f"_eval{eval_count}" if eval_count is not None else ""
    csv_path = os.path.join(result_dir, f"readout_summary{suffix}.csv")

    counts = execute_result.get_counts()
    if not counts:
        raise ValueError("execute_result.get_counts() empty")

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


def get_optimizer_name(opt_value):
    if isinstance(opt_value, int):
        if opt_value not in OPTIMIZER_MAP:
            raise ValueError(f"Unsupported optimizer value: {opt_value}. Use one of {list(OPTIMIZER_MAP.keys())}")
        return OPTIMIZER_MAP[opt_value]

    opt_str = str(opt_value).strip().upper()
    reverse_map = {v.upper(): v for v in OPTIMIZER_MAP.values()}
    if opt_str in reverse_map:
        return reverse_map[opt_str]

    raise ValueError(f"Unsupported optimizer value: {opt_value}. Use 0/1/2 or AQGD/COBYLA/SPSA")


def process_one_configuration(trial, reps, optimizer, input_number, iterations=10000):
    current_optim = get_optimizer_name(optimizer)

    result_path_prefix = 'base/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/'
    response_path = (
        result_path_prefix
        + '/'
        + str(iterations)
        + '_Iterations/'
        + str(input_number)
        + '_predicates-newQUBO'
    )

    json_problem_path = (
        'base/ExperimentalAnalysis/IBMQ/QPUPerformance/Problems/JSON/'
        + str(input_number)
        + '_predicates'
    )

    energy_csv_path = (
        f'base/Week23/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/'
        f'iterations_{iterations}/reps_{reps}/{current_optim}/input{input_number}/'
        f'energy_per_iteration_{iterations}_{current_optim}_{reps}_{trial}.csv'
    )

    print("\n====================================================")
    print(f"Running configuration: trial={trial}, reps={reps}, optimizer={current_optim}, input={input_number}")
    print("====================================================")

    response = IBMQExperiments.load_pickled_result(response_path)

    card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(
        json_problem_path,
        generated_problems=False
    )

    qubo, penalty_weight = QUBOGenerator1.generate_IBMQ_QUBO_for_left_deep_trees_v2(
        card, pred, pred_sel
    )

    res = convert_callback_csv_to_history(energy_csv_path)

    batch_run_callback_history_and_postprocess(
        res,
        qubo=qubo,
        card=card,
        pred=pred,
        pred_sel=pred_sel,
        PS1=PS1,
        reps=reps,
        trial_id=trial,
        tag=reps,
        current_optim=current_optim,
        iterations=iterations,
        input_id=input_number
    )

    Postprocessing.postprocess_qiskit_with_readout(response, card, pred, pred_sel)

    op, offset = qubo.to_ising()
    print(f"Ising Hamiltonian is: {op}; with offset of {offset}")

    solver = NumPyMinimumEigensolver()
    result = solver.compute_minimum_eigenvalue(op)

    print("minimum eigenvalue:", result.eigenvalue)
    print("minimum eigenstate:", result.eigenstate)
    print("minimum energy with offset:", result.eigenvalue.real + offset)


def main():
    parser = argparse.ArgumentParser(
        description="Run callback-history postprocessing for one or more trial/reps/optimizer/input combinations."
    )
    parser.add_argument(
        "--trial",
        type=int,
        nargs="+",
        required=True,
        help="One or more trial ids. Example: --trial 1 2 3"
    )
    parser.add_argument(
        "--reps",
        type=int,
        nargs="+",
        required=True,
        help="One or more reps values. Example: --reps 1 2 3"
    )
    parser.add_argument(
        "--optimizer",
        nargs="+",
        required=True,
        help="One or more optimizers. Use 0 1 2 or AQGD COBYLA SPSA. Example: --optimizer 1 2 or --optimizer COBYLA SPSA"
    )
    parser.add_argument(
        "--input-number",
        type=int,
        nargs="+",
        required=True,
        help="One or more input numbers. Example: --input-number 0 1 2"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Iterations value used in file paths. Default: 10000"
    )

    args = parser.parse_args()

    for trial, reps, optimizer, input_number in product(
        args.trial,
        args.reps,
        args.optimizer,
        args.input_number
    ):
        process_one_configuration(
            trial=trial,
            reps=reps,
            optimizer=optimizer,
            input_number=input_number,
            iterations=args.iterations
        )


if __name__ == '__main__':
    main()
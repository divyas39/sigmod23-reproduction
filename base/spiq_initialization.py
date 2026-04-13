#!/usr/bin/env python
# coding: utf-8

# Script to run SPIQ/CAFQA for generating a QAOA initial point by modelling the QUBO for join order optimization.

import os
import json
import argparse
import numpy as np

import Scripts.ProblemGenerator as ProblemGenerator
import Scripts.QUBOGenerator1 as QUBOGenerator1

from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz

from clapton.circuit_manipulation import (
    generate_qiskit_param_map,
    modify_circuit,
    qiskit_to_stim,
    transform_to_allowed_gates,
)
from clapton.clapton import claptonize
from clapton.depolarization import GateGeneralDepolarizationModel


def build_problem(input_idx: int, threshold: int, num_decimal_pos: int = 3):
    """
    Build the same QUBO instance style used by IBMQExperiments.py.
    """
    json_path = (
        f"ExperimentalAnalysis/IBMQ/QPUPerformance/Problems/JSON/{input_idx}_predicates"
    )

    card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(
        json_path,
        generated_problems=False,
    )

    qubo, penalty_weight = QUBOGenerator1.generate_IBMQ_QUBO_for_left_deep_trees_v2(
        card,
        pred,
        pred_sel
    )

    return qubo, card, pred, pred_sel, penalty_weight


def build_vanilla_spiq_objects(qubo, reps: int):
    """
    Create a vanilla QAOA ansatz and the stim circuit needed by SPIQ,
    while preserving vanilla QAOA parameter count/order.
    """
    op, _ = qubo.to_ising()

    # Vanilla QAOA ansatz with 2 * reps parameters in Qiskit order
    qaoa_ansatz = QAOAAnsatz(op, reps=reps)

    # because IBMQExperiments.py expects the vanilla QAOA initial point shape.
    modified_circ = modify_circuit(qaoa_ansatz)
    pcirc = transform_to_allowed_gates(modified_circ)

    stim_circ = qiskit_to_stim(pcirc)
    param_map = generate_qiskit_param_map(pcirc)
    stim_circ.define_parameter_map(param_map)

    return op, qaoa_ansatz, pcirc, stim_circ, param_map


def run_spiq_initialization(
    qubo,
    reps: int,
    n_gens: int,
    n_proc: int = 32,
    n_starts: int = 4,
    n_rounds: int = 1,
    err: float = None,
    out_file: str = None,
):
    """
    Run SPIQ/CAFQA and return an ordered Qiskit-compatible initial point.
    """
    op, qaoa_ansatz, pcirc, stim_circ, param_map = build_vanilla_spiq_objects(
        qubo, reps=reps
    )

    paulis = op.paulis.to_labels()
    coeffs = op.coeffs.real
    reversed_paulis = [p[::-1] for p in paulis]

    if err is not None:
        nm = GateGeneralDepolarizationModel(p1=err, p2=10 * err)
        stim_circ.add_depolarization_model(nm)

    (
        ks_best,
        noisy_energy_best,
        energy_best,
        best_cafqa_gen_params,
        best_cafqa_gen_fitness,
    ) = claptonize(
        reversed_paulis,
        coeffs,
        stim_circ,
        n_proc=n_proc,
        n_starts=n_starts,
        n_rounds=n_rounds,
        callback=None,
        budget=n_gens // 2,
        out_file=out_file,
    )

    # Convert the best SPIQ assignment into an ordered list matching pcirc.parameters
    initial_point = []
    for qiskit_param in pcirc.parameters:
        mapped_key = param_map.get(qiskit_param, qiskit_param)

        if mapped_key in ks_best:
            initial_point.append(float(ks_best[mapped_key]))
            continue

        if qiskit_param in ks_best:
            initial_point.append(float(ks_best[qiskit_param]))
            continue

        found = False
        for k, v in ks_best.items():
            if str(k) == str(mapped_key) or str(k) == str(qiskit_param):
                initial_point.append(float(v))
                found = True
                break

        if not found:
            raise KeyError(f"Could not map SPIQ parameter for {qiskit_param}")

    expected_len = 2 * reps
    if len(initial_point) != expected_len:
        raise ValueError(
            f"Expected vanilla QAOA initial point of length {expected_len}, "
            f"but got {len(initial_point)}"
        )

    return {
        "initial_point": initial_point,
        "energy_best": float(energy_best),
        "noisy_energy_best": None if noisy_energy_best is None else float(noisy_energy_best),
        "best_cafqa_gen_fitness": (
            None if best_cafqa_gen_fitness is None else float(best_cafqa_gen_fitness)
        ),
        "ks_best_str": {str(k): float(v) for k, v in ks_best.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_idx", type=int, default=0, help="Problem index")
    parser.add_argument("--reps", type=int, default=1, help="QAOA reps / p")
    # parser.add_argument("--threshold", type=int, default=150, help="Threshold for QUBO generation")
    # parser.add_argument("--num_decimal_pos", type=int, default=3, help="Decimal precision")
    parser.add_argument("--n_gens", type=int, default=200, help="SPIQ generation budget")
    parser.add_argument("--n_proc", type=int, default=32, help="Number of processes for SPIQ")
    parser.add_argument("--n_starts", type=int, default=4, help="Number of SPIQ starts")
    parser.add_argument("--n_rounds", type=int, default=1, help="Number of SPIQ rounds")
    parser.add_argument("--err", type=float, default=None, help="Optional depolarization error")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="spiq_init_outputs",
        help="Directory for SPIQ output files",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    qubo, card, pred, pred_sel, penalty_weight = build_problem(
        input_idx=args.input_idx,
    )

    out_file = os.path.join(
        args.out_dir,
        f"spiq_trace_input{args.input_idx}_reps{args.reps}.txt",
    )

    result = run_spiq_initialization(
        qubo=qubo,
        reps=args.reps,
        n_gens=args.n_gens,
        n_proc=args.n_proc,
        n_starts=args.n_starts,
        n_rounds=args.n_rounds,
        err=args.err,
        out_file=out_file,
    )

    json_out = os.path.join(
        args.out_dir,
        f"spiq_initial_point_input{args.input_idx}_reps{args.reps}.json",
    )

    payload = {
        "input_idx": args.input_idx,
        "reps": args.reps,
        "threshold": args.threshold,
        "num_decimal_pos": args.num_decimal_pos,
        "n_gens": args.n_gens,
        "initial_point": result["initial_point"],
        "energy_best": result["energy_best"],
        "noisy_energy_best": result["noisy_energy_best"],
        "best_cafqa_gen_fitness": result["best_cafqa_gen_fitness"],
        "ks_best_str": result["ks_best_str"],
        "spiq_trace_file": out_file,
    }

    with open(json_out, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n===== SPIQ INITIALIZATION COMPLETE =====")
    print(f"Input index          : {args.input_idx}")
    print(f"Reps                 : {args.reps}")
    print(f"Best energy          : {result['energy_best']}")
    print(f"SPIQ trace file      : {out_file}")
    print(f"JSON output          : {json_out}")
    print("\nUse this initial point in IBMQExperiments.py:\n")
    print(result["initial_point"])
    print("========================================\n")


if __name__ == "__main__":
    main()
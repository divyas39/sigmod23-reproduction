#!/usr/bin/env python
# coding: utf-8

# Script to run SPIQ/CAFQA for generating a QAOA initial point by modelling the QUBO for join order optimization.

import os
import json
import argparse
import numpy as np

import Scripts.ProblemGenerator as ProblemGenerator
import Scripts.QUBOGenerator1 as QUBOGenerator1
# from qiskit_algorithms import NumPyMinimumEigensolver


from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz

from clapton.circuit_manipulation import (
    generate_qiskit_param_map,
    modify_circuit,
    qiskit_to_stim,
    relax_qaoa_parameters,
    transform_to_allowed_gates,
)
from clapton.clapton import claptonize
from clapton.depolarization import GateGeneralDepolarizationModel


def build_problem(input_idx: int):
    """
    Build the same QUBO instance style used by IBMQExperiments.py.
    """
    json_path = os.path.join(
        os.path.dirname(__file__),
        f"ExperimentalAnalysis/IBMQ/QPUPerformance/Problems/JSON/{input_idx}_predicates",
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
    Create a QAOA ansatz and the stim circuit needed by SPIQ.
    Parameters are relaxed (one per gate) so the stim 1-to-1 param map works.
    Returns angle_multipliers so the caller can map back to vanilla QAOA.
    """
    op, _ = qubo.to_ising()

    qaoa_ansatz = QAOAAnsatz(op, reps=reps)

    modified_circ = modify_circuit(qaoa_ansatz)
    pcirc = transform_to_allowed_gates(modified_circ)
    pcirc, _, angle_multipliers = relax_qaoa_parameters(pcirc)

    stim_circ = qiskit_to_stim(pcirc)
    param_map = generate_qiskit_param_map(pcirc)
    stim_circ.define_parameter_map(param_map)

    return op, qaoa_ansatz, pcirc, stim_circ, param_map, angle_multipliers


def _clifford_to_vanilla_initial_point(ks_best, pcirc, angle_multipliers, qaoa_ansatz):
    """
    Convert per-gate Clifford values from SPIQ back to a vanilla QAOA
    initial point (2 * reps floats in qaoa_ansatz.parameters order).

    Each relaxed param named '{mult}*gamma_N' or '{mult}*beta_N' implies
    a vanilla angle theta = (k * pi/2) / mult. We average over all gates
    that share the same type (gamma / beta) per QAOA rep.
    """
    ordered_names = [p.name for p in pcirc.parameters]

    gamma_thetas = []
    beta_thetas = []

    for i, name in enumerate(ordered_names):
        k = int(ks_best[i])
        angle = k * np.pi / 2.0
        mult = angle_multipliers.get(name, 1.0)
        theta = angle / abs(mult) if mult != 0 else 0.0

        if "gamma" in name:
            gamma_thetas.append(theta)
        elif "beta" in name:
            beta_thetas.append(theta)

    vanilla_params = qaoa_ansatz.parameters
    reps = len(vanilla_params) // 2
    gamma_avg = float(np.mean(gamma_thetas)) if gamma_thetas else 0.0
    beta_avg = float(np.mean(beta_thetas)) if beta_thetas else 0.0

    initial_point = []
    for p in vanilla_params:
        pname = p.name
        if "\u03b3" in pname or "gamma" in pname.lower():
            initial_point.append(gamma_avg)
        elif "\u03b2" in pname or "beta" in pname.lower():
            initial_point.append(beta_avg)
        else:
            initial_point.append(0.0)

    return initial_point


# def evaluate_exact_energy():
#     """
#     Solve the problem classically using the NumPyMinimumEigensolver.

#     Returns:
#         The exact energy value.
#     """
#     eigensolver = NumPyMinimumEigensolver()
#     exact_solution = eigensolver.compute_minimum_eigenvalue(
#         cost_hamiltonian
#     ).eigenvalue.real
#     print("Exact Energy from Eigensolver:", exact_solution)
#     exact_energy = exact_solution
#     return exact_solution


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
    op, qaoa_ansatz, pcirc, stim_circ, param_map, angle_multipliers = (
        build_vanilla_spiq_objects(qubo, reps=reps)
    )

    print("Length of stim_circ:", stim_circ.gates.__len__())

    paulis = op.primitive.paulis.to_labels()
    coeffs = op.primitive.coeffs.real
    reversed_paulis = [p[::-1] for p in paulis]

    # if err is not None:
    #     nm = GateGeneralDepolarizationModel(p1=err, p2=10 * err)
    #     stim_circ.add_depolarization_model(nm)

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

    # initial_point = _clifford_to_vanilla_initial_point(
    #     ks_best, pcirc, angle_multipliers, qaoa_ansatz
    # )

    stim_circ.assign(ks_best)
    print("Ks Best: ", ks_best)
    print("Energy Best: ", energy_best)
    print("Length of pcirc:", len(pcirc))
    
    ordered_params = [param.name for param in pcirc.parameters]
    angle_multipliers = [-np.pi/4 if 'gamma' in param else np.pi/4 for param in ordered_params]

    random_params = np.random.uniform(0, 2 * np.pi, len(ordered_params))
    cafqa_params = [param * (np.pi/2) for param, multiplier in zip(ks_best, angle_multipliers)] #This has to be in the order we come across the gates.

    expected_len = 2 * reps
    if len(cafqa_params) != expected_len:
        raise ValueError(
            f"Expected vanilla QAOA initial point of length {expected_len}, "
            f"but got {len(cafqa_params)}"
        )

    print("CAFQA params (angle values for each relaxed param): ", cafqa_params)

    return {
        "initial_point": cafqa_params,
        "energy_best": float(energy_best),
        "noisy_energy_best": None if noisy_energy_best is None else float(noisy_energy_best),
        "best_cafqa_gen_fitness": (
            None if best_cafqa_gen_fitness is None
            else [float(v) for v in best_cafqa_gen_fitness]
            if isinstance(best_cafqa_gen_fitness, (list, np.ndarray))
            else float(best_cafqa_gen_fitness)
        ),
        "ks_best_raw": [int(k) for k in ks_best],
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
        # "threshold": args.threshold,
        # "num_decimal_pos": args.num_decimal_pos,
        "n_gens": args.n_gens,
        "initial_point": result["initial_point"],
        "energy_best": result["energy_best"],
        "noisy_energy_best": result["noisy_energy_best"],
        "best_cafqa_gen_fitness": result["best_cafqa_gen_fitness"],
        "ks_best_raw": result["ks_best_raw"],
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
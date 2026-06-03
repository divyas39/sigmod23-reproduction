#!/usr/bin/env python
# coding: utf-8

# Script to run SPIQ/CAFQA for generating a QAOA initial point by modelling the QUBO for join order optimization.

import os
import re
import json
import argparse
import numpy as np
from cafqa_selection import select_clustering_stratified_parameters
import time

import Scripts.ProblemGenerator as ProblemGenerator
import Scripts.QUBOGenerator1 as QUBOGenerator1
from qiskit.circuit import ParameterExpression
from qiskit.converters import circuit_to_dag
from qiskit.algorithms import NumPyMinimumEigensolver

try:
    from qiskit.qpy import dump as qpy_dump
except ImportError:
    from qiskit.circuit.qpy_serialization import dump as qpy_dump

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


def _parse_rep_index(param_name: str) -> int:
    m = re.search(r"(\d+)", param_name)
    return int(m.group(1)) if m else 0


def _extract_true_multipliers(pcirc):
    dag = circuit_to_dag(pcirc)
    true_mults: dict[str, float] = {}

    for node in dag.op_nodes():
        if not node.op.params:
            continue

        p0 = node.op.params[0]
        if not isinstance(p0, ParameterExpression) or not p0.parameters:
            continue

        param = list(p0.parameters)[0]

        try:
            mult = float(p0.bind({param: 1.0}))
        except Exception:
            try:
                mult = float(p0.subs({param: 1.0}))
            except Exception:
                continue

        true_mults[param.name] = mult

    return true_mults


def _build_relaxed_name_to_rep(pre_relax_circ):
    dag = circuit_to_dag(pre_relax_circ)

    gamma_counter, beta_counter = 0, 0
    name_to_rep: dict[str, int] = {}

    for node in dag.op_nodes():
        if not node.op.params:
            continue

        p0 = node.op.params[0]
        if not isinstance(p0, ParameterExpression) or not p0.parameters:
            continue

        original_name = list(p0.parameters)[0].name
        rep = _parse_rep_index(original_name)
        multiplier = float(str(p0).split("*")[0])

        if "β" in original_name or "beta" in original_name:
            relaxed_name = f"{multiplier}*beta_{beta_counter}"
            beta_counter += 1
        elif "γ" in original_name or "gamma" in original_name:
            relaxed_name = f"{multiplier}*gamma_{gamma_counter}"
            gamma_counter += 1
        else:
            continue

        name_to_rep[relaxed_name] = rep

    return name_to_rep


def build_problem(input_idx: int):
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
        pred_sel,
    )

    return qubo, card, pred, pred_sel, penalty_weight


def create_QAOA_circuit(qubo, reps: int = 1):
    op, _ = qubo.to_ising()
    return QAOAAnsatz(op, reps=reps).decompose()


def decompose_circuit(circuit, max_unchanged_repetitions=3):
    unchanged_counter = 0

    while unchanged_counter < max_unchanged_repetitions:
        old_depth = circuit.depth()
        circuit = circuit.decompose()

        if old_depth == circuit.depth():
            unchanged_counter += 1
        else:
            unchanged_counter = 0

    return circuit


def build_vanilla_spiq_objects(qubo, reps: int):
    op, ising_offset = qubo.to_ising()

    circuit = create_QAOA_circuit(qubo, reps=reps)
    qaoa_ansatz = decompose_circuit(circuit)

    modified_circ = modify_circuit(qaoa_ansatz)
    print("Length of modified_circ:", len(modified_circ))

    pcirc = transform_to_allowed_gates(modified_circ)

    name_to_rep = _build_relaxed_name_to_rep(pcirc)

    observed_reps = set(name_to_rep.values())
    expected_reps = set(range(reps))

    if observed_reps != expected_reps:
        raise RuntimeError(
            "Rep-index extraction from relaxed parameters did not match the "
            f"expected QAOA layers. Expected {sorted(expected_reps)}, got "
            f"{sorted(observed_reps)}. This usually means the original QAOA "
            "parameter names do not match the regex in `_parse_rep_index`."
        )

    pcirc_new, _, angle_multipliers = relax_qaoa_parameters(pcirc)

    stim_circ = qiskit_to_stim(pcirc_new)
    param_map = generate_qiskit_param_map(pcirc_new)
    stim_circ.define_parameter_map(param_map)

    return (
        op,
        ising_offset,
        qaoa_ansatz,
        pcirc_new,
        stim_circ,
        param_map,
        angle_multipliers,
        name_to_rep,
    )


def _clifford_to_vanilla_initial_point(
    ks_best,
    pcirc,
    angle_multipliers,
    qaoa_ansatz,
    name_to_rep,
):
    ordered_names = [p.name for p in pcirc.parameters]
    vanilla_params = qaoa_ansatz.parameters
    reps = max(1, len(vanilla_params) // 2)

    gamma_per_rep: list[list[float]] = [[] for _ in range(reps)]
    beta_per_rep: list[list[float]] = [[] for _ in range(reps)]

    for i, name in enumerate(ordered_names):
        k = int(ks_best[i])
        angle = k * np.pi / 2.0
        mult = angle_multipliers.get(name, 1.0)

        theta = angle / mult if mult != 0 else 0.0

        rep = name_to_rep.get(name, 0)
        if rep >= reps:
            continue

        if "gamma" in name:
            gamma_per_rep[rep].append(theta)
        elif "beta" in name:
            beta_per_rep[rep].append(theta)

    gamma_avg = [float(np.mean(g)) if g else 0.0 for g in gamma_per_rep]
    beta_avg = [float(np.mean(b)) if b else 0.0 for b in beta_per_rep]

    initial_point = []

    for p in vanilla_params:
        pname = p.name
        rep = _parse_rep_index(pname)
        rep = rep if rep < reps else 0

        if "γ" in pname or "gamma" in pname.lower():
            initial_point.append(gamma_avg[rep])
        elif "β" in pname or "beta" in pname.lower():
            initial_point.append(beta_avg[rep])
        else:
            initial_point.append(0.0)

    return initial_point


def run_spiq_initialization(
    qubo,
    reps: int,
    n_gens: int,
    n_proc: int = 32,
    n_starts: int = 4,
    n_rounds: int = 1,
    err: float = None,
    out_file: str = None,
    selection_strategy="single", 
    num_select=5, 
    selection_seed=None,
):
    (
        op,
        ising_offset,
        qaoa_ansatz,
        pcirc,
        stim_circ,
        param_map,
        angle_multipliers,
        name_to_rep,
    ) = build_vanilla_spiq_objects(qubo, reps=reps)

    print("Length of stim_circ:", len(stim_circ.gates))

    paulis = op.primitive.paulis.to_labels()
    coeffs = op.primitive.coeffs.real
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

    stim_circ.assign(ks_best)

    energy_best_ising = float(energy_best)
    energy_best_qubo = float(energy_best_ising + ising_offset)

    print("Ks Best: ", ks_best)
    print("Energy Best Ising: ", energy_best_ising)
    print("Ising offset: ", float(ising_offset))
    print("Energy Best QUBO / offset-adjusted: ", energy_best_qubo)
    print("Length of pcirc:", len(pcirc))

    applied_multipliers = _extract_true_multipliers(pcirc)

    n_differ = sum(
        1
        for name in angle_multipliers
        if name in applied_multipliers
        and abs(angle_multipliers[name] - applied_multipliers[name]) > 1e-9
    )

    print(
        f"[diag] applied (pcirc) vs original (name-encoded) multipliers differ "
        f"on {n_differ}/{len(angle_multipliers)} gates -- expected when "
        f"relax_qaoa_parameters hoists the coeff into the parameter name."
    )

    initial_point = _clifford_to_vanilla_initial_point(
        ks_best,
        pcirc,
        angle_multipliers,
        qaoa_ansatz,
        name_to_rep,
    )

    expected_len = 2 * reps
    if len(initial_point) != expected_len:
        raise ValueError(
            f"Expected vanilla QAOA initial point of length {expected_len}, "
            f"but got {len(initial_point)}"
        )

    relaxed_param_names = [p.name for p in pcirc.parameters]

    relaxed_initial_point = []
    for i, name in enumerate(relaxed_param_names):
        applied_mult = applied_multipliers.get(name, 1.0)

        if applied_mult == 0:
            relaxed_initial_point.append(0.0)
            continue

        relaxed_initial_point.append(
            (int(ks_best[i]) * np.pi / 2.0) / applied_mult
        )

    selected_initial_points = []
    selected_relaxed_initial_points = []
    selected_energy_best = []
    selected_energy_best_qubo = []
    selected_grad_norms = []
    if selection_strategy == "clustering" and best_cafqa_gen_params is not None:
        sel_ks, sel_fit, sel_grad = select_clustering_stratified_parameters(
            best_cafqa_gen_params, best_cafqa_gen_fitness,
            pcirc=pcirc, hamiltonian=op,
            num_select=num_select, seed=selection_seed,
        )

        selected_energy_best = [float(e) for e in sel_fit]
        selected_energy_best_qubo = [float(e + ising_offset) for e in sel_fit]
        selected_grad_norms = [float(g) for g in sel_grad]

        for ks_vec in sel_ks:
            ip = _clifford_to_vanilla_initial_point(
                ks_vec, pcirc, angle_multipliers, qaoa_ansatz, name_to_rep,
            )
            selected_initial_points.append(ip)

            rip = []
            for i, name in enumerate(relaxed_param_names):
                applied_mult = applied_multipliers.get(name, 1.0)
                if applied_mult == 0:
                    rip.append(0.0)
                else:
                    rip.append((int(ks_vec[i]) * np.pi / 2.0) / applied_mult)
            selected_relaxed_initial_points.append(rip)

    try:
        from qiskit.quantum_info import Statevector

        def _ising_expectation(theta_vec):
            bound = pcirc.assign_parameters(
                {p: float(v) for p, v in zip(pcirc.parameters, theta_vec)},
                inplace=False,
            )
            sv = Statevector.from_instruction(bound)
            return float(np.real(sv.expectation_value(op)))

        e_qiskit_ising = _ising_expectation(relaxed_initial_point)
        e_qiskit_qubo = float(e_qiskit_ising + ising_offset)

        rel_err = abs(e_qiskit_ising - energy_best_ising) / max(
            abs(energy_best_ising),
            1e-9,
        )

        print(
            f"[diag] stim energy_best Ising = {energy_best_ising:.6f}; "
            f"qiskit Statevector <H>_Ising at relaxed_initial_point = "
            f"{e_qiskit_ising:.6f}; "
            f"qiskit Statevector <H>_QUBO = {e_qiskit_qubo:.6f}; "
            f"offset = {float(ising_offset):.6f}; "
            f"rel_err = {rel_err:.3e}"
        )

        if rel_err > 1e-4:
            print(
                "[diag] WARNING: qiskit pcirc does not reproduce stim's "
                "Clifford state. Per-gate details (first 20):"
            )

            for i, name in enumerate(relaxed_param_names[:20]):
                print(
                    f"    [{i:3d}] name={name!r:40s} "
                    f"applied_mult={applied_multipliers.get(name, 1.0):+.4f} "
                    f"original_mult={angle_multipliers.get(name, 1.0):+.4f} "
                    f"k={int(ks_best[i])} "
                    f"theta={relaxed_initial_point[i]:+.4f}"
                )

    except Exception as exc:
        print(
            f"[diag] statevector verification unavailable ({exc!r}); "
            f"skipping internal sanity check."
        )

    neg_mults = {
        name: angle_multipliers[name]
        for name in relaxed_param_names
        if angle_multipliers.get(name, 0.0) < 0
    }

    print(
        f"original angle_multipliers (used for vanilla conversion only): "
        f"{len(angle_multipliers)} gates, {len(neg_mults)} with negative sign."
    )

    print(
        "Vanilla QAOA initial point (in qaoa_ansatz.parameters order): ",
        initial_point,
    )

    for rep_idx in range(reps):
        rep_slice = [
            float(v)
            for v, p in zip(initial_point, qaoa_ansatz.parameters)
            if _parse_rep_index(p.name) == rep_idx
        ]
        print(f"  rep {rep_idx}: {rep_slice}")

    print(
        f"Relaxed per-gate initial point: {len(relaxed_initial_point)} angles "
        "(use with pcirc, not qaoa_ansatz)"
    )

    return {
        "initial_point": initial_point,
        "relaxed_initial_point": [float(v) for v in relaxed_initial_point],
        "relaxed_param_names": relaxed_param_names,
        "pcirc": pcirc,
        "selected_initial_points": selected_initial_points,
        "selected_relaxed_initial_points": selected_relaxed_initial_points,
        "selected_energy_best": selected_energy_best,
        "selected_energy_best_qubo": selected_energy_best_qubo,
        "selected_grad_norms": selected_grad_norms,


        # Raw Ising-scale energy from SPIQ/CAFQA.
        # Keep this as `energy_best` because IBMQExperiments.py sanity check
        # expects this field to be Ising-scale and compares QUBO - offset to it.
        "energy_best": energy_best_ising,

        # QUBO-scale value, comparable against energy_per_iteration CSVs from
        # the SPIQ-mode IBMQ flow because that path evaluates qubo.objective.
        "energy_best_qubo": energy_best_qubo,
        "ising_offset": float(ising_offset),

        "noisy_energy_best": None if noisy_energy_best is None else float(noisy_energy_best),
        "best_cafqa_gen_fitness": (
            None
            if best_cafqa_gen_fitness is None
            else [float(v) for v in best_cafqa_gen_fitness]
            if isinstance(best_cafqa_gen_fitness, (list, np.ndarray))
            else float(best_cafqa_gen_fitness)
        ),
        "ks_best_raw": [int(k) for k in ks_best],
    }


def evaluate_exact_ground_state_energy(qubo):
    """
    Compute theoretical ground-state energy for the join-ordering QUBO.

    Returns:
        exact Ising-scale energy, QUBO-scale energy, and Ising offset.
    """
    op, offset = qubo.to_ising()

    solver = NumPyMinimumEigensolver()
    result = solver.compute_minimum_eigenvalue(op)

    exact_ising_energy = float(result.eigenvalue.real)
    exact_qubo_energy = exact_ising_energy + float(offset)

    print("\n===== EXACT GROUND STATE ENERGY =====")
    print("Exact ground-state energy, Ising scale:", exact_ising_energy)
    print("Ising/QUBO offset:", float(offset))
    print("Exact ground-state energy, QUBO scale:", exact_qubo_energy)
    print("=====================================\n")

    return {
        "exact_ising_energy": exact_ising_energy,
        "ising_offset": float(offset),
        "exact_qubo_energy": exact_qubo_energy,
    }


def main():
    total_start = time.perf_counter()

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_idx", type=int, default=0, help="Problem index")
    parser.add_argument("--reps", type=int, default=1, help="QAOA reps / p")
    parser.add_argument("--n_gens", type=int, default=200, help="SPIQ generation budget")
    parser.add_argument("--n_proc", type=int, default=32, help="Number of processes for SPIQ")
    parser.add_argument("--n_starts", type=int, default=4, help="Number of SPIQ starts")
    parser.add_argument("--n_rounds", type=int, default=1, help="Number of SPIQ rounds")
    parser.add_argument("--err", type=float, default=None, help="Optional depolarization error")

    parser.add_argument("--selection-strategy", choices=["single","clustering"], default="single")
    parser.add_argument("--num-select", type=int, default=5)
    parser.add_argument("--selection-seed", type=int, default=None)

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

    exact_ground_state = evaluate_exact_ground_state_energy(qubo)

    shape_suffix = f"{len(card)}rel_{len(pred)}pred"
    base_stub = f"input{args.input_idx}_reps{args.reps}_{shape_suffix}"

    out_file = os.path.join(
        args.out_dir,
        f"spiq_trace_{base_stub}.txt",
    )

    spiq_start = time.perf_counter()

    result = run_spiq_initialization(
        qubo=qubo,
        reps=args.reps,
        n_gens=args.n_gens,
        n_proc=args.n_proc,
        n_starts=args.n_starts,
        n_rounds=args.n_rounds,
        err=args.err,
        out_file=out_file,
        selection_strategy=args.selection_strategy,
        num_select=args.num_select,
        selection_seed=args.selection_seed,
    )
    spiq_runtime_sec = time.perf_counter() - spiq_start

    json_out = os.path.join(
        args.out_dir,
        f"spiq_initial_point_{base_stub}.json",
    )

    pcirc_qpy = os.path.join(
        args.out_dir,
        f"spiq_pcirc_{base_stub}.qpy",
    )

    with open(pcirc_qpy, "wb") as fqpy:
        qpy_dump(result["pcirc"], fqpy)

    payload = {
        "input_idx": args.input_idx,
        "reps": args.reps,
        "n_gens": args.n_gens,
        "problem_input": {
            "input_idx": args.input_idx,
            "source": (
                f"ExperimentalAnalysis/IBMQ/QPUPerformance/Problems/JSON/"
                f"{args.input_idx}_predicates"
            ),
            "num_relations": len(card),
            "num_predicates": len(pred),
            "card": card,
            "pred": [list(p) for p in pred],
            "pred_sel": pred_sel,
            "penalty_weight": float(penalty_weight),
        },
        "initial_point": result["initial_point"],
        "relaxed_initial_point": result["relaxed_initial_point"],

        "selected_initial_points": result["selected_initial_points"],
        "selected_relaxed_initial_points": result["selected_relaxed_initial_points"],
        "selected_energy_best": result["selected_energy_best"],
        "selected_energy_best_qubo": result["selected_energy_best_qubo"],
        "selected_grad_norms": result["selected_grad_norms"],
        "relaxed_param_names": result["relaxed_param_names"],
        "pcirc_qpy": pcirc_qpy,

        # Raw Ising value. Keep for IBMQExperiments.py sanity check.
        "energy_best": result["energy_best"],

        # Offset-adjusted QUBO value. Use this when comparing to IBMQ
        # energy_per_iteration CSV values from the SPIQ-mode flow.
        "energy_best_qubo": result["energy_best_qubo"],
        "ising_offset": result["ising_offset"],

        "noisy_energy_best": result["noisy_energy_best"],
        "best_cafqa_gen_fitness": result["best_cafqa_gen_fitness"],
        "ks_best_raw": result["ks_best_raw"],
        "spiq_trace_file": out_file,
        "exact_ground_state": exact_ground_state,
        "runtime_sec": {
    "spiq_initialization_core": float(spiq_runtime_sec),
    "spiq_initialization_total": float(time.perf_counter() - total_start),
},
    }

    with open(json_out, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n===== SPIQ INITIALIZATION COMPLETE =====")
    print(f"Input index          : {args.input_idx}")
    print(f"Reps                 : {args.reps}")
    print(f"Best energy Ising    : {result['energy_best']}")
    print(f"Ising offset         : {result['ising_offset']}")
    print(f"Best energy QUBO     : {result['energy_best_qubo']}")
    print(f"SPIQ trace file      : {out_file}")
    print(f"JSON output          : {json_out}")
    print(f"Relaxed pcirc (QPY)  : {pcirc_qpy}")
    print(f"SPIQ core runtime sec: {spiq_runtime_sec:.3f}")
    print(f"SPIQ total runtime sec: {time.perf_counter() - total_start:.3f}")

    print("\nVanilla QAOA initial point (2*reps angles):\n")
    print(result["initial_point"])

    print(
        f"\nRelaxed per-gate initial point "
        f"({len(result['relaxed_initial_point'])} angles, one per relaxed gate):"
    )
    print(result["relaxed_initial_point"])
    print("========================================\n")


if __name__ == "__main__":
    main()
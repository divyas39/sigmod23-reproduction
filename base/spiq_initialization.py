#!/usr/bin/env python
# coding: utf-8

# Script to run SPIQ/CAFQA for generating a QAOA initial point by modelling the QUBO for join order optimization.

import os
import re
import json
import argparse
import numpy as np

import Scripts.ProblemGenerator as ProblemGenerator
import Scripts.QUBOGenerator1 as QUBOGenerator1
from qiskit.circuit import ParameterExpression
from qiskit.converters import circuit_to_dag

try:
    from qiskit.qpy import dump as qpy_dump
except ImportError:
    from qiskit.circuit.qpy_serialization import dump as qpy_dump

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


def _parse_rep_index(param_name: str) -> int:
    """
    Extract the rep/layer index from a QAOA parameter name like
    'γ[1]', 'β[0]', 'gamma_2', or 'beta_1'. Returns 0 if none found.
    """
    m = re.search(r"(\d+)", param_name)
    return int(m.group(1)) if m else 0


def _extract_true_multipliers(pcirc):
    """
    Re-extract the coefficient of each parametric gate in `pcirc` by
    numerically evaluating its ParameterExpression at param=1.0.

    The string-parsing approach used by `relax_qaoa_parameters`
    (`float(str(expr).split("*")[0])`) is fragile: if sympy/symengine
    doesn't auto-fold into a single numeric factor, only the first
    token is captured and the rest of the coefficient is silently
    dropped. Evaluating expr at 1.0 gives the true scalar regardless
    of string form.

    Returns a dict {param_name: true_multiplier} that can shadow the
    `angle_multipliers` dict built by `relax_qaoa_parameters`.
    """
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
    """
    Walk the DAG of the pre-relaxation circuit in the same order that
    `relax_qaoa_parameters` does, and produce a dict mapping the
    future relaxed parameter name (e.g. '2.0*gamma_3') to the original
    QAOA rep/layer index encoded in its source parameter name (e.g. 'γ[1]').

    We replicate the counter logic of `relax_qaoa_parameters` exactly so
    the emitted names match 1-to-1 what ends up in `pcirc`.
    """
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
    Returns angle_multipliers and a map of relaxed-param-name to QAOA rep
    index so the caller can map back to vanilla QAOA per layer.
    """
    op, _ = qubo.to_ising()

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
            "parameter names don't match the regex in `_parse_rep_index` "
            "(e.g. your qiskit version names them differently than "
            "'\u03b3[0]' / '\u03b2[1]')."
        )

    pcirc_new, _, angle_multipliers = relax_qaoa_parameters(pcirc)

    stim_circ = qiskit_to_stim(pcirc_new)
    param_map = generate_qiskit_param_map(pcirc_new)
    stim_circ.define_parameter_map(param_map)

    return (
        op,
        qaoa_ansatz,
        pcirc_new,
        stim_circ,
        param_map,
        angle_multipliers,
        name_to_rep,
    )


def create_QAOA_circuit(qubo, reps: int = 1):
    """
    Build a parameterized QAOA circuit for the given QUBO with `reps` layers.
    Returns the QAOAAnsatz with its original γ[p]/β[p] parameters intact so
    the relaxation step downstream can recover per-layer angles.
    """
    op, _ = qubo.to_ising()
    return QAOAAnsatz(op, reps=reps).decompose()


# Decompose the circuit to its basis gates
def decompose_circuit(circuit, max_unchanged_repetitions=3):
    unchanged_counter = 0
    while unchanged_counter < max_unchanged_repetitions:
        old_depth = circuit.depth()
        circuit = circuit.decompose()
        if old_depth == circuit.depth():
            # Increase the counter if the circuit has not changed
            unchanged_counter = unchanged_counter + 1
        else:
            # Otherwise reset the counter
            unchanged_counter = 0
    return circuit



def _clifford_to_vanilla_initial_point(
    ks_best, pcirc, angle_multipliers, qaoa_ansatz, name_to_rep
):
    """
    Convert per-gate Clifford values from SPIQ back to a vanilla QAOA
    initial point (2 * reps floats in qaoa_ansatz.parameters order).

    Each relaxed param named '{mult}*gamma_N' or '{mult}*beta_N' implies
    a vanilla angle theta = (k * pi/2) / mult (signed division: the
    unrelaxed QAOAAnsatz applies Rz(mult*gamma) literally, so the sign
    of mult matters -- dropping it via abs(mult) flips S <-> Sdg on any
    gate whose coeff is negative). We bucket these thetas by
    (param_type, rep_index) so every QAOA layer gets its own gamma/beta,
    averaged over the relaxed gates that came from it.
    """
    ordered_names = [p.name for p in pcirc.parameters]
    vanilla_params = qaoa_ansatz.parameters
    reps = max(1, len(vanilla_params) // 2)

    gamma_per_rep: list[list[float]] = [[] for _ in range(reps)]
    beta_per_rep: list[list[float]] = [[] for _ in range(reps)]

    for i, name in enumerate(ordered_names):
        k = int(ks_best[i])
        angle = k * np.pi / 2.0
        mult = angle_multipliers.get(name, 1.0)
        # Signed division: qiskit_to_stim emits a raw `Rz(k*pi/2)` for each
        # parametric gate, ignoring the pcirc-side multiplier. So to make the
        # qiskit pcirc reproduce the same Clifford state as stim we need
        # mult * theta == k*pi/2  =>  theta = k*pi/2 / mult (signed).
        # Using abs(mult) flips S <-> Sdg wherever mult < 0 and scrambles the
        # Clifford state.
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
        if "\u03b3" in pname or "gamma" in pname.lower():
            initial_point.append(gamma_avg[rep])
        elif "\u03b2" in pname or "beta" in pname.lower():
            initial_point.append(beta_avg[rep])
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
    (
        op,
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

    stim_circ.assign(ks_best)
    print("Ks Best: ", ks_best)
    print("Energy Best: ", energy_best)
    print("Length of pcirc:", len(pcirc))

    # Two different multipliers live in this pipeline and they do NOT play
    # the same role:
    #   * `angle_multipliers` (from clapton's string-parse of the relaxed
    #     parameter name, e.g. "-0.977*gamma_0") is the ORIGINAL coefficient
    #     from QAOAAnsatz. In the UNRELAXED ansatz the cost-layer Rz applies
    #     Rz(mult_original * gamma), so that's what we need to invert when
    #     reconstructing a vanilla gamma from stim's k choice.
    #   * `applied_multipliers` (from numerical eval of pcirc's parameter
    #     expression at param=1) is the coefficient that the RELAXED pcirc
    #     actually multiplies the parameter value by. Because
    #     `relax_qaoa_parameters` only bakes the original coeff into the
    #     parameter's *name*, every relaxed gate is literally Rz(1.0*param),
    #     i.e. applied_mult == 1.0 for every gate. So for the relaxed
    #     binding the only correct formula is theta = k*pi/2 (no division).
    applied_multipliers = _extract_true_multipliers(pcirc)
    n_differ = sum(
        1 for name in angle_multipliers
        if name in applied_multipliers
        and abs(angle_multipliers[name] - applied_multipliers[name]) > 1e-9
    )
    print(
        f"[diag] applied (pcirc) vs original (name-encoded) multipliers differ "
        f"on {n_differ}/{len(angle_multipliers)} gates -- expected when "
        f"relax_qaoa_parameters hoists the coeff into the parameter name."
    )

    initial_point = _clifford_to_vanilla_initial_point(
        ks_best, pcirc, angle_multipliers, qaoa_ansatz, name_to_rep
    )

    expected_len = 2 * reps
    if len(initial_point) != expected_len:
        raise ValueError(
            f"Expected vanilla QAOA initial point of length {expected_len}, "
            f"but got {len(initial_point)}"
        )

    relaxed_param_names = [p.name for p in pcirc.parameters]
    # Relaxed binding: pcirc applies Rz(applied_mult * theta); stim applies
    # Rz(k*pi/2). For equivalence: applied_mult * theta = k*pi/2.
    relaxed_initial_point = []
    for i, name in enumerate(relaxed_param_names):
        applied_mult = applied_multipliers.get(name, 1.0)
        if applied_mult == 0:
            relaxed_initial_point.append(0.0)
            continue
        relaxed_initial_point.append(
            (int(ks_best[i]) * np.pi / 2.0) / applied_mult
        )

    # Empirical verification: bind relaxed_initial_point into pcirc,
    # compute the Ising expectation exactly with Statevector, and compare
    # to stim's reported energy_best. Clifford states give noise-free
    # expectations, so a match here should be at machine precision.
    try:
        from qiskit.quantum_info import Statevector

        def _ising_expectation(theta_vec):
            bound = pcirc.assign_parameters(
                {p: float(v) for p, v in zip(pcirc.parameters, theta_vec)},
                inplace=False,
            )
            sv = Statevector.from_instruction(bound)
            return float(np.real(sv.expectation_value(op)))

        e_qiskit = _ising_expectation(relaxed_initial_point)
        rel_err = abs(e_qiskit - energy_best) / max(abs(energy_best), 1e-9)
        print(
            f"[diag] stim energy_best = {energy_best:.6f}; "
            f"qiskit Statevector <H>_Ising at relaxed_initial_point = "
            f"{e_qiskit:.6f}; rel_err = {rel_err:.3e}"
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
        print(f"[diag] statevector verification unavailable ({exc!r}); "
              f"skipping internal sanity check.")

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
    pcirc_qpy = os.path.join(
        args.out_dir,
        f"spiq_pcirc_input{args.input_idx}_reps{args.reps}.qpy",
    )

    with open(pcirc_qpy, "wb") as fqpy:
        qpy_dump(result["pcirc"], fqpy)

    payload = {
        "input_idx": args.input_idx,
        "reps": args.reps,
        # "threshold": args.threshold,
        # "num_decimal_pos": args.num_decimal_pos,
        "n_gens": args.n_gens,
        "initial_point": result["initial_point"],
        "relaxed_initial_point": result["relaxed_initial_point"],
        "relaxed_param_names": result["relaxed_param_names"],
        "pcirc_qpy": pcirc_qpy,
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
    print(f"Relaxed pcirc (QPY)  : {pcirc_qpy}")
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
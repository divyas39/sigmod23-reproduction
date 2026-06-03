#!/usr/bin/env python
# coding: utf-8

# In[1]:

import time
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
import numpy as np
import pickle
from decimal import *
import sys
from types import SimpleNamespace

from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
import Scripts.QUBOGenerator1 as QUBOGenerator1

from qiskit.algorithms.optimizers import AQGD,COBYLA,SPSA
from qiskit.algorithms import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.utils import QuantumInstance
from qiskit import IBMQ
from qiskit.providers.aer import QasmSimulator

try:
    from qiskit import qpy as _qpy
except ImportError:
    from qiskit.circuit import qpy_serialization as _qpy

qpy = _qpy

import config as config

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--trial", type=int, default=1, help="Trial number passed from bash")
parser.add_argument("--reps", type=int, default=1, help="Tag for distinguishing runs")
parser.add_argument("--optimizer",type=int,default=0,help="Check with optimizer is used")
parser.add_argument(
    "--spiq_json",
    type=str,
    default=None,
    help=(
        "Path to SPIQ initialization JSON produced by spiq_initialization.py. "
        "If provided, run COBYLA over the relaxed ansatz (per-gate angles) "
        "instead of the vanilla QAOAAnsatz."
    ),
)
parser.add_argument("--input_idx", type=int, default=0, help="Problem input index")
args, _ = parser.parse_known_args() 

TRIAL_ID = args.trial  
TAG = args.reps
optmi=args.optimizer
INPUT_IDX = args.input_idx
SPIQ_JSON = args.spiq_json
if(optmi==0):
    current_optim="AQGD"
elif(optmi==1):
    current_optim="COBYLA"
else:
    current_optim="SPSA"


# In[2]:


def get_rounded_val(val, decimal_pos = 4):
    return Decimal(val).quantize(Decimal(10) ** -decimal_pos, rounding=ROUND_HALF_EVEN)

def save_results(path_string, depths):

    datapath = os.path.abspath(path_string)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path_string + '/depth.txt')
    mode = 'a' if os.path.exists(datafile) else 'w'
    print("Saving " + path_string)
    with open(datafile, mode) as file:
        json.dump(depths, file)
        
def load_result(path_string):
    datafile = os.path.abspath(path_string + '/depth.txt')
    results = None
    with open(datafile, 'rb') as file:
        results = json.load(file)
    return results
      
def pickle_results(path_string, results):

    datapath = os.path.abspath(path_string)
    pathlib.Path(datapath).mkdir(parents=True, exist_ok=True) 
    
    datafile = os.path.abspath(path_string + '/results.txt')
    print("Saving " + path_string)
    with open(datafile, 'wb') as file:
        pickle.dump(results, file)
        
def load_pickled_result(path_string):
    datapath = os.path.abspath(path_string)
    datafile = os.path.abspath(path_string + '/results.txt')
    results = None
    with open(datafile, 'rb') as file:
        results = pickle.load(file)
    return results

def get_optimizer_string(tket_optimizer, optimization_level):
    if tket_optimizer:
        return 'Tket_optimizer'
    else:
        return 'Qiskit_Lv_' + str(optimization_level) + '_optimizer'

def get_qiskit_IBMQ_gateset(restrict_to_native_gates):
    if restrict_to_native_gates:
        return CircuitGenerator.get_IBMQ_basis_gates()
    else:
        return None
    
def get_qiskit_Rigetti_gateset(restrict_to_native_gates):
    if restrict_to_native_gates:
        return CircuitGenerator.get_Rigetti_basis_gates()
    else:
        return None
    
def get_qiskit_IonQ_gateset(restrict_to_native_gates):
    if restrict_to_native_gates:
        return CircuitGenerator.get_IonQ_basis_gates()
    else:
        return None

def get_tket_IBMQ_gateset_pass(restrict_to_native_gates):
    if restrict_to_native_gates:
        return CircuitGenerator.fetch_IBMQ_rebase_pass()
    else:
        return None
    
def get_tket_IonQ_gateset_pass(restrict_to_native_gates):
    if restrict_to_native_gates:
        return CircuitGenerator.fetch_IonQ_rebase_pass()
    else:
        return None
    
def get_tket_Rigetti_gateset_pass(restrict_to_native_gates):
    if restrict_to_native_gates:
        return CircuitGenerator.fetch_Rigetti_rebase_pass()
    else:
        return None

def get_QAOA_circuit(qubo):
    circuit = CircuitGenerator.create_QAOA_circuit(qubo)
    circuit = CircuitGenerator.decompose_circuit(circuit)
    print("print number of quibits",circuit.num_qubits)
    
    return circuit

def determine_circuit_depth(qubo, coupling_map, tket_optimizer, optimization_level):
    circuit = get_QAOA_circuit(qubo)
    if tket_optimizer:
        circuit = CircuitGenerator.transpile_circuit_with_tket(circuit, coupling_map, get_tket_IBMQ_gateset_pass(True))
    else:
        circuit = CircuitGenerator.transpile_circuit_with_qiskit(circuit, coupling_map, optimization_level, get_qiskit_IBMQ_gateset(True))
    return circuit.depth()


# In[3]:


def get_IBMQ_QASM_backend():
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    backend = provider.get_backend('ibmq_qasm_simulator')
    quantum_instance = QuantumInstance(backend=backend)
    return quantum_instance
    
# Returns a local simulator backend for testing the implementation
def get_local_QASM_backend():
    backend = QasmSimulator()
    quantum_instance = QuantumInstance(backend=backend,shots=10240)
    return quantum_instance

def get_IBMQ_backend():
    ibmq_hub = config.configuration["ibmq-hub"]
    ibmq_group = config.configuration["ibmq-group"]
    ibmq_project = config.configuration["ibmq-project"]
    ibmq_backend = config.configuration["ibmq-backend"]

    # The provider needs to enable access to the ibm_auckland, or a similar 27-qubit, QPU (this is not the case for the default "open" group)
    provider = IBMQ.get_provider(hub=ibmq_hub, group=ibmq_group, project=ibmq_project)
    backend = provider.get_backend(ibmq_backend)
    quantum_instance = QuantumInstance(backend=backend)
    return quantum_instance


def _make_spiq_sample(x, fval, probability):
    """Return a pickle-safe sample object shaped like qiskit_optimization
    samples (.x / .fval / .probability). Using SimpleNamespace avoids the
    `Can't get attribute '_SpiqResponse' on <module '__main__'>` failure
    that custom classes produce when the pickle is loaded from a different
    entrypoint (e.g. Temp.py) than the one that wrote it."""
    return SimpleNamespace(x=x, fval=fval, probability=probability)


def _make_spiq_response(x, fval, samples):
    """Pickle-safe response shaped like MinimumEigenOptimizer.solve()'s
    result (.x / .fval / .samples)."""
    return SimpleNamespace(x=x, fval=fval, samples=samples)


def _load_spiq_initialization(json_path):
    """
    Load the JSON + QPY artifacts produced by spiq_initialization.py.
    Returns (pcirc, relaxed_initial_point, spiq_meta).

    The `pcirc_qpy` field in the JSON may be a relative path. We resolve
    it in this order:
      1. As-is (works if the caller's cwd matches where SPIQ was run).
      2. Relative to the JSON file's own directory (robust across cwd's).
      3. Relative to the JSON file's parent directory (covers
         cwd = base/ launching a JSON that lives at the repo root).
    """
    json_path = os.path.abspath(json_path)
    with open(json_path, "r") as f:
        meta = json.load(f)

    raw_qpy = meta["pcirc_qpy"]
    json_dir = os.path.dirname(json_path)
    candidates = [
        raw_qpy,
        os.path.join(json_dir, os.path.basename(raw_qpy)),
        os.path.join(json_dir, raw_qpy),
        os.path.join(os.path.dirname(json_dir), raw_qpy),
    ]
    qpy_path = next((p for p in candidates if os.path.exists(p)), None)
    if qpy_path is None:
        raise FileNotFoundError(
            f"Could not locate pcirc QPY (declared as {raw_qpy!r} in {json_path}). "
            f"Tried: {candidates}"
        )

    with open(qpy_path, "rb") as f:
        pcirc = qpy.load(f)[0]
    return pcirc, list(meta["relaxed_initial_point"]), meta


def _evaluate_expected_qubo_energy(counts, qubo):
    """
    Diagonal cost Hamiltonian: <H_C> = sum_x p(x) * qubo.objective.evaluate(x).
    Qiskit bitstrings are MSB-first with spaces; we reverse to match qubo var order.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    energy = 0.0
    for bitstring, cnt in counts.items():
        b = bitstring.replace(" ", "")[::-1]
        x = [int(ch) for ch in b]
        try:
            e = float(qubo.objective.evaluate(x))
        except Exception:
            e = float(qubo.objective.evaluate(list(x)))
        energy += (cnt / total) * e
    return energy


def solve_with_QAOA_spiq(
    qubo,
    iterations,
    pcirc,
    relaxed_initial_point,
    reps=TAG,
    use_local_simulator=False,
    result_dir="./9/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data",
    card=None,
    pred=None,
    pred_sel=None,
    thres=None,
    inputNumber=0,
    expected_energy=None,
    sanity_check_tolerance=0.25,
    sanity_check_shots=10240,
):
    """
    Variant of solve_with_QAOA that optimizes over the RELAXED ansatz
    (one parameter per gate, as produced by SPIQ) using a classical
    optimizer over the diagonal QUBO cost. Returns the same
    (response, final_point, used_eval, min_state_buffer) tuple shape as
    `solve_with_QAOA` for drop-in compatibility downstream.
    """
    if use_local_simulator:
        quantum_instance = get_local_QASM_backend()
    else:
        quantum_instance = get_IBMQ_backend()

    os.makedirs(result_dir, exist_ok=True)
    energy_log_path = os.path.join(
        result_dir,
        f"energy_per_iteration_{iterations}_{current_optim}_{TAG}_{TRIAL_ID}.csv",
    )

    energies = []
    min_energy_seen = float("inf")
    min_params = None
    min_eval_idx = None
    last_params = {"val": None}
    last_eval = {"val": 0}

    ordered_params = list(pcirc.parameters)
    n_params = len(ordered_params)
    if len(relaxed_initial_point) != n_params:
        raise ValueError(
            f"relaxed_initial_point length ({len(relaxed_initial_point)}) "
            f"does not match pcirc.parameters length ({n_params})."
        )

    measured_circ = pcirc.copy()
    measured_circ.measure_all()

    def _bind(theta_vec):
        return measured_circ.assign_parameters(
            {p: float(v) for p, v in zip(ordered_params, theta_vec)},
            inplace=False,
        )
    
    def evaluate_qubo_energy_only(theta_vec):
        qc = _bind(theta_vec)
        result = quantum_instance.execute(qc)
        counts = result.get_counts()
        return _evaluate_expected_qubo_energy(counts, qubo)

    def cost_fn(theta_vec):
        nonlocal min_energy_seen, min_params, min_eval_idx
        qc = _bind(theta_vec)
        result = quantum_instance.execute(qc)
        counts = result.get_counts()
        energy = _evaluate_expected_qubo_energy(counts, qubo)

        eval_count = last_eval["val"] + 1
        last_eval["val"] = eval_count
        last_params["val"] = list(theta_vec)
        # 4th column mirrors vanilla QAOA's 'std' column. We don't compute
        # one for the custom cost_fn (it's a deterministic scalar), so emit
        # 0.0 to keep the CSV schema uniform with downstream parsers like
        # Temp.convert_callback_csv_to_history.
        energies.append((eval_count, energy, list(theta_vec), 0.0))

        if energy < min_energy_seen:
            min_energy_seen = energy
            min_params = list(theta_vec)
            min_eval_idx = int(eval_count)
        return energy

    if expected_energy is not None:
        # SPIQ's energy_best is the Ising-basis expectation (no offset).
        # Our simulator evaluates qubo.objective on sampled bitstrings, which
        # is in the QUBO basis. The two differ exactly by the offset returned
        # by qubo.to_ising(), so subtract it before comparing.
        _op, ising_offset = qubo.to_ising()
        sanity_qi = QuantumInstance(backend=QasmSimulator(), shots=sanity_check_shots)
        sanity_qc = _bind(relaxed_initial_point)
        sanity_counts = sanity_qi.execute(sanity_qc).get_counts()
        sanity_energy_qubo = _evaluate_expected_qubo_energy(sanity_counts, qubo)
        sanity_energy_ising = sanity_energy_qubo - float(ising_offset)
        rel_err = abs(sanity_energy_ising - expected_energy) / max(abs(expected_energy), 1e-9)
        print(
            f"[spiq-sanity] expected_energy (from SPIQ, Ising) = {expected_energy:.6f}, "
            f"simulator <H_C>_QUBO at relaxed_initial_point = {sanity_energy_qubo:.6f}, "
            f"simulator <H_C>_Ising (= QUBO - offset {ising_offset:.4f}) = {sanity_energy_ising:.6f}, "
            f"rel_err = {rel_err:.3f}"
        )
        if rel_err > sanity_check_tolerance:
            raise RuntimeError(
                f"Sanity check failed: simulator Ising energy {sanity_energy_ising:.6f} diverges "
                f"from SPIQ reported {expected_energy:.6f} (relative error {rel_err:.3f} > "
                f"tolerance {sanity_check_tolerance}). This usually means the relaxed "
                "pcirc is not being bound to the same Clifford state as stim. "
                "Common causes: dropped sign on the gate multiplier when computing "
                "`relaxed_initial_point`, bitstring endianness mismatch in "
                "`_evaluate_expected_qubo_energy`, or a stale QPY file."
            )

    if optmi == 1:
        optimizer = COBYLA(maxiter=iterations, rhobeg=2.0, tol=1e-12, disp=True)
    elif optmi == 2:
        optimizer = SPSA(maxiter=iterations)
    else:
        optimizer = AQGD(maxiter=iterations, eta=0.01)

     # Now let optimizer start from the SPIQ point
    opt_result = optimizer.minimize(
        fun=cost_fn,
        x0=np.asarray(relaxed_initial_point)
    )

    initial_energy = evaluate_qubo_energy_only(np.asarray(relaxed_initial_point))

    # energies.append((
    #     0,
    #     initial_energy,
    #     list(relaxed_initial_point),
    #     0.0
    # ))

    print(f"[spiq-init] logged iteration 0 energy = {initial_energy}")
    
    with open(energy_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "energy", "paramter", "std"])
        writer.writerows(energies)

    final_point = last_params["val"] if last_params["val"] is not None else list(relaxed_initial_point)
    used_eval = last_eval["val"]

    checkpoint_qi = QuantumInstance(backend=QasmSimulator(), shots=10240)
    samples = []
    best_bitstring = None
    best_fval = float("inf")

    if min_params is not None:
        try:
            qc = _bind(min_params)
            result = checkpoint_qi.execute(qc)
            counts = result.get_counts()
            shots = sum(counts.values()) if counts else 0
            for bitstring, cnt in counts.items():
                b = bitstring.replace(" ", "")[::-1]
                x = [int(ch) for ch in b]
                try:
                    e = float(qubo.objective.evaluate(x))
                except Exception:
                    e = float(qubo.objective.evaluate(list(x)))
                prob = cnt / shots if shots else 0.0
                samples.append({"x": x, "fval": e, "probability": prob, "count": cnt})
                if e < best_fval:
                    best_fval = e
                    best_bitstring = x
        except Exception as exc:
            print(f"[spiq-solver] sampling best-seen params failed: {exc}")

    min_state_buffer = None
    if min_params is not None:
        min_state_buffer = {
            "min_energy": float(min_energy_seen),
            "min_eval_idx": min_eval_idx,
            "min_params": min_params,
            "shots": sum(s["count"] for s in samples) if samples else 0,
            "samples": samples,
        }

    response_samples = [
        _make_spiq_sample(s["x"], s["fval"], s["probability"]) for s in samples
    ]
    response = _make_spiq_response(
        x=best_bitstring if best_bitstring is not None else [0] * len(ordered_params),
        fval=best_fval if samples else float(opt_result.fun),
        samples=response_samples,
    )

    print(
        f"[spiq-solver] done: n_params={n_params}, "
        f"used_eval={used_eval}, best_energy={min_energy_seen}, "
        f"opt_result.fun={opt_result.fun}"
    )

    return response, final_point, used_eval, min_state_buffer


def solve_with_QAOA(qubo, iterations, reps=TAG, use_local_simulator=False,result_dir="./9/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data",
                    card=None, pred=None, pred_sel=None, thres=None, inputNumber=0):
    if use_local_simulator:
        quantum_instance = get_local_QASM_backend()
    else:
        quantum_instance = get_IBMQ_backend()
    

    os.makedirs(result_dir, exist_ok=True)
    energy_log_path = os.path.join(result_dir, f"energy_per_iteration_{iterations}_{current_optim}_{TAG}_{TRIAL_ID}.csv")
    energies = []
    if current_optim == "AQGD":
        checkpoint_every = 910
    elif current_optim == "SPSA":
        checkpoint_every = 210
    else:  
        checkpoint_every = 0
    last_params = {"val": None}
    last_eval = {"val": 0}

    # checkpoint_shots = 128


    # checkpoint_backend = QasmSimulator(seed_simulator=12345)
    # checkpoint_qi = QuantumInstance(backend=checkpoint_backend, shots=checkpoint_shots)
    # op, _ = qubo.to_ising()
    # checkpoint_ansatz = QAOAAnsatz(op, reps).decompose()

    # Qiskit QAOA callback reports the Ising expectation value.
    # Samples/postprocessing in this file use qubo.objective.evaluate(x),
    # which is in the original QUBO objective scale.  Convert every
    # callback energy to QUBO scale by adding the Ising offset once here.
    # This makes energy_per_iteration_*.csv directly comparable with the
    # average energy computed from sampled bitstring distributions.
    _, ising_offset = qubo.to_ising()
    ising_offset = float(ising_offset)
    print(f"[energy-scale] vanilla QAOA callback energies will be logged as QUBO-scale: Ising mean + offset ({ising_offset})")

    min_order, alt_min_order, _ = Postprocessing.get_optimal_join_order(card, pred, pred_sel)
    # Track minimum QUBO-scale energy seen during optimization and the corresponding parameters
    min_energy_seen = float('inf')
    min_params = None
    min_eval_idx = None
    def _ensure_header(path, header):
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)
    


    def _run_checkpoint(eval_count, parameters):
        if (not checkpoint_every) or (int(eval_count) % checkpoint_every != 0):
            return

        if card is None or pred is None or pred_sel is None or thres is None:
            print(f"[checkpoint {int(eval_count)}] skip: missing card/pred/pred_sel/thres")
            return
        
        ckpt_tag = f"eval{int(eval_count)}_{iterations}_{current_optim}_reps{TAG}_trial{TRIAL_ID}_input{inputNumber}"
        all_path   = os.path.join(result_dir, f"checkpoint_ALL_{ckpt_tag}.csv")
        valid_path = os.path.join(result_dir, f"checkpoint_VALID_{ckpt_tag}.csv")
        opt_path   = os.path.join(result_dir, f"checkpoint_OPTIMAL_{ckpt_tag}.csv")

        header_all = ["eval_count","bitstring","count","prob","energy"]
        header_vo = ["eval_count","bitstring","count","prob","energy","cost","is_optimal","join_order_json"]

        _ensure_header(all_path, header_all)
        _ensure_header(valid_path, header_vo)
        _ensure_header(opt_path, header_vo)



        
        param_map = {p: v for p, v in zip(checkpoint_ansatz.parameters, parameters)}
        qc = checkpoint_ansatz.assign_parameters(param_map, inplace=False)
        qc = qc.copy()
        qc.measure_all()


        result = checkpoint_qi.execute(qc)
        counts = result.get_counts()
        if not counts:
            return

        total = sum(counts.values())
        sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

        valid_ratio = 0.0
        optimal_ratio = 0.0
        best_cost = None
        best_join_order = None

        with open(all_path, "a", newline="") as fa, \
            open(valid_path, "a", newline="") as fv, \
            open(opt_path, "a", newline="") as fo:

            wa, wv, wo = csv.writer(fa), csv.writer(fv), csv.writer(fo)

            for rank, (bitstring, cnt) in enumerate(sorted_items, start=1):
                b = bitstring.replace(" ", "")
                prob = cnt / total

                x = np.array([int(ch) for ch in b[::-1]], dtype=int)
                b= b[::-1]
                try:
                    energy = float(qubo.objective.evaluate(x))
                except Exception:
                    energy = float(qubo.objective.evaluate(list(x)))

                sample = {i: int(v) for i, v in enumerate(x)}
                join_order = Postprocessing.get_join_tree_leaves(sample, len(card))
                is_valid = join_order is not None

                cost = ""
                is_opt = 0
                join_order_json = ""

                if is_valid:
                    cost_val = Postprocessing.get_actual_costs_for_sample(join_order, card, pred, pred_sel, thres)
                    cost = float(cost_val)
                    valid_ratio += prob

                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_join_order = join_order

                    if join_order == min_order or join_order == alt_min_order:
                        is_opt = 1
                        optimal_ratio += prob

                    join_order_json = json.dumps(join_order)

                row_all = [int(eval_count), b, int(cnt), prob, energy]
                wa.writerow(row_all)

                if is_valid:
                    row_vo = [int(eval_count), b, int(cnt), prob, energy, cost, int(is_opt), join_order_json]

                    wv.writerow(row_vo)

                    if is_opt == 1:
                        wo.writerow(row_vo)

    def callback(eval_count, parameters, mean, metadata):

        
        # `mean` is the raw Ising expectation from Qiskit.
        # Add the offset so the logged callback energy is on the same
        # QUBO-objective scale as qubo.objective.evaluate(x) used for samples.
        energy_value_ising = float(np.real(mean))
        energy_value = energy_value_ising + ising_offset
        energies.append((eval_count, energy_value,parameters,metadata))
        last_params['val']=list(parameters)
        last_eval['val']=int(eval_count)
        # update min-energy buffer if this is the best seen so far
        nonlocal min_energy_seen, min_params, min_eval_idx
        try:
            if energy_value < min_energy_seen:
                min_energy_seen = energy_value
                min_params = list(parameters)
                min_eval_idx = int(eval_count)
        except Exception:
            pass
        # _run_checkpoint(eval_count, parameters)


    if(optmi==1):
        ## increase rhobeg from 1 to 3, decrease to to 1e-8
        optimizer=COBYLA(maxiter=iterations,rhobeg=2.0,tol=1e-12,disp=True)
        ### 1e-10 if stop early try 1e-12. 
        ### remember to print out op, _ = qubo.to_ising()
        
    elif(optmi==2):
        optimizer=SPSA(maxiter=iterations)
    else:
        optimizer = AQGD(maxiter=iterations,eta=0.01)
    initial_point=[0., 0.]
    if TAG==2:
            # uninitialized
            initial_point = [0.5, 0.5, 0.5, 0.5]
            # 3 table join, sel 0.1
            # initial_point=[1.0799224746714913, 1.0799224746714913, 1.281536766330277, 1.281536766330277]
            # 3 table join, sel 0.9
            # initial_point=[1.1780972450961724, 1.1780972450961724, 6.221667292016157, 6.221667292016157]
            # 4 table join 
            # initial_point = [1.2566370614359172, 1.2566370614359172, 5.600714276673461, 5.600714276673461]
    if TAG==3:
            # uninitialized
            # initial_point = [0.5, 0.5, 0.5, 0.5]
            # 3 table join, sel 0.1
            # initial_point=[1.0799224746714913, 1.0799224746714913, 1.281536766330277, 1.281536766330277]
            # 3 table join, sel 0.9
            initial_point=[1.1780972450961724, 1.1780972450961724, 6.221667292016157, 6.221667292016157]
            # 4 table join
            # initial_point = [1.2566370614359172, 1.2566370614359172, 5.600714276673461, 5.600714276673461]
    qaoa_meas = QAOA(optimizer=optimizer, quantum_instance=quantum_instance, reps=reps, initial_point=initial_point,callback=callback)
    qaoa = MinimumEigenOptimizer(qaoa_meas)
    qaoa_result = qaoa.solve(qubo)
    # Prepare a checkpoint ansatz and quantum instance for later sampling of the best-found parameters
    try:
        # hamiltonian - print 
        op, _ = qubo.to_ising()
        checkpoint_backend = QasmSimulator()
        checkpoint_qi = QuantumInstance(backend=checkpoint_backend, shots=10240)
        checkpoint_ansatz = QAOAAnsatz(op, reps)
    except Exception:
        checkpoint_ansatz = None
        checkpoint_qi = None

    with open(energy_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "energy","paramter","std"])
        writer.writerows(energies)
    
    final_point = last_params["val"] if last_params["val"] is not None else initial_point
    used_eval = last_eval["val"]

    # Build a small buffer describing the minimum-energy quantum state found (parameters + measured samples)
    min_state_buffer = None
    if min_params is not None and checkpoint_ansatz is not None and checkpoint_qi is not None:
        try:
            # assign parameters to the ansatz
            param_map = {p: v for p, v in zip(checkpoint_ansatz.parameters, min_params)}
            qc = checkpoint_ansatz.assign_parameters(param_map, inplace=False)
            qc = qc.copy()
            qc.measure_all()
            result = checkpoint_qi.execute(qc)
            counts = result.get_counts()
            shots = sum(counts.values()) if counts else 0

            samples = []
            for bitstring, cnt in counts.items():
                b = bitstring.replace(' ', '')[::-1]
                x = [int(ch) for ch in b]
                prob = cnt / shots if shots else 0.0
                try:
                    energy = float(qubo.objective.evaluate(x))
                except Exception:
                    energy = float(qubo.objective.evaluate(list(x)))
                samples.append({'x': x, 'fval': energy, 'probability': prob, 'count': cnt})

            min_state_buffer = {
                'min_energy': float(min_energy_seen),
                'min_eval_idx': min_eval_idx,
                'min_params': min_params,
                'shots': shots,
                'samples': samples
            }
        except Exception:
            min_state_buffer = None

    return qaoa_result, final_point, used_eval, min_state_buffer

def conduct_IBMQ_QPU_experiments():
    experiment_start = time.perf_counter()
    processing = config.configuration["ibmq-processing"]
    if processing == "qpu":
        token = config.configuration["ibmq-token"]
        IBMQ.save_account(token)
        IBMQ.load_account()
    
    iterations_categories = [10000]
    thres = {0:[150],1:[200],2:[300]}
    num_decimal_pos = 3
    optimal_solution = 0
    step=10
    
    for iterations in iterations_categories:
        for i in [INPUT_IDX]:
            init_point = None
            

            card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem('ExperimentalAnalysis/IBMQ/QPUPerformance/Problems/JSON/' + str(i) + '_predicates', generated_problems=False)

            # qubo, weight_a = QUBOGenerator.generate_QUBO_for_IBMQ(card, thres[i], num_decimal_pos, pred, pred_sel)
            print("Generating qubo",flush=True)

            # qubo, penalty_weight=QUBOGenerator1.generate_IBMQ_QUBO_for_left_deep_trees(card, pred, pred_sel, thres[i][0], num_decimal_pos)

            qubo, penalty_weight=QUBOGenerator1.generate_IBMQ_QUBO_for_left_deep_trees_v2(card, pred, pred_sel)


            # qubo = ProblemGenerator.get_join_ordering_qubo('ExperimentalAnalysis/IBMQ/QPUPerformance/Problems/QUBO/' + str(i) + '_predicates')
            check_qubit_from_qubo_and_exit(qubo, max_qubits=23)
            currentWeek="/workspace/Week88"
            
            response = None
            currentPath = f'{currentWeek}/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data'
            result_dir = os.path.join(
                        currentPath,
                        f"iterations_{iterations}",
                        f"reps_{TAG}",
                        f"{current_optim}",
                        f"input{i}",
                        f"trial{TRIAL_ID}"
                    )
            spiq_bundle = None
            if SPIQ_JSON:
                try:
                    spiq_bundle = _load_spiq_initialization(SPIQ_JSON)
                    print(
                        f"[spiq] loaded {SPIQ_JSON}: "
                        f"{len(spiq_bundle[1])} relaxed angles, "
                        f"pcirc qpy={spiq_bundle[2].get('pcirc_qpy')}"
                    )
                except Exception as exc:
                    print(f"[spiq] failed to load {SPIQ_JSON}: {exc}. Falling back to vanilla QAOA.")
                    spiq_bundle = None

            if processing == "qpu":
                        if spiq_bundle is not None:
                            pcirc_spiq, single_rip, spiq_meta = spiq_bundle
                            all_rips = spiq_meta.get("selected_relaxed_initial_points") or [single_rip]
                            print(f"[multistart] {len(all_rips)} initial points to try (qpu)")
                            selected_expected_energies = spiq_meta.get("selected_energy_best")

                            best = None
                            for k, rip in enumerate(all_rips):
                                print(f"[multistart] === run {k+1}/{len(all_rips)} ===")
                                if selected_expected_energies is not None and k < len(selected_expected_energies):
                                    expected_k = selected_expected_energies[k]
                                elif np.allclose(rip, single_rip):
                                    expected_k = spiq_meta.get("energy_best")
                                else:
                                    expected_k = None
                                resp_k, init_k, used_k, buf_k = solve_with_QAOA_spiq(
                                    qubo, iterations, pcirc_spiq, rip,
                                    use_local_simulator=False,
                                    expected_energy=expected_k,
                                )
                                fval = float(getattr(resp_k, "fval", float("inf")))
                                if best is None or fval < best["fval"]:
                                    best = {"resp": resp_k, "init": init_k, "used": used_k, "buf": buf_k, "fval": fval}
                                print(f"[multistart] run {k+1} fval = {fval}")

                            response, init_point, used_eval, min_state_buffer = (
                                best["resp"], best["init"], best["used"], best["buf"]
                            )
                            print(f"[multistart] best fval = {best['fval']}")
                        else:
                            response,init_point, used_eval, min_state_buffer = solve_with_QAOA(qubo, iterations, use_local_simulator=False)
                        result_path_prefix = 'ExperimentalAnalysis/IBMQ/QPUPerformance/Results/QPU_Data'
                        # If we obtained a buffered minimum-state, postprocess it for readout summary
                        if min_state_buffer is not None:
                            try:
                                # Build a minimal response-like object expected by postprocessing
                                class BufSample:
                                    def __init__(self, x, fval, probability):
                                        self.x = x
                                        self.fval = fval
                                        self.probability = probability

                                class BufResponse:
                                    def __init__(self, samples):
                                        self.samples = samples

                                buf_samples = []
                                for s in min_state_buffer['samples']:
                                    # samples store x as list LSB-first, convert to array-like
                                    buf_samples.append(BufSample(s['x'], s['fval'], s['probability']))

                                buf_resp = BufResponse(buf_samples)
                                # call postprocessing that handles qiskit readout-style results
                                Postprocessing.postprocess_qiskit_with_readout(
                                    buf_resp,
                                    card,
                                    pred,
                                    pred_sel,
                                    card_dict=None,
                                    scale=1000,
                                    opt_time_ms=0.0,
                                    trial_id=TRIAL_ID,
                                    tag=TAG,
                                    current_optim=current_optim,
                                    iterations=iterations,
                                    base_dir=currentPath,
                                )
                            except Exception:
                                pass
            else:
                        if spiq_bundle is not None:
                            pcirc_spiq, single_rip, spiq_meta = spiq_bundle
                            all_rips =spiq_meta.get("selected_relaxed_initial_points") or  [single_rip]
                            selected_expected_energies = spiq_meta.get("selected_energy_best")
                            # spiq_meta.get("selected_relaxed_initial_points") or 
                            print(f"[multistart] {len(all_rips)} initial points to try (cpu)")

                            best = None
                            for k, rip in enumerate(all_rips):
                                print(f"[multistart] === run {k+1}/{len(all_rips)} ===")
                                if selected_expected_energies is not None and k < len(selected_expected_energies):
                                    expected_k = selected_expected_energies[k]
                                elif np.allclose(rip, single_rip):
                                    expected_k = spiq_meta.get("energy_best")
                                else:
                                    expected_k = None
                                resp_k, init_k, used_k, buf_k = solve_with_QAOA_spiq(
                                    qubo, iterations, pcirc_spiq, rip,
                                    use_local_simulator=True, result_dir=result_dir, reps=TAG,
                                    card=card, pred=pred, pred_sel=pred_sel, thres=thres[i], inputNumber=i,
                                    expected_energy=expected_k,
                                )
                                fval = float(getattr(resp_k, "fval", float("inf")))
                                if best is None or fval < best["fval"]:
                                    best = {"resp": resp_k, "init": init_k, "used": used_k, "buf": buf_k, "fval": fval}
                                print(f"[multistart] run {k+1} fval = {fval}")

                            response, init_point, used_eval, min_state_buffer = (
                                best["resp"], best["init"], best["used"], best["buf"]
                            )
                            print(f"[multistart] best fval = {best['fval']}")
                        else:
                            response,init_point, used_eval, min_state_buffer = solve_with_QAOA(qubo, iterations, use_local_simulator=True,result_dir=result_dir,reps=TAG,
                                                                         card=card, pred=pred, pred_sel=pred_sel, thres=thres[i],inputNumber=i)
                        result_path_prefix = f'ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data'
                        # If min_state_buffer exists, write a CSV of bitstring/energy/prob for later analysis
                        if min_state_buffer is not None:
                            try:
                                out_dir = result_dir
                                os.makedirs(out_dir, exist_ok=True)
                                out_path = os.path.join(out_dir, 'min_state_readout.csv')
                                with open(out_path, 'w', newline='') as fout:
                                    w = csv.writer(fout)
                                    w.writerow(['bitstring', 'energy', 'prob'])
                                    for s in min_state_buffer['samples']:
                                        bitlist = s['x']
                                        bitstring = ''.join(str(int(b)) for b in bitlist)
                                        w.writerow([bitstring, s['fval'], s['probability']])
                            except Exception:
                                pass
        #     for cumulative_iters in range(step, iterations + 1, step):
        #         currentPath = f'{curentWeek}/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/checkpoint{cumulative_iters}'
        #         result_dir = os.path.join(
        #                 currentPath,
        #                 f"iterations_{cumulative_iters}",
        #                 f"reps_{TAG}",
        #                 f"{current_optim}",
        #                 f"input{i}"
        #             )
        #         if processing == "qpu":
        #                 response,init_point, used_eval = solve_with_QAOA(qubo, step, use_local_simulator=False)
        #                 result_path_prefix = 'ExperimentalAnalysis/IBMQ/QPUPerformance/Results/QPU_Data'
        #         else:
        #                 response,init_point, used_eval = solve_with_QAOA(qubo, step, use_local_simulator=True,result_dir=result_dir,reps=TAG,initial_point=init_point)
        #                 result_path_prefix = f'ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data'
    
        #         if not os.path.exists(checkpoint_csv):
        #          with open(checkpoint_csv, "w", newline="") as f:
        #             w = csv.writer(f)
        #             w.writerow(["cumulative_iters", "input_i", "best_join_order", "best_cost", "valid_ratio", "optimal_ratio"])

        #         pickle_results(
        #     result_path_prefix + f"/TOTAL_{iterations}/" + str(cumulative_iters) + "_Iterations/" + str(i) + "_predicates",
        #     response
        # )
        #         best_join_order, best_cost, valid_ratio, optimal_ratio = Postprocessing.postprocess_IBMQ_response(
        #     response, card, pred, pred_sel, thres,base_dir=currentPath,trial_id1=TRIAL_ID,tag1=TAG,current_optim1=current_optim,iterations1=cumulative_iters,inputNumber=i
        # )
        #         with open(checkpoint_csv, "a", newline="") as f:
        #             w = csv.writer(f)
        #             w.writerow([cumulative_iters, i, json.dumps(best_join_order), best_cost, get_rounded_val(valid_ratio), get_rounded_val(optimal_ratio),init_point])
        #         if optmi == 1 and used_eval < step:
        #             break
            pickle_results(result_path_prefix + '/' + str(iterations) + '_Iterations/' + str(i) + '_predicates-newQUBO/trial' + str(TRIAL_ID), response)

            experiment_runtime_sec = time.perf_counter() - experiment_start

            runtime_path = os.path.join(
                result_dir,
                f"runtime_conduct_IBMQ_QPU_experiments_{current_optim}_reps{TAG}_trial{TRIAL_ID}_input{i}.json",
            )

            with open(runtime_path, "w") as f:
                json.dump(
                    {
                        "runtime_sec": float(experiment_runtime_sec),
                        "runtime_min": float(experiment_runtime_sec / 60.0),
                        "trial": TRIAL_ID,
                        "reps": TAG,
                        "optimizer": current_optim,
                        "input": i,
                        "iterations": iterations,
                        "spiq_json": SPIQ_JSON,
                    },
                    f,
                    indent=2,
                )

            print(f"[runtime] conduct_IBMQ_QPU_experiments runtime sec = {experiment_runtime_sec:.3f}")
            print(f"[runtime] saved to {runtime_path}")

def conduct_IBMQ_transpilation_experiments(tket_optimizer, optimization_level, sample_size = 20):
    result_path_prefix = 'ExperimentalAnalysis/IBMQ/Embeddings/Results/'
    if tket_optimizer:
        circuit_transpilation = config.configuration["tket-circuit-transpilation"]
    else:
        circuit_transpilation = config.configuration["qiskit-circuit-transpilation"]
    
    if circuit_transpilation == "retranspile":
        result_path_prefix = 'ExperimentalAnalysis/IBMQ/Embeddings/Results/Retranspiled_Data'
    else:
        result_path_prefix = 'ExperimentalAnalysis/IBMQ/Embeddings/Results/Collected_Data'


    for i in range(3):
        for k in range(sample_size):
            ## Experiments for varying predicate numbers, for the IBMQ Auckland topology
            qubo = ProblemGenerator.get_join_ordering_qubo('ExperimentalAnalysis/IBMQ/Embeddings/Problems/QUBO/predicate_variation/' + str(i) + '_predicates')
            coupling_map = TopologyGenerator.get_IBM_Mumbai_topology()
            depth = determine_circuit_depth(qubo, coupling_map, tket_optimizer, optimization_level)
            path_string = result_path_prefix + '/predicate_variation/Auckland_topology/' + get_optimizer_string(tket_optimizer, optimization_level) + '/' + str(i) + '_predicates/sample_' + str(k)
            save_results(path_string, [depth])
            
            ## Experiments for varying cont. variable discretization precisions
            qubo = ProblemGenerator.get_join_ordering_qubo('ExperimentalAnalysis/IBMQ/Embeddings/Problems/QUBO/precision_variation/' + str(i) + '_decpos')
            coupling_map = TopologyGenerator.get_IBM_Mumbai_topology()
            depth = determine_circuit_depth(qubo, coupling_map, tket_optimizer, optimization_level)
            path_string = result_path_prefix + '/precision_variation/Auckland_topology/' + get_optimizer_string(tket_optimizer, optimization_level) + '/' + str(i) + '_decpos/sample_' + str(k)
            save_results(path_string, [depth])
            
            ## Experiments for varying predicate numbers, for the IBMQ Auckland topology
            qubo = ProblemGenerator.get_join_ordering_qubo('ExperimentalAnalysis/IBMQ/Embeddings/Problems/QUBO/predicate_variation/' + str(i) + '_predicates')
            coupling_map = TopologyGenerator.get_IBM_Washington_topology()
            depth = determine_circuit_depth(qubo, coupling_map, tket_optimizer, optimization_level)
            path_string = result_path_prefix + '/predicate_variation/Washington_topology/' + get_optimizer_string(tket_optimizer, optimization_level) + '/' + str(i) + '_predicates/sample_' + str(k)
            save_results(path_string, [depth])
            
            ## Experiments for varying cont. variable discretization precisions
            qubo = ProblemGenerator.get_join_ordering_qubo('ExperimentalAnalysis/IBMQ/Embeddings/Problems/QUBO/precision_variation/' + str(i) + '_decpos')
            coupling_map = TopologyGenerator.get_IBM_Washington_topology()
            depth = determine_circuit_depth(qubo, coupling_map, tket_optimizer, optimization_level)
            path_string = result_path_prefix + '/precision_variation/Washington_topology/' + get_optimizer_string(tket_optimizer, optimization_level) + '/' + str(i) + '_decpos/sample_' + str(k)
            save_results(path_string, [depth])


# In[4]:


def save_to_csv(data, path, filename):
    
    sd = os.path.abspath(path)
    pathlib.Path(sd).mkdir(parents=True, exist_ok=True) 
    
    f = open(path + '/' + filename, 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(data)
    f.close()

def process_data(depths):
    minimum_depth = np.amin(depths)
    mean_depth = int(np.ceil(np.mean(depths)))
    median_depth = int(np.ceil(np.median(depths)))
    maximum_depth = np.amax(depths)
    return minimum_depth, mean_depth, median_depth, maximum_depth

def parse_QPU_data(include_header=True,currentInput=0):
    if include_header:
        save_to_csv(['num_qaoa_iterations', 'num_predicates', 'valid_ratio', 'opt_ratio',f'Trial: {TRIAL_ID}',f'Reps:{TAG}',f'Optimizer:{current_optim}'], 'ExperimentalAnalysis/IBMQ/QPUPerformance/Results', 'results.txt')
 
    processing = config.configuration["ibmq-processing"]
    if processing == "qpu":
        result_path_prefix = 'ExperimentalAnalysis/IBMQ/QPUPerformance/Results/QPU_Data'
    elif processing == "cpu":
        result_path_prefix = 'ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/'
    else:
        result_path_prefix = 'ExperimentalAnalysis/IBMQ/QPUPerformance/Results/Collected_Data/'
        
    iterations_categories = [10000]
    thres_vals = {0:[150],1:[200],2:[300], 3: [10]}
    
    for iterations in iterations_categories:
        for i in [INPUT_IDX]:

            card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem('ExperimentalAnalysis/IBMQ/QPUPerformance/Problems/JSON/' + str(i) + '_predicates', generated_problems=False)
            response = load_pickled_result(result_path_prefix + '/' + str(iterations) + '_Iterations/' + str(i) + '_predicates')
            
            best_join_order, best_join_order_costs, valid_ratio, optimal_ratio = Postprocessing.postprocess_IBMQ_response(response, card, pred, pred_sel, thres_vals[i],trial_id1=TRIAL_ID,tag1=TAG,current_optim1=current_optim,iterations1=iterations,inputNumber=i)
            save_to_csv([iterations, i, get_rounded_val(valid_ratio), get_rounded_val(optimal_ratio)], 'ExperimentalAnalysis/IBMQ/QPUPerformance/Results', 'results.txt')


def check_qubit_from_qubo_and_exit(qubo, max_qubits=23):
    try:
        num_qubits = qubo.get_num_binary_vars()
    except Exception:
        num_qubits = len(qubo.variables)

    print(f"[INFO] qubit number = {num_qubits}",flush=True)

    if num_qubits > max_qubits:
        print(f"[STOP] qubit number {num_qubits} > {max_qubits}, terminate script.")
        sys.exit(0)

    return num_qubits

def parse_transpilation_data(optimizers, topologies, opt_levels, aggregate_results = False, include_header=True):
    if include_header:
        if aggregate_results:
            save_to_csv(['optimizer','opt_level','num_values','num_qubits','topology','strategy','min_depth','mean_depth','med_depth','max_depth'], 'ExperimentalAnalysis/IBMQ/Embeddings/Results', 'aggregated_depths.txt')
        else:
            save_to_csv(['optimizer', 'opt_level', 'value_index', 'num_qubits', 'topology', 'strategy', 'sample', 'depth'], 'ExperimentalAnalysis/IBMQ/Embeddings/Results', 'depths.txt')

    qiskit_circuit_transpilation = config.configuration["qiskit-circuit-transpilation"]        
    tket_circuit_transpilation = config.configuration["tket-circuit-transpilation"]
    if qiskit_circuit_transpilation == "retranspile":
        qiskit_result_path_prefix = 'ExperimentalAnalysis/IBMQ/Embeddings/Results/Retranspiled_Data'
    else:
        qiskit_result_path_prefix = 'ExperimentalAnalysis/IBMQ/Embeddings/Results/Collected_Data'
    if tket_circuit_transpilation == "retranspile":
        tket_result_path_prefix = 'ExperimentalAnalysis/IBMQ/Embeddings/Results/Retranspiled_Data'
    else:
        tket_result_path_prefix = 'ExperimentalAnalysis/IBMQ/Embeddings/Results/Collected_Data'
     
    qubits = [18, 21, 24, 27]
    for optimizer in optimizers:
        for i in [INPUT_IDX]:
            for opt_level in opt_levels:
                optimizer_string = None
                samplesize = 0
                if optimizer == "Tket" and opt_level != 1:
                    continue
                if optimizer == "Tket":
                    optimizer_string = get_optimizer_string(True, 1)
                    samplesize = 1
                    result_path_prefix = tket_result_path_prefix
                else:
                    optimizer_string = get_optimizer_string(False, opt_level)
                    samplesize = 20
                    result_path_prefix = qiskit_result_path_prefix
                for topology in topologies:
                    depths = []
                    for j in range(samplesize):
                        path_string = result_path_prefix + '/predicate_variation/' + topology + '_topology/' + optimizer_string + '/' + str(i) + '_predicates/sample_' + str(j)
                        depth = load_result(path_string)[0]
                        if aggregate_results:
                            depths.append(depth)
                        else:
                            save_to_csv([optimizer, opt_level, i, qubits[i], topology, 'predicates', j, depth], 'ExperimentalAnalysis/IBMQ/Embeddings/Results', 'depths.txt')

                    if aggregate_results:
                        minimum_depth, mean_depth, median_depth, maximum_depth = process_data(depths)
                        save_to_csv([optimizer, opt_level, i, qubits[i], topology, 'predicates', minimum_depth, mean_depth, median_depth, maximum_depth], 'ExperimentalAnalysis/IBMQ/Embeddings/Results', 'aggregated_depths.txt')

                    depths = []
                    for j in range(samplesize):
                        path_string = result_path_prefix + '/precision_variation/' + topology + '_topology/' + optimizer_string + '/' + str(i) + '_decpos/sample_' + str(j)
                        depth = load_result(path_string)[0]
                        if aggregate_results:
                            depths.append(depth)
                        else:
                            save_to_csv([optimizer, opt_level, i, qubits[i], topology, 'precision', j, depth], 'ExperimentalAnalysis/IBMQ/Embeddings/Results', 'depths.txt')

                    if aggregate_results:
                        minimum_depth, mean_depth, median_depth, maximum_depth = process_data(depths)
                        save_to_csv([optimizer, opt_level, i, qubits[i], topology, 'precision', minimum_depth, mean_depth, median_depth, maximum_depth], 'ExperimentalAnalysis/IBMQ/Embeddings/Results', 'aggregated_depths.txt')


# In[5]:

if __name__ == '__main__':

    qiskit_circuit_transpilation = config.configuration["qiskit-circuit-transpilation"]
    if qiskit_circuit_transpilation == "retranspile":
        conduct_IBMQ_transpilation_experiments(False, 1, sample_size = 20)
        conduct_IBMQ_transpilation_experiments(False, 2, sample_size = 20)
        conduct_IBMQ_transpilation_experiments(False, 3, sample_size = 20)
        
    tket_circuit_transpilation = config.configuration["tket-circuit-transpilation"]  
    if tket_circuit_transpilation == "retranspile":
        conduct_IBMQ_transpilation_experiments(True, 1, sample_size = 1)
    
    optimizers = ["Tket", "Qiskit"]
    topologies = ["Auckland", "Washington"]
    opt_levels = [1, 2, 3]
    ## parse_transpilation_data(optimizers, topologies, opt_levels, aggregate_results=False)

    processing = config.configuration["ibmq-processing"]
    if processing != "collected":
        print("COndu")

        conduct_IBMQ_QPU_experiments()
    
    # Postprocessing.write_bitstring_energy_prob(response, out_path)
    # parse_QPU_data()


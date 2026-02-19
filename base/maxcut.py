import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
from qiskit import Aer, execute
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.opflow import PauliSumOp
from qiskit.circuit.library import QAOAAnsatz
import networkx as nx
from qiskit.utils import QuantumInstance
from qiskit_optimization.applications import Maxcut
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import AQGD
def maxcut_ising_manual(edges, num_nodes=None):
    if num_nodes is None:
        max_idx = max(max(i, j) for i, j, _ in edges)
        num_nodes = max_idx + 1

    J = defaultdict(float)
    total_w = 0.0
    for i, j, w in edges:
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        J[(a, b)] += 0.5 * float(w) 
        total_w += float(w)

    offset = -0.5 * total_w

    terms = []
    for (i, j), coeff in J.items():
        label = ["I"] * num_nodes
        label[num_nodes - 1 - i] = "Z"
        label[num_nodes - 1 - j] = "Z"
        terms.append(("".join(label), float(coeff)))
    if not terms:
        terms = [("I" * num_nodes, 0.0)]
    return SparsePauliOp.from_list(terms), offset


def cut_value_from_bitstring(bitstring, edges):
    def b(i): 
        return int(bitstring[-1 - i])
    val = 0.0
    for i, j, w in edges:
        if b(i) != b(j):
            val += float(w)
    return val
objective_trace = []
def cost_func_statevector(params, ansatz, H_cost_sparse):
    qc = ansatz.bind_parameters(params)
    psi = Statevector.from_instruction(qc)
    e = float(np.real(psi.expectation_value(H_cost_sparse)))
    objective_trace.append(e)
    return e
def run_with_qaoa_ansatz(edges, num_nodes=5, reps=2, method="COBYLA", shots=10000):
    H_cost_sparse, offset = maxcut_ising_manual(edges, num_nodes=num_nodes)
    H_cost_opflow = PauliSumOp(H_cost_sparse)

    ansatz = QAOAAnsatz(cost_operator=H_cost_opflow, reps=reps)

    _ = ansatz.num_parameters

    x0 = 0.1 * np.ones(ansatz.num_parameters)

    objective_trace.clear()
    res = minimize(cost_func_statevector, x0, args=(ansatz, H_cost_sparse), method=method)

    final_E = float(res.fun)
    expected_cut = 2.5 - final_E 
    print(" Optimizer Result")
    print("Final <H_cost> (no offset):", final_E)
    print("Final energy + offset:", final_E + offset)
    backend = Aer.get_backend("qasm_simulator")
    qc = ansatz.bind_parameters(res.x)
    qc.measure_all()
    counts = execute(qc, backend=backend, shots=shots).result().get_counts()
    best_by_cut, best_cut = None, -1
    for bs in counts:
        cv = cut_value_from_bitstring(bs, edges)
        if cv > best_cut:
            best_cut, best_by_cut = cv, bs

    most_freq = max(counts, key=counts.get)
    print("\nSampling Result")
    print("Most frequent:", most_freq, "prob≈", counts[most_freq]/shots,
          "cut=", cut_value_from_bitstring(most_freq, edges))
    print("Best-by-cut:", best_by_cut, "cut=", best_cut)

    return res, counts
edges = [(0,1,1.0),(0,2,1.0),(0,3,1.0),(1,2,1.0),(2,3,1.0)]
run_with_qaoa_ansatz(edges, num_nodes=5, reps=2, method="COBYLA", shots=10000)

####AQGD

def cut_value_from_bitstring(bitstring, edges):
    def b(i):  
        return int(bitstring[-1 - i])
    val = 0.0
    for i, j, w in edges:
        if b(i) != b(j):
            val += float(w)
    return val


def run_maxcut_qaoa_aqgd(edges, num_nodes=5, reps=2, maxiter=300, eta=0.2, shots=10000, seed=7):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for i, j, w in edges:
        G.add_edge(i, j, weight=float(w))

    mc = Maxcut(G)
    qp = mc.to_quadratic_program()
    op, offset = qp.to_ising()  

    print("Offset:", float(offset))
    print("Ising operator type:", type(op))
    print(op)

    sv_backend = Aer.get_backend("statevector_simulator")
    qi = QuantumInstance(sv_backend, seed_simulator=seed, seed_transpiler=seed)

    optimizer = AQGD(maxiter=maxiter, eta=eta)

    qaoa = QAOA(optimizer=optimizer, reps=reps, quantum_instance=qi)
    result = qaoa.compute_minimum_eigenvalue(op)

    final_energy = float(np.real(result.eigenvalue))
    print("\n Optimizer Result (AQGD) ")
    print("Final <H_cost> (includes whatever constant is in op):", final_energy)

    qasm_backend = Aer.get_backend("qasm_simulator")
    ansatz = qaoa.ansatz
    qc = ansatz.bind_parameters(result.optimal_point)
    qc.measure_all()
    counts = execute(qc, backend=qasm_backend, shots=shots,
                     seed_simulator=seed, seed_transpiler=seed).result().get_counts()

    most_freq = max(counts, key=counts.get)
    best_by_cut = max(counts, key=lambda bs: cut_value_from_bitstring(bs, edges))

    print("\n Sampling Result =====")
    print("Most frequent:", most_freq, "prob≈", counts[most_freq]/shots,
          "cut=", cut_value_from_bitstring(most_freq, edges))
    print("Best-by-cut:", best_by_cut, "cut=", cut_value_from_bitstring(best_by_cut, edges))

    return result, counts




if __name__ == '__main__':
    edges = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    res=run_with_qaoa_ansatz(edges, num_nodes=5, reps=2, method="COBYLA", shots=10000)





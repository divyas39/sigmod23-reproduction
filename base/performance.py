
import os
import csv
import pathlib
import sys

import Scripts.ProblemGenerator as ProblemGenerator
import Scripts.QUBOGenerator1 as QUBOGenerator1
import Scripts.CircuitGeneration as CircuitGenerator
import Scripts.TopologyGenerator as TopologyGenerator


def get_QAOA_circuit(qubo):
    circuit = CircuitGenerator.create_QAOA_circuit(qubo)
    circuit = CircuitGenerator.decompose_circuit(circuit)
    return circuit


def count_two_qubit_gates(circuit):
    count = 0
    for instr, qargs, cargs in circuit.data:
        if len(qargs) == 2:
            count += 1
    return count


def save_resource_row(csv_path, row, header=None):
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header and header is not None:
            writer.writerow(header)
        writer.writerow(row)


def get_num_qubits_from_qubo(qubo):
    try:
        return qubo.get_num_binary_vars()
    except Exception:
        return len(qubo.variables)


def build_problem_by_size(size, predicate_mode):
    """
    这里需要你根据自己的项目替换。
    如果你已经有现成 JSON / problem files，就从文件读。
    如果你有生成器函数，就在这里调用。
    """
    # ===== 下面只是示例写法，你需要按你的项目改 =====
    card, pred, pred_sel = ProblemGenerator.generate_join_ordering_problem_by_size(
        size=size,
        predicate_mode=predicate_mode
    )
    return card, pred, pred_sel


def build_qubo(card, pred, pred_sel):
    qubo, penalty_weight = QUBOGenerator1.generate_IBMQ_QUBO_for_left_deep_trees_v2(
        card, pred, pred_sel
    )
    return qubo


def conduct_QAOA_resource_estimation():
    problem_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    predicate_settings = ["0_predicate", "credit", "n_predicate", "2n_predicate"]

    out_csv = "week51/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/resource_estimation/qaoa_resource_estimation.csv"
    header = [
        "problem_size",
        "predicate_setting",
        "num_qubits",
        "circuit_depth",
        "two_qubit_gate_count"
    ]

    for size in problem_sizes:
        for pred_mode in predicate_settings:
            print(f"[RESOURCE] size={size}, predicate_setting={pred_mode}", flush=True)

            try:
                card, pred, pred_sel = build_problem_by_size(size, pred_mode)
            except Exception as e:
                print(f"[SKIP] failed to build problem for size={size}, pred_mode={pred_mode}: {e}", flush=True)
                continue

            try:
                qubo = build_qubo(card, pred, pred_sel)
            except Exception as e:
                print(f"[SKIP] failed to build QUBO for size={size}, pred_mode={pred_mode}: {e}", flush=True)
                continue

            try:
                num_qubits = get_num_qubits_from_qubo(qubo)
                circuit = get_QAOA_circuit(qubo)
                circuit_depth = circuit.depth()
                two_qubit_gate_count = count_two_qubit_gates(circuit)
            except Exception as e:
                print(f"[SKIP] failed to analyze circuit for size={size}, pred_mode={pred_mode}: {e}", flush=True)
                continue

            save_resource_row(
                out_csv,
                [size, pred_mode, num_qubits, circuit_depth, two_qubit_gate_count],
                header=header
            )

            print(
                f"[DONE] size={size}, pred_mode={pred_mode}, "
                f"qubits={num_qubits}, depth={circuit_depth}, two_qubit={two_qubit_gate_count}",
                flush=True
            )


if __name__ == "__main__":
    conduct_QAOA_resource_estimation()
"""
CAFQA/SPIQ candidate-selection module.

Extracted and refactored from gradient_norm_select_maxcut.py so it does
NOT depend on a QAOASolver wrapper or on the MaxCut problem domain.
It works on any QAOA pcirc + SparsePauliOp Hamiltonian, including the
join-ordering QUBO flow used in spiq_initialization.py.

Public functions:
    compute_gradient_norm(pcirc, hamiltonian, params)
    deduplicate_parameters(params, fitness, tolerance)
    select_clustering_stratified_parameters(
        best_cafqa_parameters, best_cafqa_fitness_values,
        pcirc, hamiltonian,
        num_select=5, seed=None, verbose=True,
    )
"""

from __future__ import annotations

import math
import numpy as np
from sklearn.cluster import KMeans


def _as_sparse_pauli_op(op):
    """Accept either a PauliSumOp wrapper or a SparsePauliOp directly."""
    return op.primitive if hasattr(op, "primitive") else op


def compute_gradient_norm(pcirc, hamiltonian, params, verbose=False):
    """Use qiskit.quantum_info.Statevector — no qiskit_aer needed."""
    from qiskit.quantum_info import Statevector

    hamiltonian = _as_sparse_pauli_op(hamiltonian)
    params = np.asarray(params, dtype=float)
    shift = np.pi / 2.0

    def _expect(theta_vec):
        bound = pcirc.assign_parameters(
            {p: float(v) for p, v in zip(pcirc.parameters, theta_vec)},
            inplace=False,
        )
        sv = Statevector.from_instruction(bound)
        return float(np.real(sv.expectation_value(hamiltonian)))

    gradients = []
    for i in range(params.size):
        plus = params.copy()
        plus[i] += shift
        minus = params.copy()
        minus[i] -= shift
        gradients.append((_expect(plus) - _expect(minus)) / 2.0)

    grad_norm = float(np.linalg.norm(gradients))
    if verbose:
        print(f"  ||grad||_2 = {grad_norm:.6f}")
    return grad_norm


def deduplicate_parameters(parameters, fitness_values, tolerance=1e-10):
    """
    Remove numerically identical parameter rows.

    Args:
        parameters: array-like, shape (N, D).
        fitness_values: array-like, shape (N,).
        tolerance: atol passed to np.allclose.

    Returns:
        (unique_parameters, unique_fitness, original_indices)
    """
    parameters = np.asarray(parameters)
    fitness_values = np.asarray(fitness_values)

    unique_params, unique_fitness, original_indices = [], [], []
    for i, (p, f) in enumerate(zip(parameters, fitness_values)):
        if any(np.allclose(p, q, atol=tolerance) for q in unique_params):
            continue
        unique_params.append(p)
        unique_fitness.append(f)
        original_indices.append(i)

    return (
        np.array(unique_params),
        np.array(unique_fitness),
        original_indices,
    )


def select_clustering_stratified_parameters(
    best_cafqa_parameters,
    best_cafqa_fitness_values,
    pcirc,
    hamiltonian,
    num_select: int = 5,
    seed: int | None = None,
    verbose: bool = True,
):
    """
    Pick `num_select` diverse, non-barren CAFQA candidates from a SPIQ
    generation history.

    Pipeline:
        1. Deduplicate.
        2. K-means on (cos, sin) of CAFQA-space integers, with
           n_clusters = ceil(n_unique / 10), clipped to [1, n_unique].
        3. Within each cluster, sort by energy, stratify into 3 layers,
           sample 2 from the low layer and 1 from the mid layer.
        4. Compute gradient norms; drop candidates with grad approx 0
           (barren plateau).
        5. Final pick: 2 lowest-energy survivors + up to 3 random
           survivors (total = num_select).

    Args:
        best_cafqa_parameters: array-like, shape (N, D), CAFQA-space
            integers from the claptonize generation history.
        best_cafqa_fitness_values: array-like, shape (N,), Ising-scale
            energies aligned with `best_cafqa_parameters`.
        pcirc: relaxed pcirc, used only for gradient evaluation.
        hamiltonian: SparsePauliOp (or PauliSumOp; auto-unwrapped).
        num_select: number of final candidates (default 5).
        seed: RNG seed for reproducibility of sampling and K-means.
        verbose: progress logging.

    Returns:
        selected_ks (ndarray, shape (M, D), CAFQA-space ints),
        selected_fitness (ndarray, shape (M,)),
        selected_gradnorms (ndarray, shape (M,)).

    Notes:
        - `selected_ks` is returned in CAFQA-space (NOT radians) so the
          existing `_clifford_to_vanilla_initial_point` in
          spiq_initialization.py can consume it directly as `ks_best`.
        - Multiply by pi/2 before passing to gradient evaluation or to
          a Qiskit Estimator.
    """
    rng = np.random.default_rng(seed)

    parameters_array = np.asarray(best_cafqa_parameters)
    fitness_array = np.asarray(best_cafqa_fitness_values)

    if parameters_array.size == 0:
        raise ValueError("best_cafqa_parameters is empty.")
    if parameters_array.ndim == 1:
        parameters_array = parameters_array.reshape(1, -1)

    if verbose:
        print("\n" + "=" * 70)
        print("CLUSTERING-BASED STRATIFIED SELECTION WITH GRADIENT FILTERING")
        print("=" * 70)
        print(f"  Total points : {len(fitness_array)}")
        print(f"  Target       : {num_select} diverse points")

    # --- Step 1: deduplicate -------------------------------------------
    if verbose:
        print("\nStep 1: deduplication")
    unique_params, unique_fitness, _ = deduplicate_parameters(
        parameters_array, fitness_array
    )
    if verbose:
        print(f"  removed {len(parameters_array) - len(unique_params)} duplicates")
        print(f"  unique points: {len(unique_params)}")
    if len(unique_params) == 0:
        raise ValueError("No unique parameters to select from.")

    # --- Step 2: k-means on the unit circle ----------------------------
    n_clusters = max(
        1, min(math.ceil(len(unique_params) / 10), len(unique_params))
    )
    if verbose:
        print(f"\nStep 2: K-means, n_clusters = {n_clusters}")

    # Periodicity: CAFQA {0,1,2,3} -> (cos, sin) on the unit circle.
    X_periodic = np.column_stack(
        [
            np.cos(unique_params * np.pi / 2.0),
            np.sin(unique_params * np.pi / 2.0),
        ]
    )

    if n_clusters == 1:
        cluster_labels = np.zeros(len(unique_params), dtype=int)
    else:
        km_seed = 42 if seed is None else int(seed)
        kmeans = KMeans(n_clusters=n_clusters, random_state=km_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(X_periodic)

    # --- Step 3: stratified sampling within each cluster ---------------
    if verbose:
        print("\nStep 3: stratified sampling within each cluster")

    sampled = []
    for cid in range(n_clusters):
        mask = cluster_labels == cid
        c_params = unique_params[mask]
        c_fitness = unique_fitness[mask]
        if len(c_params) == 0:
            continue

        if verbose:
            print(
                f"  cluster {cid + 1}: {len(c_params)} pts, "
                f"E in [{c_fitness.min():.6f}, {c_fitness.max():.6f}]"
            )

        order = np.argsort(c_fitness)
        sp = c_params[order]
        sf = c_fitness[order]

        unique_energies = np.unique(sf)
        ne = len(unique_energies)

        if ne == 1:
            n_sample = min(3, len(sp))
            picks = rng.choice(len(sp), n_sample, replace=False)
            for i in picks:
                sampled.append(
                    {"params": sp[i], "energy": float(sf[i]), "cluster": cid}
                )
                if verbose:
                    print(f"    + flat cluster pick: E={sf[i]:.6f}")
            continue

        third = max(1, ne // 3)
        low_th = unique_energies[third - 1]
        mid_th = unique_energies[min(2 * third - 1, ne - 1)]

        low_idx = np.where(sf <= low_th)[0]
        mid_idx = np.where((sf > low_th) & (sf <= mid_th))[0]

        low_taken: set[int] = set()
        if len(low_idx) > 0:
            n = min(2, len(low_idx))
            picks = rng.choice(low_idx, n, replace=False)
            low_taken.update(int(p) for p in picks)
            for i in picks:
                sampled.append(
                    {"params": sp[i], "energy": float(sf[i]), "cluster": cid}
                )
                if verbose:
                    print(f"    + low : E={sf[i]:.6f}")

        if len(mid_idx) > 0:
            i = int(rng.choice(mid_idx, 1)[0])
            sampled.append(
                {"params": sp[i], "energy": float(sf[i]), "cluster": cid}
            )
            if verbose:
                print(f"    + mid : E={sf[i]:.6f}")
        elif len(low_idx) > len(low_taken):
            extra = [int(j) for j in low_idx if int(j) not in low_taken]
            if extra:
                i = int(rng.choice(extra, 1)[0])
                sampled.append(
                    {"params": sp[i], "energy": float(sf[i]), "cluster": cid}
                )
                if verbose:
                    print(f"    + extra low: E={sf[i]:.6f}")

    if verbose:
        print(f"\n  total candidates: {len(sampled)}")
    if not sampled:
        raise ValueError("No candidates produced by stratified sampling.")

    # --- Step 4: gradient-norm filter ----------------------------------
    if verbose:
        print("\nStep 4: gradient-norm filter (parameter shift, stabilizer)")
    hamiltonian = _as_sparse_pauli_op(hamiltonian)

    valid = []
    for k, cand in enumerate(sampled):
        angles = np.asarray(cand["params"]) * (np.pi / 2.0)
        try:
            gnorm = compute_gradient_norm(pcirc, hamiltonian, angles)
        except Exception as exc:
            if verbose:
                print(f"  cand {k+1}: gradient eval failed ({exc!r}), skipped")
            continue
        if gnorm > 1e-10:
            cand["grad_norm"] = float(gnorm)
            valid.append(cand)
            if verbose:
                print(
                    f"  cand {k+1} (cl {cand['cluster']+1}): "
                    f"E={cand['energy']:.6f}, grad={gnorm:.6f} keep"
                )
        elif verbose:
            print(
                f"  cand {k+1} (cl {cand['cluster']+1}): "
                f"E={cand['energy']:.6f}, grad approx 0, drop"
            )

    if not valid:
        raise ValueError("No candidates survived the gradient-norm filter.")

    # --- Step 5: final selection ---------------------------------------
    valid.sort(key=lambda c: c["energy"])
    n_low = min(2, len(valid))
    selected = list(valid[:n_low])

    rest = valid[n_low:]
    if rest:
        n_rand = min(num_select - n_low, len(rest))
        idx = rng.choice(len(rest), n_rand, replace=False)
        for i in idx:
            selected.append(rest[i])

    selected = selected[:num_select]

    if verbose:
        print("\n" + "=" * 70)
        print(f"FINAL SELECTION: {len(selected)} points")
        print("=" * 70)
        for k, p in enumerate(selected):
            print(
                f"  point {k+1}: cluster {p['cluster']+1}, "
                f"E={p['energy']:.6f}, grad={p['grad_norm']:.6f}"
            )

    sel_ks = np.array([p["params"] for p in selected])
    sel_fit = np.array([p["energy"] for p in selected])
    sel_grad = np.array([p["grad_norm"] for p in selected])
    return sel_ks, sel_fit, sel_grad
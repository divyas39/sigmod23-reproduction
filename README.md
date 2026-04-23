# Join Order Optimisation on Quantum Hardware — SPIQ-Initialised QAOA

This repository extends the SIGMOD 2023 reproduction package *"Ready to Leap
(by Co-Design)? Join Order Optimisation on Quantum Hardware"* with a
**SPIQ / CAFQA-based initiailization pipeline** for Join Ordering QAOA Simulation. The
`base/IBMQExperiments.py` driver has been refactored so it can either run
vanilla QAOA with an optimized QUBO encoding or QUBO encoding optimised over a relaxed,
per-gate-parameterised ansatz that is pre-seeded by a Clifford-space
search.

The rest of this document focuses on **our work** — the SPIQ
pre-initialisation, the relaxed-ansatz solver. The original paper's reproduction instructions are
preserved at the bottom (see [§5 Upstream Reproduction](#4-upstream-reproduction)).

---

## 1. High-level pipeline

```
          ┌──────────────────────────┐
          │ card.txt / pred.txt /    │     (join-order inputs, from
          │ pred_sel.txt             │      base/ExperimentalAnalysis/)
          └─────────────┬────────────┘
                        │ ProblemGenerator
                        ▼
          ┌──────────────────────────┐
          │ QUBOGenerator1           │     left-deep-tree QUBO
          │ .generate_IBMQ_QUBO_     │
          │  for_left_deep_trees_v2  │
          └─────────────┬────────────┘
                        │ qubo
          ┌─────────────┴───────────────────────────────┐
          │                                             │
          ▼                                             ▼
 ┌──────────────────────────┐               ┌────────────────────────┐
 │ base/spiq_initialization │               │ base/IBMQExperiments   │
 │ .py    (CAFQA / SPIQ)    │               │ .py   (QAOA driver)    │
 │                          │               │                        │
 │ 1. QAOAAnsatz(op, reps)  │               │ <if init with spiq>    |
 |                          |               |  --spiq_json <path>    │
 │ 2. relax to per-gate     │               │ loads:                 │
 │    parameters            │───── writes ─>│  • relaxed_initial_pt  |
 │ 3. qiskit → stim         │    JSON+QPY   │  • pcirc.qpy           │
 │ 4. claptonize() over     │               │  • energy_best (sanity)│
 │    Clifford K-vector     │               │                        │
 │ 5. K → vanilla θ and     │               │ solve_with_QAOA_spiq   │
 │    relaxed θ (per gate)  │               │ (COBYLA/SPSA/AQGD over │
 └──────────────────────────┘               │  per-gate θ, pcirc as  │
                                            │  ansatz)               | 
                                            | OR,                    |
                                            |  solve_with_QAOA       │
                                            └────────────────────────┘
```

Two stages, one QUBO:

1. **SPIQ pre-pass** (`base/spiq_initialization.py`) runs CAFQA-style
   Clifford-space search (`clapton.claptonize`) to pick a good warm-start
   for QAOA without burning any shots.
2. **QAOA refinement** (`base/IBMQExperiments.py`) takes SPIQ's output and
   continues the optimisation in the relaxed (per-gate) parameter space,
   with a sanity check that the qiskit-side pcirc reproduces the same
   Clifford state as stim.

---

## 2. SPIQ implementation (`base/spiq_initialization.py` + `clapton/`) 

### 2.1 What SPIQ does here

The Clifford-space search we use is the one from the **CAFQA/SPIQ** line
of work [3], implemented in the
vendored `clapton/` package.
Every parametric `Rz` in the QAOA ansatz is restricted to a Clifford
rotation `Rz(k·π/2)` with `k ∈ {0, 1, 2, 3}`. That makes the whole
circuit Clifford-simulable in polynomial time via **stim**, and a
classical GA (`claptonize`) searches the integer vector `K` for the one
that minimises `⟨H_C⟩` over the Pauli decomposition of the QUBO.

Because the optimum Clifford point is often close to a good QAOA angle,
using it as `initial_point` gives QAOA a far better starting iterate than
the zero / random defaults in the upstream paper's setup.

### 2.2 The tricky part: relaxation + angle recovery

The hard work in `spiq_initialization.py` is turning a stim-optimised
`ks_best` vector back into angles that qiskit's parametric circuits will
execute identically. Three subtleties had to be resolved.

1. **Relaxation bakes the coefficient into the parameter *name*.**
   `clapton.relax_qaoa_parameters` replaces each original QAOA parameter
   expression such as `-0.977 * γ[0]` with a *fresh* per-gate parameter
   literally named `"-0.977*gamma_0"`, applied as `Rz(1.0 * θ)`. So the
   relaxed pcirc multiplies every parameter by `1.0`, even though the
   string encoding still carries the original coefficient. We call this
   the **applied multiplier** (always 1.0) versus the **original
   multiplier** (the number encoded in the name).

2. **`str(expr).split("*")[0]` is fragile.** For some sympy / symengine
   configurations the coefficient isn't folded into a single token, so
   the upstream string-parse silently drops part of it. We bypass this
   by numerically evaluating the `ParameterExpression` at `param=1.0`
   inside `_extract_true_multipliers` — that's the authoritative
   multiplier used at runtime.

3. **Signs matter.** The unrelaxed QAOA cost layer applies
   `Rz(mult·γ)` *literally*, including the sign of `mult`. Converting a
   stim choice `k` back to a vanilla angle therefore requires
   **signed** division: `θ = k·π/2 / mult`. Using `abs(mult)` flips
   `S ↔ S†` on every gate whose coefficient is negative, producing a
   completely different Clifford state.

Putting it all together, `spiq_initialization.py` produces **two**
initial points:

* **Vanilla QAOA initial point** — `2 · reps` floats in
  `qaoa_ansatz.parameters` order. Per-gate θ's are bucketed by
  `(gamma|beta, rep)` and averaged, so every QAOA layer gets its own
  `γ, β`. Use this with a stock `QAOAAnsatz`.
* **Relaxed per-gate initial point** — one θ per gate in `pcirc`. For
  the relaxed binding, the applied multiplier is `1.0`, so
  `θ_i = k_i · π/2`. Use this with `pcirc` (the relaxed circuit that
  SPIQ itself searched).

### 2.3 Built-in verification

Before returning, the script binds `relaxed_initial_point` into
`pcirc`, computes `⟨H_C⟩` with qiskit's `Statevector`, and compares it
to stim's reported `energy_best`. Clifford states are noise-free, so
matching to machine precision is a strong signal that the qiskit and
stim sides are genuinely describing the same state. A warning and
per-gate dump are emitted if the relative error exceeds `1e-4`.

### 2.4 Outputs

For each run, `spiq_initialization.py` writes three artifacts under
`spiq_init_outputs/`:

| File | Purpose |
|------|---------|
| `spiq_initial_point_input{idx}_reps{r}_{N}rel_{M}pred.json` | All numeric outputs — `initial_point`, `relaxed_initial_point`, `relaxed_param_names`, `energy_best`, `ks_best_raw`, problem shape, penalty weight, path to the QPY file. |
| `spiq_pcirc_input{idx}_reps{r}_{N}rel_{M}pred.qpy` | QPY-serialised relaxed pcirc. Ensures the downstream QAOA run binds angles to the *exact same* circuit SPIQ scored. |
| `spiq_trace_input{idx}_reps{r}_{N}rel_{M}pred.txt` | Generational trace from `claptonize` (useful for debugging the GA). |

The shape suffix (`{N}rel_{M}pred`) prevents accidental clobbering when
several different problem instances share the same `(input_idx, reps)`.

### 2.5 Running the SPIQ pre-pass

```bash
cd base
python3 spiq_initialization.py \
    --input_idx 0 \
    --reps 2 \
    --n_gens 4000
```

The three flags we actually vary:

* `--input_idx` — which join-order problem under
  `base/ExperimentalAnalysis/IBMQ/QPUPerformance/Problems/JSON/` to load.
* `--reps` — QAOA depth `p`. Controls both how many vanilla angles
  the search maps back to and how many relaxed gates need Clifford
  values.
* `--n_gens` — GA budget for `claptonize`. We split it in half
  internally (`budget = n_gens // 2`) to match the CAFQA reference.

Other `argparse` flags (`--n_proc`, `--n_starts`, `--n_rounds`, `--err`,
`--out_dir`) exist but we leave them at their defaults; outputs go to
`spiq_init_outputs/`.

> The SPIQ side requires an additional set of packages that are **not**
> in the base `requirements.txt` (they're the commented block at the
> bottom). Uncomment them and reinstall, or run SPIQ in a separate venv.
> See [install.txt](./install.txt) for the concrete list.

---

## 3. QAOA driver (`base/IBMQExperiments.py`)

`IBMQExperiments.py` is the unchanged entry point from the paper, plus a
new branch that activates when you pass `--spiq_json`.

### 3.1 CLI surface

```bash
python3 base/IBMQExperiments.py \
    --trial     <int>    # trial id, used for result dir names
    --reps      <int>    # QAOA depth p (= 2 in most of our runs)
    --optimizer <0|1|2>  # 0=AQGD, 1=COBYLA, 2=SPSA
    [--spiq_json <path>] # activates SPIQ warm-start mode
```

Which backend / simulator is used is still controlled by `base/config.py`
(`ibmq-processing = "qpu"` | `"cpu"` | `"collected"`), exactly as in
the paper.

### 3.2 Two solvers, one driver

`conduct_IBMQ_QPU_experiments()` now branches on whether a SPIQ bundle
was loaded:

* **`solve_with_QAOA`** — the vanilla path. Builds a standard `QAOA` /
  `MinimumEigenOptimizer` with a `callback` that streams per-iteration
  `(eval, energy, parameters, metadata)` tuples to
  `energy_per_iteration_{iterations}_{optim}_{reps}_{trial}.csv`, and
  keeps a running buffer of the **minimum-energy** callback so the best
  parameters can be re-measured on a local `QasmSimulator` at the end
  for a bitstring-level readout summary.

* **`solve_with_QAOA_spiq`** — our relaxed-ansatz path. Instead of
  `QAOAAnsatz(op, reps)` with `2·reps` angles, we:

  1. Load `pcirc` from QPY (keeping the *exact* relaxed circuit SPIQ
     searched).
  2. Optionally run a **sanity check**: bind `relaxed_initial_point`
     into `pcirc`, measure 10 240 shots on a local `QasmSimulator`,
     evaluate `⟨H_C⟩_QUBO`, subtract the Ising offset returned by
     `qubo.to_ising()`, and compare to `energy_best`. 
  3. Minimise the diagonal cost directly with COBYLA (default)
     or SPSA / AQGD over the full per-gate parameter vector
     (one float per gate). The cost function binds the vector into
     `pcirc`, measures, and computes `Σ_x p(x) · qubo.objective.evaluate(x)`.
  4. Write the same `energy_per_iteration_…csv` schema the vanilla
     path emits, so downstream parsing (`Temp.convert_callback_csv_to_history`,
     plotting) is drop-in compatible. A 4th `std` column is always 0.0
     for this path since our cost is a deterministic scalar.
  5. Re-measure the best-seen parameters and emit `min_state_readout.csv`
     and a pickled response with the same `.x / .fval / .samples`
     shape that `MinimumEigenOptimizer.solve()` returns. We use
     `SimpleNamespace` for these (not a custom class) so the pickle
     survives being unpickled by `Temp.py` — a custom class triggered
     `Can't get attribute '_SpiqResponse' on <module '__main__'>`
     when the loader module differed from the writer.

The tuple shape returned by both solvers is deliberately identical:
`(response, final_point, used_eval, min_state_buffer)`. Everything
downstream — the pickle path, the readout CSV, the postprocessing
call — is unchanged.

### 3.3 Result layout

Both solvers write into a trial-scoped directory:

```
{week}/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/
    iterations_{N}/reps_{TAG}/{OPTIM}/input{i}/trial{TRIAL_ID}/
        energy_per_iteration_{N}_{OPTIM}_{TAG}_{TRIAL}.csv
        min_state_readout.csv
```

…plus the pickled response at

```
ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/
    {N}_Iterations/{i}_predicates-newQUBO/trial{TRIAL}/results.txt
```

`Postprocessing.postprocess_qiskit_with_readout` is called on the
QPU-processing branch to produce the per-sample energy/probability CSVs
plotted by `plot.rmd` / `visualization.rmd`.

### 3.4 End-to-end example

```bash
cd base

python3 spiq_initialization.py \
    --input_idx 0 --reps 2 --n_gens 4000 \
    --out_dir ../spiq_init_outputs

python3 IBMQExperiments.py \
    --trial 1 --reps 2 --optimizer 1 \
    --spiq_json ../spiq_init_outputs/spiq_initial_point_input0_reps2_4rel_2pred.json
```

Drop `--spiq_json` to fall back to the paper's vanilla QAOA. Nothing
else changes.

---

## 4. Upstream Reproduction

The sections below are from the original SIGMOD 2023 reproduction
package and are preserved verbatim.

### Build Docker Image

```
docker build -t sigmod-repro .
```

### Create Container

```
docker run --name sigmod-repro -it sigmod-repro [<option>]
```

The `<option>` specifies which operations are performed on container
start. Available options are:

* `ibmq_only` — Runs only IBM Q experiments
* `dwave_only` — Runs only D-Wave experiments
* `codesign_only` — Runs only DB-QPU co-design experiments
* `all` — Runs all of the above
* `bash` — Launches an interactive shell

For a full reproduction, we recommend running

```
docker run --name sigmod-repro -it sigmod-repro all
```

Finally, to move the generated plots onto the host system:

```
docker cp sigmod-repro:/home/repro/sigmod-repro/plots <destination_path>
```

### Project structure (upstream)

* `base/IBMQExperiments.py`, `base/DWaveExperiments.py` — experiment
  drivers for IBM-Q and D-Wave (we've extended the former).
* `base/TranspilationExperiment.py` — co-designed-QPU transpilation
  feasibility analysis.
* `base/Scripts/` — QUBO generation, circuit generation, topology
  builders, postprocessing.
* `base/couplings/` — qubit-topology files for contemporary QPUs.
* `base/ExperimentalAnalysis/`, `base/TranspilationExperiment/` —
  query data and experimental results.
* `scripts/` — top-level shell drivers invoked by the Docker entry
  point.
* `docs/Quantum_Foundations.pdf`, `docs/readme.pdf` — paper supplement
  and original reproduction walkthrough.

### Query data format

* `card.txt` — relation cardinalities, indexed by relation id.
* `pred.txt` — join predicates as `[[i, j], …]` pairs of relation
  indices.
* `pred_sel.txt` — predicate selectivities, order-aligned with
  `pred.txt`.
* `thres.txt` — threshold values for approximating intermediate
  cardinalities.

### References

[1] Michael Steinbrunn, Guido Moerkotte, and Alfons Kemper. 1997.
    *Heuristic and randomized optimization for the join ordering
    problem.* The VLDB journal 6 (1997), 191–208.

[2] Riccardo Mancini, Srinivas Karthik, Bikash Chandra, Vasilis
    Mageirakos, and Anastasia Ailamaki. 2022. *Efficient massively
    parallel join optimization for large queries.* SIGMOD '22, 122–135.

[3] Bharadwaj, D., Hou, Y., Li, G. Y., & Ravi, G. S. (2026). Scalable Clifford-Based Classical Initialization for the Quantum Approximate Optimization Algorithm. ArXiv. https://arxiv.org/abs/2602.14327

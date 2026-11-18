# Running Dr. Frank Zickert's *Hands-On Quantum Machine Learning With Python* on MW (`mwqc`)

Volumes 1 and 2, step by step, on a laptop CPU.

---

## 1. The books

Dr. Frank Zickert's **Hands-On Quantum Machine Learning With Python** (PyQML, [pyqml.com](https://www.pyqml.com))
teaches quantum computing the way a programmer learns: every idea is a short Python program you run and look at.
It assumes no physics background and builds up from a single qubit to working algorithms.

| | Volume 1: *Get Started* (2021) | Volume 2: *Combinatorial Optimization* (2023) |
|---|---|---|
| Code | [GitHub, Vol-1](https://github.com/quantum-machine-learning/Hands-On-Quantum-Machine-Learning-With-Python-Vol-1), 14 chapter notebooks | [GitHub, Vol2](https://github.com/quantum-machine-learning/Hands-On-Quantum-Machine-Learning-With-Python-Vol2), 13 notebooks |
| Topics | qubit states and measurement, parameterised circuits, variational hybrid algorithms, Naïve Bayes, entanglement, no-cloning, quantum Bayesian networks, qubit phase and phase kickback, amplitude amplification (Grover), quantum rejection sampling. Running example: Titanic survival. | observables, Hamiltonian decomposition into Pauli terms, expectation values, ansatz and optimiser (VQE by hand), Max-Cut as an Ising problem, QAOA by hand and with Qiskit, weighted Max-Cut, applications (graph partitioning, cliques, stable sets, travelling salesman, knapsack). Running example: a Marvel-hero social network. |
| Data | `data/train.csv`, `data/test.csv` (Titanic) | `data/hero-network.csv` (22 MB) |

The books' code repositories have no license file, so nothing from them is copied into this repository.
You clone them yourself (step 3 below). Buy the books for the text: the notebooks are the listings, and the
explanations are in the books.

**Why the code needs help today.** Both volumes are written against Qiskit before version 1.0
(`from qiskit import Aer, execute`, `execute(qc, backend).result().get_counts()`; Vol. 2 also uses
`qiskit.algorithms.QAOA/VQE` and `QuantumInstance`). Qiskit 1.0 (2024) removed all of these, so the
notebooks no longer run on current Qiskit. This repository gives two routes:

* **MW route:** the books' circuits run on `mwqc` with one import line changed. This covers all of
  Vol. 1 and the hand-built chapters of Vol. 2.
* **Legacy route:** the four Vol. 2 notebooks that call Qiskit's ready-made algorithm library run unchanged in a
  pinned Qiskit 0.41 environment (the version current when Vol. 2 was published).

---

## 2. Where MW fits: the Qiskit layer

```
   the book's code                    QuantumCircuit(...), qc.h(0), qc.cx(0,1), execute(qc, backend), get_counts()
        │
   ┌────▼──────────────────────┐      IBM:  qiskit (circuit API)   + qiskit-aer (C++ simulator)
   │ circuit API + simulator   │      MW:   mwqc.qiskit_compat     + mwqc statevector / MPS engines
   └────┬──────────────────────┘
        │  (optional, not used for any result in the books)
   quantum hardware via the cloud
```

**MW replaces the middle box.** It accepts the same circuit API and returns results in the same form. The
book's code above it does not change.

**Why that layer.** Every number printed in both books comes from **Aer, a classical simulator running on the
reader's computer**, not from a quantum chip. The job the books need done is therefore: take a circuit, simulate
it exactly, return counts, statevectors or unitaries. That is a classical computation, and it is the
computation `mwqc` does.

**How something this small does the same job.**

| | Qiskit 2.4 + Aer 0.17 (installed) | mwqc |
|---|---|---|
| size | ~177,000 lines of Python + ~26 MB of compiled Rust/C++ | **~2,500 lines of Python** (the book-facing layer, `qiskit_compat.py`, is ~700) |
| what the book layer imports | the full package | numpy only |

1. **The books use a narrow, well-defined slice of Qiskit.** Across both volumes, the chapters that run on
   mwqc use **25 circuit methods** (`h x y z i sdg rx ry rz cx cz cry crz rzz ccx mcx swap initialize measure
   measure_all barrier append to_gate decompose draw`), **3 backends** (`statevector_simulator`,
   `qasm_simulator`, `unitary_simulator`) and **3 result calls** (`get_counts`, `get_statevector`,
   `get_unitary`). The circuits use at most 12 qubits. Implementing that slice exactly is a small,
   checkable piece of code. Most of Qiskit's size serves things the books do not touch: transpiling to real
   chips, pulse control, noise, hardware providers and the algorithm libraries.
2. **MW's sparse-Hilbert-space engine.** A physically meaningful state of n qubits rarely needs all 2ⁿ
   amplitudes. `mwqc` stores it as a chain of small tensors, one Bloch-sphere-like site per qubit, linked by bonds
   whose size grows only with entanglement (a matrix product state). Book-sized circuits are run exactly. The
   same API then takes the book's style of program to 100-200 qubits when entanglement stays bounded
   (verified against exact solutions in `USE_CASES.md`), with the discarded weight reported as an error bar.
3. **Checked, not assumed.** The same source runs on both stacks and the outputs are compared call by call
   (`pyqml_port/port_book.py --validate`, §5). Book-style reference programs agree with Qiskit + Aer:

| technique (Vol. 1 chapter) | agreement, Qiskit + Aer vs mwqc |
|---|---|
| amplitude initialisation (2-3) | 1 − F = 0 |
| measured superposition, 4,000 shots (5) | TVD 0.015 (shot noise) |
| unitary simulator (6) | max \|ΔU\| 1.4e-16 |
| registers + Toffoli oracle with ancillas (6-8) | TVD 0.006 |
| quantum Bayesian network, sampled / exact (9-10) | TVD 0.004 / 1 − F ≈ 1e-15 |
| Grover amplification with `mcx` + `ZGate().control` (13) | 1 − F = 1.7e-14 |
| phase kickback (11) | 1 − F = 4e-16 |
| Vol. 2 gates `rzz` / `rxx` / `ryy`, `Parameter` binding, hand-built QAOA circuit | match Qiskit to 1e-12 (`tests/test_compat.py`) |

(1 − F: infidelity of the output state. TVD: total-variation distance between count histograms; 0.01-0.02 is
the shot noise of 4,000 samples.)

**Where Qiskit is the right tool instead:** real hardware access, noise models, transpilation, the ready-made
algorithm and optimisation libraries, C++ speed on large dense simulations, and a large community.
`mwqc` does not try to replace those.

---

## 3. One-time setup

```bash
cd ~/Quantum_Compute_MW
pip install -r requirements.txt                       # numpy, scipy, torch, pyscf, pytest
pip install matplotlib pandas scikit-learn networkx jupyterlab
python3 -m pytest -q tests/test_compat.py             # the compat layer vs Qiskit (skips the Qiskit checks if Qiskit is absent)
python3 pyqml_port/bookstyle_examples.py              # 3 s; needs qiskit + qiskit-aer for the comparison column
```

Optional, for comparing against IBM's stack (§5): `pip install "qiskit>=2.0" "qiskit-aer>=0.17"`.

Clone the books' code next to this repository's files (both folders are git-ignored here):

```bash
git clone https://github.com/quantum-machine-learning/Hands-On-Quantum-Machine-Learning-With-Python-Vol-1 book_vol1
git clone https://github.com/quantum-machine-learning/Hands-On-Quantum-Machine-Learning-With-Python-Vol2  book_vol2
```

---

## 4. Volume 1 on MW

```bash
python3 pyqml_port/port_book.py --book book_vol1 --out pyqml_port/ported_vol1
jupyter lab pyqml_port/ported_vol1
```

Open a chapter and run the cells top to bottom, reading along in the book. What the porter changed:

```python
from qiskit import QuantumCircuit, Aer, execute      ->  from mwqc.qiskit_compat import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram      ->  from mwqc.qiskit_compat import plot_histogram
from qiskit.circuit.library import ZGate             ->  from mwqc.qiskit_compat import ZGate
import qiskit                                        ->  import mwqc.qiskit_compat as qiskit
from qutip import Bloch                              ->  (skipped: QuTiP 3-D Bloch rendering)
```

A first cell adds this repository to `sys.path`, and `data/` is linked into the ported folder, so the notebooks
run from there.

| chapter | content | on mwqc |
|---|---|---|
| 1 | setting up Ubuntu | shell commands only; nothing to run |
| 2, 4 | Titanic data, classical baseline, Naïve Bayes | classical (pandas / scikit-learn) |
| 3 | quantum states, observer effect, parameterised circuit, variational hybrid classifier | runs |
| 5, 6, 8 | superposition maths, entanglement, no-cloning | runs |
| 7, 9, 10, 14 | quantum Naïve Bayes, quantum Bayesian networks, rejection sampling | runs (`QuantumRegister` / `ClassicalRegister`, `ccx`, `cry`) |
| 11, 12 | qubit phase, phase kickback, amplitude amplification | runs; `plot_bloch_multivector` / `plot_state_qsphere` draw a simpler matplotlib Bloch plot, QuTiP renderings skipped |
| 13 | amplitude amplification with an oracle, multiple results (`mcx`, `ZGate().control(k)`) | runs |

Run one or more chapters through the side-by-side check:

```bash
python3 pyqml_port/port_book.py --book book_vol1 --validate                     # all chapters
python3 pyqml_port/port_book.py --book book_vol1 --validate --chapters 3,9,13   # chapter numbers
```

---

## 5. Volume 2 on MW (plus the legacy route)

### 5.1 Hand-built chapters on MW

```bash
python3 pyqml_port/port_book.py --book book_vol2 --out pyqml_port/ported_vol2
jupyter lab pyqml_port/ported_vol2
```

The porter prints which notebooks contain cells for the legacy route. Those cells are left unchanged and start with
`# [mwqc port] needs Qiskit's pre-1.0 algorithm stack`.

| notebook | content | route |
|---|---|---|
| `v2_b_01_pandemic`, `v2_b_03_classical` | hero network, the problem solved classically | classical (pandas / networkx) |
| `v2_c_05_observable` | observables | **MW** |
| `v2_d_01_decomposition` | Hamiltonians as sums of Pauli terms (1, 2, n qubits) | **MW** |
| `v2_d_02_variational` | expectation value | **MW** |
| `v2_d_03_ansatz` | finding the ground state with an ansatz | **MW** (QuTiP rendering skipped) |
| `v2_d_05_optimizer` | the classical optimiser loop (VQE by hand) | **MW** |
| `v2_f_02_adjacency`, `v2_f_03_solving_ising` | Max-Cut as an Ising problem, solved | **MW** |
| `v2_h_01_qaoa`, up to *Using Qiskit* | QAOA by hand: `rzz` problem circuit, `rx` mixer, `Parameter`, COBYLA | **MW** |
| `v2_h_01_qaoa`, from *Using Qiskit* | Qiskit's `QAOA` on the hero network | legacy |
| `v2_f_07_qiskit` | Qiskit's `VQE`, `EfficientSU2`, `QuantumInstance` | legacy |
| `v2_h_03_weighted_maxcut` | Qiskit's `QAOA` + `MinimumEigenOptimizer` | legacy |
| `v2_m_01_applications` | `qiskit_optimization` applications, noise models, mitiq, IBMQ | legacy |

Symbolic angles work as in the book: `Parameter("gamma")` in `rx` / `ry` / `rz` / `rzz` / `rxx` / `ryy` and
controlled rotations, `draw()`, `decompose()`, and `qc.assign_parameters({gamma: 0.8})` before `execute`. An
unbound circuit raises a clear error.

Side-by-side check (`--chapters` takes any part of a file name):

```bash
python3 pyqml_port/port_book.py --book book_vol2 --validate
python3 pyqml_port/port_book.py --book book_vol2 --validate --chapters d_01,f_03,qaoa
```

Legacy cells are skipped on both stacks and counted (`[n legacy-env cells skipped]`). Cells after them that need
their results fail the same way on both stacks and are listed as errors, not as differences.

### 5.2 Legacy route: Qiskit's algorithm chapters, unchanged

A separate virtual environment with Qiskit as it was in January 2023. The pins in
`pyqml_port/requirements-legacy-qiskit.txt` (Qiskit 0.41 / Terra 0.23.1 / Aer 0.11.2, qiskit-optimization 0.5.0,
mitiq 0.23, pandas 1.4.4, networkx 2.8.8) were installed and checked: Qiskit's QAOA and VQE both find the optimal
max cut on a test graph.

```bash
python3.10 -m venv ~/venvs/pyqml_legacy               # Python 3.10 or 3.11
source ~/venvs/pyqml_legacy/bin/activate
pip install -r pyqml_port/requirements-legacy-qiskit.txt
python -m ipykernel install --user --name pyqml_legacy --display-name "PyQML legacy Qiskit 0.41"
jupyter lab book_vol2
deactivate
```

In Jupyter, open the **original** notebooks `v2_f_07_qiskit`, `v2_h_01_qaoa` (from *Using Qiskit*),
`v2_h_03_weighted_maxcut` and `v2_m_01_applications`, and choose the kernel *PyQML legacy Qiskit 0.41*.
In `v2_m_01`, cells that call `IBMQ.load_account()` need an IBM Quantum account, and the IBMQ provider of that
era connects to an IBM service that has since been retired. The simulator and noise-model cells run locally.

---

## 6. Your own book-style code

Change the import line and leave the rest:

```python
from mwqc.qiskit_compat import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute, plot_histogram

qc = QuantumCircuit(2)
qc.h(0); qc.cx(0, 1); qc.measure_all()
print(execute(qc, Aer.get_backend("qasm_simulator"), shots=1000).result().get_counts())
```

For circuits too large for a statevector, `Aer.get_backend("mps_simulator")` runs the same code on the MW
sparse-Hilbert-space engine.

**Not supported** (each raises a clear error): a gate on a qubit after it has been measured (mid-circuit
measurement), classical control (`c_if`), `reset`, noise models. None of these appears in the chapters marked MW above.

---

## 7. If something fails

| symptom | cause / fix |
|---|---|
| `ModuleNotFoundError: mwqc` in a ported notebook | run the first cell (it adds the repo to `sys.path`), or port again with `--out` |
| `FileNotFoundError: ./data/...` | open the notebook from the ported folder; the porter links `data/` there |
| `ImportError: cannot import name 'execute' from 'qiskit'` | the notebook was opened unported on Qiskit ≥ 1.0: use the ported copy (MW) or the legacy kernel |
| `NameError: quantum_instance` (or `qaoa`) in a ported Vol. 2 notebook | a legacy-route cell was skipped; run that part with the legacy kernel (§5.2) |
| `circuit has unbound parameters` | call `qc.assign_parameters({p: value})` before `execute` |
| counts differ slightly between runs or stacks | shot noise: TVD of 0.01-0.02 at 4,000 shots is expected; statevectors and unitaries agree to ~1e-14 |

Anything else: run `python3 pyqml_port/port_book.py --book <book> --validate --chapters <name>` and keep the
printed line. It shows which call differs and whether the error happens on both stacks or only on mwqc.

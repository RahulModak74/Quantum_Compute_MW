# Quantum_Compute_MW

**A CPU-first quantum-computing toolkit built on the Modak-Walawalkar (MW) "Bloch manifold"
idea, made rigorous, plus two applied use cases: Kohn-Sham (DFT) eigenvalues and EUV lithography.**

Package: `mwqc` (MW Quantum Compute) · Python ≥ 3.9 · numpy, scipy, torch · optional PySCF, Qiskit (only for cross-checks)
Status: research code, v0.1.0, 2026-11-18. 30 tests pass. Every headline number below is
reproduced by a script in this repo, and every fast or approximate path is checked against an exact one.

---

## 0. TL;DR

| Claim | Status | Evidence |
|---|---|---|
| 1000-qubit QFT runs on a laptop CPU, gate by gate (499,500 two-qubit gates) | **Done, verified** | `benchmarks/qft_1000.py`: basis input exact to 2.7e-15 rad in a **32 kB** state (~2 min); general product input needs bond dim **7**, **1.44 MB**, fidelity estimate 1.000000 (~3 min) |
| The tracked error bar is honest | **Verified** at n = 20 | tracked fidelity 0.9613 vs true 0.9601 (χ = 2); 0.99999367 vs 0.99999372 (χ = 4) |
| Qiskit-style API: circuits, Statevector, SparsePauliOp, Estimator/Sampler V2 shapes | **Done** (small slice, §5) | statevectors equal Qiskit's to 1e-10; spectra equal `SparsePauliOp`'s |
| Chemistry the IBM way (PySCF → Jordan-Wigner → UCCSD-VQE) | **Done, verified** | H₂, LiH (12 qubits), H₂O (14 qubits): VQE within chemical accuracy of FCI in **8/8** runs |
| Kohn-Sham eigenvalue surrogate on real PySCF data | **Done**, below chemical accuracy | **36.8 ± 4.4 meV** MAE from geometry alone (chem. acc. 43.6 meV); ~0.2 ms vs ~40 s per PySCF solve |
| Lithography aerial-image/CD surrogate with physics constraint | **Done** | CD error **0.17 nm**, ±2σ coverage 93 %, unseen-pattern detection AUROC 1.000 (§4.2) |
| Beyond-50-qubit physics: 100-200-spin quench dynamics and ground states | **Done, verified vs exact** | dynamics: every site within 1e-4 of exact (N=100); ground energy error 5.6e-7 (N=100, critical) (§4.3, USE_CASES.md) |
| Book code (Zickert, PyQML Vol. 1) runs on mwqc by changing one import line | **Done**; 8 book-style programs agree with Qiskit+Aer | exact outputs to ≤2e-14; sampled counts within shot noise (TVD ≤ 0.015) (§5.6) |
| scikit-learn API (`fit/predict`, Pipeline, GridSearchCV) | **Done** | surrogate regressor with `return_std`, quantum-kernel SVM, quantum feature map (§5.5) |
| "MW handles any 1000-qubit computation" / "10^26 compute reduction" | **Not true. Withdrawn** | §6. Cost grows with *entanglement*, not qubit count; random depth-20 circuits on 40 qubits already break χ = 64 |

---

## 1. The idea, restated so that it is correct

The original MW picture (LinkedIn article, 2 May 2026) put each qubit on its own Bloch sphere,
2 numbers per qubit, 2n numbers in total, and chained the spheres as a "fibre bundle"
(qubit k conditioned on qubits 0..k-1).

That picture has a precise name in physics: a **matrix product state (MPS)**.

| MW language | Standard language | In `mwqc` |
|---|---|---|
| n independent Bloch spheres (θ, φ per qubit) | product state = MPS with bond dimension **χ = 1** | `BlochState` |
| fibre bundle: qubit k conditioned on the qubits before it | MPS bond carrying χ-dimensional context left→right | `MPSState(chi_max=χ)` |
| "physically realizable states live on a small manifold" | states with low entanglement (area law) have small χ | `entanglement_entropy()`, `bond_dims()` |
| "Bayesian uncertainty, reducible with effort" | truncation error: discarded singular-value weight, reducible by raising χ | `truncation_error`, `fidelity_estimate`, Estimator `stds` |

So the honest form of the insight is:

> **Memory is linear in qubit count as long as entanglement stays bounded.** For product
> states (χ = 1) and low-entanglement states (small χ), 1000 qubits fit in kilobytes or megabytes on a
> laptop. When entanglement grows, χ must grow, up to 2^(n/2) in the worst case. No
> representation avoids that worst case.

This is well-established physics (Vidal 2003; DMRG, White 1992). IBM's own
`qiskit-aer` ships an MPS simulator. `mwqc` does not claim a new simulation method.
What it adds is a small, transparent, CPU-first stack that:
1. keeps the MW framing and the Bloch picture as a first-class, *diagnosable* backend (`BlochState`
   tells you exactly how much weight it lost when the state stopped being a product);
2. reports an error bound with every expectation value;
3. couples the simulator to MW-style **Bayesian physics surrogates** (`MWSurrogate`) whose stated
   uncertainty is checked by held-out coverage.

---

## 2. Repository layout

```
Quantum_Compute_MW/
├── mwqc/                      the reusable package
│   ├── gates.py               gate tensors (24 standard gates + arbitrary 1q/2q unitaries)
│   ├── circuit.py             MWCircuit (Qiskit-style builder) + to_qiskit()/from_qiskit()
│   ├── statevector.py         exact dense simulator (n ≤ 28), the reference for everything else
│   ├── mps.py                 MPSState (any n; error tracked) and BlochState (χ = 1, the MW picture)
│   ├── library.py             qft, qft_lnn (nearest-neighbour QFT), closed-form QFT|j>, ghz, HEA, random circuits
│   ├── pauli.py               PauliSum ≈ SparsePauliOp: Pauli algebra, sparse matrix, expectation (dense & MPS)
│   ├── primitives.py          MWEstimator / MWSampler (V2-style results: .data.evs, .data.stds, counts)
│   ├── chem.py                PySCF → MolecularProblem → Jordan-Wigner → exact sector diag → UCCSD-VQE (validation reference)
│   ├── surrogate.py           MWSurrogate: deep-ensemble Bayesian surrogate, physics penalties, calibration, OOD score
│   ├── dynamics.py            TEBD real/imaginary time, Ising/Heisenberg chains, exact free-fermion references
│   ├── qiskit_compat.py       legacy Qiskit API (QuantumCircuit/registers/Aer/execute) for books & tutorials
│   └── sklearn_api.py         MWSurrogateRegressor, QuantumKernelClassifier, QuantumFeatureMap
├── usecases/
│   ├── kohn_sham/
│   │   ├── ks_surrogate.py    KS eigenvalues of the Fe-O cluster from geometry (real PySCF data from v13)
│   │   ├── ks_vqe.py          KS/HF orbitals → qubits → VQE vs FCI (the "quantum" route to the same chemistry)
│   │   └── data/              PySCF caches copied from SOLVE_QUANTUM/Kohn_Sham_ScreenApproach
│   ├── lithography/
│   │   ├── litho_physics.py   corrected EUV forward model (Abbe partial coherence, defocus, acid blur, CD)
│   │   └── litho_surrogate.py CNN-ensemble surrogate, band-limit physics constraint vs none
│   └── many_body/             use cases 3-4: 100-200-qubit dynamics and ground states (see USE_CASES.md)
├── pyqml_port/                PyQML book port: converter + validation harness, Qiskit legacy shim, book-style checks
├── benchmarks/
│   ├── qft_1000.py            the 1000-qubit QFT, checked (Parts A-D)
│   └── limits.py              entanglement wall, GHZ(1000), side-by-side with qiskit-aer MPS
├── examples/quickstart.py     20-second tour
├── tests/                     30 tests: MPS = statevector = Qiskit, closed forms, JW = FCI, compat, sklearn
├── USE_CASES.md               the four use cases in one place
├── results/                   JSON + logs of every run quoted in this README
└── docs/legacy_audit.md       file-by-file audit of the earlier code
```

### Run it

```bash
cd ~/Quantum_Compute_MW
pip install -r requirements.txt            # numpy>=2, scipy, torch, pyscf, pytest
python3 -m pytest -q                       # 30 tests, ~20 s
python3 examples/quickstart.py             # 20 s
python3 benchmarks/qft_1000.py             # ~5 min on CPU (Part A ~2 min, Part B ~3 min)
python3 benchmarks/limits.py               # ~10 min; uses qiskit-aer if installed
python3 usecases/kohn_sham/ks_vqe.py       # ~3 min
python3 usecases/kohn_sham/ks_surrogate.py # ~10 min CPU (--quick: 1 min)
python3 usecases/lithography/litho_surrogate.py   # ~50 min (GPU helps; --quick: 2 min)
```

### Five lines of API

```python
from mwqc import MWCircuit, MWEstimator, PauliSum
from mwqc.library import ghz
qc = ghz(300)                                                     # 300 qubits
H  = PauliSum.single(300, 0, "Z") * PauliSum.single(300, 299, "Z")
r  = MWEstimator(method="mps").run([(qc, H)]).result()[0]
r.data.evs, r.data.stds, r.metadata["max_bond_used"]              # 1.0, 0.0 (error bound), 2
```

---

## 3. What is done: core package (`mwqc`)

### 3.1 Backends

| Backend | Exact? | Scale | Use |
|---|---|---|---|
| `StatevectorState` | exact | n ≤ 28 (16·2^n bytes) | reference; chemistry VQE |
| `MPSState(chi_max)` | exact while no truncation; otherwise error tracked | any n; cost ~ n·χ³ per gate | 1000-qubit low-entanglement circuits |
| `BlochState` (χ = 1) | exact for product-state circuits | any n; 32 bytes/qubit | the MW picture; QFT on basis states; diagnostic |

Every backend is cross-checked in `tests/test_core.py`: MPS vs statevector on random circuits with
long-range gates (fidelity > 1-1e-10), statevector vs `qiskit.quantum_info.Statevector`, QFT vs
`numpy.fft`, and `from_qiskit` round trips of Qiskit's own `QFTGate`.

### 3.2 The 1000-qubit QFT, `benchmarks/qft_1000.py`

`qft_lnn(n)` writes the QFT as n(n-1)/2 merged CP·SWAP gates between *adjacent* qubits (Fowler,
Devitt & Hollenberg 2004). The SWAP network also performs the final bit reversal. This is what
makes a gate-by-gate MPS run at n = 1000 practical.

| Part | Input | Result (n = 1000, 500,500 gates) |
|---|---|---|
| A | basis state \|j⟩, j a random **1000-bit** integer | χ = 1 throughout; truncation error 0; max angle error vs closed form **2.7e-15 rad**; state **32 kB**; 123 s |
| B | random point on every Bloch sphere (output is **entangled**) | max bond **7**, fidelity estimate **1.000000**, state **1.44 MB**, entropies 0.06-0.53 bits per cut; 177 s |
| C | validation of B's method at n = 20 vs exact statevector | χ=1: true F 0.230 (est. 0.238); χ=2: 0.9600 (0.9613); χ=4: 0.9999937 (0.9999937); χ≥8: 1.0 |
| D | the v6 training set (j < 2^53) | **93.6 %** of target phases are within 1 mrad of 0 for every sample |

Reading: (A) confirms the closed form the v6 script trained on, without any training.
(B) is the result worth quoting: a general 1000-qubit QFT is cheap because **the QFT creates
little entanglement**, not because qubits are independent. (C) shows that the χ = 1 Bloch picture
alone would be badly wrong here (23 % fidelity), and that the tracked error estimate is reliable.

> Single-threaded Python on an i7 laptop (LOQ 15IRH8); peak process memory 557 MB, almost all of it the
> stored circuit (500,500 gate matrices), not the quantum state. `results/qft_1000.json`.

### 3.3 Limits, measured (`benchmarks/limits.py`)

Random brick-wall circuits, 40 qubits, χ ≤ 64 (shows where the approach stops working):

| depth | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|
| max bond | 2 | 4 | 8 | 16 | 32 | 64 | 64 | 64 |
| fidelity estimate | 1 | 1 | 1 | 1 | 1 | 1.000 | **0.869** | **0.384** |
| mid-chain entropy (bits) | 0.36 | 1.06 | 2.13 | 2.94 | 3.44 | 3.66 | 5.19 | 5.20 |

Entanglement grows linearly with depth, χ grows exponentially, and at fixed χ the fidelity
collapses. Wall time follows χ: 0.2 s at depth 10 → 23-26 s at depth 16-20. GHZ(1000), by contrast,
needs χ = 2 (0.13 MB, 0.8 s for 1000 shots) despite being "maximally global".

Side by side with IBM's own MPS simulator, on the same circuit (random product state + QFT):

| n | `mwqc` MPS (Python) | `qiskit-aer` MPS (C++) | Qiskit dense `Statevector` | bond dim |
|---|---|---|---|---|
| 20 | 0.06 s | 0.02 s | 8.1 s | 7 |
| 50 | 0.60 s | 0.13 s | — (needs 18 PB) | 7 |
| 100 | 1.7 s | 0.48 s | — | 7 |
| 200 | 7.5 s | 2.7 s | — | 7 |

`mwqc` agrees with Qiskit's statevector to fidelity 1.0 at n = 20. **Aer is 2.8-4.6× faster**: same
algorithm class, compiled code. The low-entanglement 1000-qubit capability is therefore *not* unique
to MW. Qiskit users already have it. What `mwqc` adds is the error bound on every result,
the χ = 1 Bloch diagnostic, and the coupling to MW surrogates.

### 3.4 Chemistry layer (`mwqc.chem`)

`MolecularProblem.from_pyscf` → MO integrals (HF or **Kohn-Sham** orbitals) → `jordan_wigner`
(explicit Pauli algebra, Qiskit-Nature's blocked spin ordering) → `exact_ground_energy`
(sparse diagonalisation in the correct N_α, N_β sector) → `vqe_uccsd` (Trotterised UCCSD, exact
adjoint gradient, BFGS).

### 3.5 `MWSurrogate`

Deep ensemble (default 5 members), heteroscedastic Gaussian likelihood, optional physics penalty
(soft prior) or physics layer inside the network (hard constraint), `calibrate()` on a validation split,
`coverage()` on held-out data, `ood_score()` = epistemic σ for routing a sample back to the real solver.
This replaces the single Pyro VAE of v12/v13. Earlier versions *stated* uncertainty; this one *checks* it.

---

## 4. Use cases

### 4.1 Kohn-Sham (DFT)

**(a) `ks_surrogate.py`: KS eigenvalue surrogate on real PySCF data.**
System: the v13 Fe-O cluster (Fe O₄, charge +2, spin 4; a *proxy* motivated by LiFePO₄), 6 valence
KS eigenvalues, 200 geometries at PBE/STO-3G that v13 had already computed (~40 s each, ≈ 2.2 CPU-hours).
v13 never saved the geometries, but generated them from a seeded RNG. They are replayed here, and
re-running PySCF on replayed geometries reproduces the cached eigenvalues to 3×10⁻⁸ Ha.

*E1: the v13 inputs leak the answer.*

| inputs (linear fit, 5-fold CV) | MAE (meV) |
|---|---|
| the 7 v13 features computed from the eigenvalues themselves | **24.9** |
| the other non-constant v13 features (all functions of `li_occupancy`, which never enters PySCF) | 213.0 |
| predict-the-mean | ~210 |

So v13's reported accuracy measured how well the network inverted its own feature formulas.

*E2/E3: the honest problem, geometry (10 interatomic distances) → 6 eigenvalues, 5-fold CV:*

| model | MAE (meV) |
|---|---|
| predict the mean | 210.4 ± 7.7 |
| ridge regression on distances | 86.0 ± 6.1 |
| **MWSurrogate** (5-member ensemble) | **36.8 ± 4.4** |
| MWSurrogate + aufbau-ordering prior (v13 prior 3) | 36.8 ± 4.3 |

* **Below chemical accuracy (43.6 meV)** against this PySCF reference, from geometry alone.
  Per-orbital: 46 / 45 / 32 / 32 / 48 / 19 meV, so the two lowest and the 5th orbitals sit at the edge.
* ±2σ coverage on held-out folds: 66 % raw → **93 % after calibration** (target 95 %).
* Inference ≈ 0.2 ms vs ≈ 40 s per PySCF solve (~10⁵×). This is *amortised*: the 2.2 CPU-hours of
  training data must be paid first, and the surrogate is valid only near the sampled geometries (±0.08 Å).
* The aufbau prior changed nothing measurable: the data already respect it.
* The 25-geometry PBE0/def2-SVP cache is too small for a network: leave-one-out ridge gives 143 meV
  vs 202 meV for the mean (and the regularisation was chosen post hoc, so even 143 is optimistic).

**(b) `ks_vqe.py`: the quantum-computer route, done the IBM way and checked.** STO-3G:

| system | qubits | Pauli terms | UCCSD params | E_FCI (Ha) | VQE − FCI (mHa) | HF − FCI (mHa) |
|---|---|---|---|---|---|---|
| H₂ 0.735 Å | 4 | 15 | 3 | −1.137306 | 0.000 | 20.3 |
| H₂ 1.50 Å (stretched) | 4 | 15 | 3 | −0.998149 | 0.000 | 87.3 |
| LiH 1.595 Å | 12 | 631 | 92 | −7.882402 | 0.010 | 20.4 |
| H₂O | 14 | 1086 | 140 | −75.012578 | 0.097 | 49.6 |

Both orbital choices (KS-PBE and HF) were run: same FCI, VQE within chemical accuracy in **8/8** runs,
qubit-mapping error ≤ 1.5×10⁻⁹ mHa. Note that E_KS(PBE) is a different method (a density functional),
so it is listed in `results/ks_vqe.json` but is not comparable with FCI.

What this means for the MW story: KS-DFT is mean-field and needs no qubits at all. Qubits
(real or emulated) enter only for correlation *beyond* DFT, and the cost is visible in the table:
qubits = 2 × orbitals, Pauli terms ∝ orbitals⁴. A periodic LiFePO₄ cell is far outside dense emulation;
the realistic path is the surrogate in (a), trained on periodic DFT (VASP/GPAW/QE) instead of a cluster.

### 4.2 EUV lithography

`litho_physics.py`: λ = 13.5 nm, NA = 0.33, annular σ 0.3-0.7 (52 source points), 64×64 px at 4 nm,
±30 nm defocus, 3 nm acid blur, dose-referenced threshold, sub-pixel CD on the centre row. Four
training families (line/space, contacts, isolated line ± SRAF, line ends) and two unseen out-of-distribution
families (diagonal gratings, random pixels). The four bugs of the earlier demo (pixel pitch in
`fftfreq`, per-image max normalisation, coherent-only illumination, unmeasured speed-up) are fixed.

**The MW physics prior as a hard constraint:** the spectrum of any aerial image through a pupil of radius
NA/λ is confined to |f| ≤ 2NA/λ. The `band` variant projects every prediction onto that set (an FFT
mask inside the network), so it *cannot* output a physically impossible image. The `plain` variant can.

Training set 2,000 masks (500 per family), test 500, OOD 250, 4-member CNN ensembles, MSE loss:

| metric (test set) | `plain` CNN | `band` CNN (physics layer) |
|---|---|---|
| aerial-image RMSE (in-distribution, intensity units; clear field = 1) | 0.0045 | **0.0039** |
| out-of-band spectral power (physically impossible part) | 5.2×10⁻⁵ | **2.3×10⁻¹⁵** (zero by construction) |
| CD MAE (nm) | 0.18 | **0.17** |
| CD MAE by family: line/space · contacts · iso+SRAF · line-end (nm) | 0.14 · 0.46 · 0.10 · 0.23 | 0.17 · 0.33 · 0.08 · 0.24 |
| print / no-print mismatches | 1 / 500 | 1 / 500 |
| ±2σ CD coverage, raw → calibrated (target 95 %) | 50 % → 94 % | 42 % → 93 % |
| OOD detection AUROC (diagonal gratings · random pixels) | 1.000 · 1.000 | 1.000 · 1.000 |
| image RMSE on OOD masks | 0.0196 | 0.0169 |
| time per mask: forward model vs surrogate | 20.7 ms vs 2.2 ms (9×) | same |

Reading:
* CD from the surrogate is within **~0.2 nm** of the optics model at 4 nm pixels (sub-pixel), with
  calibrated uncertainty and perfect flagging of unseen pattern types.
* The hard physics layer costs nothing and helps slightly. More importantly, it guarantees the output is a
  physically realisable image. That is the MW "stay on the physical manifold" idea, enforced exactly.
* The speed-up is only **9×** because the forward model here is a cheap scalar FFT model (21 ms). Against
  rigorous EUV solvers (RCWA/FDTD, seconds to minutes per clip) the same surrogate would be orders of magnitude
  faster, but that has not been measured here and is not claimed.
* **A failed first attempt is kept on record:** training with a heteroscedastic Gaussian likelihood collapsed
  the ensemble (image RMSE 0.17, CD σ ≈ 0, coverage 0 %). For a deterministic solver there is no noise to
  learn. `MWSurrogate(likelihood="mse")` fixes it (`results/litho_surrogate_gaussian_nll_FAILED.*`).

### 4.3 Beyond 50 qubits: quantum magnets (use cases 3 and 4)

100-200-qubit **quench dynamics** and **ground states** of interacting spin chains on the MPS engine
(`mwqc.dynamics`: TEBD in real and imaginary time), checked against the exact free-fermion solution at any N
and against dense diagonalisation at N = 12. Verified so far: all 100 sites × all times within 1e-4 of exact
(dynamics, N = 100); critical-Ising ground energy at N = 100 within 5.6e-7 (relative 4e-9).
Full details, limits and run commands: **[USE_CASES.md](USE_CASES.md)**.

---

## 5. Mapping to IBM's quantum stack (Qiskit): what applies to MW, and coverage

### 5.1 The MW position, and why some of Qiskit does not apply

MW does not encode chemistry onto qubits and does not target quantum hardware. Its position
(see `QUESTION_FEYNMAN.md` in MW_Geometric_Intelligence) is that physically realisable systems
live in a **sparse corner of Hilbert space**, which classical computation can reach directly.
In `mwqc` that statement is made *measurable*: "sparse" = small MPS bond dimension χ / low
entanglement entropy, reported with every run (§3.2, §3.3). Where χ stays small, MW computes the
state on a CPU. Where it does not (random deep circuits, Shor-type circuits), MW says so.

So the Qiskit stack splits in two:

**Not applicable by design (excluded from the score):** IBM Runtime and hardware access, the transpiler
(hardware layout, routing, basis translation), error mitigation and suppression, noise models and pulse
control (these only emulate imperfect hardware), and **qubit-encoded chemistry (Qiskit Nature) as a
production route**. In MW, chemistry is done by Bayesian surrogates on DFT data (§4.1a). The Jordan-Wigner/VQE
module (`mwqc.chem`) is kept only as a *validation reference*: it proves the toolkit reproduces the
qubit route exactly (8/8 within chemical accuracy) without depending on it.

### 5.2 Coverage of the applicable part

"Covered" means implemented **and** tested against Qiskit or an exact answer. The percentages are
judgement calls on feature surface, stated so they can be argued with.

| Applicable Qiskit component | What `mwqc` has (tested) | Main gaps | Coverage |
|---|---|---|---|
| Circuit building (`QuantumCircuit`, registers, gates) | 24 named gates + k-qubit unitaries, registers, `ccx/mcx/cry/ch`, `append` of sub-circuits and controlled gates, `initialize`, end-of-circuit `measure`, Qiskit round-trip | `Parameter` objects, control flow, mid-circuit measure/reset, QASM | ~60 % |
| Legacy API used by books/tutorials (`execute`, `Aer.get_backend`, `get_counts`, ...) | full set used by Zickert's PyQML Vol. 1; validated program-by-program against Qiskit + Aer (§5.4) | `c_if`, reset | ~85 % |
| Primitives V2 (Estimator, Sampler) | `run([(circuit, observables)])` → `.data.evs/.stds`, counts | parameter broadcasting, precision targets | ~60 % |
| Pure-state simulators (Aer statevector / MPS / unitary) | statevector, **MPS** (same algorithm class as Aer's, with an error bound), unitary, χ=1 Bloch | GPU, C++ speed (Aer MPS is 2.8-4.6× faster, §3.3) | ~70 % |
| `quantum_info` (relevant part) | Statevector, SparsePauliOp-like `PauliSum`, reduced states, bipartite entropy, overlaps, 2-site expectations | DensityMatrix, Operator algebra, Clifford | ~35 % |
| `circuit.library` (relevant part) | QFT (+ LNN form), GHZ, ZZ feature map, RY-CX n-local, Trotter steps for Ising/Heisenberg chains | most n-local variants, arithmetic, evolution gates for arbitrary Hamiltonians | ~30 % |
| qiskit-machine-learning | `QuantumKernelClassifier` (≈ QSVC), `QuantumFeatureMap`, scikit-learn API (§5.5) | VQC/QNN, TorchConnector | ~40 % |
| Algorithms (time evolution, ground states) | TEBD real- and imaginary-time evolution on MPS (§4.3) | QPE, amplitude estimation, VarQTE | ~30 % |

**Overall: ≈ 50 % of the Qiskit surface that applies to MW** (unweighted mean of the eight rows),
and ≈ 85 % of what an existing textbook QML codebase actually calls. Across *all* of Qiskit, including
the excluded hardware parts, it would be ≈ 20 %. That figure is not meaningful for MW and is given only
for completeness. The earlier `mw_quantum` package mirrored class *names* but performed none of
the operations: ~0 % functional (docs/legacy_audit.md §1).

### 5.3 Where MW is ahead (capabilities Qiskit + hardware do not offer today)

| Capability | MW / `mwqc` (measured here) | Qiskit simulators / quantum hardware |
|---|---|---|
| **Live inference** on a physics question | KS eigenvalues **0.2 ms** vs ~40 s PySCF (~10⁵×); EUV aerial image + CD **2.2 ms** | hardware: queue + thousands of shots per job (seconds to hours); simulators: full re-simulation per query |
| **Uncertainty you can trust** | stated σ checked on held-out data: **93 %** (KS) and **94 %** (litho) inside ±2σ after calibration | shot-noise std only; hardware error is device noise, not a calibrated predictive σ |
| **"Is this input outside what I know?"** | epistemic-σ novelty detection, AUROC **1.000** on unseen mask families | no equivalent |
| **Error bar on every simulated expectation** | MPS truncation bound returned in `stds`; tracked fidelity matched true fidelity at n = 20 | Aer exposes truncation only in logs |
| **Runs where the data is** | CPU only, no cloud, no cryogenics; 1000-qubit low-entanglement circuits in MB of RAM | hardware: dilution refrigerators, cloud access |
| **Physics constraints as hard guarantees** | band-limited optics layer: predictions *cannot* violate the 2NA/λ spectrum (out-of-band power 10⁻¹⁵) | n/a |

### 5.4 State of quantum hardware, as of November 2026 (sourced, so it cannot be dismissed)

> **NO QUANTUM COMPUTER HAS YET PRODUCED A USEFUL CHEMISTRY OR MATERIALS RESULT THAT CLASSICAL
> METHODS CANNOT MATCH.**
>
> **IBM'S 127-QUBIT "UTILITY" EXPERIMENT (KIM ET AL., NATURE 618, 500, 2023) WAS REPRODUCED ON
> CLASSICAL COMPUTERS WITHIN WEEKS USING TENSOR NETWORKS AND PAULI PROPAGATION (TINDALL ET AL., PRX
> QUANTUM 2024; BEGUŠIĆ & CHAN, SCI. ADV. 2024): THE SAME FAMILY OF METHOD AS `mwqc`'S MPS.**
>
> **FAULT-TOLERANT MACHINES ARE ON VENDOR ROADMAPS FOR ~2029 AND LATER, NOT AVAILABLE NOW.**

To be precise about what hardware *has* done: it has run circuits on 50-150 physical qubits (Google
Sycamore 53 qubits in 2019; IBM Eagle 127 in 2023; Google Willow 105 in 2024, with error correction
improving as the code grows). What it has not done is beat classical methods on a problem someone
needed solved. That gap is MW's opening, and it should be argued with the sourced facts above.

### 5.5 scikit-learn API (adoption path for developers)

```python
from mwqc.sklearn_api import MWSurrogateRegressor, QuantumKernelClassifier, QuantumFeatureMap
reg = MWSurrogateRegressor(n_members=5).fit(X, y);  mu, sd = reg.predict(X_new, return_std=True)
clf = QuantumKernelClassifier(reps=2, C=5).fit(X, y)          # works in GridSearchCV / Pipeline
pipe = make_pipeline(QuantumFeatureMap(), LogisticRegression())
```

Honest numbers (two-moons, 80 points, 3-fold CV): RBF-SVC 0.975; quantum kernel 0.926 (ZZ map) and 0.963
without entanglement. `QuantumFeatureMap` + logistic regression 0.85 = logistic regression on raw features.
The kernels run on MPS beyond 20 features (statevector == MPS kernel to 1e-10, tested), so 50-200-feature
quantum kernels fit on a laptop. Whether they *help* is an empirical question per dataset.

### 5.6 PyQML (Zickert, *Hands-On Quantum Machine Learning with Python*, Vol. 1) on mwqc

The book's code uses the pre-1.0 Qiskit API (`execute`, `Aer.get_backend`, `qc.i`), which today's Qiskit has
removed, so the original notebooks no longer run on Qiskit 2.x either. `mwqc.qiskit_compat` provides that API.
Porting is one import line. Validation runs the *same* source on both stacks (`pyqml_port/legacy_qiskit.py`
restores the old API on Qiskit 2.4 + Aer 0.17):

| book technique (chapter) | agreement Qiskit+Aer vs mwqc |
|---|---|
| amplitude initialisation (2-3) | 1 − F = 0 |
| measured superposition, 4,000 shots (5) | TVD 0.015 (shot noise) |
| unitary simulator (6) | max \|ΔU\| 1.4e-16 |
| registers + Toffoli oracle with ancillas (6-8) | TVD 0.006 |
| quantum Bayesian network, sampled (9-10) | TVD 0.004 |
| quantum Bayesian network, exact | 1 − F ≈ 1e-15; P(survive) 0.3946 = classical Bayes formula 0.3946 |
| Grover amplification with `mcx` + `ZGate().control` (13) | 1 − F = 1.7e-14; P(marked) 0.9966 on both |
| phase kickback (11) | 1 − F = 4e-16 |

The book repository has no license, so its notebooks are not redistributed. `pyqml_port/port_book.py --book <clone>
--out <dir>` writes ported notebooks locally, and `--validate` runs every chapter on both stacks. *Running
the full book was blocked in the build session (third-party code execution) and is left for the owner to run.*

**Where Qiskit is simply better:** breadth, C++ performance, noise models, hardware access, community,
maintenance. `mwqc` is ~2,500 lines of Python.

---

## 6. Issues: what earlier work claimed vs what holds

Detailed per-file evidence is in **`docs/legacy_audit.md`**. Summary:

| Earlier claim (article / code) | What holds |
|---|---|
| "1,000 qubits on a laptop" | **True** for product and low-entanglement states, now shown gate by gate. Not new as a capability (MPS, 2003; Aer MPS, ITensor, quimb all do it). |
| "10^302 GB → 2.7 GB", "10^26 compute reduction" | A ratio of representation sizes, not a measured saving. States that need 2^n amplitudes still need them. **Withdraw.** |
| "Zero decoherence, 0 % circuit error at n = 1000" | Trivially true of any classical simulation. The real error is truncation, now reported with every result. |
| "Phase MAE 4.1°" | 93.6 % of the 1000 target phases were ≈ 0 by construction (j capped at 2^53). The full-range closed form needs no learning. |
| "100 % detection of Haar-random states" | The "Haar" samples were random *pure* Bloch points (r = 1); real Haar states have r ≈ 0 per qubit. Synthetic separation. |
| `mw_quantum` "drop-in replacement for Qiskit/Aer" | Hard-coded eigenvalues, circular estimator (output = reference + noise), placeholder counts, brute-force classical period finding. Replaced. |
| `oxygen_pyscf_validation_v2.py` many-body fit | Only n = 1, 2 are real; n ≥ 3 are generated by the formula that is then fitted. |
| KS v13 accuracy | Leaky features (§4.1). Fixed: geometry inputs give 36.8 meV honestly. |
| Lithography "240,000× vs RCWA" | No RCWA was run; optics had a pixel-pitch bug. Fixed model; speed-up measured vs the model actually used. |
| "MW covers 60-70 % of the commercial quantum market" | Unsupported by anything here. Chemistry/materials *ground states* in 3D are often strongly entangled; DMRG for chemistry needs χ in the thousands. Do not repeat. |

**Open issues in this repo (known, not hidden):**
1. MPS is pure Python/numpy: ~200 µs per two-qubit gate at small χ. qiskit-aer's C++ MPS is the
   performance reference; port the gate kernel to C/numba if speed matters.
2. Long-range gates are routed with SWAP chains (O(distance)). Fine for LNN-friendly circuits; slow
   for all-to-all connectivity. MPO-based long-range gates are not implemented.
3. No noise models or density matrices (Aer has both).
4. UCCSD-VQE runs on the dense statevector in the particle sector, so ≤ ~16 qubits. VQE over MPS
   (DMRG-style) is the natural next step and would reuse `MPSState`.
5. KS surrogate: cluster proxy, not periodic LiFePO₄; valid only inside the sampled ±0.08 Å
   displacement range; 200 training points. The PBE0/def2-SVP set (25 points) is too small.
6. Lithography: scalar Kirchhoff thin-mask optics, no mask-3D effects, no polarisation, threshold resist.
   Training on RCWA/FDTD output is required before any claim about real EUV.
7. Qiskit interop is circuit-level only (no `BaseEstimatorV2` subclassing, no Parameter binding).
8. Uncertainty calibration uses a single validation split. Conformal prediction would give guarantees.

---

## 7. Suggested next steps

1. **DMRG / MPS-VQE for chemistry** on top of `MPSState`. This is the legitimate route to
   "many-qubit chemistry on a laptop" (LiH → N₂ → small transition-metal clusters at χ ≈ 100-1000).
2. **Periodic DFT data** (Quantum ESPRESSO/GPAW, free) for real LiFePO₄ and MXene cells, then
   `MWSurrogate` + `ood_score` as an active-learning loop ("call DFT only when σ is large").
3. **RCWA ground truth** for lithography (open-source RCWA such as S4 / `rcwa`). The surrogate code is unchanged.
4. Numba/C kernel for `_apply_adjacent`, then re-benchmark against qiskit-aer MPS.
5. Rewrite the public article's quantitative claims along §6. The verified
   1000-qubit QFT results (§3.2) stand on their own.

---

## 8. Provenance

| Source | Used for |
|---|---|
| `~/SOLVE_QUANTUM/mw_quantum/` (`mw_quantum` package, `quantum_fourier_manifold_e2e_v6.py`) | API shape; Bloch/fibre-bundle idea → `BlochState`/`MPSState`; audit |
| `~/FULL_REVIEW/Potential_use/physics_math/MW_FRAMEWORK_IMPLEMENTATION/oxygen_pyscf_validation_v2.py`, `log_of_1000qubits_2.txt` | audit |
| `~/FULL_REVIEW/Potential_use/physics_math/MW_Framework/` | Bayesian-manifold pattern → `MWSurrogate` |
| `~/FULL_REVIEW/Potential_use/physics_math/QUANTUM_MECH_MW/QM_MW_FRAMEWORK_CAPABILITIES.md` | read for intent; its claims (e.g. measurement collapse via heavy-tailed priors) are not implemented or endorsed here |
| `~/SOLVE_QUANTUM/Kohn_Sham_ScreenApproach/ks_mw_inference_v13.py` + PySCF caches | KS use case data, geometry replay, prior 3 |
| `~/SOLVE_QUANTUM/LITHOGRAPHY_1/mw_lithography_demo.py` | litho use case (forward model rewritten) |

References: Vidal, PRL 91, 147902 (2003) · White, PRL 69, 2863 (1992) · Fowler, Devitt & Hollenberg,
Quantum Inf. Comput. 4, 237 (2004) · Nielsen & Chuang, *Quantum Computation and Quantum Information*, §5.1 ·
Lakshminarayanan et al., deep ensembles, NeurIPS 2017.

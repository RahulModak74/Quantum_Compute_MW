# Quantum_Compute_MW: the four use cases

Each use case has a runnable script, is checked against an independent reference, and states
both what it shows and what it does not. Numbers are from runs on a Lenovo LOQ 15IRH8 laptop
(CPU; GPU used only for CNN training in use case 2). Raw outputs are in `results/`.

| # | Use case | Size | Reference it is checked against | Headline |
|---|---|---|---|---|
| 1 | Kohn-Sham (DFT) eigenvalues, Fe-O cluster | 6 valence KS levels, 200 geometries | PySCF PBE (real DFT runs) | **36.8 meV** MAE from geometry alone (< 43.6 meV chemical accuracy); 0.2 ms vs ~40 s per solve |
| 2 | EUV lithography aerial image + CD | 64×64 masks, ±30 nm defocus | Abbe partially-coherent optics model | CD error **0.17 nm**, calibrated ±2σ, unseen patterns flagged (AUROC 1.000) |
| 3 | Quench dynamics of a quantum magnet | **100-200 qubits** | exact free-fermion solution (any N) | every site, every time step within **1e-4** of exact (N = 100, t ≤ 2) |
| 4 | Ground states of quantum magnets | **100-200 qubits** | exact free-fermion energy; dense ED at N = 12; Bethe ansatz | critical Ising N = 100: energy error **5.6e-7** (rel. 4e-9); Heisenberg N = 100 matches Bethe bulk energy to 5 digits |

Use cases 1-2 are **MW surrogates** (Bayesian physics priors, live inference, checked uncertainty).
Use cases 3-4 are **beyond-50-qubit quantum simulations** on the MW sparse-Hilbert-space engine (MPS).

---

## Use case 1: Kohn-Sham eigenvalues for battery-cathode chemistry (MW surrogate)

**Question.** Predict the 6 valence Kohn-Sham eigenvalues of the LiFePO₄-motivated Fe-O cluster
(Fe O₄, charge +2, spin 4) for a new geometry without running DFT.

**How.** 200 real PySCF PBE/STO-3G calculations from the earlier v13 work. Their geometries were
recovered by replaying v13's seeded random generator, and re-running PySCF reproduces the cached eigenvalues to 3×10⁻⁸ Ha.
Input = 10 interatomic distances; output = 6 eigenvalues ± σ. `MWSurrogate`, 5-fold cross-validation.

| model | MAE (meV) |
|---|---|
| predict the mean | 210.4 ± 7.7 |
| ridge regression | 86.0 ± 6.1 |
| **MW surrogate** | **36.8 ± 4.4** |

* Below chemical accuracy (43.6 meV) against this reference. ±2σ coverage 93 % after calibration.
* 0.2 ms per prediction vs ~40 s per PySCF solve.
* Also found and fixed: the v13 inputs leaked the answer (7 of 28 features were computed from the target
  eigenvalues; a linear fit on them alone gave 24.9 meV).
* **Limits:** cluster proxy, not periodic LiFePO₄; valid inside the sampled ±0.08 Å displacements.
  Next step: train on periodic DFT (Quantum ESPRESSO / GPAW).

Run: `python3 usecases/kohn_sham/ks_surrogate.py`. Reference route, done the IBM way for comparison:
`python3 usecases/kohn_sham/ks_vqe.py` (PySCF → Jordan-Wigner → UCCSD-VQE, within chemical accuracy of FCI 8/8,
H₂ to H₂O at 14 qubits).

---

## Use case 2: EUV lithography, aerial image and critical dimension (MW surrogate)

**Question.** For a mask pattern and defocus, predict the printed image and the critical dimension (CD),
with an uncertainty, in milliseconds.

**How.** Corrected forward model: λ = 13.5 nm, NA = 0.33, annular illumination (52 source points),
defocus, 3 nm acid blur, dose-referenced threshold. 2,000 training masks in 4 families.
CNN ensemble with a **hard physics layer**: every predicted image is projected onto the band
|f| ≤ 2NA/λ that real optics can transmit.

| metric | result |
|---|---|
| CD mean absolute error | **0.17 nm** (pixels are 4 nm) |
| image RMSE | 0.0039 (clear field = 1) |
| physically impossible spectral content | 2×10⁻¹⁵ (zero by construction) |
| ±2σ CD coverage after calibration | 93 % |
| detection of never-seen pattern families | AUROC 1.000 |
| time per mask | 2.2 ms vs 20.7 ms for the optics model |

* **Limits:** scalar thin-mask optics (no mask-3D, no polarisation), threshold resist. The 9× speed-up is
  against a cheap model; real value comes when the training data are rigorous RCWA/FDTD solves.

Run: `python3 usecases/lithography/litho_surrogate.py`.

---

## Use case 3: real-time dynamics of 100-200 interacting quantum spins (beyond 50 qubits)

**Question.** A chain of N spin-½ magnets (transverse-field Ising model) starts aligned along x, and the
coupling is switched on. How do magnetisation, correlations and entanglement evolve at every site?
This is the 1-D version of the model in IBM's 127-qubit "utility" experiment (Nature 2023), and of real
quantum magnets such as CoNb₂O₆.

**Why it is beyond the usual tools.** A dense simulator (Qiskit `Statevector`, Aer statevector) needs 2¹⁰⁰
amplitudes, about 2×10²² GB. Current hardware gives noisy estimates at this width.

**How.** Second-order TEBD on the MPS engine (`mwqc.dynamics.tebd`), dt = 0.02.
**Checked against the exact solution** (Jordan-Wigner → free Majorana fermions, exact for any N),
which is computed independently. Then the same engine on the *non-integrable* chaotic Ising chain
(longitudinal field 0.5), where no exact answer exists, checked by convergence in bond dimension χ.

Verified so far (quick run, N = 100, t ≤ 2):

| run | result |
|---|---|
| integrable, N = 100, χ = 8 / 16 / 32 | max error over **all 100 sites and all times 1.0×10⁻⁴** (the Trotter error of dt = 0.02), ~5 s each |
| chaotic, N = 100 | χ = 16 vs χ = 32: max difference 2.3×10⁻⁷; χ = 8 already 1.4×10⁻³ |

The full run (N = 100 and 200, t ≤ 8, χ up to 128) writes `results/uc1_quench_dynamics.json`. It reports the
**time horizon** each χ reaches before the error exceeds 10⁻³.

* **Limits (stated up front):** after a quench, entanglement grows linearly in time, so the bond dimension
  needed grows exponentially in time. Any classical method, MW included, reaches a finite time horizon.
  That horizon is measured, not hidden. Long-time dynamics of large 2-D systems remains genuinely hard.

Run: `python3 usecases/many_body/uc1_quench_dynamics.py` (`--quick` ≈ 1 min).

---

## Use case 4: ground states of 100-200-spin quantum magnets (beyond 50 qubits)

**Question.** Lowest-energy state and energy of (a) the critical transverse-field Ising chain, the hardest
point (gapless, logarithmic entanglement), and (b) the spin-½ Heisenberg antiferromagnet, the model of
real chain compounds (KCuF₃, Sr₂CuO₃).

**Why it is beyond the usual tools.** On hardware this would be a 100-qubit VQE with deep circuits and
millions of shots. Dense simulation is impossible at N = 100.

**How.** Imaginary-time TEBD on the MPS engine with a shrinking step schedule (`mwqc.dynamics.ground_state`).

| system | result |
|---|---|
| critical Ising, N = 100, χ ≤ 32 | E = −126.96187618 vs exact −126.96187674: **error 5.6×10⁻⁷** (relative 4×10⁻⁹), bond dimension 25, 4 min |
| Heisenberg, N = 12 (validation) | MPS −5.14209063 vs dense exact −5.14209063: **error 7×10⁻¹⁰** |
| Heisenberg, N = 100, χ = 16 / 32 | E = −44.12231 / −44.12513; energy per bond at the chain centre **−0.44315**, equal to the Bethe-ansatz infinite-chain value 1/4 − ln 2 = −0.44315 (χ = 32, 9 min) |

* **Limits:** imaginary-time convergence is set by the energy gap. Critical chains need long schedules
  (a DMRG solver would converge faster and is the next step). 1-D chains are the MPS sweet spot. 2-D lattices
  need PEPS or larger χ.

Run: `python3 usecases/many_body/uc2_ground_states.py` (`--quick` ≈ 10 min).

---

## What use cases 3-4 do and do not claim

* They **do** show 100-200-qubit quantum simulations, verified against exact answers, on a laptop CPU in
  minutes. That size is out of reach for dense simulators, and today's hardware returns only noisy results at it.
* They do **not** show something tensor networks could not already do: IBM's own `qiskit-aer` has an MPS
  method, and IBM's 127-qubit 2023 result was matched classically with tensor networks within weeks.
  The MW contribution is the integrated, verified, CPU-first stack: an error bar on every number, a
  χ = 1 "Bloch" diagnostic for when the sparse picture holds, and Bayesian surrogates (use cases 1-2) for live
  inference.

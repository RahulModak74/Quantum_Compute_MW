# Audit of the earlier MW quantum code (what was kept, what was fixed, what was dropped)

Reviewed 2026-11-18. Every point below was checked against the source file named.
The goal was to find out what each piece really computes. It is not a verdict on
the ideas behind it: most of those ideas survive in `mwqc` in a form that can now be checked.

## 1. `~/SOLVE_QUANTUM/mw_quantum/` — the "Qiskit-equivalent" package (`mw_quantum`)

| File / feature | What it claims | What it actually does | Status in `mwqc` |
|---|---|---|---|
| `driver.py` `MWDriver._BUILTIN_SYSTEMS` | PySCF-verified eigenvalues for LiFePO4, MXene | Hard-coded numbers (`eps_ref`, `E_rhf`). The s-only Gaussian integral engine is built but its eigenvalues are **discarded** in favour of the table. Fe/P/Ti "STO-3G" exponents are invented (real Fe 1s exponent is ~1300, not 12.5). | Replaced by `mwqc.chem.MolecularProblem.from_pyscf` — real integrals, no tables |
| `driver.py` custom-atom path | PySCF driver | Real PySCF RHF when installed. This part was sound. | Kept in spirit |
| `mapper.py` `MWManifoldMapper` | Replaces Jordan-Wigner | Copies `H_core`/`S_mat` into a dataclass and reports `num_qubits = 2*n_basis`. No operator mapping happens. | Replaced by `mwqc.chem.jordan_wigner` (verified against FCI to 1e-9 mHa) |
| `estimator.py` `MWBayesianEstimator._solve` | Ground-state energy with Bayesian sigma | Training data = **the reference eigenvalues plus Gaussian noise**, then `E_tot = E_rhf_ref + 2*sum(eps_mw - eps_ref)`. The output is the input plus a small shift: circular. | Replaced by `MWEstimator` (exact or MPS with a truncation bound) + `vqe_uccsd` |
| `estimator.py` `get_counts()` | Aer-style measurement counts | Uniform placeholder counts over the first 8 bitstrings. | `MWSampler` samples exactly from the MPS / statevector |
| `estimator.py` `_shor_counts` | Shor/QPE via manifold | Finds the period by classical brute force `for period in range(1, N+1)` — exponential in bit-length — and writes ideal QPE peaks. | Dropped. Shor is out of scope (see README §6) |
| `get_statevector()` | Statevector | Always returns |00…0⟩. | `StatevectorState`, `MPSState.to_vector()` |
| `transpile()` | Compile | Returns the circuit unchanged. | Not provided (no hardware target) |

**Net functional overlap of `mw_quantum` with Qiskit: effectively 0 %.** The API names mirror
Qiskit's, but none of the quantum operations are performed.

## 2. `quantum_fourier_manifold_e2e_v6.py` — the 1000-qubit QFT run (article of 2 May 2026)

1. **The training target is a closed-form product state.** `qft_bloch_analytical` encodes the
   textbook result (Nielsen & Chuang eq. 5.4) that QFT|j⟩ = ⊗_q (|0⟩ + e^{2πi φ_q}|1⟩)/√2. A product
   state needs no state vector at any n. `mwqc` now *runs* the 499,500-gate circuit on 1000 qubits
   and matches that formula to 3e-15 rad, without any learning (`benchmarks/qft_1000.py`, Part A).
2. **`j` was capped at 2^53** (`max_j = min(2**n, 2**53-1)`), so for n = 1000 the output phase
   φ_m = 2πj/2^(m+1) is ≈ 0 for most qubits. Measured: **93.6 % of the 1000 target phases are within
   1 mrad of 0 for every sample.** The headline "circular MAE 4.1°" is averaged over mostly trivial targets.
3. **The "Haar-random anomalies" are not Haar-random states.** v6 draws a random point on the
   surface of each Bloch sphere (|r| = 1). A genuinely Haar-random n-qubit state has single-qubit
   reduced states that are almost maximally mixed (|r| ≈ 0). The 100 % detection rate separates two
   synthetic distributions that differ by construction.
4. **Superpositions are approximated by averaging Bloch angles** (`qft_bloch_superposition`).
   That is not how quantum states combine. The true reduced state of a superposition is a different
   object, and its Bloch vector shrinks when the state is entangled.
5. The public log `log_of_1000qubits_2.txt` (line 1032-1033) prints `loss= 0.00` for every epoch of the
   50-epoch 1000-qubit run. That suggests degenerate training. Not re-run here.
6. "10^26 / 10^298 compute reduction" divides representation sizes (2^n vs 2n). It is not a
   measured saving. For states that really need 2^n amplitudes, no representation avoids the cost.

What survives, now rigorous: the per-qubit Bloch description (`BlochState`, χ = 1) and the
fibre-bundle chain idea (MPS bond dimension χ > 1). Run on a *general* product input, the QFT
output is entangled. It still needs only χ = 7 at n = 1000, a 1.44 MB state, with the error tracked.

## 3. `oxygen_pyscf_validation_v2.py`

* Only n = 1, 2 O2 molecules are real PySCF CASSCF energies. For n ≥ 3, `synthetic_energy()` *generates*
  energies from a pairwise formula. A linear pairwise model is then fitted to them, so its accuracy
  on those points is guaranteed by construction (circular).
* The file is honest about this in its docstring ("REPRESENTATION ratio"). The conclusion "manifold
  reproduces many-body energetics to chemical accuracy" is not supported beyond n = 2.
* Not ported. The checked replacement is `usecases/kohn_sham/ks_vqe.py` (FCI-verified).

## 4. `Kohn_Sham_ScreenApproach/ks_mw_inference_v13.py` (the tested one)

The strongest of the earlier pieces. It trains against **real PySCF eigenvalues** and is candid about
the 5-atom cluster proxy. Two problems remain:

1. **Target leakage.** 7 of the 28 inputs (`fe_d_band_centre`, `o_p_band_centre`, `exchange_splitting`,
   `charge_transfer`, `fermi_level`, `band_gap`, `dos_at_fermi`) are computed *from the eigenvalues
   being predicted* (`_features_from_pyscf`). A plain linear fit on those 7 gives **24.9 meV**, with no
   neural network and no physics. The remaining non-constant inputs (`li_occupancy`, `volume_strain`,
   `discharge_voltage`) depend only on `li_occupancy`, which **never enters the PySCF calculation**:
   they score 213 meV, no better than predicting the mean (~210 meV).
   So v13's reported MAE measures how well the network inverts its own feature formulas.
2. The cluster is `Fe O4` with charge +2, spin 4. There is no Li and no P, so "LiFePO4" names
   the motivation, not the system.

Fix in `usecases/kohn_sham/ks_surrogate.py`: the geometries were never saved, but v13 generated them
from a seeded RNG. They are **replayed bit-for-bit** (checked by re-running PySCF: 3e-8 Ha agreement).
The surrogate then learns the real map *geometry → eigenvalues*.

## 5. `LITHOGRAPHY_1/mw_lithography_demo.py`

1. **Frequency-grid bug:** `np.fft.fftfreq(nx, d=lambda_nm)` treats every pixel as 13.5 nm while the
   code assumes 1 nm. The NA cutoff therefore lands in the wrong place.
2. **Per-image max normalisation:** `I / I.max()` makes the resist threshold (dose) change from mask
   to mask, so printed CD is not dose-referenced.
3. **Coherent on-axis illumination only.** EUV scanners use partially coherent sources.
4. The "quantum" priors are resist photochemistry (absorption, acid yield). That is not quantum computing.
5. The "240,000× vs RCWA" speedup compares an assumed 2-minute RCWA time with network inference. No
   RCWA was run.

All of the above are fixed in `usecases/lithography/litho_physics.py`, and the surrogate is
measured against the forward model it actually learned from.

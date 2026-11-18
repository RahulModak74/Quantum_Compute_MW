#!/usr/bin/env python3
"""
Use case B (>50 qubits): ground states of 100-200-spin quantum magnets.

  B1  Critical transverse-field Ising chain (g = J, the hardest point: gapless, log entanglement),
      N = 100 and 200, plus g = 0.5 and 1.5 at N = 100. Checked against the EXACT free-fermion energy.
  B2  Spin-1/2 Heisenberg antiferromagnet, N = 100: the model of real chain materials
      (KCuF3, Sr2CuO3, CuPzN). Not free-fermion solvable. Validated at N = 12 against dense exact
      diagonalisation, then chi-convergence at N = 100 and comparison with the Bethe-ansatz bulk
      energy per bond, 1/4 - ln 2 = -0.443147.

Method: imaginary-time TEBD on the MPS (mwqc.dynamics.ground_state).
A VQE on hardware for a 100-site chain would need 100 qubits, deep circuits and millions of shots;
dense simulation needs 2^100 amplitudes.

Usage: python3 usecases/many_body/uc2_ground_states.py [--quick]
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from mwqc import MPSState                                                         # noqa: E402
from mwqc.dynamics import (bond_energy, dense_hamiltonian, ground_state, heisenberg_bonds,   # noqa: E402
                           tfim_bonds, tfim_exact_ground_energy)
from mwqc.gates import X, Y, Z                                                    # noqa: E402

QUICK_SCHEDULE = ((0.1, 300), (0.05, 150), (0.02, 150), (0.01, 100))


def neel(N, chi):
    return MPSState.from_bits([i % 2 for i in range(N)], chi_max=chi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    kw = {"schedule": QUICK_SCHEDULE} if a.quick else {}
    out = {"B1": [], "B2": []}

    cases = [(100, 1.0, 32)] if a.quick else [(100, 1.0, 32), (200, 1.0, 32), (100, 0.5, 16), (100, 1.5, 16)]
    for N, g, chi in cases:
        st = MPSState.from_bloch_angles([np.pi / 2] * N, [0.0] * N, chi_max=chi)
        t0 = time.time()
        ground_state(st, tfim_bonds(N, 1.0, g), **kw)
        e = bond_energy(st, tfim_bonds(N, 1.0, g))
        ex = tfim_exact_ground_energy(N, 1.0, g)
        row = {"N": N, "g": g, "chi": chi, "E_mps": e, "E_exact": ex, "abs_err": e - ex,
               "rel_err": (e - ex) / abs(ex), "max_bond": max(st.bond_dims()),
               "S_mid_bits": st.entanglement_entropy(N // 2 - 1), "seconds": time.time() - t0}
        out["B1"].append(row)
        print(f"[B1] TFIM N={N} g={g}: E={e:.8f} exact {ex:.8f}  err {e - ex:.1e} "
              f"(rel {row['rel_err']:.1e}), bond {row['max_bond']}, S_mid {row['S_mid_bits']:.2f} bits, "
              f"{row['seconds']:.0f}s")

    # B2 validation at N=12
    n = 12
    Hd = dense_hamiltonian(heisenberg_bonds(n), n)
    e_dense = float(np.linalg.eigvalsh(Hd)[0])
    st = neel(n, 64)
    ground_state(st, heisenberg_bonds(n), **kw)
    e12 = bond_energy(st, heisenberg_bonds(n))
    out["B2_validation_N12"] = {"E_mps": e12, "E_dense": e_dense, "abs_err": e12 - e_dense}
    print(f"[B2] Heisenberg N=12 validation: MPS {e12:.8f} vs dense exact {e_dense:.8f} (err {e12 - e_dense:.1e})")
    N = 100
    for chi in ((16, 32) if a.quick else (16, 32, 64)):
        st = neel(N, chi)
        t0 = time.time()
        ground_state(st, heisenberg_bonds(N), **kw)
        e = bond_energy(st, heisenberg_bonds(N))
        mid = [np.real(st.expect_2site(0.25 * (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z)), b))
               for b in range(N // 2 - 5, N // 2 + 5)]
        row = {"N": N, "chi": chi, "E": e, "E_per_bond": e / (N - 1), "bulk_bond_energy_centre": float(np.mean(mid)),
               "bethe_bulk": 0.25 - np.log(2), "max_bond": max(st.bond_dims()),
               "S_mid_bits": st.entanglement_entropy(N // 2 - 1), "seconds": time.time() - t0,
               "fidelity_estimate": st.fidelity_estimate}
        out["B2"].append(row)
        print(f"[B2] Heisenberg N={N} chi={chi:>3}: E={e:.6f}, centre bond energy {row['bulk_bond_energy_centre']:.5f} "
              f"(Bethe bulk {0.25 - np.log(2):.5f}), S_mid {row['S_mid_bits']:.2f} bits, {row['seconds']:.0f}s")
    p = os.path.join(HERE, "..", "..", "results", "uc2_ground_states.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"saved {os.path.normpath(p)}")


if __name__ == "__main__":
    main()

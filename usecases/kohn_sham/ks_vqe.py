#!/usr/bin/env python3
"""
Kohn-Sham use case, part 2 — the "quantum computer" route to the same chemistry,
done the way IBM's Qiskit Nature does it, but with every number checked.

  PySCF integrals in a Kohn-Sham (PBE) or Hartree-Fock orbital basis
    -> second-quantised Hamiltonian
    -> Jordan-Wigner qubit Hamiltonian (mwqc.chem.jordan_wigner)
    -> VQE with a UCCSD ansatz (mwqc.chem.vqe_uccsd)
    -> compared with exact diagonalisation and PySCF FCI

Why this matters for the MW story: KS-DFT itself is a mean-field method that never
needs qubits. A quantum computer (or this classical emulation of one) enters only when
you want the correlation energy *beyond* DFT. The table this prints is the honest
size of that job: qubits = 2 x orbitals, Pauli terms ~ orbitals^4.

Usage:  python3 usecases/kohn_sham/ks_vqe.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from mwqc.chem import MolecularProblem, exact_ground_energy, jordan_wigner, vqe_uccsd   # noqa: E402

MHA = 1000.0
CHEM_ACC_MHA = 1.6

SYSTEMS = [
    ("H2  r=0.735", "H 0 0 0; H 0 0 0.735"),
    ("H2  r=1.50 (stretched)", "H 0 0 0; H 0 0 1.50"),
    ("LiH r=1.595", "Li 0 0 0; H 0 0 1.595"),
    ("H2O", "O 0 0 0; H 0.7572 0.5865 0; H -0.7572 0.5865 0"),
]


def main():
    rows = []
    print(f"{'system':<24}{'orb':>5}{'qubits':>8}{'terms':>7}{'params':>8}"
          f"{'E_KS(PBE)':>12}{'E_HF':>12}{'E_FCI':>12}{'VQE-FCI':>10}{'HF-FCI':>9}{'time':>7}")
    print(f"{'':<24}{'':>5}{'':>8}{'':>7}{'':>8}{'Ha':>12}{'Ha':>12}{'Ha':>12}{'mHa':>10}{'mHa':>9}{'s':>7}")
    for name, atom in SYSTEMS:
        for orbs in ("ks", "hf"):
            t0 = time.time()
            prob = MolecularProblem.from_pyscf(atom, basis="sto-3g", orbitals=orbs, xc="pbe")
            if prob.num_qubits > 14:
                print(f"{name:<24}{prob.n_orb:>5}{prob.num_qubits:>8}   skipped (dense sector > laptop budget)")
                rows.append({"system": name, "orbitals": orbs, "qubits": prob.num_qubits, "skipped": True,
                             "e_hf": prob.e_hf, "e_fci": prob.e_fci})
                break
            H = jordan_wigner(prob)
            e_ex = exact_ground_energy(H, prob)
            r = vqe_uccsd(prob, H, maxiter=300)
            dt = time.time() - t0
            row = {"system": name, "orbitals": orbs, "n_orb": prob.n_orb, "qubits": prob.num_qubits,
                   "pauli_terms": len(H), "uccsd_params": r.n_params, "e_ks": prob.e_ks, "e_hf": prob.e_hf,
                   "e_fci": prob.e_fci, "e_exact_qubit": e_ex, "e_vqe": r.energy,
                   "vqe_minus_fci_mHa": (r.energy - prob.e_fci) * MHA,
                   "hf_minus_fci_mHa": (prob.e_hf - prob.e_fci) * MHA,
                   "mapping_error_mHa": (e_ex - prob.e_fci) * MHA, "seconds": dt}
            rows.append(row)
            eks = f"{prob.e_ks:12.6f}" if prob.e_ks is not None else f"{'-':>12}"
            print(f"{name + ' [' + orbs + ']':<24}{prob.n_orb:>5}{prob.num_qubits:>8}{len(H):>7}{r.n_params:>8}"
                  f"{eks}{prob.e_hf:12.6f}{prob.e_fci:12.6f}{row['vqe_minus_fci_mHa']:10.3f}"
                  f"{row['hf_minus_fci_mHa']:9.2f}{dt:7.1f}")
    ok = [r for r in rows if not r.get("skipped")]
    print(f"\nQubit-mapping error (exact diag of JW Hamiltonian vs PySCF FCI): "
          f"max {max(abs(r['mapping_error_mHa']) for r in ok):.1e} mHa")
    print(f"VQE within chemical accuracy ({CHEM_ACC_MHA} mHa) of FCI: "
          f"{sum(abs(r['vqe_minus_fci_mHa']) < CHEM_ACC_MHA for r in ok)}/{len(ok)} runs")
    p = os.path.join(HERE, "..", "..", "results", "ks_vqe.json")
    with open(p, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"saved {os.path.normpath(p)}")


if __name__ == "__main__":
    main()

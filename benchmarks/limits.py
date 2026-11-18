#!/usr/bin/env python3
"""
Where the MW/MPS approach is cheap, and where it is not — measured.

L1  Entanglement wall: random brick-wall circuits on 40 qubits, chi capped at 64.
    Fidelity stays ~1 while entanglement fits in chi, then collapses with depth.
    (This is the regime of Shor, Grover, random-circuit sampling.)
L2  GHZ on 1000 qubits: maximally "global" correlation, yet only 1 ebit -> chi = 2.
L3  Side-by-side with IBM's own MPS simulator (qiskit-aer, method="matrix_product_state")
    on the same QFT circuits, and Qiskit's dense Statevector where it still fits.

Usage:  python3 benchmarks/limits.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mwqc import MPSState, MWSampler                       # noqa: E402
from mwqc.library import ghz, qft, qft_lnn, random_circuit  # noqa: E402


def main():
    only_l3 = "--only-l3" in sys.argv
    out = json.load(open("results/limits.json")) if only_l3 and os.path.exists("results/limits.json") else {}
    if not only_l3:
        l1_l2(out)
    l3(out)
    os.makedirs("results", exist_ok=True)
    with open("results/limits.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/limits.json")


def l1_l2(out):
    print("[L1] random brick-wall circuits, n=40, chi<=64")
    out["L1"] = []
    for depth in (2, 4, 6, 8, 10, 12, 16, 20):
        qc = random_circuit(40, depth, seed=depth)
        t = time.time()
        st = MPSState(40, chi_max=64).run(qc)
        row = {"depth": depth, "seconds": time.time() - t, "max_bond": max(st.bond_dims()),
               "fidelity_estimate": st.fidelity_estimate,
               "mid_entropy_bits": st.entanglement_entropy(19)}
        out["L1"].append(row)
        print(f"  depth {depth:>2}: max bond {row['max_bond']:>3}, fidelity est {row['fidelity_estimate']:.4f}, "
              f"S(mid) {row['mid_entropy_bits']:.2f} bits, {row['seconds']:.1f}s")

    print("\n[L2] GHZ(1000) sampled with MPS")
    t = time.time()
    r = MWSampler("mps", seed=0).run([ghz(1000)], shots=1000).result()[0]
    c = r.data.get_counts()
    out["L2"] = {"seconds": time.time() - t, "distinct_outcomes": len(c),
                 "max_bond": r.metadata["max_bond_used"], "memory_MB": r.metadata["memory_MB"]}
    print(f"  {len(c)} distinct outcomes (all-0 / all-1), max bond {r.metadata['max_bond_used']}, "
          f"{r.metadata['memory_MB']:.2f} MB, {out['L2']['seconds']:.1f}s")


def l3(out):

    print("\n[L3] QFT on a random product state: mwqc MPS vs qiskit-aer MPS vs Qiskit Statevector")
    out["L3"] = []
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        from qiskit.quantum_info import Statevector
        have_q = True
    except Exception as e:  # pragma: no cover
        print(f"  qiskit/aer unavailable: {e}")
        have_q = False
    for n in (20, 50, 100, 200):
        rng = np.random.default_rng(n)
        th, ph = rng.uniform(0, np.pi, n), rng.uniform(0, 2 * np.pi, n)
        row = {"n": n}
        t = time.time()
        st = MPSState.from_bloch_angles(th, ph, chi_max=64).run(qft_lnn(n))
        row["mwqc_mps_s"] = time.time() - t
        row["mwqc_max_bond"] = max(st.bond_dims())
        if have_q:
            qc = QuantumCircuit(n)
            for q in range(n):
                qc.ry(th[q], q)
                qc.p(ph[q], q)
            qc.compose(qft(n).to_qiskit(), inplace=True)
            sim = AerSimulator(method="matrix_product_state", matrix_product_state_max_bond_dimension=64)
            # ry/p/h/cp/swap are Aer-native: no transpile (transpiling to the backend caps width at 63)
            tq = qc
            t = time.time()
            sim.run(tq, shots=1).result()
            row["aer_mps_s"] = time.time() - t
            if n <= 26:
                qd = QuantumCircuit(n)
                for q in range(n):
                    qd.ry(th[q], q)
                    qd.p(ph[q], q)
                qd.compose(qft(n).to_qiskit(), inplace=True)
                t = time.time()
                sv = Statevector(qd)
                row["qiskit_statevector_s"] = time.time() - t
                row["fidelity_mwqc_vs_qiskit"] = float(abs(np.vdot(st.to_vector(), sv.data)) ** 2)
        out["L3"].append(row)
        print("  " + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))



if __name__ == "__main__":
    main()

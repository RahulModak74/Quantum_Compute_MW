#!/usr/bin/env python3
"""
The 1000-qubit QFT, done as a real gate-by-gate circuit simulation and checked.

Part A  basis-state input |j>, j a random 1000-bit integer (all 2^1000 values reachable,
        not capped at 2^53 as in quantum_fourier_manifold_e2e_v6.py).
        BlochState (chi = 1), all 499,500 two-qubit gates, compared to the closed form.
Part B  general product-state input (random point on every Bloch sphere).
        Here the QFT output IS entangled. MPS with a bond cap; truncation error reported.
Part C  validation of Part B's method at n = 20 against the exact statevector.

Usage:  python3 benchmarks/qft_1000.py [--n 1000] [--chi 32] [--skip-a] [--skip-b]
Results merge into results/qft_<n>.json, so parts can be re-run separately.
"""
import argparse
import json
import math
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mwqc import BlochState, MPSState, StatevectorState          # noqa: E402
from mwqc.library import qft, qft_basis_state_bloch, qft_lnn     # noqa: E402


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def circ_err(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def run_a(n, rng, circ, out):
    j = int.from_bytes(rng.bytes((n + 7) // 8), "little") & ((1 << n) - 1)
    bits = [(j >> q) & 1 for q in range(n)]
    st = BlochState.from_bits(bits)
    t = time.time()
    st.run(circ)
    ta = time.time() - t
    theta, phi = qft_basis_state_bloch(j, n)
    errs, rs = [], []
    for q in range(n):
        th, ph, r = st.bloch_angles(q)
        errs.append(max(abs(th - theta[q]), circ_err(ph, phi[q])))
        rs.append(r)
    out["A_basis_input"] = {
        "seconds": ta, "truncation_error": st.truncation_error,
        "max_angle_error_rad": float(max(errs)), "min_bloch_radius": float(min(rs)),
        "memory_state_bytes": st.memory_bytes(), "j_bits_set": int(sum(bits)),
    }
    print(f"\n[A] |j> input, chi=1:  {ta:.1f}s, truncation error {st.truncation_error:.1e}, "
          f"max angle error vs closed form {max(errs):.2e} rad, state = {st.memory_bytes()/1e3:.1f} kB")


def run_c(rng, chi_cap, out):
    nv = 20
    th0, ph0 = rng.uniform(0, np.pi, nv), rng.uniform(0, 2 * np.pi, nv)
    prod = MPSState.from_bloch_angles(th0, ph0)
    exact = StatevectorState(nv, prod.to_vector()).run(qft(nv)).to_vector()
    out["C_validation_n20"] = []
    for chi in (1, 2, 4, 8, chi_cap):
        m = MPSState.from_bloch_angles(th0, ph0, chi_max=chi).run(qft_lnn(nv))
        f = abs(np.vdot(m.to_vector(), exact)) ** 2
        out["C_validation_n20"].append({"chi": chi, "true_fidelity": f,
                                        "estimated_fidelity": m.fidelity_estimate,
                                        "max_bond": max(m.bond_dims())})
        print(f"[C] n=20 product input, chi<={chi:>3}: true fidelity {f:.10f}, "
              f"tracked estimate {m.fidelity_estimate:.10f}, max bond {max(m.bond_dims())}")


def run_b(n, rng, chi_cap, circ, out):
    thb, phb = rng.uniform(0, np.pi, n), rng.uniform(0, 2 * np.pi, n)
    m = MPSState.from_bloch_angles(thb, phb, chi_max=chi_cap)
    t = time.time()
    m.run(circ)
    tb = time.time() - t
    ent = [m.entanglement_entropy(k) for k in range(0, n - 1, max(1, n // 10))]
    out["B_product_input"] = {"seconds": tb, **m.summary(), "entropy_bits_sampled": ent}
    print(f"\n[B] random product input, chi<={chi_cap}: {tb:.1f}s, max bond {max(m.bond_dims())}, "
          f"fidelity estimate {m.fidelity_estimate:.6f}, state {m.memory_bytes()/1e6:.2f} MB")
    print("    entanglement entropy across sampled cuts (bits): " + " ".join(f"{e:.2f}" for e in ent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--chi", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    a = ap.parse_args()
    n = a.n
    os.makedirs("results", exist_ok=True)
    path = f"results/qft_{n}.json"
    out = json.load(open(path)) if os.path.exists(path) else {}
    out["n"] = n

    circ = None
    if not (a.skip_a and a.skip_b):
        print(f"Building LNN QFT circuit for n={n} ...")
        t = time.time()
        circ = qft_lnn(n)
        out["gates"], out["two_qubit_gates"] = len(circ), circ.num_nonlocal_gates()
        print(f"  {len(circ):,} gates ({circ.num_nonlocal_gates():,} two-qubit), built in {time.time() - t:.1f}s")
        print(f"  dense statevector would need 16 * 2^{n} bytes = "
              f"10^{n * math.log10(2) + math.log10(16) - 9:.0f} GB")

    if not a.skip_a:
        run_a(n, np.random.default_rng(a.seed), circ, out)
    run_c(np.random.default_rng(a.seed + 1), a.chi, out)
    if not a.skip_b:
        run_b(n, np.random.default_rng(a.seed + 2), a.chi, circ, out)

    out["peak_rss_MB_last_run"] = rss_mb()
    print(f"\nPeak process RSS (this run): {out['peak_rss_MB_last_run']:.0f} MB")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()

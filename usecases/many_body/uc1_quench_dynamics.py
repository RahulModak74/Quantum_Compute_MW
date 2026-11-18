#!/usr/bin/env python3
"""
Use case A (>50 qubits): real-time quench dynamics of 100-200 interacting spins.

Physics: transverse-field Ising chain, the model behind IBM's 127-qubit "utility" experiment
(Kim et al., Nature 2023, kicked Ising on heavy-hex) and behind real quantum magnets (CoNb2O6).
Start in |+>^N, switch on the Ising coupling, watch magnetisation and entanglement evolve.

  A1  N = 100 and 200, integrable case: every MPS number is checked against the EXACT free-fermion
      solution (any N, no state vector). We report the time horizon each chi reaches at < 1e-3 error.
  A2  N = 100, non-integrable (longitudinal field hz = 0.5, the standard chaotic Ising point):
      no exact answer exists; convergence in chi is shown instead.

A dense simulator needs 2^100 amplitudes (2e22 GB). Today's hardware gives noisy answers at this
width. This runs on a laptop CPU. It is the regime where IBM's 127-qubit result was matched
classically with tensor networks. Limits are reported: entanglement grows linearly in time,
so the reachable time horizon grows only logarithmically with chi.

Usage: python3 usecases/many_body/uc1_quench_dynamics.py [--quick]
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from mwqc import MPSState                                                     # noqa: E402
from mwqc.dynamics import tebd, tfim_bonds, tfim_exact_quench                 # noqa: E402
from mwqc.gates import I2, X, Z                                               # noqa: E402

XI, ZZ = np.kron(X, I2), np.kron(Z, Z)


def run(N, J, g, hz, chi, dt, T, every):
    st = MPSState.from_bloch_angles([np.pi / 2] * N, [0.0] * N, chi_max=chi)
    terms = tfim_bonds(N, J, g, hz)
    rec = {"t": [], "X": [], "ZZ": [], "S_mid": [], "fid": [], "bond": []}

    def cb(step, s):
        rec["t"].append(step * dt)
        rec["X"].append([s.expect_2site(XI, i).real for i in range(N - 1)] + [s.bloch_vector(N - 1)[0]])
        rec["ZZ"].append([s.expect_2site(ZZ, i).real for i in range(N - 1)])
        rec["S_mid"].append(s.entanglement_entropy(N // 2 - 1))
        rec["fid"].append(s.fidelity_estimate)
        rec["bond"].append(max(s.bond_dims()))
    t0 = time.time()
    tebd(st, terms, dt, int(round(T / dt)), callback=cb, every=every)
    rec["seconds"] = time.time() - t0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    dt, T, every = 0.02, (2.0 if a.quick else 8.0), 10
    chis = (8, 16, 32) if a.quick else (16, 32, 64)
    out = {"A1": [], "A2": []}
    J, g = 1.0, 0.5
    for N in ((100,) if a.quick else (100, 200)):
        ex = None
        for chi in chis:
            r = run(N, J, g, 0.0, chi, dt, T, every)
            if ex is None:
                ex = tfim_exact_quench(N, J, g, r["t"])
            errX = np.max(np.abs(np.array(r["X"]) - ex["X"]), axis=1)
            errZ = np.max(np.abs(np.array(r["ZZ"]) - ex["ZZ"]), axis=1)
            err = np.maximum(errX, errZ)
            bad = np.where(err >= 1e-3)[0]
            horizon = r["t"][-1] if len(bad) == 0 else (r["t"][bad[0] - 1] if bad[0] > 0 else 0.0)
            row = {"N": N, "chi": chi, "seconds": r["seconds"], "max_err_all_sites": float(err.max()),
                   "time_horizon_err_lt_1e-3": float(horizon), "final_bond": r["bond"][-1],
                   "final_S_mid_bits": r["S_mid"][-1], "final_fidelity_estimate": r["fid"][-1],
                   "t": r["t"], "err_t": err.tolist(), "S_mid_t": r["S_mid"],
                   "X_center_mps": [x[N // 2] for x in r["X"]], "X_center_exact": ex["X"][:, N // 2].tolist()}
            out["A1"].append(row)
            print(f"[A1] N={N} chi={chi:>3}: max |error| over all sites & times {err.max():.1e}; "
                  f"exact to <1e-3 up to t={horizon:.2f} (of {T}); S_mid(T)={r['S_mid'][-1]:.2f} bits; "
                  f"{r['seconds']:.0f}s")
    # A2: non-integrable
    N, g, hz = 100, 1.05, 0.5
    T2 = 2.0 if a.quick else 4.0
    runs = {}
    for chi in chis + ((128,) if not a.quick else ()):
        runs[chi] = run(N, 1.0, g, hz, chi, dt, T2, every)
        print(f"[A2] N={N} chaotic Ising chi={chi:>3}: S_mid(T)={runs[chi]['S_mid'][-1]:.2f} bits, "
              f"fidelity est {runs[chi]['fid'][-1]:.4f}, {runs[chi]['seconds']:.0f}s")
    ref = max(runs)
    for chi, r in runs.items():
        d = np.max(np.abs(np.array(r["X"]) - np.array(runs[ref]["X"])), axis=1)
        out["A2"].append({"chi": chi, "max_diff_vs_chi_max": float(d.max()), "seconds": r["seconds"],
                          "final_fidelity_estimate": r["fid"][-1], "t": r["t"], "diff_t": d.tolist(),
                          "S_mid_t": r["S_mid"]})
        if chi != ref:
            print(f"[A2] chi={chi:>3} vs chi={ref}: max |d<X_i>| {d.max():.1e}")
    p = os.path.join(HERE, "..", "..", "results", "uc1_quench_dynamics.json")
    json.dump(out, open(p, "w"))
    print(f"saved {os.path.normpath(p)}")


if __name__ == "__main__":
    main()

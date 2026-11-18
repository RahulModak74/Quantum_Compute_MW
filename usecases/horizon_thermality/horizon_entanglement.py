#!/usr/bin/env python3
"""
Horizon thermality from vacuum entanglement.

The VACUUM of a quantum field, restricted to one side of a boundary, is a THERMAL state:
  * Bisognano-Wichmann theorem / Unruh effect: the reduced state of half-space is
        rho_A ~ exp(-2 pi K_boost / v),
    i.e. a local temperature T(d) = v / (2 pi d) at distance d from the boundary. This is the
    flat-space twin of Hawking radiation (same derivation, horizon = boundary).
  * Entanglement entropy obeys an area law, with a logarithmic correction set by the central charge c
    (Calabrese-Cardy), the lattice analogue of black-hole entropy counting.

Nothing thermal goes in. Input: a Hamiltonian (critical transverse-field Ising chain = free Majorana
field, c = 1/2, velocity v = 2J). Output: a temperature profile and an entropy law, checked against CFT.

  H1  exact (Gaussian / free-fermion) entanglement Hamiltonian of half of an N = 40 chain (N = 24 with --quick):
      local inverse temperature beta(x) extracted site by site, compared with the CFT/BW prediction
      beta(x) = (2N/v) [cos(pi x/N) - cos(pi l/N)] / sin(pi l/N)  -> 2 pi d / v near the cut.
  H2  entanglement entropy S(l) for every cut: fitted central charge vs the exact c = 1/2.
  H3  the same vacuum computed independently on the mwqc MPS engine (imaginary-time TEBD),
      N = 100: entanglement entropy and Schmidt spectrum compared with the exact solution.

Usage: python3 usecases/horizon_thermality/horizon_entanglement.py [--quick]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from mwqc import MPSState                                                       # noqa: E402
from mwqc.dynamics import ground_state, tfim_bonds, tfim_majorana_matrix        # noqa: E402


def ground_covariance(N, J=1.0, g=1.0):
    """Majorana covariance Gamma_ab = (i/2)<[g_a, g_b]> of the ground state (checked vs dense ED)."""
    A = tfim_majorana_matrix(N, J, g)
    w, U = np.linalg.eigh(1j * A)
    return (1j * (U @ np.diag(np.sign(w)) @ U.conj().T)).real, A


def block_nu(G, l):
    GA = G[:2 * l, :2 * l]
    try:
        nu, V = np.linalg.eigh(1j * GA)
    except np.linalg.LinAlgError:             # rare LAPACK non-convergence: use a different driver
        import scipy.linalg as sla
        nu, V = sla.eigh(1j * GA, driver="evr")
    return nu, V


def entanglement_hamiltonian_mp(N, l, J=1.0, g=1.0, dps=60):
    """Entanglement Hamiltonian in extended precision (mpmath). Needed because deep modes have
    nu = 1 - 1e-30 and beyond: double precision cannot resolve their entanglement energies, and
    clipping them corrupts K everywhere (checked: rho = exp(-H_E)/Z to 3e-15 at N = 8)."""
    import mpmath as mp
    mp.mp.dps = dps
    A = tfim_majorana_matrix(N, J, g)
    w, U = mp.eighe(mp.matrix((1j * A).tolist()))
    G = (U * mp.diag([mp.sign(x) for x in w]) * U.H) * 1j
    nu, V = mp.eighe(G[0:2 * l, 0:2 * l] * 1j)
    K = (V * mp.diag([-2 * mp.atanh(x) for x in nu]) * V.H) * (-1j)
    return np.array([[float(mp.re(K[i, j])) for j in range(2 * l)] for i in range(2 * l)]), A


def entropy_bits(nu):
    p = np.clip((1 + nu[nu > 0]) / 2, 1e-300, 1)
    q = np.clip(1 - p, 1e-300, 1)
    return float(np.sum(-p * np.log2(p) - q * np.log2(q)))


def entanglement_hamiltonian(G, l):
    """K with rho_A ~ exp(-(i/4) gamma^T K gamma): iK = V diag(-2 artanh nu) V^dagger."""
    nu, V = block_nu(G, l)
    nu = np.clip(nu, -1 + 1e-15, 1 - 1e-15)
    iK = V @ np.diag(-2 * np.arctanh(nu)) @ V.conj().T
    return (-1j * iK).real


def schmidt_from_nu(nu, top=12, branch=20):
    """Largest Schmidt weights of a Gaussian state from its single-mode nu_k > 0.
    Each mode contributes a factor (1+nu)/2 or (1-nu)/2; branch over the most MIXED modes
    (smallest nu), and multiply in the near-pure ones at their dominant factor."""
    nk = np.sort(nu[nu > 0])                       # most mixed first
    pk = (1 + nk) / 2
    probs = [1.0]
    for p in pk[:branch]:
        probs = sorted([x * p for x in probs] + [x * (1 - p) for x in probs], reverse=True)[:512]
    rest = float(np.prod(pk[branch:])) if len(pk) > branch else 1.0
    return np.array(probs[:top]) * rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = {}
    J = g = 1.0
    v = 2 * J

    # ── H1: Bisognano-Wichmann temperature profile (extended precision) ──
    N = 24 if a.quick else 40
    l = N // 2
    K, A = entanglement_hamiltonian_mp(N, l, J, g)
    rows = []
    for i in range(l):
        for kind, (p, q), x in (("site", (2 * i, 2 * i + 1), i + 0.5), ("bond", (2 * i + 1, 2 * i + 2), i + 1.0)):
            if q >= 2 * l:
                continue
            beta_fit = K[p, q] / A[p, q]
            beta_cft = (2 * N / v) * (np.cos(np.pi * x / N) - np.cos(np.pi * l / N)) / np.sin(np.pi * l / N)
            rows.append({"kind": kind, "x": x, "d_from_cut": l - x, "beta_fit": beta_fit, "beta_cft": beta_cft,
                         "beta_unruh": 2 * np.pi * (l - x) / v})
    near = [r for r in rows if r["d_from_cut"] <= 5]
    rel = np.array([r["beta_fit"] / r["beta_cft"] - 1 for r in near])
    out["H1"] = {"N": N, "cut": l, "profile": rows,
                 "near_cut_rel_dev_mean": float(np.mean(np.abs(rel))),
                 "near_cut_rel_dev_max": float(np.max(np.abs(rel))),
                 "unruh_slope_fit": float(np.polyfit([r["d_from_cut"] for r in near],
                                                     [r["beta_fit"] for r in near], 1)[0]),
                 "unruh_slope_theory_2pi_over_v": 2 * np.pi / v}
    print(f"[H1] N={N}, half-chain vacuum. Local inverse temperature beta(d) from the exact entanglement Hamiltonian:")
    print(f"     {'d':>5} {'term':<5} {'beta (computed)':>16} {'Unruh 2 pi d/v':>15} {'finite-size CFT':>16}")
    for r in rows[::-1][:10]:
        print(f"     {r['d_from_cut']:5.1f} {r['kind']:<5} {r['beta_fit']:16.4f} {r['beta_unruh']:15.4f} {r['beta_cft']:16.4f}")
    print(f"     d <= 5: mean |rel. deviation from CFT| {100 * out['H1']['near_cut_rel_dev_mean']:.2f} %, "
          f"max {100 * out['H1']['near_cut_rel_dev_max']:.2f} %;  slope d(beta)/dd = "
          f"{out['H1']['unruh_slope_fit']:.4f} vs 2 pi / v = {np.pi:.4f}")

    # ── H2: entanglement entropy, central charge ─────────────────────────
    N = 200
    l = N // 2
    G, _ = ground_covariance(N, J, g)
    ls = np.arange(4, N - 3)
    S = np.array([entropy_bits(block_nu(G, k)[0]) for k in ls])
    xcc = np.log2((2 * N / np.pi) * np.sin(np.pi * ls / N))     # open chain: S = (c/6) log(...) + const
    # even/odd lattice oscillations are known for open chains: fit on even l
    ev = ls % 2 == 0
    c_fit = 6 * np.polyfit(xcc[ev], S[ev], 1)[0]
    out["H2"] = {"c_fit": float(c_fit), "c_exact": 0.5, "S_mid_bits": float(S[ls == l][0])}
    print(f"[H2] entanglement entropy S(l), l = 4..{N - 4}: fitted central charge c = {c_fit:.4f} (exact 1/2)")

    # ── H3: the same vacuum on the mwqc MPS engine ───────────────────────
    Nm = 60 if a.quick else 100
    Gm, _ = ground_covariance(Nm, J, g)
    st = MPSState.from_bloch_angles([np.pi / 2] * Nm, [0.0] * Nm, chi_max=32)
    t0 = time.time()
    ground_state(st, tfim_bonds(Nm, J, g))
    S_mps = np.array([st.entanglement_entropy(b) for b in range(Nm - 1)])
    S_ex = np.array([entropy_bits(block_nu(Gm, b + 1)[0]) for b in range(Nm - 1)])
    st._move_center(Nm // 2 - 1)
    aM = st.A[Nm // 2 - 1]
    sv = np.linalg.svd(aM.reshape(-1, aM.shape[2]), compute_uv=False) ** 2
    sv_ex = schmidt_from_nu(block_nu(Gm, Nm // 2)[0], top=8)
    out["H3"] = {"N": Nm, "seconds": time.time() - t0, "max_abs_entropy_err_bits": float(np.max(np.abs(S_mps - S_ex))),
                 "schmidt_top8_mps": sv[:8].tolist(), "schmidt_top8_exact": sv_ex.tolist()}
    print(f"[H3] mwqc MPS vacuum, N={Nm}: entropy at every cut vs exact, max |error| "
          f"{out['H3']['max_abs_entropy_err_bits']:.1e} bits ({out['H3']['seconds']:.0f}s)")
    print("     top Schmidt weights (entanglement spectrum)  MPS: " + " ".join(f"{x:.5f}" for x in sv[:6]))
    print("                                                 exact: " + " ".join(f"{x:.5f}" for x in sv_ex[:6]))
    p = os.path.join(HERE, "..", "..", "results", "horizon_thermality.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"saved {os.path.normpath(p)}")


if __name__ == "__main__":
    main()

"""
Many-body dynamics and ground states on MPS (TEBD), plus exact references.

    terms = tfim_bonds(N, J=1, g=1)               # list of 4x4 bond Hamiltonians
    st    = MPSState.from_bloch_angles(...)       # or MPSState(N)
    tebd(st, terms, dt=0.05, steps=40)            # real time:   exp(-i H t)
    tebd(st, terms, dt=0.05, steps=400, imaginary=True)   # ground state: exp(-tau H), renormalised
    bond_energy(st, terms)                        # <H>

Exact references (any N, no state vector):
    tfim_exact_ground_energy(N, J, g)             # free fermions (Jordan-Wigner -> Majoranas)
    tfim_exact_quench(N, J, g, times)             # <X_i(t)>, <Z_i Z_i+1(t)> from |+...+>

Model conventions (open boundary chain of N spin-1/2 sites):
    TFIM:        H = -J sum_i Z_i Z_{i+1} - g sum_i X_i - hz sum_i Z_i   (hz != 0 breaks integrability)
    Heisenberg:  H =  J sum_i (S^x S^x + S^y S^y + Delta S^z S^z),  S = sigma/2
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import scipy.linalg as sla

from .gates import I2, X, Y, Z
from .mps import MPSState


def _kron(a, b):
    return np.kron(a, b)


def tfim_bonds(N: int, J: float = 1.0, g: float = 1.0, hz: float = 0.0) -> List[np.ndarray]:
    """Bond terms h_b on (b, b+1); on-site fields split so each site gets its full field once."""
    terms = []
    for b in range(N - 1):
        wl = 1.0 if b == 0 else 0.5
        wr = 1.0 if b == N - 2 else 0.5
        h = -J * _kron(Z, Z)
        h += -g * (wl * _kron(X, I2) + wr * _kron(I2, X))
        h += -hz * (wl * _kron(Z, I2) + wr * _kron(I2, Z))
        terms.append(h)
    return terms


def heisenberg_bonds(N: int, J: float = 1.0, delta: float = 1.0) -> List[np.ndarray]:
    h = 0.25 * J * (_kron(X, X) + _kron(Y, Y) + delta * _kron(Z, Z))
    return [h.copy() for _ in range(N - 1)]


def bond_energy(st: MPSState, terms) -> float:
    return float(sum(st.expect_2site(h, b).real for b, h in enumerate(terms)))


def tebd(st: MPSState, terms, dt: float, steps: int, imaginary: bool = False,
         callback: Optional[Callable] = None, every: int = 1):
    """Second-order Trotter: exp(-H_even dt/2) exp(-H_odd dt) exp(-H_even dt/2) per step.
    Real time uses exp(-i h dt); imaginary time uses exp(-h dt) (state renormalised by the SVD).
    Trotter error per unit time is O(dt^2)."""
    f = (lambda h, t: sla.expm(-h * t)) if imaginary else (lambda h, t: sla.expm(-1j * h * t))
    half = {b: f(h, dt / 2).reshape(2, 2, 2, 2) for b, h in enumerate(terms) if b % 2 == 0}
    full = {b: f(h, dt).reshape(2, 2, 2, 2) for b, h in enumerate(terms) if b % 2 == 1}
    for s in range(steps):
        for b in sorted(half):
            st._apply_adjacent(half[b], b)
        for b in sorted(full):
            st._apply_adjacent(full[b], b)
        for b in sorted(half):
            st._apply_adjacent(half[b], b)
        if callback is not None and (s + 1) % every == 0:
            callback(s + 1, st)
    return st


# ── exact free-fermion solution of the TFIM ──────────────────────────────────
# Majoranas: gamma_{2i} = (prod_{j<i} X_j) Z_i,  gamma_{2i+1} = (prod_{j<i} X_j) Y_i
#   X_i = i gamma_{2i} gamma_{2i+1},   Z_i Z_{i+1} = i gamma_{2i+1} gamma_{2i+2}
# H = (i/4) sum_ab A_ab gamma_a gamma_b  with A real antisymmetric.
def tfim_majorana_matrix(N: int, J: float = 1.0, g: float = 1.0) -> np.ndarray:
    A = np.zeros((2 * N, 2 * N))
    for i in range(N):                    # -g X_i = -g i g2i g2i+1
        A[2 * i, 2 * i + 1] += -2 * g
        A[2 * i + 1, 2 * i] -= -2 * g
    for i in range(N - 1):                # -J Z_i Z_i+1 = -J i g2i+1 g2i+2
        A[2 * i + 1, 2 * i + 2] += -2 * J
        A[2 * i + 2, 2 * i + 1] -= -2 * J
    return A


def tfim_exact_ground_energy(N: int, J: float = 1.0, g: float = 1.0) -> float:
    A = tfim_majorana_matrix(N, J, g)
    ev = np.linalg.eigvalsh(1j * A)       # +/- eps_k
    return float(-0.25 * np.sum(np.abs(ev)))


def tfim_exact_quench(N: int, J: float, g: float, times) -> dict:
    """Start in |+>^N (all X = +1), evolve with H(J, g). Returns arrays over times x sites."""
    A = tfim_majorana_matrix(N, J, g)
    M0 = np.zeros((2 * N, 2 * N))
    for i in range(N):                    # M_ab = (i/2)<[g_a, g_b]>;  <X_i> = M_{2i,2i+1}
        M0[2 * i, 2 * i + 1], M0[2 * i + 1, 2 * i] = 1.0, -1.0
    xs, zzs = [], []
    for t in times:
        R = sla.expm(A * t)
        M = R @ M0 @ R.T
        xs.append([M[2 * i, 2 * i + 1] for i in range(N)])
        zzs.append([M[2 * i + 1, 2 * i + 2] for i in range(N - 1)])
    return {"X": np.array(xs), "ZZ": np.array(zzs)}


def dense_hamiltonian(terms, N: int) -> np.ndarray:
    """Dense H from bond terms (validation only, N <= 14)."""
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for b, h in enumerate(terms):
        H += np.kron(np.kron(np.eye(2 ** b), h), np.eye(2 ** (N - b - 2)))
    return H   # basis ordering: site 0 most significant


# total imaginary time ~85: enough for gaps >~0.03 (critical chains of ~100 sites); start from a
# state with good overlap (|+...+> for the TFIM). Measured: N=100 critical TFIM to 5.6e-7 abs.
DEFAULT_SCHEDULE = ((0.1, 600), (0.05, 300), (0.02, 300), (0.01, 300), (0.005, 200))


def ground_state(st: MPSState, terms, schedule=DEFAULT_SCHEDULE, verbose: bool = False):
    """Imaginary-time TEBD with a shrinking step (removes the O(tau^2) Trotter bias).
    Returns the list of (tau, energy) after each stage."""
    hist = []
    for tau, n in schedule:
        tebd(st, terms, tau, n, imaginary=True)
        e = bond_energy(st, terms)
        hist.append((tau, e))
        if verbose:
            print(f"    tau={tau:<6} steps={n:<4} E={e:.8f}  max bond {max(st.bond_dims())}")
    return hist

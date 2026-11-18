"""
Circuit library (mirrors a slice of qiskit.circuit.library) plus the closed-form
QFT for basis-state inputs.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence

import numpy as np

from . import gates as G
from .circuit import MWCircuit


# ── QFT ──────────────────────────────────────────────────────────────────────
def qft(n: int, do_swaps: bool = True, approximation_degree: int = 0) -> MWCircuit:
    """Textbook QFT, same unitary as qiskit.circuit.library.QFT(n) (little-endian):
        QFT|j> = 2^{-n/2} sum_k exp(2 pi i j k / 2^n) |k>
    Long-range controlled-phase gates; fine for the dense backend, slow for MPS at
    large n (each long-range gate becomes a SWAP chain). Use qft_lnn() for MPS.
    """
    qc = MWCircuit(n, name=f"qft_{n}")
    for j in reversed(range(n)):
        qc.h(j)
        for k in reversed(range(j)):
            d = j - k
            if approximation_degree and d > n - approximation_degree:
                continue
            qc.cp(math.pi / 2 ** d, k, j)
    if do_swaps:
        for q in range(n // 2):
            qc.swap(q, n - 1 - q)
    return qc


def qft_lnn(n: int, approximation_cutoff: float = 0.0) -> MWCircuit:
    """Same unitary as qft(n), but written for a linear nearest-neighbour chain
    (Fowler, Devitt & Hollenberg 2004): n(n-1)/2 merged CP*SWAP gates, all between
    adjacent qubits. The SWAP network also performs the final bit reversal.
    This is what makes a 1000-qubit QFT run in about a minute on an MPS.

    approximation_cutoff: drop the phase (keep the SWAP) when the angle is below it
    — Coppersmith's approximate QFT. 0 keeps every gate exactly.
    """
    qc = MWCircuit(n, name=f"qft_lnn_{n}")
    site = lambda pos: n - 1 - pos          # big-endian line position -> qubit index
    swap4 = G.SWAP.reshape(4, 4)
    for i in range(n):
        qc.h(site(0))
        for k in range(n - 1 - i):
            ang = math.pi / 2 ** (k + 1)
            cp4 = G.cp(ang).reshape(4, 4) if ang >= approximation_cutoff else np.eye(4)
            m = swap4 @ cp4
            qc.unitary(m, (site(k), site(k + 1)))
    return qc


def qft_basis_state_bloch(j: int, n: int):
    """Closed form for QFT|j> (j an n-bit integer, any n, exact integer arithmetic).

    QFT|j> is a PRODUCT state (Nielsen & Chuang eq. 5.4):
        qubit q  ->  (|0> + exp(2 pi i phi_q)|1>)/sqrt(2),   phi_q = frac(j / 2^(n-q))
    so every output qubit sits on the Bloch equator: theta = pi/2, phi = 2 pi phi_q.

    This is why a 1000-qubit QFT on a basis input needs no state vector: the answer is
    known in closed form and needs no entanglement. Returns (theta[], phi[]) radians.
    """
    j = int(j)
    theta = np.full(n, math.pi / 2)
    phi = np.empty(n)
    for q in range(n):
        m = n - q
        frac = Fraction(j % (1 << m), 1 << m)
        phi[q] = 2 * math.pi * float(frac)
    return theta, phi


# ── standard states / ansatze ────────────────────────────────────────────────
def ghz(n: int) -> MWCircuit:
    qc = MWCircuit(n, name=f"ghz_{n}")
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    return qc


def hardware_efficient(n: int, reps: int, params: Sequence[float],
                       initial_bits: Sequence[int] = (), entanglement: str = "linear") -> MWCircuit:
    """RY layers + CX ladders (qiskit TwoLocal('ry','cx') shape). Param count n*(reps+1)."""
    params = list(params)
    need = n * (reps + 1)
    if len(params) != need:
        raise ValueError(f"need {need} params, got {len(params)}")
    qc = MWCircuit(n, name="hea")
    for q, b in enumerate(initial_bits):
        if b:
            qc.x(q)
    it = iter(params)
    for r in range(reps + 1):
        for q in range(n):
            qc.ry(next(it), q)
        if r < reps:
            pairs = [(q, q + 1) for q in range(n - 1)]
            if entanglement == "circular" and n > 2:
                pairs.append((n - 1, 0))
            for a, b in pairs:
                qc.cx(a, b)
    return qc


def random_circuit(n: int, depth: int, seed: int = 0, two_qubit: str = "cx",
                   local: bool = True) -> MWCircuit:
    """Brick-wall random circuit (random 1q rotations + entanglers). Used to show where
    MPS stops being cheap: entanglement grows linearly with depth."""
    rng = np.random.default_rng(seed)
    qc = MWCircuit(n, name=f"rand_{n}x{depth}")
    for d in range(depth):
        for q in range(n):
            qc.u(*rng.uniform(0, 2 * np.pi, 3), q)
        start = d % 2
        for q in range(start, n - 1, 2):
            if local:
                qc.cx(q, q + 1) if two_qubit == "cx" else qc.cz(q, q + 1)
            else:
                b = int(rng.integers(0, n))
                if b != q:
                    qc.cx(q, b)
    return qc

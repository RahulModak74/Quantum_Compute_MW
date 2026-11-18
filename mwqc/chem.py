"""
Quantum-chemistry layer — the Qiskit-Nature slice of mwqc.

    prob = MolecularProblem.from_pyscf("H 0 0 0; H 0 0 0.735", basis="sto-3g")
    H    = jordan_wigner(prob)                   # PauliSum on 2*n_orb qubits
    res  = vqe_uccsd(prob, H)                    # VQE with a UCCSD ansatz
    res.energy, prob.e_fci, prob.e_hf

Unlike the earlier `mw_quantum.MWDriver`, nothing here is pre-tabulated: integrals
come from PySCF, the qubit Hamiltonian is built by an explicit Jordan-Wigner
transform, and the exact (FCI) answer is computed independently so every VQE
number can be checked.

Qubit layout (same as qiskit-nature's default "blocked" ordering):
    alpha spin-orbital p -> qubit p,   beta spin-orbital p -> qubit p + n_orb
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional

import numpy as np

from .pauli import PauliSum


@dataclass
class MolecularProblem:
    atom: str
    basis: str
    n_orb: int
    n_alpha: int
    n_beta: int
    h1: np.ndarray          # (n_orb, n_orb) MO one-electron integrals
    eri: np.ndarray         # (n_orb,)*4 MO two-electron integrals, chemist notation (pq|rs)
    e_nuc: float
    e_hf: float
    e_fci: Optional[float]
    mo_energy: np.ndarray
    orbitals: str = "hf"
    e_ks: Optional[float] = None
    meta: dict = field(default_factory=dict)

    @property
    def num_qubits(self) -> int:
        return 2 * self.n_orb

    @classmethod
    def from_pyscf(cls, atom: str, basis: str = "sto-3g", charge: int = 0, spin: int = 0,
                   orbitals: str = "hf", xc: str = "pbe", fci: bool = True):
        """orbitals='hf' uses RHF canonical orbitals; 'ks' uses Kohn-Sham orbitals (xc functional).
        The FCI energy is basis-invariant, so both give the same exact answer; the reference
        determinant (and thus VQE's starting point) differs."""
        from pyscf import ao2mo, dft, fci as pyfci, gto, scf
        mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)
        mf = scf.RHF(mol).run()
        e_ks = None
        C, mo_e = mf.mo_coeff, mf.mo_energy
        if orbitals == "ks":
            ks = dft.RKS(mol)
            ks.xc = xc
            ks.run()
            e_ks = float(ks.e_tot)
            C, mo_e = ks.mo_coeff, ks.mo_energy
        h1 = C.T @ (mol.intor("int1e_kin") + mol.intor("int1e_nuc")) @ C
        n = C.shape[1]
        eri = ao2mo.restore(1, ao2mo.kernel(mol, C), n)
        na, nb = mol.nelec
        e_fci = None
        if fci and n <= 14:
            e_fci = float(pyfci.FCI(mf).kernel()[0])
        return cls(atom, basis, n, na, nb, h1, eri, float(mol.energy_nuc()), float(mf.e_tot),
                   e_fci, np.asarray(mo_e), orbitals, e_ks)

    def hf_bits(self) -> List[int]:
        bits = [0] * self.num_qubits
        for p in range(self.n_alpha):
            bits[p] = 1
        for p in range(self.n_beta):
            bits[p + self.n_orb] = 1
        return bits


# ── Jordan-Wigner ────────────────────────────────────────────────────────────
def jordan_wigner(prob: MolecularProblem, atol: float = 1e-12) -> PauliSum:
    """Second-quantised electronic Hamiltonian -> qubit PauliSum (includes E_nuc)."""
    n, nq = prob.n_orb, prob.num_qubits
    cre = [_ladder_exact(nq, j, True) for j in range(nq)]
    ann = [_ladder_exact(nq, j, False) for j in range(nq)]
    H = PauliSum.identity(nq, prob.e_nuc)
    terms: dict = {}

    def acc(ps: PauliSum, c):
        for k, v in ps.terms.items():
            terms[k] = terms.get(k, 0) + c * v

    for p in range(n):
        for q in range(n):
            if abs(prob.h1[p, q]) < atol:
                continue
            for sp in (0, n):
                acc(cre[p + sp] * ann[q + sp], prob.h1[p, q])
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    g = prob.eri[p, q, r, s]
                    if abs(g) < atol:
                        continue
                    for sa in (0, n):
                        for sb in (0, n):
                            P, Q, R, S = p + sa, q + sa, r + sb, s + sb
                            if P == R or S == Q:
                                continue          # a_P^+ a_P^+ = 0
                            acc(cre[P] * cre[R] * ann[S] * ann[Q], 0.5 * g)
    for k, v in terms.items():
        H.terms[k] = H.terms.get(k, 0) + v
    return H.simplify(atol)


def _ladder_exact(nq: int, j: int, dagger: bool) -> PauliSum:
    zmask = (1 << j) - 1
    x = 1 << j
    X = PauliSum(nq, {(x, 0): 1.0})
    Y = PauliSum(nq, {(x, x): 1.0})        # P = i^{x&z} X Z = Y on qubit j
    Zs = PauliSum(nq, {(0, zmask): 1.0})
    op = X * 0.5 + Y * (-0.5j if dagger else 0.5j)
    return Zs * op


# ── exact reference in the right particle-number sector ──────────────────────
def sector_indices(prob: MolecularProblem) -> np.ndarray:
    n, nq = prob.n_orb, prob.num_qubits
    k = np.arange(1 << nq, dtype=np.uint64)
    amask = np.uint64((1 << n) - 1)
    na = np.bitwise_count(k & amask)
    nb = np.bitwise_count(k >> np.uint64(n))
    return np.where((na == prob.n_alpha) & (nb == prob.n_beta))[0]


def exact_ground_energy(H: PauliSum, prob: MolecularProblem) -> float:
    """Lowest eigenvalue of the qubit Hamiltonian restricted to (N_alpha, N_beta)."""
    import scipy.sparse.linalg as sla
    M = H.to_sparse()
    idx = sector_indices(prob)
    sub = M[idx][:, idx]
    if sub.shape[0] <= 400:
        return float(np.linalg.eigvalsh(sub.toarray()).min())
    return float(sla.eigsh(sub, k=1, which="SA")[0][0])


# ── UCCSD-VQE (statevector) ──────────────────────────────────────────────────
def uccsd_generators(prob: MolecularProblem) -> List[PauliSum]:
    """Anti-Hermitian generators A = T - T^dagger for spin-conserving singles and doubles."""
    n, nq = prob.n_orb, prob.num_qubits
    cre = [_ladder_exact(nq, j, True) for j in range(nq)]
    ann = [_ladder_exact(nq, j, False) for j in range(nq)]
    occ = [p for p in range(prob.n_alpha)] + [p + n for p in range(prob.n_beta)]
    vir = [p for p in range(prob.n_alpha, n)] + [p + n for p in range(prob.n_beta, n)]
    spin = lambda q: q >= n
    gens = []
    for i in occ:
        for a in vir:
            if spin(i) == spin(a):
                T = cre[a] * ann[i]
                gens.append((T + T_dag(T) * -1).simplify())
    for x, i in enumerate(occ):
        for j in occ[x + 1:]:
            for y, a in enumerate(vir):
                for b in vir[y + 1:]:
                    if sorted([spin(i), spin(j)]) != sorted([spin(a), spin(b)]):
                        continue
                    T = cre[a] * cre[b] * ann[j] * ann[i]
                    g = (T + T_dag(T) * -1).simplify()
                    if len(g):
                        gens.append(g)
    return gens


def T_dag(ps: PauliSum) -> PauliSum:
    # Pauli strings are Hermitian, so the adjoint just conjugates coefficients
    return PauliSum(ps.num_qubits, {k: np.conj(c) for k, c in ps.terms.items()})


@dataclass
class VQEResult:
    energy: float
    params: np.ndarray
    n_params: int
    n_evals: int
    seconds: float
    history: list


def vqe_uccsd(prob: MolecularProblem, H: PauliSum, maxiter: int = 200, verbose: bool = False) -> VQEResult:
    """Trotterised UCCSD, psi(theta) = prod_k exp(theta_k A_k)|HF>, minimised with BFGS.
    Restricted to the (N_alpha, N_beta) sector for speed (the generators conserve it)."""
    from scipy.optimize import minimize
    from scipy.sparse.linalg import expm_multiply

    t0 = time.time()
    idx = sector_indices(prob)
    pos = {int(k): i for i, k in enumerate(idx)}
    Hs = H.to_sparse()[idx][:, idx].tocsr()
    gens = [g.to_sparse()[idx][:, idx].tocsc() for g in uccsd_generators(prob)]
    hf = sum(b << q for q, b in enumerate(prob.hf_bits()))
    psi0 = np.zeros(len(idx), dtype=complex)
    psi0[pos[hf]] = 1.0
    hist = []

    def state(th):
        psi = psi0
        for t, A in zip(th, gens):
            if t != 0.0:
                psi = expm_multiply(A * t, psi)
        return psi

    def energy(th):
        psi = state(th)
        e = float(np.vdot(psi, Hs @ psi).real)
        hist.append(e)
        return e

    def grad(th):
        # exact gradient by the product rule: d/dt_k = 2 Re <psi| H U_{>k} A_k U_{<=k} |0>
        fwd = [psi0]
        psi = psi0
        for t, A in zip(th, gens):
            psi = expm_multiply(A * t, psi) if t != 0.0 else psi
            fwd.append(psi)
        lam = Hs @ psi
        g = np.zeros(len(th))
        for k in range(len(th) - 1, -1, -1):
            g[k] = 2 * np.vdot(lam, gens[k] @ fwd[k + 1]).real
            if th[k] != 0.0:
                lam = expm_multiply(-gens[k] * th[k], lam)   # U_k^dagger = exp(-t A)
        return g

    x0 = np.zeros(len(gens))
    r = minimize(energy, x0, jac=grad, method="BFGS", options={"maxiter": maxiter, "gtol": 1e-7})
    return VQEResult(float(r.fun), r.x, len(gens), len(hist), time.time() - t0, hist)

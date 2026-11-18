"""
MPSState — matrix-product-state simulator. This is the "MW Bloch manifold"
made rigorous.

How it relates to the original MW idea
--------------------------------------
* The MW "product of n Bloch spheres" (one (theta, phi) per qubit, 2n numbers)
  is exactly an MPS with bond dimension chi = 1, i.e. a product state.
* The MW "fibre-bundle chain" (qubit k conditioned on qubits 0..k-1) is what an
  MPS bond does: tensor A[k] carries chi-dimensional context from the left.
  chi = 1 -> no entanglement; chi = 2**(n/2) -> any state at all.
* Memory is  sum_k chi_l * 2 * chi_r * 16 bytes,  linear in n for bounded chi.
  The price: chi must grow with entanglement. When chi is capped, singular
  values are discarded and this class *records exactly how much weight was
  thrown away* (`truncation_error`, `fidelity_estimate`). That is the honest
  error bar.

Conventions: site k == qubit k. Tensors A[k] have shape (chi_left, 2, chi_right).
Mixed-canonical form is maintained around `self.center`.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .circuit import MWCircuit


class MPSState:
    def __init__(self, num_qubits: int, chi_max: int = 64, cutoff: float = 1e-12,
                 dtype=np.complex128):
        self.n = int(num_qubits)
        self.chi_max = int(chi_max)
        self.cutoff = float(cutoff)
        self.dtype = dtype
        self.A: List[np.ndarray] = []
        for _ in range(self.n):
            a = np.zeros((1, 2, 1), dtype=dtype)
            a[0, 0, 0] = 1.0
            self.A.append(a)
        self.center = 0
        self.truncation_error = 0.0      # sum of discarded squared singular values
        self.fidelity_estimate = 1.0     # prod (1 - discarded_i); standard MPS estimate
        self.n_truncations = 0
        self.max_bond_seen = 1

    # ── construction ─────────────────────────────────────────────────────
    @classmethod
    def from_bits(cls, bits: Sequence[int], **kw) -> "MPSState":
        """bits[k] is the value of qubit k."""
        st = cls(len(bits), **kw)
        for k, b in enumerate(bits):
            if b:
                st.A[k][0, 0, 0], st.A[k][0, 1, 0] = 0.0, 1.0
        return st

    @classmethod
    def from_product(cls, single_qubit_states: Sequence[np.ndarray], **kw) -> "MPSState":
        """Product state from a list of 2-vectors (one per qubit) — the chi=1 'Bloch' state."""
        st = cls(len(single_qubit_states), **kw)
        for k, v in enumerate(single_qubit_states):
            v = np.asarray(v, dtype=st.dtype)
            st.A[k] = (v / np.linalg.norm(v)).reshape(1, 2, 1)
        return st

    @classmethod
    def from_bloch_angles(cls, theta: Sequence[float], phi: Sequence[float], **kw) -> "MPSState":
        vecs = [np.array([np.cos(t / 2), np.exp(1j * f) * np.sin(t / 2)]) for t, f in zip(theta, phi)]
        return cls.from_product(vecs, **kw)

    # ── canonical-form bookkeeping ───────────────────────────────────────
    def _move_center(self, k: int):
        while self.center < k:
            i = self.center
            a = self.A[i]
            dl, d, dr = a.shape
            q, r = np.linalg.qr(a.reshape(dl * d, dr))
            self.A[i] = q.reshape(dl, d, q.shape[1])
            self.A[i + 1] = np.tensordot(r, self.A[i + 1], axes=(1, 0))
            self.center += 1
        while self.center > k:
            i = self.center
            a = self.A[i]
            dl, d, dr = a.shape
            q, r = np.linalg.qr(a.reshape(dl, d * dr).T)
            self.A[i] = q.T.reshape(q.shape[1], d, dr)
            self.A[i - 1] = np.tensordot(self.A[i - 1], r.T, axes=(2, 0))
            self.center -= 1

    # ── gates ────────────────────────────────────────────────────────────
    def apply_1q(self, g: np.ndarray, q: int):
        self.A[q] = np.einsum("st,ltr->lsr", g, self.A[q])

    def _apply_adjacent(self, g: np.ndarray, i: int):
        """g is a (2,2,2,2) tensor acting on sites (i, i+1) in that order."""
        self._move_center(i)
        a, b = self.A[i], self.A[i + 1]
        dl, dr = a.shape[0], b.shape[2]
        theta = np.tensordot(a, b, axes=(2, 0))                    # (dl,2,2,dr)
        theta = np.einsum("stuv,luvr->lstr", g, theta)
        m = theta.reshape(dl * 2, 2 * dr)
        try:
            U, S, Vh = np.linalg.svd(m, full_matrices=False)
        except np.linalg.LinAlgError:
            U, S, Vh = _svd_fallback(m)
        norm2 = float(np.sum(S ** 2))
        keep = len(S)
        if self.cutoff > 0:
            keep = max(1, int(np.sum(S ** 2 > self.cutoff * norm2)))
        keep = min(keep, self.chi_max)
        discarded = float(np.sum(S[keep:] ** 2)) / norm2 if norm2 > 0 else 0.0
        if keep < len(S) and discarded > 0:
            self.truncation_error += discarded
            self.fidelity_estimate *= (1.0 - discarded)
            self.n_truncations += 1
        U, S, Vh = U[:, :keep], S[:keep], Vh[:keep]
        S = S / np.sqrt(np.sum(S ** 2))
        self.A[i] = U.reshape(dl, 2, keep)
        self.A[i + 1] = (S[:, None] * Vh).reshape(keep, 2, dr)
        self.center = i + 1
        if keep > self.max_bond_seen:
            self.max_bond_seen = keep

    def apply_2q(self, g: np.ndarray, qa: int, qb: int):
        g = np.asarray(g).reshape(2, 2, 2, 2)
        if qa > qb:
            qa, qb = qb, qa
            g = g.transpose(1, 0, 3, 2)
        if qb == qa + 1:
            self._apply_adjacent(g, qa)
            return
        # route qb next to qa with nearest-neighbour SWAPs, apply, route back
        from .gates import SWAP
        for k in range(qb - 1, qa, -1):
            self._apply_adjacent(SWAP, k)
        self._apply_adjacent(g, qa)
        for k in range(qa + 1, qb):
            self._apply_adjacent(SWAP, k)

    def apply(self, g: np.ndarray, qubits):
        if len(qubits) == 1:
            self.apply_1q(np.asarray(g).reshape(2, 2), qubits[0])
        elif len(qubits) == 2:
            self.apply_2q(g, qubits[0], qubits[1])
        else:
            raise NotImplementedError("decompose 3+ qubit gates first")
        return self

    def run(self, circuit: MWCircuit):
        if circuit.num_qubits != self.n:
            raise ValueError("circuit width != state width")
        for ins in circuit.data:
            self.apply(ins.tensor(), ins.qubits)
        return self

    # ── observables ──────────────────────────────────────────────────────
    def norm(self) -> float:
        self._move_center(self.center)
        return float(np.linalg.norm(self.A[self.center]))

    def reduced_density_matrix(self, q: int) -> np.ndarray:
        self._move_center(q)
        a = self.A[q]
        return np.einsum("lsr,ltr->st", a, a.conj())

    def bloch_vector(self, q: int) -> np.ndarray:
        rho = self.reduced_density_matrix(q)
        return np.array([2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real])

    def bloch_angles(self, q: int):
        """(theta, phi, r): spherical coordinates of the reduced Bloch vector; r<1 means entangled."""
        x, y, z = self.bloch_vector(q)
        r = math.sqrt(x * x + y * y + z * z)
        theta = math.acos(max(-1.0, min(1.0, z / r))) if r > 1e-15 else 0.0
        phi = math.atan2(y, x) % (2 * math.pi)
        return theta, phi, r

    def expectation_pauli(self, paulis: Dict[int, str]) -> complex:
        """<psi| P |psi> for a Pauli string given as {qubit: 'X'|'Y'|'Z'}."""
        from .gates import PAULI
        sites = sorted(q for q, c in paulis.items() if c != "I")
        if not sites:
            return 1.0 + 0j
        lo, hi = sites[0], sites[-1]
        self._move_center(lo)
        env = None
        for k in range(lo, hi + 1):
            a = self.A[k]
            op = PAULI[paulis.get(k, "I")]
            oa = np.einsum("st,ltr->lsr", op, a)
            if env is None:
                env = np.einsum("lsr,lsq->rq", a.conj(), oa)
            else:
                env = np.einsum("lm,lsr,msq->rq", env, a.conj(), oa)
        return complex(np.trace(env))

    def entanglement_entropy(self, bond: int) -> float:
        """von Neumann entropy (bits) across the cut between qubit `bond` and `bond+1`."""
        self._move_center(bond)
        a = self.A[bond]
        dl, d, dr = a.shape
        s = np.linalg.svd(a.reshape(dl * d, dr), compute_uv=False)
        p = s ** 2
        p = p[p > 1e-16]
        return float(-np.sum(p * np.log2(p)))

    def bond_dims(self) -> List[int]:
        return [a.shape[2] for a in self.A[:-1]]

    def memory_bytes(self) -> int:
        return int(sum(a.nbytes for a in self.A))

    def expect_2site(self, op: np.ndarray, i: int) -> complex:
        """<psi| op |psi> for a 2-site operator on adjacent sites (i, i+1);
        op is 4x4 (or 2,2,2,2) in the mwqc convention (site i = most significant)."""
        self._move_center(i)
        th = np.tensordot(self.A[i], self.A[i + 1], axes=(2, 0))          # (l,s,t,r)
        o = np.asarray(op).reshape(2, 2, 2, 2)
        return complex(np.einsum("lstr,stuv,luvr->", th.conj(), o, th))

    def overlap(self, other: "MPSState") -> complex:
        """<self|other>, O(n chi^3). Both states must have the same number of qubits."""
        if other.n != self.n:
            raise ValueError("width mismatch")
        env = np.ones((1, 1), dtype=complex)
        for a, b in zip(self.A, other.A):
            env = np.einsum("lm,lsr,msq->rq", env, a.conj(), b)
        return complex(env[0, 0])

    # ── sampling ─────────────────────────────────────────────────────────
    def sample(self, shots: int = 1024, seed: int | None = None) -> np.ndarray:
        """Exact sampling from |psi|^2. Returns (shots, n) array of bits (column k = qubit k)."""
        rng = np.random.default_rng(seed)
        self._move_center(0)
        L = np.ones((shots, 1), dtype=self.dtype)
        out = np.zeros((shots, self.n), dtype=np.int8)
        for k in range(self.n):
            M = np.einsum("nl,lsr->nsr", L, self.A[k])           # (shots,2,dr)
            p = np.sum(np.abs(M) ** 2, axis=2).real              # (shots,2)
            p = p / p.sum(axis=1, keepdims=True)
            s = (rng.random(shots) < p[:, 1]).astype(np.int8)
            out[:, k] = s
            Ms = M[np.arange(shots), s]                          # (shots,dr)
            Ms /= np.linalg.norm(Ms, axis=1, keepdims=True)
            L = Ms
        return out

    def counts(self, shots: int = 1024, seed: int | None = None) -> dict:
        """Qiskit-style counts dict; bitstring label has qubit 0 rightmost."""
        bits = self.sample(shots, seed)
        out: dict = {}
        for row in bits:
            key = "".join(str(int(b)) for b in row[::-1])
            out[key] = out.get(key, 0) + 1
        return out

    # ── dense export (small n only) ──────────────────────────────────────
    def to_vector(self) -> np.ndarray:
        if self.n > 28:
            raise MemoryError("to_vector is only for n <= 28")
        t = self.A[0]
        for a in self.A[1:]:
            t = np.tensordot(t, a, axes=(t.ndim - 1, 0))
        t = t.reshape((2,) * self.n)
        return np.transpose(t, list(range(self.n))[::-1]).reshape(-1)

    def summary(self) -> dict:
        bd = self.bond_dims() or [1]
        return {
            "num_qubits": self.n,
            "chi_max": self.chi_max,
            "max_bond_used": int(max(bd)),
            "mean_bond": float(np.mean(bd)),
            "memory_MB": self.memory_bytes() / 1e6,
            "truncation_error": self.truncation_error,
            "fidelity_estimate": self.fidelity_estimate,
            "n_truncations": self.n_truncations,
        }


def _svd_fallback(m):
    import scipy.linalg
    return scipy.linalg.svd(m, full_matrices=False, lapack_driver="gesvd")


def simulate_mps(circuit: MWCircuit, chi_max: int = 64, cutoff: float = 1e-12,
                 initial: MPSState | None = None) -> MPSState:
    st = initial or MPSState(circuit.num_qubits, chi_max=chi_max, cutoff=cutoff)
    return st.run(circuit)


class BlochState(MPSState):
    """The original MW representation: n independent Bloch spheres (chi = 1).

    Exact for circuits whose state stays a product state (single-qubit gates,
    and controlled gates whose control is in a computational-basis state, which
    covers QFT on basis-state inputs). For anything else each entangling gate is
    projected to the nearest product state and the lost weight is recorded in
    `truncation_error` — so you always know when the Bloch picture stopped being valid.
    """

    def __init__(self, num_qubits: int, **kw):
        kw.pop("chi_max", None)
        super().__init__(num_qubits, chi_max=1, **kw)

    def angles(self):
        return [self.bloch_angles(q) for q in range(self.n)]

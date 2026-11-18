"""
PauliSum — a minimal SparsePauliOp equivalent, plus Pauli algebra (used by the
Jordan-Wigner mapper).

Internal form: {(xmask, zmask): coeff} with P = i^{popcount(x&z)} X^x Z^z,
so Y = iXZ. Masks are Python ints, so there is no qubit limit.
Labels follow Qiskit: rightmost character = qubit 0.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

_I_POW = [1, 1j, -1, -1j]


def _pc(x: int) -> int:
    return bin(x).count("1")


class PauliSum:
    def __init__(self, num_qubits: int, terms: Dict[Tuple[int, int], complex] | None = None):
        self.num_qubits = int(num_qubits)
        self.terms: Dict[Tuple[int, int], complex] = dict(terms or {})

    # ── construction ─────────────────────────────────────────────────────
    @staticmethod
    def label_to_masks(label: str) -> Tuple[int, int]:
        x = z = 0
        for q, c in enumerate(reversed(label)):
            if c in "XY":
                x |= 1 << q
            if c in "ZY":
                z |= 1 << q
            if c not in "IXYZ":
                raise ValueError(f"bad Pauli char {c!r}")
        return x, z

    @staticmethod
    def masks_to_label(x: int, z: int, n: int) -> str:
        out = []
        for q in range(n):
            bx, bz = (x >> q) & 1, (z >> q) & 1
            out.append("Y" if bx and bz else "X" if bx else "Z" if bz else "I")
        return "".join(reversed(out))

    @classmethod
    def from_list(cls, items: Iterable[Tuple[str, complex]], num_qubits: int | None = None):
        items = list(items)
        n = num_qubits or len(items[0][0])
        ps = cls(n)
        for label, c in items:
            ps._add(cls.label_to_masks(label), complex(c))
        return ps

    @classmethod
    def identity(cls, n, c=1.0):
        return cls(n, {(0, 0): complex(c)})

    @classmethod
    def single(cls, n, q, ch, c=1.0):
        lab = ["I"] * n
        lab[n - 1 - q] = ch
        return cls.from_list([("".join(lab), c)], n)

    def _add(self, key, c):
        self.terms[key] = self.terms.get(key, 0) + c

    # ── algebra ──────────────────────────────────────────────────────────
    def __add__(self, other: "PauliSum"):
        out = PauliSum(self.num_qubits, self.terms)
        for k, c in other.terms.items():
            out._add(k, c)
        return out

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return PauliSum(self.num_qubits, {k: c * other for k, c in self.terms.items()})
        out = PauliSum(self.num_qubits)
        for (x1, z1), c1 in self.terms.items():
            for (x2, z2), c2 in other.terms.items():
                x, z = x1 ^ x2, z1 ^ z2
                ph = _pc(x1 & z1) + _pc(x2 & z2) - _pc(x & z) + 2 * _pc(z1 & x2)
                out._add((x, z), c1 * c2 * _I_POW[ph % 4])
        return out

    __rmul__ = __mul__

    def simplify(self, atol: float = 1e-12) -> "PauliSum":
        return PauliSum(self.num_qubits, {k: c for k, c in self.terms.items() if abs(c) > atol})

    def to_list(self) -> List[Tuple[str, complex]]:
        return [(self.masks_to_label(x, z, self.num_qubits), c) for (x, z), c in self.terms.items()]

    def __len__(self):
        return len(self.terms)

    def __repr__(self):
        return f"PauliSum(num_qubits={self.num_qubits}, terms={len(self.terms)})"

    # ── evaluation ───────────────────────────────────────────────────────
    def expectation_vector(self, psi: np.ndarray) -> float:
        """<psi|H|psi> for a dense Qiskit-ordered statevector (n <= ~26)."""
        n = self.num_qubits
        k = np.arange(1 << n, dtype=np.uint64)
        tot = 0.0 + 0j
        for (x, z), c in self.terms.items():
            ph = _I_POW[_pc(x & z) % 4]
            sign = 1 - 2 * (np.bitwise_count(k & np.uint64(z)) & 1).astype(np.int8)
            tot += c * ph * np.vdot(psi[k ^ np.uint64(x)], sign * psi)
        return float(tot.real)

    def to_sparse(self):
        """scipy.sparse CSR matrix (n <= ~20)."""
        import scipy.sparse as sp
        n = self.num_qubits
        N = 1 << n
        k = np.arange(N, dtype=np.uint64)
        mat = sp.csr_matrix((N, N), dtype=complex)
        for (x, z), c in self.terms.items():
            ph = _I_POW[_pc(x & z) % 4]
            sign = 1 - 2 * (np.bitwise_count(k & np.uint64(z)) & 1).astype(np.int8)
            mat = mat + sp.csr_matrix((c * ph * sign, ((k ^ np.uint64(x)).astype(np.int64),
                                                       k.astype(np.int64))), shape=(N, N))
        return mat

    def expectation_mps(self, mps) -> float:
        tot = 0.0 + 0j
        for (x, z), c in self.terms.items():
            d = {}
            for q in range(self.num_qubits):
                bx, bz = (x >> q) & 1, (z >> q) & 1
                if bx or bz:
                    d[q] = "Y" if bx and bz else "X" if bx else "Z"
            tot += c * mps.expectation_pauli(d)
        return float(tot.real)

    def to_qiskit(self):
        from qiskit.quantum_info import SparsePauliOp
        return SparsePauliOp.from_list(self.to_list())

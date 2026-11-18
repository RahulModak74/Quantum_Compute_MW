"""
MWCircuit — a small Qiskit-style circuit builder.

    qc = MWCircuit(3)
    qc.h(0); qc.cx(0, 1); qc.rz(0.3, 2)

Method names and argument order follow qiskit.QuantumCircuit
(parameters first, then qubits; for controlled gates control first).
`to_qiskit()` / `MWCircuit.from_qiskit()` convert both ways when Qiskit is installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from . import gates as G


@dataclass
class Instruction:
    name: str
    qubits: Tuple[int, ...]
    params: Tuple = ()
    matrix: np.ndarray | None = None   # only for name == "unitary"

    def tensor(self) -> np.ndarray:
        if self.name == "unitary":
            return self.matrix
        return G.gate_tensor(self.name, self.params)


@dataclass
class MWCircuit:
    num_qubits: int
    name: str = "circuit"
    data: List[Instruction] = field(default_factory=list)

    # ── internal ─────────────────────────────────────────────────────────
    def _add(self, name, qubits, params=(), matrix=None):
        for q in qubits:
            if not 0 <= q < self.num_qubits:
                raise IndexError(f"qubit {q} out of range for {self.num_qubits}-qubit circuit")
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"duplicate qubits in {name}{qubits}")
        self.data.append(Instruction(name, tuple(int(q) for q in qubits),
                                     tuple(float(p) for p in params), matrix))
        return self

    # ── one-qubit gates ──────────────────────────────────────────────────
    def id(self, q): return self._add("id", (q,))
    def x(self, q): return self._add("x", (q,))
    def y(self, q): return self._add("y", (q,))
    def z(self, q): return self._add("z", (q,))
    def h(self, q): return self._add("h", (q,))
    def s(self, q): return self._add("s", (q,))
    def sdg(self, q): return self._add("sdg", (q,))
    def t(self, q): return self._add("t", (q,))
    def tdg(self, q): return self._add("tdg", (q,))
    def sx(self, q): return self._add("sx", (q,))
    def rx(self, theta, q): return self._add("rx", (q,), (theta,))
    def ry(self, theta, q): return self._add("ry", (q,), (theta,))
    def rz(self, theta, q): return self._add("rz", (q,), (theta,))
    def p(self, theta, q): return self._add("p", (q,), (theta,))
    def u(self, theta, phi, lam, q): return self._add("u", (q,), (theta, phi, lam))

    # ── two-qubit gates ──────────────────────────────────────────────────
    def cx(self, c, t): return self._add("cx", (c, t))
    def cy(self, c, t): return self._add("cy", (c, t))
    def cz(self, c, t): return self._add("cz", (c, t))
    def swap(self, a, b): return self._add("swap", (a, b))
    def cp(self, theta, c, t): return self._add("cp", (c, t), (theta,))
    def crz(self, theta, c, t): return self._add("crz", (c, t), (theta,))
    def rzz(self, theta, a, b): return self._add("rzz", (a, b), (theta,))
    def rxx(self, theta, a, b): return self._add("rxx", (a, b), (theta,))
    def ryy(self, theta, a, b): return self._add("ryy", (a, b), (theta,))

    def unitary(self, matrix, qubits: Sequence[int]):
        """Arbitrary k-qubit gate. `matrix` uses the mwqc convention (first qarg = most
        significant index), NOT Qiskit's little-endian matrix. k > 2 runs on the dense
        statevector backend only (MPS supports 1- and 2-qubit gates)."""
        k = len(qubits)
        m = np.asarray(matrix, dtype=complex)
        if m.size != 4 ** k:
            raise ValueError(f"matrix size {m.size} does not fit {k} qubits")
        m = m.reshape(2, 2) if k == 1 else m.reshape((2,) * (2 * k))
        return self._add("unitary", tuple(qubits), (), m)

    # ── helpers ──────────────────────────────────────────────────────────
    def compose(self, other: "MWCircuit"):
        if other.num_qubits != self.num_qubits:
            raise ValueError("compose needs equal widths")
        self.data.extend(other.data)
        return self

    def size(self) -> int:
        return len(self.data)

    def count_ops(self) -> dict:
        out: dict = {}
        for ins in self.data:
            out[ins.name] = out.get(ins.name, 0) + 1
        return out

    def num_nonlocal_gates(self) -> int:
        return sum(1 for ins in self.data if len(ins.qubits) == 2)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"MWCircuit(num_qubits={self.num_qubits}, size={self.size()}, ops={self.count_ops()})"

    # ── Qiskit interop ───────────────────────────────────────────────────
    _QISKIT_NAMES = {"id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "rx", "ry",
                     "rz", "p", "u", "cx", "cy", "cz", "swap", "cp", "crz", "rzz",
                     "rxx", "ryy"}

    def to_qiskit(self):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(self.num_qubits)
        for ins in self.data:
            if ins.name == "unitary":
                from qiskit.circuit.library import UnitaryGate
                qc.append(UnitaryGate(mw_to_qiskit_matrix(ins.matrix, len(ins.qubits))), list(ins.qubits))
            elif ins.name == "id":
                qc.id(ins.qubits[0])
            else:
                getattr(qc, ins.name)(*ins.params, *ins.qubits)
        return qc

    @classmethod
    def from_qiskit(cls, qc) -> "MWCircuit":
        out = cls(qc.num_qubits, name=getattr(qc, "name", "circuit"))
        for ci in qc.data:
            op = ci.operation
            qs = [qc.find_bit(q).index for q in ci.qubits]
            nm = op.name
            if nm in ("barrier", "measure"):
                continue
            if nm in cls._QISKIT_NAMES:
                out._add(nm, qs, [float(p) for p in op.params])
                continue
            mat = np.asarray(op.to_matrix(), dtype=complex)
            out._add("unitary", qs, (), qiskit_to_mw_matrix(mat, len(qs)))
        return out


def qiskit_to_mw_matrix(mat: np.ndarray, k: int) -> np.ndarray:
    """Qiskit little-endian 2^k x 2^k matrix over qargs (q0..q_{k-1}) -> mwqc tensor."""
    if k == 1:
        return np.asarray(mat, dtype=complex).reshape(2, 2)
    t = np.asarray(mat, dtype=complex).reshape((2,) * (2 * k))   # [o_{k-1}..o_0, i_{k-1}..i_0]
    perm = list(range(k))[::-1] + [k + j for j in range(k)][::-1]
    return t.transpose(perm)


def mw_to_qiskit_matrix(t: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return np.asarray(t).reshape(2, 2)
    perm = list(range(k))[::-1] + [k + j for j in range(k)][::-1]
    return np.asarray(t).transpose(perm).reshape(2 ** k, 2 ** k)

"""
Exact dense statevector simulator — the reference for every other backend.

Memory is 16 * 2**n bytes, so it is only for n <= ~28. It exists so that the
MPS / Bloch backends can be checked against exact answers on small cases.
"""
from __future__ import annotations

import numpy as np

from .circuit import MWCircuit

MAX_QUBITS = 28


class StatevectorState:
    def __init__(self, num_qubits: int, psi: np.ndarray | None = None):
        if num_qubits > MAX_QUBITS:
            raise MemoryError(
                f"dense statevector at n={num_qubits} needs {16 * 2.0**num_qubits / 1e9:.3g} GB; "
                f"use MPSState instead")
        self.n = num_qubits
        if psi is None:
            t = np.zeros((2,) * num_qubits, dtype=complex)
            t[(0,) * num_qubits] = 1.0
        else:
            # psi is in Qiskit order (qubit 0 = LSB): reshape puts qubit n-1 on axis 0, so reverse
            t = np.asarray(psi, dtype=complex).reshape((2,) * num_qubits)
            t = np.transpose(t, list(range(num_qubits))[::-1])
        self.t = t   # axis i == qubit i

    @classmethod
    def from_label(cls, bits: str):
        """bits in Qiskit label order: rightmost char = qubit 0."""
        n = len(bits)
        st = cls(n)
        st.t[(0,) * n] = 0
        st.t[tuple(int(b) for b in reversed(bits))] = 1.0
        return st

    def apply(self, g: np.ndarray, qubits):
        k = len(qubits)
        g = np.asarray(g).reshape((2,) * (2 * k))
        # contract gate input indices with the state axes
        out = np.tensordot(g, self.t, axes=(list(range(k, 2 * k)), list(qubits)))
        # tensordot puts gate output axes first; move them back into place
        self.t = np.moveaxis(out, list(range(k)), list(qubits))
        return self

    def run(self, circuit: MWCircuit):
        for ins in circuit.data:
            self.apply(ins.tensor(), ins.qubits)
        return self

    def to_vector(self) -> np.ndarray:
        """Qiskit little-endian ordering (qubit 0 = LSB)."""
        return np.transpose(self.t, list(range(self.n))[::-1]).reshape(-1)

    def probabilities(self) -> np.ndarray:
        return np.abs(self.to_vector()) ** 2

    def bloch_vector(self, q: int) -> np.ndarray:
        """<X>,<Y>,<Z> of the reduced single-qubit state of qubit q."""
        rho = self.reduced_density_matrix(q)
        return np.array([2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real])

    def reduced_density_matrix(self, q: int) -> np.ndarray:
        t = np.moveaxis(self.t, q, 0).reshape(2, -1)
        return t @ t.conj().T


def simulate_statevector(circuit: MWCircuit, initial: StatevectorState | None = None) -> StatevectorState:
    st = initial or StatevectorState(circuit.num_qubits)
    return st.run(circuit)

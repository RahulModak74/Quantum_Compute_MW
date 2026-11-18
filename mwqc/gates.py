"""
Gate library.

Convention (used everywhere in mwqc):
  * A k-qubit gate acting on qargs (a, b, ...) is stored as a tensor of shape
    (2,)*2k indexed  G[out_a, out_b, ..., in_a, in_b, ...].
    The FIRST qarg is the FIRST tensor index. For CX the first qarg is the control.
  * Full statevectors are returned in Qiskit's little-endian order
    (qubit 0 = least-significant bit), so results compare 1:1 with
    qiskit.quantum_info.Statevector.
"""
from __future__ import annotations

import numpy as np

_SQ2 = 1.0 / np.sqrt(2.0)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = _SQ2 * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG = S.conj().T
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
TDG = T.conj().T
SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)

PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(t):
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]], dtype=complex)


def p(t):
    return np.array([[1, 0], [0, np.exp(1j * t)]], dtype=complex)


def u(theta, phi, lam):
    """Qiskit U gate."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [[c, -np.exp(1j * lam) * s],
         [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def controlled(m: np.ndarray) -> np.ndarray:
    """|0><0| (x) I + |1><1| (x) m, as a (2,2,2,2) tensor, control = first qarg."""
    g = np.zeros((2, 2, 2, 2), dtype=complex)
    g[0, :, 0, :] = np.eye(2)
    g[1, :, 1, :] = m
    return g


def two_qubit(mat4: np.ndarray) -> np.ndarray:
    """4x4 matrix in basis |a b> (a = first qarg, most significant) -> tensor."""
    return np.asarray(mat4, dtype=complex).reshape(2, 2, 2, 2)


CX = controlled(X)
CY = controlled(Y)
CZ = controlled(Z)
SWAP = two_qubit(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))


def cp(t):
    return controlled(p(t))


def crz(t):
    return controlled(rz(t))


def rzz(t):
    d = np.exp(-0.5j * t * np.array([1, -1, -1, 1]))
    return two_qubit(np.diag(d))


def rxx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    m = c * np.eye(4) - 1j * s * np.kron(X, X)
    return two_qubit(m)


def ryy(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    m = c * np.eye(4) - 1j * s * np.kron(Y, Y)
    return two_qubit(m)


ONE_QUBIT_FIXED = {"id": I2, "x": X, "y": Y, "z": Z, "h": H, "s": S, "sdg": SDG,
                   "t": T, "tdg": TDG, "sx": SX}
ONE_QUBIT_PARAM = {"rx": rx, "ry": ry, "rz": rz, "p": p, "u": u}
TWO_QUBIT_FIXED = {"cx": CX, "cy": CY, "cz": CZ, "swap": SWAP}
TWO_QUBIT_PARAM = {"cp": cp, "crz": crz, "rzz": rzz, "rxx": rxx, "ryy": ryy}


def gate_tensor(name: str, params=()) -> np.ndarray:
    """Return the tensor for a named gate."""
    if name in ONE_QUBIT_FIXED:
        return ONE_QUBIT_FIXED[name]
    if name in ONE_QUBIT_PARAM:
        return ONE_QUBIT_PARAM[name](*params)
    if name in TWO_QUBIT_FIXED:
        return TWO_QUBIT_FIXED[name]
    if name in TWO_QUBIT_PARAM:
        return TWO_QUBIT_PARAM[name](*params)
    raise KeyError(f"unknown gate {name!r}")

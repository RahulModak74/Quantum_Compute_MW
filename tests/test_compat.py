"""Legacy-Qiskit compatibility layer vs real Qiskit."""
import numpy as np
import pytest

from mwqc import qiskit_compat as C

qiskit = pytest.importorskip("qiskit")
from qiskit.quantum_info import Statevector, Operator   # noqa: E402


def _both(build):
    a = C.QuantumCircuit(*build["regs"](C))
    b = qiskit.QuantumCircuit(*build["regs"](qiskit))
    build["body"](a, C)
    build["body"](b, qiskit)
    return a, b


def test_gates_registers_append_initialize():
    from qiskit.circuit.library import ZGate as QZ

    def regs(m):
        return [m.QuantumRegister(3, "p"), m.QuantumRegister(2, "r")]

    def body(qc, m):
        qc.initialize([np.sqrt(0.2), np.sqrt(0.8)], 0)
        qc.h([1, 2]); qc.ry(0.7, 3); qc.cry(1.1, 1, 4); qc.ccx(0, 1, 3)
        qc.mcx([0, 1, 2], 4); qc.cz(2, 3); qc.x(4); qc.rz(0.3, 0); qc.ch(3, 2)
        z = (m.circuit.library.ZGate() if m is qiskit else C.ZGate()).control(2)
        qc.append(z, [qc.qubits[0], qc.qubits[1], qc.qubits[4]])

    a, b = _both({"regs": regs, "body": body})
    va = C.execute(a, C.Aer.get_backend("statevector_simulator")).result().get_statevector()
    vb = Statevector(b).data
    assert abs(abs(np.vdot(va, vb)) - 1) < 1e-10


def test_counts_format_and_unitary():
    qr, cr1, cr2 = C.QuantumRegister(3), C.ClassicalRegister(1), C.ClassicalRegister(2)
    qc = C.QuantumCircuit(qr, cr1, cr2)
    qc.x(0); qc.x(2)
    qc.measure(qr[0], cr1[0]); qc.measure(qr[1], cr2[0]); qc.measure(qr[2], cr2[1])
    c = C.execute(qc, C.Aer.get_backend("qasm_simulator"), shots=50).result().get_counts()
    assert c == {"10 1": 50}                                   # Qiskit's register-spaced format
    u = C.QuantumCircuit(2); u.h(0); u.cx(0, 1)
    U = C.execute(u, C.Aer.get_backend("unitary_simulator")).result().get_unitary()
    q = qiskit.QuantumCircuit(2); q.h(0); q.cx(0, 1)
    assert np.allclose(U, Operator(q).data)


def test_subcircuit_append_and_probabilities():
    sub = C.QuantumCircuit(2); sub.h(0); sub.cx(0, 1)
    qc = C.QuantumCircuit(3); qc.append(sub, [2, 0])
    p = C.execute(qc, C.Aer.get_backend("statevector_simulator")).result().get_counts()
    assert set(p) == {"000", "101"} and abs(p["101"] - 0.5) < 1e-12


def test_mid_circuit_measurement_is_refused():
    qc = C.QuantumCircuit(1, 1); qc.h(0); qc.measure(0, 0)
    with pytest.raises(NotImplementedError):
        qc.x(0)

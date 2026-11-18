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


# ── PyQML Vol. 2 additions: rzz/rxx/ryy, Parameter, .data on results, porter rules ──
def test_two_qubit_rotations_match_qiskit():
    for g in ("rzz", "rxx", "ryy"):
        a, b = C.QuantumCircuit(3), qiskit.QuantumCircuit(3)
        for qc in (a, b):
            qc.h(0); qc.ry(0.4, 1); qc.rx(0.9, 2)
            getattr(qc, g)(0.73, 0, 2); getattr(qc, g)(-1.2, 2, 1)
        ua = C.execute(a, C.Aer.get_backend("unitary_simulator")).result().get_unitary()
        ub = Operator(b).data
        k = np.unravel_index(np.argmax(np.abs(ub)), ub.shape)
        assert np.allclose(ua * (ub[k] / ua[k]), ub, atol=1e-12), g


def test_parameter_bind_append_and_qaoa_circuit():
    # the hand-built QAOA of PyQML Vol. 2 ch. "QAOA": rzz(gamma) on edges, rx(beta) mixer, appended sub-circuits
    edges, n = [(0, 1), (1, 2)], 3

    def build(m, P):
        g, b = P("gamma"), P("beta")
        qp = m.QuantumCircuit(n)
        for e in edges:
            qp.rzz(g, *e)
        qa = m.QuantumCircuit(n)
        for i in range(n):
            qa.rx(b, i)
        qc = m.QuantumCircuit(n)
        qc.h(range(n)); qc.append(qp, range(n)); qc.append(qa, range(n))
        return qc, g, b

    from qiskit.circuit import Parameter as QP
    a, ga, ba = build(C, C.Parameter)
    b, gb, bb = build(qiskit, QP)
    assert [p.name for p in a.parameters] == ["beta", "gamma"]
    assert a.decompose() is a and "gamma" in str(a.draw())
    with pytest.raises(ValueError, match="unbound"):
        C.execute(a, C.Aer.get_backend("statevector_simulator"))
    a1 = a.assign_parameters({ga: 0.8, ba: 0.35})
    b1 = b.assign_parameters({gb: 0.8, bb: 0.35})
    va = C.execute(a1, C.Aer.get_backend("statevector_simulator")).result().get_statevector()
    assert abs(abs(np.vdot(va, Statevector(b1).data)) - 1) < 1e-12
    assert a.parameters and not a1.parameters          # original left unbound


def test_result_data_attribute():
    qc = C.QuantumCircuit(2)
    qc.rzz(np.pi, 0, 1)
    u = C.execute(qc, C.Aer.get_backend("unitary_simulator")).result().get_unitary(qc, 2)
    assert isinstance(u.data, np.ndarray) and u.data.shape == (4, 4)
    assert np.allclose(u.data, np.diag([-1j, 1j, 1j, -1j]))


def test_porter_rewrite_rules():
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pyqml_port"))
    from port_book import chapter_selected, needs_legacy, rewrite
    src = "from qiskit import Aer, execute\nfrom qiskit.circuit import Parameter\nimport qutip\n"
    out = rewrite(src, "mwqc")
    assert "from mwqc.qiskit_compat import Aer" in out and "from mwqc.qiskit_compat import Parameter" in out
    assert "import qutip" not in out
    legacy = "from qiskit import Aer\nfrom qiskit.algorithms import QAOA\n"
    assert needs_legacy(legacy) and rewrite(legacy, "mwqc") == legacy
    assert chapter_selected("v2_h_01_qaoa.ipynb", ["qaoa"]) and not chapter_selected("v2_h_01_qaoa.ipynb", ["2"])

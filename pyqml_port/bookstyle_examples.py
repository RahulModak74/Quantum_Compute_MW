#!/usr/bin/env python3
"""
Book-style validation: eight small programs written here (not copied from the book), each
using the same legacy-Qiskit idioms as a chapter of *Hands-On Quantum Machine Learning
with Python, Vol. 1*. Each program is written ONCE against a module `M` and executed
twice: M = pyqml_port.legacy_qiskit (Qiskit 2.x + qiskit-aer) and M = mwqc.qiskit_compat.

    python3 pyqml_port/bookstyle_examples.py
"""
import os
import sys
from math import asin, pi, sqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)


def prob_to_angle(p):
    return 2 * asin(sqrt(p))


# ── programs ─────────────────────────────────────────────────────────────────
def p_initialize(M):                       # ch 2-3: a qubit from amplitudes
    qc = M.QuantumCircuit(1)
    qc.initialize([sqrt(0.3), sqrt(0.7)], 0)
    return M.execute(qc, M.Aer.get_backend("statevector_simulator")).result()


def p_superposition_qasm(M):               # ch 5: measure a superposition
    qc = M.QuantumCircuit(3, 3)
    qc.h(0); qc.ry(prob_to_angle(0.25), 1); qc.cx(0, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return M.execute(qc, M.Aer.get_backend("qasm_simulator"), shots=4000).result()


def p_unitary(M):                          # ch 6: unitary simulator
    qc = M.QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1); qc.i(0); qc.z(1)
    return M.execute(qc, M.Aer.get_backend("unitary_simulator")).result()


def p_registers_oracle(M):                 # ch 6-8: registers, aux qubits, ccx oracle
    qr, cr = M.QuantumRegister(7), M.ClassicalRegister(1)
    qc = M.QuantumCircuit(qr, cr)
    for q in range(4):
        qc.h(qr[q])
    qc.ccx(0, 1, 4); qc.ccx(2, 3, 5); qc.ccx(4, 5, 6)
    qc.measure(qr[6], cr[0])
    return M.execute(qc, M.Aer.get_backend("qasm_simulator"), shots=4000).result()


def p_bayes_net(M):                        # ch 9-10: quantum Bayesian network with cry
    qr, cr = M.QuantumRegister(3), M.ClassicalRegister(1)
    qc = M.QuantumCircuit(qr, cr)
    p_child, p_female = 0.08, 0.35
    p_surv = {(0, 0): 0.19, (0, 1): 0.74, (1, 0): 0.45, (1, 1): 0.69}
    qc.ry(prob_to_angle(p_child), 0)
    qc.ry(prob_to_angle(p_female), 1)
    for (c, f), p in p_surv.items():       # set the target for each parent configuration
        if not c: qc.x(0)
        if not f: qc.x(1)
        # controlled-controlled RY via two cry + cx (as in the book)
        qc.cry(prob_to_angle(p) / 2, 1, 2); qc.cx(0, 1); qc.cry(-prob_to_angle(p) / 2, 1, 2)
        qc.cx(0, 1); qc.cry(prob_to_angle(p) / 2, 0, 2)
        if not c: qc.x(0)
        if not f: qc.x(1)
    qc.measure(qr[2], cr[0])
    return M.execute(qc, M.Aer.get_backend("qasm_simulator"), shots=8000).result()


def p_bayes_net_exact(M):                  # same network, exact probabilities
    qc = M.QuantumCircuit(3)
    p_surv = {(0, 0): 0.19, (0, 1): 0.74, (1, 0): 0.45, (1, 1): 0.69}
    qc.ry(prob_to_angle(0.08), 0); qc.ry(prob_to_angle(0.35), 1)
    for (c, f), p in p_surv.items():
        if not c: qc.x(0)
        if not f: qc.x(1)
        qc.cry(prob_to_angle(p) / 2, 1, 2); qc.cx(0, 1); qc.cry(-prob_to_angle(p) / 2, 1, 2)
        qc.cx(0, 1); qc.cry(prob_to_angle(p) / 2, 0, 2)
        if not c: qc.x(0)
        if not f: qc.x(1)
    return M.execute(qc, M.Aer.get_backend("statevector_simulator")).result()


def p_grover(M):                           # ch 13: amplitude amplification, mcx + ZGate().control
    n = 6
    qr = M.QuantumRegister(n, name="items")
    qc = M.QuantumCircuit(qr)
    qc.h(qr)
    marked = "101101"

    def oracle():
        o = M.QuantumCircuit(n)
        for i, b in enumerate(reversed(marked)):
            if b == "0": o.x(i)
        o.h(n - 1); o.mcx(list(range(n - 1)), n - 1); o.h(n - 1)
        for i, b in enumerate(reversed(marked)):
            if b == "0": o.x(i)
        return o

    def amplifier():
        a = M.QuantumCircuit(n)
        a.h(range(n)); a.x(range(n))
        a.append(M.ZGate().control(n - 1), list(range(n)))
        a.x(range(n)); a.h(range(n))
        return a

    for _ in range(int(round(pi / 4 * sqrt(2 ** n)))):
        qc.append(oracle(), qr)
        qc.append(amplifier(), qr)
    return M.execute(qc, M.Aer.get_backend("statevector_simulator")).result()


def p_phase_kickback(M):                   # ch 11: phase kickback
    qc = M.QuantumCircuit(2)
    qc.h(0); qc.x(1); qc.h(1); qc.cz(0, 1); qc.cp(pi / 4, 0, 1); qc.rz(0.4, 0)
    return M.execute(qc, M.Aer.get_backend("statevector_simulator")).result()


PROGRAMS = [p_initialize, p_superposition_qasm, p_unitary, p_registers_oracle, p_bayes_net,
            p_bayes_net_exact, p_grover, p_phase_kickback]


def tvd(a, b):
    ta, tb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a.get(k, 0) / ta - b.get(k, 0) / tb) for k in set(a) | set(b))


def main():
    import legacy_qiskit as Q
    import mwqc.qiskit_compat as W
    rows = []
    print(f"{'program':<24}{'kind':<12}{'agreement':>14}   note")
    for prog in PROGRAMS:
        rq, rw = prog(Q), prog(W)
        it = rq._items[0]
        if "statevector" in it:
            a, b = rq.get_statevector(), rw.get_statevector()
            f = abs(np.vdot(a, b)) ** 2
            pa, pb = rq.get_counts(), rw.get_counts()
            dp = max(abs(pa.get(k, 0) - pb.get(k, 0)) for k in set(pa) | set(pb))
            rows.append((prog.__name__, "statevector", 1 - f))
            note = f"max |dp| {dp:.1e}"
            if prog is p_grover:
                note += f", P(marked) qiskit {pa.get('101101', 0):.4f} / mwqc {pb.get('101101', 0):.4f}"
            if prog is p_bayes_net_exact:
                ps = sum(v for k, v in pb.items() if k[0] == "1")
                exact = sum(pc * pf * p for (c, f_), p in {(0, 0): 0.19, (0, 1): 0.74, (1, 0): 0.45,
                                                           (1, 1): 0.69}.items()
                            for pc, pf in [((0.08 if c else 0.92), (0.35 if f_ else 0.65))])
                note += f", P(survive) {ps:.4f} vs Bayes formula {exact:.4f}"
            print(f"{prog.__name__:<24}{'statevec':<12}{'1-F=' + format(1 - f, '.1e'):>14}   {note}")
        elif "unitary" in it:
            Ua, Ub = rq.get_unitary(), rw.get_unitary()
            d = float(np.max(np.abs(Ua - Ub)))
            rows.append((prog.__name__, "unitary", d))
            print(f"{prog.__name__:<24}{'unitary':<12}{'|dU|=' + format(d, '.1e'):>14}")
        else:
            ca, cb = rq.get_counts(), rw.get_counts()
            t = tvd(ca, cb)
            rows.append((prog.__name__, "counts", t))
            shots = sum(ca.values())
            print(f"{prog.__name__:<24}{'counts':<12}{'TVD=' + format(t, '.3f'):>14}   "
                  f"{shots} shots each; keys equal: {set(ca) == set(cb)}")
    import json
    os.makedirs(os.path.join(os.path.dirname(HERE), "results"), exist_ok=True)
    json.dump(rows, open(os.path.join(os.path.dirname(HERE), "results", "pyqml_bookstyle.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

"""
The same legacy API (Aer.get_backend / execute / old Result semantics) implemented on
REAL Qiskit 2.x + qiskit-aer. The book's code targets Qiskit < 1.0, where these calls
existed; modern Qiskit removed them. This shim lets the unchanged book code run on
IBM's current stack so it can be compared with mwqc call-for-call.
"""
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile   # noqa: F401
from qiskit.circuit.library import HGate, RYGate, XGate, YGate, ZGate              # noqa: F401
from qiskit_aer import AerSimulator

RECORD = None

# Qiskit >= 1.0 removed QuantumCircuit.i (the book uses it); restore the alias.
if not hasattr(QuantumCircuit, "i"):
    QuantumCircuit.i = QuantumCircuit.id


class _Backend:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class _Aer:
    def get_backend(self, name="qasm_simulator"):
        return _Backend(name)


Aer = _Aer()
BasicAer = Aer


class _Result:
    def __init__(self, items):
        self._items = items

    def _pick(self, c):
        if c is None or isinstance(c, int):
            return self._items[c or 0]
        for it in self._items:
            if it["circuit"] is c:
                return it
        return self._items[0]

    def get_counts(self, circuit=None):
        it = self._pick(circuit)
        if "counts" in it:
            return dict(it["counts"])
        v = it["statevector"]
        p = np.round(np.abs(v) ** 2, 15)
        return {format(i, f"0{it['n']}b"): float(p[i]) for i in np.nonzero(p)[0]}

    def get_statevector(self, circuit=None, decimals=None):
        v = self._pick(circuit)["statevector"]
        return np.round(v, decimals) if decimals is not None else v

    def get_unitary(self, circuit=None, decimals=None):
        return self._pick(circuit)["unitary"]


class _Job:
    def __init__(self, r):
        self._r = r

    def result(self):
        return self._r


def execute(circuits, backend=None, shots=1024, seed_simulator=None, **kw):
    name = backend.name() if backend is not None else "qasm_simulator"
    circs = circuits if isinstance(circuits, (list, tuple)) else [circuits]
    items = []
    for qc in circs:
        it = {"circuit": qc, "n": qc.num_qubits}
        if name in ("statevector_simulator", "aer_simulator_statevector"):
            c2 = qc.remove_final_measurements(inplace=False)
            c2.save_statevector()
            sim = AerSimulator(method="statevector")
            it["statevector"] = np.asarray(sim.run(transpile(c2, sim)).result().get_statevector())
        elif name == "unitary_simulator":
            c2 = qc.remove_final_measurements(inplace=False)
            c2.save_unitary()
            sim = AerSimulator(method="unitary")
            it["unitary"] = np.asarray(sim.run(transpile(c2, sim)).result().get_unitary())
        else:
            sim = AerSimulator()
            r = sim.run(transpile(qc, sim), shots=shots, seed_simulator=seed_simulator).result()
            it["counts"] = r.get_counts()
        items.append(it)
        if RECORD is not None:
            RECORD.append({"backend": name, **{k: v for k, v in it.items() if k != "circuit"}, "shots": shots})
    return _Job(_Result(items))


def plot_histogram(*a, **k):
    return None


plot_bloch_multivector = plot_state_qsphere = plot_histogram

"""
Legacy-Qiskit compatibility layer (the pre-1.0 API used by most books and tutorials,
including Dr. Frank Zickert's *Hands-On Quantum Machine Learning with Python*).
[We recommend you buy these books for Quantum ML learning.]

Porting a script is one import line:

    from qiskit import QuantumCircuit, Aer, execute            # before
    from mwqc.qiskit_compat import QuantumCircuit, Aer, execute  # after

Supported
  QuantumRegister / ClassicalRegister / QuantumCircuit(ints or registers)
  gates: i id x y z h s sdg t tdg sx rx ry rz p u1 u cx cnot cy cz ch swap crx cry crz cp cu1
         rzz rxx ryy ccx toffoli mcx  + append(sub-circuit | XGate/ZGate/...(.control(k)) | qiskit gate)
  initialize(vector, qubits) on fresh qubits, measure, measure_all, barrier, draw, decompose
  Parameter("name") in rotation gates, for drawing; bind with assign_parameters({p: value}) before execute
  Aer.get_backend('statevector_simulator' | 'qasm_simulator' | 'unitary_simulator'
                  | 'aer_simulator' | 'mps_simulator')
  execute(qc, backend, shots=...).result().get_counts() / get_statevector() / get_unitary()
  plot_histogram / plot_bloch_multivector (simple matplotlib versions)

Not supported (raise a clear error): mid-circuit measurement followed by more gates on that
qubit, classical control (c_if), reset, noise models.

Semantics follow the old Aer: on the statevector backend, get_counts() returns the exact
probability dictionary over all qubits (what plot_histogram showed in the book).
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from . import gates as G
from .circuit import MWCircuit, qiskit_to_mw_matrix, mw_to_qiskit_matrix
from .mps import MPSState
from .statevector import StatevectorState


class Parameter:
    """Symbolic gate angle (as qiskit.circuit.Parameter). A circuit holding one can be drawn;
    bind it with QuantumCircuit.assign_parameters({p: value}) before execute."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"Parameter({self.name})"

    __str__ = lambda self: self.name


def _ctrl(f):
    return lambda th: _controlled_matrix(f(th), 1)


# name -> theta -> matrix in the form QuantumCircuit._gate takes
_PARAM_GATES = {
    "rx": G.rx, "ry": G.ry, "rz": G.rz, "p": G.p,
    "crx": _ctrl(G.rx), "cry": _ctrl(G.ry), "crz": _ctrl(G.rz), "cp": _ctrl(G.p),
    "rzz": lambda th: G.rzz(th).reshape(4, 4), "rxx": lambda th: G.rxx(th).reshape(4, 4),
    "ryy": lambda th: G.ryy(th).reshape(4, 4),
}

# every executed result is appended here when RECORD is a list (used by the validation harness)
RECORD: list | None = None


# ── registers ────────────────────────────────────────────────────────────────
class Bit:
    def __init__(self, register, index):
        self.register, self.index = register, index

    def __repr__(self):
        return f"{self.register.name}[{self.index}]"


class _Register:
    _count = 0
    prefix = "r"

    def __init__(self, size: int, name: str | None = None):
        self.size = int(size)
        if name is None:
            name = f"{self.prefix}{type(self)._count}"
            type(self)._count += 1
        self.name = name
        self._bits = [Bit(self, i) for i in range(self.size)]

    def __getitem__(self, k):
        return self._bits[k]

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self._bits)


class QuantumRegister(_Register):
    prefix = "q"


class ClassicalRegister(_Register):
    prefix = "c"


# ── gate objects (for qc.append) ─────────────────────────────────────────────
class Gate:
    def __init__(self, name: str, matrix: np.ndarray, num_qubits: int):
        self.name, self.num_qubits = name, num_qubits
        self.mw_matrix = matrix          # mwqc convention

    def control(self, num_ctrl_qubits: int = 1, label=None, ctrl_state=None):
        if ctrl_state is not None:
            raise NotImplementedError("ctrl_state is not supported")
        base = self.mw_matrix.reshape(2 ** self.num_qubits, 2 ** self.num_qubits)
        return Gate(f"c{num_ctrl_qubits}{self.name}", _controlled_matrix(base, num_ctrl_qubits),
                    self.num_qubits + num_ctrl_qubits)

    def to_matrix(self):
        return mw_to_qiskit_matrix(self.mw_matrix, self.num_qubits)


def _controlled_matrix(base: np.ndarray, n_ctrl: int) -> np.ndarray:
    """Controls are the FIRST qargs (most significant in the mwqc convention)."""
    d = base.shape[0]
    D = (2 ** n_ctrl) * d
    m = np.eye(D, dtype=complex)
    m[D - d:, D - d:] = base
    return m


def XGate(): return Gate("x", G.X, 1)
def YGate(): return Gate("y", G.Y, 1)
def ZGate(): return Gate("z", G.Z, 1)
def HGate(): return Gate("h", G.H, 1)
def RYGate(theta): return Gate("ry", G.ry(theta), 1)
def RXGate(theta): return Gate("rx", G.rx(theta), 1)
def RZGate(theta): return Gate("rz", G.rz(theta), 1)


# ── circuit ──────────────────────────────────────────────────────────────────
class QuantumCircuit:
    def __init__(self, *regs, name: str | None = None):
        self.name = name or "circuit"
        self.qregs: List[QuantumRegister] = []
        self.cregs: List[ClassicalRegister] = []
        self.qubits: List[Bit] = []
        self.clbits: List[Bit] = []
        ints = [r for r in regs if isinstance(r, (int, np.integer))]
        if ints:
            self.add_register(QuantumRegister(ints[0], "q"))
            if len(ints) > 1:
                self.add_register(ClassicalRegister(ints[1], "c"))
        for r in regs:
            if isinstance(r, _Register):
                self.add_register(r)
        self.ops: list = []          # ('g', mw_tensor, qidx) | ('init', vec, qidx) | ('m', q, c) | ('barrier',)

    def add_register(self, reg):
        if isinstance(reg, QuantumRegister):
            self.qregs.append(reg)
            self.qubits.extend(reg)
        else:
            self.cregs.append(reg)
            self.clbits.extend(reg)

    @property
    def num_qubits(self):
        return len(self.qubits)

    @property
    def num_clbits(self):
        return len(self.clbits)

    def size(self):
        return sum(1 for o in self.ops if o[0] in ("g", "init", "sym"))

    # ── argument resolution ──────────────────────────────────────────────
    def _q(self, spec) -> List[int]:
        if isinstance(spec, (int, np.integer)):
            if not 0 <= spec < self.num_qubits:
                raise IndexError(f"qubit {spec} out of range")
            return [int(spec)]
        if isinstance(spec, Bit):
            return [self._find(self.qubits, spec)]
        if isinstance(spec, QuantumRegister):
            return [self._find(self.qubits, b) for b in spec]
        if isinstance(spec, (list, tuple, range)):
            out = []
            for s in spec:
                out += self._q(s)
            return out
        raise TypeError(f"cannot interpret qubit spec {spec!r}")

    def _c(self, spec) -> List[int]:
        if isinstance(spec, (int, np.integer)):
            return [int(spec)]
        if isinstance(spec, Bit):
            return [self._find(self.clbits, spec)]
        if isinstance(spec, ClassicalRegister):
            return [self._find(self.clbits, b) for b in spec]
        if isinstance(spec, (list, tuple, range)):
            out = []
            for s in spec:
                out += self._c(s)
            return out
        raise TypeError(f"cannot interpret clbit spec {spec!r}")

    @staticmethod
    def _find(bits, b):
        for i, x in enumerate(bits):
            if x is b:
                return i
        raise ValueError(f"{b} is not in this circuit")

    def _gate(self, mat, *qspecs):
        qs = []
        for s in qspecs:
            qs += self._q(s)
        if len(set(qs)) != len(qs):
            raise ValueError(f"duplicate qubits {qs}")
        self._check_not_measured(qs)
        k = len(qs)
        t = np.asarray(mat, dtype=complex).reshape((2,) * (2 * k)) if k > 1 else np.asarray(mat).reshape(2, 2)
        self.ops.append(("g", t, tuple(qs)))
        return self

    def _broadcast1(self, mat, q):
        for x in self._q(q):
            self._gate(mat, x)
        return self

    def _check_not_measured(self, qs):
        measured = {o[1] for o in self.ops if o[0] == "m"}
        if measured & set(qs):
            raise NotImplementedError("gate after measurement on the same qubit (mid-circuit "
                                      "measurement) is not supported by mwqc.qiskit_compat")

    # ── 1-qubit gates ────────────────────────────────────────────────────
    def i(self, q): return self._broadcast1(G.I2, q)
    id = i
    def x(self, q): return self._broadcast1(G.X, q)
    def y(self, q): return self._broadcast1(G.Y, q)
    def z(self, q): return self._broadcast1(G.Z, q)
    def h(self, q): return self._broadcast1(G.H, q)
    def s(self, q): return self._broadcast1(G.S, q)
    def sdg(self, q): return self._broadcast1(G.SDG, q)
    def t(self, q): return self._broadcast1(G.T, q)
    def tdg(self, q): return self._broadcast1(G.TDG, q)
    def sx(self, q): return self._broadcast1(G.SX, q)
    def _rot1(self, name, th, q):
        if isinstance(th, Parameter):
            for x in self._q(q):
                self._sym(name, th, x)
            return self
        return self._broadcast1(_PARAM_GATES[name](th), q)

    def _rotk(self, name, th, *qspecs):
        if isinstance(th, Parameter):
            return self._sym(name, th, *qspecs)
        return self._gate(_PARAM_GATES[name](th), *qspecs)

    def _sym(self, name, th, *qspecs):
        qs = []
        for s_ in qspecs:
            qs += self._q(s_)
        if len(set(qs)) != len(qs):
            raise ValueError(f"duplicate qubits {qs}")
        self._check_not_measured(qs)
        self.ops.append(("sym", name, th, tuple(qs)))
        return self

    def rx(self, th, q): return self._rot1("rx", th, q)
    def ry(self, th, q): return self._rot1("ry", th, q)
    def rz(self, th, q): return self._rot1("rz", th, q)
    def p(self, th, q): return self._rot1("p", th, q)
    u1 = p
    def u(self, th, ph, lam, q): return self._broadcast1(G.u(th, ph, lam), q)
    u3 = u

    # ── controlled gates ─────────────────────────────────────────────────
    def _c1(self, base, c, t):
        return self._gate(_controlled_matrix(base, 1), c, t)

    def cx(self, c, t): return self._c1(G.X, c, t)
    cnot = cx
    def cy(self, c, t): return self._c1(G.Y, c, t)
    def cz(self, c, t): return self._c1(G.Z, c, t)
    def ch(self, c, t): return self._c1(G.H, c, t)
    def crx(self, th, c, t): return self._rotk("crx", th, c, t)
    def cry(self, th, c, t): return self._rotk("cry", th, c, t)
    def crz(self, th, c, t): return self._rotk("crz", th, c, t)
    def cp(self, th, c, t): return self._rotk("cp", th, c, t)
    def rzz(self, th, a, b): return self._rotk("rzz", th, a, b)
    def rxx(self, th, a, b): return self._rotk("rxx", th, a, b)
    def ryy(self, th, a, b): return self._rotk("ryy", th, a, b)
    cu1 = cp
    def swap(self, a, b): return self._gate(G.SWAP.reshape(4, 4), a, b)
    def ccx(self, c1, c2, t): return self._gate(_controlled_matrix(G.X, 2), c1, c2, t)
    toffoli = ccx

    def mcx(self, controls, target, ancilla_qubits=None, mode=None):
        cs = self._q(controls)
        return self._gate(_controlled_matrix(G.X, len(cs)), cs, target)

    def mcp(self, lam, controls, target):
        cs = self._q(controls)
        return self._gate(_controlled_matrix(G.p(lam), len(cs)), cs, target)

    def unitary(self, matrix, qubits, label=None):
        qs = self._q(qubits)
        return self._gate(qiskit_to_mw_matrix(np.asarray(matrix), len(qs)), qs)

    # ── state preparation, measurement, structure ────────────────────────
    def initialize(self, params, qubits=None, normalize: bool = False):
        qs = self._q(qubits) if qubits is not None else list(range(self.num_qubits))
        vec = np.asarray(params, dtype=complex).reshape(-1)
        if vec.size != 2 ** len(qs):
            raise ValueError(f"initialize: {vec.size} amplitudes for {len(qs)} qubits")
        nrm = np.linalg.norm(vec)
        if not normalize and abs(nrm - 1) > 1e-8:
            raise ValueError("initialize: vector is not normalised")
        vec = vec / nrm
        touched = {q for o in self.ops if o[0] in ("g", "init") for q in o[-1]}
        if touched & set(qs):
            raise NotImplementedError("initialize on qubits that already have gates needs a reset; "
                                      "not supported (put initialize first)")
        self._check_not_measured(qs)
        U = _state_prep_unitary(vec)            # Qiskit little-endian over qs
        self.ops.append(("init", qiskit_to_mw_matrix(U, len(qs)), tuple(qs)))
        return self

    def measure(self, qubit, clbit):
        qs, cs = self._q(qubit), self._c(clbit)
        if len(qs) != len(cs):
            raise ValueError("measure: qubit/clbit count mismatch")
        for q, c in zip(qs, cs):
            self.ops.append(("m", q, c))
        return self

    def measure_all(self, inplace=True):
        cr = ClassicalRegister(self.num_qubits, "meas")
        self.add_register(cr)
        return self.measure(list(range(self.num_qubits)), list(range(self.num_clbits - self.num_qubits,
                                                                     self.num_clbits)))

    def barrier(self, *args):
        return self

    def append(self, instruction, qargs=None, cargs=None):
        qs = self._q(qargs) if qargs is not None else list(range(self.num_qubits))
        if isinstance(instruction, QuantumCircuit):
            if instruction.num_qubits != len(qs):
                raise ValueError("append: sub-circuit width mismatch")
            for o in instruction.ops:
                if o[0] in ("g", "init"):
                    if o[0] == "init":
                        raise NotImplementedError("append of a circuit containing initialize")
                    self._gate(o[1].reshape(2 ** len(o[2]), 2 ** len(o[2])) if len(o[2]) > 1 else o[1],
                               [qs[i] for i in o[2]])
                elif o[0] == "sym":
                    self._sym(o[1], o[2], [qs[i] for i in o[3]])
                elif o[0] == "m":
                    raise NotImplementedError("append of a circuit containing measurements")
            return self
        if isinstance(instruction, Gate):
            return self._gate(instruction.mw_matrix.reshape(2 ** instruction.num_qubits, -1), qs)
        if hasattr(instruction, "to_matrix"):                       # a real Qiskit gate
            return self._gate(qiskit_to_mw_matrix(np.asarray(instruction.to_matrix()), len(qs))
                              .reshape(2 ** len(qs), -1), qs)
        if hasattr(instruction, "definition") or hasattr(instruction, "num_qubits"):
            from qiskit.quantum_info import Operator               # pragma: no cover
            return self._gate(qiskit_to_mw_matrix(Operator(instruction).data, len(qs))
                              .reshape(2 ** len(qs), -1), qs)
        raise TypeError(f"cannot append {instruction!r}")

    compose_into = append

    def to_gate(self, label=None):
        return self

    to_instruction = to_gate

    def copy(self):
        import copy
        return copy.deepcopy(self)

    @property
    def parameters(self):
        seen = []
        for o in self.ops:
            if o[0] == "sym" and o[2] not in seen:
                seen.append(o[2])
        return sorted(seen, key=lambda p_: p_.name)

    def assign_parameters(self, values, inplace=False):
        """values: {Parameter: number} or a sequence in the order of self.parameters."""
        if not isinstance(values, dict):
            values = dict(zip(self.parameters, values))
        qc = self if inplace else self.copy()
        ops, qc.ops = qc.ops, []
        for o in ops:
            if o[0] == "sym":
                hit = [v for k, v in values.items() if k is o[2] or getattr(k, "name", None) == o[2].name]
                if hit:
                    qc._gate(_PARAM_GATES[o[1]](float(hit[0])), list(o[3]))
                    continue
            qc.ops.append(o)
        return None if inplace else qc

    bind_parameters = assign_parameters

    def decompose(self, *a, **k):
        return self          # sub-circuits are already flattened into gates on append

    # ── compile to mwqc ──────────────────────────────────────────────────
    def _unitary_part(self) -> MWCircuit:
        if any(o[0] == "sym" for o in self.ops):
            raise ValueError(f"circuit has unbound parameters {self.parameters}; "
                             "bind them with qc.assign_parameters({p: value}) before execute")
        mc = MWCircuit(self.num_qubits, name=self.name)
        for o in self.ops:
            if o[0] in ("g", "init"):
                mc._add("unitary", o[2], (), o[1])
        return mc

    def _measurements(self):
        return [(o[1], o[2]) for o in self.ops if o[0] == "m"]

    # ── display ──────────────────────────────────────────────────────────
    def to_qiskit(self):
        from qiskit import ClassicalRegister as QCR, QuantumCircuit as QQC, QuantumRegister as QQR
        from qiskit.circuit.library import UnitaryGate
        qc = QQC(*[QQR(r.size, r.name) for r in self.qregs], *[QCR(r.size, r.name) for r in self.cregs])
        qparams = {}
        for o in self.ops:
            if o[0] == "init":
                k = len(o[2])
                U = mw_to_qiskit_matrix(o[1], k)
                qc.initialize(U[:, 0], list(o[2]))
            elif o[0] == "g":
                k = len(o[2])
                qc.append(UnitaryGate(mw_to_qiskit_matrix(o[1], k)), list(o[2]))
            elif o[0] == "sym":
                from qiskit.circuit import Parameter as QP
                qp = qparams.setdefault(o[2].name, QP(o[2].name))
                getattr(qc, o[1])(qp, *o[3])
            elif o[0] == "m":
                qc.measure(o[1], o[2])
        return qc

    def draw(self, output=None, **kw):
        try:
            return self.to_qiskit().draw(output=output or "text", **kw)
        except Exception:
            return str(self)

    def __str__(self):
        return (f"QuantumCircuit({self.num_qubits} qubits, {self.num_clbits} clbits, "
                f"{self.size()} gates, {len(self._measurements())} measurements)")

    __repr__ = __str__


def _state_prep_unitary(psi: np.ndarray) -> np.ndarray:
    """Unitary U with U|0> = psi (Householder reflector times a phase)."""
    d = len(psi)
    a = psi[0] / abs(psi[0]) if abs(psi[0]) > 1e-15 else 1.0
    phi = psi / a
    e0 = np.zeros(d, dtype=complex)
    e0[0] = 1
    u = e0 - phi
    nu = np.vdot(u, u).real
    H = np.eye(d, dtype=complex) if nu < 1e-24 else np.eye(d) - 2 * np.outer(u, u.conj()) / nu
    return a * H


# ── backends, execute, results ───────────────────────────────────────────────
class _Backend:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name

    def __repr__(self):
        return f"MWBackend({self._name!r})"

    def run(self, circuits, shots=1024, seed_simulator=None, **kw):
        return execute(circuits, self, shots=shots, seed_simulator=seed_simulator)


class _Aer:
    NAMES = {"statevector_simulator", "qasm_simulator", "unitary_simulator", "aer_simulator",
             "aer_simulator_statevector", "mps_simulator"}

    def get_backend(self, name="qasm_simulator"):
        if name not in self.NAMES:
            raise ValueError(f"unknown backend {name!r}; have {sorted(self.NAMES)}")
        return _Backend(name)

    def backends(self):
        return [_Backend(n) for n in sorted(self.NAMES)]


Aer = _Aer()
BasicAer = Aer


def _probs_dict(vec: np.ndarray, n: int, decimals: int = 15) -> dict:
    p = np.round(np.abs(vec) ** 2, decimals)
    return {format(i, f"0{n}b"): float(p[i]) for i in np.nonzero(p)[0]}


class _Result:
    def __init__(self, items):
        self._items = items          # list of dicts, one per circuit

    def _pick(self, circuit):
        if circuit is None or isinstance(circuit, int):
            return self._items[circuit or 0]
        for it in self._items:
            if it["circuit"] is circuit:
                return it
        return self._items[0]

    def get_counts(self, circuit=None):
        it = self._pick(circuit)
        if "counts" in it:
            return dict(it["counts"])
        if "statevector" in it:
            return _probs_dict(it["statevector"], it["n"])
        raise ValueError("no counts for this experiment")

    def get_statevector(self, circuit=None, decimals=None):
        v = self._pick(circuit)["statevector"]
        return _OpArray.of(np.round(v, decimals) if decimals is not None else v)

    def get_unitary(self, circuit=None, decimals=None):
        u = self._pick(circuit)["unitary"]
        return _OpArray.of(np.round(u, decimals) if decimals is not None else u)

    def success(self):
        return True


class _OpArray(np.ndarray):
    """ndarray that also answers `.data` with the matrix itself, as the Statevector / Operator
    objects of later pre-1.0 Qiskit did (books print `result.get_unitary(qc).data`)."""

    @classmethod
    def of(cls, a):
        return np.asarray(a).view(cls)

    @property
    def data(self):
        return self.view(np.ndarray)


class _Job:
    def __init__(self, r):
        self._r = r

    def result(self):
        return self._r

    def status(self):
        return "DONE"


def _run_state(qc: QuantumCircuit, backend_name: str):
    mc = qc._unitary_part()
    if backend_name == "mps_simulator":
        return MPSState(qc.num_qubits, chi_max=256).run(mc)
    return StatevectorState(qc.num_qubits).run(mc)


def execute(circuits, backend=None, shots: int = 1024, seed_simulator=None, **kw):
    backend = backend or Aer.get_backend("qasm_simulator")
    name = backend.name() if hasattr(backend, "name") and callable(backend.name) else str(backend)
    circs = circuits if isinstance(circuits, (list, tuple)) else [circuits]
    rng = np.random.default_rng(seed_simulator)
    items = []
    for qc in circs:
        n = qc.num_qubits
        it = {"circuit": qc, "n": n}
        if name == "unitary_simulator":
            if n > 12:
                raise MemoryError("unitary_simulator limited to 12 qubits")
            cols = []
            for k in range(2 ** n):
                e = np.zeros(2 ** n, dtype=complex)
                e[k] = 1
                cols.append(StatevectorState(n, e).run(qc._unitary_part()).to_vector())
            it["unitary"] = np.array(cols).T
        elif name in ("statevector_simulator", "aer_simulator_statevector"):
            it["statevector"] = _run_state(qc, name).to_vector()
        else:   # qasm / aer / mps: sample measured qubits into classical bits
            st = _run_state(qc, name)
            meas = qc._measurements()
            if name == "mps_simulator":
                bits = st.sample(shots, seed=int(rng.integers(1 << 31)))
            else:
                p = st.probabilities()
                idx = rng.choice(len(p), size=shots, p=p / p.sum())
                bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
            it["counts"] = _counts_from_bits(bits, meas, qc)
        items.append(it)
        if RECORD is not None:
            RECORD.append({"backend": name, **{k: v for k, v in it.items() if k != "circuit"},
                           "shots": shots})
    return _Job(_Result(items))


def _counts_from_bits(bits: np.ndarray, meas, qc: QuantumCircuit) -> dict:
    nc = qc.num_clbits
    if nc == 0:
        return {"": int(len(bits))}
    cl = np.zeros((len(bits), nc), dtype=np.int8)
    for q, c in meas:
        cl[:, c] = bits[:, q]
    # key: registers in reverse order, separated by spaces; within a register, highest bit left
    offsets, o = [], 0
    for r in qc.cregs:
        offsets.append((o, r.size))
        o += r.size
    out: dict = {}
    for row in cl:
        parts = ["".join(str(int(row[s + j])) for j in reversed(range(size))) for s, size in offsets]
        key = " ".join(reversed(parts))
        out[key] = out.get(key, 0) + 1
    return out


# ── plotting (minimal) ───────────────────────────────────────────────────────
def plot_histogram(data, figsize=None, title=None, **kw):
    try:
        import matplotlib.pyplot as plt
    except Exception:       # pragma: no cover
        return None
    dd = data if isinstance(data, list) else [data]
    fig, ax = plt.subplots(figsize=figsize or (7, 4))
    keys = sorted(set().union(*[d.keys() for d in dd]))
    w = 0.8 / len(dd)
    for i, d in enumerate(dd):
        tot = sum(d.values())
        vals = [d.get(k, 0) / tot for k in keys]
        ax.bar(np.arange(len(keys)) + i * w, vals, width=w)
    ax.set_xticks(np.arange(len(keys)) + 0.4 - w / 2)
    ax.set_xticklabels(keys, rotation=70)
    ax.set_ylabel("Probability")
    if title:
        ax.set_title(title)
    return fig


def plot_bloch_multivector(state, **kw):
    try:
        import matplotlib.pyplot as plt
    except Exception:       # pragma: no cover
        return None
    vec = np.asarray(state)
    n = int(round(math.log2(len(vec))))
    st = StatevectorState(n, vec)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)
    for q in range(n):
        x, y, z = st.bloch_vector(q)
        ax = axes[0, q]
        ax.add_patch(__import__("matplotlib").patches.Circle((0, 0), 1, fill=False))
        ax.arrow(0, 0, x, z, head_width=0.05)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.set_title(f"qubit {q}: (x,y,z)=({x:.2f},{y:.2f},{z:.2f})", fontsize=8)
    return fig


plot_state_qsphere = plot_bloch_multivector


# `import qiskit; qiskit.__qiskit_version__` is printed in PyQML Vol. 1 ch. 3
def _versions():
    from . import __version__
    return {"mwqc": __version__, "qiskit-compat-api": "pre-1.0 (execute / Aer)"}


__qiskit_version__ = _versions()

"""Correctness tests: every fast/approximate path is checked against an exact one."""
import math

import numpy as np
import pytest

from mwqc import BlochState, MPSState, MWCircuit, MWEstimator, MWSampler, PauliSum, StatevectorState
from mwqc.library import ghz, hardware_efficient, qft, qft_basis_state_bloch, qft_lnn, random_circuit


def _fidelity(a, b):
    return abs(np.vdot(a, b)) ** 2


# ── statevector is right (vs numpy FFT and vs Qiskit) ────────────────────────
@pytest.mark.parametrize("n", [1, 3, 5])
def test_qft_matches_fft(n):
    rng = np.random.default_rng(n)
    psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    psi /= np.linalg.norm(psi)
    out = StatevectorState(n, psi).run(qft(n))
    expect = np.fft.ifft(psi) * np.sqrt(2 ** n)    # QFT|j> = sum_k e^{+2 pi i jk/N}|k>/sqrt(N)
    assert np.allclose(out.to_vector(), expect, atol=1e-10)


def test_statevector_matches_qiskit():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector
    qc = random_circuit(6, 6, seed=3)
    qc.cp(0.7, 0, 5).swap(1, 4).rzz(0.3, 2, 3).crz(1.1, 5, 1).sdg(2).t(0)
    mine = StatevectorState(6).run(qc).to_vector()
    ref = Statevector(qc.to_qiskit()).data
    assert _fidelity(mine, ref) > 1 - 1e-10


def test_from_qiskit_roundtrip():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import QFTGate
    from qiskit.quantum_info import Statevector
    q0 = QuantumCircuit(5)
    q0.append(QFTGate(5), range(5))
    q = transpile(q0, basis_gates=["h", "cp", "swap", "u", "cx"])
    mine = StatevectorState(5).run(MWCircuit.from_qiskit(q)).to_vector()
    assert _fidelity(mine, Statevector(q).data) > 1 - 1e-10


# ── MPS == statevector ───────────────────────────────────────────────────────
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mps_exact_when_untruncated(seed):
    qc = random_circuit(8, 8, seed=seed)
    qc.cx(0, 7).cp(0.4, 6, 1).swap(2, 5)               # long-range gates go through SWAP routing
    sv = StatevectorState(8).run(qc).to_vector()
    mps = MPSState(8, chi_max=64).run(qc)
    assert mps.truncation_error < 1e-12
    assert _fidelity(mps.to_vector(), sv) > 1 - 1e-10


def test_statevector_roundtrip_qiskit_order():
    rng = np.random.default_rng(5)
    psi = rng.normal(size=32) + 1j * rng.normal(size=32)
    assert np.allclose(StatevectorState(5, psi).to_vector(), psi)
    m = MPSState.from_bits([1, 0, 0, 0, 0])          # qubit 0 set -> index 1
    assert np.allclose(StatevectorState(5, m.to_vector()).to_vector()[1], 1)


def test_mps_truncation_is_reported():
    qc = random_circuit(12, 12, seed=7)
    sv = StatevectorState(12).run(qc).to_vector()
    mps = MPSState(12, chi_max=4).run(qc)
    f = _fidelity(mps.to_vector(), sv)
    assert mps.truncation_error > 1e-3                 # it did have to throw weight away
    assert f < 0.999                                   # and the state really is worse
    assert mps.fidelity_estimate < 0.999               # and it says so


def test_qft_lnn_equals_qft():
    n = 7
    rng = np.random.default_rng(0)
    pre = MWCircuit(n)
    for q in range(n):
        pre.u(*rng.uniform(0, 6, 3), q)
    a = StatevectorState(n).run(pre).run(qft(n)).to_vector()
    b = MPSState(n, chi_max=128).run(pre).run(qft_lnn(n)).to_vector()
    assert _fidelity(a, b) > 1 - 1e-10


# ── the original MW claim, checked: QFT|j> is a product state ────────────────
@pytest.mark.parametrize("n,j", [(6, 0), (6, 37), (10, 1000), (12, 4095)])
def test_qft_basis_state_closed_form(n, j):
    bits = [(j >> q) & 1 for q in range(n)]
    st = BlochState.from_bits(bits)
    st.run(qft_lnn(n))
    assert st.truncation_error < 1e-12                 # chi = 1 was enough: no entanglement
    theta, phi = qft_basis_state_bloch(j, n)
    for q in range(n):
        th, ph, r = st.bloch_angles(q)
        assert abs(r - 1) < 1e-9
        assert abs(th - theta[q]) < 1e-9
        d = (ph - phi[q] + math.pi) % (2 * math.pi) - math.pi
        assert abs(d) < 1e-9


def test_bloch_flags_entanglement():
    st = BlochState(2).run(ghz(2))
    assert st.truncation_error == pytest.approx(0.5, abs=1e-9)   # GHZ is half lost at chi=1


# ── Pauli algebra, estimator, sampler ────────────────────────────────────────
def test_pauli_algebra():
    X = PauliSum.from_list([("X", 1)])
    Y = PauliSum.from_list([("Y", 1)])
    Z = PauliSum.from_list([("Z", 1)])
    assert (X * Y).terms == (Z * 1j).terms
    assert (Y * X).terms == (Z * -1j).terms
    assert (Y * Y).simplify().terms == {(0, 0): 1}


def test_estimator_backends_agree():
    qc = hardware_efficient(6, 2, np.linspace(0.1, 1.7, 18))
    H = PauliSum.from_list([("ZZIIII", 0.5), ("IIXXII", -0.3), ("YIIIIY", 0.2), ("IIIIIZ", 1.0)])
    e_sv = MWEstimator("statevector").run([(qc, H)]).result()[0].data.evs
    r = MWEstimator("mps", chi_max=64).run([(qc, H)]).result()[0]
    assert abs(r.data.evs - e_sv) < 1e-10
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector
    e_q = Statevector(qc.to_qiskit()).expectation_value(H.to_qiskit()).real
    assert abs(e_q - e_sv) < 1e-10


def test_sampler_ghz_100():
    res = MWSampler("mps", seed=1).run([ghz(100)], shots=400).result()[0]
    counts = res.data.get_counts()
    assert set(counts) == {"0" * 100, "1" * 100}
    assert res.metadata["max_bond_used"] == 2

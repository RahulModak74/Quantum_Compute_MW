"""Chemistry path: PySCF integrals -> Jordan-Wigner -> exact & VQE, checked against PySCF FCI."""
import numpy as np
import pytest

pytest.importorskip("pyscf")
from mwqc.chem import MolecularProblem, exact_ground_energy, jordan_wigner, vqe_uccsd


def test_h2_jw_and_vqe_reach_fci():
    prob = MolecularProblem.from_pyscf("H 0 0 0; H 0 0 0.735", basis="sto-3g")
    H = jordan_wigner(prob)
    assert prob.num_qubits == 4
    e_exact = exact_ground_energy(H, prob)
    assert abs(e_exact - prob.e_fci) < 1e-8                 # mapping is exact
    hf = sum(b << q for q, b in enumerate(prob.hf_bits()))
    psi = np.zeros(16); psi[hf] = 1
    assert abs(H.expectation_vector(psi) - prob.e_hf) < 1e-8  # HF determinant energy
    r = vqe_uccsd(prob, H)
    assert abs(r.energy - prob.e_fci) < 1.6e-3 / 10          # well inside chemical accuracy


def test_h2_jw_matches_qiskit_spectrum():
    """Same physics as Qiskit's SparsePauliOp: compare full spectra."""
    pytest.importorskip("qiskit")
    prob = MolecularProblem.from_pyscf("H 0 0 0; H 0 0 0.9", basis="sto-3g")
    H = jordan_wigner(prob)
    a = np.linalg.eigvalsh(H.to_sparse().toarray())
    b = np.linalg.eigvalsh(H.to_qiskit().to_matrix())
    assert np.allclose(a, b, atol=1e-10)


def test_ks_orbitals_same_fci():
    hf = MolecularProblem.from_pyscf("H 0 0 0; H 0 0 0.735", orbitals="hf")
    ks = MolecularProblem.from_pyscf("H 0 0 0; H 0 0 0.735", orbitals="ks")
    e_hf = exact_ground_energy(jordan_wigner(hf), hf)
    e_ks = exact_ground_energy(jordan_wigner(ks), ks)
    assert abs(e_hf - e_ks) < 1e-8

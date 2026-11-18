"""
mwqc — MW Quantum Compute: a CPU-only quantum-circuit toolkit in the spirit of
IBM's Qiskit, built around the MW "Bloch manifold" idea made rigorous as a
matrix-product state (MPS).

    from mwqc import MWCircuit, MWEstimator, MWSampler, MPSState, BlochState
    from mwqc.library import qft, qft_lnn, ghz, hardware_efficient
    from mwqc.pauli import PauliSum
    from mwqc.chem import MolecularProblem, jordan_wigner, vqe_uccsd
    from mwqc.surrogate import MWSurrogate

See README.md for what is exact, what is approximate, and the Qiskit coverage map.
"""
from .circuit import MWCircuit, Instruction
from .mps import MPSState, BlochState, simulate_mps
from .statevector import StatevectorState, simulate_statevector
from .pauli import PauliSum
from .primitives import MWEstimator, MWSampler, run_circuit

# Qiskit-style aliases so ported scripts read naturally
QuantumCircuit = MWCircuit
SparsePauliOp = PauliSum
Estimator = MWEstimator
Sampler = MWSampler

__version__ = "0.1.0"

#!/usr/bin/env python3
"""Five-minute tour of mwqc. Runs in ~20 s on a laptop CPU."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mwqc import BlochState, MPSState, MWCircuit, MWEstimator, MWSampler, PauliSum
from mwqc.library import ghz, qft_basis_state_bloch, qft_lnn

# 1. Qiskit-style circuit + V2-style primitives
qc = MWCircuit(3)
qc.h(0).cx(0, 1).cx(1, 2)
print("GHZ(3) counts:", MWSampler(seed=0).run([qc], shots=1000).result()[0].data.get_counts())
H = PauliSum.from_list([("ZZI", 1.0), ("XXX", 0.5)])
r = MWEstimator().run([(qc, H)]).result()[0]
print(f"<ZZI + 0.5 XXX> = {r.data.evs:.4f}   (method={r.metadata['method']})")

# 2. The same API at 300 qubits: switch the method to MPS
big = ghz(300)
r = MWEstimator(method="mps").run([(big, PauliSum.single(300, 299, "Z") * PauliSum.single(300, 0, "Z"))]).result()[0]
print(f"GHZ(300) <Z0 Z299> = {r.data.evs:.4f}, error bound {r.data.stds:.1e}, "
      f"max bond {r.metadata['max_bond_used']}, {r.metadata['memory_MB']:.2f} MB")

# 3. The original MW picture: one Bloch sphere per qubit (chi = 1), exact for QFT|j>
n, j = 64, 0xDEADBEEFCAFEF00D
st = BlochState.from_bits([(j >> q) & 1 for q in range(n)]).run(qft_lnn(n))
th, ph = qft_basis_state_bloch(j, n)
err = max(abs(((st.bloch_angles(q)[1] - ph[q] + np.pi) % (2 * np.pi)) - np.pi) for q in range(n))
print(f"QFT|j> on {n} qubits with chi=1: truncation {st.truncation_error:.1e}, max phase error {err:.1e} rad")

# 4. ...and where chi = 1 stops being enough (entanglement is measured, not assumed)
st = BlochState(2).run(ghz(2))
print(f"GHZ(2) forced into chi=1: discarded weight {st.truncation_error:.2f}  -> use MPSState(chi_max>1)")

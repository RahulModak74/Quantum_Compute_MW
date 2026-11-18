"""
Estimator / Sampler primitives shaped like Qiskit's V2 primitives:

    est = MWEstimator(method="mps", chi_max=64)
    res = est.run([(circuit, observable)])
    res[0].data.evs, res[0].data.stds, res[0].metadata

`method`:
  "statevector" — exact, n <= 28
  "mps"         — matrix product state, any n, error tracked in metadata
  "bloch"       — chi = 1 product state (the original MW Bloch-sphere picture)
  "auto"        — statevector for n <= 20, otherwise mps

`stds` is the *truncation-error bound* of the simulation (0 for exact methods),
not a shot-noise or "Bayesian" sigma. For an observable with ||O|| <= sum|c|,
|<O>_mps - <O>_exact| <= 2 * sum|c| * sqrt(2 * truncation_error), using
||psi_mps - psi_exact||^2 <= 2 * sum(discarded weights). That bound is what we report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Sequence

import numpy as np

from .circuit import MWCircuit
from .mps import BlochState, MPSState
from .pauli import PauliSum
from .statevector import StatevectorState


def run_circuit(circuit: MWCircuit, method: str = "auto", chi_max: int = 64, cutoff: float = 1e-12):
    n = circuit.num_qubits
    if method == "auto":
        method = "statevector" if n <= 20 else "mps"
    if method == "statevector":
        return StatevectorState(n).run(circuit), method
    if method == "mps":
        return MPSState(n, chi_max=chi_max, cutoff=cutoff).run(circuit), method
    if method == "bloch":
        return BlochState(n, cutoff=cutoff).run(circuit), method
    raise ValueError(f"unknown method {method!r}")


@dataclass
class _Data:
    evs: Any = None
    stds: Any = None
    counts: dict | None = None

    def get_counts(self):
        return self.counts


@dataclass
class PubResult:
    data: _Data
    metadata: dict = field(default_factory=dict)


class PrimitiveResult(list):
    pass


def _as_list(obs):
    return obs if isinstance(obs, (list, tuple)) else [obs]


class MWEstimator:
    def __init__(self, method: str = "auto", chi_max: int = 64, cutoff: float = 1e-12):
        self.method, self.chi_max, self.cutoff = method, chi_max, cutoff

    def run(self, pubs: Sequence):
        out = PrimitiveResult()
        for pub in pubs:
            circuit, observables = pub[0], pub[1]
            st, used = run_circuit(circuit, self.method, self.chi_max, self.cutoff)
            evs, stds = [], []
            for ob in _as_list(observables):
                if isinstance(ob, str):
                    ob = PauliSum.from_list([(ob, 1.0)])
                if used == "statevector":
                    evs.append(ob.expectation_vector(st.to_vector()))
                    stds.append(0.0)
                else:
                    evs.append(ob.expectation_mps(st))
                    l1 = sum(abs(c) for c in ob.terms.values())
                    stds.append(2 * l1 * math.sqrt(2 * st.truncation_error))
            single = not isinstance(observables, (list, tuple))
            meta = {"method": used}
            if used != "statevector":
                meta.update(st.summary())
            out.append(PubResult(_Data(evs=evs[0] if single else np.array(evs),
                                       stds=stds[0] if single else np.array(stds)), meta))
        return _Job(out)


class MWSampler:
    def __init__(self, method: str = "auto", chi_max: int = 64, cutoff: float = 1e-12, seed=None):
        self.method, self.chi_max, self.cutoff, self.seed = method, chi_max, cutoff, seed

    def run(self, circuits: Sequence, shots: int = 1024):
        out = PrimitiveResult()
        for c in circuits:
            circuit = c[0] if isinstance(c, tuple) else c
            st, used = run_circuit(circuit, self.method, self.chi_max, self.cutoff)
            if used == "statevector":
                rng = np.random.default_rng(self.seed)
                p = st.probabilities()
                idx = rng.choice(len(p), size=shots, p=p / p.sum())
                counts: dict = {}
                for i in idx:
                    k = format(int(i), f"0{circuit.num_qubits}b")
                    counts[k] = counts.get(k, 0) + 1
                meta = {"method": used}
            else:
                counts = st.counts(shots, self.seed)
                meta = {"method": used, **st.summary()}
            out.append(PubResult(_Data(counts=counts), meta))
        return _Job(out)


class _Job:
    def __init__(self, result):
        self._r = result

    def result(self):
        return self._r

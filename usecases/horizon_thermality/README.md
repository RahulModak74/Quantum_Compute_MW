# Horizon thermality from vacuum entanglement

`horizon_entanglement.py`

**The vacuum of a quantum field, restricted to one side of a boundary, is a thermal state.** This is the
Bisognano-Wichmann theorem / Unruh effect, the flat-space counterpart of Hawking radiation. Its local
temperature is T(d) = v/(2πd) at distance d from the boundary. Entanglement entropy follows an area law
with a logarithmic correction set by the central charge (Calabrese-Cardy).

Input: only a Hamiltonian, the critical transverse-field Ising chain (a free Majorana field, c = ½, v = 2).
Nothing thermal goes in.

## H1: the local temperature of the vacuum

N = 40 chain, half-chain, entanglement Hamiltonian in 60-digit arithmetic (double precision cannot
resolve the deep modes):

| distance from boundary d | 0.5 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 |
|---|---|---|---|---|---|---|
| inverse temperature β(d), computed | 1.5705 | 3.1395 | 6.2672 | 9.3711 | 12.4393 | 15.4593 |
| Unruh / Bisognano-Wichmann, 2πd/v | 1.5708 | 3.1416 | 6.2832 | 9.4248 | 12.5664 | 15.7080 |
| finite-size CFT prediction | 1.5704 | 3.1384 | 6.2574 | 9.3378 | 12.3607 | 15.3073 |

Within 0.4 % of the CFT prediction on average (max 1.0 %) for d ≤ 5; slope 3.089 vs 2π/v = 3.142.
The vacuum seen from one side of a boundary is a thermal state with the Unruh temperature profile.

## H2: entanglement entropy law

N = 200, every cut: fitted central charge **c = 0.5135** vs exact ½ (Calabrese-Cardy log law; the 2.7 %
offset is the usual finite-lattice correction).

## H3: the same vacuum on the mwqc MPS engine

N = 100, imaginary-time TEBD, χ ≤ 32: entanglement entropy at all 99 cuts within **1.9×10⁻³ bits** of exact;
entanglement spectrum (top Schmidt weights 0.7818 / 0.2051 / 0.0100 / 0.0026) within 6×10⁻⁴ of exact
(0.7812 / 0.2056 / 0.0101 / 0.0027).

## Scope

Quantum field theory on a fixed background: spacetime is not dynamical here.

## Run

```
python3 usecases/horizon_thermality/horizon_entanglement.py          # ~15 min
python3 usecases/horizon_thermality/horizon_entanglement.py --quick  # ~3 min
```

Results: `results/horizon_thermality.json`, `results/horizon_thermality.log`.

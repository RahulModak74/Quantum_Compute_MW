# Fibre-bundle chain: a learned QFT manifold, qubit by qubit

In the MW picture each qubit is its own Bloch-sphere sub-manifold, and the qubits are joined as a
**fibre bundle**: qubit k is conditioned on the qubits before it. `mwqc/mps.py` implements this
chain as a matrix product state. The two scripts here build the same chain as a **learned**
Bayesian model, with one small Bloch sub-VAE per qubit.

| Script | Chain | Cost |
|---|---|---|
| `qft_fibre_bundle_v6.py` | qubit k conditions on the concatenated latents z₀…z₍k₋₁₎ (the explicit fibre bundle) | O(n²) |
| `qft_fibre_chain_On.py` | qubit k conditions on a fixed-size running context hₖ = Cell(zₖ, hₖ₋₁); the context width plays the role of the bond dimension χ | O(n) |

## Measured (2026-11-24, laptop, RTX 6 GB)

`python3 qft_fibre_chain_On.py --epochs 200` (trained once at n = 128, weights shared at every n):

| n | QFT circular phase error |
|---|---|
| 64 | 0.252 rad |
| 128 | 0.019 rad |
| 256 | 0.013 rad |
| 512 | 0.009 rad |

- Forward-pass scaling: n^1.04 for the running-context chain against n^1.63 for the concatenated
  chain; the O(n) chain is 2.0× faster at n = 2048.
- Entanglement readout: Haar-random input carries 2.6× the step mutual information of QFT inputs
  and is flagged as overflowing the context capacity (log₂ 32 = 5 bits per cut).

`python3 qft_fibre_bundle_v6.py --qubits 8 --quick` runs the full v6 pipeline end to end (8 Bloch
sub-VAEs, phase prediction, entanglement profile, Haar-random contrast).

## Scope

The chain is efficient for area-law (sparse) states such as the QFT of physical inputs. States
whose entanglement exceeds the context width are flagged by the readout, which is the same
sparse-Hilbert-space scope as the rest of `mwqc`.

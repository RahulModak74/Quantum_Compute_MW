#!/usr/bin/env python3
"""
Kohn-Sham use case, part 1 — Bayesian surrogate for KS eigenvalues of the
LiFePO4 Fe-O cluster proxy, trained on REAL PySCF data from the v13 runs.

Data: the PySCF caches written by SOLVE_QUANTUM/Kohn_Sham_ScreenApproach/ks_mw_inference_v13.py
  data/pyscf_cache_pbe_sto-3g_seed42.npz       200 geometries, PBE/STO-3G  (main set)
  data/pyscf_cache_pbe0_def2-svp_seed137.npz    25 geometries, PBE0/def2-SVP (too small; LOO only)
Each cache stores (28 "features", 6 valence eigenvalues). v13 never stored the geometries,
but it generated them from a seeded RNG, so they are replayed here bit-for-bit
(checked: re-running PySCF on replayed geometries reproduces the cached eigenvalues
to 3e-8 Ha).

What this script shows
  E1  The v13 feature set leaks the answer: 7 of the 28 inputs are computed from the
      very eigenvalues being predicted (band centres, splitting, gap, Fermi level...).
      The other non-constant inputs depend only on li_occupancy, which never enters the
      PySCF calculation. So v13's MAE measures how well the network inverts its own
      feature formulas, not chemistry.
  E2  The honest problem: geometry (10 interatomic distances) -> 6 eigenvalues.
      Baselines (predict-the-mean, ridge) vs MWSurrogate, 5-fold CV, in meV,
      compared with chemical accuracy (1.6 mHa = 43.6 meV).
  E3  Is the stated uncertainty honest? Coverage of +/- 2 sigma after calibration.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from mwqc.surrogate import MWSurrogate, mlp_factory     # noqa: E402

HA_TO_MEV = 27211.386
CHEM_ACC_MEV = 43.6
BASE_ATOM = "Fe 0 0 0; O 2.073 0 0; O -2.073 0 0; O 0 2.073 0; O 0 -2.073 0"
SCALE = 0.08
LEAKY = ["fe_d_band_centre", "o_p_band_centre", "exchange_splitting", "charge_transfer", "fermi_level",
         "band_gap", "dos_at_fermi"]   # all computed from the PySCF eigenvalues (HOMO/LUMO incl.)
FEATURE_NAMES = [
    "formation_energy", "band_gap", "fermi_level", "dos_at_fermi",
    "fe_d_band_centre", "o_p_band_centre", "charge_transfer", "exchange_splitting",
    "li_occupancy", "li_site_energy", "migration_barrier", "diffusivity_log",
    "butler_volmer_eta", "li_fe_distance", "po4_distortion",
    "lattice_a", "lattice_b", "lattice_c", "volume", "volume_strain",
    "bulk_modulus", "shear_modulus",
    "surface_energy", "sei_thickness", "electrochem_potential",
    "interfacial_resistance", "discharge_voltage", "capacity_retention"]


def perturb_geometry(base, scale, rng):          # identical to v13
    out = []
    for p in base.split(";"):
        tok = p.split()
        xyz = [float(v) + rng.normal(0, scale) for v in tok[1:]]
        out.append(np.array(xyz))
    return np.array(out)


def replay_geometries(seed, n):
    rng = np.random.default_rng(seed)
    geoms, li = [], []
    for _ in range(n):
        li.append(float(rng.uniform(0.5, 1.0)))
        geoms.append(perturb_geometry(BASE_ATOM, SCALE, rng))
    return np.array(geoms), np.array(li)


def distances(g):
    iu = np.triu_indices(len(g), 1)
    d = np.linalg.norm(g[:, None] - g[None], axis=-1)
    return d[iu]


def kfold(n, k, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return np.array_split(idx, k)


def ridge(Xtr, ytr, Xte, lam=1e-3):
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-12
    A, B = (Xtr - m) / s, (Xte - m) / s
    y0 = ytr.mean(0)                                   # intercept is not penalised
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (ytr - y0))
    return B @ W + y0


def mae_mev(a, b):
    return float(np.mean(np.abs(a - b)) * HA_TO_MEV)


def aufbau_penalty(y, _x):
    # v13 PRIOR 3 as a soft constraint: valence eigenvalues come out sorted
    import torch
    return torch.clamp(y[:, :-1] - y[:, 1:], min=0).pow(2).mean() * 1e4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.members, a.epochs, a.folds = 3, 200, 3

    out = {}
    d = np.load(os.path.join(HERE, "data", "pyscf_cache_pbe_sto-3g_seed42.npz"))
    F, Y = d["feats"].astype(np.float64), d["epss"].astype(np.float64)
    n = int(d["done"])
    assert n == len(Y), "cache has failed geometries; replay mapping would be off"
    G, LI = replay_geometries(42, n)
    assert np.allclose(LI, F[:, FEATURE_NAMES.index("li_occupancy")], atol=1e-6), "replay mismatch"
    D = np.array([distances(g) for g in G])
    print(f"PBE/STO-3G set: {n} geometries, eigenvalue spread (std) "
          f"{np.round(Y.std(0) * HA_TO_MEV).astype(int).tolist()} meV")

    # ── E1: leakage ──────────────────────────────────────────────────────
    folds = kfold(n, a.folds)
    li = [FEATURE_NAMES.index(k) for k in LEAKY]
    keep = [i for i in range(28) if i not in li and F[:, i].std() > 1e-9]
    e1 = {"ridge_on_v13_28_features": [], "ridge_on_v13_minus_leaky": [], "ridge_on_7_leaky_only": []}
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        e1["ridge_on_v13_28_features"].append(mae_mev(ridge(F[tr], Y[tr], F[f]), Y[f]))
        e1["ridge_on_v13_minus_leaky"].append(mae_mev(ridge(F[tr][:, keep], Y[tr], F[f][:, keep]), Y[f]))
        e1["ridge_on_7_leaky_only"].append(mae_mev(ridge(F[tr][:, li], Y[tr], F[f][:, li]), Y[f]))
    e1 = {k: float(np.mean(v)) for k, v in e1.items()}
    e1["non_constant_non_leaky_features"] = [FEATURE_NAMES[i] for i in keep]
    out["E1_leakage"] = e1
    print("\n[E1] v13 inputs leak the target")
    print(f"  linear fit on all 28 v13 features     : {e1['ridge_on_v13_28_features']:7.1f} meV")
    print(f"  linear fit on the 7 eigenvalue-derived: {e1['ridge_on_7_leaky_only']:7.1f} meV")
    print(f"  linear fit on the rest ({len(keep)} non-constant: {', '.join(e1['non_constant_non_leaky_features'])})"
          f": {e1['ridge_on_v13_minus_leaky']:7.1f} meV")

    # ── E2/E3: honest geometry -> eigenvalues ────────────────────────────
    res = {"mean_baseline": [], "ridge_distances": [], "mw_surrogate": [], "mw_surrogate_aufbau": [],
           "coverage_2sigma_raw": [], "coverage_2sigma_calibrated": [], "per_orbital": []}
    t_inf = []
    for fi, f in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), f)
        res["mean_baseline"].append(mae_mev(np.tile(Y[tr].mean(0), (len(f), 1)), Y[f]))
        res["ridge_distances"].append(mae_mev(ridge(D[tr], Y[tr], D[f]), Y[f]))
        # hold 15% of train for sigma calibration
        rng = np.random.default_rng(fi)
        perm = rng.permutation(tr)
        nval = max(5, len(tr) * 15 // 100)
        val, fit = perm[:nval], perm[nval:]
        for name, pen in (("mw_surrogate", None), ("mw_surrogate_aufbau", aufbau_penalty)):
            s = MWSurrogate(mlp_factory(hidden=64, depth=2), n_members=a.members, epochs=a.epochs,
                            lr=3e-3, batch_size=32, weight_decay=1e-4, physics_penalty=pen,
                            penalty_weight=1.0, seed=fi, device="cpu")
            s.fit(D[fit], Y[fit])
            if name == "mw_surrogate":
                res["coverage_2sigma_raw"].append(s.coverage(D[f], Y[f]))
                s.calibrate(D[val], Y[val])
                res["coverage_2sigma_calibrated"].append(s.coverage(D[f], Y[f]))
            t = time.perf_counter()
            mu, sd = s.predict(D[f])
            t_inf.append((time.perf_counter() - t) / len(f))
            res[name].append(mae_mev(mu, Y[f]))
            if name == "mw_surrogate":
                res["per_orbital"].append((np.abs(mu - Y[f]).mean(0) * HA_TO_MEV).tolist())
        print(f"  fold {fi + 1}/{len(folds)}: mean {res['mean_baseline'][-1]:.0f}  ridge {res['ridge_distances'][-1]:.0f}"
              f"  MW {res['mw_surrogate'][-1]:.0f}  MW+aufbau {res['mw_surrogate_aufbau'][-1]:.0f} meV")
    e2 = {k: float(np.mean(v)) for k, v in res.items() if k != "per_orbital"}
    e2["fold_std"] = {k: float(np.std(v)) for k, v in res.items() if k != "per_orbital"}
    e2["per_orbital_mae_meV"] = np.mean(res["per_orbital"], 0).round(1).tolist()
    e2["surrogate_inference_ms_per_geometry"] = float(np.mean(t_inf) * 1000)
    e2["pyscf_seconds_per_geometry_observed"] = "34-48 s (replay check, this laptop, PBE/STO-3G, warm-started SOSCF)"
    out["E2_geometry_to_eigenvalues"] = e2
    print("\n[E2] geometry (10 Fe-O/O-O distances) -> 6 valence KS eigenvalues, "
          f"{len(folds)}-fold CV, MAE in meV (chemical accuracy = {CHEM_ACC_MEV})")
    for k in ("mean_baseline", "ridge_distances", "mw_surrogate", "mw_surrogate_aufbau"):
        print(f"  {k:<22}: {e2[k]:7.1f} +/- {e2['fold_std'][k]:.1f}")
    print(f"  per-orbital MAE (MW): {e2['per_orbital_mae_meV']}")
    print(f"\n[E3] +/-2 sigma coverage on held-out folds (target 95%): raw "
          f"{100 * e2['coverage_2sigma_raw']:.0f}%, calibrated {100 * e2['coverage_2sigma_calibrated']:.0f}%")
    print(f"  inference {e2['surrogate_inference_ms_per_geometry']:.2f} ms/geometry vs PySCF ~40 s/geometry")

    # ── small high-level set (PBE0/def2-SVP, 25 geometries) ──────────────
    d2 = np.load(os.path.join(HERE, "data", "pyscf_cache_pbe0_def2-svp_seed137.npz"))
    Y2 = d2["epss"].astype(np.float64)
    G2, _ = replay_geometries(137, int(d2["done"]))
    D2 = np.array([distances(g) for g in G2])
    loo_mean, loo_ridge = [], []
    for i in range(len(Y2)):
        tr = np.setdiff1d(np.arange(len(Y2)), [i])
        loo_mean.append(mae_mev(Y2[tr].mean(0)[None], Y2[[i]]))
        loo_ridge.append(mae_mev(ridge(D2[tr], Y2[tr], D2[[i]], lam=10.0), Y2[[i]]))
    out["pbe0_def2svp_25"] = {"loo_mean_baseline": float(np.mean(loo_mean)),
                              "loo_ridge_distances": float(np.mean(loo_ridge)),
                              "note": "replay of seed 137 assumed 1:1 (done == len(epss)); not re-verified with PySCF"}
    print(f"\n[PBE0/def2-SVP, 25 geoms, LOO] mean baseline {np.mean(loo_mean):.0f} meV, "
          f"ridge on distances {np.mean(loo_ridge):.0f} meV  (too few points for a neural surrogate)")

    os.makedirs(os.path.join(HERE, "..", "..", "results"), exist_ok=True)
    p = os.path.join(HERE, "..", "..", "results", "ks_surrogate.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {os.path.normpath(p)}")


if __name__ == "__main__":
    main()

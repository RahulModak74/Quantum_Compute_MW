#!/usr/bin/env python3
"""
Lithography use case — MW physics-prior surrogate for EUV aerial images and CD.

  mask (64x64, 4 nm px) + defocus  --[litho_physics: Abbe, partial coherence]-->  aerial image -> CD
  mask + defocus                   --[MWSurrogate CNN ensemble]------------------> image +/- sigma -> CD +/- sigma

Physics prior as a HARD constraint (the MW "learn on the physical manifold" idea,
done literally): the intensity spectrum of any aerial image through a pupil of
radius NA/lambda is confined to |f| <= 2 NA/lambda. The `band` variant projects every
predicted image onto that set with an FFT mask, so it cannot output a physically
impossible image. The `plain` variant has no such layer. Both are compared.

Reported: image RMSE, CD MAE (nm), +/-2 sigma coverage of CD, out-of-distribution
detection (AUROC of epistemic sigma on mask families never seen in training),
and wall-clock per mask for physics vs surrogate.

Usage:  python3 usecases/lithography/litho_surrogate.py [--quick]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import litho_physics as LP                         # noqa: E402
from mwqc.surrogate import MWSurrogate              # noqa: E402

DEFOCUS_MAX = 30.0


# ── data ─────────────────────────────────────────────────────────────────────
def make_set(n_per_family, families, seed):
    rng = np.random.default_rng(seed)
    X, Y, CD, fam = [], [], [], []
    t = time.perf_counter()
    for name, fn in families.items():
        for _ in range(n_per_family):
            m = fn(rng)
            z = float(rng.uniform(-DEFOCUS_MAX, DEFOCUS_MAX))
            I, L, cd = LP.simulate(m, z)
            X.append(np.stack([m, np.full_like(m, z / DEFOCUS_MAX)]))
            Y.append(I)
            CD.append(cd)
            fam.append(name)
    dt = (time.perf_counter() - t) / max(1, len(X))
    return np.array(X, np.float32), np.array(Y, np.float32), np.array(CD), np.array(fam), dt


# ── network ──────────────────────────────────────────────────────────────────
class BandLimit(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("band", torch.as_tensor(LP.BAND_MASK))

    def forward(self, img):
        return torch.fft.ifft2(torch.fft.fft2(img) * self.band).real


def _block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, padding_mode="circular"), nn.SiLU(),
                         nn.Conv2d(co, co, 3, padding=1, padding_mode="circular"), nn.SiLU())


class LithoNet(nn.Module):
    """Small periodic U-Net (the field is periodic, as in the FFT optics)."""

    def __init__(self, band_limit: bool, w: int = 32):
        super().__init__()
        self.e1, self.e2, self.e3 = _block(2, w), _block(w, 2 * w), _block(2 * w, 4 * w)
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d2, self.d1 = _block(6 * w, 2 * w), _block(3 * w, w)
        self.head = nn.Conv2d(w, 2, 1)
        self.proj = BandLimit() if band_limit else None

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        d = self.d2(torch.cat([self.up(c), b], 1))
        e = self.d1(torch.cat([self.up(d), a], 1))
        o = self.head(e)
        mu, lv = o[:, 0], torch.clamp(o[:, 1], -12, 4)
        if self.proj is not None:
            mu = self.proj(mu)
        return mu, lv


def auroc(neg, pos):
    s = np.r_[neg, pos]
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cd_stats(members, cd_true):
    """CD measured on every member's predicted image -> mean and spread."""
    cds = np.array([[LP.cd_center_row(LP.resist_latent(img)) for img in mem] for mem in members])
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(cds, 0)
        sd = np.nanstd(cds, 0)
    ok = np.isfinite(cd_true) & np.isfinite(mu)
    print_mismatch = int(np.sum(np.isfinite(cd_true) != np.isfinite(mu)))
    return mu, sd, ok, print_mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=500, help="per family")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n_train, a.epochs, a.members = 120, 15, 2

    print(f"Optics: lambda={LP.LAMBDA_NM} nm, NA={LP.NA}, annular sigma {LP.SIGMA_IN}-{LP.SIGMA_OUT} "
          f"({len(LP.SOURCE)} source points), {LP.NX}x{LP.NX} px @ {LP.PIXEL_NM} nm, "
          f"acid blur {LP.ACID_BLUR_NM} nm, threshold {LP.THRESHOLD}")
    Xtr, Ytr, CDtr, _, t_phys = make_set(a.n_train, LP.FAMILIES, 1)
    Xva, Yva, CDva, _, _ = make_set(max(20, a.n_train // 8), LP.FAMILIES, 2)
    Xte, Yte, CDte, Fte, _ = make_set(max(40, a.n_train // 4), LP.FAMILIES, 3)
    Xood, Yood, CDood, Food, _ = make_set(max(40, a.n_train // 4), LP.OOD_FAMILIES, 4)
    print(f"Data: train {len(Xtr)}, val {len(Xva)}, test {len(Xte)}, OOD {len(Xood)}; "
          f"physics solver {1000 * t_phys:.1f} ms/mask")

    out = {"physics_ms_per_mask": 1000 * t_phys, "n_train": len(Xtr), "variants": {}}
    for variant in ("plain", "band"):
        print(f"\n== variant: {variant} ==")
        s = MWSurrogate(lambda d_in, d_out, b=(variant == "band"): LithoNet(b),
                        n_members=a.members, epochs=a.epochs, lr=2e-3, batch_size=32,
                        weight_decay=1e-5, norm="none", likelihood="mse", seed=7, verbose=True)
        s.fit(Xtr, Ytr)
        # image accuracy
        mu, sd = s.predict(Xte)
        rmse = float(np.sqrt(np.mean((mu - Yte) ** 2)))
        spec = np.abs(np.fft.fft2(mu)) ** 2
        out_of_band = float(np.sum(spec * (1 - LP.BAND_MASK)) / np.sum(spec))
        # CD accuracy + uncertainty, sigma calibrated on the validation split
        mem_va = s.predict_members(Xva)
        cmu_va, csd_va, ok_va, _ = cd_stats(mem_va, CDva)
        z = np.abs(cmu_va[ok_va] - CDva[ok_va]) / (2 * csd_va[ok_va] + 1e-9)
        cd_scale = float(np.quantile(z, 0.954))
        t = time.perf_counter()
        mem = s.predict_members(Xte)
        t_sur = (time.perf_counter() - t) / len(Xte)
        cmu, csd, ok, mism = cd_stats(mem, CDte)
        cd_err = np.abs(cmu[ok] - CDte[ok])
        cov_raw = float(np.mean(cd_err <= 2 * csd[ok]))
        cov_cal = float(np.mean(cd_err <= 2 * cd_scale * csd[ok]))
        per_fam = {f: float(np.mean(cd_err[Fte[ok] == f])) for f in LP.FAMILIES}
        # OOD
        sc_in, sc_ood = s.ood_score(Xte), s.ood_score(Xood)
        au = {f: auroc(sc_in, sc_ood[Food == f]) for f in LP.OOD_FAMILIES}
        mu_ood, _ = s.predict(Xood)
        rmse_ood = float(np.sqrt(np.mean((mu_ood - Yood) ** 2)))
        v = {"image_rmse": rmse, "image_rmse_ood": rmse_ood, "out_of_band_power_fraction": out_of_band,
             "cd_mae_nm": float(cd_err.mean()), "cd_mae_nm_by_family": per_fam,
             "cd_print_mismatch": mism, "cd_coverage_2sigma_raw": cov_raw,
             "cd_coverage_2sigma_calibrated": cov_cal, "cd_sigma_scale": cd_scale,
             "ood_auroc": au, "surrogate_ms_per_mask": 1000 * t_sur, "train_seconds": s.train_seconds}
        out["variants"][variant] = v
        print(f"  image RMSE {rmse:.4f} (in-dist), {rmse_ood:.4f} (OOD); out-of-band power {out_of_band:.2e}")
        print(f"  CD MAE {cd_err.mean():.2f} nm  by family " + ", ".join(f"{k} {x:.2f}" for k, x in per_fam.items()))
        print(f"  print/no-print mismatches: {mism}/{len(Xte)}")
        print(f"  CD +/-2 sigma coverage: raw {100 * cov_raw:.0f}%, calibrated {100 * cov_cal:.0f}% (target 95%)")
        print(f"  OOD AUROC (epistemic sigma): " + ", ".join(f"{k} {x:.3f}" for k, x in au.items()))
        print(f"  speed: physics {1000 * t_phys:.1f} ms/mask vs surrogate {1000 * t_sur:.2f} ms/mask "
              f"({t_phys / t_sur:.0f}x), training {s.train_seconds:.0f} s")

    p = os.path.join(HERE, "..", "..", "results", "litho_surrogate.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {os.path.normpath(p)}")


if __name__ == "__main__":
    main()

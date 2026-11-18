"""
MWSurrogate — physics-prior Bayesian surrogate (deep ensemble, heteroscedastic).

This is the reusable form of what the Kohn-Sham v13 and lithography scripts did
with a Pyro VAE: learn  inputs -> physics outputs  from an expensive solver
(PySCF, an optics model, ...), with
  * physics priors as soft penalties on the predictions  (`physics_penalty`)
    or as hard architecture constraints  (put them in your `net_factory`),
  * an uncertainty that is actually checked: `coverage()` reports how often the
    truth lands inside +/- k sigma on held-out data, and `calibrate()` rescales sigma
    on a validation split so the stated interval means what it says.

Uncertainty = aleatoric (each member predicts a variance) + epistemic
(disagreement between members). Epistemic sigma is the out-of-distribution
signal used for "send this one back to the real solver" (active learning).
"""
from __future__ import annotations

import copy
import math
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=128, depth=3, dropout=0.0):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            d = hidden
        self.body = nn.Sequential(*layers)
        self.mu = nn.Linear(d, d_out)
        self.lv = nn.Linear(d, d_out)

    def forward(self, x):
        h = self.body(x)
        return self.mu(h), torch.clamp(self.lv(h), -12, 4)


def mlp_factory(hidden=128, depth=3, dropout=0.0):
    return lambda d_in, d_out: _MLP(d_in, d_out, hidden, depth, dropout)


class MWSurrogate:
    def __init__(self, net_factory: Callable | None = None, n_members: int = 5,
                 epochs: int = 400, lr: float = 2e-3, batch_size: int = 64,
                 weight_decay: float = 1e-5,
                 physics_penalty: Optional[Callable] = None, penalty_weight: float = 1.0,
                 norm: str = "feature", likelihood: str = "gaussian", device: str | None = None, seed: int = 0,
                 verbose: bool = False):
        self.net_factory = net_factory or mlp_factory()
        self.n_members, self.epochs, self.lr = n_members, epochs, lr
        self.batch_size, self.weight_decay = batch_size, weight_decay
        self.physics_penalty, self.penalty_weight = physics_penalty, penalty_weight
        self.norm = norm
        # "gaussian": heteroscedastic NLL (aleatoric + epistemic). Use for noisy targets.
        # "mse": plain MSE, uncertainty = ensemble spread only. Use for DETERMINISTIC solvers
        #        (optics, DFT): there is no noise to learn, and the NLL variance head can
        #        absorb error and degrade the mean (observed on the lithography surrogate).
        if likelihood not in ("gaussian", "mse"):
            raise ValueError("likelihood must be 'gaussian' or 'mse'")
        self.likelihood = likelihood
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed, self.verbose = seed, verbose
        self.sigma_scale = 1.0
        self.members = []

    # ── normalisation ────────────────────────────────────────────────────
    def _stats(self, a: np.ndarray):
        if self.norm == "none":
            return np.float32(0.0), np.float32(1.0)
        if self.norm == "global":
            m, s = a.mean(), a.std() + 1e-8
            return np.float32(m), np.float32(s)
        return a.mean(0).astype(np.float32), (a.std(0) + 1e-8).astype(np.float32)

    def _t(self, a):
        return torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)

    # ── training ─────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float32)
        self.xm, self.xs = self._stats(X)
        self.ym, self.ys = self._stats(y)
        Xn = self._t((X - self.xm) / self.xs)
        yn = self._t((y - self.ym) / self.ys)
        ym_t, ys_t = self._t(self.ym), self._t(self.ys)
        X_raw = self._t(X) if self.physics_penalty is not None else None
        d_in = X.shape[1:] if X.ndim > 2 else X.shape[1]
        d_out = y.shape[1:] if y.ndim > 2 else y.shape[1]
        t0 = time.time()
        self.members = []
        for m in range(self.n_members):
            torch.manual_seed(self.seed + 1000 * m)
            net = self.net_factory(d_in, d_out).to(self.device)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
            # bootstrap-free ensemble: different init + different minibatch order
            g = torch.Generator(device="cpu").manual_seed(self.seed + m)
            N = len(Xn)
            for ep in range(self.epochs):
                perm = torch.randperm(N, generator=g).to(self.device)
                for s in range(0, N, self.batch_size):
                    idx = perm[s:s + self.batch_size]
                    mu, lv = net(Xn[idx])
                    # warm-up on MSE so the variance head cannot explain everything away
                    if self.likelihood == "mse" or ep < self.epochs // 4:
                        loss = ((mu - yn[idx]) ** 2).mean()
                    else:
                        loss = 0.5 * (lv + (mu - yn[idx]) ** 2 / lv.exp()).mean()
                    if self.physics_penalty is not None:
                        y_phys = mu * ys_t + ym_t
                        loss = loss + self.penalty_weight * self.physics_penalty(y_phys, X_raw[idx])
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step()
                sched.step()
            net.eval()
            self.members.append(net)
            if self.verbose:
                print(f"    member {m + 1}/{self.n_members} trained  ({time.time() - t0:.1f}s)")
        self.train_seconds = time.time() - t0
        return self

    # ── inference ────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, X: np.ndarray, return_parts: bool = False):
        Xn = self._t((np.asarray(X, np.float32) - self.xm) / self.xs)
        mus, vars_ = [], []
        for net in self.members:
            out = []
            lvs = []
            for s in range(0, len(Xn), 1024):
                mu, lv = net(Xn[s:s + 1024])
                out.append(mu)
                lvs.append(lv)
            mus.append(torch.cat(out))
            v = torch.cat(lvs).exp()
            vars_.append(torch.zeros_like(v) if self.likelihood == "mse" else v)
        mus = torch.stack(mus)
        ale = torch.stack(vars_).mean(0)
        epi = mus.var(0, unbiased=False)
        ys = self._t(self.ys)
        mean = (mus.mean(0) * ys + self._t(self.ym)).cpu().numpy()
        std = (torch.sqrt(ale + epi) * ys).cpu().numpy() * self.sigma_scale
        if return_parts:
            return mean, std, (torch.sqrt(epi) * ys).cpu().numpy(), (torch.sqrt(ale) * ys).cpu().numpy()
        return mean, std

    @torch.no_grad()
    def predict_members(self, X: np.ndarray) -> np.ndarray:
        """Per-member mean predictions, shape (n_members, N, ...) in physical units.
        Use it to push a derived quantity (e.g. a CD measured on a predicted image)
        through each member and read the spread."""
        Xn = self._t((np.asarray(X, np.float32) - self.xm) / self.xs)
        ys, ym = self._t(self.ys), self._t(self.ym)
        out = []
        for net in self.members:
            out.append(torch.cat([net(Xn[s:s + 1024])[0] for s in range(0, len(Xn), 1024)]) * ys + ym)
        return torch.stack(out).cpu().numpy()

    def calibrate(self, X_val, y_val, target: float = 0.954, k: float = 2.0):
        """Scale sigma so that a fraction `target` of validation points fall within +/- k sigma."""
        self.sigma_scale = 1.0
        mean, std = self.predict(X_val)
        z = np.abs(np.asarray(y_val) - mean) / (k * std + 1e-12)
        self.sigma_scale = float(np.quantile(z.reshape(-1), target))
        return self.sigma_scale

    def coverage(self, X, y, k: float = 2.0) -> float:
        mean, std = self.predict(X)
        return float(np.mean(np.abs(np.asarray(y) - mean) <= k * std))

    def ood_score(self, X) -> np.ndarray:
        """Mean epistemic sigma per sample (in normalised units)."""
        _, _, epi, _ = self.predict(X, return_parts=True)
        epi = epi / (np.asarray(self.ys) + 1e-12)
        return epi.reshape(len(epi), -1).mean(1)


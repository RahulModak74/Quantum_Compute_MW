"""
EUV lithography forward model (the "expensive solver" the surrogate learns).

Scalar Abbe imaging with partial coherence + defocus, Gaussian acid-diffusion
blur, constant-threshold resist, sub-pixel CD metrology.

Design choices
--------------
1. Frequency grid uses the true pixel pitch (d = PIXEL_NM), so the NA cutoff sits where the optics puts it.
2. Partially coherent annular illumination (Abbe summation over source points), as in EUV scanners.
3. Intensity normalised to the open-frame (clear mask) intensity = 1, so the resist threshold is a dose.
4. Photo-acid diffusion blur before thresholding.

Still a SCALAR (Kirchhoff thin-mask) model: no mask 3D effects, no absorber
stack, no polarisation. Rigorous EUV needs RCWA/FDTD — swap `aerial_image`
for such a solver and the surrogate code does not change.
"""
from __future__ import annotations

import math

import numpy as np

LAMBDA_NM = 13.5
NA = 0.33
PIXEL_NM = 4.0
NX = 64                      # 256 nm x 256 nm field
SIGMA_IN, SIGMA_OUT = 0.3, 0.7
ACID_BLUR_NM = 3.0
THRESHOLD = 0.30             # resist clears where blurred intensity > threshold (dose-normalised)

_f = np.fft.fftfreq(NX, d=PIXEL_NM)                     # cycles / nm
FX, FY = np.meshgrid(_f, _f, indexing="xy")
FR2 = FX ** 2 + FY ** 2
F_CUT = NA / LAMBDA_NM                                  # coherent cutoff
IMAGE_BAND = 2 * F_CUT                                  # intensity spectrum support (any source)
BAND_MASK = (np.sqrt(FR2) <= IMAGE_BAND + 1e-12).astype(np.float32)


def _source_points():
    """Annular source sampled on the frequency grid (integer bin shifts)."""
    df = 1.0 / (NX * PIXEL_NM)
    rmax = int(math.ceil(SIGMA_OUT * F_CUT / df))
    pts = []
    for i in range(-rmax, rmax + 1):
        for j in range(-rmax, rmax + 1):
            s = math.hypot(i * df, j * df) / F_CUT
            if SIGMA_IN <= s <= SIGMA_OUT:
                pts.append((i, j))
    return pts


SOURCE = _source_points()


def pupil(defocus_nm: float = 0.0) -> np.ndarray:
    P = (np.sqrt(FR2) <= F_CUT).astype(complex)
    if defocus_nm:
        P = P * np.exp(-1j * math.pi * LAMBDA_NM * defocus_nm * FR2)   # Fresnel defocus
    return P


def aerial_image(mask: np.ndarray, defocus_nm: float = 0.0) -> np.ndarray:
    """Abbe partially coherent image, normalised so a clear mask gives I = 1."""
    M = np.fft.fft2(mask.astype(np.float64))
    P = pupil(defocus_nm)
    I = np.zeros(mask.shape)
    for (i, j) in SOURCE:
        Ps = np.roll(np.roll(P, -j, axis=0), -i, axis=1)   # P(f + f_s)
        I += np.abs(np.fft.ifft2(M * Ps)) ** 2
    return (I / len(SOURCE)).astype(np.float32)


_gk = np.exp(-2 * (math.pi ** 2) * (ACID_BLUR_NM ** 2) * FR2)


def resist_latent(I: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifft2(np.fft.fft2(I) * _gk)).astype(np.float32)


def cd_center_row(latent: np.ndarray, threshold: float = THRESHOLD, dose: float = 1.0) -> float:
    """Mean width (nm) of cleared segments along the centre row, with linear
    interpolation of the threshold crossings. NaN if nothing prints."""
    row = latent[latent.shape[0] // 2] * dose - threshold
    widths, start = [], None
    for k in range(len(row) - 1):
        a, b = row[k], row[k + 1]
        if a <= 0 < b:
            start = k + a / (a - b)
        elif a > 0 >= b and start is not None:
            widths.append((k + a / (a - b)) - start)
            start = None
    return float(np.mean(widths) * PIXEL_NM) if widths else float("nan")


def simulate(mask, defocus_nm=0.0):
    I = aerial_image(mask, defocus_nm)
    L = resist_latent(I)
    return I, L, cd_center_row(L)


# ── mask families ────────────────────────────────────────────────────────────
def mask_line_space(rng):
    pitch = rng.uniform(32, 72)
    duty = rng.uniform(0.35, 0.6)
    x = (np.arange(NX) + 0.5) * PIXEL_NM + rng.uniform(0, pitch)
    m = ((x % pitch) < duty * pitch).astype(np.float32)
    return np.tile(m, (NX, 1))


def mask_contacts(rng):
    pitch = rng.uniform(40, 80)
    size = rng.uniform(0.35, 0.6) * pitch
    c = (np.arange(NX) + 0.5) * PIXEL_NM
    ox, oy = rng.uniform(0, pitch, 2)
    mx = ((c + ox) % pitch) < size
    my = ((c + oy) % pitch) < size
    return (my[:, None] & mx[None, :]).astype(np.float32)


def mask_iso_with_sraf(rng):
    m = np.zeros((NX, NX), np.float32)
    w = int(round(rng.uniform(16, 32) / PIXEL_NM))
    c = NX // 2 + int(rng.integers(-3, 4))
    m[:, c - w // 2: c - w // 2 + w] = 1
    if rng.random() < 0.7:                                 # sub-resolution assist features
        gap = int(round(rng.uniform(24, 40) / PIXEL_NM))
        sw = max(1, int(round(rng.uniform(4, 8) / PIXEL_NM)))
        for s in (-1, 1):
            p = c + s * (gap + w // 2)
            if 0 <= p < NX - sw:
                m[:, p:p + sw] = 1
    return m


def mask_line_end(rng):
    """Vertical lines that END a few pixels from the centre row: the measured CD
    includes line-end pull-back, a genuinely 2D proximity effect."""
    m = mask_line_space(rng)
    cut = NX // 2 + int(rng.integers(-6, 7))
    if rng.random() < 0.5:
        m[cut:, :] = 0
    else:
        m[:cut, :] = 0
    return m


FAMILIES = {"line_space": mask_line_space, "contacts": mask_contacts,
            "iso_sraf": mask_iso_with_sraf, "line_end": mask_line_end}


# out-of-distribution families (never in training)
def mask_diagonal(rng):
    pitch = rng.uniform(40, 70)
    y, x = np.mgrid[0:NX, 0:NX] * PIXEL_NM
    return (((x + y) / math.sqrt(2) % pitch) < 0.5 * pitch).astype(np.float32)


def mask_random_pixels(rng):
    return (rng.random((NX, NX)) < 0.5).astype(np.float32)


OOD_FAMILIES = {"diagonal": mask_diagonal, "random_pixels": mask_random_pixels}

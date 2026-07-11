"""preprocess.py
Spectrum cropping, normalisation (MAX / AREA / SNV), and LIR computation.

All functions operate on 2-D arrays (n_spectra × n_channels) and are
stateless — they take wavelengths as an explicit argument so they can be
used both on raw-loaded data and on subsets.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Normalization(str, Enum):
    MAX = "MAX"
    AREA = "AREA"
    SNV = "SNV"
    NONE = "NONE"


@dataclass
class PreprocessResult:
    spectra_norm: np.ndarray       # (N, n_ch) normalised spectra (cropped)
    spectra_raw_crop: np.ndarray   # (N, n_ch) raw intensities (cropped, scaled)
    wavelengths_crop: np.ndarray   # (n_ch,)
    n_zero_std: int                # number of SNV-zero-std replacements
    norm: Normalization


def crop(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    wl_min: float = 900.0,
    wl_max: float = 1100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (spectra_cropped, wavelengths_cropped)."""
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    return spectra[:, mask], wavelengths[mask]


def normalize(
    spectra: np.ndarray,
    method: Normalization | str,
    bg_subtract: bool = False,
) -> tuple[np.ndarray, int]:
    """Normalise a (N, n_ch) array of spectra.

    Returns (normalised, n_zero_std) where n_zero_std is 0 for all
    methods except SNV.

    With bg_subtract=True each spectrum has its per-spectrum minimum
    subtracted before normalisation (mirrors the LIR pre-processing
    convention).
    """
    method = Normalization(method)
    S = spectra.copy()
    if bg_subtract:
        S -= S.min(axis=1, keepdims=True)

    if method is Normalization.NONE:
        return S, 0

    if method is Normalization.MAX:
        mx = S.max(axis=1, keepdims=True)
        mx = np.where(mx == 0, 1.0, mx)
        return S / mx, 0

    if method is Normalization.AREA:
        area = S.sum(axis=1, keepdims=True)
        area = np.where(area == 0, 1.0, area)
        return S / area, 0

    # SNV
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True)
    n_zero = int((sd == 0).sum())
    sd_safe = np.where(sd == 0, 1.0, sd)
    return (S - mu) / sd_safe, n_zero


def preprocess(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    wl_min: float = 900.0,
    wl_max: float = 1100.0,
    method: Normalization | str = Normalization.SNV,
    bg_subtract: bool = False,
) -> PreprocessResult:
    """Crop then normalise spectra."""
    spectra_crop, wl_crop = crop(spectra, wavelengths, wl_min, wl_max)
    spectra_norm, n_zero = normalize(spectra_crop, method, bg_subtract)
    return PreprocessResult(
        spectra_norm=spectra_norm,
        spectra_raw_crop=spectra_crop,
        wavelengths_crop=wl_crop,
        n_zero_std=n_zero,
        norm=Normalization(method),
    )


# ─── LIR ─────────────────────────────────────────────────────────────────

@dataclass
class LIRConfig:
    hi_lo: float = 900.0
    hi_hi: float = 983.0
    low_lo: float = 983.0
    low_hi: float = 987.0


def compute_lir(
    spectra_raw_crop: np.ndarray,
    wavelengths_crop: np.ndarray,
    cfg: LIRConfig | None = None,
) -> np.ndarray:
    """Compute background-subtracted MAX-normalised LIR for every spectrum.

    Uses raw (not SNV) cropped intensities to mirror the pipeline convention:
      1. bg = per-spectrum minimum subtracted
      2. MAX normalise
      3. LIR = Σ(I, band_hi) / Σ(I, band_low)

    Returns (N,) array.
    """
    if cfg is None:
        cfg = LIRConfig()
    hi_mask = (wavelengths_crop >= cfg.hi_lo) & (wavelengths_crop <= cfg.hi_hi)
    low_mask = (wavelengths_crop >= cfg.low_lo) & (wavelengths_crop <= cfg.low_hi)
    S_bg = spectra_raw_crop - spectra_raw_crop.min(axis=1, keepdims=True)
    S_mx = S_bg.max(axis=1, keepdims=True)
    S_max = S_bg / np.where(S_mx == 0, 1.0, S_mx)
    hi_sum = S_max[:, hi_mask].sum(axis=1)
    lo_sum = np.maximum(S_max[:, low_mask].sum(axis=1), 1e-30)
    return hi_sum / lo_sum


def fit_lir_quad(
    lir_vals: np.ndarray,
    temperatures: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    """Fit a quadratic log(LIR) vs 1000/T calibration on training spectra.

    Returns the 3 polynomial coefficients [p2, p1, p0].
    """
    train_T_unique = np.unique(temperatures[train_mask])
    lir_mean_per_T = np.array(
        [lir_vals[train_mask & (temperatures == t)].mean() for t in train_T_unique]
    )
    u = 1000.0 / train_T_unique
    log_lir = np.log(np.maximum(lir_mean_per_T, 1e-30))
    return np.polyfit(u, log_lir, 2)


def predict_T_lir_quad(
    lir_vals: np.ndarray,
    poly_coeffs: np.ndarray,
    fallback_T: float,
) -> np.ndarray:
    """Invert quadratic log(LIR)=p2u²+p1u+p0 for u=1000/T."""
    p2, p1, p0 = poly_coeffs
    log_l = np.log(np.maximum(lir_vals, 1e-30))
    disc = p1**2 - 4.0 * p2 * (p0 - log_l)
    disc = np.where(disc < 0, np.nan, disc)
    u = (-p1 - np.sqrt(disc)) / (2.0 * p2)
    safe_u = np.where(np.abs(u) < 1e-12, np.nan, u)
    T_pred = 1000.0 / safe_u
    return np.where(np.isfinite(T_pred), T_pred, fallback_T)


def fit_lir_boltzmann_linear(
    lir_vals: np.ndarray,
    temperatures: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    """Linear Boltzmann-style calibration: ``log(LIR_mean) = b1·(1000/T) + b0``.

    Returns ``[b1, b0]`` (same convention as :func:`numpy.polyfit` on ``u``).
    """
    train_T_unique = np.unique(temperatures[train_mask])
    lir_mean_per_T = np.array(
        [lir_vals[train_mask & (temperatures == t)].mean() for t in train_T_unique]
    )
    u = 1000.0 / train_T_unique
    log_lir = np.log(np.maximum(lir_mean_per_T, 1e-30))
    return np.polyfit(u, log_lir, 1)


def predict_T_lir_boltzmann(
    lir_vals: np.ndarray,
    linear_coeffs: np.ndarray,
    fallback_T: float,
) -> np.ndarray:
    """Invert ``log(LIR) = b1·u + b0`` with ``u = 1000/T`` → ``T = 1000 / u``."""
    b1, b0 = float(linear_coeffs[0]), float(linear_coeffs[1])
    log_l = np.log(np.maximum(lir_vals, 1e-30))
    u = np.where(np.abs(b1) < 1e-30, np.nan, (log_l - b0) / b1)
    T_pred = 1000.0 / u
    return np.where(np.isfinite(T_pred) & (T_pred > 0), T_pred, fallback_T)

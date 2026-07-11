"""Tests for preprocess.py."""
import numpy as np
import pytest

from lt2_core.preprocess import (
    Normalization,
    LIRConfig,
    crop,
    normalize,
    preprocess,
    compute_lir,
    fit_lir_quad,
    predict_T_lir_quad,
)


def _make_spectra(n=10, n_ch=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, n_ch)) + 0.1


def _make_wavelengths(n_ch=200, lo=850.0, hi=1150.0):
    return np.linspace(lo, hi, n_ch)


# ── crop ──────────────────────────────────────────────────────────────────

def test_crop_range():
    wl = _make_wavelengths()
    S = _make_spectra(n_ch=len(wl))
    S_crop, wl_crop = crop(S, wl, 900, 1100)
    assert wl_crop.min() >= 900
    assert wl_crop.max() <= 1100
    assert S_crop.shape[1] == wl_crop.size


# ── normalize ─────────────────────────────────────────────────────────────

def test_snv_zero_mean_unit_std():
    S = _make_spectra()
    S_norm, n_zero = normalize(S, Normalization.SNV)
    np.testing.assert_allclose(S_norm.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(S_norm.std(axis=1), 1, atol=1e-12)
    assert n_zero == 0


def test_max_normalization():
    S = _make_spectra()
    S_norm, _ = normalize(S, Normalization.MAX)
    np.testing.assert_allclose(S_norm.max(axis=1), 1.0, atol=1e-12)


def test_area_normalization():
    S = _make_spectra()
    S_norm, _ = normalize(S, Normalization.AREA)
    np.testing.assert_allclose(S_norm.sum(axis=1), 1.0, atol=1e-12)


def test_snv_zero_std_guard():
    """A constant spectrum should not produce NaN."""
    S = np.ones((5, 50))
    S_norm, n_zero = normalize(S, Normalization.SNV)
    assert n_zero == 5
    assert not np.any(np.isnan(S_norm))


# ── preprocess ───────────────────────────────────────────────────────────

def test_preprocess_pipeline():
    wl = _make_wavelengths()
    S = _make_spectra(n_ch=len(wl))
    res = preprocess(S, wl, 900, 1100, Normalization.SNV)
    assert res.wavelengths_crop.min() >= 900
    assert res.spectra_norm.shape == res.spectra_raw_crop.shape
    assert res.spectra_norm.shape[1] == res.wavelengths_crop.size


# ── LIR ──────────────────────────────────────────────────────────────────

def test_lir_positive():
    wl = _make_wavelengths(n_ch=300, lo=850, hi=1150)
    S = _make_spectra(n=20, n_ch=len(wl))
    S_crop, wl_crop = crop(S, wl, 900, 1000)
    lir = compute_lir(S_crop, wl_crop, LIRConfig(hi_lo=900, hi_hi=960, low_lo=960, low_hi=990))
    assert lir.shape == (20,)
    assert (lir > 0).all()


def test_lir_quad_fit_recovers_coefficients():
    """Verify that fit_lir_quad recovers a known quadratic calibration."""
    n = 40
    T = np.linspace(300, 700, n)
    u = 1000.0 / T
    # Coefficients that match the physical sign of the real data
    true_coeffs = np.array([0.3, -1.5, 4.0])
    log_lir = np.polyval(true_coeffs, u)
    lir = np.exp(log_lir)

    train_mask = np.ones(n, dtype=bool)
    fit_c = fit_lir_quad(lir, T, train_mask)

    # Fitted coefficients should reproduce the calibration curve closely
    log_lir_pred = np.polyval(fit_c, u)
    residual = np.abs(log_lir - log_lir_pred)
    assert residual.max() < 0.01, f"LIR calibration fit error: {residual.max():.4f}"


def test_predict_T_lir_fallback_for_nan():
    """predict_T_lir_quad should never return NaN; bad inputs use fallback."""
    fallback = 500.0
    # Force discriminant < 0 by passing extreme LIR values
    coeffs = np.array([1.0, 0.0, 0.0])  # log(LIR) = u²; impossible to invert for some log_lir
    lir_bad = np.array([1e-30])  # log(LIR) ≈ -69; disc will be negative with coeffs above
    T_pred = predict_T_lir_quad(lir_bad, coeffs, fallback)
    assert np.isfinite(T_pred).all(), "predict_T_lir_quad returned non-finite value"

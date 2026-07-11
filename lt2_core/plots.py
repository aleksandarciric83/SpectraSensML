"""plots.py — figure builders for LT2 thermometry results.

This module emits only the new ``Figure N`` set agreed in the chat:

  Figure 1   - Mean Raw Spectra Per Temperature (Cropped)
  Figure 2   - Mean Normalised Spectra Per Temperature (Cropped)
  Figure 3   - LIR diagnostic                              (a–j)
  Figure 4   - PCA analysis                                (a–d)
  Figure 5   - True vs Predicted (all models)
  Figure 6   - RMSE summary                                (a,b)
  Figure 7   - Bias summary                                (a,b)
  Figure 8   - Precision (σ) summary                       (a,b)
  Figure 9   - MAE summary                                 (a,b)
  Figure 10  - MaxAbs summary                              (a,b)
  Figure 11  - P95Abs summary                              (a,b)
  Figure 12–17  - Top-5 zoom for each metric (3 sets)      (a,b)
  Figure 18–23  - Top-5 zoom for each metric (val + test + mean(val,test))  (a,b)
  Figure 24  - Sensitivity of best model

Each PNG comes with one CSV per sub-panel sharing the same base name
(e.g. ``Figure 3a.csv``, ``Figure 6b.csv``).  CSVs contain the data
actually plotted in that sub-panel.

Conventions:
  - dpi=600 raster export by default.
  - Temperature colormap ``nipy_spectral`` (same as the pipeline).
  - Ordering by ``avg(GlobalRMSE_val, GlobalRMSE_test_unseen)`` ascending,
    tie-break by test_unseen RMSE then model name (matches the leaderboard).
"""
from __future__ import annotations

import os
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .metrics import (
    METRIC_LABELS,
    METRIC_NAMES,
    metric_signed,
    per_T_table,
)


DPI_EXPORT = 600
CMAP_T = "nipy_spectral"
SETS = ("train", "val", "test_unseen")
SET_COLORS = ("tab:blue", "tab:orange", "tab:red")
SET_LABELS = ("Train", "Val", "Test_unseen")

# Default labels — overridden per call via the `quantity` dict.
DEFAULT_VAR_NAME = "T"
DEFAULT_VAR_UNIT = "K"


def _q(quantity: dict | None) -> tuple[str, str]:
    """Resolve (var_name, var_unit) with safe fallbacks."""
    if not quantity:
        return DEFAULT_VAR_NAME, DEFAULT_VAR_UNIT
    name = str(quantity.get("name") or DEFAULT_VAR_NAME).strip() or DEFAULT_VAR_NAME
    unit = str(quantity.get("unit") or DEFAULT_VAR_UNIT).strip() or DEFAULT_VAR_UNIT
    return name, unit


def _axis_label(var_name: str, var_unit: str) -> str:
    return f"True {var_name} ({var_unit})"


def _pred_label(var_name: str, var_unit: str) -> str:
    return f"Predicted {var_name} ({var_unit})"


def _metric_label(metric: str, var_unit: str) -> str:
    """Substitute the unit token in METRIC_LABELS for the user's unit."""
    base = METRIC_LABELS[metric]
    return base.replace("(K)", f"({var_unit})")


# ─── small utilities ─────────────────────────────────────────────────────


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# Mutable default extension used by _save_fig. ``export_all`` temporarily flips
# this to "svg" so every figure builder transparently emits vector output.
_SAVE_FIG_EXT = "png"


def _save_fig(fig: Figure, png_path: str, dpi: int = DPI_EXPORT) -> None:
    """Save *fig*. Path may have a ``.png`` suffix from the builder; this
    helper rewrites the extension when ``_SAVE_FIG_EXT`` is set to ``svg``.
    Returns nothing — callers track the original path for bookkeeping; we
    also write a sibling file with the actual extension used."""
    _ensure_dir(png_path)
    root, _ext = os.path.splitext(png_path)
    target_ext = ".svg" if _SAVE_FIG_EXT.lower() == "svg" else ".png"
    out_path = root + target_ext
    if target_ext == ".svg":
        fig.savefig(out_path, bbox_inches="tight")
    else:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")


def _save_csv(df: pd.DataFrame, csv_path: str) -> None:
    _ensure_dir(csv_path)
    df.to_csv(csv_path, index=False, float_format="%.6g")


def _temp_norm(temperatures: np.ndarray):
    return plt.Normalize(temperatures.min(), temperatures.max())


def _split_y(ds):
    return (
        ds.temperatures[ds.train_mask],
        ds.temperatures[ds.val_mask],
        ds.temperatures[ds.test_mask],
    )


def _avg_val_test_rmse(results, mname: str) -> float:
    mr = results.models[mname]
    v = float(mr.metrics.get("val", {}).get("RMSE", np.inf))
    t = float(mr.metrics.get("test_unseen", {}).get("RMSE", np.inf))
    return 0.5 * (v + t)


def order_models(results) -> list[str]:
    """avg(val,test) ascending → tie-break by test RMSE → name."""

    def key(m: str):
        mr = results.models[m]
        return (
            _avg_val_test_rmse(results, m),
            float(mr.metrics.get("test_unseen", {}).get("RMSE", np.inf)),
            m,
        )

    return sorted(results.models.keys(), key=key)


def top_n(results, n: int = 5) -> list[str]:
    return order_models(results)[:n]


def _set_metric(mr, set_name: str, metric: str) -> float:
    """Pull a §B sample-level metric value for a model/split."""
    return float(mr.metrics.get(set_name, {}).get(metric, np.nan))


def _per_T_values(mr, set_name: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Pull per-T arrays (T_K, value) from cached §A rows."""
    rows = mr.per_T.get(set_name, []) or []
    if not rows:
        return np.array([]), np.array([])
    T = np.array([r["T_K"] for r in rows], dtype=float)
    V = np.array([float(r.get(metric, np.nan)) for r in rows], dtype=float)
    order = np.argsort(T)
    return T[order], V[order]


# =========================================================================
# Figure 1  —  Mean Raw Spectra Per Temperature (Cropped)
# =========================================================================


def make_figure_1(prep, ds, out_dir: str, dpi: int = DPI_EXPORT,
                  quantity: dict | None = None) -> str:
    var_name, var_unit = _q(quantity)
    base = f"Figure 1 - Mean Raw Spectra Per {var_name} (Cropped)"
    png = os.path.join(out_dir, base + ".png")
    csv = os.path.join(out_dir, "Figure 1.csv")

    lam = prep.wavelengths_crop
    S = prep.spectra_raw_crop
    T_all = ds.temperatures
    T_unique = np.sort(np.unique(T_all))
    norm = _temp_norm(T_unique)
    cmap = plt.get_cmap(CMAP_T)

    fig, ax = plt.subplots(figsize=(10, 5))
    rows: list[dict] = []
    var_col = f"{var_name}_{var_unit}"
    for T in T_unique:
        mean_spec = S[T_all == T].mean(axis=0)
        ax.plot(lam, mean_spec, color=cmap(norm(T)), lw=1.0, alpha=0.9)
        for w, v in zip(lam, mean_spec):
            rows.append({var_col: float(T), "wavelength_nm": float(w),
                         "mean_intensity": float(v)})
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=f"{var_name} ({var_unit})")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean intensity (raw)")
    ax.set_title(f"Figure 1 — Mean raw spectrum per {var_name} (cropped)")
    fig.tight_layout()
    _save_fig(fig, png, dpi)
    plt.close(fig)
    _save_csv(pd.DataFrame(rows), csv)
    return png


# =========================================================================
# Figure 2  —  Mean Normalised Spectra Per Temperature (Cropped)
# =========================================================================


def make_figure_2(prep, ds, out_dir: str, dpi: int = DPI_EXPORT,
                  quantity: dict | None = None) -> str:
    var_name, var_unit = _q(quantity)
    norm_label = str(getattr(prep.norm, "value", prep.norm))
    base = f"Figure 2 - Mean Normalised Spectra Per {var_name} (Cropped)"
    png = os.path.join(out_dir, base + ".png")
    csv = os.path.join(out_dir, "Figure 2.csv")

    lam = prep.wavelengths_crop
    S = prep.spectra_norm
    T_all = ds.temperatures
    T_unique = np.sort(np.unique(T_all))
    norm = _temp_norm(T_unique)
    cmap = plt.get_cmap(CMAP_T)

    fig, ax = plt.subplots(figsize=(10, 5))
    rows: list[dict] = []
    var_col = f"{var_name}_{var_unit}"
    for T in T_unique:
        mean_spec = S[T_all == T].mean(axis=0)
        ax.plot(lam, mean_spec, color=cmap(norm(T)), lw=1.0, alpha=0.9)
        for w, v in zip(lam, mean_spec):
            rows.append({var_col: float(T), "wavelength_nm": float(w),
                         "mean_intensity_norm": float(v)})
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=f"{var_name} ({var_unit})")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(f"Mean intensity ({norm_label} normalised)")
    ax.set_title(
        f"Figure 2 — Mean {norm_label}-normalised spectrum per {var_name} (cropped)"
    )
    fig.tight_layout()
    _save_fig(fig, png, dpi)
    plt.close(fig)
    _save_csv(pd.DataFrame(rows), csv)
    return png


# =========================================================================
# Figure 3  —  LIR diagnostic   (a) raw mean + bands, (b) log(LIR) vs 1000/T,
#  (c) LIR vs T with val/test means, (d) predicted vs true,
#  (e..) per-T metric panels for the LIR quadratic model.
# =========================================================================


def _compute_lir_bg(prep, lir_cfg) -> np.ndarray:
    """LIR using background subtraction only (no normalisation)."""
    lam = prep.wavelengths_crop
    S = prep.spectra_raw_crop
    S_bg = S - S.min(axis=1, keepdims=True)
    hi_mask = (lam >= float(lir_cfg.hi_lo)) & (lam <= float(lir_cfg.hi_hi))
    lo_mask = (lam >= float(lir_cfg.low_lo)) & (lam <= float(lir_cfg.low_hi))
    I_hi = S_bg[:, hi_mask].sum(axis=1)
    I_lo = S_bg[:, lo_mask].sum(axis=1)
    denom = np.where(np.abs(I_lo) < 1e-30, np.nan, I_lo)
    return I_hi / denom


def make_figure_3(results, prep, out_dir: str, dpi: int = DPI_EXPORT,
                  quantity: dict | None = None) -> str | None:
    from .preprocess import LIRConfig

    var_name, var_unit = _q(quantity)
    base_png = "Figure 3 - LIR diagnostic.png"
    png = os.path.join(out_dir, base_png)

    cfg = (results.config.lir_cfg if results.config is not None else LIRConfig())
    ds = results.dataset
    if ds is None:
        return None

    KB_CM_INV_PER_K = 0.69503476

    all_T = ds.temperatures.astype(float)
    all_lab = ds.labels.astype(str)
    tr_mask = all_lab == "train"
    va_mask = all_lab == "val"
    te_mask = all_lab == "test_unseen"

    lam = prep.wavelengths_crop
    S = prep.spectra_raw_crop
    S_bg = S - S.min(axis=1, keepdims=True)
    LIR = _compute_lir_bg(prep, cfg)
    lir_hi = (float(cfg.hi_lo), float(cfg.hi_hi))
    lir_low = (float(cfg.low_lo), float(cfg.low_hi))

    T_tr_unique = np.sort(np.unique(all_T[tr_mask]))
    LIR_tr_mean = np.array([LIR[tr_mask & (all_T == t)].mean() for t in T_tr_unique])
    inv_T = 1000.0 / T_tr_unique
    log_LIR = np.log(np.maximum(LIR_tr_mean, 1e-30))

    t_lin_min = T_tr_unique[len(T_tr_unique) // 2] if len(T_tr_unique) >= 4 else T_tr_unique.min()
    hi_T_mask = T_tr_unique >= t_lin_min
    inv_T_hi = 1000.0 / T_tr_unique[hi_T_mask]
    log_LIR_hi = np.log(np.maximum(LIR_tr_mean[hi_T_mask], 1e-30))
    polBtz_hi = np.polyfit(inv_T_hi, log_LIR_hi, 1) if hi_T_mask.sum() >= 2 else np.array([0.0, 0.0])
    a_lin_hi, b_lin_hi = float(polBtz_hi[0]), float(polBtz_hi[1])
    de_lir_hi = -1000.0 * a_lin_hi * KB_CM_INV_PER_K

    polBtz_full = np.polyfit(inv_T, log_LIR, 1) if len(inv_T) >= 2 else np.array([0.0, 0.0])
    pol = np.polyfit(inv_T, log_LIR, 2) if len(inv_T) >= 3 else np.array([0.0, 0.0, 0.0])
    p2, p1, p0 = float(pol[0]), float(pol[1]), float(pol[2])

    def predict_T_linear(lir: np.ndarray) -> np.ndarray:
        log_lir = np.log(np.maximum(lir, 1e-30))
        denom = log_lir - b_lin_hi
        safe = np.where(np.abs(denom) < 1e-12, np.nan, denom)
        return 1000.0 * a_lin_hi / safe

    def predict_T_quadratic(lir: np.ndarray) -> np.ndarray:
        log_lir = np.log(np.maximum(lir, 1e-30))
        disc = p1 ** 2 - 4.0 * p2 * (p0 - log_lir)
        disc = np.where(disc < 0, np.nan, disc)
        if p2 == 0.0:
            return np.full_like(log_lir, np.nan, dtype=float)
        u = (-p1 - np.sqrt(disc)) / (2.0 * p2)
        safe_u = np.where(np.abs(u) < 1e-12, np.nan, u)
        return 1000.0 / safe_u

    T_pred_lin = predict_T_linear(LIR)
    T_pred_quad = predict_T_quadratic(LIR)

    cmap = plt.get_cmap(CMAP_T)
    norm = plt.Normalize(T_tr_unique.min(), T_tr_unique.max())
    n_metric_panels = len(METRIC_NAMES)
    n_top_panels = 4
    n_total = n_top_panels + n_metric_panels
    ncols = 3
    nrows = int(np.ceil(n_total / ncols))
    fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.35, wspace=0.28)
    letters = "abcdefghijklmnopqrstuvwxyz"

    panel_csvs: list[tuple[str, pd.DataFrame]] = []

    # ── (a) mean raw (bg-subtracted) spectra per T with HI/LOW bands ────
    ax_a = fig.add_subplot(gs[0, 0])
    rows_a: list[dict] = []
    for t in T_tr_unique:
        m = tr_mask & (all_T == t)
        if not m.any():
            continue
        spec = S_bg[m].mean(axis=0)
        ax_a.plot(lam, spec, color=cmap(norm(t)), lw=1.0, alpha=0.85)
        for w, v in zip(lam, spec):
            rows_a.append({"T_K": float(t), "wavelength_nm": float(w),
                           "mean_intensity_bg": float(v)})
    ax_a.axvspan(*lir_hi, color="tab:red", alpha=0.10,
                 label=f"HI {lir_hi[0]:.0f}–{lir_hi[1]:.0f} nm")
    ax_a.axvspan(*lir_low, color="tab:blue", alpha=0.20,
                 label=f"LOW {lir_low[0]:.0f}–{lir_low[1]:.0f} nm")
    ax_a.set_xlabel("Wavelength (nm)")
    ax_a.set_ylabel("Mean intensity (background-subtracted)")
    ax_a.set_title("(a) Mean spectra per training T with LIR bands")
    ax_a.legend(loc="upper right", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax_a, label=f"{var_name} ({var_unit})", pad=0.01)
    panel_csvs.append(("Figure 3a.csv", pd.DataFrame(rows_a)))

    # ── (b) log(LIR) vs 1000/T  (linear + linear-hi + quadratic) ────────
    ax_b = fig.add_subplot(gs[0, 1])
    inv_T_dense = np.linspace(inv_T.min() * 0.98, inv_T.max() * 1.02, 400)
    hi_pts = T_tr_unique >= t_lin_min
    ax_b.plot(inv_T[~hi_pts], log_LIR[~hi_pts], "o", color="0.6", ms=7, mec="k",
              label=f"train per-{var_name} mean ({var_name}<{t_lin_min:.1f} {var_unit})")
    ax_b.plot(inv_T[hi_pts], log_LIR[hi_pts], "D", color="tab:blue", ms=8, mec="k",
              label=f"train per-{var_name} mean ({var_name}≥{t_lin_min:.1f} {var_unit})")
    if hi_T_mask.sum() >= 2:
        inv_T_hi_d = np.linspace(inv_T_hi.min() * 0.98, inv_T_hi.max() * 1.02, 200)
        ax_b.plot(inv_T_hi_d, np.polyval(polBtz_hi, inv_T_hi_d), "-",
                  color="tab:red", lw=2.4,
                  label=f"linear, fit {var_name}≥{t_lin_min:.1f} {var_unit}  (slope ≈ {polBtz_hi[0]:.2f})")
    if len(inv_T) >= 2:
        ax_b.plot(inv_T_dense, np.polyval(polBtz_full, inv_T_dense), ":",
                  color="tab:red", lw=1.6, label="linear, full range")
    if len(inv_T) >= 3:
        ax_b.plot(inv_T_dense, np.polyval(pol, inv_T_dense), "--",
                  color="tab:green", lw=2.2, label="quadratic, full range")
    ax_b.set_xlabel(f"1000 / {var_name}  ({var_unit}⁻¹)")
    ax_b.set_ylabel("log(LIR)")
    ax_b.set_title(f"(b) log(LIR) vs 1000/{var_name}")
    ax_b.legend(loc="best", fontsize=8)
    ax_b.grid(True, alpha=0.3)
    panel_csvs.append((
        "Figure 3b.csv",
        pd.DataFrame({f"{var_name}_{var_unit}": T_tr_unique,
                      f"inv_{var_name}_1000_per_{var_unit}": inv_T,
                      "log_LIR": log_LIR}),
    ))

    # ── (c) LIR vs T  raw + train mean + val/test means (no fit) ────────
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.scatter(all_T[tr_mask], LIR[tr_mask], s=4, alpha=0.25,
                 color="tab:gray", label="train spectra")
    ax_c.plot(T_tr_unique, LIR_tr_mean, "D-", color="tab:red", lw=2, ms=7,
              label=f"train per-{var_name} mean")
    rows_c: list[dict] = []
    var_col = f"{var_name}_{var_unit}"
    for t in T_tr_unique:
        rows_c.append({"set": "train_mean", var_col: float(t),
                       "LIR": float(LIR_tr_mean[T_tr_unique == t][0])})
    for set_name, mask, color, marker in (
        ("val", va_mask, "tab:orange", "s"),
        ("test_unseen", te_mask, "tab:blue", "^"),
    ):
        T_u = np.sort(np.unique(all_T[mask]))
        if T_u.size == 0:
            continue
        means = np.array([LIR[mask & (all_T == t)].mean() for t in T_u])
        ax_c.plot(T_u, means, marker + "-", color=color, ms=7, lw=1.4,
                  label=f"{set_name} per-{var_name} mean")
        for t, v in zip(T_u, means):
            rows_c.append({"set": set_name, var_col: float(t), "LIR": float(v)})
    ax_c.set_xlabel(_axis_label(var_name, var_unit))
    ax_c.set_ylabel("LIR")
    ax_c.set_title(f"(c) LIR vs {var_name}  (raw + per-{var_name} means)")
    ax_c.legend(loc="best", fontsize=8)
    ax_c.grid(True, alpha=0.3)
    panel_csvs.append(("Figure 3c.csv", pd.DataFrame(rows_c)))

    # ── (d) predicted vs true ──────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    rows_d: list[dict] = []
    y_true_col = f"y_true_{var_unit}"
    y_pred_col = f"y_pred_{var_unit}"
    for set_name, mask, color, marker in (
        ("val", va_mask, "tab:orange", "o"),
        ("test_unseen", te_mask, "tab:red", "o"),
    ):
        ax_d.scatter(all_T[mask], T_pred_quad[mask], s=10, alpha=0.55,
                     color=color, marker=marker, label=f"QUAD — {set_name}")
        for yt, yp in zip(all_T[mask], T_pred_quad[mask]):
            rows_d.append({"model": "LIR_QUAD", "set": set_name,
                           y_true_col: float(yt), y_pred_col: float(yp)})
        ax_d.scatter(all_T[mask], T_pred_lin[mask], s=10, alpha=0.55,
                     color=color, marker="x", label=f"LIN — {set_name}")
        for yt, yp in zip(all_T[mask], T_pred_lin[mask]):
            rows_d.append({"model": "LIR_LIN", "set": set_name,
                           y_true_col: float(yt), y_pred_col: float(yp)})
    T_line = np.array([all_T.min() - 5, all_T.max() + 5])
    ax_d.plot(T_line, T_line, "k--", lw=1, label="identity")
    ax_d.set_xlabel(_axis_label(var_name, var_unit))
    ax_d.set_ylabel(_pred_label(var_name, var_unit))
    ax_d.set_title(f"(d) Predicted vs True {var_name}  (LIR LIN/QUAD)")
    ax_d.legend(loc="best", fontsize=7, ncol=2)
    ax_d.grid(True, alpha=0.3)
    panel_csvs.append(("Figure 3d.csv", pd.DataFrame(rows_d)))

    # ── (e..) per-T metric panels for the QUAD model ────────────────────
    y_train = all_T[tr_mask]
    y_val = all_T[va_mask]
    y_test = all_T[te_mask]
    p_train = T_pred_quad[tr_mask]
    p_val = T_pred_quad[va_mask]
    p_test = T_pred_quad[te_mask]

    finite_tr = np.isfinite(p_train)
    finite_va = np.isfinite(p_val)
    finite_te = np.isfinite(p_test)

    per_T_train = per_T_table(y_train[finite_tr], p_train[finite_tr])
    per_T_val = per_T_table(y_val[finite_va], p_val[finite_va])
    per_T_test = per_T_table(y_test[finite_te], p_test[finite_te])

    var_col = f"{var_name}_{var_unit}"
    for i, metric in enumerate(METRIC_NAMES):
        letter = letters[n_top_panels + i]
        r = (n_top_panels + i) // ncols
        c = (n_top_panels + i) % ncols
        ax = fig.add_subplot(gs[r, c])
        rows: list[dict] = []
        for set_name, rows_pT, color in (
            ("train", per_T_train, SET_COLORS[0]),
            ("val", per_T_val, SET_COLORS[1]),
            ("test_unseen", per_T_test, SET_COLORS[2]),
        ):
            if not rows_pT:
                continue
            T = np.array([r2["T_K"] for r2 in rows_pT], dtype=float)
            V = np.array([float(r2.get(metric, np.nan)) for r2 in rows_pT], dtype=float)
            order = np.argsort(T)
            T, V = T[order], V[order]
            ax.plot(T, V, "o-", ms=4, lw=1.4, color=color,
                    label=set_name)
            for t, v in zip(T, V):
                rows.append({"set": set_name, var_col: float(t),
                             "value": float(v) if np.isfinite(v) else np.nan})
        if metric == "Bias":
            ax.axhline(0.0, color="k", lw=0.6)
        ax.set_xlabel(_axis_label(var_name, var_unit))
        ax.set_ylabel(_metric_label(metric, var_unit))
        ax.set_title(f"({letter}) {_metric_label(metric, var_unit)} vs {var_name} — LIR (Quad)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        panel_csvs.append((f"Figure 3{letter}.csv", pd.DataFrame(rows)))

    fig.suptitle(
        f"Figure 3 — LIR diagnostic  ·  HI {lir_hi[0]:.0f}–{lir_hi[1]:.0f} nm  vs  "
        f"LOW {lir_low[0]:.0f}–{lir_low[1]:.0f} nm",
        fontsize=12, y=0.995,
    )
    fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.05,
                        hspace=0.38, wspace=0.28)
    _save_fig(fig, png, dpi)
    plt.close(fig)
    for fname, df in panel_csvs:
        _save_csv(df, os.path.join(out_dir, fname))
    return png


# =========================================================================
# Figure 4  —  PCA analysis (a) scree, (b) loadings, (c) PC1-PC2, (d) PC1-PC3
# =========================================================================


def make_figure_4(pca_result, prep, ds, out_dir: str, dpi: int = DPI_EXPORT,
                  quantity: dict | None = None) -> str:
    var_name, var_unit = _q(quantity)
    base = "Figure 4 - PCA analysis"
    png = os.path.join(out_dir, base + ".png")
    ev = pca_result.explained_variance_ratio
    k = len(ev)
    X_all = pca_result.X_all
    lam = prep.wavelengths_crop
    T = ds.temperatures

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    panel_csvs: list[tuple[str, pd.DataFrame]] = []

    ax = axes[0, 0]
    ev_pct = ev * 100
    cumev = np.cumsum(ev_pct)
    ax.bar(range(1, k + 1), ev_pct, color="steelblue", alpha=0.85,
           label="Individual")
    ax.plot(range(1, k + 1), cumev, "ro-", label="Cumulative")
    for i in range(k):
        ax.text(i + 1, ev_pct[i] + 1, f"{ev_pct[i]:.1f}%", ha="center")
    ax.set_xlabel("PC")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("(a) PCA explained variance")
    ax.legend()
    ax.set_ylim(0, max(110, cumev.max() + 5))
    panel_csvs.append((
        "Figure 4a.csv",
        pd.DataFrame({"PC": np.arange(1, k + 1, dtype=int),
                      "explained_pct": ev_pct, "cumulative_pct": cumev}),
    ))

    ax = axes[0, 1]
    rows_b: list[dict] = []
    for i in range(k):
        ax.plot(lam, pca_result.pca.components_[i], label=f"PC{i+1}")
        for w, v in zip(lam, pca_result.pca.components_[i]):
            rows_b.append({"PC": i + 1, "wavelength_nm": float(w),
                           "loading": float(v)})
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Loading")
    ax.set_title("(b) PCA loadings")
    ax.legend(fontsize=8)
    panel_csvs.append(("Figure 4b.csv", pd.DataFrame(rows_b)))

    var_col = f"{var_name}_{var_unit}"
    ax = axes[1, 0]
    sc = ax.scatter(X_all[:, 0],
                    X_all[:, 1] if k > 1 else np.zeros(len(X_all)),
                    c=T, cmap=CMAP_T, s=5, alpha=0.6)
    fig.colorbar(sc, ax=ax, label=f"{var_name} ({var_unit})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2" if k > 1 else "(constant)")
    ax.set_title("(c) PC1 vs PC2")
    panel_csvs.append((
        "Figure 4c.csv",
        pd.DataFrame({"PC1": X_all[:, 0],
                      "PC2": X_all[:, 1] if k > 1 else np.zeros(len(X_all)),
                      var_col: T}),
    ))

    ax = axes[1, 1]
    sc = ax.scatter(X_all[:, 0],
                    X_all[:, 2] if k > 2 else np.zeros(len(X_all)),
                    c=T, cmap=CMAP_T, s=5, alpha=0.6)
    fig.colorbar(sc, ax=ax, label=f"{var_name} ({var_unit})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC3" if k > 2 else "(constant)")
    ax.set_title("(d) PC1 vs PC3")
    panel_csvs.append((
        "Figure 4d.csv",
        pd.DataFrame({"PC1": X_all[:, 0],
                      "PC3": X_all[:, 2] if k > 2 else np.zeros(len(X_all)),
                      var_col: T}),
    ))

    fig.suptitle("Figure 4 — PCA analysis", fontsize=13, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.07,
                        hspace=0.28, wspace=0.25)
    _save_fig(fig, png, dpi)
    plt.close(fig)
    for fname, df in panel_csvs:
        _save_csv(df, os.path.join(out_dir, fname))
    return png


# =========================================================================
# Figure 5  —  Predicted vs True (all models, ordered)
# =========================================================================


def make_figure_5(results, out_dir: str, dpi: int = DPI_EXPORT,
                  quantity: dict | None = None) -> str:
    var_name, var_unit = _q(quantity)
    base = "Figure 5 - True vs Predicted (all models)"
    png = os.path.join(out_dir, base + ".png")
    ds = results.dataset
    y_tr, y_va, y_te = _split_y(ds)

    names = order_models(results)
    n = len(names)
    y_true_col = f"y_true_{var_unit}"
    y_pred_col = f"y_pred_{var_unit}"
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No models", ha="center", va="center")
        _save_fig(fig, png, dpi); plt.close(fig)
        _save_csv(pd.DataFrame(columns=["Model", "Set", y_true_col, y_pred_col]),
                  os.path.join(out_dir, "Figure 5.csv"))
        return png

    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes).reshape(nrows, ncols)
    lims = [min(y_tr.min(), y_te.min()) - 10, max(y_tr.max(), y_te.max()) + 10]

    rows: list[dict] = []
    for i, mname in enumerate(names):
        ax = axes[i // ncols, i % ncols]
        mr = results.models[mname]
        for set_name, y_true, color, label in (
            ("train", y_tr, SET_COLORS[0], "Train"),
            ("val", y_va, SET_COLORS[1], "Val"),
            ("test_unseen", y_te, SET_COLORS[2], "Test"),
        ):
            p = mr.predictions[set_name]
            rmse = _set_metric(mr, set_name, "RMSE")
            ax.scatter(y_true, p, s=4, alpha=0.4, c=color,
                       label=f"{label} ({rmse:.2f} {var_unit})")
            for yt, yp in zip(y_true, p):
                rows.append({"Model": mname, "Set": set_name,
                             y_true_col: float(yt), y_pred_col: float(yp)})
        ax.plot(lims, lims, "k--", lw=0.8)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(_axis_label(var_name, var_unit))
        ax.set_ylabel(_pred_label(var_name, var_unit))
        ax.set_title(f"{mname} [{mr.group}]", fontsize=9)
        ax.legend(fontsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    fig.suptitle(f"Figure 5 — Predicted vs True {var_name} (all models)",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    _save_fig(fig, png, dpi); plt.close(fig)
    _save_csv(pd.DataFrame(rows), os.path.join(out_dir, "Figure 5.csv"))
    return png


# =========================================================================
# Figures 6–11  —  Metric summary (Global bars + per-T on test_unseen)
# =========================================================================


def _figure_metric_summary(
    results,
    out_dir: str,
    fig_number: int,
    metric: str,
    dpi: int,
    title_suffix: str = "summary",
    model_filter: Callable[[list[str]], list[str]] | None = None,
    base_name: str | None = None,
    quantity: dict | None = None,
) -> str:
    """Two-panel: global per-set bars (a) + per-T on test_unseen (b)."""
    var_name, var_unit = _q(quantity)
    label = _metric_label(metric, var_unit)
    base = base_name or f"Figure {fig_number} - {metric} {title_suffix}"
    png = os.path.join(out_dir, base + ".png")

    names = order_models(results)
    if model_filter is not None:
        names = model_filter(names)
    if not names:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No models", ha="center", va="center")
        _save_fig(fig, png, dpi); plt.close(fig)
        return png

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(names) * 0.7 + 4), 6))

    # ── (a) Global bars, 3 splits ───────────────────────────────────────
    ax_a = axes[0]
    xs = np.arange(len(names))
    w = 0.25
    rows_a: list[dict] = []
    for si, (s, c, lbl) in enumerate(zip(SETS, SET_COLORS, SET_LABELS)):
        vals = [_set_metric(results.models[m], s, metric) for m in names]
        ax_a.bar(xs + si * w, vals, w, label=lbl, color=c, alpha=0.8)
        for m, v in zip(names, vals):
            rows_a.append({"Model": m, "Set": s, metric: float(v)})
    if metric_signed(metric):
        ax_a.axhline(0.0, color="k", lw=0.6)
    ax_a.set_xticks(xs + w)
    ax_a.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax_a.set_ylabel(label)
    ax_a.set_title(f"(a) Global {label} by model and set")
    ax_a.legend(loc="upper left", fontsize=8)

    # ── (b) Per-T on test_unseen ────────────────────────────────────────
    ax_b = axes[1]
    rows_b: list[dict] = []
    var_col = f"{var_name}_{var_unit}"
    for mname in names:
        mr = results.models[mname]
        T, V = _per_T_values(mr, "test_unseen", metric)
        if T.size == 0:
            continue
        ax_b.plot(T, V, "o-", ms=3, lw=1, alpha=0.85, label=mname)
        for t, v in zip(T, V):
            rows_b.append({"Model": mname, var_col: float(t),
                           "value": float(v) if np.isfinite(v) else np.nan})
    if metric_signed(metric):
        ax_b.axhline(0.0, color="k", lw=0.6)
    ax_b.set_xlabel(_axis_label(var_name, var_unit))
    ax_b.set_ylabel(label)
    ax_b.set_title(f"(b) Per-{var_name} {label} on test_unseen")
    ax_b.legend(fontsize=6, ncol=2)

    fig.suptitle(f"Figure {fig_number} — {label} {title_suffix}", fontsize=12, y=1.0)
    fig.tight_layout()
    _save_fig(fig, png, dpi); plt.close(fig)
    _save_csv(pd.DataFrame(rows_a), os.path.join(out_dir, f"Figure {fig_number}a.csv"))
    _save_csv(pd.DataFrame(rows_b), os.path.join(out_dir, f"Figure {fig_number}b.csv"))
    return png


def make_figures_6_11(results, out_dir: str, dpi: int = DPI_EXPORT,
                      quantity: dict | None = None) -> list[str]:
    paths: list[str] = []
    for i, metric in enumerate(METRIC_NAMES):
        fig_no = 6 + i
        paths.append(_figure_metric_summary(
            results, out_dir, fig_no, metric, dpi,
            title_suffix="summary",
            base_name=f"Figure {fig_no} - {metric} summary",
            quantity=quantity,
        ))
    return paths


# =========================================================================
# Figures 12–17  —  Top-5 zoom, all 3 sets
# =========================================================================


def make_figures_12_17(results, out_dir: str, dpi: int = DPI_EXPORT,
                       quantity: dict | None = None) -> list[str]:
    top5 = top_n(results, 5)
    keep = set(top5)
    paths: list[str] = []
    for i, metric in enumerate(METRIC_NAMES):
        fig_no = 12 + i
        paths.append(_figure_metric_summary(
            results, out_dir, fig_no, metric, dpi,
            title_suffix="zoom Top5",
            model_filter=lambda names, _keep=keep: [n for n in names if n in _keep],
            base_name=f"Figure {fig_no} - {metric} zoom Top5",
            quantity=quantity,
        ))
    return paths


# =========================================================================
# Figures 18–23  —  Top-5 zoom, val + test_unseen + mean(val,test)
# =========================================================================


def _figure_top5_val_test_mean(
    results,
    out_dir: str,
    fig_number: int,
    metric: str,
    dpi: int,
    quantity: dict | None = None,
) -> str:
    var_name, var_unit = _q(quantity)
    label = _metric_label(metric, var_unit)
    base = f"Figure {fig_number} - {metric} zoom Top5 (val,test,mean)"
    png = os.path.join(out_dir, base + ".png")

    names = top_n(results, 5)
    if not names:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No models", ha="center", va="center")
        _save_fig(fig, png, dpi); plt.close(fig)
        return png

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    xs = np.arange(len(names))
    wbar = 0.36

    # ── (a) Bars val + test + mean(val,test) ────────────────────────────
    ax_a = axes[0]
    val_vals = [_set_metric(results.models[m], "val", metric) for m in names]
    test_vals = [_set_metric(results.models[m], "test_unseen", metric) for m in names]
    if metric == "Bias":
        mean_vals = [0.5 * (v + t) for v, t in zip(val_vals, test_vals)]
    else:
        mean_vals = [0.5 * (v + t) for v, t in zip(val_vals, test_vals)]
    ax_a.bar(xs - wbar / 2, val_vals, wbar, color="tab:orange", alpha=0.6,
             label="val")
    ax_a.bar(xs + wbar / 2, test_vals, wbar, color="tab:red", alpha=0.6,
             label="test_unseen")
    ax_a.plot(xs, mean_vals, "ko-", ms=7, lw=1.4, label="mean(val, test_unseen)")
    for x, v in zip(xs, mean_vals):
        ax_a.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                  fontweight="bold")
    if metric_signed(metric):
        ax_a.axhline(0.0, color="k", lw=0.6)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax_a.set_ylabel(label)
    ax_a.set_title(f"(a) {label} — val, test_unseen, mean (Top-5)")
    ax_a.legend(loc="upper left", fontsize=9)
    rows_a = (
        [{"Model": m, "Set": "val", metric: float(v)} for m, v in zip(names, val_vals)]
        + [{"Model": m, "Set": "test_unseen", metric: float(v)} for m, v in zip(names, test_vals)]
        + [{"Model": m, "Set": "mean(val,test)", metric: float(v)} for m, v in zip(names, mean_vals)]
    )

    # ── (b) Per-T curves: val, test, mean(val,test) ─────────────────────
    ax_b = axes[1]
    rows_b: list[dict] = []
    var_col = f"{var_name}_{var_unit}"
    for mname in names:
        mr = results.models[mname]
        Tv, Vv = _per_T_values(mr, "val", metric)
        Tt, Vt = _per_T_values(mr, "test_unseen", metric)
        common = np.intersect1d(Tv, Tt) if Tv.size and Tt.size else np.array([])
        if Tv.size:
            ax_b.plot(Tv, Vv, "o:", ms=3, lw=1, alpha=0.85,
                      label=f"{mname} — val")
            for t, v in zip(Tv, Vv):
                rows_b.append({"Model": mname, "Set": "val",
                               var_col: float(t),
                               "value": float(v) if np.isfinite(v) else np.nan})
        if Tt.size:
            ax_b.plot(Tt, Vt, "o-", ms=3, lw=1.2, alpha=0.9,
                      label=f"{mname} — test_unseen")
            for t, v in zip(Tt, Vt):
                rows_b.append({"Model": mname, "Set": "test_unseen",
                               var_col: float(t),
                               "value": float(v) if np.isfinite(v) else np.nan})
        if common.size:
            vmap = {float(t): float(v) for t, v in zip(Tv, Vv)}
            tmap = {float(t): float(v) for t, v in zip(Tt, Vt)}
            means = np.array([0.5 * (vmap[float(t)] + tmap[float(t)]) for t in common])
            ax_b.plot(common, means, "o--", ms=4, lw=1.6, alpha=0.95,
                      label=f"{mname} — mean(val,test)")
            for t, v in zip(common, means):
                rows_b.append({"Model": mname, "Set": "mean(val,test)",
                               var_col: float(t),
                               "value": float(v) if np.isfinite(v) else np.nan})
    if metric_signed(metric):
        ax_b.axhline(0.0, color="k", lw=0.6)
    ax_b.set_xlabel(_axis_label(var_name, var_unit))
    ax_b.set_ylabel(label)
    ax_b.set_title(f"(b) Per-{var_name} {label} (Top-5; val, test, mean)")
    ax_b.legend(fontsize=6, ncol=2)

    fig.suptitle(f"Figure {fig_number} — {label} zoom Top5 (val, test, mean)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    _save_fig(fig, png, dpi); plt.close(fig)
    _save_csv(pd.DataFrame(rows_a), os.path.join(out_dir, f"Figure {fig_number}a.csv"))
    _save_csv(pd.DataFrame(rows_b), os.path.join(out_dir, f"Figure {fig_number}b.csv"))
    return png


def make_figures_18_23(results, out_dir: str, dpi: int = DPI_EXPORT,
                       quantity: dict | None = None) -> list[str]:
    return [
        _figure_top5_val_test_mean(results, out_dir, 18 + i, metric, dpi,
                                   quantity=quantity)
        for i, metric in enumerate(METRIC_NAMES)
    ]


# =========================================================================
# Figure 24  —  Sensitivity of best model
# =========================================================================


def make_figure_24(results, out_dir: str, dpi: int = DPI_EXPORT,
                   quantity: dict | None = None) -> str | None:
    var_name, var_unit = _q(quantity)
    base = "Figure 24 - Sensitivity of best model"
    png = os.path.join(out_dir, base + ".png")
    csv = os.path.join(out_dir, "Figure 24.csv")
    var_col = f"{var_name}_{var_unit}"
    pred_col = f"mean_pred_{var_unit}"
    names = order_models(results)
    if not names:
        return None
    winner = names[0]
    ds = results.dataset
    y_test = ds.temperatures[ds.test_mask]
    Tu = np.sort(np.unique(y_test))
    pred = results.models[winner].predictions["test_unseen"]
    if Tu.size < 2:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(
            0.5, 0.5,
            f"Sensitivity needs ≥ 2 test-unseen set-points (winner: {winner}).",
            ha="center", va="center", wrap=True,
        )
        _save_fig(fig, png, dpi); plt.close(fig)
        _save_csv(pd.DataFrame(columns=[var_col, pred_col, "S_a"]), csv)
        return png
    mean_pred = np.array([pred[y_test == t].mean() for t in Tu])
    Sa = np.abs(np.gradient(mean_pred, Tu))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(Tu, Sa, "bo-", lw=1.5)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="Ideal Sa = 1")
    ax.set_xlabel(_axis_label(var_name, var_unit))
    ax.set_ylabel(
        rf"$S_a = |d\bar{{\hat {var_name}}}/d{var_name}|$ "
        f"({var_unit}/{var_unit})"
    )
    ax.set_title(f"Figure 24 — Sensitivity of best model: {winner}")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, png, dpi); plt.close(fig)
    _save_csv(
        pd.DataFrame({var_col: Tu, pred_col: mean_pred, "S_a": Sa}),
        csv,
    )
    return png


# =========================================================================
# export_all — single entry point used by the GUI worker
# =========================================================================


def export_all(results, prep_result, pca_result, out_dir: str,
               dpi: int = DPI_EXPORT,
               quantity: dict | None = None,
               fmt: str = "png") -> dict[str, str]:
    """Emit the full new ``Figure …`` set into *out_dir*.

    Parameters
    ----------
    quantity
        Optional ``{"name": "T", "unit": "K"}`` dict so the user can override
        the temperature-dependent variable label and unit shown on axes and
        CSV column headers.  Defaults to (T, K).
    fmt
        ``"png"`` (default) or ``"svg"``.  When ``"svg"`` is selected the
        underlying matplotlib figures are written as vector SVG instead of
        rasterised PNG, but the CSV companions are unchanged.

    Returns a mapping {file basename → absolute path}.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved: dict[str, str] = {}

    # Toggle the file extension for every _save_fig call inside this run.
    global _SAVE_FIG_EXT
    prev_ext = _SAVE_FIG_EXT
    _SAVE_FIG_EXT = "svg" if str(fmt).lower() == "svg" else "png"
    target_ext = ".svg" if _SAVE_FIG_EXT == "svg" else ".png"

    def _track(path: str | None):
        if not path:
            return
        # Builders always return a *.png* path; rewrite to the actual extension
        # we wrote to disk so the caller can find the file.
        root, _ext = os.path.splitext(path)
        real = root + target_ext
        saved[os.path.basename(real)] = real

    ds = results.dataset

    try:
        _track(make_figure_1(prep_result, ds, out_dir, dpi, quantity=quantity))
        _track(make_figure_2(prep_result, ds, out_dir, dpi, quantity=quantity))
        _track(make_figure_3(results, prep_result, out_dir, dpi, quantity=quantity))
        _track(make_figure_4(pca_result, prep_result, ds, out_dir, dpi,
                             quantity=quantity))
        _track(make_figure_5(results, out_dir, dpi, quantity=quantity))
        for p in make_figures_6_11(results, out_dir, dpi, quantity=quantity):
            _track(p)
        for p in make_figures_12_17(results, out_dir, dpi, quantity=quantity):
            _track(p)
        for p in make_figures_18_23(results, out_dir, dpi, quantity=quantity):
            _track(p)
        _track(make_figure_24(results, out_dir, dpi, quantity=quantity))
    finally:
        _SAVE_FIG_EXT = prev_ext

    plt.close("all")
    return saved

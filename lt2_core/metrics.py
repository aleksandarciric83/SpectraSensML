"""metrics.py
Sensor-style regression metrics and per-temperature tables.

Conventions (locked in chat):
  e_i = y_pred_i - y_true_i              (signed; positive = over-prediction)
  Units: temperatures and all derived metrics are in K.
  Per-temperature precision (sigma) is reported only when n_k >= 3
  (ISO 5725-2 §7.4).  Otherwise NaN.

Two metric sets are exposed:

  §A — per-temperature table (one row per unique reference T_k in the split):
       n_k, Bias(T_k), MAE(T_k), Precision_sigma(T_k), RMSE(T_k),
       MaxAbs(T_k), P95Abs(T_k).

  §B — global summary for a split:
       Sample-level pool: Bias, MAE, Precision_sigma, RMSE, MaxAbs, P95Abs.
       Set-point-level averages (mean across T_k):
           MeanBin_Bias, MeanBin_MAE, MeanBin_Precision_sigma,
           MeanBin_RMSE, MeanBin_MaxAbs, MeanBin_P95Abs.
       Worst-case across T_k:
           WorstBin_Bias (|Bias(T_k)|), WorstBin_MAE,
           WorstBin_Precision_sigma, WorstBin_RMSE,
           WorstBin_MaxAbs, WorstBin_P95Abs.

R² is intentionally excluded — not a sensor figure of merit here.

The exported metric names below (METRIC_NAMES / METRIC_LABELS) are the
canonical list used to drive the auto-generated Figures 6–23.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


# ─── canonical metric vocabulary (drives Figure builders) ────────────────

METRIC_NAMES: tuple[str, ...] = (
    "RMSE",
    "Bias",
    "Precision_sigma",
    "MAE",
    "MaxAbs",
    "P95Abs",
)

METRIC_LABELS: dict[str, str] = {
    "RMSE": "RMSE (K)",
    "Bias": "Bias (K)",
    "Precision_sigma": "Precision σ (K)",
    "MAE": "MAE (K)",
    "MaxAbs": "Max |error| (K)",
    "P95Abs": "P95 |error| (K)",
}

# Minimum bin count before a sigma-based metric is trusted.
MIN_BIN_N_FOR_SIGMA = 3


# ─── §A per-temperature helpers ───────────────────────────────────────────


def _bin_metrics_one_T(e: np.ndarray) -> dict:
    """Compute §A metrics inside one set-point bin.

    Parameters
    ----------
    e : (n_k,) array of signed residuals e_i = y_pred_i - y_true_i for
        the spectra of a single set-point T_k.
    """
    n = int(e.size)
    if n == 0:
        return {
            "n": 0,
            "Bias": np.nan,
            "MAE": np.nan,
            "Precision_sigma": np.nan,
            "RMSE": np.nan,
            "MaxAbs": np.nan,
            "P95Abs": np.nan,
        }
    abs_e = np.abs(e)
    bias = float(e.mean())
    mae = float(abs_e.mean())
    rmse = float(np.sqrt(np.mean(e * e)))
    max_abs = float(abs_e.max())
    p95 = float(np.percentile(abs_e, 95))
    if n >= MIN_BIN_N_FOR_SIGMA:
        sigma = float(np.std(e, ddof=1))
    else:
        sigma = float("nan")
    return {
        "n": n,
        "Bias": bias,
        "MAE": mae,
        "Precision_sigma": sigma,
        "RMSE": rmse,
        "MaxAbs": max_abs,
        "P95Abs": p95,
    }


def per_T_table(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """§A — one row per unique reference temperature in *y_true*.

    Returns a list of dicts with keys::

        T_K, n, Bias, MAE, Precision_sigma, RMSE, MaxAbs, P95Abs

    plus legacy keys (``mean_pred``, ``mean_residual``) so older code that
    still reads ``per_T`` rows keeps working.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rows: list[dict] = []
    for T in np.unique(y_true):
        m = y_true == T
        if not m.any():
            continue
        e = y_pred[m] - T
        row = {"T_K": float(T)}
        row.update(_bin_metrics_one_T(e))
        row["mean_pred"] = float(y_pred[m].mean())
        row["mean_residual"] = row["Bias"]
        rows.append(row)
    return rows


# ─── §B global helpers ────────────────────────────────────────────────────


def _sample_pool(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    e = y_pred - y_true
    n = int(e.size)
    if n == 0:
        return {k: np.nan for k in METRIC_NAMES} | {"n": 0}
    abs_e = np.abs(e)
    bias = float(e.mean())
    mae = float(abs_e.mean())
    rmse = float(np.sqrt(np.mean(e * e)))
    max_abs = float(abs_e.max())
    p95 = float(np.percentile(abs_e, 95))
    sigma = float(np.std(e, ddof=1)) if n >= 2 else float("nan")
    return {
        "n": n,
        "Bias": bias,
        "MAE": mae,
        "Precision_sigma": sigma,
        "RMSE": rmse,
        "MaxAbs": max_abs,
        "P95Abs": p95,
    }


def _summarise_bins(rows: list[dict]) -> dict:
    """Compute MeanBin_* and WorstBin_* from a §A per-T table."""
    out: dict = {}
    if not rows:
        for m in METRIC_NAMES:
            out[f"MeanBin_{m}"] = np.nan
            out[f"WorstBin_{m}"] = np.nan
        out["K"] = 0
        return out
    out["K"] = len(rows)
    for m in METRIC_NAMES:
        vals = np.array([r[m] for r in rows], dtype=float)
        finite = np.isfinite(vals)
        if not finite.any():
            out[f"MeanBin_{m}"] = float("nan")
            out[f"WorstBin_{m}"] = float("nan")
            continue
        v = vals[finite]
        out[f"MeanBin_{m}"] = float(np.mean(v))
        if m == "Bias":
            out["WorstBin_Bias"] = float(np.max(np.abs(v)))
        else:
            out[f"WorstBin_{m}"] = float(np.max(v))
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """§B — global metrics for one split.

    Returns a flat dict with keys:
      - ``n``, ``K``
      - sample-level pool: Bias, MAE, Precision_sigma, RMSE, MaxAbs, P95Abs
      - MeanBin_* and WorstBin_* over the §A per-T rows
      - legacy keys (``MaxErr``, ``dT_mean``, ``dT_max``) preserved for any
        old code paths that still read them.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    pool = _sample_pool(y_true, y_pred)
    rows = per_T_table(y_true, y_pred)
    summary = _summarise_bins(rows)

    # Legacy keys kept so the GUI leaderboard / older callers stay working.
    legacy = {
        "MaxErr": pool["MaxAbs"],
        "dT_mean": summary.get("MeanBin_MAE", float("nan")),
        "dT_max": summary.get("WorstBin_MAE", float("nan")),
    }
    return pool | summary | legacy


# ─── public listing of canonical column orders ───────────────────────────


def global_csv_columns() -> list[str]:
    cols = ["Model", "Group", "Set", "n", "K"]
    cols += list(METRIC_NAMES)
    cols += [f"MeanBin_{m}" for m in METRIC_NAMES]
    cols += [f"WorstBin_{m}" for m in METRIC_NAMES]
    return cols


def per_T_csv_columns() -> list[str]:
    return [
        "Model",
        "Group",
        "Set",
        "T_K",
        "n",
        "Bias",
        "MAE",
        "Precision_sigma",
        "RMSE",
        "MaxAbs",
        "P95Abs",
    ]


def metric_signed(name: str) -> bool:
    """True if the metric keeps its sign (Bias)."""
    return name == "Bias"


def iter_metrics() -> Iterable[tuple[str, str]]:
    for m in METRIC_NAMES:
        yield m, METRIC_LABELS[m]

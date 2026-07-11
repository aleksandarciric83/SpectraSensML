"""benchmark.py
BenchmarkConfig dataclass and run_benchmark() function.

This module mirrors the model evaluation logic from LT2_pipeline_v2.py but
is fully importable and callable from a GUI without any print-based logging.

All hyperparameter selection uses the validation set (no LOTO, no internal CV),
matching the original pipeline exactly.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.interpolate import PchipInterpolator, RBFInterpolator
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from .dataset import Dataset
from .metrics import compute_metrics, per_T_table
from .pca_analysis import PCAResult
from .preprocess import (
    LIRConfig,
    Normalization,
    PreprocessResult,
    compute_lir,
    fit_lir_quad,
    fit_lir_boltzmann_linear,
    predict_T_lir_quad,
    predict_T_lir_boltzmann,
    preprocess,
)


# ─── optional heavy deps ──────────────────────────────────────────────────

def _try_import(name: str):
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError:
        return None


# ─── Configuration ────────────────────────────────────────────────────────

@dataclass
class ModelFlags:
    """Which model groups to include in the benchmark."""
    # Group B — simple / spline regressions
    pc1_poly_bic: bool = True
    pchip_pc1: bool = True
    sensor_fusion: bool = True
    mlr: bool = True
    poly_3pcs_deg7: bool = True
    tps_3pcs: bool = True
    lir_quad: bool = True
    lir_boltzmann: bool = True
    # Group A — trees
    random_forest: bool = True
    gradient_boosting: bool = True
    xgboost: bool = True
    lightgbm: bool = True
    extra_trees: bool = True
    catboost: bool = True
    # Group C — kernel / probabilistic
    svr_rbf: bool = True
    svr_poly: bool = True
    gpr: bool = True
    knn: bool = True
    # Group D — extras
    krr_poly: bool = True
    pls: bool = True
    bayesian_ridge: bool = True
    elasticnet: bool = True
    mlp: bool = True
    # Group E — neural (in-process sklearn + optional PyTorch CNN)
    ann_pca_mlp: bool = True
    ann_snv_mlp: bool = True
    cnn_snv_1d: bool = True


@dataclass
class BenchmarkConfig:
    """Complete description of one benchmark run."""
    wl_min: float = 900.0
    wl_max: float = 1100.0
    normalization: Normalization = Normalization.SNV
    bg_subtract: bool = False
    n_pca_components: int = 3
    rng_seed: int = 42
    models: ModelFlags = field(default_factory=ModelFlags)
    lir_cfg: LIRConfig = field(default_factory=LIRConfig)
    # Name and unit of the temperature-dependent variable the app is trained on.
    # Defaults assume Kelvin temperature; the user can override these so the
    # software is reusable for any scalar variable that depends on temperature.
    var_name: str = "T"
    var_unit: str = "K"
    # Expose for hyper-param grids (future: full grid editor)
    svr_C_grid: tuple = (0.1, 1, 10, 100)
    svr_eps_grid: tuple = (0.01, 0.1, 1.0)
    svr_poly_eps_grid: tuple = (0.1, 1.0)
    svr_poly_deg_grid: tuple = (2, 3, 4)
    knn_k_grid: tuple = (3, 5, 10)
    krr_alpha_grid: tuple = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    krr_deg_grid: tuple = (2, 3, 4)
    pls_nc_grid: tuple = (1, 2, 3, 4, 5, 7, 10, 15)
    mlp_archs: tuple = ((32,), (64,), (64, 64), (128, 64), (128, 128, 64))
    en_alpha_grid: tuple = (1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0)
    en_l1_grid: tuple = (0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
    n_gpr_restarts: int = 5
    fusion_T_grid_pts: int = 4000
    pc1_poly_max_order: int = 7
    ann_pca_archs: tuple = ((256, 128, 64), (128, 128, 64), (64,))
    ann_snv_archs: tuple = ((256, 128, 64), (128, 64), (64, 64))
    cnn_fc_hidden: int = 64
    # Defaults tuned for interactive GUI runs (CPU); increase for final paper runs.
    cnn_max_epochs: int = 48
    cnn_patience_epochs: int = 12
    # If set, load ``*.npz`` files from this directory after the sklearn benchmark.
    # Each file must contain arrays ``train``, ``val``, ``test_unseen`` and
    # strings ``model_name``, ``group`` (see LT2_pipeline_v2 virtual-model loader).
    extra_predictions_dir: str | None = None

    def __post_init__(self) -> None:
        n = self.normalization
        if isinstance(n, Normalization):
            return
        if n is None:
            object.__setattr__(self, "normalization", Normalization.SNV)
            return
        object.__setattr__(self, "normalization", Normalization(str(n)))


# ─── Result containers ───────────────────────────────────────────────────

@dataclass
class ModelResult:
    name: str
    group: str   # "A" splines/LIR | "B" trees | "C" | "D" | "E" neural
    metrics: dict[str, dict]   # set_name → {RMSE, MAE, ...}
    predictions: dict[str, np.ndarray]  # set_name → (N,) array
    per_T: dict[str, list[dict]]        # set_name → per_T_table rows


@dataclass
class BenchmarkResults:
    models: dict[str, ModelResult] = field(default_factory=dict)
    pca_result: Optional[PCAResult] = None
    preprocess_result: Optional[PreprocessResult] = None
    dataset: Optional[Dataset] = None
    config: Optional[BenchmarkConfig] = None

    def sorted_by_mean_val_test_rmse(self) -> list[str]:
        def score(n: str) -> float:
            mr = self.models[n]
            v = mr.metrics.get("val", {}).get("RMSE", np.inf)
            t = mr.metrics.get("test_unseen", {}).get("RMSE", np.inf)
            return 0.5 * (float(v) + float(t))

        return sorted(self.models, key=score)

    def sorted_by_test_rmse(self) -> list[str]:
        return sorted(
            self.models,
            key=lambda n: self.models[n].metrics.get("test_unseen", {}).get("RMSE", np.inf),
        )

    def leaderboard(self) -> list[dict]:
        rows = []
        for name in self.sorted_by_mean_val_test_rmse():
            mr = self.models[name]
            rv = mr.metrics.get("val", {}).get("RMSE", np.nan)
            rt = mr.metrics.get("test_unseen", {}).get("RMSE", np.nan)
            avg_vt = 0.5 * (float(rv) + float(rt)) if np.isfinite(rv) and np.isfinite(rt) else np.nan
            rows.append(
                {
                    "Model": name,
                    "Group": mr.group,
                    "Train RMSE": mr.metrics.get("train", {}).get("RMSE", np.nan),
                    "Val RMSE": rv,
                    "Test RMSE": rt,
                    "Avg(val,test) RMSE": avg_vt,
                    "dT_mean": mr.metrics.get("test_unseen", {}).get("dT_mean", np.nan),
                }
            )
        return rows


def merge_external_predictions_from_dir(
    results: BenchmarkResults,
    ds: Dataset,
    directory: str | None,
    log: Callable[[str], None],
) -> int:
    """Load ``*.npz`` prediction bundles and append :class:`ModelResult` entries.

    Expected NPZ keys: ``train``, ``val``, ``test_unseen`` (1-D float arrays),
    ``model_name``, ``group`` (pickled or byte strings).

    Returns the number of models successfully merged.
    """
    if not directory or not str(directory).strip():
        return 0
    directory = os.path.abspath(str(directory).strip())
    if not os.path.isdir(directory):
        log(f"[extra preds] Not a directory: {directory}")
        return 0

    y_train = ds.temperatures[ds.train_mask]
    y_val = ds.temperatures[ds.val_mask]
    y_test = ds.temperatures[ds.test_mask]

    n_ok = 0
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".npz"):
            continue
        fpath = os.path.join(directory, fname)
        try:
            blob = np.load(fpath, allow_pickle=True)
        except Exception as e:
            log(f"  [skip] {fname}: cannot load ({e})")
            continue
        need = ("train", "val", "test_unseen", "model_name", "group")
        if not all(k in blob.files for k in need):
            log(f"  [skip] {fname}: missing one of {need}")
            continue
        p_train = np.asarray(blob["train"], dtype=float).ravel()
        p_val = np.asarray(blob["val"], dtype=float).ravel()
        p_test = np.asarray(blob["test_unseen"], dtype=float).ravel()
        label = str(blob["model_name"])
        grp = str(blob["group"])
        if p_train.shape != y_train.shape or p_val.shape != y_val.shape or p_test.shape != y_test.shape:
            log(
                f"  [skip] {fname}: shape mismatch "
                f"(train {p_train.shape} vs {y_train.shape}, …)"
            )
            continue
        if label in results.models:
            log(f"  [skip] {fname}: model «{label}» already exists")
            continue
        _register(results, label, grp, y_train, y_val, y_test, p_train, p_val, p_test, log)
        n_ok += 1
    if n_ok:
        log(f"[extra preds] Merged {n_ok} model(s) from {directory}")
    else:
        log(f"[extra preds] No valid *.npz models loaded from {directory}")
    return n_ok


# ─── Internal helpers ─────────────────────────────────────────────────────

def _val_rmse(y_val, y_pred_val):
    return float(np.sqrt(mean_squared_error(y_val, y_pred_val)))


def _register(
    results: BenchmarkResults,
    name: str,
    group: str,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    p_train: np.ndarray,
    p_val: np.ndarray,
    p_test: np.ndarray,
    log: Callable[[str], None],
):
    mr = ModelResult(
        name=name,
        group=group,
        metrics={
            "train": compute_metrics(y_train, p_train),
            "val": compute_metrics(y_val, p_val),
            "test_unseen": compute_metrics(y_test, p_test),
        },
        predictions={"train": p_train, "val": p_val, "test_unseen": p_test},
        per_T={
            "train": per_T_table(y_train, p_train),
            "val": per_T_table(y_val, p_val),
            "test_unseen": per_T_table(y_test, p_test),
        },
    )
    results.models[name] = mr
    log(
        f"  [{group}] {name:38s} | "
        f"Train {mr.metrics['train']['RMSE']:7.3f}  "
        f"Val   {mr.metrics['val']['RMSE']:7.3f}  "
        f"Test  {mr.metrics['test_unseen']['RMSE']:7.3f} K"
    )


# ─── Main entry point ─────────────────────────────────────────────────────

def run_benchmark(
    ds: Dataset,
    cfg: BenchmarkConfig | None = None,
    log: Callable[[str], None] = print,
    progress: Callable[[int, int], None] | None = None,
    cancel_flag: list[bool] | None = None,   # cancel_flag[0] → True to abort
) -> BenchmarkResults:
    """Evaluate all enabled models on *ds* according to *cfg*.

    Parameters
    ----------
    ds : Dataset from load_dataset()
    cfg : BenchmarkConfig (defaults to BenchmarkConfig())
    log : callable for string messages (default: print)
    progress : callable(done, total) for progress reporting
    cancel_flag : single-element list; set [0]=True to abort the run

    Returns
    -------
    BenchmarkResults
    """
    if cfg is None:
        cfg = BenchmarkConfig()

    n_train = int((ds.labels == "train").sum())
    n_val = int((ds.labels == "val").sum())
    n_test = int((ds.labels == "test_unseen").sum())
    if n_train == 0:
        raise ValueError(
            "No training spectra (train=0). Assign at least one temperature folder "
            "to «Train/Val» in the Data tab, then Apply, before running the benchmark."
        )
    if n_val == 0:
        raise ValueError(
            "No validation spectra (val=0). Use at least two spectra per Train/Val "
            "folder or lower the train fraction so some spectra remain in validation."
        )

    def cancelled() -> bool:
        return bool(cancel_flag and cancel_flag[0])

    def tick(step: int, total: int):
        if progress is not None:
            progress(step, total)

    results = BenchmarkResults(config=cfg, dataset=ds)

    fl = cfg.models
    k = cfg.n_pca_components

    def _n_ticks() -> int:
        n = 0
        n += int(fl.pc1_poly_bic)
        n += int(fl.pchip_pc1)
        n += int(fl.sensor_fusion and k >= 2)
        n += int(fl.mlr)
        n += int(fl.poly_3pcs_deg7 and k >= 3)
        n += int(fl.tps_3pcs and k >= 3)
        n += int(fl.lir_quad)
        n += int(fl.lir_boltzmann)
        n += int(fl.random_forest)
        n += int(fl.gradient_boosting)
        n += int(fl.xgboost)
        n += int(fl.lightgbm)
        n += int(fl.extra_trees)
        n += int(fl.catboost)
        n += int(fl.svr_rbf)
        n += int(fl.svr_poly)
        n += int(fl.gpr)
        n += int(fl.knn)
        n += int(fl.krr_poly)
        n += int(fl.pls)
        n += int(fl.bayesian_ridge)
        n += int(fl.elasticnet)
        n += int(fl.mlp)
        n += int(fl.ann_pca_mlp)
        n += int(fl.ann_snv_mlp)
        n += int(fl.cnn_snv_1d)
        return max(n, 1)

    n_model_steps = _n_ticks()
    total_steps = max(3, 2 + n_model_steps)
    prog_step = 0

    def advance_progress():
        nonlocal prog_step
        prog_step += 1
        tick(prog_step, total_steps)

    def _tick():
        advance_progress()

    tick(0, total_steps)

    # ── 1. Preprocess ─────────────────────────────────────────────────────
    log("[1] Preprocessing spectra ...")
    prep = preprocess(
        ds.spectra, ds.wavelengths,
        cfg.wl_min, cfg.wl_max,
        cfg.normalization, cfg.bg_subtract,
    )
    results.preprocess_result = prep
    log(f"    cropped to {prep.wavelengths_crop.size} channels; "
        f"{prep.n_zero_std} zero-std spectra")
    advance_progress()

    # ── 2. PCA ───────────────────────────────────────────────────────────
    log(f"[2] PCA ({cfg.n_pca_components} components, fit on train) ...")
    from .pca_analysis import fit_pca
    pca_res = fit_pca(
        prep.spectra_norm, ds.train_mask, cfg.n_pca_components, cfg.rng_seed
    )
    results.pca_result = pca_res
    ev = pca_res.explained_variance_ratio
    log("    explained: " + "  ".join(f"PC{i+1}={v:.4f}" for i, v in enumerate(ev))
        + f"  Σ={ev.sum():.4f}")
    advance_progress()

    # ── split arrays ──────────────────────────────────────────────────────
    tr = ds.train_mask
    va = ds.val_mask
    te = ds.test_mask

    X_all = pca_res.X_all                    # (N, k)
    X_train = X_all[tr]; y_train = ds.temperatures[tr]
    X_val   = X_all[va]; y_val   = ds.temperatures[va]
    X_test  = X_all[te]; y_test  = ds.temperatures[te]

    S_train = prep.spectra_norm[tr]          # full-dim SNV slices (for PLS)
    S_val   = prep.spectra_norm[va]
    S_test  = prep.spectra_norm[te]

    S_raw_train = prep.spectra_raw_crop[tr]  # raw (bg-scaled) for LIR

    train_temps_unique = np.unique(y_train)
    mean_pcs_per_T = np.array(
        [X_train[y_train == t].mean(axis=0) for t in train_temps_unique]
    )

    reg = _register  # shortcut

    # ─────────────────────────────────────────────────────────────────────
    # Group A — simple regressions / splines / LIR / fusion
    # ─────────────────────────────────────────────────────────────────────
    log("\n[Group A] Splines / LIR / fusion")

    pc1_train = X_train[:, 0]
    pc1_val   = X_val[:, 0]
    pc1_test  = X_test[:, 0]

    # B1 — PC1 Polynomial (BIC order selection on calibration curve)
    if fl.pc1_poly_bic and not cancelled():
        mean_pc1_per_T = np.array(
            [pc1_train[y_train == t].mean() for t in train_temps_unique]
        )
        n_calib = len(train_temps_unique)
        bic_records = []
        poly_models: dict[int, np.ndarray] = {}
        for order in range(1, cfg.pc1_poly_max_order + 1):
            coeffs = np.polyfit(mean_pc1_per_T, train_temps_unique, order)
            pred_c = np.polyval(coeffs, mean_pc1_per_T)
            rss = float(np.sum((train_temps_unique - pred_c) ** 2))
            kk = order + 1
            sigma2 = max(rss / max(n_calib - kk, 1), 1e-12)
            log_lik = -0.5 * n_calib * (np.log(2 * np.pi * sigma2) + 1.0)
            bic = -2.0 * log_lik + kk * np.log(n_calib)
            bic_records.append((order, bic))
            poly_models[order] = coeffs
        best_ord = min(bic_records, key=lambda r: r[1])[0]
        best_c = poly_models[best_ord]
        reg(results, f"PC1 Poly (deg={best_ord}, BIC)", "A",
            y_train, y_val, y_test,
            np.polyval(best_c, pc1_train),
            np.polyval(best_c, pc1_val),
            np.polyval(best_c, pc1_test), log)
        _tick()

    # B2 — Monotone PCHIP on PC1
    if fl.pchip_pc1 and not cancelled():
        mean_pc1_per_T_pchip = np.array(
            [pc1_train[y_train == t].mean() for t in train_temps_unique]
        )
        order_ = np.argsort(train_temps_unique)
        T_sort = train_temps_unique[order_]
        PC1_sort = mean_pc1_per_T_pchip[order_]
        g_pc1 = PchipInterpolator(T_sort, PC1_sort, extrapolate=True)
        T_fine = np.linspace(T_sort.min() - 0.1, T_sort.max() + 0.1, 8000)
        PC1_fine = g_pc1(T_fine)
        sidx = np.argsort(PC1_fine)
        PC1_sorted_inv = PC1_fine[sidx]
        T_sorted_inv   = T_fine[sidx]

        def _pc1_to_T(q):
            return np.interp(q, PC1_sorted_inv, T_sorted_inv)

        reg(results, "Monotone PCHIP (PC1)", "A",
            y_train, y_val, y_test,
            _pc1_to_T(pc1_train), _pc1_to_T(pc1_val), _pc1_to_T(pc1_test), log)
        _tick()

    # B3 — Sensor fusion (inverse variance on PC1 … PCk, k ≥ 2)
    if fl.sensor_fusion and k >= 2 and not cancelled():
        n_fuse = k
        fuse_T_grid = np.linspace(
            train_temps_unique.min() - 5,
            train_temps_unique.max() + 5,
            cfg.fusion_T_grid_pts,
        )
        g_grids: list[np.ndarray] = []
        sig_grids: list[np.ndarray] = []
        for i in range(n_fuse):
            mean_PCi = mean_pcs_per_T[:, i]
            sig_per_T = np.array(
                [X_train[y_train == t, i].std() for t in train_temps_unique]
            )
            sig_per_T = np.maximum(sig_per_T, 1e-6)
            ord_i = np.argsort(train_temps_unique)
            T_s = train_temps_unique[ord_i]
            PC_s = mean_PCi[ord_i]
            sig_s = sig_per_T[ord_i]
            g_i = PchipInterpolator(T_s, PC_s, extrapolate=True)
            sig_i = PchipInterpolator(T_s, sig_s, extrapolate=True)
            g_grids.append(g_i(fuse_T_grid))
            sig_grids.append(np.maximum(sig_i(fuse_T_grid), 1e-6))
        g_grid_arr = np.stack(g_grids, axis=0)
        sig_grid_arr = np.stack(sig_grids, axis=0)

        def _predict_fused(X_k):
            X_k = np.asarray(X_k, dtype=float)
            n = X_k.shape[0]
            out = np.empty(n)
            inv_sig2 = 1.0 / sig_grid_arr**2
            CHUNK = 2000
            for s in range(0, n, CHUNK):
                e = s + CHUNK
                blk = X_k[s:e, :n_fuse]
                cost = np.zeros((blk.shape[0], fuse_T_grid.size))
                for i in range(n_fuse):
                    r = blk[:, i:i + 1] - g_grid_arr[i][None, :]
                    cost += r * r * inv_sig2[i][None, :]
                out[s:e] = fuse_T_grid[np.argmin(cost, axis=1)]
            return out

        reg(
            results,
            f"Sensor Fusion ({k} PCs, inv-var)",
            "A",
            y_train,
            y_val,
            y_test,
            _predict_fused(X_train),
            _predict_fused(X_val),
            _predict_fused(X_test),
            log,
        )
        _tick()

    # B4 — MLR on first k PCs
    if fl.mlr and not cancelled():
        mlr = LinearRegression().fit(X_train, y_train)
        reg(
            results,
            f"MLR ({k} PCs)",
            "A",
            y_train,
            y_val,
            y_test,
            mlr.predict(X_train),
            mlr.predict(X_val),
            mlr.predict(X_test),
            log,
        )
        _tick()

    # B5 — Poly Reg 3PCs deg=7
    if fl.poly_3pcs_deg7 and k >= 3 and not cancelled():
        pf = PolynomialFeatures(degree=7, include_bias=False)
        Xt = pf.fit_transform(X_train[:, :3])
        lr7 = LinearRegression().fit(Xt, y_train)
        reg(results, "Poly Reg 3PCs (deg=7)", "A",
            y_train, y_val, y_test,
            lr7.predict(Xt),
            lr7.predict(pf.transform(X_val[:, :3])),
            lr7.predict(pf.transform(X_test[:, :3])), log)
        _tick()

    # B6 — 3D TPS
    if fl.tps_3pcs and k >= 3 and not cancelled():
        tps = RBFInterpolator(mean_pcs_per_T[:, :3], train_temps_unique,
                              kernel="thin_plate_spline")
        reg(results, "3D Spline (PC1,PC2,PC3 → T, TPS)", "A",
            y_train, y_val, y_test,
            tps(X_train[:, :3]), tps(X_val[:, :3]), tps(X_test[:, :3]), log)
        _tick()

    # B7 — LIR quadratic
    if fl.lir_quad and not cancelled():
        lir_all = compute_lir(prep.spectra_raw_crop, prep.wavelengths_crop, cfg.lir_cfg)
        lir_coeffs = fit_lir_quad(lir_all, ds.temperatures, ds.train_mask)
        fallback = float(y_train.mean())
        reg(results, "LIR (Quad, MAX)", "A",
            y_train, y_val, y_test,
            predict_T_lir_quad(lir_all[tr], lir_coeffs, fallback),
            predict_T_lir_quad(lir_all[va], lir_coeffs, fallback),
            predict_T_lir_quad(lir_all[te], lir_coeffs, fallback), log)
        _tick()

    # B8 — LIR Boltzmann (linear in 1000/T)
    if fl.lir_boltzmann and not cancelled():
        lir_all = compute_lir(prep.spectra_raw_crop, prep.wavelengths_crop, cfg.lir_cfg)
        lir_lin = fit_lir_boltzmann_linear(lir_all, ds.temperatures, ds.train_mask)
        fallback = float(y_train.mean())
        reg(results, "LIR (Boltzmann, MAX)", "A",
            y_train, y_val, y_test,
            predict_T_lir_boltzmann(lir_all[tr], lir_lin, fallback),
            predict_T_lir_boltzmann(lir_all[va], lir_lin, fallback),
            predict_T_lir_boltzmann(lir_all[te], lir_lin, fallback), log)
        _tick()

    # ─────────────────────────────────────────────────────────────────────
    # Group B — trees
    # ─────────────────────────────────────────────────────────────────────
    log("\n[Group B] Tree-based models")

    def _fit_tree(name, estimator):
        if not cancelled():
            Xtr = np.asarray(X_train, dtype=np.float64, order="C")
            Xva = np.asarray(X_val, dtype=np.float64, order="C")
            Xte = np.asarray(X_test, dtype=np.float64, order="C")

            def _pred(X):
                try:
                    return estimator.predict(X, validate_features=False)
                except TypeError:
                    return estimator.predict(X)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(Xtr, y_train)
                reg(results, name, "B",
                    y_train, y_val, y_test,
                    _pred(Xtr), _pred(Xva), _pred(Xte), log)
        _tick()

    if fl.random_forest:
        _fit_tree("Random Forest",
                  RandomForestRegressor(n_estimators=200, random_state=cfg.rng_seed, n_jobs=-1))
    if fl.gradient_boosting:
        _fit_tree("Gradient Boosting",
                  GradientBoostingRegressor(n_estimators=200, random_state=cfg.rng_seed))
    if fl.xgboost:
        xgb_mod = _try_import("xgboost")
        if xgb_mod:
            _fit_tree("XGBoost",
                      xgb_mod.XGBRegressor(n_estimators=200, objective="reg:squarederror",
                                           random_state=cfg.rng_seed, verbosity=0))
        else:
            log("    [skip] xgboost not installed")
            _tick()
    if fl.lightgbm:
        lgb_mod = _try_import("lightgbm")
        if lgb_mod:
            _fit_tree("LightGBM",
                      lgb_mod.LGBMRegressor(n_estimators=200, random_state=cfg.rng_seed,
                                            verbose=-1))
        else:
            log("    [skip] lightgbm not installed")
            _tick()
    if fl.extra_trees:
        _fit_tree("Extra Trees",
                  ExtraTreesRegressor(n_estimators=300, random_state=cfg.rng_seed, n_jobs=-1))
    if fl.catboost:
        cb_mod = _try_import("catboost")
        if cb_mod:
            _fit_tree("CatBoost",
                      cb_mod.CatBoostRegressor(
                          iterations=500, depth=6, learning_rate=0.05,
                          random_seed=cfg.rng_seed, verbose=0,
                          allow_writing_files=False))
        else:
            log("    [skip] catboost not installed")
            _tick()

    # ─────────────────────────────────────────────────────────────────────
    # Group C — kernel / probabilistic
    # ─────────────────────────────────────────────────────────────────────
    log("\n[Group C] Kernel / probabilistic")

    scaler_c = StandardScaler().fit(X_train)
    Xtr_s = scaler_c.transform(X_train)
    Xva_s = scaler_c.transform(X_val)
    Xte_s = scaler_c.transform(X_test)

    # SVR(RBF)
    if fl.svr_rbf and not cancelled():
        best_v, best_m = np.inf, None
        for C in cfg.svr_C_grid:
            for eps in cfg.svr_eps_grid:
                m = SVR(kernel="rbf", C=C, epsilon=eps, gamma="scale").fit(Xtr_s, y_train)
                rv = _val_rmse(y_val, m.predict(Xva_s))
                if rv < best_v:
                    best_v, best_m = rv, m
                    best_label = f"SVR (RBF, C={C}, eps={eps})"
        reg(results, best_label, "C",
            y_train, y_val, y_test,
            best_m.predict(Xtr_s), best_m.predict(Xva_s), best_m.predict(Xte_s), log)
        _tick()

    # SVR(Poly)
    if fl.svr_poly and not cancelled():
        best_v, best_m, best_label = np.inf, None, "SVR (Poly)"
        for C in cfg.svr_C_grid:
            for eps in cfg.svr_poly_eps_grid:
                for deg in cfg.svr_poly_deg_grid:
                    m = SVR(kernel="poly", C=C, epsilon=eps, degree=deg,
                            gamma="scale", coef0=1.0).fit(Xtr_s, y_train)
                    rv = _val_rmse(y_val, m.predict(Xva_s))
                    if rv < best_v:
                        best_v, best_m = rv, m
                        best_label = f"SVR (Poly d={deg}, C={C})"
        reg(results, best_label, "C",
            y_train, y_val, y_test,
            best_m.predict(Xtr_s), best_m.predict(Xva_s), best_m.predict(Xte_s), log)
        _tick()

    # GPR — fit on per-T mean PCs
    if fl.gpr and not cancelled():
        scaler_gpr = StandardScaler().fit(mean_pcs_per_T)
        mean_pcs_sc = scaler_gpr.transform(mean_pcs_per_T)
        kern = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * RBF(1.0, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(1e-3, noise_level_bounds=(1e-8, 1.0))
        )
        gpr = GaussianProcessRegressor(
            kernel=kern, normalize_y=True,
            n_restarts_optimizer=cfg.n_gpr_restarts, random_state=cfg.rng_seed,
        )
        gpr.fit(mean_pcs_sc, train_temps_unique)
        reg(results, "GPR", "C",
            y_train, y_val, y_test,
            gpr.predict(scaler_gpr.transform(X_train)),
            gpr.predict(scaler_gpr.transform(X_val)),
            gpr.predict(scaler_gpr.transform(X_test)), log)
        _tick()

    # KNN
    if fl.knn and not cancelled():
        best_v, best_m, best_k = np.inf, None, 3
        for kk in cfg.knn_k_grid:
            m = KNeighborsRegressor(n_neighbors=kk).fit(X_train, y_train)
            rv = _val_rmse(y_val, m.predict(X_val))
            if rv < best_v:
                best_v, best_m, best_k = rv, m, kk
        reg(results, f"KNN (k={best_k})", "C",
            y_train, y_val, y_test,
            best_m.predict(X_train), best_m.predict(X_val), best_m.predict(X_test), log)
        _tick()

    # ─────────────────────────────────────────────────────────────────────
    # Group D — extras
    # ─────────────────────────────────────────────────────────────────────
    log("\n[Group D] Extra models")

    y_mean = y_train.mean()
    y_train_c = y_train - y_mean

    # KRR(Poly)
    if fl.krr_poly and not cancelled():
        best_v, best_m, best_label = np.inf, None, "KRR Poly"
        for alpha in cfg.krr_alpha_grid:
            for deg in cfg.krr_deg_grid:
                for coef0 in (0.0, 1.0):
                    m = KernelRidge(kernel="poly", alpha=alpha, degree=deg,
                                    coef0=coef0).fit(Xtr_s, y_train_c)
                    rv = _val_rmse(y_val, m.predict(Xva_s) + y_mean)
                    if rv < best_v:
                        best_v, best_m = rv, m
                        best_label = f"Kernel Ridge (Poly d={deg})"
        reg(results, best_label, "D",
            y_train, y_val, y_test,
            best_m.predict(Xtr_s) + y_mean,
            best_m.predict(Xva_s) + y_mean,
            best_m.predict(Xte_s) + y_mean, log)
        _tick()

    # PLS — on full SNV spectra
    if fl.pls and not cancelled():
        best_v, best_m, best_nc = np.inf, None, 3
        for nc in cfg.pls_nc_grid:
            if nc >= min(S_train.shape[0], S_train.shape[1]):
                continue
            m = PLSRegression(n_components=nc).fit(S_train, y_train)
            rv = _val_rmse(y_val, m.predict(S_val).ravel())
            if rv < best_v:
                best_v, best_m, best_nc = rv, m, nc
        reg(results, f"PLS (n={best_nc})", "D",
            y_train, y_val, y_test,
            best_m.predict(S_train).ravel(),
            best_m.predict(S_val).ravel(),
            best_m.predict(S_test).ravel(), log)
        _tick()

    # Bayesian Ridge on poly-3 PC features
    if fl.bayesian_ridge and not cancelled():
        pf3 = PolynomialFeatures(degree=3, include_bias=False)
        Xtr_p = pf3.fit_transform(X_train)
        Xva_p = pf3.transform(X_val)
        Xte_p = pf3.transform(X_test)
        sc_p = StandardScaler().fit(Xtr_p)
        Xtr_ps = sc_p.transform(Xtr_p)
        Xva_ps = sc_p.transform(Xva_p)
        Xte_ps = sc_p.transform(Xte_p)
        br = BayesianRidge().fit(Xtr_ps, y_train_c)
        reg(results, "Bayesian Ridge (poly3 PCs)", "D",
            y_train, y_val, y_test,
            br.predict(Xtr_ps) + y_mean,
            br.predict(Xva_ps) + y_mean,
            br.predict(Xte_ps) + y_mean, log)
        _tick()

    # ElasticNet on poly-3 PC features
    if fl.elasticnet and not cancelled():
        try:
            pf3  # reuse from Bayesian Ridge if already computed
            Xtr_ps
        except NameError:
            pf3 = PolynomialFeatures(degree=3, include_bias=False)
            Xtr_p = pf3.fit_transform(X_train)
            Xva_p = pf3.transform(X_val)
            Xte_p = pf3.transform(X_test)
            sc_p = StandardScaler().fit(Xtr_p)
            Xtr_ps = sc_p.transform(Xtr_p)
            Xva_ps = sc_p.transform(Xva_p)
            Xte_ps = sc_p.transform(Xte_p)
        best_v, best_m, best_label = np.inf, None, "ElasticNet"
        for alpha in cfg.en_alpha_grid:
            for l1 in cfg.en_l1_grid:
                m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=50000,
                               random_state=cfg.rng_seed).fit(Xtr_ps, y_train_c)
                rv = _val_rmse(y_val, m.predict(Xva_ps) + y_mean)
                if rv < best_v:
                    best_v, best_m = rv, m
                    best_label = f"ElasticNet (poly3, α={alpha}, l1={l1})"
        reg(results, best_label, "D",
            y_train, y_val, y_test,
            best_m.predict(Xtr_ps) + y_mean,
            best_m.predict(Xva_ps) + y_mean,
            best_m.predict(Xte_ps) + y_mean, log)
        _tick()

    # MLP
    if fl.mlp and not cancelled():
        best_v, best_m, best_arch = np.inf, None, (64,)
        for arch in cfg.mlp_archs:
            m = MLPRegressor(
                hidden_layer_sizes=arch, activation="relu", solver="adam",
                max_iter=3000, random_state=cfg.rng_seed, learning_rate_init=1e-3,
                early_stopping=True, n_iter_no_change=40, validation_fraction=0.15,
            ).fit(Xtr_s, y_train)
            rv = _val_rmse(y_val, m.predict(Xva_s))
            if rv < best_v:
                best_v, best_m, best_arch = rv, m, arch
        reg(results, f"MLP {best_arch}", "D",
            y_train, y_val, y_test,
            best_m.predict(Xtr_s), best_m.predict(Xva_s), best_m.predict(Xte_s), log)
        _tick()

    # Group E — neural (ANN on PCA, ANN on SNV spectrum, 1D CNN on SNV)
    if (fl.ann_pca_mlp or fl.ann_snv_mlp or fl.cnn_snv_1d) and not cancelled():
        log("\n[Group E] Neural models (ANN / CNN)")
        from .neural_models import (
            register_ann_pca_mlp,
            register_ann_snv_mlp,
            register_cnn_snv_1d,
        )

        if fl.ann_pca_mlp and not cancelled():
            register_ann_pca_mlp(
                results, cfg, X_train, X_val, X_test,
                y_train, y_val, y_test, reg, log, cancelled,
            )
            _tick()
        if fl.ann_snv_mlp and not cancelled():
            register_ann_snv_mlp(
                results, cfg, S_train, S_val, S_test,
                y_train, y_val, y_test, reg, log, cancelled,
            )
            _tick()
        if fl.cnn_snv_1d and not cancelled():
            register_cnn_snv_1d(
                results, cfg, S_train, S_val, S_test,
                y_train, y_val, y_test, reg, log, cancelled,
            )
            _tick()

    log(f"\n[Done] {len(results.models)} models evaluated.")
    return results

# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for LT2 Thermometry v1.0
#
# Run from the repo root (Yb/):
#   pyinstaller packaging/LT2_Thermometry.spec
#
# Or let the build scripts do it — see packaging/build_pyinstaller.sh/.bat.
#
# Produces a onedir bundle: dist/LT2 Thermometry/
# The build scripts then wrap that into platform installers.

import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(SPECPATH).parent          # Yb/
LT2_GUI   = REPO_ROOT / "lt2_gui"
LT2_CORE  = REPO_ROOT / "lt2_core"
ASSETS    = LT2_GUI / "assets"

# ── Collect helpers ───────────────────────────────────────────────────────────
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# Helper: skip submodules that drag in optional test/dev deps.
def _no_tests(name: str) -> bool:
    bad = ("testing", "tests", "test_", "_test", "benchmarks")
    return not any(part in name for part in bad)


# PySide6 (Qt6 GUI framework + plugins)
pyside6_datas, pyside6_binaries, pyside6_hiddenimports = collect_all(
    "PySide6", filter_submodules=_no_tests, on_error="warn"
)

# matplotlib – backends needed at runtime
mpl_datas, mpl_binaries, mpl_hi = collect_all(
    "matplotlib", filter_submodules=_no_tests, on_error="warn"
)

# scikit-learn – many cython extensions discovered only at import time
sklearn_datas, sklearn_binaries, sklearn_hi = collect_all(
    "sklearn", filter_submodules=_no_tests, on_error="warn"
)

# torch – large; collect_all also grabs native libs
torch_datas, torch_binaries, torch_hi = collect_all(
    "torch", filter_submodules=_no_tests, on_error="warn"
)

# xgboost.testing requires 'hypothesis' (a pytest extra) — skip it
xgb_datas,  xgb_binaries,  xgb_hi  = collect_all(
    "xgboost", filter_submodules=_no_tests, on_error="warn"
)
lgb_datas,  lgb_binaries,  lgb_hi  = collect_all(
    "lightgbm", filter_submodules=_no_tests, on_error="warn"
)
cb_datas,   cb_binaries,   cb_hi   = collect_all(
    "catboost", filter_submodules=_no_tests, on_error="warn"
)

# pandas (may have optional arrow/parquet extensions — keep it simple)
pd_datas,  pd_binaries,  pd_hi   = collect_all(
    "pandas", filter_submodules=_no_tests, on_error="warn"
)

# scipy
scipy_hi = collect_submodules("scipy", filter=_no_tests, on_error="warn")
scipy_datas = collect_data_files("scipy")

# ── Application data files ────────────────────────────────────────────────────
# (src_path, dest_path_inside_bundle)
app_datas = [
    (str(ASSETS), "assets"),                         # group_*.png, app_icon.*, LT2_Help.pdf
    (str(LT2_GUI / "__version__.py"), "lt2_gui"),
    (str(LT2_GUI / "citation.json"),  "lt2_gui"),    # editable citation; update without rebuild
]

# ── Combined datas / binaries / hidden imports ────────────────────────────────
all_datas = (
    app_datas
    + pyside6_datas
    + mpl_datas
    + sklearn_datas
    + torch_datas
    + xgb_datas + lgb_datas + cb_datas
    + pd_datas
    + scipy_datas
)

all_binaries = (
    pyside6_binaries
    + mpl_binaries
    + sklearn_binaries
    + torch_binaries
    + xgb_binaries + lgb_binaries + cb_binaries
    + pd_binaries
)

hidden_imports = list({
    # sklearn cython helpers commonly missed by static analysis
    "sklearn.utils._cython_blas",
    "sklearn.utils._weight_vector",
    "sklearn.utils._seq_dataset",
    "sklearn.neighbors._quad_tree",
    "sklearn.tree._utils",
    "sklearn.ensemble._gradient_boosting",
    # neural_models lazy-imports
    "torch",
    "torch.nn",
    "torch.optim",
    # tree libraries
    "xgboost",
    "lightgbm",
    "catboost",
    # matplotlib backends used in GUI and export
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_pdf",
    # scipy interpolation used by benchmark
    "scipy.interpolate",
    "scipy.interpolate._rbfinterp",
    "scipy.interpolate._pchip",
    # joblib multiprocessing backend
    "joblib",
    "joblib._multiprocessing_helpers",
    # app packages
    "lt2_gui",
    "lt2_gui.main_window",
    "lt2_gui.splash",
    "lt2_gui.workers",
    "lt2_gui.help_text",
    "lt2_gui.paths",
    "lt2_gui.widgets.data_tab",
    "lt2_gui.widgets.preprocess_tab",
    "lt2_gui.widgets.pca_tab",
    "lt2_gui.widgets.models_tab",
    "lt2_gui.widgets.run_tab",
    "lt2_gui.widgets.results_tab",
    "lt2_gui.widgets.help_tab",
    "lt2_gui.widgets.about_tab",
    "lt2_gui.widgets.help_widgets",
    "lt2_core",
    "lt2_core.benchmark",
    "lt2_core.dataset",
    "lt2_core.preprocess",
    "lt2_core.pca_analysis",
    "lt2_core.spectrum_io",
    "lt2_core.metrics",
    "lt2_core.plots",
    "lt2_core.neural_models",
} | set(pyside6_hiddenimports)
  | set(mpl_hi)
  | set(sklearn_hi)
  | set(torch_hi)
  | set(xgb_hi) | set(lgb_hi) | set(cb_hi)
  | set(pd_hi)
  | set(scipy_hi)
)

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    [str(LT2_GUI / "__main__.py")],
    pathex=[str(REPO_ROOT)],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=hidden_imports,
    hookspath=["packaging/hooks"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # dev-only & tests
        "pytest", "_pytest", "py", "IPython", "ipykernel",
        "tkinter", "_tkinter",
        # unused torch backends to reduce size
        "torch.distributed",
        "torch.testing",
        # unused jupyter/notebook deps that sometimes get dragged in
        "notebook", "nbformat", "nbconvert",
    ],
    noarchive=False,
    optimize=1,
)

# ── PYZ (bytecode archive) ────────────────────────────────────────────────────
pyz = PYZ(a.pure)

# ── EXE (launcher stub) ──────────────────────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SpectraSensML",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX compression sometimes triggers AV false-positives
    console=False,      # no terminal window on Windows/macOS
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,   # None = native; set "x86_64" or "arm64" for explicit
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ASSETS / "app_icon.ico") if (ASSETS / "app_icon.ico").exists() else None,
)

# ── COLLECT (onedir bundle) ───────────────────────────────────────────────────
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SpectraSensML",
)

# ── macOS .app bundle ─────────────────────────────────────────────────────────
# Only active on macOS; harmless to leave on other platforms (PyInstaller
# only uses BUNDLE when building on macOS).
app = BUNDLE(
    coll,
    name="SpectraSensML.app",
    icon=str(ASSETS / "app_icon.icns") if (ASSETS / "app_icon.icns").exists() else None,
    bundle_identifier="org.omasgroup.spectrasenml",
    version="1.0",
    info_plist={
        "NSPrincipalClass": "NSApplication",
        "NSAppleScriptEnabled": False,
        "CFBundleDisplayName": "SpectraSensML",
        "CFBundleShortVersionString": "1.0",
        "NSHighResolutionCapable": True,
    },
)

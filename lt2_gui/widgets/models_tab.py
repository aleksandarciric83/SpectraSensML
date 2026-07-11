"""Models tab — grouped checkboxes with replaceable PNG icons."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from lt2_core.benchmark import ModelFlags

from ..paths import resource_path
from .help_widgets import HelpButton

_ASSETS_DIR = resource_path("assets")
# Display height for group illustrations (source PNGs are large; scale down for UI).
_ICON_HEIGHT = 168


def _group_pixmap(letter: str) -> QPixmap:
    path = _ASSETS_DIR / f"group_{letter.lower()}.png"
    pix = QPixmap(str(path))
    if pix.isNull():
        return pix
    return pix.scaledToHeight(
        _ICON_HEIGHT,
        Qt.TransformationMode.SmoothTransformation,
    )

# (field, label, min_k, group) — Group A (splines) before Group B (trees) in UI
MODEL_DEFS = [
    ("pc1_poly_bic", "PC1 Polynomial (BIC order)", 1, "A"),
    ("pchip_pc1", "Monotone PCHIP (PC1)", 1, "A"),
    ("sensor_fusion", "Sensor Fusion (PCs, inv-var)", 2, "A"),
    ("mlr", "MLR (k PCs)", 1, "A"),
    ("poly_3pcs_deg7", "Poly Reg (first 3 PCs, deg=7)", 3, "A"),
    ("tps_3pcs", "3D Thin-Plate Spline (PC1–PC3)", 3, "A"),
    ("lir_quad", "LIR (Quad, MAX)", 0, "A"),
    ("lir_boltzmann", "LIR (Boltzmann, MAX)", 0, "A"),
    ("random_forest", "Random Forest", 1, "B"),
    ("gradient_boosting", "Gradient Boosting", 1, "B"),
    ("xgboost", "XGBoost", 1, "B"),
    ("lightgbm", "LightGBM", 1, "B"),
    ("extra_trees", "Extra Trees", 1, "B"),
    ("catboost", "CatBoost", 1, "B"),
    ("svr_rbf", "SVR (RBF kernel)", 1, "C"),
    ("svr_poly", "SVR (Polynomial kernel)", 1, "C"),
    ("gpr", "Gaussian Process Regressor", 1, "C"),
    ("knn", "K-Nearest Neighbours", 1, "C"),
    ("krr_poly", "Kernel Ridge (Polynomial)", 1, "D"),
    ("pls", "PLS Regression (SNV spectra)", 1, "D"),
    ("bayesian_ridge", "Bayesian Ridge (poly3 PCs)", 3, "D"),
    ("elasticnet", "ElasticNet (poly3 PCs)", 3, "D"),
    ("mlp", "MLP (sklearn, val-tuned arch)", 1, "D"),
    ("ann_pca_mlp", "ANN (MLP on k PCs, lbfgs, val-tuned)", 1, "E"),
    ("ann_snv_mlp", "ANN (MLP on SNV spectrum, val-tuned)", 1, "E"),
    ("cnn_snv_1d", "CNN (1D Conv on SNV, val-tuned)", 1, "E"),
]

GROUP_ORDER = ("A", "B", "C", "D", "E")
GROUP_LABELS = {
    "A": "Group A — Splines / LIR / fusion",
    "B": "Group B — Tree-based",
    "C": "Group C — Kernel / probabilistic",
    "D": "Group D — Regularised / neural",
    "E": "Group E — Neural (ANN / CNN)",
}


class ModelsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._checkboxes: dict[str, QCheckBox] = {}
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        btn_row = QHBoxLayout()
        sa = QPushButton("Select All")
        sa.clicked.connect(lambda: self._set_all(True))
        sn = QPushButton("Deselect All")
        sn.clicked.connect(lambda: self._set_all(False))
        btn_row.addWidget(sa)
        btn_row.addWidget(sn)
        btn_row.addStretch()
        btn_row.addWidget(HelpButton("model_checks"))
        outer.addLayout(btn_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)

        for grp in GROUP_ORDER:
            title = GROUP_LABELS[grp]
            group_box = QGroupBox()
            row = QHBoxLayout(group_box)

            left = QVBoxLayout()
            title_lb = QLabel(title)
            f = title_lb.font()
            f.setBold(True)
            title_lb.setFont(f)
            left.addWidget(title_lb)
            for field, label, min_k, g2 in MODEL_DEFS:
                if g2 != grp:
                    continue
                cb = QCheckBox(label)
                cb.setChecked(True)
                if min_k > 0:
                    cb.setToolTip(f"Requires k ≥ {min_k} PCA components")
                self._checkboxes[field] = cb
                left.addWidget(cb)
            row.addLayout(left, stretch=1)

            pic = QLabel()
            pm = _group_pixmap(grp)
            if not pm.isNull():
                pic.setPixmap(pm)
                pic.setMinimumSize(pm.size())
                pic.setMaximumSize(pm.size())
            pic.setAlignment(
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter
            )
            pic.setScaledContents(False)
            pic.setToolTip(
                f"Replace with your own image: {_ASSETS_DIR / f'group_{grp.lower()}.png'}"
            )
            row.addWidget(pic, 0)

            inner_layout.addWidget(group_box)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll)

    def _set_all(self, state: bool):
        for cb in self._checkboxes.values():
            if cb.isEnabled():
                cb.setChecked(state)

    def update_k(self, k: int):
        for field, _label, min_k, _grp in MODEL_DEFS:
            cb = self._checkboxes.get(field)
            if cb is None:
                continue
            if min_k > 0 and k < min_k:
                cb.setEnabled(False)
                cb.setToolTip(f"Disabled: requires k ≥ {min_k} (currently k={k})")
            else:
                cb.setEnabled(True)
                cb.setToolTip(
                    f"Requires k ≥ {min_k} PCA components" if min_k > 0 else ""
                )

    def model_flags(self) -> ModelFlags:
        flags = ModelFlags()
        for field, _, _, _ in MODEL_DEFS:
            cb = self._checkboxes.get(field)
            if cb is not None:
                setattr(flags, field, cb.isChecked() and cb.isEnabled())
        return flags

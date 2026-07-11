"""PCA tab — Run PCA, scree, loadings, scores, PCs vs T."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .help_widgets import labeled_help_row

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class PCATab(QWidget):
    n_components_changed = Signal(int)
    pca_run_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pca_result = None
        self._var_name = "T"
        self._var_unit = "K"
        self._build_ui()

    def set_quantity(self, name: str, unit: str) -> None:
        self._var_name = name or "T"
        self._var_unit = unit or "K"

    def _build_ui(self):
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        grp = QGroupBox("PCA")
        form = QFormLayout(grp)
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 50)
        self.k_spin.setValue(3)
        self.k_spin.valueChanged.connect(self.n_components_changed)
        form.addRow(
            labeled_help_row("n_components (k) for benchmark:", "pca_n_use"),
            self.k_spin,
        )
        self.var_label = QLabel("Run PCA on the Pre-process tab first.")
        form.addRow("Explained variance:", self.var_label)
        self.run_btn = QPushButton("Run PCA (fit on train)")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 6px 16px;")
        self.run_btn.clicked.connect(self.pca_run_requested.emit)
        form.addRow(self.run_btn)
        ctrl.addWidget(grp)
        layout.addLayout(ctrl)

        if _HAS_MPL:
            self.canvas = FigureCanvas(Figure(figsize=(10, 10.2)))
            layout.addWidget(self.canvas, 1)
        else:
            layout.addWidget(QLabel("[matplotlib required]"))

    def n_components(self) -> int:
        return self.k_spin.value()

    def update_full_pca(self, pca_result, ds, wavelengths_crop: np.ndarray):
        """Draw scree, loadings, scatter plots, and PC vs T (first three PCs)."""
        self._pca_result = pca_result
        ev = pca_result.explained_variance_ratio
        k_fit = len(ev)
        self.k_spin.setMaximum(max(1, k_fit))
        ev_pct = ev * 100
        cum = np.cumsum(ev_pct)
        self.var_label.setText(
            "  ".join(f"PC{i+1}={v:.1f}%" for i, v in enumerate(ev_pct[: min(6, k_fit)]))
            + (f"   Σ={cum[-1]:.1f}%" if len(cum) else "")
        )
        if not _HAS_MPL:
            return

        X = pca_result.X_all
        T = ds.temperatures
        labs = ds.labels.astype(str)
        fig = self.canvas.figure
        fig.clear()
        gs = fig.add_gridspec(3, 2, hspace=0.52, wspace=0.32)

        ax0 = fig.add_subplot(gs[0, 0])
        xs = np.arange(1, k_fit + 1)
        ax0.bar(xs, ev_pct, color="steelblue", alpha=0.85)
        ax0.plot(xs, cum, "ro-", ms=4, label="Cumulative")
        ax0.set_xlabel("PC")
        ax0.set_ylabel("Explained variance (%)")
        ax0.set_title("Scree")
        ax0.legend(fontsize=8)
        ax0.set_ylim(0, min(105, float(cum.max()) * 1.1 + 5))

        ax1 = fig.add_subplot(gs[0, 1])
        for i in range(k_fit):
            ax1.plot(wavelengths_crop, pca_result.pca.components_[i], label=f"PC{i+1}")
        ax1.axhline(0, color="k", lw=0.4)
        ax1.set_xlabel("λ (nm)")
        ax1.set_ylabel("Loading")
        ax1.set_title("Loadings")
        ax1.legend(fontsize=7, ncol=2)

        ax2 = fig.add_subplot(gs[1, 0])
        sc = ax2.scatter(
            X[:, 0],
            X[:, 1] if k_fit > 1 else np.zeros(len(X)),
            c=T,
            cmap="nipy_spectral",
            s=4,
            alpha=0.55,
        )
        cb_label = f"{self._var_name} ({self._var_unit})"
        fig.colorbar(sc, ax=ax2, label=cb_label)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2" if k_fit > 1 else "(n/a)")
        ax2.set_title("PC1 vs PC2")

        ax3 = fig.add_subplot(gs[1, 1])
        if k_fit > 2:
            sc2 = ax3.scatter(X[:, 0], X[:, 2], c=T, cmap="nipy_spectral", s=4, alpha=0.55)
            fig.colorbar(sc2, ax=ax3, label=cb_label)
            ax3.set_xlabel("PC1")
            ax3.set_ylabel("PC3")
            ax3.set_title("PC1 vs PC3")
        else:
            ax3.text(0.5, 0.5, "Need k ≥ 3 for PC1 vs PC3", ha="center", va="center")

        split_styles = {
            "train": ("tab:blue", "o", 4, 0.35),
            "val": ("tab:orange", "s", 4, 0.5),
            "test_unseen": ("tab:red", "^", 5, 0.75),
        }
        n_pc_plot = min(3, k_fit)
        if n_pc_plot > 0:
            sub_pc = gs[2, :].subgridspec(1, n_pc_plot)
            for j, pci in enumerate(range(n_pc_plot)):
                axp = fig.add_subplot(sub_pc[0, j])
                for split, (color, mk, ms, al) in split_styles.items():
                    m = labs == split
                    if not m.any():
                        continue
                    axp.scatter(T[m], X[m, pci], c=color, marker=mk, s=ms, alpha=al, label=split)
                axp.set_xlabel(f"{self._var_name} ({self._var_unit})")
                axp.set_ylabel(f"PC{pci+1}")
                axp.set_title(f"PC{pci+1} vs {self._var_name}")
                axp.legend(fontsize=6, markerscale=1.2)

        fig.suptitle("PCA (fitted on train spectra only)", fontsize=12, y=0.995)
        fig.subplots_adjust(
            left=0.07, right=0.98, top=0.93, bottom=0.06, hspace=0.42, wspace=0.34
        )
        self.canvas.draw()

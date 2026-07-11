"""Pre-processing tab — options + Run to preview cropped/normalised spectra."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from lt2_core.preprocess import Normalization, LIRConfig

from .help_widgets import HelpButton, labeled_help_row

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class PreprocessTab(QWidget):
    preprocess_run_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._var_name = "T"
        self._var_unit = "K"
        self._build_ui()

    def set_quantity(self, name: str, unit: str) -> None:
        """Update the variable name/unit used in axis labels and titles."""
        self._var_name = name or "T"
        self._var_unit = unit or "K"

    def _build_ui(self):
        outer = QHBoxLayout(self)
        left = QVBoxLayout()
        wl_group = QGroupBox("Wavelength Range")
        wl_layout = QFormLayout(wl_group)
        self.wl_min = QDoubleSpinBox()
        self.wl_min.setRange(200, 2000)
        self.wl_min.setValue(900)
        self.wl_min.setSuffix(" nm")
        self.wl_max = QDoubleSpinBox()
        self.wl_max.setRange(200, 2000)
        self.wl_max.setValue(1100)
        self.wl_max.setSuffix(" nm")
        wl_layout.addRow(labeled_help_row("λ min:", "wl_min"), self.wl_min)
        wl_layout.addRow(labeled_help_row("λ max:", "wl_max"), self.wl_max)
        left.addWidget(wl_group)

        norm_group = QGroupBox("Normalisation")
        norm_outer = QVBoxLayout(norm_group)
        norm_header = QHBoxLayout()
        norm_header.addStretch()
        norm_header.addWidget(HelpButton("normalization"))
        norm_outer.addLayout(norm_header)
        norm_layout = QVBoxLayout()
        norm_outer.addLayout(norm_layout)
        self._norm_bg = QButtonGroup(self)
        for i, (label, val) in enumerate(
            [
                ("SNV (Standard Normal Variate)", Normalization.SNV),
                ("Spectral Maximum (MAX)", Normalization.MAX),
                ("Spectral Area (AREA)", Normalization.AREA),
                ("None (raw intensity)", Normalization.NONE),
            ]
        ):
            rb = QRadioButton(label)
            rb.setChecked(i == 0)
            rb.setProperty("norm_value", val)
            self._norm_bg.addButton(rb, i)
            norm_layout.addWidget(rb)
        bg_row = QHBoxLayout()
        self.bg_subtract_cb = QCheckBox(
            "Background subtract before normalisation (per spectrum min)"
        )
        bg_row.addWidget(self.bg_subtract_cb)
        bg_row.addWidget(HelpButton("bg_subtract"))
        bg_row.addStretch()
        norm_layout.addLayout(bg_row)
        left.addWidget(norm_group)

        lir_outer = QGroupBox("LIR Bands (shown as shaded regions on preview)")
        lir_top = QVBoxLayout(lir_outer)
        lir_header = QHBoxLayout()
        lir_header.addStretch()
        lir_header.addWidget(HelpButton("lir_bands"))
        lir_top.addLayout(lir_header)
        lir_form_w = QWidget()
        lir_layout = QFormLayout(lir_form_w)
        lir_top.addWidget(lir_form_w)
        lir_group = lir_outer

        def _ds(lo, hi, val):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setValue(val)
            sb.setSuffix(" nm")
            return sb

        self.lir_hi_lo = _ds(200, 2000, 900.0)
        self.lir_hi_hi = _ds(200, 2000, 983.0)
        self.lir_low_lo = _ds(200, 2000, 983.0)
        self.lir_low_hi = _ds(200, 2000, 987.0)
        lir_layout.addRow("I1 low / high:", self._pair_row(self.lir_hi_lo, self.lir_hi_hi))
        lir_layout.addRow("I2 low / high:", self._pair_row(self.lir_low_lo, self.lir_low_hi))
        left.addWidget(lir_group)

        self.run_btn = QPushButton("Run pre-processing")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 6px 16px;")
        self.run_btn.clicked.connect(self.preprocess_run_requested.emit)
        left.addWidget(self.run_btn)
        left.addStretch()
        outer.addLayout(left, 0)

        if _HAS_MPL:
            self.canvas = FigureCanvas(Figure(figsize=(6.5, 4.8)))
            outer.addWidget(self.canvas, 1)
        else:
            outer.addWidget(QLabel("[matplotlib required]"))

    def _pair_row(self, a, b):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(a)
        h.addWidget(b)
        return row

    def normalization(self) -> Normalization:
        btn = self._norm_bg.checkedButton()
        raw = btn.property("norm_value") if btn else Normalization.SNV
        if isinstance(raw, Normalization):
            return raw
        if raw is None:
            return Normalization.SNV
        try:
            return Normalization(str(raw))
        except ValueError:
            return Normalization.SNV

    def bg_subtract(self) -> bool:
        return self.bg_subtract_cb.isChecked()

    def wl_range(self) -> tuple[float, float]:
        return self.wl_min.value(), self.wl_max.value()

    def lir_config(self) -> LIRConfig:
        return LIRConfig(
            hi_lo=self.lir_hi_lo.value(),
            hi_hi=self.lir_hi_hi.value(),
            low_lo=self.lir_low_lo.value(),
            low_hi=self.lir_low_hi.value(),
        )

    def show_preprocessed(self, prep, temperatures, folders):
        """Plot mean normalised spectrum + LIR band shading (30 % opacity)."""
        if not _HAS_MPL:
            return
        import numpy as np

        wl = prep.wavelengths_crop
        cfg = self.lir_config()
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        uniq_f = []
        for f in folders:
            if f not in uniq_f:
                uniq_f.append(f)
        temps = [float(np.unique(temperatures[folders == fn])[0]) for fn in uniq_f]
        t0, t1 = min(temps), max(temps)
        if t1 <= t0:
            t1 = t0 + 1.0
        norm = plt.Normalize(t0, t1)
        cmap = plt.cm.nipy_spectral
        for fn in uniq_f:
            m = folders == fn
            T = float(np.unique(temperatures[m])[0])
            y = prep.spectra_norm[m].mean(axis=0)
            ax.plot(wl, y, color=cmap(norm(T)), lw=0.85, alpha=0.9)
        ax.axvspan(cfg.hi_lo, cfg.hi_hi, color="tab:blue", alpha=0.3, label="I1 band")
        ax.axvspan(cfg.low_lo, cfg.low_hi, color="tab:orange", alpha=0.3, label="I2 band")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=f"{self._var_name} ({self._var_unit})")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(f"Normalised ({prep.norm.value})")
        ax.set_title("Mean pre-processed spectrum per folder")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        self.canvas.draw()

"""Data tab — folder picker, per-folder T and role table, spectrum preview."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

ROLE_LABELS = ("Do not use", "Train/Val", "test_unseen")
ROLE_TO_INTERNAL = {"Do not use": "omit", "Train/Val": "train_val", "test_unseen": "test_unseen"}
INTERNAL_TO_LABEL = {v: k for k, v in ROLE_TO_INTERNAL.items()}


class DataTab(QWidget):
    """Load raw files; user sets folder T(K) and role before Apply."""

    load_requested = Signal(str, dict)
    apply_roles_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._root: str = ""
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        folder_group = QGroupBox("Spectra Root Folder")
        fg_layout = QHBoxLayout(folder_group)
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText(
            "Select folder containing temperature subfolders "
            "(p/mXXX = Celsius offset → K,  or plain integers = Kelvin)"
        )
        self.folder_edit.setReadOnly(True)
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse)
        fg_layout.addWidget(self.folder_edit)
        fg_layout.addWidget(self.browse_btn)
        layout.addWidget(folder_group)

        from .help_widgets import labeled_help_row

        opt_group = QGroupBox("Load Options")
        opt_layout = QFormLayout(opt_group)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        opt_layout.addRow(labeled_help_row("RNG seed:", "rng_seed"), self.seed_spin)
        self.train_frac_spin = QSpinBox()
        self.train_frac_spin.setRange(10, 95)
        self.train_frac_spin.setValue(80)
        self.train_frac_spin.setSuffix(" %")
        opt_layout.addRow(
            labeled_help_row("Train fraction (within Train/Val folders):", "train_frac"),
            self.train_frac_spin,
        )
        self.var_name_edit = QLineEdit("T")
        opt_layout.addRow(
            labeled_help_row("Variable name (default T):", "var_name"),
            self.var_name_edit,
        )
        self.var_unit_edit = QLineEdit("K")
        opt_layout.addRow(
            labeled_help_row("Variable unit (default K):", "var_unit"),
            self.var_unit_edit,
        )
        self.integ_edit = QLineEdit("Integration Time (sec):")
        opt_layout.addRow(
            labeled_help_row("Integration-time prefix:", "integ_prefix"),
            self.integ_edit,
        )
        self.marker_edit = QLineEdit(">>>>>Begin Spectral Data<<<<<")
        opt_layout.addRow(
            labeled_help_row("Header end marker (blank = auto-detect):", "header_marker"),
            self.marker_edit,
        )
        self.ext_edit = QLineEdit("*.txt")
        opt_layout.addRow(
            labeled_help_row("File extension:", "file_ext"),
            self.ext_edit,
        )
        layout.addWidget(opt_group)

        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Load Spectra")
        self.load_btn.setEnabled(False)
        self.load_btn.setStyleSheet("font-weight: bold; padding: 6px 20px;")
        self.load_btn.clicked.connect(self._emit_load)
        self.apply_btn = QPushButton("Apply folder roles → dataset")
        self.apply_btn.setToolTip(
            "Rebuild train/val/test labels and temperatures from the table "
            "(required after editing T or Role)."
        )
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.apply_roles_requested.emit)
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.apply_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.summary_label = QLabel("No dataset loaded.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        splitter = QSplitter(Qt.Horizontal)
        self.folder_table = QTableWidget(0, 3)
        # Header is rewritten on dataset load so it matches the user's variable.
        self.folder_table.setHorizontalHeaderLabels(["Folder", "T (K)", "Role"])
        self.folder_table.horizontalHeader().setStretchLastSection(True)
        self.folder_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        splitter.addWidget(self.folder_table)

        if _HAS_MPL:
            self.spec_canvas = FigureCanvas(Figure(figsize=(5.5, 4.2)))
            splitter.addWidget(self.spec_canvas)
        else:
            splitter.addWidget(QLabel("[matplotlib required for spectrum preview]"))
        splitter.setSizes([420, 520])
        layout.addWidget(splitter, 1)

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "Select Spectra Root Folder")
        if path:
            self.folder_edit.setText(path)
            self.load_btn.setEnabled(True)

    def _emit_load(self):
        root = self.folder_edit.text()
        if not root:
            return
        self._root = root
        marker = self.marker_edit.text().strip() or None
        kwargs = {
            "rng_seed": self.seed_spin.value(),
            "train_frac": self.train_frac_spin.value() / 100.0,
            "integ_prefix": self.integ_edit.text(),
            "header_marker": marker,
            "file_ext": self.ext_edit.text() or "*.txt",
        }
        self.load_requested.emit(root, kwargs)

    def set_table_from_dataset(self, ds):
        """Populate table from unique folders in *ds* (after load or apply)."""
        import numpy as np

        vn, vu = self.var_name(), self.var_unit()
        self.folder_table.setHorizontalHeaderLabels(
            ["Folder", f"{vn} ({vu})", "Role"]
        )
        folders = []
        seen = set()
        for f in ds.folders:
            if f not in seen:
                seen.add(f)
                folders.append(f)

        self.folder_table.setRowCount(len(folders))
        for row, fname in enumerate(folders):
            mask = ds.folders == fname
            t0 = float(np.unique(ds.temperatures[mask])[0])
            lab = str(np.unique(ds.labels[mask].astype(str))[0])
            if lab == "train" or lab == "val":
                role_label = "Train/Val"
            else:
                role_label = "test_unseen"

            self.folder_table.setItem(row, 0, QTableWidgetItem(fname))
            self.folder_table.item(row, 0).setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

            sp = QSpinBox()
            sp.setRange(1, 100000)
            sp.setValue(int(round(t0)))
            sp.setSuffix(f" {vu}")
            self.folder_table.setCellWidget(row, 1, sp)

            cb = QComboBox()
            for text in ROLE_LABELS:
                cb.addItem(text)
            idx = cb.findText(role_label)
            cb.setCurrentIndex(max(0, idx))
            self.folder_table.setCellWidget(row, 2, cb)

        self.folder_table.resizeColumnsToContents()
        self.apply_btn.setEnabled(True)

    def collect_folder_roles(self) -> dict[str, tuple[float, str]]:
        """Read table → ``{folder: (T_K, 'omit'|'train_val'|'test_unseen')}``."""
        out: dict[str, tuple[float, str]] = {}
        for row in range(self.folder_table.rowCount()):
            name_item = self.folder_table.item(row, 0)
            if name_item is None:
                continue
            folder = name_item.text()
            sp = self.folder_table.cellWidget(row, 1)
            cb = self.folder_table.cellWidget(row, 2)
            if not isinstance(sp, QSpinBox) or not isinstance(cb, QComboBox):
                continue
            t_k = float(sp.value())
            role = ROLE_TO_INTERNAL[cb.currentText()]
            out[folder] = (t_k, role)
        return out

    def var_name(self) -> str:
        return self.var_name_edit.text().strip() or "T"

    def var_unit(self) -> str:
        return self.var_unit_edit.text().strip() or "K"

    def load_io_kwargs(self) -> dict:
        marker = self.marker_edit.text().strip() or None
        return {
            "rng_seed": self.seed_spin.value(),
            "train_frac": self.train_frac_spin.value() / 100.0,
            "integ_prefix": self.integ_edit.text(),
            "header_marker": marker,
            "file_ext": self.ext_edit.text() or "*.txt",
        }

    def update_dataset_summary(self, ds):
        self.summary_label.setText(ds.summary())

    def plot_loaded_spectra(self, ds):
        """Mean spectrum per folder, colour by temperature."""
        if not _HAS_MPL:
            return
        import numpy as np
        from matplotlib.ticker import MaxNLocator

        wl = ds.wavelengths
        fig = self.spec_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        folders = []
        for f in ds.folders:
            if f not in folders:
                folders.append(f)
        temps = [float(np.unique(ds.temperatures[ds.folders == fn])[0]) for fn in folders]
        t_min, t_max = min(temps), max(temps)
        if t_max <= t_min:
            t_max = t_min + 1.0
        norm = plt.Normalize(t_min, t_max)
        cmap = plt.cm.nipy_spectral
        for fn in folders:
            m = ds.folders == fn
            T = float(np.unique(ds.temperatures[m])[0])
            mean_spec = ds.spectra[m].mean(axis=0)
            ax.plot(wl, mean_spec, color=cmap(norm(T)), lw=0.9, alpha=0.9, label=None)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        vn, vu = self.var_name(), self.var_unit()
        cb = fig.colorbar(sm, ax=ax, label=f"{vn} ({vu})")
        cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity / integration time")
        ax.set_title(f"Mean spectrum per folder (colour = {vn} from table)")
        fig.tight_layout()
        self.spec_canvas.draw()

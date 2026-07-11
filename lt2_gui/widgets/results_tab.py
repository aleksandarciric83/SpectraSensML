"""Results tab — leaderboard table, embedded plot thumbnails, export."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .help_widgets import HelpButton


class ResultsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._out_dir: str | None = None
        self._results = None
        self._quantity = {"name": "T", "unit": "K"}
        self._build_ui()

    def set_quantity(self, name: str, unit: str) -> None:
        self._quantity = {"name": name or "T", "unit": unit or "K"}

    def _build_ui(self):
        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)

        # ── Left: leaderboard ─────────────────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("<b>Leaderboard (sorted by Avg(val,test) RMSE)</b>"))
        # Header text is rewritten in `update_results` so the unit matches
        # whatever the user typed in the Data tab (default K).
        self._lb_columns = [
            ("Model", "Model"),
            ("Group", "Group"),
            ("Val RMSE", "Val RMSE"),
            ("Test RMSE", "Test RMSE"),
            ("Avg(val,test) RMSE", "Avg(val,test) RMSE"),
            ("Test Bias", "Test Bias (K)"),
            ("Test Precision σ", "Test Precision σ (K)"),
            ("Test WorstBin RMSE", "Test WorstBin RMSE (K)"),
        ]
        self.leaderboard = QTableWidget(0, len(self._lb_columns))
        self.leaderboard.setHorizontalHeaderLabels([h for _, h in self._lb_columns])
        self.leaderboard.horizontalHeader().setStretchLastSection(True)
        self.leaderboard.setEditTriggers(QTableWidget.NoEditTriggers)
        left_layout.addWidget(self.leaderboard)

        # Export CSV
        export_row = QHBoxLayout()
        self.export_csv_btn = QPushButton("Export metrics_global.csv")
        self.export_csv_btn.setEnabled(False)
        self.export_csv_btn.clicked.connect(self._export_csv)
        export_row.addStretch()
        export_row.addWidget(self.export_csv_btn)
        left_layout.addLayout(export_row)
        splitter.addWidget(left)

        # ── Right: plot viewer ────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("<b>Plots</b>"))

        self.plot_list = QListWidget()
        self.plot_list.setMaximumHeight(120)
        self.plot_list.currentTextChanged.connect(self._show_plot)
        right_layout.addWidget(self.plot_list)

        self.plot_label = QLabel("Run the benchmark to generate plots.")
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setMinimumSize(400, 300)
        scroll = QScrollArea()
        scroll.setWidget(self.plot_label)
        scroll.setWidgetResizable(True)
        right_layout.addWidget(scroll)

        # DPI selector + export
        dpi_row = QHBoxLayout()
        dpi_row.addWidget(QLabel("Export DPI:"))
        self.dpi_combo = QComboBox()
        for dpi in ("150", "300", "600"):
            self.dpi_combo.addItem(dpi)
        self.dpi_combo.setCurrentText("600")
        dpi_row.addWidget(self.dpi_combo)
        dpi_row.addWidget(HelpButton("export_dpi"))

        self.export_all_btn = QPushButton("Export All Plots (PNG)")
        self.export_all_btn.setEnabled(False)
        self.export_all_btn.clicked.connect(self._export_all_plots)
        dpi_row.addStretch()
        dpi_row.addWidget(self.export_all_btn)
        right_layout.addLayout(dpi_row)

        svg_row = QHBoxLayout()
        self.export_svg_btn = QPushButton("Export All Plots as SVG")
        self.export_svg_btn.setEnabled(False)
        self.export_svg_btn.clicked.connect(self._export_all_plots_svg)
        svg_row.addStretch()
        svg_row.addWidget(self.export_svg_btn)
        right_layout.addLayout(svg_row)

        splitter.addWidget(right)
        splitter.setSizes([450, 550])
        layout.addWidget(splitter)

    # ── public API ────────────────────────────────────────────────────────

    def update_results(self, results, out_dir: str):
        self._results = results
        self._out_dir = out_dir
        vu = self._quantity.get("unit", "K") or "K"
        headers = [
            "Model", "Group", "Val RMSE", "Test RMSE", "Avg(val,test) RMSE",
            f"Test Bias ({vu})",
            f"Test Precision σ ({vu})",
            f"Test WorstBin RMSE ({vu})",
        ]
        self.leaderboard.setHorizontalHeaderLabels(headers)
        self._populate_leaderboard(results)
        self._populate_plot_list(out_dir)
        self.export_csv_btn.setEnabled(True)
        self.export_all_btn.setEnabled(True)
        self.export_svg_btn.setEnabled(True)

    def _populate_leaderboard(self, results):
        rows = self._build_rows(results)
        self.leaderboard.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, (key, _hdr) in enumerate(self._lb_columns):
                val = row.get(key, "")
                if isinstance(val, float):
                    text = f"{val:.3f}" if not (val != val) else "—"  # NaN check
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self.leaderboard.setItem(i, j, item)
        self.leaderboard.resizeColumnsToContents()

    def _build_rows(self, results) -> list[dict]:
        import numpy as np

        order = results.sorted_by_mean_val_test_rmse()
        rows: list[dict] = []
        for name in order:
            mr = results.models[name]
            val = mr.metrics.get("val", {})
            test = mr.metrics.get("test_unseen", {})
            rv = float(val.get("RMSE", float("nan")))
            rt = float(test.get("RMSE", float("nan")))
            avg_vt = (
                0.5 * (rv + rt)
                if np.isfinite(rv) and np.isfinite(rt)
                else float("nan")
            )
            rows.append(
                {
                    "Model": name,
                    "Group": mr.group,
                    "Val RMSE": rv,
                    "Test RMSE": rt,
                    "Avg(val,test) RMSE": avg_vt,
                    "Test Bias": float(test.get("Bias", float("nan"))),
                    "Test Precision σ": float(test.get("Precision_sigma", float("nan"))),
                    "Test WorstBin RMSE": float(test.get("WorstBin_RMSE", float("nan"))),
                }
            )
        return rows

    def _populate_plot_list(self, out_dir: str):
        self.plot_list.clear()
        if not out_dir or not os.path.isdir(out_dir):
            return
        for fname in sorted(os.listdir(out_dir)):
            if fname.lower().endswith((".png", ".svg")):
                self.plot_list.addItem(fname)
        if self.plot_list.count() > 0:
            self.plot_list.setCurrentRow(0)

    def _show_plot(self, fname: str):
        if not fname or not self._out_dir:
            return
        path = os.path.join(self._out_dir, fname)
        if not os.path.isfile(path):
            return
        # Decode at preview resolution so 600 DPI exports do not hit Qt's ~256MB pixmap cap.
        reader = QImageReader(path)
        if not reader.canRead():
            self.plot_label.setPixmap(QPixmap())
            self.plot_label.setText(f"Could not read image: {fname}")
            return
        sz = reader.size()
        if sz.isValid() and sz.width() > 0 and sz.height() > 0:
            reader.setScaledSize(sz.scaled(1200, 900, Qt.KeepAspectRatio))
        image = reader.read()
        if image.isNull():
            self.plot_label.setPixmap(QPixmap())
            self.plot_label.setText(
                f"Image too large to preview in the GUI: {fname}\n"
                "Open the file from disk, or export plots at 150–300 DPI."
            )
            return
        pix = QPixmap.fromImage(image)
        pix = pix.scaled(900, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.plot_label.setPixmap(pix)
        self.plot_label.resize(pix.size())

    def _export_csv(self):
        if not self._results or not self._out_dir:
            return
        src = os.path.join(self._out_dir, "metrics_global.csv")
        if os.path.isfile(src):
            import shutil
            dst, _ = QFileDialog.getSaveFileName(self, "Save CSV", "metrics_global.csv",
                                                  "CSV files (*.csv)")
            if dst:
                shutil.copy2(src, dst)

    def _export_all_plots(self):
        self._do_export("png")

    def _export_all_plots_svg(self):
        self._do_export("svg")

    def _do_export(self, fmt: str) -> None:
        if not self._results or not self._out_dir:
            return
        dpi = int(self.dpi_combo.currentText())
        import matplotlib.pyplot as plt
        from lt2_core.plots import export_all
        target = QFileDialog.getExistingDirectory(
            self, f"Select export directory ({fmt.upper()})"
        )
        if not target:
            return
        try:
            export_all(
                self._results,
                self._results.preprocess_result,
                self._results.pca_result,
                target,
                dpi=dpi,
                quantity=self._quantity,
                fmt=fmt,
            )
        finally:
            plt.close("all")
        if target == self._out_dir:
            self._populate_plot_list(target)
        QMessageBox.information(
            self,
            "Export complete",
            f"All plots and per-panel CSVs exported as {fmt.upper()} into:\n{target}",
        )

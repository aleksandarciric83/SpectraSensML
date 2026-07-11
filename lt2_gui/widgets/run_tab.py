"""Run tab — output folder, progress bar, log pane, cancel."""
from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .help_widgets import HelpButton


class RunTab(QWidget):
    run_requested = Signal(str)   # emits output_dir
    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── Output folder ─────────────────────────────────────────────────
        out_group = QGroupBox("Output Directory")
        out_layout = QHBoxLayout(out_group)
        self.out_edit = QLabel("")
        self.out_edit.setWordWrap(True)
        self._set_default_out()
        self.out_browse_btn = QPushButton("Choose…")
        self.out_browse_btn.clicked.connect(self._browse_out)
        out_layout.addWidget(self.out_edit, 1)
        out_layout.addWidget(self.out_browse_btn)
        out_layout.addWidget(HelpButton("out_dir"))
        layout.addWidget(out_group)

        extra_group = QGroupBox("Optional external NN/CNN predictions (NPZ)")
        extra_layout = QVBoxLayout(extra_group)
        extra_layout.addWidget(
            QLabel(
                "Folder with *.npz files (keys: train, val, test_unseen, model_name, group). "
                "Same spectrum order as current dataset. See LT2_pipeline_v2 virtual-model loader."
            )
        )
        extra_row = QHBoxLayout()
        self.extra_preds_edit = QLineEdit("")
        self.extra_preds_edit.setPlaceholderText("Leave empty to skip")
        self.extra_preds_browse = QPushButton("Browse…")
        self.extra_preds_browse.clicked.connect(self._browse_extra_preds)
        extra_row.addWidget(self.extra_preds_edit, 1)
        extra_row.addWidget(self.extra_preds_browse)
        extra_row.addWidget(HelpButton("extra_predictions_dir"))
        extra_layout.addLayout(extra_row)
        layout.addWidget(extra_group)

        # ── Run / Cancel ─────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run Benchmark")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 8px 24px;")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._on_run)

        self.cancel_btn = QPushButton("✕  Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)

        btn_row.addStretch()
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        # ── Progress ──────────────────────────────────────────────────────
        self.progress_status = QLabel("Idle")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_status)
        layout.addWidget(self.progress_bar)

        # ── Log pane ──────────────────────────────────────────────────────
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_pane = QPlainTextEdit()
        self.log_pane.setReadOnly(True)
        self.log_pane.setMaximumBlockCount(5000)
        log_layout.addWidget(self.log_pane)
        layout.addWidget(log_group)

    # ── helpers ───────────────────────────────────────────────────────────

    def _set_default_out(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        from pathlib import Path
        home = Path.home()
        # Prefer Desktop if it exists on this OS (Windows/macOS by default,
        # Linux distros that follow xdg-user-dirs); otherwise fall back to
        # the user's home directory so the path is always valid.
        desktop = home / "Desktop"
        base = desktop if desktop.is_dir() else home
        default = str(base / f"LT2_results_{ts}")
        self.out_edit.setText(default)

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Choose output directory")
        if path:
            self.out_edit.setText(path)

    def _browse_extra_preds(self):
        path = QFileDialog.getExistingDirectory(self, "Folder with preds_*.npz")
        if path:
            self.extra_preds_edit.setText(path)

    def extra_predictions_dir(self) -> str | None:
        s = self.extra_preds_edit.text().strip()
        return s if s else None

    def _on_run(self):
        out = self.out_edit.text().strip()
        if not out:
            self._set_default_out()
            out = self.out_edit.text()
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_status.setText("Running benchmark …")
        self.run_requested.emit(out)

    def _on_cancel(self):
        self.cancel_requested.emit()
        self.cancel_btn.setEnabled(False)

    # ── public API ────────────────────────────────────────────────────────

    def enable_run(self, state: bool):
        self.run_btn.setEnabled(state)

    def append_log(self, msg: str):
        self.log_pane.appendPlainText(msg)
        self.log_pane.ensureCursorVisible()

    def set_progress(self, done: int, total: int):
        """File-load progress (spectra folders)."""
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(done)
            self.progress_bar.setFormat(f"{done} / {total}")
            self.progress_status.setText("Loading spectra …")
        else:
            self.progress_bar.setMaximum(1)
            self.progress_bar.setValue(0)

    def reset_progress_display(self):
        """Reset bar + label after load completes or errors."""
        self.progress_bar.setMaximum(1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        self.progress_status.setText("Idle")

    def set_benchmark_progress(self, done: int, total: int):
        """Benchmark: first two steps are preprocess and PCA, then each model."""
        if total <= 0:
            return
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(min(done, total))
        pct = (100 * done) // total if total else 0
        self.progress_bar.setFormat(f"{done} / {total}  ({pct}%)")
        if done == 0:
            self.progress_status.setText("Queued — preprocessing will start …")
        elif done == 1:
            self.progress_status.setText("Preprocessing spectra …")
        elif done == 2:
            self.progress_status.setText("PCA (fit on train) …")
        elif done < total:
            n_models = max(total - 2, 1)
            cur = min(done - 2, n_models)
            self.progress_status.setText(f"Fitting benchmark models … ({cur}/{n_models})")
        else:
            self.progress_status.setText("Saving results …")

    def on_finished(self):
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.progress_status.setText("Finished.")
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def on_error(self, msg: str):
        self.append_log(f"\n[ERROR]\n{msg}")
        self.progress_status.setText("Stopped with error — see log.")
        self.reset_progress_display()
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

"""main_window.py — SpectraSensML PySide6 shell."""
from __future__ import annotations

from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtWidgets import (
    QLabel, QMainWindow, QStatusBar, QTabWidget, QVBoxLayout, QWidget,
)

from lt2_core.dataset import build_dataset_from_roles, default_folder_roles_all_train_val

from .__version__ import app_title, citation_text
from .widgets.about_tab import AboutTab
from .widgets.data_tab import DataTab
from .widgets.help_tab import HelpTab
from .widgets.models_tab import ModelsTab
from .widgets.pca_tab import PCATab
from .widgets.preprocess_tab import PreprocessTab
from .widgets.results_tab import ResultsTab
from .widgets.run_tab import RunTab
from .workers import LoadWorker, RunWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(app_title())
        self.resize(1280, 860)

        self._dataset = None
        self._prep_result = None
        self._results = None
        self._run_worker: RunWorker | None = None

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        # Citation notice bar
        citation_bar = QLabel(
            f"📄  <b>Citation required:</b> {citation_text()} — "
            "See the <b>About</b> tab for details."
        )
        citation_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        citation_bar.setWordWrap(True)
        citation_bar.setStyleSheet(
            "QLabel {"
            "  background-color: #fff8e1;"
            "  color: #5d4037;"
            "  border-bottom: 1px solid #ffe082;"
            "  padding: 5px 12px;"
            "  font-size: 11px;"
            "}"
        )

        self.tabs = QTabWidget()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(citation_bar)
        layout.addWidget(self.tabs)
        self.setCentralWidget(central)
        self.data_tab = DataTab()
        self.preproc_tab = PreprocessTab()
        self.pca_tab = PCATab()
        self.models_tab = ModelsTab()
        self.run_tab = RunTab()
        self.results_tab = ResultsTab()
        self.help_tab = HelpTab()
        self.about_tab = AboutTab()
        self.tabs.addTab(self.data_tab, "1 · Data")
        self.tabs.addTab(self.preproc_tab, "2 · Pre-process")
        self.tabs.addTab(self.pca_tab, "3 · PCA")
        self.tabs.addTab(self.models_tab, "4 · Models")
        self.tabs.addTab(self.run_tab, "5 · Run")
        self.tabs.addTab(self.results_tab, "6 · Results")
        self.tabs.addTab(self.help_tab, "7 · Help")
        self.tabs.addTab(self.about_tab, "8 · About")
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — load spectra, set folder roles, Apply, then Pre-process → PCA → Run.")

    def _connect_signals(self):
        self.data_tab.load_requested.connect(self._on_load_requested)
        self.data_tab.apply_roles_requested.connect(self._rebuild_dataset_from_table)
        self.preproc_tab.preprocess_run_requested.connect(self._on_preprocess_run)
        self.pca_tab.pca_run_requested.connect(self._on_pca_run)
        self.pca_tab.n_components_changed.connect(self.models_tab.update_k)
        self.run_tab.run_requested.connect(self._on_run_requested)
        self.run_tab.cancel_requested.connect(self._on_cancel)
        self.data_tab.var_name_edit.editingFinished.connect(self._sync_quantity)
        self.data_tab.var_unit_edit.editingFinished.connect(self._sync_quantity)

    def _sync_quantity(self) -> None:
        """Broadcast variable name/unit from the Data tab to every other tab."""
        vn = self.data_tab.var_name()
        vu = self.data_tab.var_unit()
        self.preproc_tab.set_quantity(vn, vu)
        self.pca_tab.set_quantity(vn, vu)
        self.results_tab.set_quantity(vn, vu)

    def _on_load_requested(self, root: str, kwargs: dict):
        self.status.showMessage(f"Loading {root} …")
        self.data_tab.load_btn.setEnabled(False)
        self.run_tab.append_log(f"Loading from {root} …")
        worker = LoadWorker(root, kwargs)
        worker.signals.log.connect(self.run_tab.append_log)
        worker.signals.progress.connect(self.run_tab.set_progress)
        worker.signals.finished.connect(self._on_dataset_loaded)
        worker.signals.error.connect(self._on_load_error)
        QThreadPool.globalInstance().start(worker)

    def _on_dataset_loaded(self, ds):
        self._dataset = ds
        self._prep_result = None
        self.data_tab.load_btn.setEnabled(True)
        self._sync_quantity()
        self.data_tab.set_table_from_dataset(ds)
        self.data_tab.update_dataset_summary(ds)
        self.data_tab.plot_loaded_spectra(ds)
        self.run_tab.enable_run(True)
        self.run_tab.append_log(ds.summary())
        self.models_tab.update_k(self.pca_tab.n_components())
        self.status.showMessage(
            f"Loaded {ds.spectra.shape[0]} spectra — adjust folder roles if needed, then Apply."
        )
        self.tabs.setCurrentIndex(0)
        self.run_tab.reset_progress_display()

    def _on_load_error(self, msg: str):
        self.data_tab.load_btn.setEnabled(True)
        self.run_tab.append_log(f"[LOAD ERROR] {msg}")
        self.run_tab.reset_progress_display()
        self.status.showMessage("Load failed — see Run tab log.")

    def _rebuild_dataset_from_table(self):
        if not self.data_tab.folder_edit.text():
            self.run_tab.append_log("No root folder selected.")
            return
        root = self.data_tab.folder_edit.text()
        roles = self.data_tab.collect_folder_roles()
        if not roles:
            self.run_tab.append_log("Folder table is empty.")
            return
        io = self.data_tab.load_io_kwargs()
        try:
            ds = build_dataset_from_roles(root, roles, **io)
        except Exception as e:
            self.run_tab.append_log(f"[Apply ERROR] {e}")
            self.status.showMessage("Apply failed — see log.")
            return
        self._dataset = ds
        self._prep_result = None
        self.data_tab.set_table_from_dataset(ds)
        self.data_tab.update_dataset_summary(ds)
        self.data_tab.plot_loaded_spectra(ds)
        self.run_tab.append_log(ds.summary())
        self.status.showMessage("Dataset updated from folder table.")

    def _on_preprocess_run(self):
        if self._dataset is None:
            self.run_tab.append_log("Load spectra and Apply folder roles first.")
            return
        from lt2_core.preprocess import preprocess

        wl_min, wl_max = self.preproc_tab.wl_range()
        prep = preprocess(
            self._dataset.spectra,
            self._dataset.wavelengths,
            wl_min,
            wl_max,
            self.preproc_tab.normalization(),
            self.preproc_tab.bg_subtract(),
        )
        self._prep_result = prep
        self.preproc_tab.show_preprocessed(
            prep, self._dataset.temperatures, self._dataset.folders
        )
        self.run_tab.append_log(
            f"[Pre-process] channels={prep.wavelengths_crop.size}, norm={prep.norm.value}"
        )
        self.status.showMessage("Pre-processing finished.")
        self.tabs.setCurrentIndex(1)

    def _on_pca_run(self):
        if self._dataset is None or self._prep_result is None:
            self.run_tab.append_log("Load data + Apply, then run Pre-process before PCA.")
            return
        from lt2_core.pca_analysis import fit_pca

        tr = int(self._dataset.train_mask.sum())
        if tr < 2:
            self.run_tab.append_log("Need at least 2 training spectra for PCA.")
            return
        n_ch = self._prep_result.spectra_norm.shape[1]
        k_req = self.pca_tab.n_components()
        k_max = min(50, n_ch, max(1, tr - 1))
        k = max(1, min(k_req, k_max))
        if k != k_req:
            self.run_tab.append_log(f"[PCA] Clamped k from {k_req} to {k_max} (rank / train count).")
            self.pca_tab.k_spin.blockSignals(True)
            self.pca_tab.k_spin.setValue(k)
            self.pca_tab.k_spin.blockSignals(False)
        pca_res = fit_pca(
            self._prep_result.spectra_norm,
            self._dataset.train_mask,
            k,
            self.data_tab.seed_spin.value(),
        )
        self.pca_tab.update_full_pca(
            pca_res, self._dataset, self._prep_result.wavelengths_crop
        )
        self.models_tab.update_k(self.pca_tab.n_components())
        self.run_tab.append_log(f"[PCA] k={k}  explained Σ={pca_res.explained_variance_ratio.sum():.4f}")
        self.status.showMessage("PCA finished — set k for benchmark if needed.")
        self.tabs.setCurrentIndex(2)

    def _on_run_requested(self, out_dir: str):
        if self._dataset is None:
            self.run_tab.on_error("No dataset. Load spectra and click Apply.")
            return
        from lt2_core.benchmark import BenchmarkConfig

        cfg = BenchmarkConfig(
            wl_min=self.preproc_tab.wl_range()[0],
            wl_max=self.preproc_tab.wl_range()[1],
            normalization=self.preproc_tab.normalization(),
            bg_subtract=self.preproc_tab.bg_subtract(),
            n_pca_components=self.pca_tab.n_components(),
            rng_seed=self.data_tab.seed_spin.value(),
            models=self.models_tab.model_flags(),
            lir_cfg=self.preproc_tab.lir_config(),
            extra_predictions_dir=self.run_tab.extra_predictions_dir(),
            var_name=self.data_tab.var_name(),
            var_unit=self.data_tab.var_unit(),
        )
        self._sync_quantity()
        self.status.showMessage("Running benchmark …")
        self.run_tab.progress_bar.setValue(0)
        self.run_tab.append_log(f"\n{'='*60}")
        self.run_tab.append_log(f"Starting benchmark → {out_dir}")
        self.run_tab.append_log(
            f"  k={cfg.n_pca_components}, norm={str(cfg.normalization)}, "
            f"train={int(self._dataset.train_mask.sum())} val={int(self._dataset.val_mask.sum())}"
        )
        worker = RunWorker(self._dataset, cfg, out_dir)
        self._run_worker = worker
        worker.signals.log.connect(self.run_tab.append_log)
        worker.signals.progress.connect(self.run_tab.set_benchmark_progress)
        worker.signals.finished.connect(self._on_run_finished)
        worker.signals.error.connect(self._on_run_error)
        QThreadPool.globalInstance().start(worker)

    def _on_run_finished(self, results):
        self._results = results
        out_dir = self._run_worker.out_dir if self._run_worker else ""
        self.run_tab.on_finished()
        self.results_tab.update_results(results, out_dir)
        self.tabs.setCurrentIndex(5)
        self.status.showMessage(
            f"Done — {len(results.models)} models. "
            f"Best avg(val,test) RMSE: {results.sorted_by_mean_val_test_rmse()[0]}"
            if results.models
            else "Done."
        )

    def _on_run_error(self, msg: str):
        self.run_tab.on_error(msg)
        self.status.showMessage("Benchmark failed — see log.")

    def _on_cancel(self):
        if self._run_worker is not None:
            self._run_worker.cancel()
        self.status.showMessage("Cancellation requested …")

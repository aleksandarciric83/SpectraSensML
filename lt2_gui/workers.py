"""workers.py — background loaders for spectra and benchmark."""
from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

if TYPE_CHECKING:
    pass


class WorkerSignals(QObject):
    progress = Signal(int, int)
    log = Signal(str)
    finished = Signal(object)
    error = Signal(str)


class LoadWorker(QRunnable):
    """Load all spectral folders with default role Train/Val (no digit-based split)."""

    def __init__(self, root: str, kwargs: dict):
        super().__init__()
        self.root = root
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self._cancel = [False]

    def cancel(self):
        self._cancel[0] = True

    @Slot()
    def run(self):
        from lt2_core.dataset import build_dataset_from_roles, default_folder_roles_all_train_val

        try:
            def _progress(done, total):
                if self._cancel[0]:
                    raise InterruptedError("Cancelled by user")
                self.signals.progress.emit(done, total)

            self.signals.log.emit(f"Loading spectra from {self.root} ...")
            roles = default_folder_roles_all_train_val(self.root)
            ds = build_dataset_from_roles(
                self.root,
                roles,
                train_frac=self.kwargs["train_frac"],
                rng_seed=self.kwargs["rng_seed"],
                integ_prefix=self.kwargs.get("integ_prefix", "Integration Time (sec):"),
                header_marker=self.kwargs.get("header_marker"),
                file_ext=self.kwargs.get("file_ext", "*.txt"),
                progress_callback=_progress,
            )
            self.signals.log.emit(ds.summary())
            self.signals.finished.emit(ds)
        except InterruptedError:
            self.signals.error.emit("Load cancelled.")
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class RunWorker(QRunnable):
    def __init__(self, ds, cfg, out_dir: str):
        super().__init__()
        self.ds = ds
        self.cfg = cfg
        self.out_dir = out_dir
        self.signals = WorkerSignals()
        self._cancel = [False]

    def cancel(self):
        self._cancel[0] = True

    @Slot()
    def run(self):
        from lt2_core.benchmark import merge_external_predictions_from_dir, run_benchmark
        from lt2_core.plots import export_all
        import datetime
        import json
        import os
        import platform

        import numpy as np
        import pandas as pd
        import sklearn

        try:
            os.makedirs(self.out_dir, exist_ok=True)
            self.signals.log.emit("[0] Starting benchmark (this may take several minutes) …")

            results = run_benchmark(
                self.ds,
                self.cfg,
                log=lambda msg: self.signals.log.emit(msg),
                progress=lambda d, t: self.signals.progress.emit(d, t),
                cancel_flag=self._cancel,
            )

            if self._cancel[0]:
                self.signals.error.emit("Run cancelled.")
                return

            merge_external_predictions_from_dir(
                results,
                self.ds,
                self.cfg.extra_predictions_dir,
                log=lambda msg: self.signals.log.emit(msg),
            )

            self.signals.log.emit("\nSaving CSVs …")
            from lt2_core.metrics import global_csv_columns, per_T_csv_columns

            global_rows: list[dict] = []
            for mname, mr in results.models.items():
                for sname, metrics in mr.metrics.items():
                    row = {"Model": mname, "Group": mr.group, "Set": sname}
                    row.update({k: metrics.get(k, None) for k in metrics.keys()})
                    global_rows.append(row)
            global_df = pd.DataFrame(global_rows)
            cols = [c for c in global_csv_columns() if c in global_df.columns]
            extra = [c for c in global_df.columns if c not in cols]
            global_df = global_df[cols + extra]
            global_df.to_csv(
                os.path.join(self.out_dir, "metrics_global.csv"),
                index=False,
                float_format="%.6g",
            )

            per_T_rows: list[dict] = []
            for mname, mr in results.models.items():
                for sname in ("train", "val", "test_unseen"):
                    for row in mr.per_T.get(sname, []) or []:
                        per_T_rows.append({
                            "Model": mname,
                            "Group": mr.group,
                            "Set": sname,
                            **row,
                        })
            per_T_df = pd.DataFrame(per_T_rows)
            cols = [c for c in per_T_csv_columns() if c in per_T_df.columns]
            extra = [c for c in per_T_df.columns if c not in cols]
            per_T_df = per_T_df[cols + extra]
            per_T_df.to_csv(
                os.path.join(self.out_dir, "metrics_per_temperature.csv"),
                index=False,
                float_format="%.6g",
            )

            self.signals.log.emit("Saving plots …")
            quantity = {
                "name": getattr(self.cfg, "var_name", "T") or "T",
                "unit": getattr(self.cfg, "var_unit", "K") or "K",
            }
            export_all(
                results,
                results.preprocess_result,
                results.pca_result,
                self.out_dir,
                quantity=quantity,
            )

            from lt2_gui.__version__ import APP_NAME, APP_VERSION
            info = {
                "out_dir": self.out_dir,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "platform": platform.platform(),
                "sklearn": sklearn.__version__,
                "numpy": np.__version__,
                "n_models": len(results.models),
                "winner": results.sorted_by_mean_val_test_rmse()[0] if results.models else None,
                "app": f"{APP_NAME} v{APP_VERSION}",
                "variable_name": quantity["name"],
                "variable_unit": quantity["unit"],
            }
            with open(os.path.join(self.out_dir, "run_info.json"), "w") as f:
                json.dump(info, f, indent=2)

            self.signals.log.emit(f"\n[Done] Results saved to {self.out_dir}")
            self.signals.finished.emit(results)

        except InterruptedError:
            self.signals.error.emit("Run cancelled.")
        except Exception:
            self.signals.error.emit(traceback.format_exc())

"""In-benchmark ANN (MLP) and CNN models for lt2_gui / run_benchmark.

* **ANN (PCA)** — :class:`~sklearn.neural_network.MLPRegressor` with ``solver='lbfgs'``
  on scaled PCA features (same *k* PCs as the rest of the benchmark).
* **ANN (SNV)** — MLP on flattened SNV spectra (same tensor as PLS).
* **CNN (1D)** — small Conv1d + global average pool in PyTorch (CPU).  If PyTorch
  is not installed, the benchmark logs a skip and does not register a model.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _val_rmse(y_val: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_val, y_pred)))


def register_ann_pca_mlp(
    results,
    cfg,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    reg: Callable,
    log: Callable[[str], None],
    cancelled: Callable[[], bool],
) -> None:
    """Val-pick best lbfgs MLP on first *k* PCA components."""
    k = min(cfg.n_pca_components, X_train.shape[1])
    sc = StandardScaler().fit(X_train[:, :k])
    Xt = sc.transform(X_train[:, :k])
    Xv = sc.transform(X_val[:, :k])
    Xe = sc.transform(X_test[:, :k])

    best_v, best_m, best_arch = np.inf, None, cfg.ann_pca_archs[0]
    for arch in cfg.ann_pca_archs:
        if cancelled():
            return
        m = MLPRegressor(
            hidden_layer_sizes=arch,
            activation="relu",
            solver="lbfgs",
            alpha=1e-4,
            max_iter=2500,
            random_state=cfg.rng_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            m.fit(Xt, y_train)
        rv = _val_rmse(y_val, m.predict(Xv))
        if rv < best_v:
            best_v, best_m, best_arch = rv, m, arch

    if best_m is None:
        return
    name = f"ANN (MLP on {k} PCs, lbfgs, {best_arch})"
    reg(
        results,
        name,
        "E",
        y_train,
        y_val,
        y_test,
        best_m.predict(Xt),
        best_m.predict(Xv),
        best_m.predict(Xe),
        log,
    )


def register_ann_snv_mlp(
    results,
    cfg,
    S_train: np.ndarray,
    S_val: np.ndarray,
    S_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    reg: Callable,
    log: Callable[[str], None],
    cancelled: Callable[[], bool],
) -> None:
    """Val-pick best Adam MLP on scaled full SNV spectra."""
    sc = StandardScaler().fit(S_train)
    St = sc.transform(S_train)
    Sv = sc.transform(S_val)
    Se = sc.transform(S_test)

    best_v, best_m, best_arch = np.inf, None, cfg.ann_snv_archs[0]
    for arch in cfg.ann_snv_archs:
        if cancelled():
            return
        m = MLPRegressor(
            hidden_layer_sizes=arch,
            activation="relu",
            solver="adam",
            max_iter=2500,
            random_state=cfg.rng_seed,
            learning_rate_init=1e-3,
            early_stopping=True,
            n_iter_no_change=30,
            validation_fraction=0.12,
        )
        m.fit(St, y_train)
        rv = _val_rmse(y_val, m.predict(Sv))
        if rv < best_v:
            best_v, best_m, best_arch = rv, m, arch

    if best_m is None:
        return
    name = f"ANN (MLP on SNV spectrum, {best_arch})"
    reg(
        results,
        name,
        "E",
        y_train,
        y_val,
        y_test,
        best_m.predict(St),
        best_m.predict(Sv),
        best_m.predict(Se),
        log,
    )


def register_cnn_snv_1d(
    results,
    cfg,
    S_train: np.ndarray,
    S_val: np.ndarray,
    S_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    reg: Callable,
    log: Callable[[str], None],
    cancelled: Callable[[], bool],
) -> None:
    """Small 1D CNN on SNV spectra (channels as length)."""
    try:
        import torch
        from torch import nn
    except ImportError:
        log("  [skip] CNN (1D Conv on SNV): PyTorch not installed")
        return

    try:
        # Single-thread intra-op avoids oversubscription and rare libomp/MKL deadlocks when
        # numpy, sklearn, and torch run in the same QThreadPool worker.
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

    device = torch.device("cpu")
    torch.manual_seed(cfg.rng_seed)
    np.random.seed(cfg.rng_seed)

    n_ch = S_train.shape[1]
    n_tr, n_va, n_te = S_train.shape[0], S_val.shape[0], S_test.shape[0]
    log(
        f"  [E] CNN (1D): {n_tr} train / {n_va} val / {n_te} test samples, "
        f"{n_ch} spectral points — preparing tensors …"
    )

    class SmallCNN1d(nn.Module):
        def __init__(self, n_in: int, h: int):
            super().__init__()
            c1, c2 = 16, 32
            self.conv1 = nn.Conv1d(1, c1, kernel_size=7, padding=3)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, padding=2)
            self.fc1 = nn.Linear(c2, h)
            self.fc2 = nn.Linear(h, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, n_in)
            z = x.unsqueeze(1)
            z = torch.relu(self.conv1(z))
            z = self.pool(z)
            z = torch.relu(self.conv2(z))
            z = z.mean(dim=2)
            z = torch.relu(self.fc1(z))
            return self.fc2(z).squeeze(-1)

    def to_tensor(a: np.ndarray) -> torch.Tensor:
        a32 = np.ascontiguousarray(a, dtype=np.float32)
        t = torch.from_numpy(a32)
        if not t.is_contiguous():
            t = t.contiguous()
        return t.to(device)

    Xt = to_tensor(S_train)
    Xv = to_tensor(S_val)
    yt = to_tensor(y_train.reshape(-1))

    model = SmallCNN1d(n_ch, cfg.cnn_fc_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    n = Xt.shape[0]
    bs = min(64, max(8, n // 4))
    n_batches = (n + bs - 1) // bs
    best_state = None
    best_val = np.inf
    stall = 0
    max_ep = cfg.cnn_max_epochs
    log(
        f"  [E] CNN training (CPU, up to {max_ep} epochs, early stop "
        f"{cfg.cnn_patience_epochs}, batch {bs}) …"
    )

    for ep in range(max_ep):
        if cancelled():
            return
        if ep == 0:
            log(
                f"      CNN epoch 1/{max_ep}: {n_batches} batches (size {bs}); "
                f"watch for batch milestones, then val RMSE …"
            )
        model.train()
        perm = torch.randperm(n, device=device)
        milestones = {0, n_batches // 4, n_batches // 2, (3 * n_batches) // 4, n_batches - 1}
        for bi, i in enumerate(range(0, n, bs)):
            if bi % 8 == 0 and cancelled():
                return
            idx = perm[i : i + bs]
            if ep == 0 and bi == 0:
                log("      CNN epoch 1: first batch (forward + backward) …")
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            opt.step()
            if ep == 0 and bi in milestones:
                log(f"      CNN epoch 1: batch {bi + 1}/{n_batches} done (loss={loss.item():.4f}).")

        if ep == 0:
            log(f"      CNN epoch 1/{max_ep}: training batches finished; validation forward …")
        model.eval()
        with torch.no_grad():
            pv = model(Xv).cpu().numpy()
        val_rmse = _val_rmse(y_val, pv)
        if ep == 0 or (ep + 1) % 5 == 0 or ep == max_ep - 1:
            log(f"      CNN epoch {ep + 1}/{max_ep}: val RMSE = {val_rmse:.4f} K")
        if val_rmse < best_val - 1e-5:
            best_val = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= cfg.cnn_patience_epochs:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pt = model(Xt).cpu().numpy()
        pv = model(Xv).cpu().numpy()
        pe = model(to_tensor(S_test)).cpu().numpy()

    name = f"CNN (1D Conv on SNV, ch={n_ch}, h={cfg.cnn_fc_hidden})"
    reg(results, name, "E", y_train, y_val, y_test, pt, pv, pe, log)

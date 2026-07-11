"""pca_analysis.py
Thin wrapper around sklearn PCA that enforces the 'fit on train only' rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class PCAResult:
    pca: PCA
    X_all: np.ndarray              # (N, k) projected coordinates
    explained_variance_ratio: np.ndarray  # (k,)
    n_components: int

    def explained_cumulative(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio)

    def transform(self, X: np.ndarray, k: int | None = None) -> np.ndarray:
        """Project new data; optionally restrict to first k components."""
        out = self.pca.transform(X)
        if k is not None:
            out = out[:, :k]
        return out


def fit_pca(
    spectra: np.ndarray,
    train_mask: np.ndarray,
    n_components: int = 3,
    rng_seed: int = 42,
) -> PCAResult:
    """Fit PCA on training spectra only, then transform all spectra.

    Parameters
    ----------
    spectra : (N, n_ch) normalised spectra (all splits)
    train_mask : (N,) boolean — True for training samples
    n_components : number of PCA components to keep
    rng_seed : random state for the PCA solver
    """
    pca = PCA(n_components=n_components, random_state=rng_seed)
    pca.fit(spectra[train_mask])
    X_all = pca.transform(spectra)
    return PCAResult(
        pca=pca,
        X_all=X_all,
        explained_variance_ratio=pca.explained_variance_ratio_,
        n_components=n_components,
    )


def transform_pca(pca_result: PCAResult, spectra: np.ndarray, k: int | None = None) -> np.ndarray:
    """Transform new spectra using an already-fitted PCA."""
    return pca_result.transform(spectra, k=k)

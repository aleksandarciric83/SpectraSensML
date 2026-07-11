"""Tests for pca_analysis.py."""
import numpy as np
import pytest

from lt2_core.pca_analysis import fit_pca, transform_pca


def _make_data(n=100, n_ch=50, k=3, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.random((n, k))  # latent
    W = rng.random((k, n_ch))  # mixing
    S = T @ W + rng.normal(0, 0.01, (n, n_ch))
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:70] = True
    return S, train_mask


def test_fit_pca_returns_correct_shape():
    S, train_mask = _make_data()
    res = fit_pca(S, train_mask, n_components=3)
    assert res.X_all.shape == (100, 3)
    assert res.explained_variance_ratio.shape == (3,)


def test_fit_pca_train_only():
    S, train_mask = _make_data()
    res = fit_pca(S, train_mask, n_components=3)
    # PCA fitted on train → projection should align
    from sklearn.decomposition import PCA
    pca2 = PCA(n_components=3).fit(S[train_mask])
    # components may differ in sign; check absolute cosine similarity
    for i in range(3):
        cos = np.abs(np.dot(res.pca.components_[i], pca2.components_[i]))
        cos /= (np.linalg.norm(res.pca.components_[i]) * np.linalg.norm(pca2.components_[i]))
        assert cos > 0.99


def test_explained_variance_cumulative():
    S, train_mask = _make_data()
    res = fit_pca(S, train_mask, n_components=3)
    cumvar = res.explained_cumulative()
    assert cumvar[-1] <= 1.0 + 1e-9
    assert cumvar[-1] > 0.5  # generated from 3 latent factors


def test_transform_pca_k_restriction():
    S, train_mask = _make_data()
    res = fit_pca(S, train_mask, n_components=3)
    X2 = transform_pca(res, S, k=2)
    assert X2.shape == (100, 2)

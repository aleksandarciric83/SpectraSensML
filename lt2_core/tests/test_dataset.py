"""Tests for dataset.py — uses a synthetic temp-folder tree."""
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

from lt2_core.dataset import (
    parse_folder_name,
    is_training_folder,
    load_dataset,
    default_folder_roles_all_train_val,
)


# ── unit tests ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,expected_K", [
    ("p100", 373.0),
    ("p0", 273.0),
    ("m10", 263.0),
])
def test_parse_folder_name(name, expected_K):
    assert parse_folder_name(name) == pytest.approx(expected_K)


@pytest.mark.parametrize("name,expected", [
    ("p27", True),
    ("p30", False),
    ("m13", True),
    ("m10", False),
])
def test_is_training_folder(name, expected):
    assert is_training_folder(name) == expected


# ── synthetic dataset fixture ─────────────────────────────────────────────

_SPECTRUM_TEMPLATE = """\
Integration Time (sec): 1.0
>>>>>Begin Spectral Data<<<<<
{data}
"""


def _write_spectrum(path: Path, wl_start=900.0, n_ch=10, seed=0):
    rng = np.random.default_rng(seed)
    wl = np.linspace(wl_start, wl_start + n_ch - 1, n_ch)
    inten = rng.random(n_ch) + 1.0
    data = "\n".join(f"{w:.2f}\t{v:.4f}" for w, v in zip(wl, inten))
    path.write_text(textwrap.dedent(_SPECTRUM_TEMPLATE.format(data=data)))


@pytest.fixture()
def synthetic_root(tmp_path):
    """Create a mini folder tree: 2 training folders + 1 test folder, 4 spectra each."""
    n_spectra = 4
    folders = [
        ("p27", True),   # training (last digit 7)
        ("p37", True),   # training
        ("p30", False),  # test (last digit 0)
    ]
    for fname, _ in folders:
        d = tmp_path / fname
        d.mkdir()
        for i in range(n_spectra):
            _write_spectrum(d / f"spec_{i:03d}.txt", seed=i)
    return tmp_path


def test_default_folder_roles_train_val_if_kelvin_ends_in_zero(tmp_path):
    (tmp_path / "p227").mkdir()
    (tmp_path / "p125").mkdir()
    roles = default_folder_roles_all_train_val(tmp_path)
    assert roles["p227"][1] == "train_val"
    assert roles["p125"][1] == "test_unseen"
    assert int(roles["p227"][0]) == 500
    assert int(roles["p125"][0]) == 398


def test_load_dataset_shape(synthetic_root):
    ds = load_dataset(synthetic_root, rng_seed=0)
    # 2 train folders × 4 = 8, 1 test folder × 4 = 4 → 12 total
    assert ds.spectra.shape[0] == 12
    # labels
    assert set(ds.labels) <= {"train", "val", "test_unseen"}
    assert (ds.train_mask | ds.val_mask | ds.test_mask).all()


def test_train_val_split_fractions(synthetic_root):
    ds = load_dataset(synthetic_root, rng_seed=0, train_frac=0.75)
    tr = int(ds.train_mask.sum())
    va = int(ds.val_mask.sum())
    # 2 folders × 4 spectra = 8 training spectra
    assert tr + va == 8
    assert tr == 6  # floor(0.75 × 4) × 2 = 3 × 2


def test_test_only_mode(synthetic_root):
    ds = load_dataset(synthetic_root, test_only=True)
    assert not ds.train_mask.any()
    assert not ds.val_mask.any()
    assert ds.test_mask.all()
    assert ds.spectra.shape[0] == 4   # only the p30 folder


def test_temperatures_correct(synthetic_root):
    ds = load_dataset(synthetic_root)
    T_train_folders = set(ds.temperatures[ds.train_mask | ds.val_mask])
    T_test = set(ds.temperatures[ds.test_mask])
    assert parse_folder_name("p27") in T_train_folders
    assert parse_folder_name("p37") in T_train_folders
    assert parse_folder_name("p30") in T_test


def test_summary_str(synthetic_root):
    ds = load_dataset(synthetic_root)
    s = ds.summary()
    assert "Dataset" in s
    assert "train=" in s

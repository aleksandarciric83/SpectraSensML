"""dataset.py
Load spectrum files into a Dataset dataclass.

Folder naming conventions supported:

* ``pXXX`` / ``mXXX`` — Celsius magnitude with sign (p=positive, m=negative).
  Temperature in Kelvin = ±XXX + 273.15.  Example: ``m100`` → 173.15 K.
* Plain integer / float — temperature already in Kelvin.
  Example: ``300`` → 300 K,  ``473`` → 473 K.

**Role assignment** (train/val vs ``test_unseen`` vs omit) is explicit via
:func:`build_dataset_from_roles` — the GUI drives this.
:func:`default_folder_roles_all_train_val` assigns ``train_val`` when rounded
integer T(K) ends in ``0``, otherwise ``test_unseen``.
"""
from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .spectrum_io import read_spectrum_file

Role = Literal["omit", "train_val", "test_unseen"]


def parse_folder_name(name: str) -> float:
    """Return temperature in Kelvin from a folder name.

    Supported formats:
    * ``pXXX`` / ``mXXX`` — Celsius with sign; T(K) = ±XXX + 273.15
    * Pure integer / float string — already in Kelvin; returned as-is.
    """
    if name and name[0] in ("p", "m"):
        sign = 1 if name[0] == "p" else -1
        return float(sign * float(name[1:]) + 273.15)
    # plain numeric — Kelvin directly
    return float(name)


def is_training_folder(name: str) -> bool:
    """Legacy helper: digit-based training folder (optional external scripts)."""
    if name and name[0] in ("p", "m"):
        last = int(name[1:]) % 10
        return last == 7 if name[0] == "p" else last == 3
    # Kelvin folders: use last digit 0 as training heuristic
    try:
        return int(float(name)) % 10 == 0
    except ValueError:
        return False


def _is_temp_folder(name: str) -> bool:
    """True if *name* encodes a temperature (p/m-Celsius or plain-Kelvin)."""
    if not name:
        return False
    if name[0] in ("p", "m"):
        try:
            float(name[1:])
            return True
        except ValueError:
            return False
    # plain numeric (integer or float, positive)
    try:
        val = float(name)
        return val > 0
    except ValueError:
        return False


def discover_spectral_folders(root: str | Path) -> list[str]:
    """Sorted list of temperature-encoded child folder names under *root*.

    Accepts both ``p*/m*`` (Celsius-offset) and plain numeric (Kelvin) names.
    Numeric Kelvin folders are sorted numerically; p/m folders by parsed T(K).
    All are returned sorted by their parsed temperature value.
    """
    root = Path(root)
    folders = [
        d for d in os.listdir(root)
        if (root / d).is_dir() and _is_temp_folder(d)
    ]
    return sorted(folders, key=parse_folder_name)


@dataclass
class Dataset:
    """Container for spectra, wavelengths, temperatures, split labels."""

    spectra: np.ndarray
    wavelengths: np.ndarray
    temperatures: np.ndarray
    labels: np.ndarray
    folders: np.ndarray
    file_paths: list[str] = field(default_factory=list)
    integ_times: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def train_mask(self) -> np.ndarray:
        return self.labels == "train"

    @property
    def val_mask(self) -> np.ndarray:
        return self.labels == "val"

    @property
    def test_mask(self) -> np.ndarray:
        return self.labels == "test_unseen"

    @property
    def train_temps_unique(self) -> np.ndarray:
        return np.unique(self.temperatures[self.train_mask | self.val_mask])

    @property
    def test_temps_unique(self) -> np.ndarray:
        return np.unique(self.temperatures[self.test_mask])

    def summary(self) -> str:
        n_tr = int(self.train_mask.sum())
        n_va = int(self.val_mask.sum())
        n_te = int(self.test_mask.sum())
        n_ch = self.spectra.shape[1]
        tr_T = self.train_temps_unique
        te_T = self.test_temps_unique
        if tr_T.size:
            train_line = (
                f"  train T: {len(tr_T)} in [{tr_T.min():.0f}, {tr_T.max():.0f}] K"
            )
        else:
            train_line = "  train T: (none — assign folders to Train/Val in the Data tab)"
        if te_T.size:
            test_line = (
                f"  test  T: {len(te_T)} in [{te_T.min():.0f}, {te_T.max():.0f}] K"
            )
        else:
            test_line = "  test  T: (none)"
        return (
            f"Dataset: {len(self.spectra)} spectra | {n_ch} channels\n"
            f"  train={n_tr}  val={n_va}  test_unseen={n_te}\n"
            f"{train_line}\n"
            f"{test_line}"
        )


def build_dataset_from_roles(
    root: str | Path,
    folder_roles: dict[str, tuple[float, Role]],
    train_frac: float = 0.8,
    rng_seed: int = 42,
    integ_prefix: str = "Integration Time (sec):",
    header_marker: str | None = ">>>>>Begin Spectral Data<<<<<",
    file_ext: str = "*.txt",
    wl_atol: float = 1e-2,
    spectra_per_folder_assert: int | None = None,
    progress_callback=None,
) -> Dataset:
    """Load spectra applying per-folder temperature and split role.

    Parameters
    ----------
    folder_roles
        Maps folder name ``->`` ``(T_K, role)`` where *role* is:

        * ``"omit"`` — folder skipped entirely
        * ``"train_val"`` — spectra split randomly into ``train`` / ``val``
        * ``"test_unseen"`` — every spectrum labeled ``test_unseen``

    Folders present on disk but missing from *folder_roles* are **skipped**.
    """
    root = Path(root)
    folders = discover_spectral_folders(root)
    if not folders:
        raise ValueError(
            f"No temperature folders found under {root}. "
            "Expected either 'p/mXXX' (Celsius) or plain numeric (Kelvin) "
            "sub-folder names."
        )

    rng = np.random.default_rng(rng_seed)
    all_spectra: list[np.ndarray] = []
    all_temps: list[float] = []
    all_labels: list[str] = []
    all_folders: list[str] = []
    all_paths: list[str] = []
    all_itimes: list[float] = []
    wavelengths_ref: np.ndarray | None = None

    all_files_flat: list[tuple[str, str]] = []
    for folder in folders:
        if folder not in folder_roles:
            continue
        role = folder_roles[folder][1]
        if role == "omit":
            continue
        fp = sorted(glob.glob(str(root / folder / file_ext)))
        for f in fp:
            all_files_flat.append((folder, f))
    n_total = len(all_files_flat)
    n_done = 0

    for folder in folders:
        if folder not in folder_roles:
            continue
        t_override, role = folder_roles[folder]
        if role == "omit":
            continue

        folder_path = root / folder
        files = sorted(glob.glob(str(folder_path / file_ext)))
        n = len(files)
        if n == 0:
            continue
        if spectra_per_folder_assert is not None and n != spectra_per_folder_assert:
            warnings.warn(
                f"Folder {folder} has {n} spectra (expected {spectra_per_folder_assert})",
                stacklevel=2,
            )

        folder_intens: list[np.ndarray] = []
        folder_itimes: list[float] = []
        for fpath in files:
            wl, inten, it, _ = read_spectrum_file(
                fpath, integ_prefix, header_marker
            )
            if wavelengths_ref is None:
                wavelengths_ref = wl
            else:
                np.testing.assert_allclose(
                    wl, wavelengths_ref, atol=wl_atol,
                    err_msg=f"Wavelength mismatch in {fpath}",
                )
            folder_intens.append(inten)
            folder_itimes.append(it)
            n_done += 1
            if progress_callback is not None:
                progress_callback(n_done, n_total)

        t_k = float(t_override)
        if role == "test_unseen":
            lab = np.array(["test_unseen"] * n, dtype=object)
        else:
            idx = np.arange(n)
            rng.shuffle(idx)
            if n == 1:
                lab = np.array(["train"], dtype=object)
            else:
                split = max(1, int(train_frac * n))
                if split >= n:
                    split = n - 1
                lab = np.array(["val"] * n, dtype=object)
                lab[idx[:split]] = "train"

        for i in range(n):
            all_spectra.append(folder_intens[i])
            all_temps.append(t_k)
            all_folders.append(folder)
            all_labels.append(str(lab[i]))
            all_paths.append(files[i])
            all_itimes.append(folder_itimes[i])

    if not all_spectra:
        raise ValueError(
            "No spectra loaded — every folder is omitted or has no files. "
            "Assign at least one folder to Train/Val or test_unseen."
        )
    assert wavelengths_ref is not None
    return Dataset(
        spectra=np.asarray(all_spectra, dtype=np.float64),
        wavelengths=wavelengths_ref,
        temperatures=np.asarray(all_temps, dtype=np.float64),
        labels=np.asarray(all_labels),
        folders=np.asarray(all_folders),
        file_paths=all_paths,
        integ_times=np.asarray(all_itimes, dtype=np.float64),
    )


def default_folder_roles_all_train_val(root: str | Path) -> dict[str, tuple[float, Role]]:
    """Default role per folder from parsed T(K), rounded to integer Kelvin.

    * Integer *T* ending in ``0`` → ``train_val`` (random train/val split inside folder).
    * Otherwise → ``test_unseen`` (all spectra in that folder are held out).

    Temperature stored as ``float(int(round(T_from_name)))`` so the Data tab can
    show whole kelvin without decimals.
    """
    out: dict[str, tuple[float, Role]] = {}
    for name in discover_spectral_folders(root):
        t_k = float(int(round(parse_folder_name(name))))
        role: Role = "train_val" if (int(t_k) % 10 == 0) else "test_unseen"
        out[name] = (t_k, role)
    return out


def load_dataset(
    root: str | Path,
    rng_seed: int = 42,
    train_frac: float = 0.8,
    test_only: bool = False,
    integ_prefix: str = "Integration Time (sec):",
    header_marker: str | None = ">>>>>Begin Spectral Data<<<<<",
    file_ext: str = "*.txt",
    wl_atol: float = 1e-2,
    spectra_per_folder_assert: int | None = None,
    progress_callback=None,
) -> Dataset:
    """Convenience loader: **legacy digit-based** train vs test split (tests / scripts).

    * ``test_only=True`` — only non-training-digit folders, all ``test_unseen``.
    * ``test_only=False`` — digit rule for train/val vs ``test_unseen``.
    """
    root = Path(root)
    roles: dict[str, tuple[float, Role]] = {}
    for name in discover_spectral_folders(root):
        t_k = parse_folder_name(name)
        is_tr = is_training_folder(name)
        if test_only:
            if is_tr:
                continue
            roles[name] = (t_k, "test_unseen")
        else:
            if is_tr:
                roles[name] = (t_k, "train_val")
            else:
                roles[name] = (t_k, "test_unseen")
    return build_dataset_from_roles(
        root,
        roles,
        train_frac=train_frac,
        rng_seed=rng_seed,
        integ_prefix=integ_prefix,
        header_marker=header_marker,
        file_ext=file_ext,
        wl_atol=wl_atol,
        spectra_per_folder_assert=spectra_per_folder_assert,
        progress_callback=progress_callback,
    )

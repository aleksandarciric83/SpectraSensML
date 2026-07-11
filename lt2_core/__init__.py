"""lt2_core — headless library for LT2 luminescence thermometry.

All modules import-safe: no Qt, no GUI dependencies.
"""
from .spectrum_io import read_spectrum_file, auto_detect_header
from .dataset import (
    Dataset,
    load_dataset,
    build_dataset_from_roles,
    default_folder_roles_all_train_val,
    discover_spectral_folders,
)
from .preprocess import crop, normalize, compute_lir
from .pca_analysis import fit_pca, transform_pca

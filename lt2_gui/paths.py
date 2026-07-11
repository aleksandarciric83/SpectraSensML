"""paths.py — resolve data files correctly both in dev and frozen builds.

When packaged with PyInstaller, all collected data files land under the
temporary extraction directory ``sys._MEIPASS``.  When running from source,
the package directory itself is the correct root.

Usage::

    from .paths import resource_path
    icon_dir = resource_path("assets")          # -> Path(".../assets")
    help_pdf = resource_path("assets", "LT2_Help.pdf")
"""
from __future__ import annotations

import sys
from pathlib import Path


def resource_path(*parts: str) -> Path:
    """Return an absolute Path to a data resource inside the package.

    *parts* are joined relative to the package root (``lt2_gui/``).
    Works identically in:

    * dev mode  — resolves via ``__file__``
    * PyInstaller *onedir* — resolves via ``sys._MEIPASS``
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # PyInstaller: _MEIPASS points to the extracted bundle root.
        # The spec places lt2_gui/assets/ at <bundle>/assets/, so we
        # address parts relative to _MEIPASS directly.
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    else:
        # Running from source: package root is the directory of this file.
        base = Path(__file__).resolve().parent
    return base.joinpath(*parts)

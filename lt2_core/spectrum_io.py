"""spectrum_io.py
Single-pass spectrum file reader with configurable header detection.

Two modes:
  * marker mode: caller supplies a header-end substring; data starts on the
    first line *after* the line containing that substring.
  * auto-detect mode (default, marker=None): data starts at the first line
    where both tokens parse as floats AND the second line also does.

Delimiter auto-detection:
  * The reader inspects the first candidate data line to pick the delimiter
    automatically from: tab, space/whitespace, comma, semicolon.

Integration-time scaling:
  * integ_regex: a string prefix (not a compiled regex) such as
    "Integration Time (sec):".  The reader appends a greedy numeric-capture
    pattern and searches every header line.  If not found, returns 1.0 and
    sets a warning flag so the GUI can alert the user.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def _build_integ_re(prefix: str) -> re.Pattern:
    """Escape the prefix and append a numeric capture group."""
    return re.compile(re.escape(prefix) + r"\s*([\dEe.+\-]+)")


def read_spectrum_file(
    filepath: str | Path,
    integ_prefix: str = "Integration Time (sec):",
    header_marker: str | None = ">>>>>Begin Spectral Data<<<<<",
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Read a single spectrum file.

    Returns
    -------
    wavelengths : (N,) float64
    intensities : (N,) float64  — already divided by integration time
    integ_time  : float
    integ_found : bool — False if integration time was not found in the file

    Notes
    -----
    * If `header_marker` is None, auto-detect mode is used.
    * `integ_prefix` may be an empty string to disable scaling (always 1.0).
    """
    filepath = Path(filepath)
    integ_re = _build_integ_re(integ_prefix) if integ_prefix else None
    integ_time: float | None = None
    in_data = False
    wl: list[float] = []
    inten: list[float] = []
    lines: list[str] = []

    with filepath.open("r", errors="replace") as f:
        lines = f.readlines()

    if header_marker is not None:
        # ── Marker mode ──────────────────────────────────────────────────
        # Find where data starts so we can detect delimiter before reading.
        data_start_idx = 0
        for idx, line in enumerate(lines):
            if integ_re is not None:
                m = integ_re.search(line)
                if m is not None:
                    integ_time = float(m.group(1))
            if header_marker in line:
                data_start_idx = idx + 1
                break
        delim = _detect_delimiter(lines, data_start_idx)
        for line in lines[data_start_idx:]:
            _try_append(line, wl, inten, delim)
    else:
        # ── Auto-detect mode ─────────────────────────────────────────────
        data_start: int | None = None
        for i, line in enumerate(lines):
            if integ_re is not None and integ_time is None:
                m = integ_re.search(line)
                if m is not None:
                    integ_time = float(m.group(1))
            if data_start is None and i + 1 < len(lines):
                # Try all known delimiters so CSV / semicolon files are found.
                for _d in _DELIMITERS:
                    if _is_data_line(line, _d) and _is_data_line(lines[i + 1], _d):
                        data_start = i
                        break
        if data_start is None:
            raise ValueError(
                f"Could not auto-detect numeric data block in {filepath}"
            )
        delim = _detect_delimiter(lines, data_start)
        for line in lines[data_start:]:
            _try_append(line, wl, inten, delim)

    if not wl:
        raise ValueError(f"No spectral data found in {filepath}")

    it = integ_time if integ_time is not None else 1.0
    found = integ_time is not None
    return (
        np.asarray(wl, dtype=np.float64),
        np.asarray(inten, dtype=np.float64) / it,
        it,
        found,
    )


def auto_detect_header(filepath: str | Path) -> tuple[int, bool]:
    """Return (data_start_line_idx, integ_found) without loading the whole array.
    Useful for the GUI's 'validate 5 random files' preview."""
    filepath = Path(filepath)
    with filepath.open("r", errors="replace") as f:
        lines = f.readlines()
    integ_found = any(
        re.search(r"integration time", l, re.IGNORECASE) for l in lines
    )
    for i, line in enumerate(lines[:-1]):
        # Try all delimiters so mixed-format files are detected correctly.
        for delim in _DELIMITERS:
            if _is_data_line(line, delim) and _is_data_line(lines[i + 1], delim):
                return i, integ_found
    return -1, integ_found


# ─── helpers ─────────────────────────────────────────────────────────────

# Delimiter candidates tried in priority order.
_DELIMITERS = ["\t", ",", ";", None]  # None → str.split() (any whitespace)


def _split(line: str, delim: str | None) -> list[str]:
    """Split *line* on *delim* (None = any whitespace) and strip each token."""
    if delim is None:
        return line.split()
    return [t.strip() for t in line.split(delim)]


def _detect_delimiter(lines: list[str], start: int = 0) -> str | None:
    """Scan up to 5 candidate data lines and return the delimiter that
    consistently yields ≥2 numeric tokens, or None (whitespace fallback)."""
    checked = 0
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        for delim in _DELIMITERS:
            parts = _split(line, delim)
            if len(parts) >= 2:
                try:
                    float(parts[0]); float(parts[1])
                    return delim   # first delimiter that works
                except ValueError:
                    continue
        checked += 1
        if checked >= 5:
            break
    return None  # fallback: whitespace


def _is_data_line(line: str, delim: str | None = None) -> bool:
    parts = _split(line.strip(), delim)
    if len(parts) < 2:
        return False
    try:
        float(parts[0])
        float(parts[1])
        return True
    except ValueError:
        return False


def _try_append(line: str, wl: list, inten: list, delim: str | None = None) -> None:
    line = line.strip()
    if not line:
        return
    parts = _split(line, delim)
    if len(parts) < 2:
        return
    try:
        wl.append(float(parts[0]))
        inten.append(float(parts[1]))
    except ValueError:
        pass

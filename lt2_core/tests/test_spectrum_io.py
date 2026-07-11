"""Tests for spectrum_io.py."""
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

from lt2_core.spectrum_io import read_spectrum_file, auto_detect_header


# ── helpers ──────────────────────────────────────────────────────────────

def _write_file(content: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
    tmp.write(textwrap.dedent(content))
    tmp.flush()
    return Path(tmp.name)


MARKER_FILE = """\
    SomeHeader
    Integration Time (sec): 0.5
    OtherHeader
    >>>>>Begin Spectral Data<<<<<
    900.0\t100.0
    901.0\t200.0
    902.0\t300.0
"""

AUTO_FILE = """\
    SomeHeader
    Integration Time (sec): 2.0
    AnotherHeader
    900.0\t100.0
    901.0\t200.0
    902.0\t300.0
"""

NO_INTEG_FILE = """\
    NoIntegTime
    >>>>>Begin Spectral Data<<<<<
    900.0\t50.0
    901.0\t60.0
"""


# ── tests ──────────────────────────────────────────────────────────────────

def test_marker_mode_reads_data():
    p = _write_file(MARKER_FILE)
    wl, inten, it, found = read_spectrum_file(p)
    assert found
    assert it == pytest.approx(0.5)
    assert len(wl) == 3
    np.testing.assert_allclose(wl, [900.0, 901.0, 902.0])
    np.testing.assert_allclose(inten, [100.0 / 0.5, 200.0 / 0.5, 300.0 / 0.5])


def test_auto_detect_mode():
    p = _write_file(AUTO_FILE)
    wl, inten, it, found = read_spectrum_file(p, header_marker=None)
    assert found
    assert it == pytest.approx(2.0)
    np.testing.assert_allclose(inten, [50.0, 100.0, 150.0])


def test_missing_integration_time_returns_false():
    p = _write_file(NO_INTEG_FILE)
    wl, inten, it, found = read_spectrum_file(p, integ_prefix="Integration Time (sec):")
    assert not found
    assert it == 1.0
    assert len(wl) == 2


def test_auto_detect_header_returns_correct_index():
    p = _write_file(AUTO_FILE)
    idx, integ_found = auto_detect_header(p)
    assert idx >= 0
    assert integ_found


def test_empty_integ_prefix_disables_scaling():
    """If integ_prefix is empty string, intensity should NOT be divided."""
    p = _write_file(MARKER_FILE)
    wl, inten, it, found = read_spectrum_file(p, integ_prefix="")
    # No integration time found → fallback 1.0, no division
    assert it == 1.0
    np.testing.assert_allclose(inten, [100.0, 200.0, 300.0])

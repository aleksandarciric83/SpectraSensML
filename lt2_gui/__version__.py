"""Central version / author metadata for the SpectraSensML app."""
from __future__ import annotations

import json
from pathlib import Path

APP_NAME = "SpectraSensML"
APP_VERSION = "1.0"

AUTHOR_NAME = "Aleksandar Ciric"
AUTHOR_EMAIL = "aleksandar.ciric@ff.bg.ac.rs"
AUTHOR_GROUP = "OMAS group"
AUTHOR_GROUP_URL = "https://www.omasgroup.org"

LICENSE_SHORT = "GPLv3"
LICENSE_LONG = "GNU General Public License v3.0 (GPLv3)"

CITATION_NOTE = (
    "If you use this software in published or otherwise disseminated work "
    "(scientific paper, thesis, report, presentation, application note, "
    "industrial document, etc.), you are required to cite the accompanying "
    "publication that references this software."
)

# ── Citation loaded from citation.json ──────────────────────────────────────

_CITATION_URL = "https://www.omasgroup.org/citation.json"
_CITATION_TIMEOUT = 4  # seconds


def _load_citation() -> dict:
    """Load citation data. Returns an empty dict on total failure.

    Fetch order:
    1. Remote URL (omasgroup.org/citation.json) — always tried first so the
       citation can be updated without rebuilding the app.
    2. Local citation.json bundled with the app — used as fallback when
       offline or the server is unreachable.
    """
    import sys

    # ── 1. Try remote ────────────────────────────────────────────────────────
    try:
        from urllib.request import urlopen, Request
        from urllib.error import URLError
        req = Request(_CITATION_URL, headers={"User-Agent": "SpectraSensML"})
        with urlopen(req, timeout=_CITATION_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception:
        pass  # network unavailable — fall through to local copy

    # ── 2. Fall back to bundled local file ───────────────────────────────────
    candidates: list[Path] = [Path(__file__).resolve().parent / "citation.json"]
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "lt2_gui" / "citation.json")  # type: ignore[attr-defined]
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: v for k, v in data.items() if not k.startswith("_")}
        except Exception:
            continue
    return {}


CITATION: dict = _load_citation()


def citation_text() -> str:
    """Single-line citation string suitable for the banner and About tab.

    Falls back to CITATION_NOTE if no publication data is filled in yet.
    """
    c = CITATION
    parts: list[str] = []

    authors = c.get("authors", "").strip()
    title   = c.get("title",   "").strip()
    journal = c.get("journal", "").strip()
    volume  = c.get("volume",  "").strip()
    pages   = c.get("pages",   "").strip()
    year    = c.get("year",    "").strip()
    doi     = c.get("doi",     "").strip()
    note    = c.get("note",    "").strip()

    if authors:
        parts.append(authors)
    if title:
        parts.append(f'"{title}"')
    if journal:
        jpart = journal
        if volume:
            jpart += f" {volume}"
        if pages:
            jpart += f", {pages}"
        if year:
            jpart += f" ({year})"
        parts.append(jpart)
    elif year:
        parts.append(f"({year})")
    if doi:
        parts.append(f"DOI: {doi}")
    if note and not parts:
        parts.append(note)

    if parts:
        return ", ".join(parts)
    return CITATION_NOTE


def app_title() -> str:
    return f"{APP_NAME} v{APP_VERSION}"

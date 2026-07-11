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

def _load_citation() -> dict:
    """Load citation.json. Returns an empty dict on any failure.

    Search order:
    1. Next to this file (dev mode and PyInstaller frozen build both land here).
    2. sys._MEIPASS / lt2_gui / citation.json (extra safety for frozen builds).
    """
    import sys
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

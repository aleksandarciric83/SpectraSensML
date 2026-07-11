"""About tab (Tab 8) — author, licence, citation, OS info."""
from __future__ import annotations

import platform
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..__version__ import (
    APP_NAME,
    APP_VERSION,
    AUTHOR_EMAIL,
    AUTHOR_GROUP,
    AUTHOR_GROUP_URL,
    AUTHOR_NAME,
    CITATION,
    CITATION_NOTE,
    LICENSE_LONG,
    citation_text,
)


class AboutTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignTop)

        # ── App identity ────────────────────────────────────────────────
        header = QLabel(
            f"<h1>{APP_NAME}</h1>"
            f"<p><b>Version {APP_VERSION}</b></p>"
        )
        header.setTextFormat(Qt.RichText)
        outer.addWidget(header)

        # ── Author ──────────────────────────────────────────────────────
        author = QGroupBox("Author")
        a_layout = QVBoxLayout(author)
        a_label = QLabel(
            f"<p><b>{AUTHOR_NAME}</b><br>"
            f'<a href="mailto:{AUTHOR_EMAIL}">{AUTHOR_EMAIL}</a></p>'
            f"<p>{AUTHOR_GROUP} · "
            f'<a href="{AUTHOR_GROUP_URL}">{AUTHOR_GROUP_URL}</a></p>'
        )
        a_label.setTextFormat(Qt.RichText)
        a_label.setOpenExternalLinks(True)
        a_layout.addWidget(a_label)
        outer.addWidget(author)

        # ── Licence ────────────────────────────────────────────────────
        lic = QGroupBox("License")
        l_layout = QVBoxLayout(lic)
        l_label = QLabel(
            f"<p>This software is released under the "
            f"<b>{LICENSE_LONG}</b>.</p>"
            "<p>You are free to use, modify, and redistribute it, subject "
            "to the terms of the licence; in particular, derivative works "
            "must also be licensed under the GPLv3.</p>"
            "<p>Full text: "
            '<a href="https://www.gnu.org/licenses/gpl-3.0.html">'
            "https://www.gnu.org/licenses/gpl-3.0.html</a></p>"
        )
        l_label.setTextFormat(Qt.RichText)
        l_label.setOpenExternalLinks(True)
        l_label.setWordWrap(True)
        l_layout.addWidget(l_label)
        outer.addWidget(lic)

        # ── Citation ───────────────────────────────────────────────────
        cite = QGroupBox("Citation requirement")
        c_layout = QVBoxLayout(cite)

        note_label = QLabel(f"<p>{CITATION_NOTE}</p>")
        note_label.setTextFormat(Qt.RichText)
        note_label.setWordWrap(True)
        c_layout.addWidget(note_label)

        ref = citation_text()
        doi = CITATION.get("doi", "").strip()
        doi_html = (
            f'<br><a href="https://doi.org/{doi}">https://doi.org/{doi}</a>'
            if doi else ""
        )
        note_str = CITATION.get("note", "").strip()
        note_html = (
            f'<br><span style="color:#777;">{note_str}</span>'
            if note_str and not doi else ""
        )
        ref_label = QLabel(
            f"<p style='font-family:monospace; background:#f5f5f5; "
            f"padding:6px; border:1px solid #ccc;'>{ref}</p>"
            f"{doi_html}{note_html}"
            f"<p style='color:#777; font-size:small;'>"
            f"Edit <code>lt2_gui/citation.json</code> to update this reference.</p>"
        )
        ref_label.setTextFormat(Qt.RichText)
        ref_label.setOpenExternalLinks(True)
        ref_label.setWordWrap(True)
        c_layout.addWidget(ref_label)
        outer.addWidget(cite)

        # ── Environment ────────────────────────────────────────────────
        env = QGroupBox("Runtime environment (detected)")
        e_layout = QVBoxLayout(env)
        sys_label = QLabel(self._env_block())
        sys_label.setTextFormat(Qt.RichText)
        sys_label.setWordWrap(True)
        e_layout.addWidget(sys_label)
        outer.addWidget(env)

        outer.addStretch()

    def _env_block(self) -> str:
        try:
            from PySide6.QtCore import qVersion
            qt_ver = qVersion()
        except Exception:
            qt_ver = "unknown"
        return (
            "<table style='font-family:monospace;'>"
            f"<tr><td>OS</td><td>{platform.system()} {platform.release()}</td></tr>"
            f"<tr><td>Platform</td><td>{platform.platform()}</td></tr>"
            f"<tr><td>Machine</td><td>{platform.machine()}</td></tr>"
            f"<tr><td>Python</td><td>{sys.version.split()[0]}</td></tr>"
            f"<tr><td>Qt</td><td>{qt_ver}</td></tr>"
            "</table>"
        )

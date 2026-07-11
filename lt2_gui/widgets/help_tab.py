"""Help tab (Tab 7) — rendered Help content + link to bundled PDF."""
from __future__ import annotations

import os
import subprocess
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..help_text import HELP_SECTIONS, help_markdown
from ..paths import resource_path


# Bundled PDF path — works in source and in frozen PyInstaller builds.
BUNDLED_HELP_PDF = resource_path("assets", "LT2_Help.pdf")


class HelpTab(QWidget):
    """Long-form documentation pane with a navigable side list.

    The PDF version of the same content is shipped with the app at
    ``lt2_gui/assets/LT2_Help.pdf`` (see :data:`BUNDLED_HELP_PDF`); the
    "Open Help PDF" button here just reveals that file in the OS file
    manager / default PDF viewer.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        pdf_label = QLabel(
            f"Bundled PDF version: <code>{BUNDLED_HELP_PDF}</code>"
        )
        pdf_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top_bar.addWidget(pdf_label)
        top_bar.addStretch()
        self.open_pdf_btn = QPushButton("Open Help PDF")
        self.open_pdf_btn.clicked.connect(self._open_pdf)
        top_bar.addWidget(self.open_pdf_btn)
        outer.addLayout(top_bar)

        splitter = QSplitter(Qt.Horizontal)

        self.toc = QListWidget()
        self.toc.setMinimumWidth(220)
        self.toc.setMaximumWidth(320)
        for sec in HELP_SECTIONS:
            item = QListWidgetItem(sec["title"])
            self.toc.addItem(item)
        self.toc.currentRowChanged.connect(self._jump_to_section)
        splitter.addWidget(self.toc)

        self.viewer = QTextBrowser()
        self.viewer.setOpenExternalLinks(True)
        self.viewer.setMarkdown(help_markdown())
        # QTextBrowser does not expose markdown anchors directly, so the
        # side list scrolls by searching for the section title.
        splitter.addWidget(self.viewer)
        splitter.setSizes([250, 900])
        outer.addWidget(splitter, 1)

        if self.toc.count() > 0:
            self.toc.setCurrentRow(0)

    # ── interactions ─────────────────────────────────────────────────────

    def _jump_to_section(self, row: int) -> None:
        if row < 0 or row >= len(HELP_SECTIONS):
            return
        title = HELP_SECTIONS[row]["title"]
        if not self.viewer.find(title):
            cursor = self.viewer.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            self.viewer.setTextCursor(cursor)
            self.viewer.find(title)

    def _open_pdf(self) -> None:
        path = BUNDLED_HELP_PDF
        if not path.is_file():
            QMessageBox.warning(
                self,
                "Help PDF not found",
                f"The bundled Help PDF was not found at:\n{path}\n\n"
                "Re-installing the app or running "
                "`python -m lt2_gui.build_help_pdf` will regenerate it.",
            )
            return
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            elif sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Could not open Help PDF",
                f"Failed to open {path}:\n{exc}",
            )

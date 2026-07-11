"""help_widgets.py — small reusable widgets for contextual help.

Provides:
  * ``HelpButton``     — tiny "?" button that opens a tooltip popup with
                         the per-field help text.
  * ``labeled_help_row`` — convenience: returns a QWidget containing
                         ``QLabel("Foo:")`` + ``HelpButton(field_key)`` so
                         it can be used as the label column in a
                         ``QFormLayout``.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QToolButton,
    QWidget,
)

from ..help_text import FIELD_HELP


class HelpButton(QToolButton):
    """A small ``?`` button that opens an explanatory popup for *field_key*."""

    def __init__(self, field_key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.field_key = field_key
        self.setText("?")
        self.setAutoRaise(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(QSize(18, 18))
        self.setStyleSheet(
            "QToolButton{"
            "border:1px solid palette(mid);"
            "border-radius:9px;"
            "font-weight:bold;"
            "padding:0px;"
            "}"
            "QToolButton:hover{ background: palette(highlight); color: palette(highlighted-text); }"
        )
        info = FIELD_HELP.get(field_key)
        title = info["title"] if info else field_key
        text = info["text"] if info else "(no help for this field yet)"
        self.setToolTip(text)
        self._title = title
        self._text = text
        self.clicked.connect(self._show_popup)

    def _show_popup(self) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Help — " + self._title)
        box.setText(f"<b>{self._title}</b>")
        box.setInformativeText(self._text)
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()


def labeled_help_row(label_text: str, field_key: str) -> QWidget:
    """QWidget containing ``QLabel(label_text)`` + ``HelpButton(field_key)``.

    Use as the *label* argument in ``QFormLayout.addRow(label, field_widget)``.
    """
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(4)
    lbl = QLabel(label_text)
    h.addWidget(lbl)
    h.addWidget(HelpButton(field_key))
    h.addStretch()
    return w

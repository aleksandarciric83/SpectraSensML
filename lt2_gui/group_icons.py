"""Small decorative icons for model groups (40×40 px)."""
from __future__ import annotations

from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QPainter, QPixmap, QPolygon


def _pixmap_from_draw(draw_fn) -> QPixmap:
    pm = QPixmap(40, 40)
    pm.fill(QColor(245, 245, 245))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    draw_fn(p)
    p.end()
    return pm


def icon_group_b() -> QPixmap:
    """Spline / curve hint for Group B."""

    def draw(p: QPainter):
        p.setPen(QColor(30, 100, 180))
        p.drawPolyline(
            QPolygon(
                [
                    QPoint(4, 32),
                    QPoint(12, 10),
                    QPoint(22, 28),
                    QPoint(36, 6),
                ]
            )
        )

    return _pixmap_from_draw(draw)


def icon_group_a() -> QPixmap:
    """Tree / block hint for Group A."""

    def draw(p: QPainter):
        p.setBrush(QColor(34, 139, 34))
        p.setPen(QColor(20, 90, 20))
        p.drawRect(16, 18, 10, 16)
        p.setBrush(QColor(60, 160, 60))
        p.drawEllipse(6, 8, 28, 18)

    return _pixmap_from_draw(draw)


def icon_group_c() -> QPixmap:
    """Gaussian / kernel hint for Group C."""

    def draw(p: QPainter):
        p.setPen(QColor(160, 40, 140))
        for i, y in enumerate([30, 22, 16, 12, 16, 22, 30]):
            p.drawLine(6 + i * 4, y, 6 + i * 4, 34)

    return _pixmap_from_draw(draw)


def icon_group_d() -> QPixmap:
    """Network hint for Group D."""

    def draw(p: QPainter):
        p.setPen(QColor(200, 100, 20))
        for (x1, y1), (x2, y2) in [
            ((8, 10), (20, 20)),
            ((20, 20), (32, 10)),
            ((8, 30), (20, 20)),
            ((20, 20), (32, 30)),
        ]:
            p.drawLine(x1, y1, x2, y2)
        for x, y in [(8, 10), (32, 10), (20, 20), (8, 30), (32, 30)]:
            p.setBrush(QColor(255, 140, 0))
            p.drawEllipse(x - 2, y - 2, 5, 5)

    return _pixmap_from_draw(draw)

"""splash.py — startup splash screen for SpectraSensML."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QSplashScreen


def _build_pixmap(icon_path: str, width: int = 480, height: int = 420) -> QPixmap:
    """Compose the splash pixmap: icon + app name + author + version."""
    from lt2_gui.__version__ import APP_NAME, APP_VERSION, AUTHOR_EMAIL, AUTHOR_NAME

    px = QPixmap(width, height)
    px.fill(QColor("#0f1b2d"))

    painter = QPainter(px)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    # Icon (top 60% of splash)
    icon_px = QPixmap(icon_path)
    if not icon_px.isNull():
        icon_size = int(width * 0.52)
        icon_px = icon_px.scaled(
            icon_size, icon_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (width - icon_px.width()) // 2
        painter.drawPixmap(x, 12, icon_px)
        text_top = 12 + icon_px.height() + 10
    else:
        text_top = 20

    # App name
    name_font = QFont("Arial", 26, QFont.Weight.Bold)
    painter.setFont(name_font)
    painter.setPen(QColor("#ffffff"))
    painter.drawText(0, text_top, width, 42, Qt.AlignmentFlag.AlignHCenter, APP_NAME)

    # Version
    ver_font = QFont("Arial", 12)
    painter.setFont(ver_font)
    painter.setPen(QColor("#90caf9"))
    painter.drawText(0, text_top + 44, width, 24, Qt.AlignmentFlag.AlignHCenter, f"v{APP_VERSION}")

    # Separator line
    painter.setPen(QColor("#1e3a5f"))
    sep_y = text_top + 76
    painter.drawLine(40, sep_y, width - 40, sep_y)

    # Author name
    auth_font = QFont("Arial", 11)
    painter.setFont(auth_font)
    painter.setPen(QColor("#b0bec5"))
    painter.drawText(0, sep_y + 10, width, 24, Qt.AlignmentFlag.AlignHCenter, AUTHOR_NAME)

    # Author email
    email_font = QFont("Arial", 9)
    painter.setFont(email_font)
    painter.setPen(QColor("#607d8b"))
    painter.drawText(0, sep_y + 34, width, 20, Qt.AlignmentFlag.AlignHCenter, AUTHOR_EMAIL)

    # Loading hint at very bottom
    hint_font = QFont("Arial", 8)
    painter.setFont(hint_font)
    painter.setPen(QColor("#37474f"))
    painter.drawText(0, height - 18, width, 16, Qt.AlignmentFlag.AlignHCenter, "Loading…")

    painter.end()
    return px


def show_splash(icon_path: str, duration_ms: int = 3000) -> QSplashScreen:
    """Create, show, and auto-close a splash screen.

    Returns the QSplashScreen so the caller can call
    ``splash.finish(main_window)`` after the main window is ready.
    The QTimer ensures the splash stays visible for at least *duration_ms*.
    """
    px = _build_pixmap(icon_path)
    splash = QSplashScreen(px, Qt.WindowType.WindowStaysOnTopHint)
    splash.setMask(px.mask())
    splash.show()
    # Process events so the splash is actually painted before the heavy
    # import chain of the main window starts.
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()
    # Auto-close timer (fallback — finish() is also called after main window shows)
    QTimer.singleShot(duration_ms, splash.close)
    return splash

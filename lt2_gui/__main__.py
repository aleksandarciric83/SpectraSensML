"""Entry point: python -m lt2_gui"""
import multiprocessing
import os
import platform
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

# When this file is launched directly, e.g. by the PyInstaller bootloader on
# Windows/macOS/Linux (the spec points straight at __main__.py) or by running
# `python lt2_gui/__main__.py`, Python assigns it no parent package, so the
# relative imports below would raise "attempted relative import with no
# known parent package". Detect that case, put the repo root on sys.path,
# and fall back to absolute imports; `python -m lt2_gui` keeps using the
# normal relative-import path.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lt2_gui.__version__ import APP_NAME, APP_VERSION
    from lt2_gui.main_window import MainWindow
    from lt2_gui.paths import resource_path
else:
    from .__version__ import APP_NAME, APP_VERSION
    from .main_window import MainWindow
    from .paths import resource_path


def main():
    # Cross-OS detection — `platform.system()` returns "Linux", "Darwin"
    # (macOS) or "Windows". Logged to stdout at startup so users running
    # the app from a terminal can confirm the right runtime was picked up;
    # the actual file picker dialogs come from Qt and are already native
    # on every platform.
    os_name = platform.system() or "unknown"
    print(
        f"{APP_NAME} v{APP_VERSION} — detected OS: {os_name} "
        f"({platform.platform()})"
    )

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("OMAS group")

    icon_path = str(resource_path("assets", "app_icon.png"))
    try:
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
    except Exception:
        pass

    try:
        from PySide6.QtGui import QImageReader
        lim = getattr(QImageReader, "setAllocationLimit", None)
        if callable(lim):
            lim(512)  # megabytes; large matplotlib PNGs at 600 DPI can exceed Qt defaults
    except Exception:
        pass

    # Splash screen — shown for 3 s minimum, closed when main window is ready
    splash = None
    try:
        if __package__ in (None, ""):
            from lt2_gui.splash import show_splash
        else:
            from .splash import show_splash
        if os.path.exists(icon_path):
            splash = show_splash(icon_path, duration_ms=3000)
    except Exception:
        pass

    win = MainWindow()
    win.show()

    if splash is not None:
        splash.finish(win)

    sys.exit(app.exec())


if __name__ == "__main__":
    # Required on Windows so that frozen multiprocessing workers (used by
    # joblib/sklearn inside the benchmark) do not recursively spawn the GUI.
    multiprocessing.freeze_support()
    main()

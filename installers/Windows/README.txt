SpectraSensML v1.0 — Windows Installer Build Kit
=================================================

This folder contains everything needed to produce a self-contained
Windows installer (SpectraSensML_1.0_win64_setup.exe) that end users
can run without installing Python.

REQUIREMENTS
────────────
• Windows 10 or 11, x64
• Python 3.12 (recommended) or 3.11
    Download: https://python.org/downloads/
    ✓ Check "Add Python to PATH" during install
• Inno Setup 6 (free, for the final .exe wizard)
    Download: https://jrsoftware.org/isinfo.php
• Internet connection (~4 GB download for packages + PyTorch)

HOW TO BUILD  (one step)
──────────────────────────
1. Clone or download the repository from GitHub:
     https://github.com/aleksandarciric83/SpectraSensML
   OR copy this entire folder to any Windows 10/11 x64 machine.
2. Double-click  BUILD.bat
   – OR – open cmd.exe in this folder and run:  BUILD.bat

The script automatically:
  • Creates an isolated Python venv (.venv-build\)
  • Downloads and installs all dependencies (PySide6, scikit-learn,
    PyTorch CPU, XGBoost, LightGBM, CatBoost, etc.)
  • Runs PyInstaller to create the self-contained bundle
  • Runs Inno Setup to produce the final installer

OUTPUT
──────
  packaging\Output\SpectraSensML_1.0_win64_setup.exe   ← distribute this

The installer:
  • Installs to  %ProgramFiles%\SpectraSensML\
  • Creates a Start Menu shortcut
  • Optionally creates a Desktop shortcut
  • Includes a full uninstaller

NOTES
─────
• First build takes 20-40 min (large downloads). Subsequent builds
  reuse the .venv-build\ folder and are much faster.
• Expected installer size: ~1-1.5 GB (PyTorch CPU dominates).
• The app is unsigned; Windows SmartScreen may warn on first run.
  For lab distribution this is acceptable.
• Internet is NOT needed by end users — the installer is self-contained.

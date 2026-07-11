SpectraSensML v1.0 — Linux AppImage Build Kit
=============================================

This folder contains everything needed to produce a self-contained
Linux AppImage (SpectraSensML_1.0_linux_x86_64.AppImage) that end
users can run without installing Python.

REQUIREMENTS
────────────
• Linux x86_64, Ubuntu 22.04+ recommended (or equivalent glibc 2.35+)
• Python 3.12 (recommended) or 3.11
    Ubuntu:   sudo apt-get install python3.12 python3.12-venv
    Fedora:   sudo dnf install python3.12
• Internet connection (~4 GB download for packages + PyTorch)
• sudo access (for system library installation)

HOW TO BUILD  (one step)
──────────────────────────
1. Clone or download the repository from GitHub:
     https://github.com/aleksandarciric83/SpectraSensML
   OR copy this entire folder to a Linux x86_64 machine.
2. Run:
    chmod +x BUILD.sh
    ./BUILD.sh

   OR with an explicit Python path:
    ./BUILD.sh /usr/bin/python3.12

The script automatically:
  • Installs required system libraries (libfuse2, libGL, etc.)
  • Creates an isolated Python venv (.venv-build/)
  • Downloads and installs all dependencies (PySide6, scikit-learn,
    PyTorch CPU, XGBoost, LightGBM, CatBoost, etc.)
  • Runs PyInstaller to create the self-contained bundle
  • Packages it as an AppImage using appimagetool (auto-downloaded)

OUTPUT
──────
  packaging/Output/SpectraSensML_1.0_linux_x86_64.AppImage   ← distribute this

End-user runs it with:
  chmod +x SpectraSensML_1.0_linux_x86_64.AppImage
  ./SpectraSensML_1.0_linux_x86_64.AppImage

No installation, no root access needed by end users.

NOTES
─────
• First build takes 20-40 min (large downloads). Subsequent builds
  reuse .venv-build/ and are much faster.
• Expected AppImage size: ~1.5-2 GB.
• End users need libfuse2 installed:
    Ubuntu/Debian: sudo apt-get install libfuse2
    Fedora/RHEL:   sudo dnf install fuse
• The app works on Ubuntu 22.04, Fedora 36+, Arch, and most modern
  Linux distros with glibc 2.35+. Build on Ubuntu 22.04 for the widest
  compatibility.

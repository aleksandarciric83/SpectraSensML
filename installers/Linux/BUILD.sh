#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  SpectraSensML v1.0 — Linux build script
#  Run this script on a Linux x86_64 machine to produce:
#    packaging/Output/SpectraSensML_1.0_linux_x86_64.AppImage
#
#  Tested on: Ubuntu 22.04 LTS x86_64
#  Requires:  Python 3.12 (or 3.11), internet connection
#
#  Usage:
#    chmod +x BUILD.sh
#    ./BUILD.sh
#    # OR with an explicit Python path:
#    ./BUILD.sh /usr/bin/python3.12
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

PYTHON="${1:-python3.12}"

echo "================================================================"
echo " SpectraSensML v1.0 - Linux AppImage Builder"
echo "================================================================"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv-build"

# ── Check Python ──────────────────────────────────────────────────────
if ! command -v "$PYTHON" &>/dev/null; then
    echo "Python '$PYTHON' not found. Trying python3..."
    PYTHON=python3
    if ! command -v "$PYTHON" &>/dev/null; then
        echo "ERROR: Python not found."
        echo "Install: sudo apt-get install python3.12 python3.12-venv"
        exit 1
    fi
fi
echo "Using: $PYTHON ($($PYTHON --version))"

# ── System dependencies ───────────────────────────────────────────────
echo "Checking system dependencies..."
MISSING=()
for pkg in libfuse2 libgl1-mesa-glx libglib2.0-0 libdbus-1-3 libegl1; do
    dpkg -s "$pkg" &>/dev/null || MISSING+=("$pkg")
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Installing missing system packages: ${MISSING[*]}"
    sudo apt-get install -y "${MISSING[@]}" || \
        echo "WARNING: Could not install system packages. Build may fail."
fi

# ── Build venv ────────────────────────────────────────────────────────
if [ ! -f "$VENV/bin/activate" ]; then
    echo "Creating build virtual environment..."
    "$PYTHON" -m venv "$VENV"
fi
source "$VENV/bin/activate"
echo "Active Python: $(python --version)"

# ── Install dependencies ──────────────────────────────────────────────
echo "Installing dependencies (this may take 10-20 minutes)..."
pip install --quiet --upgrade pip wheel setuptools

grep -v '^#' "$HERE/packaging/requirements-lock.txt" \
    | grep -v '^torch' \
    | grep -v '^\s*$' \
    | pip install --quiet -r /dev/stdin

echo "Installing PyTorch CPU (large download ~1 GB)..."
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu

# ── Expose app source ─────────────────────────────────────────────────
export PYTHONPATH="$HERE:${PYTHONPATH:-}"

# ── Generate Help PDF ─────────────────────────────────────────────────
echo "Generating Help PDF..."
python -m lt2_gui.build_help_pdf 2>/dev/null || echo "(PDF generation skipped)"

# ── PyInstaller ───────────────────────────────────────────────────────
echo "Running PyInstaller (this takes 5-15 minutes)..."
cd "$HERE"
python -m PyInstaller --noconfirm --clean packaging/LT2_Thermometry.spec

# ── AppImage ──────────────────────────────────────────────────────────
echo "Building AppImage..."
bash packaging/build_appimage.sh

echo ""
echo "================================================================"
echo " DONE!"
echo " Installer: packaging/Output/SpectraSensML_1.0_linux_x86_64.AppImage"
echo "================================================================"
echo ""
echo " Users run it with:"
echo "   chmod +x SpectraSensML_1.0_linux_x86_64.AppImage"
echo "   ./SpectraSensML_1.0_linux_x86_64.AppImage"

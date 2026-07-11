#!/usr/bin/env bash
# build_pyinstaller.sh — create the onedir PyInstaller bundle on macOS or Linux.
#
# Usage (run from repo root Yb/):
#   bash packaging/build_pyinstaller.sh [python3.12]
#
# The optional first argument lets you specify a custom Python 3.12 binary if
# 'python3.12' is not on PATH (e.g. '/usr/local/bin/python3.12').
#
# Outputs:
#   dist/LT2 Thermometry/          <-- the self-contained app folder
#   dist/LT2 Thermometry.app/      <-- macOS .app bundle (macOS only)

set -euo pipefail

PYTHON="${1:-python3.12}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv-build"

echo "==> Repo root : $REPO_ROOT"
echo "==> Python    : $PYTHON  ($(${PYTHON} --version 2>&1))"
echo "==> Venv      : $VENV_DIR"

# ── 1. Create / reuse build venv ──────────────────────────────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "==> Creating build venv …"
    "$PYTHON" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "==> Active Python: $(python --version)"

# ── 2. Install dependencies ───────────────────────────────────────────────────
echo "==> Upgrading pip, wheel, setuptools …"
pip install --quiet --upgrade pip wheel setuptools

echo "==> Installing runtime requirements (excluding torch) …"
# Install everything except lines beginning with 'torch' (handled separately)
grep -v '^#' "$REPO_ROOT/packaging/requirements-lock.txt" \
  | grep -v '^torch' \
  | grep -v '^\s*$' \
  | pip install --quiet -r /dev/stdin

echo "==> Installing PyTorch CPU-only wheel …"
pip install --quiet torch \
    --index-url https://download.pytorch.org/whl/cpu

# ── 3. Install the app packages themselves ────────────────────────────────────
echo "==> Installing lt2_core and lt2_gui …"
pip install --quiet -e "$REPO_ROOT"  2>/dev/null || true
# If there is no setup.py/pyproject.toml, add the path manually instead:
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# ── 4. Regenerate Help PDF (in case source changed) ───────────────────────────
echo "==> Generating Help PDF …"
python -m lt2_gui.build_help_pdf || echo "(Warning: build_help_pdf failed — continuing without fresh PDF)"

# ── 5. Run PyInstaller ────────────────────────────────────────────────────────
echo "==> Running PyInstaller …"
cd "$REPO_ROOT"
python -m PyInstaller \
    --noconfirm \
    --clean \
    packaging/LT2_Thermometry.spec

echo ""
echo "==> Build complete."
echo "    Bundle : $REPO_ROOT/dist/SpectraSensML/"
if [ "$(uname)" = "Darwin" ]; then
    echo "    .app   : $REPO_ROOT/dist/SpectraSensML.app/"
    echo "    Next   : bash packaging/build_dmg.sh"
else
    echo "    Next   : bash packaging/build_appimage.sh"
fi

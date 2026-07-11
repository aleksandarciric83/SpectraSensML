#!/usr/bin/env bash
# build_appimage.sh — package the PyInstaller onedir bundle as a Linux AppImage.
#
# Prerequisites (Ubuntu 22.04+ recommended; x86_64 only):
#   - Run  bash packaging/build_pyinstaller.sh  first.
#     The bundle must exist at  dist/LT2 Thermometry/
#   - appimagetool must be available (downloaded automatically below).
#   - libfuse2 must be installed:
#       sudo apt-get install libfuse2
#
# Usage (from repo root Yb/):
#   bash packaging/build_appimage.sh
#
# Output:
#   packaging/Output/LT2_Thermometry_1.0_linux_x86_64.AppImage

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="LT2 Thermometry"
APP_VERSION="1.0"
BUNDLE_DIR="$REPO_ROOT/dist/${APP_NAME}"
OUTPUT_DIR="$REPO_ROOT/packaging/Output"
APPIMAGE_OUT="$OUTPUT_DIR/LT2_Thermometry_${APP_VERSION}_linux_x86_64.AppImage"
APPDIR="$REPO_ROOT/packaging/AppDir"

echo "==> Repo root  : $REPO_ROOT"
echo "==> Bundle dir : $BUNDLE_DIR"
echo "==> AppImage   : $APPIMAGE_OUT"

if [ ! -d "$BUNDLE_DIR" ]; then
    echo "ERROR: $BUNDLE_DIR not found. Run build_pyinstaller.sh first."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── Download appimagetool if not on PATH ──────────────────────────────────────
APPIMAGETOOL="${APPIMAGETOOL:-appimagetool}"
if ! command -v "$APPIMAGETOOL" &>/dev/null; then
    TOOL_PATH="$REPO_ROOT/packaging/appimagetool-x86_64.AppImage"
    if [ ! -f "$TOOL_PATH" ]; then
        echo "==> Downloading appimagetool …"
        curl -fsSL -o "$TOOL_PATH" \
            "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
        chmod +x "$TOOL_PATH"
    fi
    APPIMAGETOOL="$TOOL_PATH"
fi
echo "==> appimagetool : $APPIMAGETOOL"

# ── Build AppDir structure ────────────────────────────────────────────────────
echo "==> Building AppDir …"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
mkdir -p "$APPDIR/usr/share/applications"
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"

# Copy bundle
cp -a "$BUNDLE_DIR/." "$APPDIR/usr/bin/"

# Desktop entry  (required by AppImage spec)
cat > "$APPDIR/usr/share/applications/lt2thermometry.desktop" <<'DESKTOP'
[Desktop Entry]
Name=LT2 Thermometry
Comment=Luminescence thermometry benchmark tool
Exec=lt2thermometry
Icon=lt2thermometry
Type=Application
Categories=Science;Education;
Terminal=false
StartupNotify=true
DESKTOP

# Copy a PNG icon (use one of the bundled group images as a stand-in)
ICON_SRC="$BUNDLE_DIR/assets/group_e.png"
if [ -f "$ICON_SRC" ]; then
    cp "$ICON_SRC" "$APPDIR/usr/share/icons/hicolor/256x256/apps/lt2thermometry.png"
else
    # Create a minimal 1x1 PNG placeholder if no icon found
    python3 -c "
import base64, sys
PNG_1x1=b'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
open('$APPDIR/usr/share/icons/hicolor/256x256/apps/lt2thermometry.png','wb').write(base64.b64decode(PNG_1x1))
"
fi

# AppRun launcher script
cat > "$APPDIR/AppRun" <<'APPRUN'
#!/usr/bin/env bash
# AppRun — entry point executed by the AppImage runtime.
HERE="$(dirname "$(readlink -f "${0}")")"
export LD_LIBRARY_PATH="${HERE}/usr/bin:${LD_LIBRARY_PATH:-}"
# Qt needs to find its platform plugins inside the bundle.
export QT_PLUGIN_PATH="${HERE}/usr/bin/PySide6/Qt/plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_PLUGIN_PATH}/platforms"
exec "${HERE}/usr/bin/LT2 Thermometry" "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

# Symlink top-level .desktop and icon (AppImage spec requirement)
ln -sf "usr/share/applications/lt2thermometry.desktop" "$APPDIR/lt2thermometry.desktop"
ln -sf "usr/share/icons/hicolor/256x256/apps/lt2thermometry.png" "$APPDIR/lt2thermometry.png"

# ── Run appimagetool ──────────────────────────────────────────────────────────
echo "==> Running appimagetool …"
ARCH=x86_64 "$APPIMAGETOOL" "$APPDIR" "$APPIMAGE_OUT"

chmod +x "$APPIMAGE_OUT"
APP_SIZE=$(du -sh "$APPIMAGE_OUT" | cut -f1)
echo ""
echo "==> Done: $APPIMAGE_OUT  ($APP_SIZE)"
echo ""
echo "    Users can run it directly (no install required):"
echo "      chmod +x LT2_Thermometry_${APP_VERSION}_linux_x86_64.AppImage"
echo "      ./LT2_Thermometry_${APP_VERSION}_linux_x86_64.AppImage"
echo ""
echo "    Note: libfuse2 must be installed on the user's machine."
echo "      Ubuntu/Debian: sudo apt-get install libfuse2"
echo "      Fedora/RHEL:   sudo dnf install fuse"

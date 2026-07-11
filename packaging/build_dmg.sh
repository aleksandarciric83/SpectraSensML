#!/usr/bin/env bash
# build_dmg.sh — wrap the macOS .app bundle into a drag-to-Applications DMG.
#
# Prerequisites (macOS only):
#   - Run  bash packaging/build_pyinstaller.sh  first.
#     The .app must exist at  dist/SpectraSensML.app/
#   - Optionally install create-dmg for a nicer background/layout:
#       brew install create-dmg
#   - Otherwise the script falls back to plain hdiutil.
#
# Usage (from repo root Yb/):
#   bash packaging/build_dmg.sh
#
# Output:
#   packaging/Output/SpectraSensML_1.0_macos.dmg

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="SpectraSensML"
APP_VERSION="1.0"
APP_BUNDLE="$REPO_ROOT/dist/${APP_NAME}.app"
OUTPUT_DIR="$REPO_ROOT/packaging/Output"
DMG_NAME="SpectraSensML_${APP_VERSION}_macos"
DMG_PATH="$OUTPUT_DIR/${DMG_NAME}.dmg"
STAGING_DIR="$(mktemp -d)"

echo "==> Repo root : $REPO_ROOT"
echo "==> .app      : $APP_BUNDLE"
echo "==> DMG out   : $DMG_PATH"

if [ ! -d "$APP_BUNDLE" ]; then
    echo "ERROR: $APP_BUNDLE not found. Run build_pyinstaller.sh first."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── Build DMG via hdiutil (no AppleScript/Finder dependency) ─────────────────
RW_DMG="$STAGING_DIR/rw.dmg"
VOLNAME_SAFE="SpectraSensML${APP_VERSION//./}"   # no spaces for safe mounting

APP_SIZE_KB=$(du -sk "$APP_BUNDLE" | cut -f1)
DMG_SIZE_MB=$(( (APP_SIZE_KB + 204800) / 1024 ))  # +200 MB slack

echo "==> App size: ${APP_SIZE_KB}KB — creating ${DMG_SIZE_MB}MB scratch DMG …"
hdiutil create -size "${DMG_SIZE_MB}m" -fs HFS+ -volname "$VOLNAME_SAFE" -o "$RW_DMG" -quiet

echo "==> Mounting scratch DMG …"
MOUNT_INFO=$(hdiutil attach "$RW_DMG" -readwrite -nobrowse -plist)
MOUNT_DIR=$(echo "$MOUNT_INFO" | python3 -c "
import sys, plistlib
p = plistlib.loads(sys.stdin.buffer.read())
for e in p.get('system-entities', []):
    mp = e.get('mount-point', '')
    if mp: print(mp); break
")
echo "==> Mounted at: $MOUNT_DIR"

cp -a "$APP_BUNDLE" "$MOUNT_DIR/"
ln -s /Applications "$MOUNT_DIR/Applications"

hdiutil detach "$MOUNT_DIR" -quiet

echo "==> Compressing to read-only DMG …"
rm -f "$DMG_PATH"
hdiutil convert "$RW_DMG" -format UDZO -imagekey zlib-level=9 -o "$DMG_PATH" -quiet

rm -rf "$STAGING_DIR"

DMG_SIZE=$(du -sh "$DMG_PATH" | cut -f1)
echo ""
echo "==> Done: $DMG_PATH  ($DMG_SIZE)"
echo ""
echo "    This app is ad-hoc signed only (not notarized). On recent macOS"
echo "    (Ventura+), the reliable way for users to open it after copying"
echo "    from the DMG is to clear the quarantine flag in Terminal:"
    echo "        xattr -cr \"/Applications/SpectraSensML.app\""
echo "    Right-click → Open sometimes works too, but many macOS versions"
echo "    no longer offer an 'Open Anyway' button for quarantined,"
echo "    unnotarized apps. For signed/notarized builds see Apple"
echo "    Developer documentation."

LT2 Thermometry v1.0 — macOS Installer
=======================================

File: LT2_Thermometry_1.0_macos.dmg  (402 MB)

HOW TO INSTALL
──────────────
1. Double-click  LT2_Thermometry_1.0_macos.dmg  to open it.
2. In the window that opens, drag "LT2 Thermometry.app" to the
   "Applications" shortcut on the right side.
3. Eject the DMG (drag it to Trash, or right-click → Eject).
4. Open Applications and double-click "LT2 Thermometry" to launch.

FIRST LAUNCH (unsigned app)
────────────────────────────
Because this app is only ad-hoc signed and not notarized with Apple,
macOS Gatekeeper will block it on the first open. If the DMG was
copied over a network, USB drive, AirDrop, or browser download, macOS
attaches a "quarantine" flag that makes Gatekeeper show a dialog with
no "Open Anyway" button at all, just "Move to Trash" / "Done".

Recommended fix (works on all macOS versions, incl. Ventura/Sonoma/
Sequoia and later, where the old right-click trick often no longer
appears): open Terminal (Applications → Utilities → Terminal) and run:

    xattr -cr "/Applications/LT2 Thermometry.app"

Then double-click the app in Applications — it opens normally.

Alternative (older macOS, or if the option is shown):
  • Right-click "LT2 Thermometry.app" in Applications → Open.
  • Click "Open" in the dialog that appears.
  • If no "Open Anyway" option appears, go to System Settings →
    Privacy & Security, scroll to the "Security" section, and click
    "Open Anyway" next to the LT2 Thermometry entry, then try again.

REQUIREMENTS
────────────
• macOS 12 (Monterey) or later recommended
• Apple Silicon (arm64) or Intel x86_64  — this build is arm64
• No Python installation required — everything is bundled

UNINSTALL
─────────
Drag "LT2 Thermometry.app" from Applications to Trash.

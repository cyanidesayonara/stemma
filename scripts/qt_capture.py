"""Shared helpers for scripts that render the real UI offscreen.

Used by ``render_ui_review.py`` and ``generate_screenshots.py``. Kept in its
own module so the two scripts can import each other's pieces without a
circular import.
"""

from __future__ import annotations

import os
import time

UI_FONTS = (
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
)
# Glyphs Segoe UI lacks, such as the theme toggle's sun, fall back to this.
# A real Windows session does that on its own; offscreen Qt does not.
SYMBOL_FONT = r"C:\Windows\Fonts\seguisym.ttf"
# Real bold and semibold faces, so bold UI labels and Store headlines match a
# Windows session instead of using Qt's synthesized bold.
BOLD_FONTS = (
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\seguisb.ttf",
)


def load_ui_font(app) -> str | None:
    """Register a real UI font: offscreen Qt otherwise draws tofu boxes.

    Returns the family name, or None when no font file was found.
    """
    # Deferred: callers set QT_QPA_PLATFORM before Qt is first imported.
    from PySide6.QtGui import QFont, QFontDatabase

    symbol_family = None
    if os.path.isfile(SYMBOL_FONT):
        fid = QFontDatabase.addApplicationFont(SYMBOL_FONT)
        found = QFontDatabase.applicationFontFamilies(fid)
        symbol_family = found[0] if found else None

    for path in BOLD_FONTS:
        if os.path.isfile(path):
            QFontDatabase.addApplicationFont(path)

    for path in UI_FONTS:
        if not os.path.isfile(path):
            continue
        fid = QFontDatabase.addApplicationFont(path)
        families = QFontDatabase.applicationFontFamilies(fid)
        if families:
            if symbol_family:
                QFont.insertSubstitution(families[0], symbol_family)
            app.setFont(QFont(families[0], 9))
            print(f"  ui font: {families[0]}")
            return families[0]
    print("  WARNING: no UI font found; text may render as boxes")
    return None


def pump(app, seconds: float) -> None:
    """Process events for *seconds* so async work (peaks) can land."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)

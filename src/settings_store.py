"""Where stemma's settings live.

Kept free of heavy imports so ``main.py`` can open settings before the
splash screen, ahead of sounddevice, numpy, and the UI tree.
"""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QSettings

# Points the app at an INI file instead of the per-user native store. The
# test suite and scripts/render_ui_review.py set it so they never read or
# overwrite a real user's session, window, or preference state. Frozen
# (packaged and Store) builds ignore it, so a stray variable can never
# redirect a real user's settings.
SETTINGS_FILE_ENV = "STEMMA_SETTINGS_FILE"


def open_settings() -> QSettings:
    """Open the app's settings store.

    Normally the per-user native store (the registry on Windows). When
    ``STEMMA_SETTINGS_FILE`` is set in a source run, an INI file at that
    path instead.
    """
    path = os.environ.get(SETTINGS_FILE_ENV)
    if path and not getattr(sys, "frozen", False):
        return QSettings(path, QSettings.Format.IniFormat)
    return QSettings("stemma", "stemma")

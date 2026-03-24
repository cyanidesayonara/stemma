"""Application root directory, aware of PyInstaller frozen builds."""

import os
import sys


def app_root() -> str:
    """Return the application root directory.

    In a PyInstaller frozen build, returns ``sys._MEIPASS`` (the temp
    extraction directory where bundled data files are unpacked).  Otherwise
    returns the repository root (parent of the ``src`` package).
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS  # type: ignore[attr-defined]
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

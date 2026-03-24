"""QApplication setup for stemma."""

import ctypes
import os
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from src.library import SongLibrary
from src.model_manager import ModelManager
from src.player import MultiTrackPlayer
from src.ui.main_window import MainWindow
from src.ui.styles import DARK_STYLESHEET
from src.version import __version__

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT_DIR, "data")
_ICON_PATH = os.path.join(_ROOT_DIR, "assets", "icons", "stemma.ico")


def run() -> int:
    """Create and run the stemma application. Returns the exit code."""
    # Tell Windows this is its own app, not a child of python.exe.
    # Without this, Windows uses Python's icon in the taskbar.
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("stemma.app")

    app = QApplication(sys.argv)
    app.setApplicationName("stemma")
    app.setApplicationVersion(__version__)
    app.setStyleSheet(DARK_STYLESHEET)
    if os.path.exists(_ICON_PATH):
        app.setWindowIcon(QIcon(_ICON_PATH))

    library = SongLibrary(data_dir=DATA_DIR)
    player = MultiTrackPlayer()
    model_manager = ModelManager(data_dir=DATA_DIR)

    window = MainWindow(library, player, model_manager)
    window.show()

    return app.exec()

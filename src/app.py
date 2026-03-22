"""QApplication setup for stemma."""

import os
import sys

from PySide6.QtWidgets import QApplication

from src.library import SongLibrary
from src.model_manager import ModelManager
from src.player import MultiTrackPlayer
from src.ui.main_window import MainWindow
from src.ui.styles import DARK_STYLESHEET

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def run() -> int:
    """Create and run the stemma application. Returns the exit code."""
    app = QApplication(sys.argv)
    app.setApplicationName("stemma")
    app.setStyleSheet(DARK_STYLESHEET)

    library = SongLibrary(data_dir=DATA_DIR)
    player = MultiTrackPlayer()
    model_manager = ModelManager(data_dir=DATA_DIR)

    window = MainWindow(library, player, model_manager)
    window.show()

    return app.exec()

"""QApplication setup for stemma."""

import ctypes
import os
import sys

from PySide6.QtCore import QSettings, QSharedMemory
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from src.app_settings import normalize_output_device_setting
from src.data_paths import resolve_data_dir
from src.library import SongLibrary
from src.model_manager import ModelManager
from src.paths import app_root
from src.player import MultiTrackPlayer
from src.ui.main_window import MainWindow
from src.ui.styles import get_colors, get_stylesheet
from src.version import __version__

_ROOT_DIR = app_root()
_ICON_PATH = os.path.join(_ROOT_DIR, "assets", "icons", "stemma.ico")


def run() -> int:
    """Create and run the stemma application. Returns the exit code."""
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("stemma.app")

    app = QApplication(sys.argv)
    app.setApplicationName("stemma")
    app.setApplicationVersion(__version__)

    single_lock = QSharedMemory("stemma_single_instance_v1")
    if not single_lock.create(1):
        QMessageBox.critical(
            None,
            "stemma",
            "Another instance of stemma is already running.",
        )
        return 1

    settings = QSettings("stemma", "stemma")
    theme = settings.value("theme", "dark")
    if theme not in ("dark", "light"):
        theme = "dark"
    app.setStyleSheet(get_stylesheet(theme))

    if os.path.exists(_ICON_PATH):
        app.setWindowIcon(QIcon(_ICON_PATH))

    data_dir = resolve_data_dir(_ROOT_DIR, settings)
    library = SongLibrary(data_dir=data_dir)
    player = MultiTrackPlayer()
    player.set_output_device(normalize_output_device_setting(settings))
    model_manager = ModelManager(data_dir=data_dir)

    window = MainWindow(library, player, model_manager)

    colors = get_colors(theme)
    window.apply_theme(theme, colors)

    window.show()

    return app.exec()

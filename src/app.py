"""QApplication setup for stemma."""

import os

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from src.app_settings import normalize_output_device_setting
from src.data_paths import resolve_data_dir
from src.library import SongLibrary
from src.model_manager import ModelManager
from src.paths import app_root
from src.player import MultiTrackPlayer
from src.ui.main_window import MainWindow
from src.ui.styles import get_colors

_ROOT_DIR = app_root()


def build_and_show(
    qapp: QApplication,
    settings: QSettings,
    theme: str,
    splash: "QWidget",
) -> None:
    """Heavy construction phase called after the splash is already visible.

    Creates the library, player, model manager, and main window, then
    transitions from the splash to the main window.
    """
    qapp.processEvents()

    data_dir = resolve_data_dir(_ROOT_DIR, settings)
    library = SongLibrary(data_dir=data_dir)
    qapp.processEvents()

    player = MultiTrackPlayer()
    player.set_output_device(normalize_output_device_setting(settings))
    qapp.processEvents()

    model_manager = ModelManager(data_dir=data_dir)
    qapp.processEvents()

    window = MainWindow(library, player, model_manager)
    colors = get_colors(theme)
    window.apply_theme(theme, colors)
    qapp.processEvents()

    splash.finish(window)

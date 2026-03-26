"""Entry point for stemma.

Lightweight startup that shows an animated splash screen before importing
the heavy application modules (onnxruntime, librosa, sounddevice, etc.).
"""

import ctypes
import os
import sys
from functools import partial

from PySide6.QtCore import QSettings, QSharedMemory, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from src.paths import app_root
from src.ui.splash_screen import SplashScreen
from src.ui.styles import get_stylesheet
from src.version import __version__

_ROOT_DIR = app_root()
_ICON_PATH = os.path.join(_ROOT_DIR, "assets", "icons", "stemma.ico")
_AUDIO_PATH = os.path.join(_ROOT_DIR, "assets", "audio", "arpeggio.wav")


def _finish_startup(
    qapp: QApplication,
    settings: QSettings,
    theme: str,
    splash: SplashScreen,
) -> None:
    # Deferred import: src.app pulls in sounddevice, numpy, onnxruntime,
    # librosa, and the full UI tree.  Importing here (instead of at the top
    # of the file) keeps the splash visible and animating while those heavy
    # modules load.
    from src.app import build_and_show  # noqa: PLC0415

    qapp.processEvents()
    build_and_show(qapp, settings, theme, splash)


def main() -> int:
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "stemma.app"
        )

    qapp = QApplication(sys.argv)
    qapp.setApplicationName("stemma")
    qapp.setApplicationVersion(__version__)

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
    qapp.setStyleSheet(get_stylesheet(theme))

    if os.path.exists(_ICON_PATH):
        qapp.setWindowIcon(QIcon(_ICON_PATH))

    play_sound = settings.value("startup/play_sound", True, type=bool)
    splash = SplashScreen(
        theme=theme, play_sound=play_sound, audio_path=_AUDIO_PATH
    )
    splash.start()

    QTimer.singleShot(
        0, partial(_finish_startup, qapp, settings, theme, splash)
    )

    _keep = single_lock  # prevent GC; shared memory must live until exit
    assert _keep is not None

    return qapp.exec()


sys.exit(main())

"""Entry point for stemma.

Lightweight startup that shows an animated splash screen before importing
the heavy application modules (onnxruntime, librosa, sounddevice, etc.).
"""

import ctypes
import os
import sys
import traceback
from functools import partial

from PySide6.QtCore import QSettings, QSharedMemory, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from src import crash_log
from src.diagnostics import (
    diagnostics_requested,
    main as diagnostics_main,
)
from src.paths import app_root
from src.settings_store import open_settings
from src.ui.splash_screen import SplashScreen
from src.ui.styles import apply_tooltip_palette, get_stylesheet
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
    try:
        from src.app import build_and_show  # noqa: PLC0415

        qapp.processEvents()
        build_and_show(qapp, settings, theme, splash)
    except Exception as exc:  # noqa: BLE001 - reported to the user below
        # Without this the splash stayed up forever with no message, and the
        # single-instance lock turned every relaunch into "already running".
        report_startup_failure(exc, splash)
        qapp.exit(1)


def report_startup_failure(
    exc: BaseException, splash: SplashScreen | None = None,
) -> None:
    """Log a startup failure, close the splash, and tell the user."""
    crash_log.logger.error(
        "Startup failed", exc_info=(type(exc), exc, exc.__traceback__),
    )
    if sys.stderr is not None:  # source runs keep their console traceback
        traceback.print_exception(exc)
    if splash is not None:
        splash.abort()
    QMessageBox.critical(
        None, "stemma could not start", startup_failure_text(exc),
    )


def startup_failure_text(exc: BaseException) -> str:
    """Plain-language message for an exception raised during startup."""
    if isinstance(exc, OSError) and exc.filename:
        text = (
            "stemma could not use its data folder:\n\n"
            f"{exc.filename}\n\n"
            "Check that the folder exists and that you can write to it. "
            "Windows Security's Controlled folder access can also block "
            "apps from writing to protected folders."
        )
    else:
        text = "stemma ran into a problem while starting and has to close."
    log = crash_log.real_log_path()
    if log:
        text += f"\n\nDetails were saved to:\n{log}"
    return text


def main() -> int:
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "stemma.app"
        )

    crash_log.install()
    crash_log.logger.info("stemma %s starting", __version__)

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

    splash = None
    try:
        settings = open_settings()
        theme = settings.value("theme", "dark")
        if theme not in ("dark", "light"):
            theme = "dark"
        qapp.setStyleSheet(get_stylesheet(theme))
        apply_tooltip_palette(theme)

        if os.path.exists(_ICON_PATH):
            qapp.setWindowIcon(QIcon(_ICON_PATH))

        play_sound = settings.value("startup/play_sound", True, type=bool)
        splash = SplashScreen(
            theme=theme, play_sound=play_sound, audio_path=_AUDIO_PATH
        )
        splash.start()
    except Exception as exc:  # noqa: BLE001 - reported to the user
        report_startup_failure(exc, splash)
        return 1

    QTimer.singleShot(
        0, partial(_finish_startup, qapp, settings, theme, splash)
    )

    _ = single_lock  # prevent GC; shared memory must live until exit

    return qapp.exec()


def _run() -> int:
    """Run diagnostics when requested, otherwise launch the normal GUI."""
    if diagnostics_requested(sys.argv):
        return diagnostics_main(sys.argv)
    return main()


if __name__ == "__main__":
    sys.exit(_run())

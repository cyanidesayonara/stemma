"""Every user-facing failure goes through the readable-error formatter.

Workers map the caught exception (and log the raw text); dialogs format
the message they receive, so no raw ``[WinError ...]`` text reaches a
message box.
"""

import logging
import socket
import threading
import time
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
import soundfile as sf

from src import player as player_module
from src.downloader import DownloadError
from src.exporter import ExportWorker
from src.import_messages import (
    MSG_DISK_FULL,
    MSG_OFFLINE,
    MSG_PERMISSION,
    MSG_RESET,
    MSG_UNREADABLE_AUDIO,
)
from src.mdx_separator import MdxSeparatorWorker
from src.model_manager import ModelDownloader
from src.separator import SeparatorWorker


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _disk_full() -> OSError:
    exc = OSError(28, "There is not enough space on the disk")
    exc.winerror = 112
    return exc


class TestWorkersEmitReadableText:
    def test_model_download_offline(self, tmp_path, caplog):
        dl = ModelDownloader(
            "beat_this", str(tmp_path), url="http://x/m.onnx",
            file_name="m.onnx",
        )
        errors = []
        dl.error.connect(errors.append)
        raw = urllib.error.URLError(socket.gaierror(11001, "getaddrinfo failed"))
        with caplog.at_level(logging.WARNING, logger="stemma"), patch(
            "src.model_manager.urllib.request.urlopen", side_effect=raw,
        ):
            dl.run()
        assert errors == [MSG_OFFLINE]
        assert any("getaddrinfo" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("worker_cls", [SeparatorWorker, MdxSeparatorWorker])
    def test_separation_disk_full(self, tmp_path, worker_cls):
        worker = worker_cls(
            input_path=str(tmp_path / "in.wav"),
            output_dir=str(tmp_path / "out"),
            model_path="unused.onnx",
        )
        errors = []
        worker.error.connect(errors.append)
        with patch.object(worker_cls, "_separate", side_effect=_disk_full()):
            worker.run()
        assert errors == [MSG_DISK_FULL]

    def test_export_permission_denied(self, tmp_path):
        exporter = MagicMock()
        exporter.export_mix.side_effect = PermissionError(
            13, "Permission denied", str(tmp_path / "mix.wav"),
        )
        worker = ExportWorker(exporter, str(tmp_path / "mix.wav"), set(), {})
        errors = []
        worker.error.connect(errors.append)
        worker.run()
        assert errors == [MSG_PERMISSION]

    def test_stem_load_unreadable(self):
        worker = player_module.StemLoadWorker({"vocals": "x.wav"})
        errors = []
        worker.error.connect(errors.append)
        with patch.object(
            player_module, "read_stem_files",
            side_effect=sf.LibsndfileError(0, prefix="Error opening 'x.wav': "),
        ):
            worker.run()
        assert errors == [MSG_UNREADABLE_AUDIO]

    def test_youtube_download_reset(self, tmp_path):
        from src.ui import import_dialog

        def fail(*_args, **_kwargs):
            try:
                raise ConnectionResetError(10054, "forcibly closed")
            except ConnectionResetError as exc:
                raise DownloadError(str(exc)) from exc

        worker = import_dialog._DownloadWorker(
            "https://youtu.be/abc", str(tmp_path / "a.mp3"),
        )
        errors = []
        worker.error.connect(errors.append)
        with patch.object(import_dialog, "download_audio", side_effect=fail):
            worker.run()
        assert errors == [MSG_RESET]

    def test_youtube_download_retries_stop_on_interruption(self, tmp_path):
        from src.ui import import_dialog

        started = threading.Event()
        seen = []

        def backing_off(*_args, should_cancel, **_kwargs):
            # Stands in for download_audio waiting out a retry backoff.
            seen.append(should_cancel())
            started.set()
            deadline = time.monotonic() + 5.0
            while not should_cancel() and time.monotonic() < deadline:
                time.sleep(0.01)
            seen.append(should_cancel())
            raise DownloadError("Download cancelled")

        worker = import_dialog._DownloadWorker(
            "https://youtu.be/abc", str(tmp_path / "a.mp3"),
        )
        with patch.object(
            import_dialog, "download_audio", side_effect=backing_off,
        ):
            worker.start()
            assert started.wait(5.0)
            worker.requestInterruption()  # what the dialog's reject() does
            assert worker.wait(5000)
        assert seen == [False, True]


class TestDialogsFormatMessages:
    """Dialogs receiving a raw string still show readable text."""

    def test_separation_failed_dialog(self, app):
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        stub._library.get_song.return_value = None
        with patch("src.ui.main_window.QMessageBox") as mb:
            MainWindow._on_separation_failed(
                stub, "s1", "[WinError 112] There is not enough space",
            )
        text = mb.warning.call_args.args[2]
        assert "WinError" not in text
        assert MSG_DISK_FULL in text

    def test_export_failed_dialog(self, app):
        from src.ui.main_window import MainWindow

        with patch("src.ui.main_window.QMessageBox") as mb:
            MainWindow._on_export_error(
                MagicMock(), "[Errno 13] Permission denied: 'C:\\\\mix.wav'",
            )
        text = mb.critical.call_args.args[2]
        assert "Errno" not in text
        assert MSG_PERMISSION in text

    def test_load_song_dialog(self, app):
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        worker = MagicMock()
        stub._stem_load_generation = worker.generation
        stub._loading_song_id = worker.song_id
        stub._stem_load_worker = worker
        with patch("src.ui.main_window.QMessageBox") as mb:
            MainWindow._handle_stem_load_error(
                stub, worker,
                "Error opening 'x.mp3': File does not exist or is not a "
                "regular file (possibly a pipe?).",
            )
        text = mb.warning.call_args.args[2]
        assert "possibly a pipe" not in text
        assert MSG_UNREADABLE_AUDIO in text

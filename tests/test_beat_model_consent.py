"""The beat model asks before it downloads, and offline it fails once (#205).

Separation models ask before their first download (#200). The beat model
was the one download that still started on its own, and offline it
retried with a 30 s timeout on every song load (#188).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from src.model_manager import ModelManager
from src.player import MultiTrackPlayer
from src.ui.player_controls import PlayerControls


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def player():
    p = MultiTrackPlayer()
    yield p
    p.shutdown(wait_ms=5000)


@pytest.fixture
def controls(qapp, player):
    ctrl = PlayerControls(player)
    player._stems = {"vocals": np.zeros((64, 2), dtype=np.float32)}
    yield ctrl
    ctrl._cleanup_peak_thread()
    ctrl.setParent(None)
    ctrl.deleteLater()
    qapp.processEvents()


class _FakeDownloader(QObject):
    progress = Signal(int, str)
    download_complete = Signal(str)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.started = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        pass

    def isRunning(self) -> bool:
        return False


def _manager(downloads: list) -> MagicMock:
    """A manager whose beat model is missing; records each downloader."""
    manager = MagicMock()
    manager.beat_model_path.return_value = "beat_this.onnx"
    manager.is_beat_model_downloaded.return_value = False
    manager.download_size_mb.return_value = 79
    manager.beat_model_host.return_value = "raw.githubusercontent.com"

    def make():
        downloader = _FakeDownloader()
        downloads.append(downloader)
        return downloader

    manager.download_beat_model.side_effect = make
    return manager


class TestBeatModelConsent:
    def test_asks_before_the_first_download(self, controls):
        downloads = []
        controls.set_model_manager(_manager(downloads))

        with patch.object(
            controls, "_ask_beat_model_download", return_value=True,
        ) as ask, patch.object(controls, "_run_detection"):
            controls.start_detection()

        ask.assert_called_once()
        assert len(downloads) == 1
        assert downloads[0].started

    def test_no_runs_the_fallback_without_downloading(self, controls):
        downloads = []
        controls.set_model_manager(_manager(downloads))

        with patch.object(
            controls, "_ask_beat_model_download", return_value=False,
        ), patch.object(controls, "_run_detection") as run:
            controls.start_detection(1.0, 2.0)

        assert downloads == []
        run.assert_called_once()
        assert run.call_args.args[:2] == (1.0, 2.0)
        assert controls._beat_model_path is None

    def test_no_is_remembered_for_the_session(self, controls):
        downloads = []
        controls.set_model_manager(_manager(downloads))

        with patch.object(
            controls, "_ask_beat_model_download", return_value=False,
        ) as ask, patch.object(controls, "_run_detection") as run:
            controls.start_detection()
            controls.start_detection()

        ask.assert_called_once()
        assert downloads == []
        assert run.call_count == 2

    def test_a_failed_download_is_not_retried_this_session(self, controls):
        downloads = []
        controls.set_model_manager(_manager(downloads))

        with patch.object(
            controls, "_ask_beat_model_download", return_value=True,
        ) as ask, patch.object(controls, "_run_detection") as run:
            controls.start_detection()
            downloads[0].error.emit("No connection")
            controls.start_detection()

        ask.assert_called_once()
        assert len(downloads) == 1
        # The failed attempt and the next song both use the fallback.
        assert run.call_count == 2

    def test_yes_is_not_asked_again_after_a_download(self, controls):
        downloads = []
        manager = _manager(downloads)
        controls.set_model_manager(manager)

        with patch.object(
            controls, "_ask_beat_model_download", return_value=True,
        ) as ask, patch.object(controls, "_run_detection"):
            controls.start_detection()
            manager.is_beat_model_downloaded.return_value = True
            downloads[0].download_complete.emit("beat_this.onnx")
            controls.start_detection()

        ask.assert_called_once()

    def test_progress_shows_in_the_detection_status(self, controls):
        downloads = []
        controls.set_model_manager(_manager(downloads))

        with patch.object(
            controls, "_ask_beat_model_download", return_value=True,
        ), patch.object(controls, "_run_detection"), patch.object(
            controls._song_info_bar, "show_detection_status",
        ) as status:
            controls.start_detection()
            downloads[0].progress.emit(
                42, "Downloading the beat detection model... 33 of 79 MB",
            )

        texts = [" ".join(map(str, c.args)) for c in status.call_args_list]
        assert any("42%" in text for text in texts)

    def test_prompt_names_size_host_and_purpose(self, controls):
        controls.set_model_manager(_manager([]))

        with patch(
            "src.ui.player_controls.QMessageBox.question",
            return_value=QMessageBox.StandardButton.No,
        ) as question:
            assert controls._ask_beat_model_download() is False

        text = question.call_args.args[2]
        assert "79 MB" in text
        assert "raw.githubusercontent.com" in text
        assert "downbeat" in text.lower()
        assert "tempo and key" in text.lower()


def test_manager_reports_the_beat_model_host(tmp_path):
    manager = ModelManager(str(tmp_path))
    assert manager.beat_model_host() == "raw.githubusercontent.com"
    assert manager.download_size_mb("beat_this") == 79

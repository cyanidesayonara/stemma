"""Quit during separation, cancel checks, and recording without input.

Covers the "Quit and record" part of issue #185.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from src.mdx_separator import MdxSeparatorWorker
from src.player import MultiTrackPlayer
from src.post_processing import soft_gate, wiener_filter
from src.separation_queue import SeparationJob, SeparationQueue
from src.separation_state import COMPLETION_MARKER
from src.separator import SeparatorWorker


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _stems(n_stems=2, seconds=0.5):
    rng = np.random.default_rng(0)
    frames = int(44100 * seconds)
    return rng.standard_normal((n_stems, 2, frames)).astype(np.float32) * 0.1


class TestPostProcessingCancel:
    def test_wiener_filter_stops_when_cancelled(self):
        with pytest.raises(InterruptedError):
            wiener_filter(_stems(), should_cancel=lambda: True)

    def test_soft_gate_stops_when_cancelled(self):
        with pytest.raises(InterruptedError):
            soft_gate(_stems(), should_cancel=lambda: True)

    def test_filters_run_normally_without_cancel(self):
        stems = _stems()
        assert wiener_filter(stems, should_cancel=lambda: False).shape == (
            stems.shape
        )


class TestWorkerCancelChecks:
    def _demucs(self, tmp_path):
        return SeparatorWorker(
            input_path=str(tmp_path / "in.wav"),
            output_dir=str(tmp_path / "out"),
            model_path="unused.onnx",
        )

    def test_demucs_post_process_checks_cancel(self, tmp_path):
        worker = self._demucs(tmp_path)
        worker.cancel()
        with pytest.raises(InterruptedError):
            worker._post_process(_stems(n_stems=4))

    def test_demucs_save_stops_and_writes_no_marker(self, tmp_path):
        worker = self._demucs(tmp_path)
        worker.cancel()
        with pytest.raises(InterruptedError):
            worker._save_stems(_stems(n_stems=4))
        out = tmp_path / "out"
        assert not (out / COMPLETION_MARKER).exists()
        assert not any(p.suffix == ".wav" for p in out.iterdir())

    def test_mdx_save_stops_and_writes_no_marker(self, tmp_path):
        worker = MdxSeparatorWorker(
            input_path=str(tmp_path / "in.wav"),
            output_dir=str(tmp_path / "out"),
            model_path="unused.onnx",
        )
        worker.cancel()
        audio = _stems(n_stems=1)[0]
        with pytest.raises(InterruptedError):
            worker._save(audio, audio)
        out = tmp_path / "out"
        assert not (out / COMPLETION_MARKER).exists()

    def test_cancel_during_save_reports_cancelled(self, tmp_path):
        """Cancelling while stems are written ends as a cancellation."""
        sf.write(
            str(tmp_path / "in.wav"),
            np.zeros((4410, 2), dtype=np.float32), 44100,
        )
        worker = self._demucs(tmp_path)
        errors, finished = [], []
        worker.error.connect(errors.append)
        worker.finished.connect(finished.append)

        def infer(audio, _session):
            return np.zeros((4, 2, audio.shape[1]), dtype=np.float32)

        written = []
        real_write = sf.write

        def write(path, *args, **kwargs):
            written.append(path)
            real_write(path, *args, **kwargs)
            worker.cancel()  # user cancels once the first stem is on disk

        with patch.object(worker, "_create_session"), patch.object(
            worker, "_run_segmented_inference", side_effect=infer,
        ), patch.object(
            worker, "_post_process", side_effect=lambda stems: stems,
        ), patch("src.separator.sf.write", side_effect=write):
            worker.run()
        assert finished == []
        assert errors and "cancelled" in errors[0].lower()
        assert len(written) == 1  # later stems were not written
        assert not (tmp_path / "out" / COMPLETION_MARKER).exists()


class _StubbornWorker(QThread):
    """Ignores cancel for a while, like one ONNX segment still running."""

    progress = Signal(int, str)
    finished = Signal(dict)  # shadows QThread.finished, like the real ones
    error = Signal(str)

    def __init__(self, seconds):
        super().__init__()
        self._seconds = seconds
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        time.sleep(self._seconds)
        self.error.emit("Separation cancelled by user.")


class TestQueueShutdownKeepsWorker:
    def test_shutdown_waits_for_worker_past_the_timeout(self, app, tmp_path):
        queue = SeparationQueue()
        worker = _StubbornWorker(0.4)
        with patch.object(queue, "_make_worker", return_value=worker):
            queue.enqueue(SeparationJob(
                "s1", "in.wav", str(tmp_path), "m.onnx", "htdemucs",
            ))
        assert worker.isRunning()
        assert queue.shutdown(wait_ms=5000)
        assert worker.cancelled
        assert not worker.isRunning()

    def test_stuck_worker_is_kept_and_reported(self, app, tmp_path):
        """A stage that cannot be interrupted must not hang the close."""
        queue = SeparationQueue()
        worker = _StubbornWorker(0.5)
        with patch.object(queue, "_make_worker", return_value=worker):
            queue.enqueue(SeparationJob(
                "s1", "in.wav", str(tmp_path), "m.onnx", "htdemucs",
            ))
        assert queue.shutdown(wait_ms=50) is False
        # Still referenced, so the running QThread is not destroyed.
        assert queue._stuck_worker is worker
        worker.wait()

    def test_close_ends_the_process_when_separation_is_stuck(self, app):
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        stub._confirm_quit.return_value = True
        stub._export_worker = None
        stub._separation_queue.shutdown.return_value = False
        with patch(
            "src.ui.main_window.os._exit", side_effect=SystemExit,
        ) as exit_, patch("src.ui.main_window.logging.shutdown"):
            with pytest.raises(SystemExit):
                MainWindow.closeEvent(stub, MagicMock())
        stub.hide.assert_called_once()
        stub._settings.sync.assert_called_once()
        exit_.assert_called_once_with(0)


class TestConfirmQuit:
    def _stub(self, pending):
        stub = MagicMock()
        stub._separation_queue.pending_count = pending
        return stub

    def test_idle_quits_without_asking(self, app):
        from src.ui.main_window import MainWindow

        with patch("src.ui.main_window.QMessageBox.question") as ask:
            assert MainWindow._confirm_quit(self._stub(0))
        ask.assert_not_called()

    @pytest.mark.parametrize("answer, expected", [
        (QMessageBox.StandardButton.Yes, True),
        (QMessageBox.StandardButton.No, False),
    ])
    def test_separating_asks(self, app, answer, expected):
        from src.ui.main_window import MainWindow

        with patch(
            "src.ui.main_window.QMessageBox.question", return_value=answer,
        ) as ask:
            assert MainWindow._confirm_quit(self._stub(1)) is expected
        assert "still separating" in ask.call_args.args[2]
        assert "discard" in ask.call_args.args[2]

    def test_declining_keeps_the_window_open(self, app):
        from src.ui.main_window import MainWindow

        stub = self._stub(1)
        stub._confirm_quit.return_value = False
        event = MagicMock()
        MainWindow.closeEvent(stub, event)
        event.ignore.assert_called_once()
        stub._separation_queue.shutdown.assert_not_called()
        stub._save_session.assert_not_called()


@pytest.fixture
def armed_player(tmp_path):
    path = tmp_path / "vocals.wav"
    sf.write(str(path), np.full((44100, 2), 0.1, dtype=np.float32), 44100)
    player = MultiTrackPlayer()
    player.load_stems({"vocals": str(path)})
    player.arm_recording(True)
    return player


class TestRecordingWithoutInput:
    def _play(self, player, *, default=(-1, 1), stream_error=None):
        unavailable, failed = [], []
        player.recording_unavailable.connect(unavailable.append)
        player.playback_failed.connect(failed.append)
        with patch("src.player.sd.Stream") as duplex, patch(
            "src.player.sd.OutputStream",
        ) as output, patch("src.player.sd.default") as sd_default, patch(
            "src.player.sd.query_devices",
            return_value={"max_input_channels": 0},
        ):
            sd_default.device = default
            if stream_error is not None:
                duplex.side_effect = stream_error
            player.play()
        player.stop()
        return unavailable, failed, duplex, output

    def test_no_input_device_disarms_and_plays(self, armed_player):
        unavailable, failed, duplex, output = self._play(armed_player)
        assert not armed_player.recording_armed
        assert unavailable and "input" in unavailable[0].lower()
        assert "speakers or headphones" not in unavailable[0]
        assert failed == []
        duplex.assert_not_called()
        output.assert_called_once()

    def test_duplex_open_failure_disarms_and_plays(self, armed_player):
        unavailable, failed, _duplex, output = self._play(
            armed_player, default=(0, 1),
            stream_error=sd.PortAudioError("Invalid number of channels"),
        )
        assert not armed_player.recording_armed
        assert unavailable
        assert failed == []
        output.assert_called_once()

    def test_next_play_does_not_fail_again(self, armed_player):
        self._play(armed_player)
        unavailable, failed, duplex, _output = self._play(armed_player)
        assert unavailable == []
        assert failed == []
        duplex.assert_not_called()

    def test_main_window_unchecks_record_and_explains(self, app):
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        with patch("src.ui.main_window.QMessageBox.warning") as warn:
            MainWindow._on_recording_unavailable(stub, "No input device.")
        stub._player_controls._record_btn.setChecked.assert_called_with(False)
        assert warn.call_args.args[2] == "No input device."


class TestPausedTakeSurvives:
    def test_paused_take_is_kept_when_the_input_goes(self, armed_player,
                                                     tmp_path):
        """Record, pause, unplug the mic, Play: the take is still saved."""
        armed_player.set_recording_song_dir(str(tmp_path))
        armed_player._allocate_recording_buffer()
        armed_player._recording_buffer[:1000] = 0.2
        armed_player._recording_frames_captured = 1000
        saved, unavailable = [], []
        armed_player.recording_saved.connect(saved.append)
        armed_player.recording_unavailable.connect(unavailable.append)
        with patch("src.player.sd.OutputStream"), patch(
            "src.player.sd.default",
        ) as sd_default:
            sd_default.device = (-1, 1)
            armed_player.play()
        armed_player.stop()
        assert not armed_player.recording_armed
        assert unavailable and "kept" in unavailable[0]
        assert len(saved) == 1
        assert sf.read(saved[0])[0][:1000].max() == pytest.approx(0.2, abs=1e-3)

    def test_duplex_start_failure_plays_without_recording(self, armed_player):
        unavailable, failed = [], []
        armed_player.recording_unavailable.connect(unavailable.append)
        armed_player.playback_failed.connect(failed.append)
        with patch("src.player.sd.Stream") as duplex, patch(
            "src.player.sd.OutputStream",
        ) as output, patch("src.player.sd.default") as sd_default, patch(
            "src.player.sd.query_devices",
            return_value={"max_input_channels": 2},
        ):
            sd_default.device = (0, 1)
            duplex.return_value.start.side_effect = sd.PortAudioError("busy")
            armed_player.play()
        armed_player.stop()
        assert failed == []
        assert unavailable
        assert not armed_player.recording_armed
        output.return_value.start.assert_called_once()

    def test_unavailable_is_emitted_after_playback_starts(self, armed_player):
        states = []
        armed_player.recording_unavailable.connect(
            lambda _msg: states.append(armed_player.is_playing)
        )
        with patch("src.player.sd.OutputStream"), patch(
            "src.player.sd.default",
        ) as sd_default:
            sd_default.device = (-1, 1)
            armed_player.play()
        armed_player.stop()
        assert states == [True]


class TestRecordingSaveFailure:
    def test_failed_write_keeps_the_take_and_explains(self, armed_player,
                                                      tmp_path):
        armed_player._allocate_recording_buffer()
        armed_player._recording_frames_captured = 100
        failed = []
        armed_player.recording_save_failed.connect(failed.append)
        with patch(
            "src.player.sf.write",
            side_effect=OSError(28, "No space left on device"),
        ):
            assert armed_player.save_recording(str(tmp_path)) is None
        assert failed and "could not be saved" in failed[0]
        assert "space" in failed[0].lower()
        assert armed_player._recording_buffer is not None
        assert not list(tmp_path.glob("recording_take*.wav"))
        # Once the problem is fixed, the retry saves it.
        assert armed_player.save_recording(str(tmp_path)) is not None

    def test_main_window_shows_the_save_failure(self, app):
        from src.ui.main_window import MainWindow

        with patch("src.ui.main_window.QMessageBox.warning") as warn:
            MainWindow._on_recording_save_failed(MagicMock(), "Disk full.")
        assert warn.call_args.args[2] == "Disk full."

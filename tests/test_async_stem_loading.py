"""Tests for generation-safe asynchronous song loading."""

from concurrent.futures import Future
import os
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QObject, QSettings, QThread, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from src import player as player_module
from src.beat_detector import DetectionResult
from src.library import Song
from src.ui import main_window as main_window_module
from src.ui import player_controls as player_controls_module
from src.ui.library_panel import REPEAT_ALL


def _write_wav(path, value: float, sample_rate: int = 44100) -> None:
    data = np.full((64, 2), value, dtype=np.float32)
    sf.write(str(path), data, sample_rate)


class _FakeStemLoadWorker(QObject):
    completed = Signal(dict, int)
    error = Signal(str)
    finished = Signal()

    def __init__(self, stem_paths, **metadata) -> None:
        super().__init__()
        self.stem_paths = dict(stem_paths)
        self.generation = metadata.get("generation", 0)
        self.song_id = metadata.get("song_id", "")
        self.source_stem_names = metadata.get("source_stem_names", ())
        self.started = False
        self.running = False
        self.wait_calls = 0

    def start(self) -> None:
        self.started = True
        self.running = True

    def isRunning(self) -> bool:
        return self.running

    def wait(self) -> None:
        self.wait_calls += 1
        self.running = False

    def setParent(self, _parent) -> None:
        pass

    def deleteLater(self) -> None:
        pass


class _FakeDetectionWorker(QObject):
    completed = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.started = False
        self.running = False

    def start(self) -> None:
        self.started = True
        self.running = True

    def isRunning(self) -> bool:
        return self.running

    def wait(self) -> None:
        self.running = False


class _FakeBeatDownloader(QObject):
    download_complete = Signal(str)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.started = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def isRunning(self) -> bool:
        return False


def _wait_until(qapp, predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.001)
    qapp.processEvents()
    return bool(predicate())


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _QuietMainWindow(main_window_module.MainWindow):
    """MainWindow with the startup library prune disabled.

    This replaces ``patch.object(MainWindow, "_prune_incomplete_songs")``.
    Mutating a method on a QWidget *subclass* and then instantiating that
    class segfaults PySide6 6.10.2 on Python 3.14 -- the interpreter
    AGENTS.md documents for local development -- with an access violation
    in the next Qt allocation. CI stayed green because it runs 3.12, so
    this module was uncollectable locally while looking healthy upstream.

    Verified: patching QTimer.singleShot (a Qt built-in) is fine; patching
    MainWindow._prune_incomplete_songs or PlayerControls.start_detection
    is what crashes. Subclassing leaves the base class untouched.
    """

    def _prune_incomplete_songs(self) -> None:
        pass


@pytest.fixture
def window(qapp, tmp_path):
    songs = []
    for index, song_id in enumerate(("a", "b"), start=1):
        stems_dir = tmp_path / song_id
        stems_dir.mkdir()
        _write_wav(stems_dir / "vocals.wav", index / 10)
        songs.append(
            Song(
                id=song_id,
                title=f"Song {song_id.upper()}",
                artist="Artist",
                original_path="",
                stems_path=str(stems_dir),
                model_used="htdemucs",
                date_added="",
            )
        )

    library = MagicMock()
    library.songs = songs
    library.get_song.side_effect = lambda song_id: next(
        (song for song in songs if song.id == song_id), None,
    )
    settings = QSettings("stemma", "stemma")
    settings.clear()
    player = player_module.MultiTrackPlayer()
    # QTimer.singleShot is a Qt built-in and patches safely; the two
    # subclass-method patches this used to carry do not (see
    # _QuietMainWindow), so they are replaced by a subclass override and
    # an instance attribute.
    # Suppressing the deferred startup callbacks (QTimer.singleShot) is
    # enough on its own: nothing calls start_detection during
    # construction, so the original class-level patch of it was
    # unnecessary -- and leaving a stub in place would break the tests
    # below that call the real start_detection.
    with patch.object(main_window_module.QTimer, "singleShot"):
        result = _QuietMainWindow(library, player, MagicMock())

    yield result

    if hasattr(result, "_shutdown_stem_loads"):
        result._shutdown_stem_loads()
    result._separation_queue.shutdown(0)
    result._player_controls.shutdown()
    player.shutdown(wait_ms=5000)
    result.setParent(None)
    result.deleteLater()
    qapp.processEvents()
    settings.clear()


class TestStemLoadWorker:
    def test_emits_complete_result_after_all_files_load(self, tmp_path):
        vocals = tmp_path / "vocals.wav"
        drums = tmp_path / "drums.wav"
        _write_wav(vocals, 0.1)
        _write_wav(drums, 0.2)
        results = []
        errors = []

        worker = player_module.StemLoadWorker(
            {"vocals": str(vocals), "drums": str(drums)}
        )
        worker.completed.connect(
            lambda stems, sample_rate: results.append((stems, sample_rate))
        )
        worker.error.connect(errors.append)

        worker.run()

        assert errors == []
        assert len(results) == 1
        stems, sample_rate = results[0]
        assert sample_rate == 44100
        assert set(stems) == {"vocals", "drums"}
        assert stems["vocals"].shape == (64, 2)
        assert np.allclose(stems["drums"], 0.2, atol=1e-4)

    def test_reports_error_without_partial_result(self, tmp_path):
        vocals = tmp_path / "vocals.wav"
        drums = tmp_path / "drums.wav"
        _write_wav(vocals, 0.1, sample_rate=44100)
        _write_wav(drums, 0.2, sample_rate=48000)
        results = []
        errors = []

        worker = player_module.StemLoadWorker(
            {"vocals": str(vocals), "drums": str(drums)}
        )
        worker.completed.connect(
            lambda stems, sample_rate: results.append((stems, sample_rate))
        )
        worker.error.connect(errors.append)

        worker.run()

        assert results == []
        assert len(errors) == 1
        assert "Sample rate mismatch" in errors[0]

    def test_synchronous_load_failure_preserves_previous_song(
        self, tmp_path,
    ):
        old = tmp_path / "old.wav"
        new_a = tmp_path / "new-a.wav"
        new_b = tmp_path / "new-b.wav"
        _write_wav(old, 0.4, sample_rate=44100)
        _write_wav(new_a, 0.1, sample_rate=44100)
        _write_wav(new_b, 0.2, sample_rate=48000)
        player = player_module.MultiTrackPlayer()
        player.load_stems({"vocals": str(old)})

        with pytest.raises(ValueError, match="Sample rate mismatch"):
            player.load_stems({
                "vocals": str(new_a),
                "drums": str(new_b),
            })

        assert set(player.stems) == {"vocals"}
        assert np.allclose(player.stems["vocals"], 0.4, atol=1e-4)
        assert player.sample_rate == 44100


class TestMainWindowAsyncLoading:
    def test_real_worker_callbacks_run_on_main_thread(
        self, window, qapp,
    ):
        applied_threads = []
        original_apply = window._player.apply_loaded_stems

        def record_apply(stems, sample_rate):
            applied_threads.append(QThread.currentThread())
            original_apply(stems, sample_rate)

        with patch.object(
            window._player, "apply_loaded_stems",
            side_effect=record_apply,
        ), patch.object(
            main_window_module.PlayerControls, "start_detection",
        ):
            window._on_song_selected("a")
            worker = window._stem_load_worker
            assert worker is not None
            assert _wait_until(
                qapp,
                lambda: (
                    window._current_song_id == "a"
                    and window._stem_load_worker is None
                ),
            )

        assert applied_threads == [window.thread()]
        assert window.thread() is qapp.thread()
        assert not worker.isRunning()
        assert window._orphaned_stem_load_workers == []

    def test_selection_returns_before_player_arrays_are_applied(self, window):
        workers = []

        def make_worker(paths):
            worker = _FakeStemLoadWorker(paths)
            workers.append(worker)
            return worker

        with patch.object(
            main_window_module, "StemLoadWorker",
            side_effect=make_worker, create=True,
        ), patch.object(window._player, "load_stems") as synchronous_load:
            window._on_song_selected("a")

        synchronous_load.assert_not_called()
        assert len(workers) == 1
        assert workers[0].started
        assert not window._player.has_stems

        arrays, sample_rate = player_module.read_stem_files(
            workers[0].stem_paths,
        )
        workers[0].completed.emit(arrays, sample_rate)

        assert window._current_song_id == "a"
        assert window._player.has_stems

    def test_navigation_plays_only_after_matching_load_succeeds(self, window):
        workers = []

        def make_worker(paths):
            worker = _FakeStemLoadWorker(paths)
            workers.append(worker)
            return worker

        window._current_song_id = "a"
        window._library_panel.set_repeat_mode(REPEAT_ALL)
        with patch.object(
            main_window_module, "StemLoadWorker",
            side_effect=make_worker, create=True,
        ), patch.object(window._player, "play") as play:
            window._advance_song(1)
            play.assert_not_called()

            arrays, sample_rate = player_module.read_stem_files(
                workers[0].stem_paths,
            )
            workers[0].completed.emit(arrays, sample_rate)

        play.assert_called_once_with()
        assert window._current_song_id == "b"

    def test_rapid_selection_discards_stale_completion(self, window):
        workers = []

        def make_worker(paths):
            worker = _FakeStemLoadWorker(paths)
            workers.append(worker)
            return worker

        with patch.object(
            main_window_module, "StemLoadWorker",
            side_effect=make_worker, create=True,
        ):
            window._on_song_selected("a")
            window._on_song_selected("b")

            stale_arrays, stale_rate = player_module.read_stem_files(
                workers[0].stem_paths,
            )
            window._handle_stem_load_completed(
                workers[0], stale_arrays, stale_rate,
            )

            assert window._current_song_id != "a"
            assert not window._player.has_stems

            current_arrays, current_rate = player_module.read_stem_files(
                workers[1].stem_paths,
            )
            workers[1].completed.emit(current_arrays, current_rate)

        assert window._current_song_id == "b"
        assert np.allclose(
            window._player.stems["vocals"], 0.2, atol=1e-4,
        )

    def test_completion_for_removed_song_clears_loading_state(self, window):
        workers = []

        def make_worker(paths):
            worker = _FakeStemLoadWorker(paths)
            workers.append(worker)
            return worker

        with patch.object(
            main_window_module, "StemLoadWorker",
            side_effect=make_worker, create=True,
        ):
            window._on_song_selected("a")
            arrays, sample_rate = player_module.read_stem_files(
                workers[0].stem_paths,
            )
            window._library.get_song.side_effect = lambda _song_id: None
            workers[0].completed.emit(arrays, sample_rate)

        assert window._loading_song_id is None
        assert window._current_song_id is None
        assert window.windowTitle() == "stemma"

    def test_only_active_load_error_is_surfaced(self, window):
        workers = []

        def make_worker(paths):
            worker = _FakeStemLoadWorker(paths)
            workers.append(worker)
            return worker

        with patch.object(
            main_window_module, "StemLoadWorker",
            side_effect=make_worker, create=True,
        ), patch.object(
            main_window_module.QMessageBox, "warning",
        ) as warning:
            window._on_song_selected("a")
            window._on_song_selected("b")

            window._handle_stem_load_error(workers[0], "old failure")
            warning.assert_not_called()

            workers[1].error.emit("current failure")

        warning.assert_called_once()
        assert window._current_song_id is None
        assert window._loading_song_id is None
        assert not window._player.has_stems
        assert window._library_panel._song_delegate._playing_song_id is None

    def test_session_state_applies_only_after_async_restore(self, window):
        workers = []

        def make_worker(paths):
            worker = _FakeStemLoadWorker(paths)
            workers.append(worker)
            return worker

        window._settings.setValue("session/song_id", "a")
        window._settings.setValue("session/muted_stems", '["vocals"]')
        window._settings.setValue("session/volumes", '{"vocals": 0.5}')
        window._settings.setValue("session/position", 0.001)
        window._settings.setValue("session/metronome_bpm", 135)

        with patch.object(
            main_window_module, "StemLoadWorker",
            side_effect=make_worker, create=True,
        ), patch.object(
            main_window_module.PlayerControls, "start_detection",
        ):
            window._restore_session()

            assert not window._player.has_stems
            assert window._player.muted_stems == set()

            arrays, sample_rate = player_module.read_stem_files(
                workers[0].stem_paths,
            )
            workers[0].completed.emit(arrays, sample_rate)

        assert window._current_song_id == "a"
        assert window._player.muted_stems == {"vocals"}
        assert window._player.get_volume("vocals") == pytest.approx(0.5)
        assert window._player.current_seconds == pytest.approx(0.001, abs=1e-4)
        assert window._player.metronome_bpm == 135

    def test_old_restore_callback_cannot_seek_new_song(self, window):
        workers = []

        def make_worker(paths):
            worker = _FakeStemLoadWorker(paths)
            workers.append(worker)
            return worker

        window._settings.setValue("session/song_id", "a")
        window._settings.setValue("session/speed", 0.75)
        with patch.object(
            main_window_module, "StemLoadWorker",
            side_effect=make_worker, create=True,
        ), patch.object(
            window._player, "set_speed",
        ):
            window._restore_session()
            arrays, sample_rate = player_module.read_stem_files(
                workers[0].stem_paths,
            )
            workers[0].completed.emit(arrays, sample_rate)
            pending = list(window._pending_restore_callbacks)
            assert len(pending) == 1
            late_callback = pending[0][1]

            window._on_song_selected("b")
            assert window._pending_restore_callbacks == []
            with patch.object(window._player, "seek") as seek:
                late_callback(0.75)

        seek.assert_not_called()

    def test_existing_recordings_are_read_by_the_stem_worker(self, window):
        recording = (
            window._library.get_song("a").stems_path
            + "/recording_take1.wav"
        )
        _write_wav(recording, 0.3)
        workers = []

        def make_worker(paths):
            worker = _FakeStemLoadWorker(paths)
            workers.append(worker)
            return worker

        with patch.object(
            main_window_module, "StemLoadWorker",
            side_effect=make_worker, create=True,
        ), patch.object(
            main_window_module.PlayerControls, "start_detection",
        ):
            window._on_song_selected("a")
            assert "recording_take1" in workers[0].stem_paths

            arrays, sample_rate = player_module.read_stem_files(
                workers[0].stem_paths,
            )
            with patch.object(
                main_window_module.sf, "read",
                side_effect=AssertionError("GUI thread read recording"),
            ):
                workers[0].completed.emit(arrays, sample_rate)

        assert "recording_take1" in window._player.stems
        assert "recording_take1" in window._player_controls._recording_rows

    def test_close_song_orphans_then_shutdown_drains_active_load(self, window):
        workers = []

        def make_worker(paths):
            worker = _FakeStemLoadWorker(paths)
            workers.append(worker)
            return worker

        with patch.object(
            main_window_module, "StemLoadWorker",
            side_effect=make_worker, create=True,
        ):
            window._on_song_selected("a")
            worker = workers[0]

            window._on_close_song()

            assert worker in window._orphaned_stem_load_workers
            assert worker.wait_calls == 0

            window._shutdown_stem_loads()

        assert worker.wait_calls == 1
        assert window._orphaned_stem_load_workers == []

    def test_pre_removal_drains_load_before_library_delete(self, window):
        workers = []

        def make_worker(paths):
            worker = _FakeStemLoadWorker(paths)
            workers.append(worker)
            return worker

        def remove_song(_song_id):
            assert workers[0].wait_calls == 1

        window._library.remove_song.side_effect = remove_song
        with patch.object(
            main_window_module, "StemLoadWorker",
            side_effect=make_worker, create=True,
        ), patch.object(
            main_window_module.QMessageBox, "question",
            return_value=main_window_module.QMessageBox.StandardButton.Yes,
        ):
            window._library_panel.select_song("a")
            window._library_panel._on_remove_clicked()

        window._library.remove_song.assert_called_once_with("a")
        assert workers[0] not in window._orphaned_stem_load_workers

    def test_next_during_same_song_load_binds_autoplay(self, window):
        workers = []

        def make_worker(paths):
            worker = _FakeStemLoadWorker(paths)
            workers.append(worker)
            return worker

        with patch.object(
            main_window_module, "StemLoadWorker",
            side_effect=make_worker, create=True,
        ), patch.object(
            window, "_get_next_song_id", return_value="a",
        ), patch.object(window._player, "play") as play:
            window._library_panel.select_song("a")
            window._advance_song(1)
            play.assert_not_called()

            arrays, sample_rate = player_module.read_stem_files(
                workers[0].stem_paths,
            )
            workers[0].completed.emit(arrays, sample_rate)

        play.assert_called_once_with()

    def test_selection_does_not_reload_outgoing_recording_on_gui_thread(
        self, window,
    ):
        workers = []

        def make_worker(paths):
            worker = _FakeStemLoadWorker(paths)
            workers.append(worker)
            return worker

        window._player._recording_buffer = np.zeros(
            (32, 2), dtype=np.float32,
        )
        window._player.set_recording_song_dir(
            window._library.get_song("b").stems_path,
        )
        saved_path = os.path.join(
            window._library.get_song("b").stems_path,
            "recording_take1.wav",
        )

        def save_outgoing(_song_dir):
            window._player._recording_buffer = None
            return saved_path

        with patch.object(
            main_window_module, "StemLoadWorker",
            side_effect=make_worker, create=True,
        ), patch.object(
            window._player, "save_recording", side_effect=save_outgoing,
        ) as save_recording, patch.object(
            main_window_module.sf, "read",
            side_effect=AssertionError("GUI thread recording read"),
        ) as gui_read:
            window._on_song_selected("a")

        save_recording.assert_called_once()
        gui_read.assert_not_called()
        assert len(workers) == 1

    def test_close_saves_recording_without_reload_or_peak_work(self, window):
        window._player._recording_buffer = np.zeros(
            (32, 2), dtype=np.float32,
        )
        window._player._recording_frames_captured = 32
        song_dir = window._library.get_song("a").stems_path
        window._player.set_recording_song_dir(song_dir)
        saved_path = os.path.join(song_dir, "recording_take1.wav")

        def save_outgoing(_song_dir):
            window._player._recording_buffer = None
            return saved_path

        with patch.object(
            window._player, "stop", wraps=window._player.stop,
        ) as stop, patch.object(
            window._player, "save_recording", side_effect=save_outgoing,
        ) as save_recording, patch.object(
            main_window_module.sf, "read",
            return_value=(
                np.zeros((32, 2), dtype=np.float32),
                44100,
            ),
        ) as gui_read, patch.object(
            window._player_controls, "_do_recompute_peaks",
        ) as recompute_peaks, patch.object(
            window._separation_queue, "shutdown",
        ), patch.object(
            window, "_shutdown_stem_loads",
        ), patch.object(
            window._player, "shutdown",
        ), patch.object(
            window._player_controls, "shutdown",
        ), patch.object(
            main_window_module, "shutdown_peak_pool",
        ):
            window.closeEvent(QCloseEvent())

        stop.assert_called_once_with()
        save_recording.assert_called_once_with(song_dir)
        gui_read.assert_not_called()
        recompute_peaks.assert_not_called()

    @pytest.mark.parametrize("action", ["error", "close"])
    def test_failed_or_closed_song_can_be_selected_again(
        self, window, action,
    ):
        workers = []

        def make_worker(paths):
            worker = _FakeStemLoadWorker(paths)
            workers.append(worker)
            return worker

        with patch.object(
            main_window_module, "StemLoadWorker",
            side_effect=make_worker, create=True,
        ), patch.object(
            main_window_module.QMessageBox, "warning",
        ):
            window._library_panel.select_song("a")
            if action == "error":
                workers[0].error.emit("broken")
            else:
                window._on_close_song()

            assert window._library_panel._list.currentItem() is None
            assert window._library_panel.select_song("a")

        assert len(workers) == 2

    def test_save_during_load_preserves_song_but_updates_playback_mode(
        self, window,
    ):
        workers = []

        def make_worker(paths):
            worker = _FakeStemLoadWorker(paths)
            workers.append(worker)
            return worker

        window._settings.setValue("session/song_id", "previous")
        with patch.object(
            main_window_module, "StemLoadWorker",
            side_effect=make_worker, create=True,
        ):
            window._on_song_selected("a")
            window._library_panel.set_repeat_mode(REPEAT_ALL)
            window._save_session()

        assert window._settings.value("session/song_id") == "previous"
        assert window._settings.value("session/repeat_mode") == REPEAT_ALL

    def test_removing_loading_song_invalidates_its_worker(self, window):
        workers = []

        def make_worker(paths):
            worker = _FakeStemLoadWorker(paths)
            workers.append(worker)
            return worker

        with patch.object(
            main_window_module, "StemLoadWorker",
            side_effect=make_worker, create=True,
        ):
            window._on_song_selected("a")
            window._on_song_removed("a")

        assert window._loading_song_id is None
        assert workers[0] in window._orphaned_stem_load_workers

    def test_shuffle_navigation_excludes_song_currently_loading(self, window):
        window._current_song_id = None
        window._loading_song_id = "a"
        window._shuffle_queue = []

        with patch.object(
            main_window_module.random, "shuffle",
            side_effect=lambda _values: None,
        ):
            result = window._pop_shuffle_queue(["a", "b"])

        assert result == "b"


class _FakePeakPool:
    def __init__(self, result) -> None:
        self.result = result
        self.submit_calls = 0

    def submit(self, _function, **_kwargs):
        self.submit_calls += 1
        future = Future()
        future.set_result(self.result)
        return future


class TestPeakGeneration:
    def test_cleanup_drains_detection_worker_before_releasing_it(self, window):
        controls = window._player_controls
        worker = MagicMock()
        controls._detection_worker = worker

        controls._cleanup_peak_thread()

        worker.completed.disconnect.assert_called_once_with()
        worker.error.disconnect.assert_called_once_with()
        worker.finished.disconnect.assert_called_once_with()
        worker.wait.assert_called_once_with()
        assert controls._detection_worker is None

    def test_shutdown_drains_orphaned_detection_workers(self, window):
        controls = window._player_controls
        worker = MagicMock()
        controls._orphaned_workers = [worker]

        controls.shutdown()

        worker.wait.assert_called_once_with()
        assert controls._orphaned_workers == []

    def test_stale_result_is_rejected_then_fresh_result_applies(self, window):
        controls = window._player_controls
        window._player._stems = {
            "vocals": np.zeros((64, 2), dtype=np.float32),
        }
        stale_main = np.array([0.1], dtype=np.float32)
        fresh_main = np.array([0.9], dtype=np.float32)
        stale = Future()
        stale.set_result((stale_main, {"vocals": stale_main}))
        controls._peak_generation = 2
        controls._peak_future = stale
        controls._peak_future_generation = 1
        pool = _FakePeakPool(
            (fresh_main, {"vocals": fresh_main}),
        )

        with patch.object(
            player_controls_module, "_get_peak_pool", return_value=pool,
        ), patch.object(
            controls, "_on_peaks_computed",
        ) as apply_peaks:
            controls._poll_peak_future()
            apply_peaks.assert_not_called()
            assert pool.submit_calls == 1

            controls._poll_peak_future()

        apply_peaks.assert_called_once()
        assert np.array_equal(apply_peaks.call_args.args[0], fresh_main)

    def test_recompute_invalidates_in_flight_generation(self, window):
        controls = window._player_controls
        controls._peak_generation = 4

        controls._recompute_peaks()

        controls._peaks_timer.stop()
        assert controls._peak_generation == 5


class TestDetectionGeneration:
    def test_detach_reaps_worker_finishing_before_signal_connect(
        self, window,
    ):
        controls = window._player_controls
        worker = MagicMock()
        worker.isRunning.side_effect = [True, False]
        controls._detection_worker = worker

        controls._detach_detection_worker()

        assert worker not in controls._orphaned_workers
        worker.finished.connect.assert_called_once()
        worker.setParent.assert_called_once_with(None)
        worker.deleteLater.assert_called_once_with()

    def test_late_detection_result_and_error_are_ignored(self, window):
        controls = window._player_controls
        window._player._stems = {
            "vocals": np.zeros((64, 2), dtype=np.float32),
        }
        workers = []

        def make_worker(*args, **kwargs):
            worker = _FakeDetectionWorker(*args, **kwargs)
            workers.append(worker)
            return worker

        with patch.object(
            player_controls_module, "DetectionWorker",
            side_effect=make_worker,
        ):
            controls.start_detection()
            late_completed = controls._on_detect_completed
            late_error = controls._on_detect_error
            controls.start_detection()
            active = workers[1]
            workers[0].finished.connect(controls._on_detect_finished)
            controls._key_label.setText("current")

            with patch.object(
                window._player, "set_beat_times",
            ) as set_beats, patch.object(
                controls, "_refresh_key_label",
            ) as refresh_key:
                late_completed(DetectionResult(beat_times=[0.1, 0.2]))
                late_error("old failure")
                workers[0].finished.emit()
                set_beats.assert_not_called()
                refresh_key.assert_not_called()
                assert controls._key_label.text() == "current"
                assert controls._detection_worker is active

                active.completed.emit(
                    DetectionResult(beat_times=[0.2, 0.4]),
                )

        set_beats.assert_called_once_with([0.2, 0.4], [])
        assert active._detection_generation == controls._detection_generation

    def test_late_model_download_does_not_restart_cleared_song(self, window):
        controls = window._player_controls
        downloader = _FakeBeatDownloader()
        manager = MagicMock()
        manager.beat_model_path.return_value = "beat_this.onnx"
        manager.is_beat_model_downloaded.return_value = False
        manager.download_beat_model.return_value = downloader
        controls.set_model_manager(manager)
        window._player._stems = {
            "vocals": np.zeros((64, 2), dtype=np.float32),
        }

        with patch.object(controls, "_run_detection") as run_detection:
            controls.start_detection(1.0, 2.0)
            window._player._stems.clear()
            controls.set_stem_names([])
            downloader.download_complete.emit("beat_this.onnx")

        run_detection.assert_not_called()

    def test_current_model_download_resumes_through_start_detection(
        self, window,
    ):
        controls = window._player_controls
        downloader = _FakeBeatDownloader()
        manager = MagicMock()
        manager.beat_model_path.return_value = "beat_this.onnx"
        manager.is_beat_model_downloaded.return_value = False
        manager.download_beat_model.return_value = downloader
        controls.set_model_manager(manager)
        window._player._stems = {
            "vocals": np.zeros((64, 2), dtype=np.float32),
        }
        controls.start_detection(3.0, 4.0)

        with patch.object(controls, "start_detection") as restart:
            downloader.download_complete.emit("beat_this.onnx")

        restart.assert_called_once_with(
            3.0, 4.0, _model_ready=True,
        )

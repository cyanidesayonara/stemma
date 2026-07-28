"""Tests for generation-safe asynchronous song loading."""

from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from src import player as player_module
from src.library import Song
from src.ui import main_window as main_window_module
from src.ui import player_controls as player_controls_module
from src.ui.library_panel import REPEAT_ALL


def _write_wav(path, value: float, sample_rate: int = 44100) -> None:
    data = np.full((64, 2), value, dtype=np.float32)
    sf.write(str(path), data, sample_rate)


class _FakeSignal:
    def __init__(self) -> None:
        self.slots = []

    def connect(self, slot) -> None:
        self.slots.append(slot)

    def disconnect(self) -> None:
        self.slots.clear()

    def emit(self, *args) -> None:
        for slot in list(self.slots):
            slot(*args)


class _FakeStemLoadWorker:
    def __init__(self, stem_paths) -> None:
        self.stem_paths = dict(stem_paths)
        self.completed = _FakeSignal()
        self.error = _FakeSignal()
        self.finished = _FakeSignal()
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


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


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
    with patch.object(
        main_window_module.MainWindow, "_prune_incomplete_songs",
    ), patch.object(
        main_window_module.QTimer, "singleShot",
    ), patch.object(
        main_window_module.PlayerControls, "start_detection",
    ):
        result = main_window_module.MainWindow(
            library, player, MagicMock(),
        )

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


class TestMainWindowAsyncLoading:
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
            stale_completion = workers[0].completed.slots[0]
            window._on_song_selected("b")

            stale_arrays, stale_rate = player_module.read_stem_files(
                workers[0].stem_paths,
            )
            stale_completion(stale_arrays, stale_rate)

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
            stale_error = workers[0].error.slots[0]
            window._on_song_selected("b")

            stale_error("old failure")
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

            assert worker.completed.slots == []
            assert worker.error.slots == []
            assert worker in window._orphaned_stem_load_workers
            assert worker.wait_calls == 0

            window._shutdown_stem_loads()

        assert worker.wait_calls == 1
        assert window._orphaned_stem_load_workers == []

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
        assert workers[0].completed.slots == []
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

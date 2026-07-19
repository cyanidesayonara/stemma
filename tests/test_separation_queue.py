"""Tests for the background separation queue and its UI integration.

Workers are replaced with synchronous fakes: `start()` runs the fake
inline and emits like the real engines (progress, then finished(dict)
or error(str) exactly once), so queue behavior is tested without audio
or models.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from src.separation_queue import SeparationJob, SeparationQueue


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _FakeWorker(QObject):
    """Stands in for SeparatorWorker / MdxSeparatorWorker.

    By default completes successfully the moment start() is called.
    Set ``outcome`` to "error" for failure, or "hold" to stay 'running'
    until finish()/fail() is called manually (simulates a long job).
    """

    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, job, outcome="success"):
        super().__init__()
        self.job = job
        self.outcome = outcome
        self.cancelled = False
        self._running = False

    def start(self):
        self._running = True
        self.progress.emit(50, "half way")
        if self.outcome == "success":
            self.finish()
        elif self.outcome == "error":
            self.fail("boom")
        # "hold": stay running until told otherwise.

    def finish(self):
        self._running = False
        self.finished.emit({"vocals": "v.wav"})

    def fail(self, msg):
        self._running = False
        self.error.emit(msg)

    def cancel(self):
        self.cancelled = True
        # Real engines notice the flag between windows and emit error.
        if self._running:
            self.fail("Separation cancelled by user.")

    def isRunning(self):
        return self._running

    def wait(self, _ms=0):
        return True

    def deleteLater(self):  # noqa: D401 (QObject API)
        pass


def _job(song_id, model_key="mdx_inst_hq3"):
    return SeparationJob(
        song_id=song_id, input_path="in.mp3", output_dir="out",
        model_path="m.onnx", model_key=model_key,
    )


@pytest.fixture
def queue_and_log(app):
    """A queue whose workers are fakes, plus a signal log."""
    q = SeparationQueue()
    log = []
    q.job_queued.connect(lambda s: log.append(("queued", s)))
    q.job_started.connect(lambda s: log.append(("started", s)))
    q.job_progress.connect(lambda s, p, m: log.append(("progress", s, p)))
    q.job_finished.connect(lambda s, k: log.append(("finished", s, k)))
    q.job_failed.connect(lambda s, m: log.append(("failed", s, m)))

    made = []

    def fake_make(job, outcome_by_song=None):
        w = _FakeWorker(job, (outcome_by_song or {}).get(job.song_id, "success"))
        made.append(w)
        return w

    q._outcomes = {}
    q._make_worker = lambda job: fake_make(job, q._outcomes)
    q._made_workers = made
    return q, log


class TestQueueSequencing:
    def test_single_job_full_lifecycle(self, queue_and_log):
        q, log = queue_and_log
        q.enqueue(_job("a"))
        assert log == [
            ("queued", "a"), ("started", "a"), ("progress", "a", 50),
            ("finished", "a", "mdx_inst_hq3"),
        ]
        assert q.pending_count == 0
        assert not q.is_song_pending("a")

    def test_two_jobs_run_serially_in_order(self, queue_and_log):
        q, log = queue_and_log
        q._outcomes = {"a": "hold"}
        q.enqueue(_job("a"))
        q.enqueue(_job("b"))
        # b waits while a runs.
        assert q.is_song_pending("b")
        assert q.active_song_id == "a"
        assert not any(e[0] == "started" and e[1] == "b" for e in log)

        q._made_workers[0].finish()

        assert ("finished", "a", "mdx_inst_hq3") in log
        assert ("started", "b") in log
        assert ("finished", "b", "mdx_inst_hq3") in log
        assert q.pending_count == 0

    def test_failed_job_does_not_block_next(self, queue_and_log):
        q, log = queue_and_log
        q._outcomes = {"a": "error"}
        q.enqueue(_job("a"))
        q.enqueue(_job("b"))
        assert ("failed", "a", "boom") in log
        assert ("finished", "b", "mdx_inst_hq3") in log

    def test_model_key_passed_through(self, queue_and_log):
        q, log = queue_and_log
        q.enqueue(_job("a", model_key="htdemucs_6s"))
        assert ("finished", "a", "htdemucs_6s") in log


class TestQueueCancel:
    def test_cancel_queued_job_drops_it(self, queue_and_log):
        q, log = queue_and_log
        q._outcomes = {"a": "hold"}
        q.enqueue(_job("a"))
        q.enqueue(_job("b"))

        q.cancel_song("b")

        assert any(
            e[0] == "failed" and e[1] == "b" and "cancelled" in e[2].lower()
            for e in log
        )
        assert not q.is_song_pending("b")
        # a is untouched.
        assert q.active_song_id == "a"

    def test_cancel_active_job_cancels_worker(self, queue_and_log):
        q, log = queue_and_log
        q._outcomes = {"a": "hold"}
        q.enqueue(_job("a"))

        q.cancel_song("a")

        worker = q._made_workers[0]
        assert worker.cancelled
        assert any(
            e[0] == "failed" and e[1] == "a" and "cancelled" in e[2].lower()
            for e in log
        )
        assert q.active_song_id is None

    def test_cancel_unknown_song_is_noop(self, queue_and_log):
        q, log = queue_and_log
        q.cancel_song("ghost")
        assert log == []


class TestQueueShutdown:
    def test_shutdown_cancels_active_and_drops_queued(self, queue_and_log):
        q, log = queue_and_log
        q._outcomes = {"a": "hold"}
        q.enqueue(_job("a"))
        q.enqueue(_job("b"))

        q.shutdown(wait_ms=10)

        worker = q._made_workers[0]
        assert worker.cancelled
        assert q.pending_count == 0
        # Shutdown is silent: detached before the cancel lands, queued
        # jobs dropped without signals (the app is closing).
        assert not any(e[0] == "finished" for e in log)


class TestWorkerSelection:
    def test_mdx_key_builds_mdx_worker(self, app):
        q = SeparationQueue()
        with patch("src.separation_queue.MdxSeparatorWorker") as mdx_cls, \
             patch("src.separation_queue.SeparatorWorker") as demucs_cls:
            q._make_worker(_job("a", model_key="mdx_inst_hq3"))
            mdx_cls.assert_called_once()
            demucs_cls.assert_not_called()

    def test_demucs_keys_build_demucs_worker(self, app):
        q = SeparationQueue()
        with patch("src.separation_queue.MdxSeparatorWorker") as mdx_cls, \
             patch("src.separation_queue.SeparatorWorker") as demucs_cls:
            q._make_worker(_job("a", model_key="htdemucs_6s"))
            demucs_cls.assert_called_once()
            assert demucs_cls.call_args.kwargs["is_6_stem"] is True
            mdx_cls.assert_not_called()


class TestDialogHandoff:
    """With a queue, the dialog enqueues and closes instead of running
    the worker inline."""

    def test_dialog_enqueues_and_accepts(self, app, tmp_path):
        from src.ui.import_dialog import ImportDialog

        library = MagicMock()
        mm = MagicMock()
        queue = MagicMock()
        dlg = ImportDialog(library, mm, separation_queue=queue)

        song = MagicMock()
        song.id = "s1"
        song.original_path = str(tmp_path / "in.mp3")
        song.stems_path = str(tmp_path / "stems")

        with patch.object(dlg, "accept") as accept:
            dlg._start_separation_worker(song, "model.onnx", "mdx_inst_hq3")

        queue.enqueue.assert_called_once()
        job = queue.enqueue.call_args.args[0]
        assert job.song_id == "s1"
        assert job.model_key == "mdx_inst_hq3"
        accept.assert_called_once()
        # The dialog must not treat the song as its own rollback target
        # anymore -- the queue owns the outcome now.
        assert dlg._import_song_id is None
        assert dlg._worker is None

    def test_dialog_without_queue_runs_inline(self, app, tmp_path):
        from src.ui.import_dialog import ImportDialog

        library = MagicMock()
        mm = MagicMock()
        dlg = ImportDialog(library, mm)

        song = MagicMock()
        song.id = "s1"
        song.original_path = str(tmp_path / "in.mp3")
        song.stems_path = str(tmp_path / "stems")

        with patch(
            "src.ui.import_dialog.MdxSeparatorWorker"
        ) as worker_cls:
            dlg._start_separation_worker(song, "model.onnx", "mdx_inst_hq3")
        worker_cls.assert_called_once()
        worker_cls.return_value.start.assert_called_once()


class TestLibraryPanelSeparatingState:
    def _panel(self, app):
        from src.ui.library_panel import LibraryPanel

        library = MagicMock()
        song = MagicMock()
        song.id = "s1"
        song.artist = "a"
        song.title = "t"
        library.songs = [song]
        return LibraryPanel(library)

    def test_separating_row_is_unselectable(self, app):
        from PySide6.QtCore import Qt

        panel = self._panel(app)
        panel.set_song_separating("s1", "Separating... 10%")
        item = panel._item_for_song("s1")
        assert not (item.flags() & Qt.ItemFlag.ItemIsSelectable)
        assert panel.is_song_separating("s1")

        panel.clear_song_separating("s1")
        item = panel._item_for_song("s1")
        assert item.flags() & Qt.ItemFlag.ItemIsSelectable
        assert not panel.is_song_separating("s1")

    def test_state_survives_refresh(self, app):
        from PySide6.QtCore import Qt

        panel = self._panel(app)
        panel.set_song_separating("s1", "Separating... 10%")
        panel.refresh()
        item = panel._item_for_song("s1")
        assert not (item.flags() & Qt.ItemFlag.ItemIsSelectable)

    def test_select_song_refuses_separating_row(self, app):
        panel = self._panel(app)
        panel.set_song_separating("s1", "Separating...")
        selected = []
        panel.song_selected.connect(lambda s: selected.append(s))
        panel.select_song("s1")
        assert selected == []


class TestMainWindowGlue:
    """Stub-based checks of the queue handlers (no full window)."""

    def test_finished_records_model_and_clears_state(self, app):
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        MainWindow._on_separation_finished(stub, "s1", "mdx_inst_hq3")
        stub._library.update_song.assert_called_once_with(
            "s1", model_used="mdx_inst_hq3"
        )
        stub._library_panel.clear_song_separating.assert_called_once_with("s1")
        stub._library_panel.refresh.assert_called_once()

    def test_failed_rolls_back_row(self, app):
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        stub._library.get_song.return_value = MagicMock()
        with patch("src.ui.main_window.QMessageBox") as mb:
            MainWindow._on_separation_failed(stub, "s1", "boom")
        stub._library.remove_song.assert_called_once_with("s1")
        mb.warning.assert_called_once()

    def test_cancelled_failure_shows_no_dialog(self, app):
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        stub._library.get_song.return_value = MagicMock()
        with patch("src.ui.main_window.QMessageBox") as mb:
            MainWindow._on_separation_failed(
                stub, "s1", "Separation cancelled by user."
            )
        stub._library.remove_song.assert_called_once_with("s1")
        mb.warning.assert_not_called()

    def test_prune_removes_stemless_songs(self, app, tmp_path):
        from src.ui.main_window import MainWindow

        good = MagicMock()
        good.id = "good"
        good.stems_path = str(tmp_path / "good")
        os.makedirs(good.stems_path)
        open(os.path.join(good.stems_path, "vocals.wav"), "wb").close()

        ghost = MagicMock()
        ghost.id = "ghost"
        ghost.stems_path = str(tmp_path / "ghost")
        os.makedirs(ghost.stems_path)

        stub = MagicMock()
        stub._library.songs = [good, ghost]
        MainWindow._prune_incomplete_songs(stub)

        stub._library.remove_song.assert_called_once_with("ghost")

"""Tests for the background separation queue and its UI integration.

Workers are replaced with synchronous fakes: `start()` runs the fake
inline and emits like the real engines (progress, then finished(dict)
or error(str) exactly once), so queue behavior is tested without audio
or models.
"""

import json
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

    # -- startup prune: only provably interrupted imports are removed --

    @staticmethod
    def _song(tmp_path, name, stems=(), marker=None, pending=False,
              takes=0, model_used="htdemucs"):
        song = MagicMock()
        song.id = name
        song.model_used = model_used
        song.stems_path = str(tmp_path / name)
        os.makedirs(song.stems_path)
        for stem in stems:
            open(os.path.join(song.stems_path, f"{stem}.wav"), "wb").close()
        for take in range(1, takes + 1):
            open(os.path.join(
                song.stems_path, f"recording_take{take}.wav"), "wb").close()
        if marker is not None:
            with open(os.path.join(
                song.stems_path, ".separation-complete.json",
            ), "w", encoding="utf-8") as f:
                f.write(marker if isinstance(marker, str)
                        else json.dumps(marker))
        if pending:
            open(os.path.join(
                song.stems_path, ".separation-pending"), "wb").close()
        return song

    def _prune(self, songs):
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        stub._library.songs = songs
        MainWindow._prune_incomplete_songs(stub)
        return [c.args[0] for c in stub._library.remove_song.call_args_list]

    def test_prune_removes_imports_interrupted_mid_job(self, app, tmp_path):
        """Closed or crashed mid-separation: the pending marker is there and
        no completion marker, however many stems were written."""
        songs = [
            self._song(tmp_path, "queued", pending=True, model_used=""),
            self._song(tmp_path, "four-of-six", pending=True,
                       stems=("drums", "bass", "other", "vocals"),
                       model_used=""),
        ]

        assert self._prune(songs) == ["queued", "four-of-six"]

    def test_prune_keeps_finished_songs(self, app, tmp_path):
        complete = {"version": 1, "model": "mdx_inst_hq3",
                    "stems": ["vocals", "other"]}
        songs = [
            self._song(tmp_path, "marked", stems=("vocals", "other"),
                       marker=complete),
            # A pending marker the completion write failed to delete.
            self._song(tmp_path, "marked-and-pending",
                       stems=("vocals", "other"), marker=complete,
                       pending=True),
            self._song(tmp_path, "legacy",
                       stems=("drums", "bass", "other", "vocals")),
        ]

        assert self._prune(songs) == []

    @pytest.mark.parametrize("case", [
        "legacy-partial", "garbled-marker", "marker-missing-a-stem",
        "future-marker", "unknown-model", "takes-on-a-pending-song",
        "unreadable-marker-with-stale-pending",
    ])
    def test_prune_keeps_what_it_cannot_prove_is_unfinished(
        self, app, tmp_path, case,
    ):
        """Removal deletes the folder; without the import's pending marker,
        or with the user's own takes inside, it never happens (#181)."""
        song = {
            "legacy-partial": dict(stems=("vocals",)),
            "garbled-marker": dict(
                stems=("drums", "bass", "other", "vocals"),
                marker="{garbled"),
            "marker-missing-a-stem": dict(
                stems=("vocals",),
                marker={"version": 1, "model": "mdx_inst_hq3",
                        "stems": ["vocals", "other"]}),
            "future-marker": dict(
                stems=("vocals",),
                marker={"version": 99, "model": "htdemucs",
                        "stems": ["drums", "bass", "other", "vocals"]}),
            "unknown-model": dict(model_used=""),
            "takes-on-a-pending-song": dict(pending=True, takes=1),
            # Finished (the marker is written atomically, last), but the
            # pending removal failed and the marker is unreadable now.
            "unreadable-marker-with-stale-pending": dict(
                stems=("vocals", "other"), marker="{garbled", pending=True),
        }[case]

        assert self._prune([self._song(tmp_path, case, **song)]) == []

    def test_import_then_close_then_relaunch(self, app, tmp_path):
        """End to end: a fresh import with no separation yet is removed
        on the next launch; a finished one is kept, its marker cleared."""
        from src.library import SongLibrary
        from src.separation_state import write_completion_marker
        from src.ui.main_window import MainWindow

        source = tmp_path / "song.wav"
        source.write_bytes(b"audio")
        library = SongLibrary(str(tmp_path / "data"))
        interrupted = library.add_song("Cut short", "Band", str(source))
        finished = library.add_song("Done", "Band", str(source))
        for stem in ("vocals", "other"):
            open(os.path.join(
                finished.stems_path, f"{stem}.wav"), "wb").close()
        write_completion_marker(finished.stems_path, "mdx_inst_hq3")

        relaunched = SongLibrary(str(tmp_path / "data"))
        stub = MagicMock()
        stub._library = relaunched
        MainWindow._prune_incomplete_songs(stub)

        assert [s.id for s in relaunched.songs] == [finished.id]
        assert not os.path.exists(interrupted.stems_path)
        assert not os.path.exists(os.path.join(
            finished.stems_path, ".separation-pending"))


class TestDamagedIndexKeepsAudio:
    """The reproduced #181 data loss, end to end: a damaged library.json
    plus the startup prune deleted every song imported before v2.6."""

    @pytest.mark.parametrize("damage", ["truncated", "unknown_key"])
    def test_legacy_song_survives_a_damaged_index(
        self, app, tmp_path, damage,
    ):
        from src.library import SongLibrary
        from src.ui.main_window import MainWindow

        data_dir = str(tmp_path / "data")
        song_dir = os.path.join(data_dir, "songs", "0123456789ab")
        os.makedirs(song_dir)
        for name in ("original.mp3", "drums.wav", "bass.wav",
                     "other.wav", "vocals.wav"):
            open(os.path.join(song_dir, name), "wb").close()
        entry = {
            "id": "0123456789ab", "title": "Old Song", "artist": "Band",
            "original_path": os.path.join(song_dir, "original.mp3"),
            "stems_path": song_dir, "model_used": "htdemucs",
            "date_added": "2025-01-01",
        }
        index = json.dumps([entry])
        if damage == "truncated":
            index = index[: len(index) // 2]
        else:
            index = json.dumps([{**entry, "from_the_future": 1}])
        with open(os.path.join(data_dir, "library.json"), "w") as f:
            f.write(index)

        library = SongLibrary(data_dir)
        stub = MagicMock()
        stub._library = library
        MainWindow._prune_incomplete_songs(stub)

        assert os.path.isfile(os.path.join(song_dir, "vocals.wav"))
        assert [s.id for s in library.songs] == ["0123456789ab"]


class TestCompletionMarkerWrite:
    def test_locked_pending_marker_does_not_fail_a_finished_job(
        self, tmp_path, monkeypatch,
    ):
        """An indexer or antivirus holding the pending file must not turn
        a finished separation into "Import failed" (and a rollback)."""
        from src import separation_state

        song_dir = str(tmp_path)
        separation_state.mark_separation_pending(song_dir)
        for stem in ("vocals", "other"):
            open(os.path.join(song_dir, f"{stem}.wav"), "wb").close()
        real_remove = os.remove

        def locked(path):
            if path.endswith(separation_state.PENDING_MARKER):
                raise PermissionError(13, "in use", path)
            real_remove(path)

        monkeypatch.setattr(separation_state.os, "remove", locked)

        separation_state.write_completion_marker(song_dir, "mdx_inst_hq3")

        assert separation_state.separation_is_complete(song_dir, "")
        assert not separation_state.is_interrupted_import(song_dir)

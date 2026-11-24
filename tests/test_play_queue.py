"""Tests for the PlayQueue model, PlayQueueWidget, and MainWindow queue wiring."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QPoint, QThread, Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication, QMenu

from src import player as player_module
from src.app_settings import open_settings
from src.library import Song, SongLibrary
from src.play_queue import (
    CONTEXT_LIBRARY,
    CONTEXT_SETLIST,
    REPEAT_ALL,
    REPEAT_OFF,
    REPEAT_ONE,
    PlayQueue,
)
from src.ui import main_window as main_window_module
from src.ui.library_panel import LibraryPanel
from src.ui.play_queue_widget import PlayQueueWidget
from src.ui.styles import DARK_COLORS, LIGHT_COLORS


def _write_wav(path, value: float = 0.1, sample_rate: int = 44100) -> None:
    data = np.full((64, 2), value, dtype=np.float32)
    sf.write(str(path), data, sample_rate)


def _make_songs(count: int = 5) -> list[Song]:
    return [
        Song(
            id=f"s{i}",
            title=f"Song {i}",
            artist=f"Artist {i}",
            original_path="",
            stems_path="",
            model_used="htdemucs",
            date_added="",
        )
        for i in range(1, count + 1)
    ]


def _make_mock_library(songs: list[Song]) -> MagicMock:
    library = MagicMock()
    library.songs = songs
    library.writable = True
    library.get_song.side_effect = lambda sid: next(
        (s for s in songs if s.id == sid), None
    )
    return library


@pytest.fixture(scope="module")
def qapp():
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


# ---------------------------------------------------------------------------
# Pure PlayQueue model tests
# ---------------------------------------------------------------------------


class TestPlayQueueMutations:
    def test_play_next_and_enqueue_preserve_order_and_allow_duplicates(self):
        q = PlayQueue()
        q.enqueue("s1")
        q.enqueue("s2")
        q.play_next("s3")
        q.play_next("s1")  # duplicate allowed
        assert q.up_next == ["s1", "s3", "s1", "s2"]
        assert len(q) == 4
        assert q.has_up_next is True

    def test_play_next_many_and_enqueue_many(self):
        q = PlayQueue()
        q.enqueue_many(["s1", "s2"])
        q.play_next_many(["s3", "s4"])
        assert q.up_next == ["s3", "s4", "s1", "s2"]

    @pytest.mark.parametrize("bad_id", ["", "   ", 123, None])
    def test_invalid_song_id_raises_value_error(self, bad_id):
        q = PlayQueue()
        q.enqueue("s1")
        with pytest.raises(ValueError):
            q.play_next(bad_id)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            q.enqueue(bad_id)  # type: ignore[arg-type]
        assert q.up_next == ["s1"]

    def test_many_mutators_are_atomic_on_invalid_input(self):
        q = PlayQueue()
        q.enqueue("s1")
        with pytest.raises(ValueError):
            q.enqueue_many(["s2", ""])
        assert q.up_next == ["s1"]

        with pytest.raises(ValueError):
            q.play_next_many(["s2", None])  # type: ignore[list-item]
        assert q.up_next == ["s1"]

    def test_remove_at_and_bounds(self):
        q = PlayQueue()
        q.enqueue_many(["s1", "s2", "s3"])
        assert q.remove_at(1) == "s2"
        assert q.up_next == ["s1", "s3"]
        with pytest.raises(IndexError):
            q.remove_at(-1)
        with pytest.raises(IndexError):
            q.remove_at(2)

    def test_move_upward_downward_noop_and_bounds(self):
        q = PlayQueue()
        q.enqueue_many(["s1", "s2", "s3", "s4"])
        q.move(2, 0)
        assert q.up_next == ["s3", "s1", "s2", "s4"]
        q.move(0, 3)
        assert q.up_next == ["s1", "s2", "s4", "s3"]
        q.move(1, 1)
        assert q.up_next == ["s1", "s2", "s4", "s3"]
        with pytest.raises(IndexError):
            q.move(-1, 0)
        with pytest.raises(IndexError):
            q.move(0, 4)

    def test_clear_up_next_preserves_active_setlist_context(self):
        q = PlayQueue()
        q.start_setlist("set1", ["s1", "s2", "s3"], name="Friday Gig")
        q.enqueue("s4")
        q.clear_up_next()
        assert q.up_next == []
        assert q.has_up_next is False
        assert q.is_setlist_active is True
        assert q.context_id == "set1"


class TestPlayQueueSetlistContext:
    def test_start_setlist_and_empty_clears_context(self):
        q = PlayQueue()
        q.enqueue("s9")
        first = q.start_setlist(
            "set1", ["s1", "s2", "s3"], name="Set A", start_index=1,
        )
        assert first == "s2"
        assert q.is_setlist_active is True
        assert q.context_kind == CONTEXT_SETLIST
        assert q.context_name == "Set A"
        assert q.context_index == 1
        assert q.up_next == ["s9"]  # preserved by default

        with pytest.raises(IndexError):
            q.start_setlist("set1", ["s1", "s2"], start_index=2)
        with pytest.raises(IndexError):
            q.start_setlist("set1", ["s1", "s2"], start_index=-1)

        cleared = q.start_setlist("set1", [], name="Empty", clear_up_next=True)
        assert cleared is None
        assert q.is_setlist_active is False
        assert q.context_kind == CONTEXT_LIBRARY
        assert q.up_next == []

    def test_sync_setlist_updates_order_current_removal_and_empty(self):
        q = PlayQueue()
        q.start_setlist("set1", ["s1", "s2", "s3"], name="Old Name", start_index=1)
        # Reorder with s2 still active.
        q.sync_setlist("set1", ["s3", "s1", "s2"], name="New Name", current_song_id="s2")
        assert q.context_name == "New Name"
        assert q.context_index == 2
        assert q.upcoming_context_songs() == []

        # Remove current song s2: context_index stays on preceding song ("s1"),
        # and removing s3 from middle aligns cleanly too.
        q.sync_setlist("set1", ["s3", "s1"], current_song_id="s2")
        assert q.context_songs == ["s3", "s1"]
        assert q.context_index == 1
        assert q.upcoming_context_songs() == []

        q.start_setlist("set1", ["s1", "s2", "s3"], name="New Name", start_index=1)
        q.sync_setlist("set1", ["s1", "s3"], current_song_id="s2")
        assert q.context_index == 0
        assert q.upcoming_context_songs() == ["s3"]

        # Non-matching setlist ID is ignored.
        q.sync_setlist("other", ["s9"], name="Wrong")
        assert q.context_id == "set1"
        assert q.context_name == "New Name"

        # Empty synced list clears setlist context.
        q.sync_setlist("set1", [])
        assert q.is_setlist_active is False
        assert q.context_kind == CONTEXT_LIBRARY

    def test_upcoming_context_songs(self):
        q = PlayQueue()
        assert q.upcoming_context_songs() == []
        q.start_setlist("set1", ["s1", "s2", "s3"], start_index=0)
        assert q.upcoming_context_songs() == ["s2", "s3"]
        q.clear_setlist()
        assert q.upcoming_context_songs() == []


class TestPlayQueueNavigation:
    def test_up_next_takes_priority_without_shifting_setlist_index(self):
        q = PlayQueue()
        q.start_setlist("set1", ["s1", "s2", "s3"], start_index=0)
        q.play_next("s9")

        nxt = q.advance_next(
            "s1", ["s1", "s2", "s3", "s9"],
            repeat_mode=REPEAT_OFF,
            from_auto_finish=True,
        )
        assert nxt == "s9"
        assert q.up_next == []
        assert q.context_index == 0  # unchanged while up_next played
        assert q.history[-1] == "s1"

        # After s9 finishes, setlist resumes at index 1 ("s2").
        nxt2 = q.advance_next(
            "s9", ["s1", "s2", "s3", "s9"],
            repeat_mode=REPEAT_OFF,
            from_auto_finish=True,
        )
        assert nxt2 == "s2"
        assert q.context_index == 1

    def test_setlist_sequential_autoplay_off_and_wrap_and_skip_unplayable(self):
        q = PlayQueue()
        q.start_setlist("set1", ["s1", "s2", "s3"], start_index=0)
        playable = {"s1", "s3"}  # s2 is unplayable

        nxt = q.advance_next(
            "s1", ["s1", "s2", "s3"],
            repeat_mode=REPEAT_OFF,
            from_auto_finish=True,
            is_playable=playable.__contains__,
        )
        assert nxt == "s3"
        assert q.context_index == 2

        # At end with REPEAT_OFF: stops.
        assert q.advance_next(
            "s3", ["s1", "s2", "s3"],
            repeat_mode=REPEAT_OFF,
            from_auto_finish=True,
            is_playable=playable.__contains__,
        ) is None

        # With REPEAT_ALL: wraps back to s1.
        wrapped = q.advance_next(
            "s3", ["s1", "s2", "s3"],
            repeat_mode=REPEAT_ALL,
            from_auto_finish=True,
            is_playable=playable.__contains__,
        )
        assert wrapped == "s1"
        assert q.context_index == 0

    def test_setlist_shuffle_autoplay_one_pass_when_repeat_off(self):
        q = PlayQueue()
        songs = ["s1", "s2", "s3"]
        q.start_setlist("set1", songs, start_index=0)
        seen = []
        current = "s1"
        for _ in range(2):
            nxt = q.advance_next(
                current, songs,
                repeat_mode=REPEAT_OFF,
                shuffle_enabled=True,
                from_auto_finish=True,
            )
            assert nxt is not None
            seen.append(nxt)
            current = nxt
        assert set(seen) == {"s2", "s3"}
        # One full pass completed; auto-finish with REPEAT_OFF stops.
        assert q.advance_next(
            current, songs,
            repeat_mode=REPEAT_OFF,
            shuffle_enabled=True,
            from_auto_finish=True,
        ) is None
        # Manual Next or REPEAT_ALL refills and continues.
        assert q.advance_next(
            current, songs,
            repeat_mode=REPEAT_ALL,
            shuffle_enabled=True,
            from_auto_finish=True,
        ) in {"s1", "s2", "s3"} - {current}

    def test_library_context_auto_finish_and_manual_and_shuffle(self):
        q = PlayQueue()
        songs = ["s1", "s2", "s3"]
        # REPEAT_OFF auto-finish stops when queue is empty.
        assert q.advance_next(
            "s1", songs, repeat_mode=REPEAT_OFF, from_auto_finish=True,
        ) is None

        # Manual Next advances sequentially and wraps only on REPEAT_ALL.
        assert q.advance_next("s1", songs, repeat_mode=REPEAT_OFF) == "s2"
        assert q.advance_next("s3", songs, repeat_mode=REPEAT_OFF) is None
        assert q.advance_next("s3", songs, repeat_mode=REPEAT_ALL) == "s1"

        # Shuffle excludes current song.
        picked = set()
        for _ in range(2):
            nxt = q.advance_next(
                "s1", songs, repeat_mode=REPEAT_ALL, shuffle_enabled=True,
            )
            assert nxt in {"s2", "s3"}
            picked.add(nxt)
        assert picked == {"s2", "s3"}

    def test_advance_previous_uses_history_then_falls_back(self):
        q = PlayQueue()
        songs = ["s1", "s2", "s3"]
        q.start_setlist("set1", songs, start_index=0)
        q.enqueue("s9")
        # s1 -> up_next(s9) -> s2
        assert q.advance_next("s1", songs, repeat_mode=REPEAT_OFF) == "s9"
        assert q.advance_next("s9", songs, repeat_mode=REPEAT_OFF) == "s2"

        # First Previous returns s9 from history; next returns s1 and updates index.
        assert q.advance_previous("s2", songs, repeat_mode=REPEAT_OFF) == "s9"
        assert q.advance_previous("s9", songs, repeat_mode=REPEAT_OFF) == "s1"
        assert q.context_index == 0

        # History is now empty: at start with REPEAT_OFF -> None; REPEAT_ALL -> wraps.
        assert q.advance_previous("s1", songs, repeat_mode=REPEAT_OFF) is None
        assert q.advance_previous("s1", songs, repeat_mode=REPEAT_ALL) == "s3"
        assert q.context_index == 2

    def test_advance_previous_skips_current_and_unplayable_history(self):
        q = PlayQueue()
        q.record_history("s1")
        q.record_history("dead")
        q.record_history("s3")
        nxt = q.advance_previous(
            "s3", ["s1", "s2", "s3"],
            repeat_mode=REPEAT_OFF,
            is_playable={"s1", "s2", "s3"}.__contains__,
        )
        assert nxt == "s1"


class TestPlayQueueMaintenanceAndSerialization:
    def test_remove_song_adjusts_index_zero_to_minus_one_and_clears_when_empty(self):
        q = PlayQueue()
        q.enqueue_many(["s1", "s2", "s1"])
        q.record_history("s1")
        q.record_history("s2")
        q.start_setlist("set1", ["s1", "s2"], start_index=0)

        q.remove_song("s1")
        assert q.up_next == ["s2"]
        assert q.history == ["s2"]
        assert q.context_songs == ["s2"]
        assert q.context_index == -1  # next advance should return new index 0 ("s2")
        assert q.advance_next("s1", ["s2"], repeat_mode=REPEAT_OFF) == "s2"

        q.remove_song("s2")
        assert q.up_next == []
        assert q.is_setlist_active is False

    def test_prune_removes_deleted_ids_across_all_lists(self):
        q = PlayQueue()
        q.enqueue_many(["s1", "gone", "s2"])
        q.record_history("gone")
        q.record_history("s1")
        q.start_setlist("set1", ["gone", "s1", "s2"], start_index=2)
        q.prune(["s1", "s2"])
        assert q.up_next == ["s1", "s2"]
        assert q.history == ["s1"]
        assert q.context_songs == ["s1", "s2"]
        assert q.context_index == 1

    def test_round_trip_and_defensive_from_dict(self):
        q = PlayQueue()
        q.enqueue_many(["s2", "s3"])
        q.record_history("s1")
        q.start_setlist("set1", ["s1", "s2", "s3"], name="Acoustic", start_index=1)
        data = q.to_dict()
        restored = PlayQueue.from_dict(data)
        assert restored.up_next == ["s2", "s3"]
        assert restored.history == ["s1"]
        assert restored.context_kind == CONTEXT_SETLIST
        assert restored.context_id == "set1"
        assert restored.context_name == "Acoustic"
        assert restored.context_songs == ["s1", "s2", "s3"]
        assert restored.context_index == 1

        # Defensive checks on malformed dicts.
        assert PlayQueue.from_dict("not a dict").up_next == []
        bad_setlist = PlayQueue.from_dict({
            "up_next": ["s1", "", 99],
            "context_kind": CONTEXT_SETLIST,
            "context_id": "",
            "context_songs": ["s1"],
            "context_index": True,  # bool must not pass int check
        })
        assert bad_setlist.up_next == ["s1"]
        assert bad_setlist.is_setlist_active is False

        out_of_range = PlayQueue.from_dict({
            "context_kind": CONTEXT_SETLIST,
            "context_id": "set1",
            "context_songs": ["s1", "s2"],
            "context_index": 99,
        })
        assert out_of_range.is_setlist_active is True
        assert out_of_range.context_index == -1


# ---------------------------------------------------------------------------
# PlayQueueWidget tests
# ---------------------------------------------------------------------------


class TestPlayQueueWidget:
    def test_empty_and_populated_state_buttons_and_signals(self, qapp):
        songs = _make_songs(4)
        lib = _make_mock_library(songs)
        q = PlayQueue()
        widget = PlayQueueWidget()
        try:
            widget.refresh(q, lib)
            assert not widget._empty_label.isHidden()
            assert not widget._clear_btn.isEnabled()
            assert not widget._up_btn.isEnabled()
            assert not widget._down_btn.isEnabled()
            assert not widget._remove_btn.isEnabled()
            assert widget._context_bar.isHidden()
            assert widget._context_section.isHidden()

            q.enqueue_many(["s1", "s2"])
            q.start_setlist("set1", ["s3", "s4"], name="Encore", start_index=0)
            widget.refresh(q, lib)
            assert widget._empty_label.isHidden()
            assert widget._clear_btn.isEnabled()
            assert widget._up_next_list.count() == 2
            assert not widget._context_bar.isHidden()
            assert not widget._context_section.isHidden()
            assert widget._context_list.count() == 1  # upcoming ["s4"]

            widget._up_next_list.setCurrentRow(0)
            assert not widget._up_btn.isEnabled()
            assert widget._down_btn.isEnabled()
            assert widget._remove_btn.isEnabled()

            widget._up_next_list.setCurrentRow(1)
            assert widget._up_btn.isEnabled()
            assert not widget._down_btn.isEnabled()

            moves = []
            removes = []
            clears = []
            stops = []
            play_idx = []
            play_ctx = []
            widget.move_item_requested.connect(lambda a, b: moves.append((a, b)))
            widget.remove_item_requested.connect(removes.append)
            widget.clear_queue_requested.connect(lambda: clears.append(True))
            widget.stop_setlist_requested.connect(lambda: stops.append(True))
            widget.play_queued_index_requested.connect(play_idx.append)
            widget.play_context_song_requested.connect(play_ctx.append)

            widget._up_btn.click()
            assert widget._up_next_list.currentRow() == 0
            widget._up_next_list.setCurrentRow(0)
            widget._down_btn.click()
            assert widget._up_next_list.currentRow() == 1
            widget._remove_btn.click()
            widget._clear_btn.click()
            widget._stop_setlist_btn.click()
            widget.item_double_clicked = widget._up_next_list.item(0)
            widget._up_next_list.itemDoubleClicked.emit(widget._up_next_list.item(0))
            widget._context_list.itemDoubleClicked.emit(widget._context_list.item(0))

            assert moves == [(1, 0), (0, 1)]
            assert removes == [1]
            assert clears == [True]
            assert stops == [True]
            assert play_idx == [0]
            assert play_ctx == ["s4"]
        finally:
            widget.close()
            widget.deleteLater()

    def test_qt_rows_moved_index_translation_and_theme(self, qapp):
        songs = _make_songs(3)
        lib = _make_mock_library(songs)
        q = PlayQueue()
        q.enqueue_many(["s1", "s2", "s3"])
        widget = PlayQueueWidget()
        try:
            widget.refresh(q, lib)
            moves = []
            widget.move_item_requested.connect(lambda a, b: moves.append((a, b)))
            # Simulate Qt moving row 2 upward to before row 0 -> index 0.
            widget._on_rows_moved(None, 2, 2, None, 0)
            # Simulate Qt moving row 0 downward past row 2 (destRow=3) -> index 2.
            widget._on_rows_moved(None, 0, 0, None, 3)
            assert moves == [(2, 0), (0, 2)]

            widget.apply_theme("dark", DARK_COLORS)
            widget.apply_theme("light", LIGHT_COLORS)
        finally:
            widget.close()
            widget.deleteLater()


# ---------------------------------------------------------------------------
# LibraryPanel and MainWindow integration tests
# ---------------------------------------------------------------------------


class _FakeStemLoadWorker(QThread):
    completed = Signal(dict, int)
    error = Signal(str)
    finished = Signal()

    def __init__(self, stem_paths) -> None:
        super().__init__()
        self.stem_paths = dict(stem_paths)
        self.generation = 0
        self.song_id = ""
        self.source_stem_names = ()
        self.running = False

    def start(self) -> None:
        self.running = True

    def isRunning(self) -> bool:
        return self.running

    def wait(self) -> None:
        self.running = False

    def setParent(self, _parent) -> None:
        pass

    def deleteLater(self) -> None:
        pass


class _QuietMainWindow(main_window_module.MainWindow):
    """MainWindow with startup prune disabled for isolated PySide6 3.14 tests."""

    def _prune_incomplete_songs(self) -> None:
        pass


def _build_real_library(tmp_path, count: int = 3) -> SongLibrary:
    lib = SongLibrary(str(tmp_path / "data"))
    for i in range(1, count + 1):
        wav = tmp_path / f"song_{i}.wav"
        _write_wav(wav, value=i / 10.0)
        song = lib.add_song(f"Song {i}", f"Artist {i}", str(wav))
        _write_wav(f"{song.stems_path}/vocals.wav", value=i / 10.0)
        lib.update_song(song.id, model_used="htdemucs")
    return lib


@pytest.fixture
def window(qapp, tmp_path):
    lib = _build_real_library(tmp_path, count=3)
    settings = open_settings()
    settings.clear()
    player = player_module.MultiTrackPlayer()
    with patch.object(main_window_module.QTimer, "singleShot"):
        win = _QuietMainWindow(lib, player, MagicMock())
    yield win
    if hasattr(win, "_shutdown_stem_loads"):
        win._shutdown_stem_loads()
    win._separation_queue.shutdown(0)
    win._player_controls.shutdown()
    player.shutdown(wait_ms=5000)
    win.setParent(None)
    win.deleteLater()
    qapp.processEvents()
    settings.clear()


class TestLibraryPanelQueueIntegration:
    def test_context_menu_queue_actions_do_not_change_selection(self, qapp):
        songs = _make_songs(3)
        lib = _make_mock_library(songs)
        panel = LibraryPanel(lib)
        try:
            panel._list.setCurrentRow(0)
            selected = []
            play_next = []
            add_queue = []
            panel.song_selected.connect(selected.append)
            panel.play_next_requested.connect(play_next.append)
            panel.add_to_queue_requested.connect(add_queue.append)

            item2 = panel._list.item(1)
            menu = QMenu()
            with patch("src.ui.library_panel.QMenu", return_value=menu), \
                    patch.object(panel._list, "itemAt", return_value=item2), \
                    patch.object(panel._list, "mapToGlobal", return_value=QPoint(0, 0)), \
                    patch.object(menu, "exec") as mock_exec:
                panel._on_context_menu(QPoint(0, 0))
                mock_exec.assert_called_once()
                texts = [a.text() for a in menu.actions() if not a.isSeparator()]
                assert texts == [
                    "Play Next", "Add to Queue", "Add to Setlist", "Edit...",
                ]
                menu.actions()[0].trigger()
                menu.actions()[1].trigger()

            assert selected == []
            assert panel._list.currentRow() == 0
            assert play_next == ["s2"]
            assert add_queue == ["s2"]
        finally:
            panel.close()
            panel.deleteLater()

    def test_queue_toggle_switches_stack_and_updates_header_and_count(self, qapp):
        songs = _make_songs(3)
        lib = _make_mock_library(songs)
        panel = LibraryPanel(lib)
        try:
            assert panel.is_queue_visible is False
            assert panel._header_label.text() == "Library"
            q = PlayQueue()
            q.enqueue_many(["s1", "s2"])
            panel.update_queue_display(q)
            assert panel._queue_count_label.text() == "2 queued"
            assert "2 up next" in panel._queue_btn.toolTip()

            panel.toggle_queue_visible()
            assert panel.is_queue_visible is True
            assert panel._header_label.text() == "Play Queue"
        finally:
            panel.close()
            panel.deleteLater()


class TestMainWindowPlayQueueWiring:
    def test_up_next_autoplay_with_repeat_off_and_repeat_one_precedence(self, window):
        workers = []

        def make_worker(paths):
            w = _FakeStemLoadWorker(paths)
            workers.append(w)
            return w

        s1, s2, s3 = [s.id for s in window._library.songs]
        with patch.object(
            main_window_module, "StemLoadWorker", side_effect=make_worker, create=True,
        ), patch.object(window._player, "play") as play:
            window._library_panel.select_song(s1)
            arrays, sr = player_module.read_stem_files(workers[-1].stem_paths)
            workers[-1].completed.emit(arrays, sr)

            window.play_next(s3)
            assert window.play_queue.up_next == [s3]

            # REPEAT_ONE replays current song without consuming up_next.
            window._library_panel.set_repeat_mode(REPEAT_ONE)
            with patch.object(window._player, "seek") as seek:
                window._on_play_finished()
                seek.assert_called_once_with(0)
            assert window.play_queue.up_next == [s3]

            # REPEAT_OFF auto-finishes into up_next.
            window._library_panel.set_repeat_mode(REPEAT_OFF)
            window._on_play_finished()
            assert window.play_queue.up_next == []
            arrays, sr = player_module.read_stem_files(workers[-1].stem_paths)
            workers[-1].completed.emit(arrays, sr)
            assert window._current_song_id == s3
            assert play.called

            # Next finish with empty queue and REPEAT_OFF stops cleanly.
            play.reset_mock()
            window._on_play_finished()
            assert len(workers) == 2

    def test_setlist_autoplay_interleaved_up_next_and_manual_select_reset(self, window):
        workers = []

        def make_worker(paths):
            w = _FakeStemLoadWorker(paths)
            workers.append(w)
            return w

        s1, s2, s3 = [s.id for s in window._library.songs]
        sl = window._library.create_setlist("Lesson 1", [s1, s2])

        with patch.object(
            main_window_module, "StemLoadWorker", side_effect=make_worker, create=True,
        ), patch.object(window._player, "play"):
            assert window.play_setlist(sl.id) is True
            arrays, sr = player_module.read_stem_files(workers[-1].stem_paths)
            workers[-1].completed.emit(arrays, sr)
            assert window._current_song_id == s1
            assert window.play_queue.is_setlist_active is True

            # Queue s3 with Play Next; finishing s1 plays s3, then finishing s3
            # resumes setlist at s2.
            window.play_next(s3)
            window._library_panel.set_repeat_mode(REPEAT_OFF)
            window._on_play_finished()
            arrays, sr = player_module.read_stem_files(workers[-1].stem_paths)
            workers[-1].completed.emit(arrays, sr)
            assert window._current_song_id == s3
            assert window.play_queue.is_setlist_active is True

            window._on_play_finished()
            arrays, sr = player_module.read_stem_files(workers[-1].stem_paths)
            workers[-1].completed.emit(arrays, sr)
            assert window._current_song_id == s2

            # End of setlist with REPEAT_OFF stops; with REPEAT_ALL wraps to s1.
            count_before = len(workers)
            window._on_play_finished()
            assert len(workers) == count_before

            window._library_panel.set_repeat_mode(REPEAT_ALL)
            window._on_play_finished()
            arrays, sr = player_module.read_stem_files(workers[-1].stem_paths)
            workers[-1].completed.emit(arrays, sr)
            assert window._current_song_id == s1

            # Manual library selection clears setlist context but preserves up_next.
            window.add_to_queue(s2)
            window._library_panel.select_song(s3)
            assert window.play_queue.is_setlist_active is False
            assert window.play_queue.up_next == [s2]

    def test_remove_song_prunes_queue_and_session_round_trip(self, qapp, tmp_path):
        lib = _build_real_library(tmp_path, count=3)
        s1, s2, s3 = [s.id for s in lib.songs]
        sl = lib.create_setlist("Gig", [s1, s2, s3])
        settings = open_settings()
        settings.clear()

        def make_win(library):
            player = player_module.MultiTrackPlayer()
            with patch.object(main_window_module.QTimer, "singleShot"):
                return _QuietMainWindow(library, player, MagicMock()), player

        win1, player1 = make_win(lib)
        try:
            win1.play_queue.start_setlist(sl.id, sl.song_ids, name=sl.name, start_index=0)
            win1.add_to_queue(s3)
            win1.play_queue.enqueue(s2)
            win1._on_song_removed(s2)
            assert win1.play_queue.up_next == [s3]
            win1._save_session()
            saved_queue = json.loads(settings.value("session/play_queue"))
            assert saved_queue["up_next"] == [s3]
        finally:
            win1._separation_queue.shutdown(0)
            win1._player_controls.shutdown()
            player1.shutdown(wait_ms=5000)
            win1.setParent(None)
            win1.deleteLater()
            qapp.processEvents()

        # Reorder setlist on disk before restoring session into win2.
        lib.reorder_setlist(sl.id, [s3, s1, s2])
        win2, player2 = make_win(lib)
        try:
            win2._restore_session()
            assert win2.play_queue.up_next == [s3]
            assert win2.play_queue.is_setlist_active is True
            assert win2.play_queue.context_songs == [s3, s1, s2]
        finally:
            win2._separation_queue.shutdown(0)
            win2._player_controls.shutdown()
            player2.shutdown(wait_ms=5000)
            win2.setParent(None)
            win2.deleteLater()
            qapp.processEvents()
            settings.clear()

        # Deleting the setlist clears setlist context on restore.
        lib.delete_setlist(sl.id)
        settings.setValue("session/play_queue", json.dumps({
            "up_next": [s1],
            "context_kind": CONTEXT_SETLIST,
            "context_id": sl.id,
            "context_name": "Gig",
            "context_songs": [s1, s2],
            "context_index": 0,
            "history": [],
        }))
        win3, player3 = make_win(lib)
        try:
            win3._restore_session()
            assert win3.play_queue.up_next == [s1]
            assert win3.play_queue.is_setlist_active is False
        finally:
            win3._separation_queue.shutdown(0)
            win3._player_controls.shutdown()
            player3.shutdown(wait_ms=5000)
            win3.setParent(None)
            win3.deleteLater()
            qapp.processEvents()
            settings.clear()

    def test_q_shortcut_and_view_action_toggle_queue_view(self, window, qapp):
        assert window._play_queue_action.isChecked() is False
        assert window._library_panel.is_queue_visible is False

        # Trigger via registered Q shortcut.
        q_shortcuts = [
            s for s in window.findChildren(main_window_module.QShortcut)
            if s.key() == QKeySequence(Qt.Key.Key_Q)
        ]
        assert len(q_shortcuts) == 1
        with patch.object(window, "_text_input_has_focus", return_value=False):
            q_shortcuts[0].activated.emit()
        assert window._library_panel.is_queue_visible is True
        assert window._play_queue_action.isChecked() is True

        # Trigger via View > Play &Queue action.
        window._play_queue_action.trigger()
        assert window._library_panel.is_queue_visible is False
        assert window._play_queue_action.isChecked() is False

    def test_move_buttons_follow_moved_song_and_edit_refreshes_queue(self, window):
        s1, s2, s3 = [s.id for s in window._library.songs]
        window.add_to_queue(s1)
        window.add_to_queue(s2)
        window.add_to_queue(s3)

        qw = window._library_panel.queue_widget
        qw._up_next_list.setCurrentRow(2)
        qw._up_btn.click()
        assert window.play_queue.up_next == [s1, s3, s2]
        assert qw._up_next_list.currentRow() == 1
        qw._up_btn.click()
        assert window.play_queue.up_next == [s3, s1, s2]
        assert qw._up_next_list.currentRow() == 0

        # Editing a song notifies MainWindow so PlayQueueWidget updates labels.
        window._library.update_song(s3, title="Renamed Song 3")
        window._library_panel.song_edited.emit(s3)
        assert "Renamed Song 3" in qw._up_next_list.item(0).text()

    def test_one_song_setlist_keeps_banner_at_top_with_stretch_empty_label(self, qapp):
        songs = _make_songs(1)
        lib = _make_mock_library(songs)
        q = PlayQueue()
        q.start_setlist("set1", ["s1"], name="Solo Set")
        widget = PlayQueueWidget()
        try:
            widget.resize(260, 500)
            widget.show()
            widget.refresh(q, lib)
            qapp.processEvents()
            assert widget._context_bar.isVisible()
            assert widget._context_bar.geometry().top() < 30
        finally:
            widget.close()
            widget.deleteLater()

"""Tests for setlists and tags UI, drag-reorder, and EditSongDialog fields."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QModelIndex, QPoint, QThread, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractItemView, QApplication, QMenu

from src import player as player_module
from src.app_settings import open_settings
from src.library import Song, SongLibrary
from src.play_queue import REPEAT_OFF
from src.ui import main_window as main_window_module
from src.ui.library_panel import EditSongDialog, LibraryPanel


def _write_wav(path, value: float = 0.1, sample_rate: int = 44100) -> None:
    data = np.full((64, 2), value, dtype=np.float32)
    sf.write(str(path), data, sample_rate)


def _build_real_library(tmp_path, count: int = 4) -> SongLibrary:
    lib = SongLibrary(str(tmp_path / "data"))
    for i in range(1, count + 1):
        wav = tmp_path / f"song_{i}.wav"
        _write_wav(wav, value=i / 10.0)
        song = lib.add_song(f"Song {i}", f"Artist {i}", str(wav))
        _write_wav(f"{song.stems_path}/vocals.wav", value=i / 10.0)
        lib.update_song(song.id, model_used="htdemucs")
    return lib


def _shown_ids(panel: LibraryPanel) -> list[str]:
    return [
        panel._list.item(i).data(Qt.ItemDataRole.UserRole)
        for i in range(panel._list.count())
    ]


def _run_context_menu(panel: LibraryPanel, song_id: str, label: str) -> None:
    """Open the context menu on *song_id*'s row and trigger *label*.

    Qt's modal exec never returns offscreen, so the menu instance's exec is
    replaced; *label* may also sit in the Add to Setlist submenu.
    """
    menu = QMenu()
    with patch("src.ui.library_panel.QMenu", return_value=menu), \
            patch.object(
                panel._list, "itemAt",
                return_value=panel._item_for_song(song_id),
            ), \
            patch.object(
                panel._list, "mapToGlobal", return_value=QPoint(0, 0),
            ), \
            patch.object(menu, "exec"):
        panel._on_context_menu(QPoint(0, 0))
        actions = list(menu.actions())
        for action in menu.actions():
            if action.menu() is not None:
                actions.extend(action.menu().actions())
        next(a for a in actions if a.text() == label).trigger()


@pytest.fixture(scope="module")
def qapp():
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


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


@pytest.fixture
def window(qapp, tmp_path):
    lib = _build_real_library(tmp_path, count=4)
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


# ---------------------------------------------------------------------------
# EditSongDialog tags and practice_status
# ---------------------------------------------------------------------------


class TestEditSongDialogTagsAndStatus:
    def test_prefills_tags_and_practice_status(self, qapp):
        song = Song(
            id="s1",
            title="Song 1",
            artist="Artist 1",
            original_path="",
            stems_path="",
            model_used="htdemucs",
            date_added="",
            tags=["gig", "warm up"],
            practice_status="practicing",
        )
        dlg = EditSongDialog(song)
        try:
            assert dlg.title == "Song 1"
            assert dlg.artist == "Artist 1"
            assert dlg.tags == ["gig", "warm up"]
            assert dlg.practice_status == "practicing"
            assert dlg._tags_edit.text() == "gig, warm up"
        finally:
            dlg.close()

    def test_tags_splitting_dedup_and_status_selection(self, qapp):
        song = Song(
            id="s1",
            title="Song 1",
            artist="Artist 1",
            original_path="",
            stems_path="",
            model_used="htdemucs",
            date_added="",
        )
        dlg = EditSongDialog(song)
        try:
            assert dlg.practice_status == ""
            dlg._tags_edit.setText(" Gig , learning, gig, , Acoustic ")
            assert dlg.tags == ["Gig", "learning", "Acoustic"]

            for status in ("learning", "practicing", "ready", ""):
                idx = dlg._status_combo.findData(status)
                assert idx >= 0
                dlg._status_combo.setCurrentIndex(idx)
                assert dlg.practice_status == status
        finally:
            dlg.close()

    def test_accepting_edit_persists_tags_and_status_on_library(self, qapp, tmp_path):
        lib = _build_real_library(tmp_path, count=2)
        s1 = lib.songs[0].id
        panel = LibraryPanel(lib)
        edited = []
        panel.song_edited.connect(edited.append)
        try:
            panel._list.setCurrentRow(0)

            def fake_exec(dlg):
                dlg._title_edit.setText("Song One")
                dlg._artist_edit.setText("Band One")
                dlg._tags_edit.setText("gig, Blues")
                dlg._status_combo.setCurrentIndex(
                    dlg._status_combo.findData("ready")
                )
                return True

            with patch.object(EditSongDialog, "exec", fake_exec):
                panel._on_edit_song(s1)

            updated = lib.get_song(s1)
            assert updated is not None
            assert updated.title == "Song One"
            assert updated.artist == "Band One"
            assert updated.tags == ["gig", "Blues"]
            assert updated.practice_status == "ready"
            assert edited == [s1]
            combo_items = [
                panel._collection_combo.itemData(i)
                for i in range(panel._collection_combo.count())
            ]
            assert ("tag", "Blues") in combo_items
            assert ("tag", "gig") in combo_items
        finally:
            panel.close()
            panel.deleteLater()


# ---------------------------------------------------------------------------
# Collection filter, search, setlist bar, and drag-reorder state
# ---------------------------------------------------------------------------


class TestCollectionFilterAndReorderMode:
    def test_combo_lists_setlists_and_tags_and_filters_songs(self, qapp, tmp_path):
        lib = _build_real_library(tmp_path, count=4)
        s1, s2, s3, s4 = [s.id for s in lib.songs]
        lib.set_song_tags(s1, ["gig", "jazz"])
        lib.set_song_tags(s3, ["Gig"])
        sl = lib.create_setlist("Friday Set", [s4, s2])

        panel = LibraryPanel(lib)
        try:
            combo = panel._collection_combo
            texts = [combo.itemText(i) for i in range(combo.count())]
            assert "All Songs" in texts
            assert "Setlist: Friday Set (2)" in texts
            assert "Tag: gig" in texts
            assert "Tag: jazz" in texts

            # Filter by tag (case-insensitive match on "gig" -> s1, s3).
            panel.set_collection_filter("tag", "gig")
            assert panel.filter_kind == "tag"
            assert _shown_ids(panel) == [s1, s3]
            # Browsing does not change what plays next.
            assert panel.song_ids_in_order() == [s1, s2, s3, s4]
            assert panel._setlist_bar.isHidden()
            assert (
                panel._list.dragDropMode()
                == QAbstractItemView.DragDropMode.NoDragDrop
            )

            # Filter by setlist -> exact setlist order [s4, s2] + Play Setlist bar.
            panel.set_collection_filter("setlist", sl.id)
            assert panel.filter_kind == "setlist"
            assert _shown_ids(panel) == [s4, s2]
            assert not panel._setlist_bar.isHidden()
            assert panel._play_setlist_btn.isEnabled()
            assert (
                panel._list.dragDropMode()
                == QAbstractItemView.DragDropMode.InternalMove
            )

            # Typing in search temporarily disables drag-reorder.
            panel._search_edit.setText("Song 4")
            assert (
                panel._list.dragDropMode()
                == QAbstractItemView.DragDropMode.NoDragDrop
            )
            panel._search_edit.clear()
            assert (
                panel._list.dragDropMode()
                == QAbstractItemView.DragDropMode.InternalMove
            )

            # All Songs search matches song tags too.
            panel.set_collection_filter("all")
            panel._search_edit.setText("jazz")
            visible = [
                panel._list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(panel._list.count())
                if not panel._list.item(i).isHidden()
            ]
            assert visible == [s1]
        finally:
            panel.close()
            panel.deleteLater()

    def test_deleted_setlist_or_removed_tag_falls_back_to_all(self, qapp, tmp_path):
        lib = _build_real_library(tmp_path, count=2)
        s1, s2 = [s.id for s in lib.songs]
        lib.set_song_tags(s1, ["solo"])
        sl = lib.create_setlist("Temp Set", [s1, s2])
        panel = LibraryPanel(lib)
        try:
            panel.set_collection_filter("tag", "solo")
            assert panel.filter_kind == "tag"
            lib.set_song_tags(s1, [])
            panel.refresh()
            assert panel.filter_kind == "all"
            assert panel.filter_value is None

            panel.set_collection_filter("setlist", sl.id)
            assert panel.filter_kind == "setlist"
            lib.delete_setlist(sl.id)
            panel.refresh()
            assert panel.filter_kind == "all"
        finally:
            panel.close()
            panel.deleteLater()


# ---------------------------------------------------------------------------
# Setlist CRUD, drag-reorder, context menu, and read-only safety
# ---------------------------------------------------------------------------


class TestSetlistManagementAndContextMenu:
    def test_create_rename_delete_add_remove_and_drag_reorder(self, qapp, tmp_path):
        lib = _build_real_library(tmp_path, count=3)
        s1, s2, s3 = [s.id for s in lib.songs]
        panel = LibraryPanel(lib)
        changed = []
        panel.setlist_changed.connect(changed.append)
        try:
            # Create setlist.
            with patch(
                "src.ui.library_panel.QInputDialog.getText",
                return_value=("Acoustic Set", True),
            ):
                panel._on_new_setlist([s1, s2, s3])
            assert len(lib.setlists) == 1
            sl = lib.setlists[0]
            assert sl.name == "Acoustic Set"
            assert sl.song_ids == [s1, s2, s3]
            assert changed == [sl.id]

            # Rename setlist while active.
            panel.set_collection_filter("setlist", sl.id)
            with patch(
                "src.ui.library_panel.QInputDialog.getText",
                return_value=("Friday Night", True),
            ):
                panel._on_rename_setlist()
            assert lib.get_setlist(sl.id).name == "Friday Night"

            # Simulated Qt drag-reorder: 2 -> 0 (upward) and 0 -> 2 (dest_row=3).
            panel._on_list_rows_moved(None, 2, 2, None, 0)
            assert lib.get_setlist(sl.id).song_ids == [s3, s1, s2]
            panel._on_list_rows_moved(None, 0, 0, None, 3)
            assert lib.get_setlist(sl.id).song_ids == [s1, s2, s3]

            # Remove from setlist (keeps song in library).
            panel._on_remove_song_from_setlist(sl.id, s2)
            assert lib.get_setlist(sl.id).song_ids == [s1, s3]
            assert lib.get_song(s2) is not None

            # Delete setlist returns filter to All Songs.
            with patch(
                "src.ui.library_panel.QMessageBox.question",
                return_value=main_window_module.QMessageBox.StandardButton.Yes,
            ):
                panel._on_delete_setlist()
            assert lib.setlists == []
            assert panel.filter_kind == "all"
        finally:
            panel.close()
            panel.deleteLater()

    def test_context_menu_add_reorder_and_remove_from_setlist(self, qapp, tmp_path):
        lib = _build_real_library(tmp_path, count=3)
        s1, s2, s3 = [s.id for s in lib.songs]
        sl = lib.create_setlist("Set A", [s1, s2])
        panel = LibraryPanel(lib)
        try:
            # In All Songs view: Add to Setlist -> Set A works for s3 without changing
            # selection.
            panel._list.setCurrentRow(0)
            selected = []
            panel.song_selected.connect(selected.append)
            _run_context_menu(panel, s3, "Set A")
            assert lib.get_setlist(sl.id).song_ids == [s1, s2, s3]
            assert selected == []
            assert panel._list.currentRow() == 0

            # In Setlist view: Move Up / Move Down / Remove from Setlist appear.
            panel.set_collection_filter("setlist", sl.id)
            _run_context_menu(panel, s2, "Move Up in Setlist")
            assert lib.get_setlist(sl.id).song_ids == [s2, s1, s3]
            _run_context_menu(panel, s2, "Move Down in Setlist")
            assert lib.get_setlist(sl.id).song_ids == [s1, s2, s3]
            _run_context_menu(panel, s2, "Remove from Setlist")
            assert lib.get_setlist(sl.id).song_ids == [s1, s3]
        finally:
            panel.close()
            panel.deleteLater()

    def test_model_row_move_persists_order(self, qapp, tmp_path):
        """The move Qt performs on a drop reaches the stored setlist."""
        lib = _build_real_library(tmp_path, count=3)
        s1, s2, s3 = [s.id for s in lib.songs]
        sl = lib.create_setlist("Set A", [s1, s2, s3])
        panel = LibraryPanel(lib)
        try:
            panel.set_collection_filter("setlist", sl.id)
            model = panel._list.model()
            assert model.moveRow(QModelIndex(), 0, QModelIndex(), 3)
            assert lib.get_setlist(sl.id).song_ids == [s2, s3, s1]
            assert _shown_ids(panel) == [s2, s3, s1]
        finally:
            panel.close()
            panel.deleteLater()

    def test_drag_does_not_select_but_click_does(self, qapp, tmp_path):
        lib = _build_real_library(tmp_path, count=3)
        s1, s2, s3 = [s.id for s in lib.songs]
        sl = lib.create_setlist("Set A", [s1, s2, s3])
        panel = LibraryPanel(lib)
        panel.resize(260, 600)
        panel.show()
        qapp.processEvents()
        try:
            panel.set_collection_filter("setlist", sl.id)
            assert panel.select_song(s1)
            selected = []
            panel.song_selected.connect(selected.append)
            rows = panel._list
            view = rows.viewport()

            # Press and drag: the real startDrag runs Qt's modal drag loop,
            # so stand in for the base implementation and drop during it.
            center = rows.visualItemRect(panel._item_for_song(s3)).center()
            QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=center)
            assert rows.currentRow() == 2
            assert selected == []

            def drop(_list, _actions):
                rows.model().moveRow(QModelIndex(), 2, QModelIndex(), 0)

            with patch("src.ui.library_panel.QListWidget.startDrag", drop):
                rows.startDrag(Qt.DropAction.MoveAction)
            QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=center)
            assert lib.get_setlist(sl.id).song_ids == [s3, s1, s2]
            assert selected == []
            assert rows.currentItem().data(Qt.ItemDataRole.UserRole) == s1

            # A plain click still loads the song.
            center = rows.visualItemRect(panel._item_for_song(s2)).center()
            QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=center)
            assert selected == [s2]
        finally:
            panel.close()
            panel.deleteLater()

    def test_auto_advance_during_press_and_drag(self, qapp, tmp_path):
        """A song that starts mid-press loads, and is not what gets moved."""
        lib = _build_real_library(tmp_path, count=4)
        s1, s2, s3, s4 = [s.id for s in lib.songs]
        sl = lib.create_setlist("Set A", [s1, s2, s3, s4])
        panel = LibraryPanel(lib)
        panel.resize(260, 600)
        panel.show()
        qapp.processEvents()
        try:
            panel.set_collection_filter("setlist", sl.id)
            panel.select_song(s1)
            selected = []
            panel.song_selected.connect(selected.append)
            rows = panel._list
            view = rows.viewport()

            # Held, then released without a drag.
            center = rows.visualItemRect(panel._item_for_song(s4)).center()
            QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=center)
            panel.select_song(s2)
            assert selected == [s2]
            QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=center)
            assert selected == [s2]

            # Advancing both before the drag starts and during it.
            center = rows.visualItemRect(panel._item_for_song(s3)).center()
            QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=center)
            panel.select_song(s4)
            moved = []

            def drag(_list, _actions):
                moved.append(rows.currentItem().data(Qt.ItemDataRole.UserRole))
                panel.select_song(s1)
                rows.dropEvent(MagicMock())

            def drop(_list, _event):
                assert len(rows.selectedIndexes()) == 1
                row = rows.selectedIndexes()[0].row()
                rows.model().moveRow(QModelIndex(), row, QModelIndex(), 0)

            base = "src.ui.library_panel.QListWidget."
            with patch(base + "startDrag", drag), \
                    patch(base + "dropEvent", drop):
                rows.startDrag(Qt.DropAction.MoveAction)
            QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=center)
            assert moved == [s3]
            assert lib.get_setlist(sl.id).song_ids == [s3, s1, s2, s4]
            assert selected == [s2, s4, s1]
            assert rows.currentItem().data(Qt.ItemDataRole.UserRole) == s1
        finally:
            panel.close()
            panel.deleteLater()

    def test_setlist_name_with_ampersand_in_menu(self, qapp, tmp_path):
        lib = _build_real_library(tmp_path, count=1)
        sl = lib.create_setlist("Rock & Roll", [])
        panel = LibraryPanel(lib)
        try:
            song_id = lib.songs[0].id
            _run_context_menu(panel, song_id, "Rock && Roll")
            assert lib.get_setlist(sl.id).song_ids == [song_id]
        finally:
            panel.close()
            panel.deleteLater()

    def test_playing_song_is_highlighted_after_filter_round_trip(
        self, qapp, tmp_path,
    ):
        lib = _build_real_library(tmp_path, count=2)
        s1 = lib.songs[0].id
        sl = lib.create_setlist("Empty", [])
        panel = LibraryPanel(lib)
        try:
            panel.select_song(s1)
            panel.set_playing_song(s1)
            selected = []
            panel.song_selected.connect(selected.append)
            panel.set_collection_filter("setlist", sl.id)
            assert panel._list.currentItem() is None
            panel.set_collection_filter("all")
            current = panel._list.currentItem()
            assert current.data(Qt.ItemDataRole.UserRole) == s1
            assert panel._remove_btn.isEnabled()
            assert selected == []
        finally:
            panel.close()
            panel.deleteLater()

    def test_empty_setlist_says_so(self, qapp, tmp_path):
        lib = _build_real_library(tmp_path, count=1)
        sl = lib.create_setlist("Empty", [])
        panel = LibraryPanel(lib)
        try:
            panel.set_collection_filter("setlist", sl.id)
            assert not panel._no_match_label.isHidden()
            assert panel._no_match_label.text() == "This setlist is empty"
            assert not panel._play_setlist_btn.isEnabled()
            panel.set_collection_filter("all")
            assert panel._no_match_label.isHidden()
        finally:
            panel.close()
            panel.deleteLater()

    def test_read_only_library_blocks_setlist_mutations(self, qapp, tmp_path):
        lib = _build_real_library(tmp_path, count=2)
        s1, s2 = [s.id for s in lib.songs]
        sl = lib.create_setlist("ReadOnly", [s1, s2])
        lib._writable = False
        panel = LibraryPanel(lib)
        try:
            panel.set_collection_filter("setlist", sl.id)
            assert (
                panel._list.dragDropMode()
                == QAbstractItemView.DragDropMode.NoDragDrop
            )
            with patch("src.ui.library_panel.QMessageBox.warning") as warn:
                panel._on_new_setlist([s1])
                panel._on_rename_setlist()
                panel._on_delete_setlist()
                panel._on_add_song_to_setlist(sl.id, s2)
                panel._on_remove_song_from_setlist(sl.id, s1)
                panel._reorder_active_setlist(0, 1)
            assert warn.call_count == 6
            assert lib.get_setlist(sl.id).song_ids == [s1, s2]
        finally:
            panel.close()
            panel.deleteLater()


# ---------------------------------------------------------------------------
# MainWindow integration with Setlists, Play Queue, and Search Tab Order
# ---------------------------------------------------------------------------


class TestMainWindowSetlistsAndFilterPlayback:
    def test_play_setlist_sync_edits_and_queued_song_outside_filter(self, window):
        workers = []

        def make_worker(paths):
            w = _FakeStemLoadWorker(paths)
            workers.append(w)
            return w

        s1, s2, s3, s4 = [s.id for s in window._library.songs]
        sl = window._library.create_setlist("Showtime", [s1, s2, s3])
        window._library_panel.refresh()
        window._library_panel.set_collection_filter("setlist", sl.id)

        with patch.object(
            main_window_module, "StemLoadWorker", side_effect=make_worker, create=True,
        ), patch.object(window._player, "play"):
            # Clicking Play Setlist starts first song of the setlist.
            window._library_panel._play_setlist_btn.click()
            arrays, sr = player_module.read_stem_files(workers[-1].stem_paths)
            workers[-1].completed.emit(arrays, sr)
            assert window._current_song_id == s1
            assert window.play_queue.is_setlist_active is True
            assert window.play_queue.context_songs == [s1, s2, s3]

            # Reordering/removing in LibraryPanel syncs active play_queue.
            window._library.reorder_setlist(sl.id, [s3, s1, s2])
            window._library_panel.refresh()
            window._library_panel.setlist_changed.emit(sl.id)
            assert window.play_queue.context_songs == [s3, s1, s2]
            assert window.play_queue.context_index == 1  # still at s1
            assert window.play_queue.upcoming_context_songs() == [s2]

            # Queue s4 (which is NOT in the filtered setlist) and finish s1 under
            # REPEAT_OFF: playback advances to s4 without dropping the setlist
            # filter!
            window.play_next(s4)
            window._library_panel.set_repeat_mode(REPEAT_OFF)
            window._on_play_finished()
            arrays, sr = player_module.read_stem_files(workers[-1].stem_paths)
            workers[-1].completed.emit(arrays, sr)
            assert window._current_song_id == s4
            assert window._library_panel.filter_kind == "setlist"
            assert window._library_panel.filter_value == sl.id

            # Deleting active setlist clears setlist context cleanly.
            window._library.delete_setlist(sl.id)
            window._library_panel.refresh()
            window._library_panel.setlist_changed.emit(sl.id)
            assert window.play_queue.is_setlist_active is False

    def test_finished_import_opens_while_another_view_is_shown(self, window):
        song_id = window._library.songs[0].id
        sl = window._library.create_setlist("Empty", [])
        window._library_panel.set_collection_filter("setlist", sl.id)
        window._watched_separation = song_id
        with patch.object(
            main_window_module, "StemLoadWorker",
            side_effect=_FakeStemLoadWorker, create=True,
        ):
            window._on_separation_finished(song_id, "htdemucs")
        assert window._loading_song_id == song_id
        assert window._library_panel.filter_value == sl.id

    def test_out_of_view_song_without_stems_does_not_start(self, window):
        song = window._library.songs[0]
        sl = window._library.create_setlist("Empty", [])
        window._library_panel.set_collection_filter("setlist", sl.id)
        os.remove(f"{song.stems_path}/vocals.wav")
        assert window._load_and_autoplay_song(song.id) is False
        assert window._pending_autoplay_song_id is None

"""Tests for marking songs, batch actions, batch export, and batch import."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMenu, QMessageBox

from src import player as player_module
from src.app_settings import open_settings
from src.batch_export import (
    BatchExportItem,
    BatchExportWorker,
    free_path,
    safe_file_name,
)
from src.library import SongLibrary
from src.ui import main_window as main_window_module
from src.ui.batch_export_dialog import BatchExportDialog
from src.ui.import_dialog import ImportDialog
from src.ui.library_panel import LibraryPanel

LEFT = Qt.MouseButton.LeftButton
RIGHT = Qt.MouseButton.RightButton
CTRL = Qt.KeyboardModifier.ControlModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier
NONE = Qt.KeyboardModifier.NoModifier


def _wav(path, value=0.1, frames=64):
    sf.write(str(path), np.full((frames, 2), value, dtype=np.float32), 44100)
    return str(path)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def lib(tmp_path):
    library = SongLibrary(str(tmp_path / "data"))
    for i in range(1, 5):
        song = library.add_song(
            f"Song {i}", f"Artist {i}", _wav(tmp_path / f"{i}.wav"),
        )
        _wav(f"{song.stems_path}/vocals.wav", 0.1)
        _wav(f"{song.stems_path}/drums.wav", 0.2)
    return library


@pytest.fixture
def panel(qapp, lib):
    widget = LibraryPanel(lib)
    widget.resize(260, 700)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()


@pytest.fixture
def ids(lib):
    return [s.id for s in lib.songs]


def _click(panel, song_id, modifiers=NONE, button=LEFT):
    center = panel._list.visualItemRect(
        panel._item_for_song(song_id)
    ).center()
    QTest.mouseClick(
        panel._list.viewport(), button, modifiers, center,
    )


def _menu(panel):
    menu = QMenu()
    panel._fill_batch_menu(menu)
    return menu


def _action(menu, label):
    for action in menu.actions():
        if action.text() == label:
            return action
        if action.menu() is not None:
            for sub in action.menu().actions():
                if sub.text() == label:
                    return sub
    raise AssertionError(label)


class TestMarking:
    def test_modifier_clicks_mark_without_loading(self, panel, ids):
        _click(panel, ids[0])
        selected = []
        panel.song_selected.connect(selected.append)
        _click(panel, ids[2], CTRL)
        # The selected song joins the first mark.
        assert panel.marked_song_ids() == [ids[0], ids[2]]
        _click(panel, ids[0], CTRL)
        assert panel.marked_song_ids() == [ids[2]]
        _click(panel, ids[3], SHIFT)
        _click(panel, ids[1], SHIFT)
        # The range grows from the last Ctrl+click, the first song.
        assert panel.marked_song_ids() == ids[:2]
        assert selected == []
        assert not panel._batch_bar.isHidden()
        assert panel._batch_label.text() == "2 selected"

        _click(panel, ids[3])
        assert selected == [ids[3]]
        assert panel.marked_song_ids() == []
        assert panel._batch_bar.isHidden()

    def test_right_click_does_not_load_or_unmark(self, panel, ids):
        _click(panel, ids[0])
        _click(panel, ids[1], CTRL)
        selected = []
        panel.song_selected.connect(selected.append)
        _click(panel, ids[2], button=RIGHT)
        assert selected == []
        assert panel._list.currentRow() == 0
        assert panel.marked_song_ids() == ids[:2]

    def test_keyboard(self, panel, ids):
        _click(panel, ids[1])
        QTest.keyClick(panel._list, Qt.Key.Key_A, CTRL)
        assert panel.marked_song_ids() == ids
        QTest.keyClick(panel._list, Qt.Key.Key_Space, CTRL)
        assert panel.marked_song_ids() == [ids[0], ids[2], ids[3]]
        QTest.keyClick(panel._list, Qt.Key.Key_Escape)
        assert panel.marked_song_ids() == []

    def test_modifier_double_click_does_not_edit(self, panel, ids):
        with patch.object(panel, "_on_edit_song") as edit:
            center = panel._list.visualItemRect(
                panel._item_for_song(ids[0])
            ).center()
            QTest.mouseDClick(panel._list.viewport(), LEFT, CTRL, center)
        edit.assert_not_called()

    def test_songs_out_of_sight_lose_their_mark(self, panel, ids):
        QTest.keyClick(panel._list, Qt.Key.Key_A, CTRL)
        panel._search_edit.setText("Song 2")
        assert panel.marked_song_ids() == [ids[1]]
        panel._search_edit.clear()
        assert panel.marked_song_ids() == [ids[1]]

        panel._on_mark_all()
        panel.set_song_separating(ids[0], "Separating...")
        assert panel.marked_song_ids() == ids[1:]
        _click(panel, ids[0], CTRL)
        assert panel.marked_song_ids() == ids[1:]

    def test_marking_in_a_setlist_does_not_start_a_drag(self, panel, lib, ids):
        sl = lib.create_setlist("Set", ids)
        panel.set_collection_filter("setlist", sl.id)
        selected = []
        panel.song_selected.connect(selected.append)
        _click(panel, ids[1], CTRL)
        _click(panel, ids[2], CTRL)
        assert panel.marked_song_ids() == ids[1:3]
        assert panel._pressed_songs is None
        assert selected == []

    def test_marks_are_spoken(self, panel, ids):
        role = Qt.ItemDataRole.AccessibleDescriptionRole
        panel.toggle_song_mark(ids[1])
        assert panel._item_for_song(ids[1]).data(role) == "Selected"
        panel.clear_marks()
        assert panel._item_for_song(ids[1]).data(role) == ""


class TestActions:
    def test_queue_tag_setlist_and_export(self, panel, lib, ids):
        sl = lib.create_setlist("R & B", [ids[0]])
        got = []
        panel.play_next_many_requested.connect(lambda x: got.append(("n", x)))
        panel.add_many_to_queue_requested.connect(
            lambda x: got.append(("q", x))
        )
        panel.export_songs_requested.connect(lambda x: got.append(("e", x)))
        panel._on_mark_all()
        menu = _menu(panel)
        _action(menu, "Play Next").trigger()
        _action(menu, "Add to Queue").trigger()
        _action(menu, "Export...").trigger()
        assert got == [("n", ids), ("q", ids), ("e", ids)]
        _action(menu, "R && B").trigger()
        assert lib.get_setlist(sl.id).song_ids == ids
        with patch(
            "src.ui.library_panel.QInputDialog.getItem",
            return_value=("gig, loud", True),
        ):
            menu = _menu(panel)
            _action(menu, "Add Tag...").trigger()
        assert all(s.tags == ["gig", "loud"] for s in lib.songs)

    def test_remove_asks_once_and_reports_failures(self, panel, lib, ids):
        removed = []
        panel.song_removed.connect(removed.append)
        panel.toggle_song_mark(ids[0])
        panel.toggle_song_mark(ids[1])
        panel.toggle_song_mark(ids[2])
        real = lib.remove_song

        def remove(song_id):
            if song_id == ids[1]:
                raise PermissionError(32, "in use")
            real(song_id)

        with patch.object(lib, "remove_song", remove), patch(
            "src.ui.library_panel.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ) as question, patch(
            "src.ui.library_panel.QMessageBox.warning"
        ) as warning:
            panel._remove_btn.click()
        question.assert_called_once()
        assert "these 3 songs" in question.call_args.args[2]
        warning.assert_called_once()
        assert removed == [ids[0], ids[2]]
        assert [s.id for s in lib.songs] == [ids[1], ids[3]]

    def test_read_only_library_disables_changes(self, panel, lib, ids):
        lib._writable = False
        panel._on_mark_all()
        menu = _menu(panel)
        assert not _action(menu, "Add Tag...").isEnabled()
        assert not _action(menu, "Remove...").isEnabled()
        assert _action(menu, "Export...").isEnabled()
        assert not panel._remove_btn.isEnabled()


class TestBatchExport:
    @pytest.mark.parametrize("name, expected", [
        ("AC/DC - T:N?T", "AC_DC - T_N_T"),
        ("CON", "_CON"),
        ("nul.txt", "_nul.txt"),
        ("trailing. ", "trailing"),
        ("...", "Untitled"),
        ("x" * 300, "x" * 120),
    ])
    def test_safe_file_name(self, name, expected):
        assert safe_file_name(name) == expected

    def test_free_path_never_overwrites(self, tmp_path):
        (tmp_path / "a.wav").touch()
        (tmp_path / "a (2).wav").touch()
        assert free_path(str(tmp_path), "a", ".wav").endswith("a (3).wav")

    def _run(self, *args, **kwargs):
        worker = BatchExportWorker(*args, **kwargs)
        result = []
        worker.completed.connect(lambda n, f: result.append((n, f)))
        worker.run()
        return result[0]

    def test_mix_holds_only_the_chosen_stems(self, qapp, tmp_path):
        item = BatchExportItem("Same", {
            "vocals": _wav(tmp_path / "v.wav", 0.1),
            "drums": _wav(tmp_path / "d.wav", 0.2),
        })
        out = tmp_path / "out"
        out.mkdir()
        bad = BatchExportItem("Bad", {"bass": str(tmp_path / "gone.wav")})
        exported, failures = self._run(
            [item, bad, item], str(out), ["drums", "bass"],
        )
        assert exported == 2
        assert len(failures) == 1 and failures[0].startswith("Bad: ")
        assert sorted(os.listdir(out)) == ["Same (2).wav", "Same.wav"]
        audio, _ = sf.read(str(out / "Same.wav"))
        assert np.allclose(audio, 0.2, atol=1e-3)

    def test_separate_files(self, qapp, tmp_path):
        item = BatchExportItem("Song", {
            "vocals": _wav(tmp_path / "v.wav"),
            "drums": _wav(tmp_path / "d.wav"),
        })
        out = tmp_path / "out"
        out.mkdir()
        exported, _ = self._run(
            [item], str(out), ["vocals", "drums"], separate_files=True,
        )
        assert exported == 1
        assert sorted(os.listdir(out / "Song")) == ["drums.wav", "vocals.wav"]

    def test_dialog_needs_a_stem(self, qapp):
        dialog = BatchExportDialog(2, ["vocals", "drums"], "mp3")
        assert dialog.stems == ["vocals", "drums"]
        assert dialog.extension == ".mp3"
        for check in dialog._stem_checks.values():
            check.setChecked(False)
        assert not dialog._ok.isEnabled()
        dialog.deleteLater()


class _QuietMainWindow(main_window_module.MainWindow):
    def _prune_incomplete_songs(self) -> None:
        pass


@pytest.fixture
def window(qapp, lib):
    settings = open_settings()
    settings.clear()
    player = player_module.MultiTrackPlayer()
    with patch.object(main_window_module.QTimer, "singleShot"):
        win = _QuietMainWindow(lib, player, MagicMock())
    yield win
    win._shutdown_stem_loads()
    win._separation_queue.shutdown(0)
    win._player_controls.shutdown()
    player.shutdown(wait_ms=5000)
    win.setParent(None)
    win.deleteLater()
    qapp.processEvents()
    settings.clear()


class TestMainWindow:
    def test_queue_many_keeps_order_and_skips_separating(self, window, ids):
        window._library_panel.set_song_separating(ids[1], "Queued...")
        window.add_many_to_queue(ids[2:])
        window.play_next_many(ids[:2])
        assert window.play_queue.up_next == [ids[0], ids[2], ids[3]]

    def test_export_runs_to_the_summary(self, window, ids, tmp_path, qapp):
        out = tmp_path / "exported"
        out.mkdir()
        mod = "src.ui.main_window."
        with patch(
            mod + "BatchExportDialog.exec",
            return_value=QDialog.DialogCode.Accepted,
        ), patch(
            mod + "QFileDialog.getExistingDirectory", return_value=str(out),
        ), patch(mod + "QMessageBox.information") as info:
            window._on_export_songs(ids[:2])
            assert window._export_worker.wait(20000)
            for _ in range(50):
                qapp.processEvents()
                if info.called:
                    break
            info.assert_called_once()
        assert info.call_args.args[2] == "Exported 2 songs."
        assert window._export_progress is None
        assert len(os.listdir(out)) == 2


class TestBatchImport:
    def _dialog(self, lib, paths, *, downloaded=True, queue=None):
        manager = MagicMock()
        manager.is_model_downloaded.return_value = downloaded
        manager.model_path.return_value = __file__  # any existing file
        manager.download_size_mb.return_value = 1
        return ImportDialog(
            lib, manager, file_paths=paths,
            separation_queue=queue or MagicMock(),
        )

    @pytest.fixture
    def files(self, tmp_path):
        return [_wav(tmp_path / f"{n}.wav") for n in ("One", "Two")]

    @pytest.fixture(autouse=True)
    def _memory(self):
        with patch(
            "src.ui.import_dialog.available_memory_bytes", return_value=None,
        ):
            yield

    def test_queues_every_file_with_one_model(self, qapp, lib, files):
        queue = MagicMock()
        dialog = self._dialog(lib, files, queue=queue)
        assert not dialog._title_edit.isEnabled()
        dialog._artist_edit.setText("Band")
        dialog._on_import()
        jobs = [c.args[0] for c in queue.enqueue.call_args_list]
        assert {j.model_key for j in jobs} == {"htdemucs"}
        songs = [lib.get_song(j.song_id) for j in jobs]
        assert [(s.title, s.artist) for s in songs] == [
            ("One", "Band"), ("Two", "Band"),
        ]
        assert dialog.result() == QDialog.DialogCode.Accepted
        dialog.deleteLater()

    def test_unreadable_file_adds_nothing(self, qapp, lib, files, tmp_path):
        bad = tmp_path / "bad.mp3"
        bad.write_bytes(b"nope")
        before = len(lib.songs)
        dialog = self._dialog(lib, [*files, str(bad)])
        dialog._on_import()
        assert len(lib.songs) == before
        assert "bad.mp3" in dialog._status_label.text()
        dialog.deleteLater()

    def test_model_downloads_before_the_library_changes(
        self, qapp, lib, files,
    ):
        queue = MagicMock()
        before = len(lib.songs)
        dialog = self._dialog(lib, files, downloaded=False, queue=queue)
        with patch(
            "src.ui.import_dialog.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ) as question:
            dialog._on_import()
        question.assert_called_once()
        assert len(lib.songs) == before
        dialog._on_model_download_finished("")
        assert len(lib.songs) == before + 2
        assert queue.enqueue.call_count == 2
        dialog.deleteLater()

    def test_one_failed_copy_does_not_stop_the_rest(self, qapp, lib, files):
        queue = MagicMock()
        dialog = self._dialog(lib, files, queue=queue)
        real = lib.add_song

        def add(title, **kwargs):
            if title == "One":
                raise OSError("disk full")
            return real(title=title, **kwargs)

        with patch.object(lib, "add_song", add), patch(
            "src.ui.import_dialog.QMessageBox.warning"
        ) as warning:
            dialog._on_import()
        warning.assert_called_once()
        assert queue.enqueue.call_count == 1
        assert dialog.result() == QDialog.DialogCode.Accepted
        dialog.deleteLater()

    def test_choosing_one_file_again_returns_to_single(
        self, qapp, lib, files,
    ):
        dialog = self._dialog(lib, files)
        dialog._select_files(files[:1])
        assert dialog._title_edit.isEnabled()
        assert dialog._title_edit.text() == "One"
        assert dialog._batch_paths == []
        dialog.deleteLater()

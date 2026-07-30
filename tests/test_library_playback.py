"""Tests for library playback controls: repeat, shuffle, next/prev, now-playing."""

from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QApplication

from src.library import Song
from src.ui.library_panel import (
    REPEAT_ALL,
    REPEAT_OFF,
    REPEAT_ONE,
    LibraryPanel,
)


@pytest.fixture(scope="module")
def app():
    """Ensure a QApplication exists for widget tests."""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


def _make_library(songs: list[Song]) -> MagicMock:
    library = MagicMock()
    library.songs = songs
    library.get_song.side_effect = lambda sid: next(
        (s for s in songs if s.id == sid), None
    )
    return library


def _make_songs(n: int = 5) -> list[Song]:
    """Return *n* test songs."""
    return [
        Song(
            id=str(i),
            title=f"Song {i}",
            artist=f"Artist {i}",
            original_path="",
            stems_path="",
            model_used="",
            date_added="",
        )
        for i in range(1, n + 1)
    ]


# ------------------------------------------------------------------
# Repeat button
# ------------------------------------------------------------------


class TestRepeatMode:
    def test_initial_mode_is_off(self, app):
        panel = LibraryPanel(_make_library([]))
        assert panel.repeat_mode == REPEAT_OFF

    def test_cycle_off_to_all(self, app):
        panel = LibraryPanel(_make_library([]))
        panel._on_repeat_clicked()
        assert panel.repeat_mode == REPEAT_ALL

    def test_cycle_all_to_one(self, app):
        panel = LibraryPanel(_make_library([]))
        panel._on_repeat_clicked()  # off → all
        panel._on_repeat_clicked()  # all → one
        assert panel.repeat_mode == REPEAT_ONE

    def test_cycle_one_back_to_off(self, app):
        panel = LibraryPanel(_make_library([]))
        for _ in range(3):
            panel._on_repeat_clicked()
        assert panel.repeat_mode == REPEAT_OFF

    def test_set_repeat_mode_programmatic(self, app):
        panel = LibraryPanel(_make_library([]))
        panel.set_repeat_mode(REPEAT_ALL)
        assert panel.repeat_mode == REPEAT_ALL

    def test_set_repeat_mode_invalid_falls_back(self, app):
        panel = LibraryPanel(_make_library([]))
        panel.set_repeat_mode("invalid")
        assert panel.repeat_mode == REPEAT_OFF

    def test_repeat_signal_emitted(self, app):
        panel = LibraryPanel(_make_library([]))
        received = []
        panel.repeat_mode_changed.connect(received.append)
        panel._on_repeat_clicked()
        assert received == [REPEAT_ALL]


# ------------------------------------------------------------------
# Shuffle button
# ------------------------------------------------------------------


class TestShuffle:
    def test_initial_shuffle_off(self, app):
        panel = LibraryPanel(_make_library([]))
        assert not panel.shuffle_enabled

    def test_toggle_shuffle_on(self, app):
        panel = LibraryPanel(_make_library([]))
        panel._shuffle_btn.setChecked(True)
        assert panel.shuffle_enabled

    def test_set_shuffle_programmatic(self, app):
        panel = LibraryPanel(_make_library([]))
        panel.set_shuffle_enabled(True)
        assert panel.shuffle_enabled
        assert panel._shuffle_btn.isChecked()

    def test_shuffle_signal_emitted(self, app):
        panel = LibraryPanel(_make_library([]))
        received = []
        panel.shuffle_toggled.connect(received.append)
        panel._shuffle_btn.setChecked(True)
        assert received == [True]


# ------------------------------------------------------------------
# Now-playing indicator
# ------------------------------------------------------------------


class TestNowPlaying:
    def test_set_playing_song(self, app):
        panel = LibraryPanel(_make_library(_make_songs()))
        panel.set_playing_song("2")
        assert panel._song_delegate._playing_song_id == "2"

    def test_clear_playing_song(self, app):
        panel = LibraryPanel(_make_library(_make_songs()))
        panel.set_playing_song("2")
        panel.set_playing_song(None)
        assert panel._song_delegate._playing_song_id is None


# ------------------------------------------------------------------
# Next / Previous signals
# ------------------------------------------------------------------


class TestNavSignals:
    def test_next_signal_emitted(self, app):
        panel = LibraryPanel(_make_library(_make_songs()))
        received = []
        panel.next_requested.connect(lambda: received.append("next"))
        panel._next_btn.click()
        assert received == ["next"]

    def test_previous_signal_emitted(self, app):
        panel = LibraryPanel(_make_library(_make_songs()))
        received = []
        panel.previous_requested.connect(lambda: received.append("prev"))
        panel._prev_btn.click()
        assert received == ["prev"]


# ------------------------------------------------------------------
# Song ordering helpers
# ------------------------------------------------------------------


class TestSongOrder:
    def test_song_ids_in_order(self, app):
        songs = _make_songs(3)
        panel = LibraryPanel(_make_library(songs))
        assert panel.song_ids_in_order() == ["1", "2", "3"]

    def test_song_count(self, app):
        songs = _make_songs(4)
        panel = LibraryPanel(_make_library(songs))
        assert panel.song_count() == 4

    def test_song_count_empty(self, app):
        panel = LibraryPanel(_make_library([]))
        assert panel.song_count() == 0


# ------------------------------------------------------------------
# MainWindow autoplay logic (unit-tested via extracted methods)
# ------------------------------------------------------------------


class TestNextSongResolution:
    """Test _get_next_song_id logic by calling it on a mock MainWindow."""

    def _make_main_window_stub(self, songs, current_id, repeat, shuffle):
        """Create a minimal stub with the methods we need to test."""
        panel = LibraryPanel(_make_library(songs))
        panel.set_repeat_mode(repeat)
        panel.set_shuffle_enabled(shuffle)

        stub = MagicMock()
        stub._library_panel = panel
        stub._current_song_id = current_id
        stub._shuffle_queue = []
        return stub

    def test_next_sequential(self, app):
        from src.ui.main_window import MainWindow

        songs = _make_songs(3)
        stub = self._make_main_window_stub(songs, "1", REPEAT_ALL, False)
        result = MainWindow._get_next_song_id(stub, direction=1)
        assert result == "2"

    def test_prev_sequential(self, app):
        from src.ui.main_window import MainWindow

        songs = _make_songs(3)
        stub = self._make_main_window_stub(songs, "2", REPEAT_ALL, False)
        result = MainWindow._get_next_song_id(stub, direction=-1)
        assert result == "1"

    def test_wrap_around_forward(self, app):
        from src.ui.main_window import MainWindow

        songs = _make_songs(3)
        stub = self._make_main_window_stub(songs, "3", REPEAT_ALL, False)
        result = MainWindow._get_next_song_id(stub, direction=1)
        assert result == "1"

    def test_wrap_around_backward(self, app):
        from src.ui.main_window import MainWindow

        songs = _make_songs(3)
        stub = self._make_main_window_stub(songs, "1", REPEAT_ALL, False)
        result = MainWindow._get_next_song_id(stub, direction=-1)
        assert result == "3"

    def test_no_wrap_when_repeat_off(self, app):
        from src.ui.main_window import MainWindow

        songs = _make_songs(3)
        stub = self._make_main_window_stub(songs, "3", REPEAT_OFF, False)
        result = MainWindow._get_next_song_id(stub, direction=1)
        assert result is None

    def test_empty_library(self, app):
        from src.ui.main_window import MainWindow

        stub = self._make_main_window_stub([], None, REPEAT_ALL, False)
        result = MainWindow._get_next_song_id(stub, direction=1)
        assert result is None

    def test_shuffle_returns_different_order(self, app):
        """Shuffle should return all songs before repeating any."""
        from src.ui.main_window import MainWindow

        songs = _make_songs(5)
        stub = self._make_main_window_stub(songs, "1", REPEAT_ALL, True)

        seen = set()
        for _ in range(4):  # 5 songs - 1 current = 4 others
            result = MainWindow._pop_shuffle_queue(stub, [s.id for s in songs])
            assert result is not None
            assert result != "1"  # Never returns current
            seen.add(result)

        assert seen == {"2", "3", "4", "5"}

    def test_shuffle_excludes_current(self, app):
        """Shuffle queue should never contain the current song."""
        from src.ui.main_window import MainWindow

        songs = _make_songs(3)
        stub = self._make_main_window_stub(songs, "2", REPEAT_ALL, True)

        for _ in range(10):
            result = MainWindow._pop_shuffle_queue(stub, [s.id for s in songs])
            assert result != "2"

    def test_previous_falls_through_to_sequential_during_shuffle(self, app):
        """Pressing Previous during shuffle uses sequential order (YouTube-style)."""
        from src.ui.main_window import MainWindow

        songs = _make_songs(3)
        stub = self._make_main_window_stub(songs, "2", REPEAT_ALL, True)
        # direction=-1 should NOT hit the shuffle queue.
        result = MainWindow._get_next_song_id(stub, direction=-1)
        assert result == "1"

    def test_shuffle_toggle_off_clears_queue(self, app):
        """Disabling shuffle on MainWindow clears the pending queue."""
        from src.ui.main_window import MainWindow

        songs = _make_songs(3)
        stub = self._make_main_window_stub(songs, "1", REPEAT_ALL, True)
        # Prime the queue.
        MainWindow._pop_shuffle_queue(stub, [s.id for s in songs])
        assert stub._shuffle_queue  # non-empty after first pop
        # Toggle off — should clear.
        MainWindow._on_shuffle_toggled(stub, False)
        assert stub._shuffle_queue == []


# ------------------------------------------------------------------
# Master volume via shortcut
# ------------------------------------------------------------------


class TestMasterVolumeAdjust:
    """_adjust_master_volume routes through PlayerControls.set_master_volume
    so the transport-row slider/label stay in sync with the player."""

    def test_adjust_adds_delta(self, app):
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        stub._player.master_volume = 1.0
        MainWindow._adjust_master_volume(stub, 0.1)
        stub._player_controls.set_master_volume.assert_called_once_with(
            pytest.approx(1.1)
        )

    def test_adjust_clamps_upper(self, app):
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        stub._player.master_volume = 1.95
        MainWindow._adjust_master_volume(stub, 0.1)
        # Caller clamps to 2.0 before calling set_master_volume.
        stub._player_controls.set_master_volume.assert_called_once_with(2.0)

    def test_adjust_clamps_lower(self, app):
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        stub._player.master_volume = 0.02
        MainWindow._adjust_master_volume(stub, -0.1)
        stub._player_controls.set_master_volume.assert_called_once_with(0.0)

    def test_adjust_shows_toast(self, app):
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        stub._player.master_volume = 0.5
        MainWindow._adjust_master_volume(stub, 0.1)
        stub._show_volume_toast.assert_called_once()


# ------------------------------------------------------------------
# Autoplay on play_finished
# ------------------------------------------------------------------


class TestOnPlayFinished:
    """Autoplay behaviour depends on the current repeat mode."""

    def _make_stub(self, mode):
        panel = LibraryPanel(_make_library(_make_songs(3)))
        panel.set_repeat_mode(mode)
        stub = MagicMock()
        stub._library_panel = panel
        stub._player = MagicMock()
        return stub

    def test_repeat_one_seeks_and_plays(self, app):
        from src.ui.main_window import MainWindow

        stub = self._make_stub(REPEAT_ONE)
        MainWindow._on_play_finished(stub)
        stub._player.seek.assert_called_once_with(0)
        stub._player.play.assert_called_once()
        stub._advance_song.assert_not_called()

    def test_repeat_off_is_noop(self, app):
        from src.ui.main_window import MainWindow

        stub = self._make_stub(REPEAT_OFF)
        MainWindow._on_play_finished(stub)
        stub._player.seek.assert_not_called()
        stub._player.play.assert_not_called()
        stub._advance_song.assert_not_called()

    def test_repeat_all_advances_forward(self, app):
        from src.ui.main_window import MainWindow

        stub = self._make_stub(REPEAT_ALL)
        MainWindow._on_play_finished(stub)
        stub._advance_song.assert_called_once_with(1)


# ------------------------------------------------------------------
# Shortcut focus guard
# ------------------------------------------------------------------


class TestShortcutFocusGuard:
    """_text_input_has_focus returns True only for text-entry widgets."""

    def test_no_focus_returns_false(self, app):
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        with patch(
            "src.ui.main_window.QApplication.focusWidget", return_value=None
        ):
            assert MainWindow._text_input_has_focus(stub) is False

    def test_line_edit_focus_returns_true(self, app):
        from PySide6.QtWidgets import QLineEdit

        from src.ui.main_window import MainWindow

        stub = MagicMock()
        w = QLineEdit()
        with patch(
            "src.ui.main_window.QApplication.focusWidget", return_value=w
        ):
            assert MainWindow._text_input_has_focus(stub) is True

    def test_spinbox_focus_returns_true(self, app):
        from PySide6.QtWidgets import QSpinBox

        from src.ui.main_window import MainWindow

        stub = MagicMock()
        w = QSpinBox()
        with patch(
            "src.ui.main_window.QApplication.focusWidget", return_value=w
        ):
            assert MainWindow._text_input_has_focus(stub) is True

    def test_non_editable_combo_returns_false(self, app):
        from PySide6.QtWidgets import QComboBox

        from src.ui.main_window import MainWindow

        stub = MagicMock()
        w = QComboBox()
        w.setEditable(False)
        with patch(
            "src.ui.main_window.QApplication.focusWidget", return_value=w
        ):
            assert MainWindow._text_input_has_focus(stub) is False

    def test_editable_combo_returns_true(self, app):
        from PySide6.QtWidgets import QComboBox

        from src.ui.main_window import MainWindow

        stub = MagicMock()
        w = QComboBox()
        w.setEditable(True)
        with patch(
            "src.ui.main_window.QApplication.focusWidget", return_value=w
        ):
            assert MainWindow._text_input_has_focus(stub) is True


class TestExportDoubleStartGuard:
    """A second export while one is running must be refused, not crash.

    ExportWorker is parentless; reassigning self._export_worker would
    drop the last reference to the live QThread and crash Qt.
    """

    def test_running_export_blocks_second(self, app):
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        stub._player.has_stems = True
        stub._export_worker.isRunning.return_value = True

        with patch(
            "src.ui.main_window.QMessageBox.information"
        ) as info, patch(
            "src.ui.main_window.ExportWorker"
        ) as worker_cls:
            MainWindow._on_export(stub)

        info.assert_called_once()
        worker_cls.assert_not_called()

    def test_no_running_export_proceeds_past_guard(self, app):
        """With no worker running the guard is not the thing that stops
        us -- has_stems True + a real song lookup would continue; here
        we only assert the guard message is not shown."""
        from src.ui.main_window import MainWindow

        stub = MagicMock()
        stub._player.has_stems = True
        stub._export_worker = None
        stub._current_song_id = None  # get_song(None) -> stop later, no crash

        with patch(
            "src.ui.main_window.QMessageBox.information"
        ) as info:
            MainWindow._on_export(stub)

        # The "already in progress" message must not have fired.
        for call in info.call_args_list:
            assert "already in progress" not in str(call)

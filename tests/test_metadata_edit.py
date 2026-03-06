"""Tests for song metadata editing from the library panel."""

from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QApplication

from src.library import Song
from src.ui.library_panel import LibraryPanel, EditSongDialog


@pytest.fixture(scope="module")
def app():
    """Ensure a QApplication exists for widget tests."""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


def _make_library(songs: list[Song]) -> MagicMock:
    """Create a mock SongLibrary with the given songs."""
    library = MagicMock()
    library.songs = songs
    library.get_song = MagicMock(side_effect=lambda sid: next(
        (s for s in songs if s.id == sid), None
    ))
    return library


def _make_songs() -> list[Song]:
    return [
        Song(id="1", title="Bohemian Rhapsody", artist="Queen",
             original_path="", stems_path="", model_used="", date_added=""),
        Song(id="2", title="Stairway to Heaven", artist="Led Zeppelin",
             original_path="", stems_path="", model_used="", date_added=""),
    ]


# -----------------------------------------------------------------------
# EditSongDialog
# -----------------------------------------------------------------------

class TestEditSongDialog:
    """Edit dialog pre-fills fields and returns updated values."""

    def test_fields_prefilled(self, app):
        """Dialog title and artist fields match the given song."""
        song = _make_songs()[0]
        dlg = EditSongDialog(song)
        try:
            assert dlg._title_edit.text() == "Bohemian Rhapsody"
            assert dlg._artist_edit.text() == "Queen"
        finally:
            dlg.close()

    def test_edited_values_returned(self, app):
        """After editing, title and artist properties reflect new values."""
        song = _make_songs()[0]
        dlg = EditSongDialog(song)
        try:
            dlg._title_edit.setText("New Title")
            dlg._artist_edit.setText("New Artist")
            assert dlg.title == "New Title"
            assert dlg.artist == "New Artist"
        finally:
            dlg.close()

    def test_empty_title_falls_back(self, app):
        """Empty title returns original song title as fallback."""
        song = _make_songs()[0]
        dlg = EditSongDialog(song)
        try:
            dlg._title_edit.clear()
            assert dlg.title == "Bohemian Rhapsody"
        finally:
            dlg.close()

    def test_empty_artist_falls_back(self, app):
        """Empty artist returns original song artist as fallback."""
        song = _make_songs()[0]
        dlg = EditSongDialog(song)
        try:
            dlg._artist_edit.clear()
            assert dlg.artist == "Queen"
        finally:
            dlg.close()


# -----------------------------------------------------------------------
# LibraryPanel edit integration
# -----------------------------------------------------------------------

class TestLibraryPanelEdit:
    """Library panel wires double-click to the edit dialog."""

    def test_double_click_opens_edit_dialog(self, app):
        """Double-clicking a song opens EditSongDialog."""
        songs = _make_songs()
        library = _make_library(songs)
        panel = LibraryPanel(library)
        panel._list.setCurrentRow(0)

        with patch("src.ui.library_panel.EditSongDialog") as mock_cls:
            mock_dlg = MagicMock()
            mock_dlg.exec.return_value = True
            mock_dlg.title = "Changed"
            mock_dlg.artist = "Artist"
            mock_cls.return_value = mock_dlg
            panel._on_edit_song()

        mock_cls.assert_called_once()
        library.update_song.assert_called_once_with(
            "1", title="Changed", artist="Artist"
        )

    def test_edit_cancelled_does_not_update(self, app):
        """Cancelling the edit dialog does not call update_song."""
        songs = _make_songs()
        library = _make_library(songs)
        panel = LibraryPanel(library)
        panel._list.setCurrentRow(0)

        with patch("src.ui.library_panel.EditSongDialog") as mock_cls:
            mock_dlg = MagicMock()
            mock_dlg.exec.return_value = False
            mock_cls.return_value = mock_dlg
            panel._on_edit_song()

        library.update_song.assert_not_called()

    def test_edit_refreshes_list(self, app):
        """Accepting the edit dialog refreshes the song list."""
        songs = _make_songs()
        library = _make_library(songs)
        panel = LibraryPanel(library)
        panel._list.setCurrentRow(0)

        with patch("src.ui.library_panel.EditSongDialog") as mock_cls:
            mock_dlg = MagicMock()
            mock_dlg.exec.return_value = True
            mock_dlg.title = "X"
            mock_dlg.artist = "Y"
            mock_cls.return_value = mock_dlg
            with patch.object(panel, "refresh") as mock_refresh:
                panel._on_edit_song()

        mock_refresh.assert_called_once()

    def test_no_selection_edit_is_noop(self, app):
        """Editing with no selection does nothing."""
        songs = _make_songs()
        library = _make_library(songs)
        panel = LibraryPanel(library)
        # No item selected.
        panel._list.setCurrentRow(-1)

        with patch("src.ui.library_panel.EditSongDialog") as mock_cls:
            panel._on_edit_song()

        mock_cls.assert_not_called()

    def test_context_menu_only_shows_on_item(self, app):
        """Right-click on empty space does not show a menu."""
        songs = _make_songs()
        library = _make_library(songs)
        panel = LibraryPanel(library)

        with patch("src.ui.library_panel.QMenu") as mock_menu:
            # Simulate right-click on empty area (itemAt returns None).
            with patch.object(panel._list, "itemAt", return_value=None):
                panel._on_context_menu(MagicMock())
            mock_menu.assert_not_called()

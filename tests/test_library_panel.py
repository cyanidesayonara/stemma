"""Tests for library panel search/filter functionality."""

from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QApplication

from src.library import Song
from src.ui.library_panel import LibraryPanel


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
    return library


def _make_songs() -> list[Song]:
    """Return a small set of test songs."""
    return [
        Song(id="1", title="Bohemian Rhapsody", artist="Queen",
             original_path="", stems_path="", model_used="", date_added=""),
        Song(id="2", title="Stairway to Heaven", artist="Led Zeppelin",
             original_path="", stems_path="", model_used="", date_added=""),
        Song(id="3", title="Hotel California", artist="Eagles",
             original_path="", stems_path="", model_used="", date_added=""),
    ]


def _visible_items(panel: LibraryPanel) -> list:
    """Return the non-hidden items in the panel's list widget."""
    return [panel._list.item(i) for i in range(panel._list.count())
            if not panel._list.item(i).isHidden()]


class TestSearchFilter:
    """Search box filters the library list as you type."""

    def test_search_box_exists(self, app):
        """Panel has a search input field."""
        panel = LibraryPanel(_make_library([]))
        assert hasattr(panel, "_search_edit")

    def test_all_songs_visible_with_empty_query(self, app):
        """All songs are visible when the search box is empty."""
        panel = LibraryPanel(_make_library(_make_songs()))
        assert len(_visible_items(panel)) == 3

    def test_filter_by_title(self, app):
        """Typing a title substring hides non-matching songs."""
        panel = LibraryPanel(_make_library(_make_songs()))
        panel._search_edit.setText("bohemian")
        visible = _visible_items(panel)
        assert len(visible) == 1
        assert "Bohemian" in visible[0].text()

    def test_filter_by_artist(self, app):
        """Typing an artist name filters correctly."""
        panel = LibraryPanel(_make_library(_make_songs()))
        panel._search_edit.setText("zeppelin")
        visible = _visible_items(panel)
        assert len(visible) == 1
        assert "Zeppelin" in visible[0].text()

    def test_filter_is_case_insensitive(self, app):
        """Filter matches regardless of case."""
        panel = LibraryPanel(_make_library(_make_songs()))
        panel._search_edit.setText("QUEEN")
        assert len(_visible_items(panel)) == 1

    def test_no_match_hides_all(self, app):
        """A query matching nothing hides all items."""
        panel = LibraryPanel(_make_library(_make_songs()))
        panel._search_edit.setText("xyznonexistent")
        assert len(_visible_items(panel)) == 0

    def test_clearing_search_shows_all(self, app):
        """Clearing the search box restores all items."""
        panel = LibraryPanel(_make_library(_make_songs()))
        panel._search_edit.setText("queen")
        panel._search_edit.clear()
        assert len(_visible_items(panel)) == 3

    def test_refresh_preserves_filter(self, app):
        """After refresh(), the current filter is reapplied."""
        songs = _make_songs()
        library = _make_library(songs)
        panel = LibraryPanel(library)
        panel._search_edit.setText("queen")

        # Simulate adding a new song and refreshing.
        songs.append(Song(id="4", title="We Will Rock You", artist="Queen",
                          original_path="", stems_path="", model_used="",
                          date_added=""))
        library.songs = songs
        panel.refresh()

        # Both Queen songs should be visible.
        assert len(_visible_items(panel)) == 2

    def test_remove_button_disabled_when_selection_hidden(self, app):
        """Remove button is disabled when the selected song is filtered out."""
        panel = LibraryPanel(_make_library(_make_songs()))
        # Select the second item (Led Zeppelin).
        panel._list.setCurrentRow(1)
        assert panel._remove_btn.isEnabled()
        # Filter to only Queen — Led Zeppelin is hidden.
        panel._search_edit.setText("queen")
        assert not panel._remove_btn.isEnabled()


def test_search_with_no_match_says_so(app):
    """A search that hides every song used to leave a blank list."""
    panel = LibraryPanel(_make_library(_make_songs()))
    assert panel._no_match_label.isHidden()

    panel._search_edit.setText("no such song")
    assert not panel._no_match_label.isHidden()

    panel._search_edit.setText("bohemian")
    assert panel._no_match_label.isHidden()
    panel._search_edit.setText("")
    assert panel._no_match_label.isHidden()


def test_search_box_has_an_accessible_name(app):
    """A placeholder is not a name for screen readers."""
    panel = LibraryPanel(_make_library([]))
    assert panel._search_edit.accessibleName() == "Search songs"


class TestLibraryWriteFailures:
    """A read-only or failing library is reported, not silently ignored."""

    def _panel(self, monkeypatch, writable=True):
        from src.ui import library_panel

        box = MagicMock()
        box.question.return_value = box.StandardButton.Yes
        monkeypatch.setattr(library_panel, "QMessageBox", box)
        library = _make_library(_make_songs())
        library.writable = writable
        panel = LibraryPanel(library)
        panel._list.setCurrentRow(0)
        return panel, library, box

    def _accepting_dialog(self, monkeypatch):
        from src.ui import library_panel

        dialog = MagicMock()
        dialog.return_value.exec.return_value = True
        dialog.return_value.title = "New"
        dialog.return_value.artist = "Band"
        monkeypatch.setattr(library_panel, "EditSongDialog", dialog)
        return dialog

    def test_failed_edit_shows_a_warning(self, app, monkeypatch):
        from src.library import LibraryReadOnlyError

        panel, library, box = self._panel(monkeypatch)
        self._accepting_dialog(monkeypatch)
        library.update_song.side_effect = LibraryReadOnlyError("read only")

        panel._on_edit_song()

        box.warning.assert_called_once()

    def test_failed_remove_shows_a_warning(self, app, monkeypatch):
        panel, library, box = self._panel(monkeypatch)
        library.remove_song.side_effect = PermissionError(32, "in use")
        removed = MagicMock()
        panel.song_removed.connect(removed)

        panel._on_remove_clicked()

        box.warning.assert_called_once()
        removed.assert_not_called()

    def test_read_only_remove_warns_before_closing_the_song(
        self, app, monkeypatch,
    ):
        panel, library, box = self._panel(monkeypatch, writable=False)
        requested = MagicMock()
        panel.song_removal_requested.connect(requested)

        panel._on_remove_clicked()

        requested.assert_not_called()
        library.remove_song.assert_not_called()
        box.warning.assert_called_once()

    def test_read_only_edit_opens_no_dialog(self, app, monkeypatch):
        panel, library, box = self._panel(monkeypatch, writable=False)
        dialog = self._accepting_dialog(monkeypatch)

        panel._on_edit_song()

        dialog.assert_not_called()
        box.warning.assert_called_once()

    def test_read_only_library_disables_remove(self, app, monkeypatch):
        panel, _library, _box = self._panel(monkeypatch, writable=False)

        assert not panel._remove_btn.isEnabled()

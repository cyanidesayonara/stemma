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


class TestSearchFilter:
    """Search box filters the library list as you type."""

    def test_search_box_exists(self, app):
        """Panel has a search input field."""
        panel = LibraryPanel(_make_library([]))
        assert hasattr(panel, "_search_edit")

    def test_all_songs_visible_with_empty_query(self, app):
        """All songs are visible when the search box is empty."""
        songs = _make_songs()
        panel = LibraryPanel(_make_library(songs))
        visible = [panel._list.item(i) for i in range(panel._list.count())
                   if not panel._list.item(i).isHidden()]
        assert len(visible) == 3

    def test_filter_by_title(self, app):
        """Typing a title substring hides non-matching songs."""
        songs = _make_songs()
        panel = LibraryPanel(_make_library(songs))
        panel._search_edit.setText("bohemian")
        visible = [panel._list.item(i) for i in range(panel._list.count())
                   if not panel._list.item(i).isHidden()]
        assert len(visible) == 1
        assert "Bohemian" in visible[0].text()

    def test_filter_by_artist(self, app):
        """Typing an artist name filters correctly."""
        songs = _make_songs()
        panel = LibraryPanel(_make_library(songs))
        panel._search_edit.setText("zeppelin")
        visible = [panel._list.item(i) for i in range(panel._list.count())
                   if not panel._list.item(i).isHidden()]
        assert len(visible) == 1
        assert "Zeppelin" in visible[0].text()

    def test_filter_is_case_insensitive(self, app):
        """Filter matches regardless of case."""
        songs = _make_songs()
        panel = LibraryPanel(_make_library(songs))
        panel._search_edit.setText("QUEEN")
        visible = [panel._list.item(i) for i in range(panel._list.count())
                   if not panel._list.item(i).isHidden()]
        assert len(visible) == 1

    def test_no_match_hides_all(self, app):
        """A query matching nothing hides all items."""
        songs = _make_songs()
        panel = LibraryPanel(_make_library(songs))
        panel._search_edit.setText("xyznonexistent")
        visible = [panel._list.item(i) for i in range(panel._list.count())
                   if not panel._list.item(i).isHidden()]
        assert len(visible) == 0

    def test_clearing_search_shows_all(self, app):
        """Clearing the search box restores all items."""
        songs = _make_songs()
        panel = LibraryPanel(_make_library(songs))
        panel._search_edit.setText("queen")
        panel._search_edit.clear()
        visible = [panel._list.item(i) for i in range(panel._list.count())
                   if not panel._list.item(i).isHidden()]
        assert len(visible) == 3

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

        visible = [panel._list.item(i) for i in range(panel._list.count())
                   if not panel._list.item(i).isHidden()]
        # Both Queen songs should be visible.
        assert len(visible) == 2

    def test_clear_button_resets_filter(self, app):
        """The clear button on the search box resets to show all."""
        songs = _make_songs()
        panel = LibraryPanel(_make_library(songs))
        panel._search_edit.setText("queen")
        # QLineEdit with setClearButtonEnabled — simulate clearing.
        panel._search_edit.clear()
        visible = [panel._list.item(i) for i in range(panel._list.count())
                   if not panel._list.item(i).isHidden()]
        assert len(visible) == 3

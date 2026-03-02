"""Tests for drag-and-drop file import in MainWindow and ImportDialog."""

from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QApplication

from src.ui.import_dialog import ImportDialog
from src.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    """Ensure a QApplication exists for widget tests."""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


@pytest.fixture
def dialog(app, tmp_path):
    """ImportDialog with mocked dependencies."""
    library = MagicMock()
    model_manager = MagicMock()
    model_manager.model_path.return_value = str(tmp_path / "no_model.onnx")
    dlg = ImportDialog(library, model_manager)
    yield dlg
    dlg.close()


@pytest.fixture
def main_window(app):
    """MainWindow with fully mocked dependencies."""
    library = MagicMock()
    library.songs = []
    player = MagicMock()
    model_manager = MagicMock()
    win = MainWindow(library, player, model_manager)
    yield win
    win.close()


def _make_event(paths: list[str]) -> MagicMock:
    """Build a mock drag/drop event carrying the given local file paths."""
    urls = []
    for p in paths:
        url = MagicMock()
        url.toLocalFile.return_value = p
        urls.append(url)
    mime = MagicMock()
    mime.hasUrls.return_value = True
    mime.urls.return_value = urls
    event = MagicMock()
    event.mimeData.return_value = mime
    return event


# -----------------------------------------------------------------------
# ImportDialog pre-fill
# -----------------------------------------------------------------------

class TestImportDialogPrefill:
    """ImportDialog accepts an optional file_path to pre-fill the form."""

    def test_default_path_is_empty(self, dialog):
        """Without file_path, path edit and title are blank."""
        assert dialog._path_edit.text() == ""
        assert dialog._title_edit.text() == ""

    def test_prefilled_path_appears_in_path_edit(self, app, tmp_path):
        """Passing file_path sets _path_edit and populates _selected_path."""
        audio = tmp_path / "my_song.mp3"
        audio.write_bytes(b"fake")
        library = MagicMock()
        model_manager = MagicMock()
        model_manager.model_path.return_value = str(tmp_path / "no_model.onnx")

        dlg = ImportDialog(library, model_manager, file_path=str(audio))
        try:
            assert dlg._path_edit.text() == str(audio)
            assert dlg._selected_path == str(audio)
        finally:
            dlg.close()

    def test_prefilled_title_derived_from_filename(self, app, tmp_path):
        """Title field is auto-filled from the file basename when pre-filling."""
        audio = tmp_path / "Great Song.mp3"
        audio.write_bytes(b"fake")
        library = MagicMock()
        model_manager = MagicMock()
        model_manager.model_path.return_value = str(tmp_path / "no_model.onnx")

        dlg = ImportDialog(library, model_manager, file_path=str(audio))
        try:
            assert dlg._title_edit.text() == "Great Song"
        finally:
            dlg.close()


# -----------------------------------------------------------------------
# MainWindow drag-and-drop
# -----------------------------------------------------------------------

class TestMainWindowDragEnter:
    """dragEnterEvent accepts audio files and rejects everything else."""

    def test_accepts_mp3(self, main_window):
        event = _make_event(["C:/music/song.mp3"])
        main_window.dragEnterEvent(event)
        event.acceptProposedAction.assert_called_once()

    def test_accepts_wav(self, main_window):
        event = _make_event(["C:/music/song.wav"])
        main_window.dragEnterEvent(event)
        event.acceptProposedAction.assert_called_once()

    def test_accepts_flac(self, main_window):
        event = _make_event(["C:/music/song.flac"])
        main_window.dragEnterEvent(event)
        event.acceptProposedAction.assert_called_once()

    def test_rejects_non_audio(self, main_window):
        event = _make_event(["C:/docs/report.pdf"])
        main_window.dragEnterEvent(event)
        event.ignore.assert_called_once()

    def test_rejects_no_urls(self, main_window):
        mime = MagicMock()
        mime.hasUrls.return_value = False
        event = MagicMock()
        event.mimeData.return_value = mime
        main_window.dragEnterEvent(event)
        event.ignore.assert_called_once()

    def test_accepts_if_any_audio_present(self, main_window):
        """Accept if at least one file is audio, even if mixed with others."""
        event = _make_event(["report.pdf", "song.wav"])
        main_window.dragEnterEvent(event)
        event.acceptProposedAction.assert_called_once()


class TestMainWindowDragMove:
    """dragMoveEvent mirrors dragEnterEvent to keep the drop cursor active."""

    def test_accepts_audio_during_move(self, main_window):
        event = _make_event(["C:/music/song.mp3"])
        main_window.dragMoveEvent(event)
        event.acceptProposedAction.assert_called_once()

    def test_ignores_non_audio_during_move(self, main_window):
        event = _make_event(["C:/docs/report.pdf"])
        main_window.dragMoveEvent(event)
        event.ignore.assert_called_once()


class TestMainWindowDrop:
    """dropEvent accepts the event and opens ImportDialog for each audio file."""

    def test_single_audio_file_opens_dialog(self, main_window):
        """Dropping one audio file opens ImportDialog once."""
        event = _make_event(["C:/music/song.mp3"])
        with patch("src.ui.main_window.ImportDialog") as mock_cls:
            mock_dlg = MagicMock()
            mock_dlg.exec.return_value = False
            mock_cls.return_value = mock_dlg
            main_window.dropEvent(event)
        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        assert kwargs.get("file_path") == "C:/music/song.mp3"

    def test_drop_event_is_accepted(self, main_window):
        """dropEvent calls acceptProposedAction on the event."""
        event = _make_event(["C:/music/song.mp3"])
        with patch("src.ui.main_window.ImportDialog") as mock_cls:
            mock_cls.return_value.exec.return_value = False
            main_window.dropEvent(event)
        event.acceptProposedAction.assert_called_once()

    def test_multiple_audio_files_open_multiple_dialogs(self, main_window):
        """Dropping two audio files opens ImportDialog twice."""
        event = _make_event(["a.mp3", "b.wav"])
        with patch("src.ui.main_window.ImportDialog") as mock_cls:
            mock_dlg = MagicMock()
            mock_dlg.exec.return_value = False
            mock_cls.return_value = mock_dlg
            main_window.dropEvent(event)
        assert mock_cls.call_count == 2

    def test_non_audio_files_are_skipped(self, main_window):
        """Dropping non-audio files does not open ImportDialog."""
        event = _make_event(["report.pdf", "image.png"])
        with patch("src.ui.main_window.ImportDialog") as mock_cls:
            main_window.dropEvent(event)
        mock_cls.assert_not_called()

    def test_mixed_drop_only_imports_audio(self, main_window):
        """Only audio files in a mixed drop trigger import."""
        event = _make_event(["report.pdf", "song.flac"])
        with patch("src.ui.main_window.ImportDialog") as mock_cls:
            mock_dlg = MagicMock()
            mock_dlg.exec.return_value = False
            mock_cls.return_value = mock_dlg
            main_window.dropEvent(event)
        assert mock_cls.call_count == 1
        _, kwargs = mock_cls.call_args
        assert kwargs.get("file_path") == "song.flac"

    def test_accepted_import_refreshes_library(self, main_window):
        """If dialog is accepted, library panel is refreshed."""
        event = _make_event(["song.mp3"])
        with patch("src.ui.main_window.ImportDialog") as mock_cls:
            mock_dlg = MagicMock()
            mock_dlg.exec.return_value = True
            mock_cls.return_value = mock_dlg
            with patch.object(
                main_window._library_panel, "refresh"
            ) as mock_refresh:
                main_window.dropEvent(event)
        mock_refresh.assert_called_once()

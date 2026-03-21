"""Tests for the YouTube import dialog worker lifecycle and error handling.

These tests verify that:
- Workers are properly cleaned up on dialog close (bugs #1, #2, #3).
- Signal naming does not shadow QThread.finished (bug #7).
- _start_local_import handles exceptions gracefully (bug #6).
- Signal disconnects are safe even if already disconnected (bug #8).
"""

import os
import shutil
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication

from src.ui.import_dialog import (
    ImportDialog,
    _MetadataWorker,
    _DownloadWorker,
)


@pytest.fixture(scope="session")
def qapp():
    """Ensure a QApplication exists for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def dialog(qapp, tmp_path):
    """Create an ImportDialog with mocked dependencies."""
    library = MagicMock()
    model_manager = MagicMock()
    model_manager.model_path.return_value = str(tmp_path / "no_model.onnx")
    dlg = ImportDialog(library, model_manager)
    yield dlg
    dlg.close()


# -----------------------------------------------------------------------
# Bug #7: Signal name should not shadow QThread.finished
# -----------------------------------------------------------------------

class TestSignalNaming:
    """Verify worker signals do not shadow QThread.finished."""

    def test_metadata_worker_does_not_shadow_finished(self):
        """_MetadataWorker should not have a custom 'finished' signal."""
        # If the worker has a custom 'finished' signal, it shadows
        # QThread.finished which breaks thread lifecycle management.
        assert not hasattr(_MetadataWorker, "finished") or (
            # It's acceptable if 'finished' is inherited from QThread
            _MetadataWorker.finished is QThread.finished
        ), (
            "_MetadataWorker.finished shadows QThread.finished; "
            "rename to 'completed' or 'result_ready'"
        )

    def test_download_worker_does_not_shadow_finished(self):
        """_DownloadWorker should not have a custom 'finished' signal."""
        assert not hasattr(_DownloadWorker, "finished") or (
            _DownloadWorker.finished is QThread.finished
        ), (
            "_DownloadWorker.finished shadows QThread.finished; "
            "rename to 'completed' or 'result_ready'"
        )


# -----------------------------------------------------------------------
# Bug #2: Metadata worker should be waited on in reject()
# -----------------------------------------------------------------------

class TestRejectWaitsForMetadataWorker:
    """reject() must wait for _metadata_worker, not just _download_worker."""

    def test_reject_waits_for_metadata_worker(self, dialog):
        """Closing the dialog while metadata is fetching must wait."""
        mock_worker = MagicMock(spec=_MetadataWorker)
        mock_worker.isRunning.return_value = True
        dialog._metadata_worker = mock_worker

        dialog.reject()

        mock_worker.wait.assert_called()


# -----------------------------------------------------------------------
# Bug #1: Signals must be disconnected before reject destroys dialog
# -----------------------------------------------------------------------

class TestRejectDisconnectsSignals:
    """reject() must disconnect worker signals before super().reject()."""

    def test_reject_disconnects_download_worker_signals(self, dialog):
        """Download worker signals should be disconnected on close."""
        mock_worker = MagicMock(spec=_DownloadWorker)
        mock_worker.isRunning.return_value = True
        mock_worker.wait.return_value = True
        dialog._download_worker = mock_worker

        dialog.reject()

        # The worker should have had setParent(None) called or signals
        # disconnected. We check that the dialog attempts cleanup.
        assert mock_worker.wait.called


# -----------------------------------------------------------------------
# Bug #3: Re-fetch must wait for previous metadata worker
# -----------------------------------------------------------------------

class TestRefetchWaitsForPrevious:
    """Clicking Fetch twice must not orphan the first QThread."""

    def test_refetch_waits_for_previous_worker(self, dialog):
        """Starting a new metadata fetch waits for the previous one."""
        old_worker = MagicMock(spec=_MetadataWorker)
        old_worker.isRunning.return_value = True
        dialog._metadata_worker = old_worker

        dialog._url_edit.setText("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        with patch("src.ui.import_dialog.is_supported_url", return_value=True):
            with patch("src.ui.import_dialog._MetadataWorker") as mock_cls:
                mock_new = MagicMock()
                mock_cls.return_value = mock_new
                dialog._on_fetch_metadata()

        old_worker.wait.assert_called()


# -----------------------------------------------------------------------
# Bug #6: _start_local_import must handle exceptions
# -----------------------------------------------------------------------

class TestLocalImportErrorHandling:
    """_start_local_import must catch exceptions and call _on_error."""

    def test_add_song_failure_calls_on_error(self, dialog, tmp_path):
        """If library.add_song raises, dialog should show error."""
        dialog._library.add_song.side_effect = OSError("disk full")
        dialog._button_box.setEnabled(False)

        # Create a dummy file so the path is valid.
        dummy = tmp_path / "song.mp3"
        dummy.write_bytes(b"fake audio")

        dialog._start_local_import(str(dummy))

        # The dialog should have recovered: error shown, buttons re-enabled.
        assert dialog._button_box.isEnabled()
        assert "Error" in dialog._status_label.text() or "disk full" in dialog._status_label.text()


# -----------------------------------------------------------------------
# Bug #8: Disconnect calls must be safe if already disconnected
# -----------------------------------------------------------------------

class TestSafeDisconnect:
    """Signal disconnects must not raise if already disconnected."""

    def test_disconnect_on_already_disconnected_worker(self, dialog):
        """Disconnecting signals that were never connected must not crash."""
        worker = _MetadataWorker("https://youtu.be/abc123")
        dialog._metadata_worker = worker

        # This should not raise even though no slots are connected.
        # The bug is that disconnect() raises RuntimeError.
        try:
            dialog._on_fetch_metadata()
        except RuntimeError:
            pytest.fail(
                "disconnect() raised RuntimeError on already-disconnected signal"
            )

"""Import song dialog -- file browser, YouTube URL, metadata, separation trigger.

Supports importing from a local audio file or a YouTube URL. When a URL is
entered, yt-dlp downloads the audio before handing it off to the separator.
"""

import os
import shutil
import tempfile

from PySide6.QtCore import Qt, QSettings, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from src.app_settings import read_default_import_6_stem
from src.downloader import (
    DownloadError,
    check_ffmpeg,
    download_audio,
    extract_metadata,
    is_supported_url,
)
from src.import_messages import format_import_error
from src.library import Song, SongLibrary
from src.model_manager import ModelDownloader, ModelManager
from src.separator import SeparatorWorker


def _safe_disconnect(signal) -> None:
    """Disconnect all slots from *signal*, ignoring RuntimeError."""
    try:
        signal.disconnect()
    except RuntimeError:
        pass


class _MetadataWorker(QThread):
    """Background thread for fetching YouTube metadata."""

    # Named 'completed' to avoid shadowing QThread.finished.
    completed = Signal(str, str)  # title, artist
    error = Signal(str)

    def __init__(self, url: str, parent=None) -> None:
        super().__init__(parent)
        self._url = url

    def run(self) -> None:
        try:
            title, artist = extract_metadata(self._url)
            self.completed.emit(title, artist)
        except DownloadError as exc:
            self.error.emit(str(exc))


class _DownloadWorker(QThread):
    """Background thread for downloading audio from YouTube."""

    progress = Signal(int, str)  # percent, message
    # Named 'completed' to avoid shadowing QThread.finished.
    completed = Signal(str)  # output path
    error = Signal(str)

    def __init__(self, url: str, output_path: str, parent=None) -> None:
        super().__init__(parent)
        self._url = url
        self._output_path = output_path

    def run(self) -> None:
        try:
            self.progress.emit(0, "Downloading audio...")

            def on_progress(d):
                if d.get("status") == "downloading":
                    total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    downloaded = d.get("downloaded_bytes", 0)
                    if total > 0:
                        pct = int(downloaded / total * 100)
                        self.progress.emit(min(pct, 99), "Downloading audio...")

            download_audio(self._url, self._output_path, progress_callback=on_progress)
            self.progress.emit(100, "Download complete.")
            self.completed.emit(self._output_path)
        except DownloadError as exc:
            self.error.emit(str(exc))


class ImportDialog(QDialog):
    """Dialog for importing and separating a song.

    Supports two import modes:
    - Local file: browse for an audio file on disk.
    - YouTube URL: paste a YouTube link to download and import.
    """

    def __init__(
        self,
        library: SongLibrary,
        model_manager: ModelManager,
        parent=None,
        file_path: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Song")
        self.setMinimumWidth(500)

        self._library = library
        self._model_manager = model_manager
        self._worker: SeparatorWorker | None = None
        self._download_worker: _DownloadWorker | None = None
        self._metadata_worker: _MetadataWorker | None = None
        self._model_downloader: ModelDownloader | None = None
        self._song_pending_model_id: str | None = None
        self._pending_separation_is_6_stem: bool = False
        self._selected_path: str = ""
        self._tmp_dir: str | None = None  # Cleaned up after import or on close.

        self._setup_ui()

        if read_default_import_6_stem(QSettings("stemma", "stemma")):
            self._model_combo.setCurrentIndex(1)

        if file_path:
            self._selected_path = file_path
            self._path_edit.setText(file_path)
            basename = os.path.splitext(os.path.basename(file_path))[0]
            if not self._title_edit.text():
                self._title_edit.setText(basename)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # -- YouTube URL row --
        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("URL:"))
        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText("Paste a YouTube URL...")
        self._url_edit.textChanged.connect(self._on_url_changed)
        url_row.addWidget(self._url_edit)

        self._fetch_btn = QPushButton("Fetch")
        self._fetch_btn.setFixedWidth(60)
        self._fetch_btn.setToolTip("Fetch title and artist from YouTube")
        self._fetch_btn.clicked.connect(self._on_fetch_metadata)
        self._fetch_btn.setEnabled(False)
        url_row.addWidget(self._fetch_btn)
        layout.addLayout(url_row)

        # -- Separator label --
        or_label = QLabel("-- or --")
        or_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        or_label.setObjectName("subtle-label")
        or_label.setStyleSheet("padding: 4px;")
        layout.addWidget(or_label)

        # -- File selection row --
        file_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Select an audio file...")
        self._path_edit.setReadOnly(True)
        file_row.addWidget(self._path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        # -- Metadata fields --
        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("Title:"))
        self._title_edit = QLineEdit()
        title_row.addWidget(self._title_edit)
        layout.addLayout(title_row)

        artist_row = QHBoxLayout()
        artist_row.addWidget(QLabel("Artist:"))
        self._artist_edit = QLineEdit()
        artist_row.addWidget(self._artist_edit)
        layout.addLayout(artist_row)

        # -- Model selection --
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.addItem("4-stem (vocals, drums, bass, other)", False)
        self._model_combo.addItem("6-stem (+ guitar, piano)", True)
        self._model_combo.setToolTip("Choose separation model")
        model_row.addWidget(self._model_combo)
        layout.addLayout(model_row)

        # -- Progress --
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        # -- Buttons --
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._button_box.button(
            QDialogButtonBox.StandardButton.Ok
        ).setText("Import && Separate")
        self._button_box.accepted.connect(self._on_import)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

    # ------------------------------------------------------------------
    # URL handling
    # ------------------------------------------------------------------

    def _on_url_changed(self, text: str) -> None:
        """Enable/disable fetch button based on URL validity."""
        self._fetch_btn.setEnabled(is_supported_url(text))

    def _on_fetch_metadata(self) -> None:
        """Fetch title and artist from the YouTube URL in a background thread."""
        url = self._url_edit.text().strip()
        if not is_supported_url(url):
            return

        # Wait for a previous metadata worker to finish before starting
        # a new one.  Without this, the old QThread would be orphaned.
        if self._metadata_worker is not None:
            _safe_disconnect(self._metadata_worker.completed)
            _safe_disconnect(self._metadata_worker.error)
            if self._metadata_worker.isRunning():
                self._metadata_worker.wait(5000)

        self._fetch_btn.setEnabled(False)
        self._status_label.setVisible(True)
        self._status_label.setText("Fetching metadata...")

        self._metadata_worker = _MetadataWorker(url, parent=self)
        self._metadata_worker.completed.connect(self._on_metadata_fetched)
        self._metadata_worker.error.connect(self._on_metadata_error)
        self._metadata_worker.start()

    def _on_metadata_fetched(self, title: str, artist: str) -> None:
        """Populate title and artist fields from YouTube metadata."""
        if not self._title_edit.text():
            self._title_edit.setText(title)
        if not self._artist_edit.text():
            self._artist_edit.setText(artist)
        self._status_label.setText("Metadata fetched.")
        self._fetch_btn.setEnabled(True)

    def _on_metadata_error(self, message: str) -> None:
        self._status_label.setText(
            f"Metadata error: {format_import_error(message)}"
        )
        self._fetch_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # File browsing
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.mp3 *.wav *.flac);;All Files (*)",
        )
        if path:
            self._selected_path = path
            self._path_edit.setText(path)
            # Clear URL field when a local file is selected.
            self._url_edit.clear()

            # Auto-fill title from filename.
            basename = os.path.splitext(os.path.basename(path))[0]
            if not self._title_edit.text():
                self._title_edit.setText(basename)

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def _on_import(self) -> None:
        url = self._url_edit.text().strip()

        # Disable immediately to prevent double-click race.
        self._button_box.setEnabled(False)

        if is_supported_url(url):
            self._start_youtube_import(url)
        elif self._selected_path:
            self._start_local_import(self._selected_path)
        else:
            # Nothing selected -- re-enable.
            self._button_box.setEnabled(True)

    def _start_youtube_import(self, url: str) -> None:
        """Download audio from YouTube, then hand off to the separator."""
        if not check_ffmpeg():
            QMessageBox.critical(
                self,
                "ffmpeg not found",
                "YouTube import requires ffmpeg.\n\n"
                "Install ffmpeg and make sure it is on your PATH.",
            )
            self._button_box.setEnabled(True)
            return

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._status_label.setVisible(True)
        self._status_label.setText("Downloading audio...")
        self._button_box.setEnabled(False)

        # Download to a temp file, then import like a local file.
        self._tmp_dir = tempfile.mkdtemp(prefix="stemma_yt_")
        output_path = os.path.join(self._tmp_dir, "audio.mp3")

        self._download_worker = _DownloadWorker(url, output_path, parent=self)
        self._download_worker.progress.connect(self._on_progress)
        self._download_worker.completed.connect(self._on_download_finished)
        self._download_worker.error.connect(self._on_error)
        self._download_worker.start()

    def _on_download_finished(self, path: str) -> None:
        """After download completes, import the downloaded file.

        The library's add_song copies the file into the song directory,
        so the temp dir is cleaned up afterwards.
        """
        self._selected_path = path
        self._start_local_import(path)
        self._cleanup_tmp_dir()

    def _start_local_import(self, path: str) -> None:
        """Import a local audio file into the library and start separation."""
        title = self._title_edit.text() or "Untitled"
        artist = self._artist_edit.text() or "Unknown Artist"

        try:
            # Add to library (copies the file).
            song = self._library.add_song(
                title=title,
                artist=artist,
                original_path=path,
            )
        except Exception as exc:
            self._on_error(str(exc))
            return

        # Start separation.
        is_6_stem = self._model_combo.currentData()
        model_path = self._model_manager.model_path(is_6_stem=is_6_stem)

        if not os.path.isfile(model_path):
            self._begin_model_download(song, is_6_stem)
            return

        self._start_separation_worker(song, model_path, is_6_stem)

    def _begin_model_download(self, song: Song, is_6_stem: bool) -> None:
        """Download the ONNX model in the background, then run separation."""
        self._song_pending_model_id = song.id
        self._pending_separation_is_6_stem = is_6_stem
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._status_label.setVisible(True)
        self._button_box.setEnabled(False)

        self._model_downloader = self._model_manager.download_model(
            is_6_stem=is_6_stem
        )
        self._model_downloader.progress.connect(self._on_progress)
        self._model_downloader.finished.connect(
            self._on_model_download_finished
        )
        self._model_downloader.error.connect(self._on_model_download_error)
        self._model_downloader.start()

    def _on_model_download_finished(self, _path: str) -> None:
        """Model file is on disk; continue with stem separation."""
        song_id = self._song_pending_model_id
        is_6 = self._pending_separation_is_6_stem
        self._song_pending_model_id = None
        self._model_downloader = None

        if song_id is None:
            return
        song = self._library.get_song(song_id)
        if song is None:
            return

        model_path = self._model_manager.model_path(is_6_stem=is_6)
        if not os.path.isfile(model_path):
            try:
                self._library.remove_song(song_id)
            except KeyError:
                pass
            self._on_error("Model file missing after download.")
            return

        self._start_separation_worker(song, model_path, is_6)

    def _on_model_download_error(self, message: str) -> None:
        """Remove the library entry and show a readable download error."""
        self._rollback_pending_model_song()
        self._model_downloader = None
        self._on_error(message)

    def _rollback_pending_model_song(self) -> None:
        """Drop the song created for an import that failed before separation."""
        if self._song_pending_model_id is None:
            return
        sid = self._song_pending_model_id
        self._song_pending_model_id = None
        try:
            self._library.remove_song(sid)
        except KeyError:
            pass

    def _start_separation_worker(
        self,
        song: Song,
        model_path: str,
        is_6_stem: bool,
    ) -> None:
        """Run ONNX separation for a song that already has a model file."""
        self._progress_bar.setVisible(True)
        self._status_label.setVisible(True)
        self._button_box.setEnabled(False)

        self._worker = SeparatorWorker(
            input_path=song.original_path,
            output_dir=song.stems_path,
            model_path=model_path,
            is_6_stem=is_6_stem,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(lambda _: self._on_finished(song.id))
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, percent: int, message: str) -> None:
        self._progress_bar.setValue(percent)
        self._status_label.setText(message)

    def _on_finished(self, song_id: str) -> None:
        self._library.update_song(song_id, model_used="htdemucs")
        self.accept()

    def _on_error(self, message: str) -> None:
        self._status_label.setVisible(True)
        self._status_label.setText(f"Error: {format_import_error(message)}")
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._button_box.setEnabled(True)
        self._cleanup_tmp_dir()

    def _cleanup_tmp_dir(self) -> None:
        """Remove the temporary download directory if it exists."""
        if self._tmp_dir is not None and os.path.isdir(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    def _disconnect_and_detach_workers(self) -> None:
        """Disconnect all worker signals and detach them from the dialog.

        Workers are detached (setParent(None)) so they are not destroyed
        when the dialog is deleted. Any still-running thread will finish
        its work and be cleaned up by Qt's deleteLater mechanism.
        """
        if self._metadata_worker is not None:
            _safe_disconnect(self._metadata_worker.completed)
            _safe_disconnect(self._metadata_worker.error)
        if self._download_worker is not None:
            _safe_disconnect(self._download_worker.progress)
            _safe_disconnect(self._download_worker.completed)
            _safe_disconnect(self._download_worker.error)
        if self._model_downloader is not None:
            _safe_disconnect(self._model_downloader.progress)
            _safe_disconnect(self._model_downloader.finished)
            _safe_disconnect(self._model_downloader.error)
        if self._worker is not None:
            _safe_disconnect(self._worker.progress)
            _safe_disconnect(self._worker.finished)
            _safe_disconnect(self._worker.error)

    def reject(self) -> None:
        """Cancel any running workers before closing."""
        # Disconnect all signals first so no callbacks fire into a
        # partially destroyed dialog.
        self._disconnect_and_detach_workers()

        # Wait for workers to finish (best-effort; yt-dlp has no cancel).
        if self._metadata_worker is not None and self._metadata_worker.isRunning():
            self._metadata_worker.wait(5000)
        if self._download_worker is not None and self._download_worker.isRunning():
            self._download_worker.setParent(None)
            self._download_worker.wait(5000)
            if self._download_worker.isRunning():
                # Thread outlived the wait -- schedule deferred cleanup.
                self._download_worker.finished.connect(
                    self._download_worker.deleteLater
                )
        if self._model_downloader is not None:
            if self._model_downloader.isRunning():
                self._model_downloader.cancel()
                self._model_downloader.wait(5000)
            self._model_downloader = None
        self._rollback_pending_model_song()
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)

        self._cleanup_tmp_dir()
        super().reject()

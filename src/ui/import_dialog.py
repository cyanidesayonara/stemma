"""Import song dialog -- file browser, YouTube URL, metadata, separation trigger.

Supports importing from a local audio file or a YouTube URL. When a URL is
entered, yt-dlp downloads the audio before handing it off to the separator.
"""

import os
import shutil
import tempfile

import soundfile as sf
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from src.app_settings import (
    SEPARATION_MODEL_TOOLTIP,
    SEPARATION_MODELS,
    open_settings,
    read_default_import_model,
)
from src.crash_log import logger
from src.downloader import (
    DownloadCancelled,
    DownloadError,
    check_ffmpeg,
    download_audio,
    extract_metadata,
    ffmpeg_missing_message,
    is_supported_url,
)
from src.import_messages import (
    MSG_UNREADABLE_AUDIO,
    MSG_YT_ONLY,
    describe_error,
    format_import_error,
)
from src.library import Song, SongLibrary
from src.mdx_separator import MdxSeparatorWorker
from src.model_manager import ModelDownloader, ModelManager, model_label
from src.qt_signal_utils import safe_disconnect as _safe_disconnect
from src.separator import (
    SeparatorWorker,
    available_memory_bytes,
    estimate_separation_memory,
)


# Separation loads the full source into RAM; warn above this size (bytes).
_LARGE_SOURCE_WARN_BYTES = 100 * 1024 * 1024


def _is_readable_audio(path: str) -> bool:
    """True if soundfile can open *path* and it holds some audio.

    Checked before the song is added or a model is downloaded, so a
    corrupt or unsupported file fails in the dialog with a clear message
    instead of deep inside separation.
    """
    try:
        info = sf.info(path)
    except Exception as exc:  # noqa: BLE001 - any failure means unreadable
        describe_error(exc, f"Import check failed for {path}")
        return False
    return info.frames > 0 and info.channels > 0 and info.samplerate > 0


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
            self.error.emit(describe_error(exc, "YouTube metadata failed"))


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
        # The bar never moves backwards: a retry starts its download over,
        # so its early progress is below what the failed attempt reached.
        shown = 0

        def on_progress(d):
            nonlocal shown
            if d.get("status") != "downloading":
                return
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                pct = min(int(downloaded / total * 100), 99)
                if pct > shown:
                    shown = pct
                    self.progress.emit(shown, "Downloading audio...")

        def on_retry(attempt, _delay):
            self.progress.emit(shown, f"Retrying download (attempt {attempt})...")

        try:
            self.progress.emit(0, "Downloading audio...")
            download_audio(
                self._url,
                self._output_path,
                progress_callback=on_progress,
                # Closing the dialog stops any automatic retry.
                should_cancel=self.isInterruptionRequested,
                on_retry=on_retry,
            )
            self.progress.emit(100, "Download complete.")
            self.completed.emit(self._output_path)
        except DownloadCancelled as exc:
            # The user closed the dialog: not a failure worth a warning.
            logger.info("YouTube download cancelled: %s", exc)
        except DownloadError as exc:
            self.error.emit(describe_error(exc, "YouTube download failed"))


def _rule() -> QFrame:
    """A 1 px horizontal line in the divider colour."""
    line = QFrame()
    line.setObjectName("divider-line")
    line.setFixedHeight(1)
    return line


def _or_divider() -> QWidget:
    """A rule with "or" in the middle, between the two ways to import."""
    row = QWidget()
    box = QHBoxLayout(row)
    box.setContentsMargins(0, 2, 0, 2)
    label = QLabel("or")
    label.setObjectName("subtle-label")
    box.addWidget(_rule(), 1)
    box.addWidget(label)
    box.addWidget(_rule(), 1)
    return row


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
        separation_queue=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Song")
        self.setMinimumWidth(500)

        self._library = library
        self._model_manager = model_manager
        # When a SeparationQueue is provided (the main window always
        # passes one), separation is handed off to it and the dialog
        # closes immediately after acquisition + model download; the
        # library row then shows live progress. Without a queue the
        # dialog falls back to running the worker inline (blocking),
        # which standalone/test usage relies on.
        self._separation_queue = separation_queue
        self._worker: SeparatorWorker | MdxSeparatorWorker | None = None
        self._download_worker: _DownloadWorker | None = None
        self._metadata_worker: _MetadataWorker | None = None
        self._model_downloader: ModelDownloader | None = None
        self._import_song_id: str | None = None
        self._pending_model_key: str = "htdemucs"
        self._selected_path: str = ""
        self._tmp_dir: str | None = None  # Cleaned up after import or on close.
        # Model keys the user already agreed to download in this dialog,
        # so a YouTube import asks once, before the audio download.
        self._download_consent: set[str] = set()

        self._setup_ui()

        self._model_combo.setCurrentIndex(max(0, self._model_combo.findData(
            read_default_import_model(open_settings()),
        )))

        if file_path:
            self._selected_path = file_path
            self._path_edit.setText(file_path)
            basename = os.path.splitext(os.path.basename(file_path))[0]
            if not self._title_edit.text():
                self._title_edit.setText(basename)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        # One label column for every field: separate rows started each
        # field at a different x, after labels of different widths.
        form = QFormLayout()
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        form.setVerticalSpacing(8)
        layout.addLayout(form)

        # -- YouTube URL row --
        url_row = QHBoxLayout()
        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText("Paste a YouTube URL...")
        self._url_edit.textChanged.connect(self._on_url_changed)
        url_row.addWidget(self._url_edit)

        self._fetch_btn = QPushButton("Fetch")
        self._fetch_btn.setToolTip("Fetch title and artist from YouTube")
        self._fetch_btn.clicked.connect(self._on_fetch_metadata)
        self._fetch_btn.setEnabled(False)
        url_row.addWidget(self._fetch_btn)
        # A row that is a layout gets no buddy from addRow, so link the
        # label by hand: screen readers name the field from it.
        url_label = QLabel("YouTube &URL:")
        url_label.setBuddy(self._url_edit)
        form.addRow(url_label, url_row)

        form.addRow(_or_divider())

        # -- File selection row --
        file_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Select an audio file...")
        self._path_edit.setReadOnly(True)
        file_row.addWidget(self._path_edit)

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse)
        file_row.addWidget(browse_btn)
        # Both source rows end at the same x.
        button_width = max(
            browse_btn.sizeHint().width(), self._fetch_btn.sizeHint().width()
        )
        browse_btn.setFixedWidth(button_width)
        self._fetch_btn.setFixedWidth(button_width)
        file_label = QLabel("Audio &file:")
        file_label.setBuddy(self._path_edit)
        form.addRow(file_label, file_row)

        # A little air between the source and the song's details.
        form.addItem(QSpacerItem(0, 1))

        # -- Metadata fields --
        self._title_edit = QLineEdit()
        form.addRow("&Title:", self._title_edit)
        self._artist_edit = QLineEdit()
        form.addRow("&Artist:", self._artist_edit)

        # -- Model selection --
        self._model_combo = QComboBox()
        for model_key, label in SEPARATION_MODELS:
            self._model_combo.addItem(label, model_key)
        self._model_combo.setToolTip(SEPARATION_MODEL_TOOLTIP)
        form.addRow("&Model:", self._model_combo)

        # -- Progress --
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        # Error text is a full sentence or two; wrap instead of clipping.
        self._status_label.setWordWrap(True)
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        self._retry_btn = QPushButton("Retry import")
        self._retry_btn.setVisible(False)
        self._retry_btn.clicked.connect(self._on_retry_import)
        layout.addWidget(self._retry_btn)

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

    def _set_busy(self, busy: bool) -> None:
        """Block a second import while one runs; Cancel always stays on.

        Cancel is how the user stops a model or audio download, so only
        the import button is disabled.
        """
        self._button_box.button(
            QDialogButtonBox.StandardButton.Ok
        ).setEnabled(not busy)

    def _set_status(self, text: str) -> None:
        """Show *text* in the status line, growing the dialog to fit it."""
        self._status_label.setVisible(True)
        self._status_label.setText(text)
        # After the caller has finished showing and hiding the other rows,
        # so the label's width is final.
        QTimer.singleShot(0, self._fit_status)

    def _fit_status(self) -> None:
        """Reserve the wrapped status text's full height.

        The label wraps, but this layout has no height-for-width, so the
        dialog never grew taller and a second line was cut off under the
        buttons. A minimum height makes the dialog grow to fit.
        """
        label = self._status_label
        if label.isVisible() and label.text():
            label.setMinimumHeight(label.heightForWidth(label.width()))
        else:
            label.setMinimumHeight(0)

    def resizeEvent(self, event) -> None:
        """Refit the status text when the width changes its wrapping."""
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self._fit_status()

    def _show_problem(self, message: str) -> None:
        """Show *message* in the status line and allow another try."""
        self._set_status(message)
        self._progress_bar.setVisible(False)
        self._set_busy(False)

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
        self._set_status("Fetching metadata...")

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
        self._set_status("Metadata fetched.")
        self._fetch_btn.setEnabled(True)

    def _on_metadata_error(self, message: str) -> None:
        self._set_status(
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

        self._retry_btn.setVisible(False)

        # Disable immediately to prevent double-click race.
        self._set_busy(True)

        if is_supported_url(url):
            self._start_youtube_import(url)
        elif url:
            # Something that is not a YouTube link used to do nothing.
            self._show_problem(MSG_YT_ONLY)
        elif self._selected_path:
            if not self._warn_large_source_ok(self._selected_path):
                self._set_busy(False)
                return
            self._start_local_import(self._selected_path)
        else:
            # Nothing selected -- re-enable.
            self._set_busy(False)

    def _start_youtube_import(self, url: str) -> None:
        """Download audio from YouTube, then hand off to the separator."""
        if not check_ffmpeg():
            QMessageBox.critical(
                self, "ffmpeg not found", ffmpeg_missing_message(),
            )
            self._set_busy(False)
            return

        # Ask about a model download now, not after the audio arrives.
        if not self._confirm_model_download(self._model_combo.currentData()):
            self._set_busy(False)
            return

        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._set_status("Downloading audio...")
        self._set_busy(True)

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
        if not self._warn_large_source_ok(path):
            self._set_busy(False)
            self._cleanup_tmp_dir()
            return
        self._start_local_import(path)
        self._cleanup_tmp_dir()

    def _start_local_import(self, path: str) -> None:
        """Import a local audio file into the library and start separation.

        The file is checked, and any model download agreed to, before the
        library is touched, so a refusal leaves nothing behind.
        """
        model_key = self._model_combo.currentData()
        if not _is_readable_audio(path):
            self._show_problem(MSG_UNREADABLE_AUDIO)
            return
        if not model_key.startswith("mdx_"):
            is_6_stem = model_key == "htdemucs_6s"
            if not self._check_memory_ok(path, is_6_stem):
                self._set_busy(False)
                return
        if not self._confirm_model_download(model_key):
            self._set_busy(False)
            return

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
            self._on_error(describe_error(exc, "Import failed"))
            return

        self._import_song_id = song.id

        # Start separation.
        model_path = self._model_path_for(model_key)

        if not self._model_downloaded_for(model_key):
            self._begin_model_download(song, model_key)
            return

        self._start_separation_worker(song, model_path, model_key)

    def _model_path_for(self, model_key: str) -> str:
        """Resolve the on-disk ONNX path for a separation model key."""
        if model_key.startswith("mdx_"):
            return self._model_manager.mdx_model_path(model_key)
        return self._model_manager.model_path(
            is_6_stem=model_key == "htdemucs_6s"
        )

    def _model_downloaded_for(self, model_key: str) -> bool:
        """True when all artifacts for *model_key* exist on disk.

        Uses the manager checks rather than a bare isfile() so the
        two-artifact HTDemucs models aren't treated as cached when only
        the graph (and not the .onnx.data weights) survived.
        """
        if model_key.startswith("mdx_"):
            return self._model_manager.is_mdx_model_downloaded(model_key)
        return self._model_manager.is_model_downloaded(
            is_6_stem=model_key == "htdemucs_6s"
        )

    def _confirm_model_download(self, model_key: str) -> bool:
        """Ask before the first download of *model_key*; True to go on.

        Returns True without asking when the model is already on disk or
        the user agreed earlier in this dialog.
        """
        if model_key in self._download_consent:
            return True
        if self._model_downloaded_for(model_key):
            return True
        size_mb = self._model_manager.download_size_mb(model_key)
        reply = QMessageBox.question(
            self,
            "Download model",
            f"stemma needs to download the {model_label(model_key)} "
            f"(about {size_mb} MB) once. It stays on this PC for later "
            "imports.\n\nDownload it now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return False
        self._download_consent.add(model_key)
        return True

    def _begin_model_download(self, song: Song, model_key: str) -> None:
        """Download the ONNX model in the background, then run separation."""
        self._pending_model_key = model_key
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._set_status(f"Downloading the {model_label(model_key)}...")
        self._set_busy(True)

        if model_key.startswith("mdx_"):
            self._model_downloader = self._model_manager.download_mdx_model(
                model_key
            )
        else:
            self._model_downloader = self._model_manager.download_model(
                is_6_stem=model_key == "htdemucs_6s"
            )
        self._model_downloader.progress.connect(self._on_progress)
        self._model_downloader.download_complete.connect(
            self._on_model_download_finished
        )
        self._model_downloader.error.connect(self._on_model_download_error)
        self._model_downloader.start()

    def _on_model_download_finished(self, _path: str) -> None:
        """Model file is on disk; continue with stem separation."""
        model_key = self._pending_model_key
        self._model_downloader = None

        song_id = self._import_song_id
        if song_id is None:
            return
        song = self._library.get_song(song_id)
        if song is None:
            return

        model_path = self._model_path_for(model_key)
        if not os.path.isfile(model_path):
            try:
                self._library.remove_song(song_id)
            except KeyError:
                pass
            self._import_song_id = None
            self._on_error("Model file missing after download.")
            return

        self._start_separation_worker(song, model_path, model_key)

    def _on_model_download_error(self, message: str) -> None:
        """Show a readable download error and roll back the library entry."""
        self._model_downloader = None
        self._on_error(message)

    def _discard_failed_import_song(self) -> None:
        """Remove the song row created for an import that did not complete."""
        if self._import_song_id is None:
            return
        sid = self._import_song_id
        self._import_song_id = None
        if self._library.get_song(sid) is not None:
            try:
                self._library.remove_song(sid)
            except KeyError:
                pass

    def _warn_large_source_ok(self, path: str) -> bool:
        """If *path* is very large, ask the user to confirm. Return False to abort."""
        try:
            size = os.path.getsize(path)
        except OSError:
            return True
        if size < _LARGE_SOURCE_WARN_BYTES:
            return True
        mb = size / (1024 * 1024)
        reply = QMessageBox.question(
            self,
            "Large file",
            f"This file is about {mb:.0f} MB. Separation loads the full track "
            "into memory and may be slow or fail on systems with limited RAM.\n\n"
            "Continue with import?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _on_retry_import(self) -> None:
        """Re-run import with the same path or URL after an error."""
        self._retry_btn.setVisible(False)
        self._on_import()

    def _start_separation_worker(
        self,
        song: Song,
        model_path: str,
        model_key: str,
    ) -> None:
        """Start separation: hand off to the queue, or run inline.

        The caller has already completed the Demucs RAM confirmation
        against the source path before adding the song to the library.
        """
        if self._separation_queue is not None:
            # Background path: enqueue and close. The song row is
            # finalized (model_used) or rolled back by the main window
            # when the queue reports the job's end; the dialog's own
            # reject-time rollback must no longer touch it.
            from src.separation_queue import SeparationJob

            self._separation_queue.enqueue(SeparationJob(
                song_id=song.id,
                input_path=song.original_path,
                output_dir=song.stems_path,
                model_path=model_path,
                model_key=model_key,
            ))
            self._import_song_id = None
            self.accept()
            return

        self._progress_bar.setVisible(True)
        self._status_label.setVisible(True)
        self._set_busy(True)

        if model_key.startswith("mdx_"):
            self._worker = MdxSeparatorWorker(
                input_path=song.original_path,
                output_dir=song.stems_path,
                model_path=model_path,
                model_key=model_key,
            )
        else:
            self._worker = SeparatorWorker(
                input_path=song.original_path,
                output_dir=song.stems_path,
                model_path=model_path,
                is_6_stem=model_key == "htdemucs_6s",
            )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(
            lambda _: self._on_finished(song.id, model_key)
        )
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _check_memory_ok(self, audio_path: str, is_6_stem: bool) -> bool:
        """Warn and ask the user if available RAM looks insufficient.

        Returns True to proceed, False to abort.
        """
        try:
            info = sf.info(audio_path)
            duration = info.duration
        except Exception:
            return True

        needed = estimate_separation_memory(duration, is_6_stem)
        avail = available_memory_bytes()
        if avail is None or avail >= needed:
            return True

        needed_gb = needed / (1024 ** 3)
        avail_gb = avail / (1024 ** 3)
        reply = QMessageBox.warning(
            self,
            "Low memory",
            f"Stem separation needs roughly {needed_gb:.1f} GB of free memory, "
            f"but only {avail_gb:.1f} GB is available.\n\n"
            "Close other applications or try a shorter audio file.\n\n"
            "Continue anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _on_progress(self, percent: int, message: str) -> None:
        self._progress_bar.setValue(percent)
        self._set_status(message)

    def _on_finished(self, song_id: str, model_key: str) -> None:
        self._import_song_id = None
        # The model key doubles as the stored model_used value; the
        # demucs keys match the strings recorded by earlier versions.
        self._library.update_song(song_id, model_used=model_key)
        self.accept()

    def _on_error(self, message: str) -> None:
        self._discard_failed_import_song()
        self._set_status(f"Error: {format_import_error(message)}")
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._set_busy(False)
        self._retry_btn.setVisible(True)
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
            _safe_disconnect(self._model_downloader.download_complete)
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
            # yt-dlp cannot stop mid-attempt, but no retry starts after this.
            self._download_worker.requestInterruption()
            self._download_worker.setParent(None)
            self._download_worker.wait(5000)
            if self._download_worker.isRunning():
                # Thread outlived the wait -- schedule deferred cleanup.
                self._download_worker.finished.connect(
                    self._download_worker.deleteLater
                )
        if self._model_downloader is not None:
            dl = self._model_downloader
            self._model_downloader = None
            if dl.isRunning():
                dl.cancel()
                dl.wait(5000)
            if dl.isRunning():
                dl.setParent(None)
                dl.finished.connect(dl.deleteLater)
            else:
                dl.deleteLater()
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)
        self._discard_failed_import_song()

        self._cleanup_tmp_dir()
        super().reject()

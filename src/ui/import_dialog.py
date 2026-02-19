"""Import song dialog -- file browser, metadata, separation trigger.

Full implementation in ticket #11. This provides the minimal working
dialog needed for the main window's File > Import action.
"""

import os

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from src.library import SongLibrary
from src.model_manager import ModelManager
from src.separator import SeparatorWorker


class ImportDialog(QDialog):
    """Dialog for importing and separating a song."""

    def __init__(
        self,
        library: SongLibrary,
        model_manager: ModelManager,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Song")
        self.setMinimumWidth(500)

        self._library = library
        self._model_manager = model_manager
        self._worker: SeparatorWorker | None = None
        self._selected_path: str = ""

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # File selection row.
        file_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Select an audio file...")
        self._path_edit.setReadOnly(True)
        file_row.addWidget(self._path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        # Metadata fields.
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

        # Progress.
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        # Buttons.
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

            # Auto-fill title from filename.
            basename = os.path.splitext(os.path.basename(path))[0]
            if not self._title_edit.text():
                self._title_edit.setText(basename)

    def _on_import(self) -> None:
        if not self._selected_path:
            return

        title = self._title_edit.text() or "Untitled"
        artist = self._artist_edit.text() or "Unknown Artist"

        # Add to library (copies the file).
        song = self._library.add_song(
            title=title,
            artist=artist,
            original_path=self._selected_path,
        )

        # Start separation.
        model_path = self._model_manager.model_path(is_6_stem=False)

        if not os.path.isfile(model_path):
            # Model not downloaded yet -- skip separation for now.
            self._library.update_song(song.id, model_used="pending")
            self.accept()
            return

        self._progress_bar.setVisible(True)
        self._status_label.setVisible(True)
        self._button_box.setEnabled(False)

        self._worker = SeparatorWorker(
            input_path=song.original_path,
            output_dir=song.stems_path,
            model_path=model_path,
            is_6_stem=False,
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
        self._status_label.setText(f"Error: {message}")
        self._button_box.setEnabled(True)

    def reject(self) -> None:
        """Cancel any running separation before closing."""
        if hasattr(self, "_worker") and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)
        super().reject()

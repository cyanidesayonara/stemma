"""Main application window layout.

Left panel: song library list.
Center: player controls and stem mixer.
Menu bar: File > Import / Export.
"""

import os

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
)

from src.exporter import ExportWorker, StemExporter
from src.library import SongLibrary
from src.model_manager import ModelManager
from src.player import MultiTrackPlayer
from src.ui.library_panel import LibraryPanel
from src.ui.player_controls import PlayerControls

# Try loading all components in this preferred visual layout order
ALL_STEM_NAMES = ("vocals", "drums", "bass", "other", "guitar", "piano")


class MainWindow(QMainWindow):
    """Top-level application window for stemma."""

    def __init__(
        self,
        library: SongLibrary,
        player: MultiTrackPlayer,
        model_manager: ModelManager,
    ) -> None:
        super().__init__()
        self.setWindowTitle("stemma")
        self.setMinimumSize(900, 600)

        self._library = library
        self._player = player
        self._model_manager = model_manager
        self._current_song_id: str | None = None
        self._export_worker: ExportWorker | None = None

        self._settings = QSettings("stemma", "stemma")

        self._setup_ui()
        self._setup_menu()
        self._setup_shortcuts()
        self._connect_signals()
        self._restore_state()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        """Build the main window layout."""
        self._library_panel = LibraryPanel(self._library)
        self._player_controls = PlayerControls(self._player)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._library_panel)
        splitter.addWidget(self._player_controls)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        self.setCentralWidget(splitter)

    def _setup_menu(self) -> None:
        """Create the menu bar."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        import_action = file_menu.addAction("&Import Song...")
        import_action.triggered.connect(self._on_import)

        export_action = file_menu.addAction("&Export Mix...")
        export_action.triggered.connect(self._on_export)

        file_menu.addSeparator()

        quit_action = file_menu.addAction("&Quit")
        quit_action.triggered.connect(self.close)

    def _setup_shortcuts(self) -> None:
        """Register global keyboard shortcuts."""
        # Transport
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(
            self._on_shortcut_play_pause
        )
        QShortcut(QKeySequence(Qt.Key.Key_S), self).activated.connect(
            self._player.stop
        )
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(
            lambda: self._player.seek(
                max(0.0, self._player.current_seconds - 5.0)
            )
        )
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(
            lambda: self._player.seek(self._player.current_seconds + 5.0)
        )

        # A-B loop
        QShortcut(QKeySequence(Qt.Key.Key_A), self).activated.connect(
            self._player_controls._on_set_loop_a
        )
        QShortcut(QKeySequence(Qt.Key.Key_B), self).activated.connect(
            self._player_controls._on_set_loop_b
        )
        QShortcut(QKeySequence(Qt.Key.Key_L), self).activated.connect(
            self._player_controls.toggle_looping
        )

        # Number keys 1–6 toggle mute on corresponding stem
        stem_order = list(ALL_STEM_NAMES)
        for i, key in enumerate([
            Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3,
            Qt.Key.Key_4, Qt.Key.Key_5, Qt.Key.Key_6,
        ]):
            stem_name = stem_order[i]
            QShortcut(QKeySequence(key), self).activated.connect(
                self._make_mute_toggler(stem_name)
            )

    def _make_mute_toggler(self, stem_name: str):
        """Return a callable that toggles mute for a stem and updates the UI."""
        def toggle():
            self._player_controls.toggle_stem_mute(stem_name)
        return toggle

    def _on_shortcut_play_pause(self) -> None:
        """Toggle play/pause via keyboard shortcut."""
        if self._player.is_playing:
            self._player.pause()
        else:
            self._player.play()

    def _restore_state(self) -> None:
        """Restore saved window geometry and state."""
        geometry = self._settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = self._settings.value("window/state")
        if state is not None:
            self.restoreState(state)

    def closeEvent(self, event) -> None:
        """Save window geometry/state and clean up background threads."""
        self._settings.setValue("window/geometry", self.saveGeometry())
        self._settings.setValue("window/state", self.saveState())

        # Wait for any in-flight export to finish.
        if self._export_worker is not None and self._export_worker.isRunning():
            self._export_worker.wait(5000)

        # Stop playback.
        self._player.stop()

        super().closeEvent(event)

    def _connect_signals(self) -> None:
        """Wire up signals between panels."""
        self._library_panel.song_selected.connect(self._on_song_selected)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_song_selected(self, song_id: str) -> None:
        """Load stems for the selected song into the player."""
        song = self._library.get_song(song_id)
        if song is None:
            return

        stem_names = ALL_STEM_NAMES
        stem_paths = {}
        for name in stem_names:
            path = os.path.join(song.stems_path, f"{name}.wav")
            if os.path.isfile(path):
                stem_paths[name] = path

        if stem_paths:
            self._player.stop()
            self._player.load_stems(stem_paths)
            self._player_controls.set_stem_names(list(stem_paths.keys()))
            self._current_song_id = song_id

    def _on_import(self) -> None:
        """Open the import dialog."""
        from src.ui.import_dialog import ImportDialog

        dialog = ImportDialog(
            library=self._library,
            model_manager=self._model_manager,
            parent=self,
        )
        if dialog.exec():
            self._library_panel.refresh()

    def _on_export(self) -> None:
        """Export the current mix as a WAV file."""
        if not self._player.has_stems:
            QMessageBox.information(
                self, "Export", "No stems loaded. Import and separate a song first."
            )
            return

        song = self._library.get_song(self._current_song_id) if self._current_song_id else None
        if song is None:
            return

        stem_paths = {}
        for name in ALL_STEM_NAMES:
            path = os.path.join(song.stems_path, f"{name}.wav")
            if os.path.isfile(path):
                stem_paths[name] = path

        if not stem_paths:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Mix", "",
            "WAV Files (*.wav);;MP3 Files (*.mp3)"
        )
        if path:
            exporter = StemExporter(stem_paths)
            volumes = {
                name: self._player.get_volume(name)
                for name in stem_paths
            }
            self._export_worker = ExportWorker(
                exporter=exporter,
                output_path=path,
                muted_stems=self._player.muted_stems,
                volumes=volumes,
            )
            self._export_worker.finished.connect(self._on_export_finished)
            self._export_worker.error.connect(self._on_export_error)
            
            self.statusBar().showMessage("Exporting mix... please wait.", 10000)
            self._export_worker.start()

    def _on_export_finished(self, path: str) -> None:
        self.statusBar().showMessage("Export complete.", 5000)
        QMessageBox.information(self, "Export", f"Mix successfully exported to {path}")

    def _on_export_error(self, err: str) -> None:
        self.statusBar().clearMessage()
        QMessageBox.critical(self, "Export Error", f"Failed to export mix:\n{err}")

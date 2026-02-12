"""Main application window layout.

Left panel: song library list.
Center: player controls and stem mixer.
Menu bar: File > Import / Export.
"""

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
)

from src.exporter import StemExporter
from src.library import SongLibrary
from src.model_manager import ModelManager
from src.player import MultiTrackPlayer
from src.separator import STEMS_4, STEMS_6
from src.ui.library_panel import LibraryPanel
from src.ui.player_controls import PlayerControls

ALL_STEM_NAMES = STEMS_6  # Superset: try loading all possible stems.


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

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()

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
            self, "Export Mix", "", "WAV Files (*.wav)"
        )
        if path:
            exporter = StemExporter(stem_paths)
            volumes = {
                name: self._player.get_volume(name)
                for name in stem_paths
            }
            exporter.export_mix(
                path,
                muted_stems=self._player.muted_stems,
                volumes=volumes,
            )
            QMessageBox.information(self, "Export", f"Mix exported to {path}")

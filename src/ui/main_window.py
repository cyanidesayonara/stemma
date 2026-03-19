"""Main application window layout.

Left panel: song library list.
Center: player controls and stem mixer.
Menu bar: File > Import / Export, View > Theme.
"""

import os

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
)

from src.exporter import ExportWorker, StemExporter
from src.library import SongLibrary
from src.model_manager import ModelManager
from src.player import MultiTrackPlayer
from src.ui.import_dialog import ImportDialog
from src.ui.library_panel import LibraryPanel
from src.ui.player_controls import PlayerControls
from src.ui.styles import get_colors, get_stylesheet
from src.version import __version__

# Try loading all components in this preferred visual layout order
ALL_STEM_NAMES = ("vocals", "drums", "bass", "other", "guitar", "piano")
_AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".flac"})


def _is_audio_path(path: str) -> bool:
    """Return True if *path* has an audio file extension."""
    _, ext = os.path.splitext(path)
    return ext.lower() in _AUDIO_EXTENSIONS


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
        self._theme = self._settings.value("theme", "dark")
        if self._theme not in ("dark", "light"):
            self._theme = "dark"

        self._setup_ui()
        self._setup_menu()
        self._setup_shortcuts()
        self._connect_signals()
        self._restore_state()
        self.setAcceptDrops(True)

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
        self.setStatusBar(None)  # No global status bar

    def _setup_menu(self) -> None:
        """Create the menu bar."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        import_action = file_menu.addAction("&Import Song...")
        import_action.triggered.connect(self._on_import)

        export_action = file_menu.addAction("&Export Mix...")
        export_action.triggered.connect(self._on_export)

        close_song_action = file_menu.addAction("&Close Song")
        close_song_action.triggered.connect(self._on_close_song)

        file_menu.addSeparator()

        quit_action = file_menu.addAction("&Quit")
        quit_action.triggered.connect(self.close)

        view_menu = menu_bar.addMenu("&View")
        self._theme_action = view_menu.addAction("&Light Theme")
        self._theme_action.setCheckable(True)
        self._theme_action.setChecked(self._theme == "light")
        self._theme_action.toggled.connect(self._on_theme_toggled)

        help_menu = menu_bar.addMenu("&Help")

        # Corner toggle button (right side of menu bar)
        self._theme_btn = QPushButton()
        self._theme_btn.setObjectName("theme-toggle")
        self._theme_btn.setAccessibleName("Toggle theme")
        self._theme_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._theme_btn.clicked.connect(
            lambda: self._theme_action.setChecked(
                not self._theme_action.isChecked()
            )
        )
        self._update_theme_btn()
        menu_bar.setCornerWidget(self._theme_btn)
        about_action = help_menu.addAction("&About stemma")
        about_action.triggered.connect(self._on_about)

    def _setup_shortcuts(self) -> None:
        """Register global keyboard shortcuts."""
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

        QShortcut(QKeySequence(Qt.Key.Key_A), self).activated.connect(
            self._player_controls.set_loop_a
        )
        QShortcut(QKeySequence(Qt.Key.Key_B), self).activated.connect(
            self._player_controls.set_loop_b
        )
        QShortcut(QKeySequence(Qt.Key.Key_L), self).activated.connect(
            self._player_controls.toggle_looping
        )

        QShortcut(QKeySequence(Qt.Key.Key_BracketRight), self).activated.connect(
            lambda: self._player_controls.cycle_speed(1)
        )
        QShortcut(QKeySequence(Qt.Key.Key_BracketLeft), self).activated.connect(
            lambda: self._player_controls.cycle_speed(-1)
        )

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

    def _update_theme_btn(self) -> None:
        """Update the corner toggle button text and tooltip for the active theme."""
        if self._theme == "dark":
            self._theme_btn.setText("\u2600")
            self._theme_btn.setToolTip("Switch to light theme")
        else:
            self._theme_btn.setText("\u263D")
            self._theme_btn.setToolTip("Switch to dark theme")

    def _on_theme_toggled(self, checked: bool) -> None:
        """Switch between light and dark themes."""
        self._theme = "light" if checked else "dark"
        self._settings.setValue("theme", self._theme)

        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(get_stylesheet(self._theme))

        colors = get_colors(self._theme)
        self._player_controls.apply_theme(self._theme, colors)
        self._update_theme_btn()

    def _on_about(self) -> None:
        """Show the About dialog with the main logo rendered from SVG."""
        dlg = QDialog(self)
        dlg.setWindowTitle("About stemma")
        dlg.setFixedSize(540, 280)

        from src.ui.player_controls import _ROOT_DIR, _logo_variant, _render_svg

        variant = _logo_variant(self._theme)
        svg_path = os.path.join(
            _ROOT_DIR, "assets", "icons", f"logo_main_{variant}.svg"
        )

        outer = QHBoxLayout(dlg)
        outer.setContentsMargins(20, 20, 20, 20)

        logo_label = QLabel()
        pixmap = _render_svg(svg_path, 300, 240)
        if pixmap:
            logo_label.setPixmap(pixmap)
        logo_label.setFixedSize(300, 240)
        outer.addWidget(logo_label, alignment=Qt.AlignmentFlag.AlignTop)

        right = QVBoxLayout()
        right.setSpacing(6)

        info = QLabel(
            f"<h2 style='margin:0'>stemma</h2>"
            f"<p>Version {__version__}</p>"
            f"<p>A music player with AI stem separation.</p>"
            f'<p><a href="https://github.com/cyanidesayonara/stemma">'
            f"github.com/cyanidesayonara/stemma</a></p>"
            f"<p>MIT License</p>"
        )
        info.setOpenExternalLinks(True)
        info.setWordWrap(True)
        right.addWidget(info)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dlg.accept)
        right.addWidget(buttons, alignment=Qt.AlignmentFlag.AlignRight)

        outer.addLayout(right)
        dlg.exec()

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

        if self._export_worker is not None and self._export_worker.isRunning():
            self._export_worker.wait(5000)

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
            self.setWindowTitle(f"{song.artist} \u2014 {song.title} \u2014 stemma")

    def _on_close_song(self) -> None:
        """Stop playback and return to the empty logo state."""
        self._player.stop()
        self._player_controls.clear_song()
        self._current_song_id = None
        self.setWindowTitle("stemma")

    def _on_import(self) -> None:
        """Open the import dialog."""
        self._open_import_dialog()

    def _open_import_dialog(self, file_path: str = "") -> None:
        """Open the import dialog, optionally pre-filled with a file path."""
        dialog = ImportDialog(
            library=self._library,
            model_manager=self._model_manager,
            parent=self,
            file_path=file_path,
        )
        if dialog.exec():
            self._library_panel.refresh()

    # ------------------------------------------------------------------
    # Drag-and-drop import
    # ------------------------------------------------------------------

    def _is_audio_drop(self, event) -> bool:
        """Return True if the drag event contains at least one audio file."""
        mime = event.mimeData()
        return mime.hasUrls() and any(
            _is_audio_path(url.toLocalFile()) for url in mime.urls()
        )

    def dragEnterEvent(self, event) -> None:
        """Accept drag if any dropped URL is an audio file."""
        if self._is_audio_drop(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        """Keep the drop cursor active while dragging over the window."""
        if self._is_audio_drop(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        """Open ImportDialog for each dropped audio file."""
        mime = event.mimeData()
        if not mime.hasUrls():
            return
        event.acceptProposedAction()
        for url in mime.urls():
            path = url.toLocalFile()
            if _is_audio_path(path):
                self._open_import_dialog(file_path=path)

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

            self._export_worker.start()

    def _on_export_finished(self, path: str) -> None:
        QMessageBox.information(self, "Export", f"Mix successfully exported to {path}")

    def _on_export_error(self, err: str) -> None:
        QMessageBox.critical(self, "Export Error", f"Failed to export mix:\n{err}")

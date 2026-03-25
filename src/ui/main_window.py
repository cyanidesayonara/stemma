"""Main application window layout.

Left panel: song library list.
Center: player controls and stem mixer.
Menu bar: File / Edit / Help; theme toggle in the menu bar corner.
"""

import json
import os

from PySide6.QtCore import QPointF, QSettings, QSize, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QIcon,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
)
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
    QWidget,
)

from src.app_settings import (
    normalize_output_device_setting,
    output_device_indices_with_output,
    read_default_export_format,
    read_default_mp3_bitrate,
)
from src.data_paths import consume_data_dir_reset_notice
from src.exporter import ExportWorker, StemExporter
from src.library import SongLibrary
from src.model_manager import ModelManager
from src.player import MultiTrackPlayer
from src.ui.import_dialog import ImportDialog
from src.ui.preferences_dialog import PreferencesDialog
from src.ui.library_panel import LibraryPanel
from src.ui.player_controls import PlayerControls, _ROOT_DIR, _logo_variant, _render_svg
from src.ui.styles import get_colors, get_stylesheet
from src.version import __version__

# Try loading all components in this preferred visual layout order
ALL_STEM_NAMES = ("vocals", "drums", "bass", "other", "guitar", "piano")
_AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".flac"})


def _is_audio_path(path: str) -> bool:
    """Return True if *path* has an audio file extension."""
    _, ext = os.path.splitext(path)
    return ext.lower() in _AUDIO_EXTENSIONS


_THEME_TOGGLE_ICON_PX = 18


def _moon_icon(color: QColor) -> QIcon:
    """Crisp crescent moon for the theme button (avoids weak font glyphs on Windows)."""
    s = _THEME_TOGGLE_ICON_PX
    pix = QPixmap(s, s)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    cx = s * 0.5
    cy = s * 0.5
    r_outer = s * 0.42
    p.drawEllipse(QPointF(cx, cy), r_outer, r_outer)
    p.setCompositionMode(
        QPainter.CompositionMode.CompositionMode_DestinationOut
    )
    p.drawEllipse(QPointF(cx + s * 0.20, cy), r_outer * 0.90, r_outer * 0.90)
    p.end()
    return QIcon(pix)


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

        QTimer.singleShot(0, self._maybe_show_data_dir_reset_notice)
        QTimer.singleShot(0, self._maybe_warn_no_audio_output)
        QTimer.singleShot(0, self._restore_session)

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _maybe_show_data_dir_reset_notice(self) -> None:
        """Tell the user if startup fell back from an invalid custom data path."""
        msg = consume_data_dir_reset_notice(self._settings)
        if msg:
            QMessageBox.information(self, "Data folder", msg)

    def _maybe_warn_no_audio_output(self) -> None:
        """Warn once at startup if PortAudio reports no output devices."""
        valid = output_device_indices_with_output()
        if valid is not None and len(valid) == 0:
            QMessageBox.warning(
                self,
                "Audio output",
                "No audio output devices were found. Playback will not work "
                "until you connect speakers or headphones, or install an audio "
                "driver.\n\n"
                "You can still import and separate songs.",
            )

    def _on_playback_failed(self, message: str) -> None:
        """Show a dialog when the player cannot open an output stream."""
        QMessageBox.warning(self, "Playback", message)

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

        edit_menu = menu_bar.addMenu("&Edit")
        prefs_action = edit_menu.addAction("&Preferences...")
        prefs_action.triggered.connect(self._on_preferences)

        help_menu = menu_bar.addMenu("&Help")

        self._theme_btn = QPushButton()
        self._theme_btn.setObjectName("theme-toggle")
        self._theme_btn.setAccessibleName("Toggle theme")
        self._theme_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._theme_btn.setIconSize(
            QSize(_THEME_TOGGLE_ICON_PX, _THEME_TOGGLE_ICON_PX)
        )
        self._theme_btn.setFixedSize(30, 26)
        self._theme_btn.clicked.connect(self._toggle_theme)

        self._theme_corner = QWidget(menu_bar)
        corner_l = QHBoxLayout(self._theme_corner)
        corner_l.setContentsMargins(0, 4, 10, 0)
        corner_l.setSpacing(0)
        corner_l.addWidget(self._theme_btn)
        self._update_theme_btn()
        menu_bar.setCornerWidget(self._theme_corner)

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

        QShortcut(QKeySequence(Qt.Key.Key_M), self).activated.connect(
            self._player_controls.toggle_metronome
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

    def _on_preferences(self) -> None:
        """Open preferences; apply audio device without restart."""
        dlg = PreferencesDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._player.set_output_device(
                normalize_output_device_setting(self._settings)
            )

    def apply_theme(self, theme: str, colors: dict[str, str]) -> None:
        """Apply a theme to all child widgets that need explicit updates."""
        self._player_controls.apply_theme(theme, colors)

    def _update_theme_btn(self) -> None:
        """Update the corner toggle (sun glyph in dark mode, drawn moon in light)."""
        if self._theme == "dark":
            self._theme_btn.setIcon(QIcon())
            self._theme_btn.setText("\u2600")
            self._theme_btn.setToolTip("Switch to light theme")
        else:
            self._theme_btn.setText("")
            ink = QColor(get_colors(self._theme)["text"])
            self._theme_btn.setIcon(_moon_icon(ink))
            self._theme_btn.setToolTip("Switch to dark theme")

    def _toggle_theme(self) -> None:
        """Switch between light and dark themes."""
        self._theme = "light" if self._theme == "dark" else "dark"
        self._settings.setValue("theme", self._theme)

        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(get_stylesheet(self._theme))

        colors = get_colors(self._theme)
        self.apply_theme(self._theme, colors)
        self._update_theme_btn()

    def _on_about(self) -> None:
        """Show the About dialog with the main logo rendered from SVG."""
        dlg = QDialog(self)
        dlg.setWindowTitle("About stemma")
        dlg.setFixedSize(540, 280)

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

    def _save_session(self) -> None:
        """Persist current player state so it can be restored on next launch."""
        self._settings.setValue(
            "session/song_id", self._current_song_id or ""
        )
        self._settings.setValue(
            "session/position", self._player.current_seconds
        )
        self._settings.setValue(
            "session/muted_stems",
            json.dumps(sorted(self._player.muted_stems)),
        )
        self._settings.setValue(
            "session/soloed_stems",
            json.dumps(sorted(self._player.soloed_stems)),
        )
        self._settings.setValue(
            "session/volumes", json.dumps(self._player.volumes)
        )
        loop_a = self._player.loop_a
        loop_b = self._player.loop_b
        self._settings.setValue("session/loop_a", loop_a if loop_a is not None else -1)
        self._settings.setValue("session/loop_b", loop_b if loop_b is not None else -1)
        self._settings.setValue("session/looping", self._player.looping)
        self._settings.setValue("session/speed", self._player.speed)
        self._settings.setValue(
            "session/metronome_bpm", self._player.metronome_bpm
        )
        self._settings.setValue(
            "session/metronome_enabled", self._player.metronome_enabled
        )
        self._settings.setValue(
            "session/metronome_volume", self._player.metronome_volume
        )

    def _restore_session(self) -> None:
        """Reload the last song and player state from QSettings."""
        song_id = self._settings.value("session/song_id", "")
        if not song_id:
            return

        song = self._library.get_song(song_id)
        if song is None:
            return

        if not self._library_panel.select_song(song_id):
            return

        # Stem mute/solo/volume
        try:
            muted = set(json.loads(self._settings.value("session/muted_stems", "[]")))
        except (json.JSONDecodeError, TypeError):
            muted = set()
        try:
            soloed = set(json.loads(self._settings.value("session/soloed_stems", "[]")))
        except (json.JSONDecodeError, TypeError):
            soloed = set()
        try:
            volumes = json.loads(self._settings.value("session/volumes", "{}"))
        except (json.JSONDecodeError, TypeError):
            volumes = {}

        self._player_controls.restore_stem_state(muted, soloed, volumes)

        # Loop points
        try:
            loop_a = float(self._settings.value("session/loop_a", -1))
        except (TypeError, ValueError):
            loop_a = -1
        try:
            loop_b = float(self._settings.value("session/loop_b", -1))
        except (TypeError, ValueError):
            loop_b = -1
        looping = self._settings.value("session/looping", False)
        if isinstance(looping, str):
            looping = looping.lower() == "true"

        self._player_controls.restore_loop_state(
            loop_a if loop_a >= 0 else None,
            loop_b if loop_b >= 0 else None,
            bool(looping),
        )

        # Speed (async — seek after stretch completes)
        try:
            speed = float(self._settings.value("session/speed", 1.0))
        except (TypeError, ValueError):
            speed = 1.0

        try:
            position = float(self._settings.value("session/position", 0.0))
        except (TypeError, ValueError):
            position = 0.0

        if speed != 1.0:
            def _after_speed_applied(_s: float, pos=position) -> None:
                self._player.speed_changed.disconnect(_after_speed_applied)
                self._player.seek(pos)

            self._player.speed_changed.connect(_after_speed_applied)
            self._player_controls._speed_combo.blockSignals(True)
            label = f"{speed}x"
            idx = self._player_controls._speed_combo.findText(label)
            if idx >= 0:
                self._player_controls._speed_combo.setCurrentIndex(idx)
            self._player_controls._speed_combo.blockSignals(False)
            self._player_controls._speed_status.setText("Stretching...")
            self._player.set_speed(speed)
        else:
            self._player.seek(position)

        # Metronome state
        try:
            met_bpm = int(
                float(self._settings.value("session/metronome_bpm", 120))
            )
        except (TypeError, ValueError):
            met_bpm = 120
        met_enabled = self._settings.value("session/metronome_enabled", False)
        if isinstance(met_enabled, str):
            met_enabled = met_enabled.lower() == "true"
        try:
            met_vol = float(
                self._settings.value("session/metronome_volume", 0.5)
            )
        except (TypeError, ValueError):
            met_vol = 0.5
        self._player_controls.restore_metronome_state(
            met_bpm, bool(met_enabled), met_vol
        )

    def closeEvent(self, event) -> None:
        """Save window geometry/state, session, and clean up background threads."""
        try:
            self._save_session()
        except Exception:
            pass  # Never prevent the window from closing.
        self._settings.setValue("window/geometry", self.saveGeometry())
        self._settings.setValue("window/state", self.saveState())

        if self._export_worker is not None and self._export_worker.isRunning():
            self._export_worker.wait(5000)

        self._player.stop()

        super().closeEvent(event)

    def _connect_signals(self) -> None:
        """Wire up signals between panels."""
        self._library_panel.song_selected.connect(self._on_song_selected)
        self._player.playback_failed.connect(self._on_playback_failed)

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
        """Export the current mix as WAV or MP3."""
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

        fmt = read_default_export_format(self._settings)
        if fmt == "mp3":
            filters = "MP3 Files (*.mp3);;WAV Files (*.wav)"
        else:
            filters = "WAV Files (*.wav);;MP3 Files (*.mp3)"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Mix", "", filters
        )
        if path:
            exporter = StemExporter(stem_paths)
            volumes = {
                name: self._player.get_volume(name)
                for name in stem_paths
            }
            bitrate = read_default_mp3_bitrate(self._settings)
            self._export_worker = ExportWorker(
                exporter=exporter,
                output_path=path,
                muted_stems=self._player.muted_stems,
                volumes=volumes,
                mp3_bitrate=bitrate,
            )
            self._export_worker.finished.connect(self._on_export_finished)
            self._export_worker.error.connect(self._on_export_error)

            self._export_worker.start()

    def _on_export_finished(self, path: str) -> None:
        QMessageBox.information(self, "Export", f"Mix successfully exported to {path}")

    def _on_export_error(self, err: str) -> None:
        QMessageBox.critical(self, "Export Error", f"Failed to export mix:\n{err}")

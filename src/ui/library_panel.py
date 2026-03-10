"""Song library panel -- left sidebar listing imported songs.

Displays the song library as a list and emits a signal when a song is
selected. Full implementation in ticket #10.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.library import Song, SongLibrary


class EditSongDialog(QDialog):
    """Small dialog for editing a song's title and artist."""

    def __init__(self, song: Song, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Song")
        self.setMinimumWidth(400)
        self._song = song

        layout = QVBoxLayout(self)

        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("Title:"))
        self._title_edit = QLineEdit(song.title)
        title_row.addWidget(self._title_edit)
        layout.addLayout(title_row)

        artist_row = QHBoxLayout()
        artist_row.addWidget(QLabel("Artist:"))
        self._artist_edit = QLineEdit(song.artist)
        artist_row.addWidget(self._artist_edit)
        layout.addLayout(artist_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def title(self) -> str:
        """Return the edited title, falling back to the original."""
        return self._title_edit.text().strip() or self._song.title

    @property
    def artist(self) -> str:
        """Return the edited artist, falling back to the original."""
        return self._artist_edit.text().strip() or self._song.artist


class LibraryPanel(QWidget):
    """Left sidebar showing the song library.

    Signals:
        song_selected(str): Emitted with the song ID when a song is clicked.
    """

    song_selected = Signal(str)

    def __init__(self, library: SongLibrary, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._library = library

        self._setup_ui()
        self.refresh()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QLabel("Library")
        header.setObjectName("title-label")
        layout.addWidget(header)

        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search songs...")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self._search_edit)

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_item_changed)
        self._list.itemDoubleClicked.connect(lambda _: self._on_edit_song())
        self._list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self._list)

        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.setEnabled(False)
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        layout.addWidget(self._remove_btn)

    def refresh(self) -> None:
        """Reload the song list from the library."""
        self._list.clear()
        for song in self._library.songs:
            item = QListWidgetItem(f"{song.artist} - {song.title}")
            item.setData(Qt.ItemDataRole.UserRole, song.id)
            self._list.addItem(item)
        self._apply_filter(self._search_edit.text())

    def _apply_filter(self, query: str) -> None:
        """Show only items matching *query* (case-insensitive substring)."""
        query = query.lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(query not in item.text().lower())
        # Disable Remove if the current selection is hidden or gone.
        current = self._list.currentItem()
        if current is None or current.isHidden():
            self._remove_btn.setEnabled(False)

    def _on_item_changed(self, current: QListWidgetItem | None, _previous) -> None:
        if current is not None:
            song_id = current.data(Qt.ItemDataRole.UserRole)
            if song_id:
                self.song_selected.emit(song_id)
                self._remove_btn.setEnabled(True)
        else:
            self._remove_btn.setEnabled(False)

    def _on_context_menu(self, pos) -> None:
        """Show a right-click context menu on the song list."""
        item = self._list.itemAt(pos)
        if item is None:
            return
        # Select the right-clicked item so _on_edit_song targets it,
        # not whatever was previously selected.
        self._list.setCurrentItem(item)
        menu = QMenu(self)
        menu.addAction("Edit...").triggered.connect(self._on_edit_song)
        menu.exec(self._list.mapToGlobal(pos))

    def _on_edit_song(self) -> None:
        """Open the edit dialog for the currently selected song."""
        current = self._list.currentItem()
        if current is None:
            return
        song_id = current.data(Qt.ItemDataRole.UserRole)
        if not song_id:
            return
        song = self._library.get_song(song_id)
        if song is None:
            return
        dlg = EditSongDialog(song, parent=self)
        if dlg.exec():
            self._library.update_song(song_id, title=dlg.title, artist=dlg.artist)
            self.refresh()

    def _on_remove_clicked(self) -> None:
        current = self._list.currentItem()
        if current is not None:
            song_id = current.data(Qt.ItemDataRole.UserRole)
            if song_id:
                reply = QMessageBox.question(
                    self,
                    "Remove Song",
                    f"Are you sure you want to remove '{current.text()}' and delete its separated audio files?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._library.remove_song(song_id)
                    self.refresh()

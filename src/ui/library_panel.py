"""Song library panel -- left sidebar listing imported songs.

Displays the song library as a list and emits a signal when a song is
selected. Full implementation in ticket #10.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.library import SongLibrary


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
            item.setData(256, song.id)  # Qt.UserRole = 256
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
            song_id = current.data(256)
            if song_id:
                self.song_selected.emit(song_id)
                self._remove_btn.setEnabled(True)
        else:
            self._remove_btn.setEnabled(False)

    def _on_remove_clicked(self) -> None:
        current = self._list.currentItem()
        if current is not None:
            song_id = current.data(256)
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

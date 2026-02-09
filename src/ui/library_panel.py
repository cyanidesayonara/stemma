"""Song library panel -- left sidebar listing imported songs.

Displays the song library as a list and emits a signal when a song is
selected. Full implementation in ticket #10.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
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

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list)

    def refresh(self) -> None:
        """Reload the song list from the library."""
        self._list.clear()
        for song in self._library.songs:
            item = QListWidgetItem(f"{song.artist} - {song.title}")
            item.setData(256, song.id)  # Qt.UserRole = 256
            self._list.addItem(item)

    def _on_item_changed(self, current: QListWidgetItem | None, _previous) -> None:
        if current is not None:
            song_id = current.data(256)
            if song_id:
                self.song_selected.emit(song_id)

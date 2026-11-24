"""Play queue panel widget shown inside the sidebar stack."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from src.library import SongLibrary
from src.play_queue import PlayQueue
from src.ui.styles import DARK_COLORS, ON_ACCENT

_ARTIST_ROLE = Qt.ItemDataRole.UserRole + 1
_TITLE_ROLE = Qt.ItemDataRole.UserRole + 2


class _QueueItemDelegate(QStyledItemDelegate):
    """Two-line row delegate matching the library song list styling."""

    _V_PADDING = 4

    def __init__(
        self,
        parent=None,
        separator_color: str = DARK_COLORS["surface1"],
        accent_color: str = DARK_COLORS["accent"],
        selected_text_color: str = ON_ACCENT,
        subtext_color: str = DARK_COLORS["subtext"],
    ) -> None:
        super().__init__(parent)
        self._separator_color = QColor(separator_color)
        self._accent_color = QColor(accent_color)
        self._selected_text_color = QColor(selected_text_color)
        self._subtext_color = QColor(subtext_color)

    def apply_colors(self, colors: dict[str, str]) -> None:
        self._separator_color = QColor(colors["surface1"])
        self._accent_color = QColor(colors["accent"])
        self._selected_text_color = QColor(colors["on_accent"])
        self._subtext_color = QColor(colors["subtext"])

    def paint(self, painter: QPainter, option, index) -> None:  # noqa: D401
        self.initStyleOption(option, index)
        painter.save()

        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        if selected:
            painter.fillRect(option.rect, self._accent_color)
            text_color = QColor(self._selected_text_color)
            sub_color = QColor(text_color)
        else:
            text_color = option.palette.text().color()
            sub_color = QColor(self._subtext_color)

        artist = index.data(_ARTIST_ROLE) or ""
        title = index.data(_TITLE_ROLE) or ""
        rect = option.rect.adjusted(6, self._V_PADDING, -6, -self._V_PADDING)

        bold_font = QFont(option.font)
        bold_font.setBold(True)
        top_rect = rect.adjusted(0, 0, 0, -rect.height() // 2)
        bottom_rect = rect.adjusted(0, rect.height() // 2, 0, 0)
        painter.setFont(bold_font)
        painter.setPen(text_color)
        painter.drawText(
            top_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            painter.fontMetrics().elidedText(
                title or artist, Qt.TextElideMode.ElideRight, top_rect.width(),
            ),
        )

        normal_font = QFont(option.font)
        normal_font.setPointSizeF(option.font.pointSizeF() * 0.9)
        painter.setFont(normal_font)
        metrics = painter.fontMetrics()
        painter.setPen(sub_color)
        painter.drawText(
            bottom_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            metrics.elidedText(
                artist if title else "", Qt.TextElideMode.ElideRight,
                max(0, bottom_rect.width()),
            ),
        )

        painter.setPen(QPen(self._separator_color, 1))
        y = option.rect.bottom()
        painter.drawLine(option.rect.left() + 6, y, option.rect.right() - 6, y)
        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        fm = QFontMetrics(option.font)
        return QSize(0, fm.height() * 2 + self._V_PADDING * 2 + 4)


def _song_display(song_id: str, library: SongLibrary) -> tuple[str, str]:
    """Return a clean (title, artist) pair for *song_id*."""
    song = library.get_song(song_id)
    if song is not None:
        title = getattr(song, "title", None)
        artist = getattr(song, "artist", None)
        if isinstance(title, str):
            return (title or song_id, artist if isinstance(artist, str) else "")
    return (song_id, "")


def _make_song_item(song_id: str, library: SongLibrary) -> QListWidgetItem:
    title, artist = _song_display(song_id, library)
    display = f"{artist} - {title}" if artist else title
    item = QListWidgetItem(display)
    item.setData(Qt.ItemDataRole.UserRole, song_id)
    item.setData(_TITLE_ROLE, title)
    item.setData(_ARTIST_ROLE, artist)
    return item


class PlayQueueWidget(QWidget):
    """Queue manager showing Up Next items and remaining setlist tracks."""

    play_queued_index_requested = Signal(int)
    move_item_requested = Signal(int, int)
    remove_item_requested = Signal(int)
    clear_queue_requested = Signal()
    stop_setlist_requested = Signal()
    play_context_song_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._repopulating = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Active setlist banner (hidden in library mode).
        self._context_bar = QWidget()
        ctx_row = QHBoxLayout(self._context_bar)
        ctx_row.setContentsMargins(0, 0, 0, 0)
        ctx_row.setSpacing(6)
        self._context_label = QLabel()
        self._context_label.setObjectName("subtle-label")
        self._context_label.setMinimumWidth(0)
        ctx_row.addWidget(self._context_label, 1)

        self._stop_setlist_btn = QPushButton("Stop")
        self._stop_setlist_btn.setToolTip("Stop playing setlist")
        self._stop_setlist_btn.setAccessibleName("Stop active setlist")
        self._stop_setlist_btn.clicked.connect(self.stop_setlist_requested)
        ctx_row.addWidget(self._stop_setlist_btn)
        self._context_bar.setVisible(False)
        layout.addWidget(self._context_bar)

        # Up Next header row.
        up_header = QHBoxLayout()
        up_header.setContentsMargins(0, 0, 0, 0)
        up_title = QLabel("Up Next")
        up_title.setObjectName("subtle-label")
        up_header.addWidget(up_title)
        up_header.addStretch()

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setToolTip("Clear queued songs")
        self._clear_btn.setAccessibleName("Clear play queue")
        self._clear_btn.setEnabled(False)
        self._clear_btn.clicked.connect(self.clear_queue_requested)
        up_header.addWidget(self._clear_btn)
        layout.addLayout(up_header)

        # Up Next list (drag-reorderable).
        self._up_next_list = QListWidget()
        self._queue_delegate = _QueueItemDelegate(self._up_next_list)
        self._up_next_list.setItemDelegate(self._queue_delegate)
        self._up_next_list.setAccessibleName("Play queue")
        self._up_next_list.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove
        )
        self._up_next_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._up_next_list.setDragEnabled(True)
        self._up_next_list.setAcceptDrops(True)
        self._up_next_list.setDropIndicatorShown(True)
        self._up_next_list.model().rowsMoved.connect(self._on_rows_moved)
        self._up_next_list.currentRowChanged.connect(
            lambda _row: self._sync_buttons()
        )
        self._up_next_list.itemDoubleClicked.connect(
            self._on_up_next_double_clicked
        )
        self._up_next_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._up_next_list.customContextMenuRequested.connect(
            self._on_up_next_context_menu
        )
        layout.addWidget(self._up_next_list, 1)

        self._empty_label = QLabel(
            "Queue is empty.\nRight-click a song to play next or add to queue."
        )
        self._empty_label.setObjectName("subtle-label")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setWordWrap(True)
        layout.addWidget(self._empty_label)

        # Keyboard-friendly reorder and remove buttons.
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(4)

        self._up_btn = QPushButton("Up")
        self._up_btn.setAccessibleName("Move queued song up")
        self._up_btn.setEnabled(False)
        self._up_btn.clicked.connect(lambda: self._move_selected(-1))
        actions.addWidget(self._up_btn)

        self._down_btn = QPushButton("Down")
        self._down_btn.setAccessibleName("Move queued song down")
        self._down_btn.setEnabled(False)
        self._down_btn.clicked.connect(lambda: self._move_selected(1))
        actions.addWidget(self._down_btn)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.setAccessibleName("Remove from play queue")
        self._remove_btn.setEnabled(False)
        self._remove_btn.clicked.connect(self._remove_selected)
        actions.addWidget(self._remove_btn)
        layout.addLayout(actions)

        # Upcoming setlist section (hidden unless setlist has remaining tracks).
        self._context_section = QWidget()
        ctx_col = QVBoxLayout(self._context_section)
        ctx_col.setContentsMargins(0, 4, 0, 0)
        ctx_col.setSpacing(4)

        self._context_header = QLabel("Next from Setlist")
        self._context_header.setObjectName("subtle-label")
        ctx_col.addWidget(self._context_header)

        self._context_list = QListWidget()
        self._context_delegate = _QueueItemDelegate(self._context_list)
        self._context_list.setItemDelegate(self._context_delegate)
        self._context_list.setAccessibleName("Upcoming setlist songs")
        self._context_list.itemDoubleClicked.connect(
            self._on_context_double_clicked
        )
        ctx_col.addWidget(self._context_list, 1)
        self._context_section.setVisible(False)
        layout.addWidget(self._context_section, 1)

    def refresh(self, queue: PlayQueue, library: SongLibrary) -> None:
        """Synchronize the widget with *queue* and *library*."""
        prev_row = self._up_next_list.currentRow()
        up_next = queue.up_next

        self._repopulating = True
        try:
            self._up_next_list.clear()
            for song_id in up_next:
                self._up_next_list.addItem(_make_song_item(song_id, library))
        finally:
            self._repopulating = False

        if up_next and prev_row >= 0:
            self._up_next_list.setCurrentRow(min(prev_row, len(up_next) - 1))

        if queue.is_setlist_active:
            label = queue.context_name or "Setlist"
            self._context_label.setText(f"Setlist: {label}")
            self._context_label.setToolTip(f"Playing setlist: {label}")
            self._context_bar.setVisible(True)
        else:
            self._context_bar.setVisible(False)

        upcoming = queue.upcoming_context_songs()
        self._context_list.clear()
        for song_id in upcoming:
            self._context_list.addItem(_make_song_item(song_id, library))
        if upcoming:
            ctx_name = queue.context_name or "Setlist"
            self._context_header.setText(f"Next from {ctx_name}")
            self._context_section.setVisible(True)
        else:
            self._context_section.setVisible(False)

        self._up_next_list.setVisible(bool(up_next))
        self._empty_label.setVisible(not up_next and not upcoming)
        self._sync_buttons()

    def apply_theme(self, _theme: str, colors: dict[str, str]) -> None:
        """Update item delegates for the active theme."""
        self._queue_delegate.apply_colors(colors)
        self._context_delegate.apply_colors(colors)
        self._up_next_list.viewport().update()
        self._context_list.viewport().update()

    def _sync_buttons(self) -> None:
        count = self._up_next_list.count()
        row = self._up_next_list.currentRow()
        has_sel = 0 <= row < count
        self._clear_btn.setEnabled(count > 0)
        self._up_btn.setEnabled(has_sel and row > 0)
        self._down_btn.setEnabled(has_sel and row < count - 1)
        self._remove_btn.setEnabled(has_sel)

    def _move_selected(self, delta: int) -> None:
        row = self._up_next_list.currentRow()
        target = row + delta
        if 0 <= row < self._up_next_list.count() and (
            0 <= target < self._up_next_list.count()
        ):
            self.move_item_requested.emit(row, target)

    def _remove_selected(self) -> None:
        row = self._up_next_list.currentRow()
        if 0 <= row < self._up_next_list.count():
            self.remove_item_requested.emit(row)

    def _on_rows_moved(
        self, _parent, start: int, _end: int, _dest_parent, dest_row: int,
    ) -> None:
        if self._repopulating:
            return
        # Qt reports dest_row in pre-move coordinates; moving downward inserts
        # one slot past the final post-move index.
        target = dest_row - 1 if dest_row > start else dest_row
        if start != target:
            self.move_item_requested.emit(start, target)

    def _on_up_next_double_clicked(self, item: QListWidgetItem) -> None:
        row = self._up_next_list.row(item)
        if row >= 0:
            self.play_queued_index_requested.emit(row)

    def _on_context_double_clicked(self, item: QListWidgetItem) -> None:
        song_id = item.data(Qt.ItemDataRole.UserRole)
        if song_id:
            self.play_context_song_requested.emit(song_id)

    def _on_up_next_context_menu(self, pos) -> None:
        item = self._up_next_list.itemAt(pos)
        if item is None:
            return
        row = self._up_next_list.row(item)
        if row < 0:
            return
        self._up_next_list.setCurrentRow(row)
        count = self._up_next_list.count()

        menu = QMenu(self)
        menu.addAction("Play Now").triggered.connect(
            lambda: self.play_queued_index_requested.emit(row)
        )
        move_up = menu.addAction("Move Up")
        move_up.setEnabled(row > 0)
        move_up.triggered.connect(lambda: self.move_item_requested.emit(row, row - 1))
        move_down = menu.addAction("Move Down")
        move_down.setEnabled(row < count - 1)
        move_down.triggered.connect(
            lambda: self.move_item_requested.emit(row, row + 1)
        )
        menu.addSeparator()
        menu.addAction("Remove from Queue").triggered.connect(
            lambda: self.remove_item_requested.emit(row)
        )
        menu.exec(self._up_next_list.mapToGlobal(pos))

"""Song library panel -- left sidebar listing imported songs.

Displays the song library as a list and emits a signal when a song is
selected. Includes repeat/shuffle/prev/next playback controls and a
"now playing" indicator. Full implementation in ticket #10.
"""

import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
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
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from src.library import Song, SongLibrary
from src.ui.styles import DARK_COLORS

# Custom data roles for two-line display.
_ARTIST_ROLE = Qt.ItemDataRole.UserRole + 1
_TITLE_ROLE = Qt.ItemDataRole.UserRole + 2

_CTRL_ICON = 18  # Icon size for library control buttons.
_CTRL_BTN = 28   # Button size for library control buttons.


# ---------------------------------------------------------------------------
# QPainter icon draw helpers
# ---------------------------------------------------------------------------

def _make_icon(draw_fn, color: QColor, size: int = _CTRL_ICON) -> QIcon:
    """Create a QIcon by painting with *draw_fn(painter, size)*."""
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.GlobalColor.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    draw_fn(p, size)
    p.end()
    return QIcon(pixmap)


def _make_toggle_icon(draw_fn, normal_color: QColor,
                      size: int = _CTRL_ICON) -> QIcon:
    """Icon with distinct normal and checked pixmaps."""
    checked_color = QColor("#1e1e2e")
    icon = QIcon()
    for color, state in [
        (normal_color, QIcon.State.Off),
        (checked_color, QIcon.State.On),
    ]:
        pixmap = QPixmap(QSize(size, size))
        pixmap.fill(Qt.GlobalColor.transparent)
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        draw_fn(p, size)
        p.end()
        icon.addPixmap(pixmap, QIcon.Mode.Normal, state)
    return icon


def _draw_repeat(p: QPainter, s: int) -> None:
    """Cycle/repeat arrows icon (repeat-all)."""
    cx = s / 2.0
    m = s * 0.20
    r = (s - 2 * m) / 2.0
    pen = QPen(p.brush().color(), s * 0.09)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    arc_rect = QRectF(m, m, s - 2 * m, s - 2 * m)
    p.drawArc(arc_rect, 20 * 16, 140 * 16)
    p.drawArc(arc_rect, 200 * 16, 140 * 16)
    p.setPen(Qt.PenStyle.NoPen)
    a1 = math.radians(20)
    ax1 = cx + r * math.cos(a1)
    ay1 = cx - r * math.sin(a1)
    ah = s * 0.12
    p.drawPolygon(QPolygonF([
        QPointF(ax1 + ah, ay1 - ah * 0.6),
        QPointF(ax1 - ah * 0.3, ay1 - ah * 0.8),
        QPointF(ax1, ay1 + ah * 0.5),
    ]))
    a2 = math.radians(200)
    ax2 = cx + r * math.cos(a2)
    ay2 = cx - r * math.sin(a2)
    p.drawPolygon(QPolygonF([
        QPointF(ax2 - ah, ay2 + ah * 0.6),
        QPointF(ax2 + ah * 0.3, ay2 + ah * 0.8),
        QPointF(ax2, ay2 - ah * 0.5),
    ]))


def _draw_repeat_one(p: QPainter, s: int) -> None:
    """Repeat-one icon: repeat arrows with a '1' in the center."""
    _draw_repeat(p, s)
    # Draw "1" in the center.
    font = p.font()
    font.setPixelSize(max(6, int(s * 0.38)))
    font.setBold(True)
    p.setFont(font)
    p.setPen(p.brush().color())
    p.drawText(QRectF(0, 0, s, s), Qt.AlignmentFlag.AlignCenter, "1")
    p.setPen(Qt.PenStyle.NoPen)


def _draw_shuffle(p: QPainter, s: int) -> None:
    """Two crossed arrows — shuffle icon."""
    m = s * 0.18
    pen = QPen(p.brush().color(), s * 0.09)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    # Two crossing lines.
    p.drawLine(QPointF(m, s * 0.35), QPointF(s - m, s * 0.65))
    p.drawLine(QPointF(m, s * 0.65), QPointF(s - m, s * 0.35))
    p.setPen(Qt.PenStyle.NoPen)
    # Arrowhead at top-right.
    ah = s * 0.12
    tip_x = s - m
    tip_y = s * 0.35
    p.drawPolygon(QPolygonF([
        QPointF(tip_x + ah * 0.3, tip_y - ah * 0.3),
        QPointF(tip_x - ah, tip_y - ah * 0.3),
        QPointF(tip_x, tip_y + ah * 0.7),
    ]))
    # Arrowhead at bottom-right.
    tip_y2 = s * 0.65
    p.drawPolygon(QPolygonF([
        QPointF(tip_x + ah * 0.3, tip_y2 + ah * 0.3),
        QPointF(tip_x - ah, tip_y2 + ah * 0.3),
        QPointF(tip_x, tip_y2 - ah * 0.7),
    ]))


def _draw_prev(p: QPainter, s: int) -> None:
    """Previous track icon: bar + left-pointing triangle."""
    m = s * 0.22
    bar_w = s * 0.1
    # Vertical bar on left.
    p.drawRect(QRectF(m, m, bar_w, s - 2 * m))
    # Left-pointing triangle.
    p.drawPolygon(QPolygonF([
        QPointF(s - m, m),
        QPointF(m + bar_w + 2, s / 2.0),
        QPointF(s - m, s - m),
    ]))


def _draw_next(p: QPainter, s: int) -> None:
    """Next track icon: right-pointing triangle + bar."""
    m = s * 0.22
    bar_w = s * 0.1
    # Right-pointing triangle.
    p.drawPolygon(QPolygonF([
        QPointF(m, m),
        QPointF(s - m - bar_w - 2, s / 2.0),
        QPointF(m, s - m),
    ]))
    # Vertical bar on right.
    p.drawRect(QRectF(s - m - bar_w, m, bar_w, s - 2 * m))


# ---------------------------------------------------------------------------
# Song delegate
# ---------------------------------------------------------------------------

class _SongDelegate(QStyledItemDelegate):
    """Two-line delegate: artist (bold) on top, title (subdued) below."""

    _V_PADDING = 4
    _PLAYING_BAR_WIDTH = 3

    def __init__(
        self, parent=None, separator_color: str = DARK_COLORS["surface1"],
        accent_color: str = DARK_COLORS["accent"],
    ):
        super().__init__(parent)
        self._separator_color = QColor(separator_color)
        self._accent_color = QColor(accent_color)
        self._playing_song_id: str | None = None

    def set_separator_color(self, color: str) -> None:
        self._separator_color = QColor(color)

    def set_accent_color(self, color: str) -> None:
        self._accent_color = QColor(color)

    def set_playing_song(self, song_id: str | None) -> None:
        """Set the currently playing song ID for the 'now playing' indicator."""
        self._playing_song_id = song_id

    def paint(self, painter, option, index):  # noqa: D401
        self.initStyleOption(option, index)
        painter.save()

        song_id = index.data(Qt.ItemDataRole.UserRole)
        is_playing = song_id and song_id == self._playing_song_id

        # Draw selection / hover background.
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        if selected:
            painter.fillRect(option.rect, self._accent_color)
            text_color = QColor("#1e1e2e")  # Dark text on accent.
            sub_color = QColor(text_color)
            sub_color.setAlphaF(0.7)
        else:
            text_color = option.palette.text().color()
            # Subdued color for title line.
            sub_color = QColor(text_color)
            sub_color.setAlphaF(0.6)

        # "Now playing" indicator — accent bar on the left edge.
        left_inset = 6
        if is_playing and not selected:
            bar_rect = QRectF(
                option.rect.left(),
                option.rect.top() + self._V_PADDING,
                self._PLAYING_BAR_WIDTH,
                option.rect.height() - 2 * self._V_PADDING,
            )
            painter.fillRect(bar_rect, self._accent_color)
            left_inset = self._PLAYING_BAR_WIDTH + 6

        artist = index.data(_ARTIST_ROLE) or ""
        title = index.data(_TITLE_ROLE) or ""
        rect = option.rect.adjusted(left_inset, self._V_PADDING, -4, -self._V_PADDING)

        # Artist line (bold).
        bold_font = QFont(option.font)
        bold_font.setBold(True)
        painter.setFont(bold_font)
        painter.setPen(text_color)
        artist_rect = rect.adjusted(0, 0, 0, -rect.height() // 2)
        painter.drawText(
            artist_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            artist,
        )

        # Title line (normal, slightly smaller, subdued).
        normal_font = QFont(option.font)
        normal_font.setPointSizeF(option.font.pointSizeF() * 0.9)
        painter.setFont(normal_font)
        painter.setPen(sub_color)
        title_rect = rect.adjusted(0, rect.height() // 2, 0, 0)
        painter.drawText(
            title_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            title,
        )

        # Separator line at the bottom of each item.
        painter.setPen(QPen(self._separator_color, 1))
        y = option.rect.bottom()
        painter.drawLine(option.rect.left() + 6, y, option.rect.right() - 6, y)

        painter.restore()

    def sizeHint(self, option, index):
        fm = QFontMetrics(option.font)
        return QSize(0, fm.height() * 2 + self._V_PADDING * 2 + 4)


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


# ---------------------------------------------------------------------------
# Repeat modes
# ---------------------------------------------------------------------------

REPEAT_OFF = "off"
REPEAT_ALL = "all"
REPEAT_ONE = "one"
_REPEAT_CYCLE = [REPEAT_OFF, REPEAT_ALL, REPEAT_ONE]


class LibraryPanel(QWidget):
    """Left sidebar showing the song library.

    Signals:
        song_selected(str): Emitted with the song ID when a song is clicked.
        repeat_mode_changed(str): Emitted when repeat mode cycles.
        shuffle_toggled(bool): Emitted when shuffle is toggled.
        previous_requested(): Emitted when the user clicks Previous.
        next_requested(): Emitted when the user clicks Next.
    """

    song_selected = Signal(str)
    repeat_mode_changed = Signal(str)
    shuffle_toggled = Signal(bool)
    previous_requested = Signal()
    next_requested = Signal()

    def __init__(self, library: SongLibrary, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._library = library
        self._repeat_mode: str = REPEAT_OFF
        self._shuffle_enabled: bool = False

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
        self._song_delegate = _SongDelegate(self._list)
        self._list.setItemDelegate(self._song_delegate)
        self._list.setAccessibleName("Song library")
        self._list.currentItemChanged.connect(self._on_item_changed)
        self._list.itemDoubleClicked.connect(lambda _: self._on_edit_song())
        self._list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self._list)

        # -- Playback control bar --
        ctrl_bar = QHBoxLayout()
        ctrl_bar.setSpacing(4)

        icon_color = QColor(DARK_COLORS["text"])

        # Repeat (cycles: off → all → one)
        self._repeat_btn = QPushButton()
        self._repeat_btn.setObjectName("icon-btn")
        self._repeat_btn.setFixedSize(_CTRL_BTN, _CTRL_BTN)
        self._repeat_btn.setToolTip("Repeat: off")
        self._repeat_btn.setAccessibleName("Repeat")
        self._repeat_btn.clicked.connect(self._on_repeat_clicked)
        self._repeat_icons = {
            REPEAT_OFF: _make_icon(_draw_repeat, icon_color),
            REPEAT_ALL: _make_icon(_draw_repeat, icon_color),
            REPEAT_ONE: _make_icon(_draw_repeat_one, icon_color),
        }
        self._repeat_btn.setIcon(self._repeat_icons[REPEAT_OFF])
        self._repeat_btn.setIconSize(QSize(_CTRL_ICON, _CTRL_ICON))
        ctrl_bar.addWidget(self._repeat_btn)

        # Shuffle (toggle)
        self._shuffle_btn = QPushButton()
        self._shuffle_btn.setObjectName("icon-btn")
        self._shuffle_btn.setCheckable(True)
        self._shuffle_btn.setFixedSize(_CTRL_BTN, _CTRL_BTN)
        self._shuffle_btn.setToolTip("Shuffle: off")
        self._shuffle_btn.setAccessibleName("Shuffle")
        self._shuffle_btn.setIcon(
            _make_toggle_icon(_draw_shuffle, icon_color)
        )
        self._shuffle_btn.setIconSize(QSize(_CTRL_ICON, _CTRL_ICON))
        self._shuffle_btn.toggled.connect(self._on_shuffle_toggled)
        ctrl_bar.addWidget(self._shuffle_btn)

        ctrl_bar.addStretch()

        # Previous
        self._prev_btn = QPushButton()
        self._prev_btn.setObjectName("icon-btn")
        self._prev_btn.setFixedSize(_CTRL_BTN, _CTRL_BTN)
        self._prev_btn.setToolTip("Previous song (P)")
        self._prev_btn.setAccessibleName("Previous song")
        self._prev_btn.setIcon(_make_icon(_draw_prev, icon_color))
        self._prev_btn.setIconSize(QSize(_CTRL_ICON, _CTRL_ICON))
        self._prev_btn.clicked.connect(self.previous_requested)
        ctrl_bar.addWidget(self._prev_btn)

        # Next
        self._next_btn = QPushButton()
        self._next_btn.setObjectName("icon-btn")
        self._next_btn.setFixedSize(_CTRL_BTN, _CTRL_BTN)
        self._next_btn.setToolTip("Next song (N)")
        self._next_btn.setAccessibleName("Next song")
        self._next_btn.setIcon(_make_icon(_draw_next, icon_color))
        self._next_btn.setIconSize(QSize(_CTRL_ICON, _CTRL_ICON))
        self._next_btn.clicked.connect(self.next_requested)
        ctrl_bar.addWidget(self._next_btn)

        layout.addLayout(ctrl_bar)

        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.setEnabled(False)
        self._remove_btn.setToolTip("Remove selected song from library")
        self._remove_btn.setAccessibleName("Remove selected song")
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        layout.addWidget(self._remove_btn)

    # ------------------------------------------------------------------
    # Repeat / Shuffle
    # ------------------------------------------------------------------

    @property
    def repeat_mode(self) -> str:
        return self._repeat_mode

    @property
    def shuffle_enabled(self) -> bool:
        return self._shuffle_enabled

    def set_repeat_mode(self, mode: str) -> None:
        """Set repeat mode without emitting the signal (for session restore)."""
        if mode not in _REPEAT_CYCLE:
            mode = REPEAT_OFF
        self._repeat_mode = mode
        self._update_repeat_ui()

    def set_shuffle_enabled(self, enabled: bool) -> None:
        """Set shuffle state without emitting the signal (for session restore)."""
        self._shuffle_enabled = enabled
        self._shuffle_btn.blockSignals(True)
        self._shuffle_btn.setChecked(enabled)
        self._shuffle_btn.blockSignals(False)
        self._update_shuffle_ui()

    def _on_repeat_clicked(self) -> None:
        idx = _REPEAT_CYCLE.index(self._repeat_mode)
        self._repeat_mode = _REPEAT_CYCLE[(idx + 1) % len(_REPEAT_CYCLE)]
        self._update_repeat_ui()
        self.repeat_mode_changed.emit(self._repeat_mode)

    def _on_shuffle_toggled(self, checked: bool) -> None:
        self._shuffle_enabled = checked
        self._update_shuffle_ui()
        self.shuffle_toggled.emit(checked)

    def _update_repeat_ui(self) -> None:
        self._repeat_btn.setIcon(self._repeat_icons[self._repeat_mode])
        labels = {REPEAT_OFF: "off", REPEAT_ALL: "all", REPEAT_ONE: "one"}
        tip = f"Repeat: {labels[self._repeat_mode]}"
        self._repeat_btn.setToolTip(tip)
        # Visual: when active (all/one), show as "checked" style.
        self._repeat_btn.setStyleSheet(
            "" if self._repeat_mode == REPEAT_OFF else ""
        )
        # Use the checked property to style active state via QSS.
        self._repeat_btn.setCheckable(True)
        self._repeat_btn.setChecked(self._repeat_mode != REPEAT_OFF)
        self._repeat_btn.setCheckable(False)  # Prevent toggle on next click.

    def _update_shuffle_ui(self) -> None:
        tip = f"Shuffle: {'on' if self._shuffle_enabled else 'off'}"
        self._shuffle_btn.setToolTip(tip)

    # ------------------------------------------------------------------
    # Now-playing indicator
    # ------------------------------------------------------------------

    def set_playing_song(self, song_id: str | None) -> None:
        """Mark a song as 'now playing' with a visual indicator."""
        self._song_delegate.set_playing_song(song_id)
        self._list.viewport().update()

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def apply_theme(self, theme: str, colors: dict[str, str]) -> None:
        """Update delegate colors and control icons for the current theme."""
        self._song_delegate.set_separator_color(colors["surface1"])
        self._song_delegate.set_accent_color(colors["accent"])

        icon_color = QColor(colors["text"])
        self._repeat_icons = {
            REPEAT_OFF: _make_icon(_draw_repeat, icon_color),
            REPEAT_ALL: _make_icon(_draw_repeat, icon_color),
            REPEAT_ONE: _make_icon(_draw_repeat_one, icon_color),
        }
        self._update_repeat_ui()
        self._shuffle_btn.setIcon(
            _make_toggle_icon(_draw_shuffle, icon_color)
        )
        self._prev_btn.setIcon(_make_icon(_draw_prev, icon_color))
        self._next_btn.setIcon(_make_icon(_draw_next, icon_color))

        self._list.viewport().update()

    # ------------------------------------------------------------------
    # Song list management
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Reload the song list from the library."""
        self._list.clear()
        for song in self._library.songs:
            # Display text used for search filtering; delegate paints the rows.
            item = QListWidgetItem(f"{song.artist} - {song.title}")
            item.setData(Qt.ItemDataRole.UserRole, song.id)
            item.setData(_ARTIST_ROLE, song.artist)
            item.setData(_TITLE_ROLE, song.title)
            self._list.addItem(item)
        self._apply_filter(self._search_edit.text())

    def song_count(self) -> int:
        """Return the number of songs in the library."""
        return len(self._library.songs)

    def song_ids_in_order(self) -> list[str]:
        """Return song IDs in current library display order."""
        return [s.id for s in self._library.songs]

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

    def select_song(self, song_id: str) -> bool:
        """Programmatically select a song by its ID.

        Returns True if the song was found and selected, False otherwise.
        """
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == song_id:
                self._list.setCurrentItem(item)
                return True
        return False

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

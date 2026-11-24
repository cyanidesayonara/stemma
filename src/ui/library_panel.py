"""Song library panel -- left sidebar listing imported songs.

Displays the song library as a list and emits a signal when a song is
selected. Includes repeat/shuffle/prev/next playback controls and a
"now playing" indicator. Full implementation in ticket #10.
"""

from collections.abc import Callable
import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionComboBox,
    QStylePainter,
    QVBoxLayout,
    QWidget,
)

from src.import_messages import describe_error
from src.library import (
    LIBRARY_READ_ONLY_MESSAGE,
    PRACTICE_STATUSES,
    Setlist,
    Song,
    SongLibrary,
    normalize_tags,
)
from src.play_queue import PlayQueue
from src.practice_state import PracticeState
from src.ui.control_primitives import (
    COMPACT_BUTTON,
    add_disabled_pixmaps,
    fix_button_size,
    repolish,
)
from src.ui.play_queue_widget import PlayQueueWidget
from src.ui.styles import DARK_COLORS, ON_ACCENT, scaled_px

# Custom data roles for two-line display.
_ARTIST_ROLE = Qt.ItemDataRole.UserRole + 1
_TITLE_ROLE = Qt.ItemDataRole.UserRole + 2
# Progress text while the song is still separating in the background
# (e.g. "Separating... 42%"); unset/empty for ready songs.
_PROGRESS_ROLE = Qt.ItemDataRole.UserRole + 3
# Precomputed lowercase search text covering artist, title, and tags.
_SEARCH_TEXT_ROLE = Qt.ItemDataRole.UserRole + 4
# Practice status label ("Learning") drawn as a badge; empty for none.
_STATUS_ROLE = Qt.ItemDataRole.UserRole + 5
# Compact practice state ("A-B 0.75x +2"); empty for none.
_PRACTICE_ROLE = Qt.ItemDataRole.UserRole + 6
# True for songs marked for a batch action (Ctrl or Shift click).
_MARKED_ROLE = Qt.ItemDataRole.UserRole + 7

_CTRL_ICON = 18  # Icon size for library control buttons.

_PRACTICE_STATUS_LABELS = {
    "": "None",
    "learning": "Learning",
    "practicing": "Practicing",
    "ready": "Ready",
}


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
    icon = QIcon(pixmap)
    add_disabled_pixmaps(icon, draw_fn, color, size)
    return icon


def _make_toggle_icon(draw_fn, normal_color: QColor,
                      checked_color: QColor | None = None,
                      size: int = _CTRL_ICON) -> QIcon:
    """Icon with distinct normal and checked pixmaps.

    *checked_color*: color used when the button is in checked state.
    Defaults to ON_ACCENT (near-black) for readability on the teal accent
    fill — works in both dark and light themes.
    """
    if checked_color is None:
        checked_color = QColor(ON_ACCENT)
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
    add_disabled_pixmaps(icon, draw_fn, normal_color, size)
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


def _draw_queue(p: QPainter, s: int) -> None:
    """Play queue icon: stacked list lines with a small play triangle."""
    m = s * 0.18
    line_h = max(1.5, s * 0.10)
    full_w = s - 2 * m
    short_w = full_w * 0.52
    p.drawRect(QRectF(m, s * 0.22, full_w, line_h))
    p.drawRect(QRectF(m, s * 0.45, full_w, line_h))
    p.drawRect(QRectF(m, s * 0.68, short_w, line_h))
    # Small play indicator at bottom-right.
    tri_left = m + short_w + s * 0.08
    tri_right = s - m
    tri_top = s * 0.56
    tri_bottom = s * 0.84
    p.drawPolygon(QPolygonF([
        QPointF(tri_left, tri_top),
        QPointF(tri_right, (tri_top + tri_bottom) / 2.0),
        QPointF(tri_left, tri_bottom),
    ]))


# ---------------------------------------------------------------------------
# Song delegate
# ---------------------------------------------------------------------------

class _SongDelegate(QStyledItemDelegate):
    """Two-line delegate: title (bold) on top, artist (subdued) below."""

    _V_PADDING = 4
    _PLAYING_BAR_WIDTH = 3

    def __init__(
        self, parent=None, separator_color: str = DARK_COLORS["surface1"],
        accent_color: str = DARK_COLORS["accent"],
        selected_text_color: str = ON_ACCENT,
        subtext_color: str = DARK_COLORS["subtext"],
    ):
        super().__init__(parent)
        self._separator_color = QColor(separator_color)
        self._accent_color = QColor(accent_color)
        self._selected_text_color = QColor(selected_text_color)
        self._subtext_color = QColor(subtext_color)
        self._playing_song_id: str | None = None

    def set_separator_color(self, color: str) -> None:
        self._separator_color = QColor(color)

    def set_accent_color(self, color: str) -> None:
        self._accent_color = QColor(color)

    def set_selected_text_color(self, color: str) -> None:
        self._selected_text_color = QColor(color)

    def set_subtext_color(self, color: str) -> None:
        self._subtext_color = QColor(color)

    @property
    def playing_song(self) -> str | None:
        """The song with the 'now playing' indicator."""
        return self._playing_song_id

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
        # The title line is subdued by the subtext token, not by alpha:
        # 60-70% alpha dropped it under 4.5:1 (on the light accent 3.6:1).
        if selected:
            painter.fillRect(option.rect, self._accent_color)
            text_color = QColor(self._selected_text_color)
            sub_color = QColor(text_color)
        else:
            text_color = option.palette.text().color()
            sub_color = QColor(self._subtext_color)

        # Marked rows keep their colours under an accent outline, so a marked
        # row and the open song's filled row stay tell-apart.
        if index.data(_MARKED_ROLE):
            outline = QColor(
                self._selected_text_color if selected else self._accent_color
            )
            painter.setPen(QPen(outline, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(option.rect.adjusted(1, 1, -1, -2))

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

        # Title on top (bold), artist below (subdued), as most players do.
        # Both elide rather than clip mid-glyph, and the separation progress
        # keeps its own room on the second line instead of overlapping it.
        bold_font = QFont(option.font)
        bold_font.setBold(True)
        top_rect = rect.adjusted(0, 0, 0, -rect.height() // 2)
        bottom_rect = rect.adjusted(0, rect.height() // 2, 0, 0)
        normal_font = QFont(option.font)
        normal_font.setPointSizeF(option.font.pointSizeF() * 0.9)
        status = index.data(_STATUS_ROLE)
        if status:
            painter.setFont(normal_font)
            badge_width = (
                painter.fontMetrics().horizontalAdvance(status)
                + 2 * scaled_px(5)
            )
            badge_rect = QRectF(
                top_rect.right() + 1 - badge_width,
                top_rect.top() + 1,
                badge_width,
                top_rect.height() - 2,
            )
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(sub_color, 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(
                badge_rect.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4,
            )
            painter.drawText(
                badge_rect, Qt.AlignmentFlag.AlignCenter, status,
            )
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            top_rect.setRight(top_rect.right() - int(badge_width) - 6)
        painter.setFont(bold_font)
        painter.setPen(text_color)
        painter.drawText(
            top_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            painter.fontMetrics().elidedText(
                title or artist, Qt.TextElideMode.ElideRight, top_rect.width(),
            ),
        )

        painter.setFont(normal_font)
        metrics = painter.fontMetrics()
        progress = index.data(_PROGRESS_ROLE)
        practice = index.data(_PRACTICE_ROLE)
        artist_width = bottom_rect.width()
        if practice and not progress:
            painter.setPen(sub_color)
            painter.drawText(
                bottom_rect,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                practice,
            )
            artist_width -= metrics.horizontalAdvance(practice) + 8
        if progress:
            # On a selected row the fill is the accent itself.
            painter.setPen(
                sub_color if selected else QColor(self._accent_color)
            )
            painter.drawText(
                bottom_rect,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                progress,
            )
            artist_width -= metrics.horizontalAdvance(progress) + 8
        painter.setPen(sub_color)
        painter.drawText(
            bottom_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            metrics.elidedText(
                artist if title else "", Qt.TextElideMode.ElideRight,
                max(0, artist_width),
            ),
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
    """Dialog for editing a song's title, artist, tags, and practice status."""

    def __init__(self, song: Song, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Song")
        self.setMinimumWidth(400)
        self._song = song

        layout = QVBoxLayout(self)
        # A form keeps both fields starting at the same x; the labels
        # have different widths.
        form = QFormLayout()
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        form.setVerticalSpacing(8)
        self._title_edit = QLineEdit(song.title)
        form.addRow("Title:", self._title_edit)
        self._artist_edit = QLineEdit(song.artist)
        form.addRow("Artist:", self._artist_edit)
        self._tags_edit = QLineEdit(", ".join(getattr(song, "tags", ())))
        self._tags_edit.setPlaceholderText("Comma-separated tags (e.g. gig, learning)")
        self._tags_edit.setAccessibleName("Tags")
        form.addRow("Tags:", self._tags_edit)

        self._status_combo = QComboBox()
        self._status_combo.setAccessibleName("Practice status")
        for status in PRACTICE_STATUSES:
            self._status_combo.addItem(
                _PRACTICE_STATUS_LABELS.get(status, status or "None"),
                status,
            )
        status_idx = self._status_combo.findData(
            getattr(song, "practice_status", "")
        )
        if status_idx >= 0:
            self._status_combo.setCurrentIndex(status_idx)
        form.addRow("Practice status:", self._status_combo)
        layout.addLayout(form)

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

    @property
    def tags(self) -> list[str]:
        """Return normalized comma-separated tags."""
        return normalize_tags(self._tags_edit.text().split(","))

    @property
    def practice_status(self) -> str:
        """Return the selected practice status key."""
        return str(self._status_combo.currentData() or "")


# ---------------------------------------------------------------------------
# Repeat modes
# ---------------------------------------------------------------------------

REPEAT_OFF = "off"
REPEAT_ALL = "all"
REPEAT_ONE = "one"
_REPEAT_CYCLE = (REPEAT_OFF, REPEAT_ALL, REPEAT_ONE)


class _ElidingComboBox(QComboBox):
    """Combo box that elides long text; a styled QComboBox clips it."""

    def paintEvent(self, _event) -> None:
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        field = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox,
            option,
            QStyle.SubControl.SC_ComboBoxEditField,
            self,
        )
        option.currentText = option.fontMetrics.elidedText(
            option.currentText,
            Qt.TextElideMode.ElideRight,
            field.width() - scaled_px(4),
        )
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)


class _SongList(QListWidget):
    """Song list that tells a click apart from the start of a drag.

    Qt makes a row current on mouse press. Selecting a song loads it, so
    while rows can be dragged the panel waits for the release: a press
    that turns into a drag reorders without touching playback. Selection
    changes from anywhere else (auto-advance) are never held back.
    """

    row_pressed = Signal(str)  # song id, before the press takes effect
    press_released = Signal()
    drag_started = Signal()
    mark_toggled = Signal(str)  # Ctrl+click on a song
    mark_range_requested = Signal(str)  # Shift+click on a song
    mark_all_requested = Signal()
    marks_cleared = Signal()
    drag_finished = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._in_press = False
        self._press_active = False
        self._mark_press = False
        self._dragged_id: str | None = None

    @property
    def in_press(self) -> bool:
        """True while a mouse press on a draggable row is being handled."""
        return self._in_press

    def item_for_song(self, song_id: str) -> QListWidgetItem | None:
        """Return the row showing *song_id*, if any."""
        for i in range(self.count()):
            item = self.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == song_id:
                return item
        return None

    def set_current_silently(self, item: QListWidgetItem | None) -> None:
        """Make *item* (or nothing) current without announcing it."""
        blocked = self.blockSignals(True)
        try:
            self.setCurrentItem(item)
            if item is None:
                self.clearSelection()
        finally:
            self.blockSignals(blocked)
        self.viewport().update()

    def mousePressEvent(self, event) -> None:
        item = self.itemAt(event.position().toPoint())
        # Ctrl and Shift clicks mark songs for a batch action. They never
        # reach Qt, so they cannot load a song or start a drag.
        modifiers = event.modifiers()
        marking = modifiers & (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.ShiftModifier
        )
        if event.button() == Qt.MouseButton.RightButton:
            # Qt would make the row current, loading the song; the context
            # menu arrives as its own event.
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if marking:
                self._press_active = False
                self._mark_press = True
                event.accept()
                if item is not None:
                    song_id = item.data(Qt.ItemDataRole.UserRole)
                    if modifiers & Qt.KeyboardModifier.ShiftModifier:
                        self.mark_range_requested.emit(song_id)
                    else:
                        self.mark_toggled.emit(song_id)
                return
            self._mark_press = False
            self.marks_cleared.emit()
        self._press_active = (
            self.dragEnabled()
            and event.button() == Qt.MouseButton.LeftButton
            and item is not None
        )
        if not self._press_active:
            super().mousePressEvent(event)
            return
        self.row_pressed.emit(item.data(Qt.ItemDataRole.UserRole))
        self._in_press = True
        try:
            super().mousePressEvent(event)
        finally:
            self._in_press = False

    def mouseMoveEvent(self, event) -> None:
        if self._mark_press and event.buttons():
            return
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        # A quick second Ctrl or Shift click is another mark, not an edit.
        if event.modifiers() & (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.ShiftModifier
        ):
            self.mousePressEvent(event)
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        modifiers = event.modifiers()
        current = self.currentItem()
        if event.matches(QKeySequence.StandardKey.SelectAll):
            self.mark_all_requested.emit()
        elif event.key() == Qt.Key.Key_Escape:
            self.marks_cleared.emit()
        elif (
            event.key() == Qt.Key.Key_Space
            and modifiers == Qt.KeyboardModifier.ControlModifier
            and current is not None
        ):
            self.mark_toggled.emit(current.data(Qt.ItemDataRole.UserRole))
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._mark_press and event.button() == Qt.MouseButton.LeftButton:
            self._mark_press = False
            event.accept()
            return
        super().mouseReleaseEvent(event)
        if self._press_active:
            self._press_active = False
            self.press_released.emit()

    def startDrag(self, supported_actions) -> None:
        self._press_active = False
        # The panel puts the pressed row back first, in case another song
        # became current between the press and the drag.
        self.drag_started.emit()
        current = self.currentItem()
        self._dragged_id = (
            current.data(Qt.ItemDataRole.UserRole)
            if current is not None
            else None
        )
        try:
            super().startDrag(supported_actions)
        finally:
            self._dragged_id = None
        self.drag_finished.emit()

    def dropEvent(self, event) -> None:
        # Qt moves the selected row; playback may have advanced to another
        # song mid-drag, so select the dragged one again.
        if self._dragged_id is not None:
            item = self.item_for_song(self._dragged_id)
            if item is None:
                event.ignore()
                return
            self.set_current_silently(item)
        super().dropEvent(event)


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
    song_edited = Signal(str)
    song_removal_requested = Signal(str)  # emitted before files are deleted
    song_removed = Signal(str)  # emitted with the removed song's id
    cancel_separation_requested = Signal(str)  # song_id still separating
    play_next_requested = Signal(str)
    add_to_queue_requested = Signal(str)
    play_next_many_requested = Signal(list)
    add_many_to_queue_requested = Signal(list)
    export_songs_requested = Signal(list)
    queue_toggled = Signal(bool)
    play_setlist_requested = Signal(str)
    setlist_changed = Signal(str)
    repeat_mode_changed = Signal(str)
    shuffle_toggled = Signal(bool)
    previous_requested = Signal()
    next_requested = Signal()

    def __init__(self, library: SongLibrary, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._library = library
        self._repeat_mode: str = REPEAT_OFF
        self._shuffle_enabled: bool = False
        self._filter_kind: str = "all"
        self._filter_value: str | None = None
        self._repopulating_list: bool = False
        # (song id to select again after a drag, pressed song id) from a
        # press on a draggable row until its click or drag ends.
        self._pressed_songs: tuple[str | None, str] | None = None
        # Songs marked for a batch action, and where Shift+click ranges start.
        self._marked: set[str] = set()
        self._mark_anchor: str | None = None
        # Looks up a song's stored loop/speed/pitch (set by the main window).
        self._practice_provider: Callable[[str], PracticeState] | None = None
        # song_id -> progress text for songs still separating in the
        # background. Rows in this state are unselectable (their stems
        # are half-written) and offer only "Cancel separation".
        self._separating: dict[str, str] = {}

        self._setup_ui()
        self.refresh()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        self._header_label = QLabel("Library")
        self._header_label.setObjectName("title-label")
        header_row.addWidget(self._header_label)
        header_row.addStretch()
        self._queue_count_label = QLabel("")
        self._queue_count_label.setObjectName("subtle-label")
        self._queue_count_label.setVisible(False)
        header_row.addWidget(self._queue_count_label)
        layout.addLayout(header_row)

        # Page 0 keeps _search_edit as the first focusable control in the
        # panel so window Tab order stays top-to-bottom across panels.
        self._stack = QStackedWidget()

        library_page = QWidget()
        library_layout = QVBoxLayout(library_page)
        library_layout.setContentsMargins(0, 0, 0, 0)

        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search songs…")
        # A placeholder is not a name for screen readers.
        self._search_edit.setAccessibleName("Search songs")
        self._search_edit.setClearButtonEnabled(True)
        self._search_edit.textChanged.connect(self._apply_filter)
        library_layout.addWidget(self._search_edit)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.setSpacing(4)
        self._collection_combo = _ElidingComboBox()
        self._collection_combo.setAccessibleName("Filter by setlist or tag")
        self._collection_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._collection_combo.setMinimumContentsLength(10)
        self._collection_combo.currentIndexChanged.connect(
            self._on_collection_changed
        )
        filter_row.addWidget(self._collection_combo, 1)

        self._setlist_menu_btn = QPushButton("...")
        self._setlist_menu_btn.setObjectName("icon-btn")
        fix_button_size(
            self._setlist_menu_btn, COMPACT_BUTTON, COMPACT_BUTTON,
        )
        self._setlist_menu_btn.setToolTip("Manage setlists")
        self._setlist_menu_btn.setAccessibleName("Manage setlists")
        self._setlist_menu_btn.clicked.connect(self._on_setlist_menu_clicked)
        filter_row.addWidget(self._setlist_menu_btn)
        library_layout.addLayout(filter_row)

        self._setlist_bar = QWidget()
        setlist_row = QHBoxLayout(self._setlist_bar)
        setlist_row.setContentsMargins(0, 0, 0, 0)
        setlist_row.setSpacing(6)
        self._play_setlist_btn = QPushButton("Play Setlist")
        self._play_setlist_btn.setAccessibleName("Play setlist")
        self._play_setlist_btn.clicked.connect(self._on_play_setlist_clicked)
        setlist_row.addWidget(self._play_setlist_btn)
        self._setlist_hint_label = QLabel("Drag to reorder")
        self._setlist_hint_label.setObjectName("subtle-label")
        setlist_row.addWidget(self._setlist_hint_label)
        setlist_row.addStretch()
        self._setlist_bar.setVisible(False)
        library_layout.addWidget(self._setlist_bar)

        self._list = _SongList()
        self._song_delegate = _SongDelegate(self._list)
        self._list.setItemDelegate(self._song_delegate)
        self._list.setAccessibleName("Song library")
        self._list.currentItemChanged.connect(self._on_item_changed)
        self._list.itemDoubleClicked.connect(lambda _: self._on_edit_song())
        self._list.model().rowsMoved.connect(self._on_list_rows_moved)
        self._list.mark_toggled.connect(self.toggle_song_mark)
        self._list.mark_range_requested.connect(self._on_mark_range)
        self._list.mark_all_requested.connect(self._on_mark_all)
        self._list.marks_cleared.connect(self.clear_marks)
        self._list.row_pressed.connect(self._on_list_row_pressed)
        self._list.drag_started.connect(self._on_list_drag_started)
        self._list.press_released.connect(self._on_list_press_released)
        self._list.drag_finished.connect(self._on_list_drag_finished)
        self._list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        library_layout.addWidget(self._list)

        # A search with no match, or an empty setlist, would otherwise
        # leave a blank list.
        self._no_match_label = QLabel("No songs match your search")
        self._no_match_label.setObjectName("subtle-label")
        self._no_match_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_match_label.setVisible(False)
        library_layout.addWidget(self._no_match_label)

        self._batch_bar = QWidget()
        batch_row = QHBoxLayout(self._batch_bar)
        batch_row.setContentsMargins(0, 0, 0, 0)
        batch_row.setSpacing(4)
        self._batch_label = QLabel("")
        batch_row.addWidget(self._batch_label, 1)
        self._batch_actions_btn = QPushButton("Actions")
        self._batch_actions_btn.clicked.connect(self._on_batch_menu_clicked)
        batch_row.addWidget(self._batch_actions_btn)
        self._batch_clear_btn = QPushButton("Clear")
        self._batch_clear_btn.setAccessibleName("Clear selection")
        self._batch_clear_btn.clicked.connect(self.clear_marks)
        batch_row.addWidget(self._batch_clear_btn)
        self._batch_bar.setVisible(False)
        library_layout.addWidget(self._batch_bar)

        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.setEnabled(False)
        self._remove_btn.setToolTip("Remove selected song from library")
        self._remove_btn.setAccessibleName("Remove selected song")
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        library_layout.addWidget(self._remove_btn)

        self._queue_widget = PlayQueueWidget()
        self._stack.addWidget(library_page)
        self._stack.addWidget(self._queue_widget)
        layout.addWidget(self._stack, 1)

        # -- Playback control bar --
        ctrl_bar = QHBoxLayout()
        ctrl_bar.setSpacing(4)

        icon_color = QColor(DARK_COLORS["text"])
        checked_icon_color = QColor(DARK_COLORS["base"])

        # Repeat (cycles: off → all → one)
        self._repeat_btn = QPushButton()
        self._repeat_btn.setObjectName("icon-btn")
        fix_button_size(self._repeat_btn, COMPACT_BUTTON, COMPACT_BUTTON)
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
        fix_button_size(self._shuffle_btn, COMPACT_BUTTON, COMPACT_BUTTON)
        self._shuffle_btn.setToolTip("Shuffle: off")
        self._shuffle_btn.setAccessibleName("Shuffle")
        self._shuffle_btn.setIcon(
            _make_toggle_icon(_draw_shuffle, icon_color, checked_icon_color)
        )
        self._shuffle_btn.setIconSize(QSize(_CTRL_ICON, _CTRL_ICON))
        self._shuffle_btn.toggled.connect(self._on_shuffle_toggled)
        ctrl_bar.addWidget(self._shuffle_btn)

        # Play queue view toggle
        self._queue_btn = QPushButton()
        self._queue_btn.setObjectName("icon-btn")
        self._queue_btn.setCheckable(True)
        fix_button_size(self._queue_btn, COMPACT_BUTTON, COMPACT_BUTTON)
        self._queue_btn.setToolTip("Play queue (Q)")
        self._queue_btn.setAccessibleName("Play queue")
        self._queue_btn.setIcon(
            _make_toggle_icon(_draw_queue, icon_color, checked_icon_color)
        )
        self._queue_btn.setIconSize(QSize(_CTRL_ICON, _CTRL_ICON))
        self._queue_btn.toggled.connect(self._on_queue_btn_toggled)
        ctrl_bar.addWidget(self._queue_btn)

        ctrl_bar.addStretch()

        # Previous
        self._prev_btn = QPushButton()
        self._prev_btn.setObjectName("icon-btn")
        fix_button_size(self._prev_btn, COMPACT_BUTTON, COMPACT_BUTTON)
        self._prev_btn.setToolTip("Previous song (P)")
        self._prev_btn.setAccessibleName("Previous song")
        self._prev_btn.setIcon(_make_icon(_draw_prev, icon_color))
        self._prev_btn.setIconSize(QSize(_CTRL_ICON, _CTRL_ICON))
        self._prev_btn.clicked.connect(self.previous_requested)
        ctrl_bar.addWidget(self._prev_btn)

        # Next
        self._next_btn = QPushButton()
        self._next_btn.setObjectName("icon-btn")
        fix_button_size(self._next_btn, COMPACT_BUTTON, COMPACT_BUTTON)
        self._next_btn.setToolTip("Next song (N)")
        self._next_btn.setAccessibleName("Next song")
        self._next_btn.setIcon(_make_icon(_draw_next, icon_color))
        self._next_btn.setIconSize(QSize(_CTRL_ICON, _CTRL_ICON))
        self._next_btn.clicked.connect(self.next_requested)
        ctrl_bar.addWidget(self._next_btn)

        layout.addLayout(ctrl_bar)

    # ------------------------------------------------------------------
    # Play queue view
    # ------------------------------------------------------------------

    @property
    def queue_widget(self) -> PlayQueueWidget:
        """Embedded play queue management widget."""
        return self._queue_widget

    @property
    def is_queue_visible(self) -> bool:
        """True when the sidebar is showing the Play Queue view."""
        return self._stack.currentIndex() == 1

    def set_queue_visible(self, visible: bool) -> None:
        """Switch between the Library collection view and the Play Queue view."""
        visible = bool(visible)
        target_index = 1 if visible else 0
        changed = self._stack.currentIndex() != target_index
        self._stack.setCurrentIndex(target_index)
        if self._queue_btn.isChecked() != visible:
            self._queue_btn.blockSignals(True)
            self._queue_btn.setChecked(visible)
            self._queue_btn.blockSignals(False)
        self._header_label.setText("Play Queue" if visible else "Library")
        if changed:
            self.queue_toggled.emit(visible)

    def toggle_queue_visible(self) -> None:
        """Toggle between the Library view and the Play Queue view."""
        self.set_queue_visible(not self.is_queue_visible)

    def _on_queue_btn_toggled(self, checked: bool) -> None:
        self.set_queue_visible(checked)

    def update_queue_display(self, queue: PlayQueue) -> None:
        """Refresh queue widget contents and sidebar count badge."""
        self._queue_widget.refresh(queue, self._library)
        count = len(queue)
        has_setlist = queue.is_setlist_active
        if has_setlist and count:
            badge = f"Setlist: {count} queued"
        elif has_setlist:
            badge = "Setlist"
        elif count:
            badge = f"{count} queued"
        else:
            badge = ""
        self._queue_count_label.setText(badge)
        self._queue_count_label.setVisible(bool(badge))

        if count:
            self._queue_btn.setToolTip(f"Play queue ({count} up next) (Q)")
            self._queue_btn.setAccessibleName(f"Play queue, {count} up next")
        else:
            self._queue_btn.setToolTip("Play queue (Q)")
            self._queue_btn.setAccessibleName("Play queue")

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
        self._repeat_btn.setToolTip(f"Repeat: {labels[self._repeat_mode]}")
        self._repeat_btn.setAccessibleName(
            f"Repeat {labels[self._repeat_mode]}"
        )
        # Repeat cycles three modes, so it stays a plain button: checkable, it
        # became a checkbox to assistive tech, whose Toggle action flipped the
        # look without changing the mode. The "active" property lets the app
        # sheet style it with the :checked rules (accent fill plus hover and
        # focus feedback); a widget stylesheet here outranked all of that.
        active = self._repeat_mode != REPEAT_OFF
        if self._repeat_btn.property("active") != active:
            self._repeat_btn.setProperty("active", active)
            repolish(self._repeat_btn)

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
        # Selected rows and checked buttons sit on the accent fill, so text
        # and glyphs there use the theme's on-accent colour.
        self._song_delegate.set_selected_text_color(colors["on_accent"])
        self._song_delegate.set_subtext_color(colors["subtext"])

        icon_color = QColor(colors["text"])
        on_accent_color = QColor(colors["on_accent"])
        self._repeat_icons = {
            # REPEAT_OFF has a neutral (grey) background — use theme text color.
            REPEAT_OFF: _make_icon(_draw_repeat, icon_color),
            # REPEAT_ALL / REPEAT_ONE paint on the teal background — use on-accent.
            REPEAT_ALL: _make_icon(_draw_repeat, on_accent_color),
            REPEAT_ONE: _make_icon(_draw_repeat_one, on_accent_color),
        }
        self._update_repeat_ui()
        self._shuffle_btn.setIcon(
            _make_toggle_icon(_draw_shuffle, icon_color, on_accent_color)
        )
        self._queue_btn.setIcon(
            _make_toggle_icon(_draw_queue, icon_color, on_accent_color)
        )
        self._prev_btn.setIcon(_make_icon(_draw_prev, icon_color))
        self._next_btn.setIcon(_make_icon(_draw_next, icon_color))

        self._list.viewport().update()
        self._queue_widget.apply_theme(theme, colors)

    # ------------------------------------------------------------------
    # Setlist / tag filtering and management
    # ------------------------------------------------------------------

    @property
    def filter_kind(self) -> str:
        """Current sidebar collection filter ('all', 'setlist', or 'tag')."""
        return self._filter_kind

    @property
    def filter_value(self) -> str | None:
        """ID of the active setlist or name of the active tag filter."""
        return self._filter_value

    def set_collection_filter(
        self, kind: str, value: str | None = None,
    ) -> None:
        """Programmatically select a collection, setlist, or tag filter."""
        if kind not in ("all", "setlist", "tag"):
            kind = "all"
            value = None
        self._filter_kind = kind
        self._filter_value = None if kind == "all" else value
        self.refresh()

    def _library_setlists(self) -> list[Setlist]:
        raw = getattr(self._library, "setlists", ())
        if isinstance(raw, (list, tuple)):
            return [sl for sl in raw if isinstance(sl, Setlist)]
        return []

    def _library_all_tags(self) -> list[str]:
        fn = getattr(self._library, "all_tags", None)
        if callable(fn):
            res = fn()
            if isinstance(res, (list, tuple)):
                return [t for t in res if isinstance(t, str)]
        return []

    def _library_get_setlist(self, setlist_id: str | None) -> Setlist | None:
        if not setlist_id:
            return None
        fn = getattr(self._library, "get_setlist", None)
        if callable(fn):
            sl = fn(setlist_id)
            return sl if isinstance(sl, Setlist) else None
        return None

    def _rebuild_collection_combo(self) -> None:
        setlists = self._library_setlists()
        tags = self._library_all_tags()
        self._collection_combo.blockSignals(True)
        try:
            self._collection_combo.clear()
            self._collection_combo.addItem("All Songs", ("all", None))
            if setlists:
                self._collection_combo.insertSeparator(
                    self._collection_combo.count()
                )
                for sl in setlists:
                    self._collection_combo.addItem(
                        f"Setlist: {sl.name} ({len(sl.song_ids)})",
                        ("setlist", sl.id),
                    )
            if tags:
                self._collection_combo.insertSeparator(
                    self._collection_combo.count()
                )
                for tag in tags:
                    self._collection_combo.addItem(f"Tag: {tag}", ("tag", tag))

            target_idx = -1
            for i in range(self._collection_combo.count()):
                data = self._collection_combo.itemData(i)
                if not isinstance(data, tuple) or len(data) != 2:
                    continue
                kind, val = data
                if kind != self._filter_kind:
                    continue
                if kind == "all":
                    target_idx = i
                    break
                if kind == "setlist" and val == self._filter_value:
                    target_idx = i
                    break
                if (
                    kind == "tag"
                    and isinstance(val, str)
                    and isinstance(self._filter_value, str)
                    and val.casefold() == self._filter_value.casefold()
                ):
                    self._filter_value = val
                    target_idx = i
                    break
            if target_idx < 0:
                self._filter_kind = "all"
                self._filter_value = None
                target_idx = 0
            self._collection_combo.setCurrentIndex(target_idx)
            # The combo elides long names at the narrow sidebar width.
            self._collection_combo.setToolTip(
                self._collection_combo.currentText()
            )
        finally:
            self._collection_combo.blockSignals(False)

    def _on_collection_changed(self, index: int) -> None:
        data = self._collection_combo.itemData(index)
        if isinstance(data, tuple) and len(data) == 2:
            self._filter_kind, self._filter_value = data
        else:
            self._filter_kind = "all"
            self._filter_value = None
        self.refresh()

    def _filtered_songs(self) -> list[Song]:
        if self._filter_kind == "setlist" and self._filter_value:
            sl = self._library_get_setlist(self._filter_value)
            if sl is None:
                return []
            songs: list[Song] = []
            for sid in sl.song_ids:
                song = self._library.get_song(sid)
                if song is not None:
                    songs.append(song)
            return songs
        if self._filter_kind == "tag" and self._filter_value:
            target = self._filter_value.casefold()
            matched: list[Song] = []
            for song in self._library.songs:
                tags = getattr(song, "tags", ())
                if isinstance(tags, (list, tuple)) and any(
                    isinstance(t, str) and t.casefold() == target for t in tags
                ):
                    matched.append(song)
            return matched
        return list(self._library.songs)

    def _sync_drag_reorder_mode(self) -> None:
        writable = bool(getattr(self._library, "writable", True))
        can_drag = (
            self._filter_kind == "setlist"
            and bool(self._filter_value)
            and not self._search_edit.text().strip()
            and writable
        )
        if can_drag:
            self._list.setDragDropMode(
                QAbstractItemView.DragDropMode.InternalMove
            )
            self._list.setDefaultDropAction(Qt.DropAction.MoveAction)
            self._list.setDragEnabled(True)
            self._list.setAcceptDrops(True)
            self._list.setDropIndicatorShown(True)
        else:
            self._list.setDragDropMode(
                QAbstractItemView.DragDropMode.NoDragDrop
            )
            self._list.setDragEnabled(False)
            self._list.setAcceptDrops(False)
            self._list.setDropIndicatorShown(False)
        self._setlist_hint_label.setVisible(can_drag and self._list.count() > 1)

    def _on_list_rows_moved(
        self, _parent, start: int, _end: int, _dest_parent, dest_row: int,
    ) -> None:
        if (
            self._repopulating_list
            or self._filter_kind != "setlist"
            or not self._filter_value
        ):
            return
        # Qt reports dest_row in pre-move coordinates; moving downward inserts
        # one slot past the final post-move index.
        target = dest_row - 1 if dest_row > start else dest_row
        if start != target:
            self._reorder_active_setlist(start, target)

    def _on_play_setlist_clicked(self) -> None:
        if self._filter_kind == "setlist" and self._filter_value:
            self.play_setlist_requested.emit(self._filter_value)

    def _on_setlist_menu_clicked(self) -> None:
        writable = bool(getattr(self._library, "writable", True))
        menu = QMenu(self)
        new_act = menu.addAction("New Setlist...")
        new_act.setEnabled(writable)
        new_act.triggered.connect(lambda _c=False: self._on_new_setlist())
        if self._filter_kind == "setlist" and self._filter_value:
            menu.addSeparator()
            play_act = menu.addAction("Play Setlist")
            play_act.setEnabled(bool(self._filtered_songs()))
            play_act.triggered.connect(self._on_play_setlist_clicked)
            rename_act = menu.addAction("Rename Setlist...")
            rename_act.setEnabled(writable)
            rename_act.triggered.connect(self._on_rename_setlist)
            del_act = menu.addAction("Delete Setlist")
            del_act.setEnabled(writable)
            del_act.triggered.connect(self._on_delete_setlist)
        menu.exec(
            self._setlist_menu_btn.mapToGlobal(
                self._setlist_menu_btn.rect().bottomLeft()
            )
        )

    def _on_new_setlist(
        self, initial_song_ids: list[str] | None = None,
    ) -> None:
        if not self._library_writable("New Setlist"):
            return
        name, ok = QInputDialog.getText(
            self, "New Setlist", "Setlist name:",
        )
        if not ok or not name.strip():
            return
        try:
            sl = self._library.create_setlist(name.strip(), initial_song_ids)
        except (OSError, KeyError, ValueError) as exc:
            QMessageBox.warning(
                self, "New Setlist",
                describe_error(exc, "Could not create setlist"),
            )
            return
        if initial_song_ids is None:
            self._filter_kind = "setlist"
            self._filter_value = sl.id
        self.refresh()
        self.setlist_changed.emit(sl.id)

    def _on_rename_setlist(self) -> None:
        if self._filter_kind != "setlist" or not self._filter_value:
            return
        if not self._library_writable("Rename Setlist"):
            return
        sl = self._library_get_setlist(self._filter_value)
        if sl is None:
            return
        name, ok = QInputDialog.getText(
            self, "Rename Setlist", "Setlist name:", text=sl.name,
        )
        if not ok or not name.strip():
            return
        try:
            self._library.rename_setlist(sl.id, name.strip())
        except (OSError, KeyError, ValueError) as exc:
            QMessageBox.warning(
                self, "Rename Setlist",
                describe_error(exc, "Could not rename setlist"),
            )
            return
        self.refresh()
        self.setlist_changed.emit(sl.id)

    def _on_delete_setlist(self) -> None:
        if self._filter_kind != "setlist" or not self._filter_value:
            return
        if not self._library_writable("Delete Setlist"):
            return
        sl = self._library_get_setlist(self._filter_value)
        if sl is None:
            return
        reply = QMessageBox.question(
            self,
            "Delete Setlist",
            f"Delete setlist '{sl.name}'? Songs remain in your library.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        sid = sl.id
        try:
            self._library.delete_setlist(sid)
        except (OSError, KeyError, ValueError) as exc:
            QMessageBox.warning(
                self, "Delete Setlist",
                describe_error(exc, "Could not delete setlist"),
            )
            return
        self._filter_kind = "all"
        self._filter_value = None
        self.refresh()
        self.setlist_changed.emit(sid)

    def _on_add_song_to_setlist(self, setlist_id: str, song_id: str) -> None:
        self._add_to_setlist(setlist_id, [song_id])

    def _add_to_setlist(self, setlist_id: str, song_ids: list[str]) -> None:
        if not self._library_writable("Add to Setlist"):
            return
        try:
            self._library.add_songs_to_setlist(setlist_id, song_ids)
        except (OSError, KeyError, ValueError) as exc:
            QMessageBox.warning(
                self, "Add to Setlist",
                describe_error(exc, "Could not update setlist"),
            )
            return
        self.refresh()
        self.setlist_changed.emit(setlist_id)

    def _on_remove_song_from_setlist(
        self, setlist_id: str, song_id: str,
    ) -> None:
        if not self._library_writable("Remove from Setlist"):
            return
        try:
            self._library.remove_song_from_setlist(setlist_id, song_id)
        except (OSError, KeyError, ValueError) as exc:
            QMessageBox.warning(
                self, "Remove from Setlist",
                describe_error(exc, "Could not update setlist"),
            )
            return
        self.refresh()
        self.setlist_changed.emit(setlist_id)

    def _reorder_active_setlist(
        self, from_index: int, to_index: int,
    ) -> None:
        if self._filter_kind != "setlist" or not self._filter_value:
            return
        if not self._library_writable("Reorder Setlist"):
            self.refresh()
            return
        sl = self._library_get_setlist(self._filter_value)
        if sl is None:
            self.refresh()
            return
        ids = list(sl.song_ids)
        if (
            not (0 <= from_index < len(ids))
            or not (0 <= to_index < len(ids))
            or from_index == to_index
        ):
            self.refresh()
            return
        moved = ids.pop(from_index)
        ids.insert(to_index, moved)
        try:
            self._library.reorder_setlist(sl.id, ids)
        except (OSError, KeyError, ValueError) as exc:
            QMessageBox.warning(
                self, "Reorder Setlist",
                describe_error(exc, "Could not reorder setlist"),
            )
            self.refresh()
            return
        self.refresh()
        self.setlist_changed.emit(sl.id)

    # ------------------------------------------------------------------
    # Song list management
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Reload the song list from the library without re-triggering selection."""
        self._rebuild_collection_combo()
        songs = self._filtered_songs()

        current = self._list.currentItem()
        selected_id = (
            current.data(Qt.ItemDataRole.UserRole)
            if current is not None
            else None
        )
        if selected_id is None:
            # Coming back from a view without it, the open song is
            # highlighted again.
            selected_id = self._song_delegate.playing_song
        selected_item = None
        self._repopulating_list = True
        self._list.blockSignals(True)
        try:
            self._list.clear()
            for song in songs:
                # Display text used for search filtering; delegate paints the rows.
                item = QListWidgetItem(f"{song.artist} - {song.title}")
                item.setData(Qt.ItemDataRole.UserRole, song.id)
                item.setData(_ARTIST_ROLE, song.artist)
                item.setData(_TITLE_ROLE, song.title)
                tags = getattr(song, "tags", ())
                tags_text = (
                    " ".join(t for t in tags if isinstance(t, str))
                    if isinstance(tags, (list, tuple))
                    else ""
                )
                item.setData(
                    _SEARCH_TEXT_ROLE,
                    f"{song.artist} {song.title} {tags_text}".lower(),
                )
                self._apply_separating_state(item, song.id)
                self._apply_practice_state(item, song)
                if song.id in self._marked:
                    item.setData(_MARKED_ROLE, True)
                self._list.addItem(item)
                if selected_id and song.id == selected_id:
                    selected_item = item
            if selected_item is not None:
                self._list.setCurrentItem(selected_item)
        finally:
            self._list.blockSignals(False)
            self._repopulating_list = False

        in_setlist = (
            self._filter_kind == "setlist" and bool(self._filter_value)
        )
        self._setlist_bar.setVisible(in_setlist)
        self._play_setlist_btn.setEnabled(in_setlist and bool(songs))
        self._apply_filter(self._search_edit.text())
        if selected_item is not None and not selected_item.isHidden():
            self._remove_btn.setEnabled(bool(self._library.writable))
        self._sync_marks()

    # ------------------------------------------------------------------
    # Batch marks
    # ------------------------------------------------------------------

    def marked_song_ids(self) -> list[str]:
        """Return the marked songs in the order shown."""
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self._markable_items()
            if item.data(Qt.ItemDataRole.UserRole) in self._marked
        ]

    def toggle_song_mark(self, song_id: str) -> None:
        """Mark or unmark *song_id* for a batch action.

        The first mark takes the selected song along, as Ctrl+click does in
        any file list.
        """
        shown = [
            i.data(Qt.ItemDataRole.UserRole) for i in self._markable_items()
        ]
        if song_id not in shown:
            return
        if not self._marked:
            current = self._list.currentItem()
            if current is not None:
                current_id = current.data(Qt.ItemDataRole.UserRole)
                if current_id in shown:
                    self._marked.add(current_id)
            self._marked.add(song_id)
        else:
            self._marked ^= {song_id}
        self._mark_anchor = song_id
        self._sync_marks()

    def clear_marks(self) -> None:
        """Forget every mark."""
        self._mark_anchor = None
        if self._marked:
            self._marked.clear()
            self._sync_marks()

    def _markable_items(self) -> list[QListWidgetItem]:
        """Rows that are shown and not still separating."""
        rows = (self._list.item(i) for i in range(self._list.count()))
        return [
            item for item in rows
            if not item.isHidden()
            and not self.is_song_separating(
                item.data(Qt.ItemDataRole.UserRole)
            )
        ]

    def _on_mark_range(self, song_id: str) -> None:
        shown = [
            i.data(Qt.ItemDataRole.UserRole) for i in self._markable_items()
        ]
        if song_id not in shown:
            return
        anchor = self._mark_anchor
        if anchor not in shown:
            current = self._list.currentItem()
            anchor = (
                current.data(Qt.ItemDataRole.UserRole)
                if current is not None
                else None
            )
        if anchor not in shown:
            anchor = song_id
        first, last = sorted((shown.index(anchor), shown.index(song_id)))
        self._marked = set(shown[first:last + 1])
        self._mark_anchor = anchor
        self._sync_marks()

    def _on_mark_all(self) -> None:
        self._marked = {
            i.data(Qt.ItemDataRole.UserRole) for i in self._markable_items()
        }
        self._sync_marks()

    def _sync_marks(self) -> None:
        """Drop marks on songs no longer shown and redraw the rest.

        A batch action must never reach a song the user cannot see.
        """
        self._marked &= {
            i.data(Qt.ItemDataRole.UserRole) for i in self._markable_items()
        }
        for i in range(self._list.count()):
            item = self._list.item(i)
            song_id = item.data(Qt.ItemDataRole.UserRole)
            marked = song_id in self._marked
            if bool(item.data(_MARKED_ROLE)) != marked:
                item.setData(_MARKED_ROLE, marked)
                song = self._library.get_song(song_id)
                if song is not None:
                    self._apply_practice_state(item, song)
        count = len(self._marked)
        self._batch_label.setText(f"{count} selected")
        self._batch_bar.setVisible(count > 0)
        noun = "song" if count == 1 else "songs"
        self._remove_btn.setAccessibleName(
            f"Remove {count} selected {noun}"
            if count
            else "Remove selected song"
        )
        current = self._list.currentItem()
        self._remove_btn.setEnabled(
            bool(self._library.writable)
            and (
                count > 0
                or (current is not None and not current.isHidden())
            )
        )

    def _on_batch_menu_clicked(self) -> None:
        menu = QMenu(self)
        self._fill_batch_menu(menu)
        menu.exec(
            self._batch_actions_btn.mapToGlobal(
                self._batch_actions_btn.rect().bottomLeft()
            )
        )

    def _fill_batch_menu(self, menu: QMenu) -> None:
        """Add the actions for every marked song to *menu*."""
        ids = self.marked_song_ids()
        writable = bool(getattr(self._library, "writable", True))
        menu.addAction("Play Next").triggered.connect(
            lambda _c=False: self.play_next_many_requested.emit(ids)
        )
        menu.addAction("Add to Queue").triggered.connect(
            lambda _c=False: self.add_many_to_queue_requested.emit(ids)
        )
        setlist_menu = menu.addMenu("Add to Setlist")
        setlist_menu.setEnabled(writable)
        setlists = self._library_setlists()
        for sl in setlists:
            setlist_menu.addAction(
                sl.name.replace("&", "&&")
            ).triggered.connect(
                lambda _c=False, slid=sl.id: self._add_to_setlist(slid, ids)
            )
        if setlists:
            setlist_menu.addSeparator()
        setlist_menu.addAction("New Setlist...").triggered.connect(
            lambda _c=False: self._on_new_setlist(ids)
        )
        tag_act = menu.addAction("Add Tag...")
        tag_act.setEnabled(writable)
        tag_act.triggered.connect(lambda _c=False: self._on_add_tag(ids))
        menu.addSeparator()
        menu.addAction("Export...").triggered.connect(
            lambda _c=False: self.export_songs_requested.emit(ids)
        )
        remove_act = menu.addAction("Remove...")
        remove_act.setEnabled(writable)
        remove_act.triggered.connect(lambda _c=False: self._remove_songs(ids))

    def _on_add_tag(self, song_ids: list[str]) -> None:
        if not self._library_writable("Add Tag"):
            return
        tag, ok = QInputDialog.getItem(
            self, "Add Tag", "Tag:", ["", *self._library_all_tags()], 0, True,
        )
        # Commas separate tags in the Edit Song dialog.
        tags = normalize_tags(tag.split(",")) if ok else []
        for name in tags:
            try:
                self._library.add_tag_to_songs(song_ids, name)
            except (OSError, KeyError, ValueError) as exc:
                QMessageBox.warning(
                    self, "Add Tag", describe_error(exc, "Could not add tag"),
                )
                break
        if tags:
            self.refresh()

    # ------------------------------------------------------------------
    # Background-separation state
    # ------------------------------------------------------------------

    def set_song_separating(self, song_id: str, text: str) -> None:
        """Mark *song_id* as separating and show *text* on its row."""
        self._separating[song_id] = text
        if song_id in self._marked:
            self._sync_marks()
        item = self._item_for_song(song_id)
        if item is not None:
            self._apply_separating_state(item, song_id)

    def clear_song_separating(self, song_id: str) -> None:
        """Return *song_id*'s row to the normal, selectable state."""
        self._separating.pop(song_id, None)
        item = self._item_for_song(song_id)
        if item is not None:
            self._apply_separating_state(item, song_id)

    def is_song_separating(self, song_id: str) -> bool:
        """True while *song_id* is separating in the background."""
        return song_id in self._separating

    # ------------------------------------------------------------------
    # Practice state
    # ------------------------------------------------------------------

    def set_practice_provider(
        self, provider: Callable[[str], PracticeState],
    ) -> None:
        """Set where rows look up a song's loop, speed, and pitch."""
        self._practice_provider = provider
        self.refresh()

    def update_song_practice(self, song_id: str) -> None:
        """Redraw *song_id*'s practice badges after a change."""
        item = self._item_for_song(song_id)
        song = self._library.get_song(song_id)
        if item is not None and song is not None:
            self._apply_practice_state(item, song)

    def _apply_practice_state(self, item: QListWidgetItem, song: Song) -> None:
        """Set a row's status badge, practice text, and spoken details."""
        state = (
            self._practice_provider(song.id)
            if self._practice_provider is not None
            else PracticeState()
        )
        raw_status = getattr(song, "practice_status", "")
        status = (
            _PRACTICE_STATUS_LABELS.get(raw_status, "")
            if isinstance(raw_status, str) and raw_status
            else ""
        )
        tags = getattr(song, "tags", ())
        tags = (
            [t for t in tags if isinstance(t, str)]
            if isinstance(tags, (list, tuple))
            else []
        )
        item.setData(_STATUS_ROLE, status)
        item.setData(_PRACTICE_ROLE, state.badge_text())
        details = ["Selected"] if song.id in self._marked else []
        if status:
            details.append(f"Status: {status}")
        if tags:
            details.append(f"Tags: {', '.join(tags)}")
        details.extend(state.description_lines())
        item.setToolTip(
            "\n".join([item.text(), *details]) if details else ""
        )
        item.setData(
            Qt.ItemDataRole.AccessibleDescriptionRole, ", ".join(details),
        )

    def _item_for_song(self, song_id: str) -> QListWidgetItem | None:
        return self._list.item_for_song(song_id)

    def _apply_separating_state(self, item: QListWidgetItem, song_id: str) -> None:
        """Sync an item's progress role and selectability with the state.

        Separating rows are made unselectable at the Qt level: their
        stems are half-written, so loading (or removing) them must be
        impossible until the job ends.
        """
        text = self._separating.get(song_id)
        item.setData(_PROGRESS_ROLE, text or None)
        flags = item.flags()
        if text:
            item.setFlags(flags & ~Qt.ItemFlag.ItemIsSelectable)
        else:
            item.setFlags(flags | Qt.ItemFlag.ItemIsSelectable)

    def song_count(self) -> int:
        """Return the number of songs in the library."""
        return len(self._library.songs)

    def song_ids_in_order(self) -> list[str]:
        """Return every song ID in library order, whatever the view shows.

        Playback walks this list, so browsing a setlist or tag does not
        change what plays next.
        """
        return [s.id for s in self._library.songs]

    def _apply_filter(self, query: str) -> None:
        """Show only items matching *query* (case-insensitive substring)."""
        query = query.lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            haystack = item.data(_SEARCH_TEXT_ROLE)
            if not isinstance(haystack, str):
                haystack = item.text().lower()
            item.setHidden(query not in haystack)
        visible = any(
            not self._list.item(i).isHidden() for i in range(self._list.count())
        )
        empty_setlist = (
            self._filter_kind == "setlist" and self._list.count() == 0
        )
        self._no_match_label.setText(
            "This setlist is empty"
            if empty_setlist
            else "No songs match your search"
        )
        self._no_match_label.setVisible(
            empty_setlist or (bool(query) and not visible)
        )
        # Disable Remove if the current selection is hidden or gone.
        current = self._list.currentItem()
        if current is None or current.isHidden():
            self._remove_btn.setEnabled(False)
        if self._marked:
            self._sync_marks()
        self._sync_drag_reorder_mode()

    def select_song(self, song_id: str) -> bool:
        """Programmatically select a song by its ID.

        Returns True if the song was found and selected, False otherwise.
        """
        item = self._item_for_song(song_id)
        if item is None:
            return False
        if (
            self._pressed_songs is not None
            and self._pressed_songs[0] != song_id
            and item is self._list.currentItem()
        ):
            # The press in progress made this row current without announcing
            # it, so setCurrentItem would not announce it either.
            self._on_item_changed(item, None)
        else:
            self._list.setCurrentItem(item)
        return True

    def clear_selection(self) -> None:
        """Clear the current row so it can be selected again explicitly."""
        self._list.setCurrentItem(None)
        self._list.clearSelection()

    def _on_list_row_pressed(self, song_id: str) -> None:
        """Remember what was current before a press on a draggable row."""
        current = self._list.currentItem()
        self._pressed_songs = (
            current.data(Qt.ItemDataRole.UserRole)
            if current is not None
            else None,
            song_id,
        )

    def _on_list_press_released(self) -> None:
        """Load the pressed song now that the press proved to be a click."""
        pressed, self._pressed_songs = self._pressed_songs, None
        current = self._list.currentItem()
        if (
            pressed is not None
            and pressed[0] != pressed[1]
            and current is not None
            and current.data(Qt.ItemDataRole.UserRole) == pressed[1]
        ):
            self._on_item_changed(current, None)

    def _on_list_drag_started(self) -> None:
        """Make the pressed row the one Qt picks up."""
        if self._pressed_songs is not None:
            item = self._item_for_song(self._pressed_songs[1])
            if item is not None:
                self._list.set_current_silently(item)

    def _on_list_drag_finished(self) -> None:
        """Put the selection back where playback has it after a drag."""
        pressed, self._pressed_songs = self._pressed_songs, None
        if pressed is None:
            return
        previous = self._item_for_song(pressed[0]) if pressed[0] else None
        self._list.set_current_silently(previous)
        self._remove_btn.setEnabled(
            previous is not None and bool(self._library.writable)
        )

    def _on_item_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if self._list.in_press:
            return  # A click or a drag: decided on release.
        if self._pressed_songs is not None:
            # Selected from elsewhere mid-press; a drag must not undo it.
            self._pressed_songs = (
                current.data(Qt.ItemDataRole.UserRole)
                if current is not None
                else None,
                self._pressed_songs[1],
            )
        if current is not None:
            song_id = current.data(Qt.ItemDataRole.UserRole)
            if song_id and not self.is_song_separating(song_id):
                self.song_selected.emit(song_id)
                self._remove_btn.setEnabled(bool(self._library.writable))
        else:
            self._remove_btn.setEnabled(False)

    def _on_context_menu(self, pos) -> None:
        """Show a right-click context menu on the song list."""
        item = self._list.itemAt(pos)
        if item is None:
            return
        song_id = item.data(Qt.ItemDataRole.UserRole)
        if song_id and self.is_song_separating(song_id):
            # The only sensible action on a half-separated song is to
            # abort the job (the main window rolls the row back).
            menu = QMenu(self)
            menu.addAction("Cancel separation").triggered.connect(
                lambda: self.cancel_separation_requested.emit(song_id)
            )
            menu.exec(self._list.mapToGlobal(pos))
            return
        if song_id in self._marked:
            menu = QMenu(self)
            self._fill_batch_menu(menu)
            menu.exec(self._list.mapToGlobal(pos))
            return
        # Do not select the right-clicked item: right-clicking another song
        # to queue or edit it must not interrupt the currently loaded song.
        writable = bool(getattr(self._library, "writable", True))
        menu = QMenu(self)
        if song_id:
            play_next = menu.addAction("Play Next")
            play_next.triggered.connect(
                lambda _checked=False, sid=song_id: (
                    self.play_next_requested.emit(sid)
                )
            )
            add_queue = menu.addAction("Add to Queue")
            add_queue.triggered.connect(
                lambda _checked=False, sid=song_id: (
                    self.add_to_queue_requested.emit(sid)
                )
            )

            add_setlist_menu = menu.addMenu("Add to Setlist")
            add_setlist_menu.setEnabled(writable)
            setlists = self._library_setlists()
            for sl in setlists:
                act = add_setlist_menu.addAction(
                    # A lone "&" would become a mnemonic underline.
                    sl.name.replace("&", "&&")
                )
                if song_id in sl.song_ids:
                    act.setEnabled(False)
                else:
                    act.setEnabled(writable)
                    act.triggered.connect(
                        lambda _c=False, slid=sl.id, sid=song_id: (
                            self._on_add_song_to_setlist(slid, sid)
                        )
                    )
            if setlists:
                add_setlist_menu.addSeparator()
            new_sl_act = add_setlist_menu.addAction("New Setlist...")
            new_sl_act.setEnabled(writable)
            new_sl_act.triggered.connect(
                lambda _c=False, sid=song_id: self._on_new_setlist([sid])
            )

            if self._filter_kind == "setlist" and self._filter_value:
                active_sl = self._library_get_setlist(self._filter_value)
                if active_sl is not None and song_id in active_sl.song_ids:
                    idx = active_sl.song_ids.index(song_id)
                    menu.addSeparator()
                    up_act = menu.addAction("Move Up in Setlist")
                    up_act.setEnabled(writable and idx > 0)
                    up_act.triggered.connect(
                        lambda _c=False, i=idx: (
                            self._reorder_active_setlist(i, i - 1)
                        )
                    )
                    down_act = menu.addAction("Move Down in Setlist")
                    down_act.setEnabled(
                        writable and idx < len(active_sl.song_ids) - 1
                    )
                    down_act.triggered.connect(
                        lambda _c=False, i=idx: (
                            self._reorder_active_setlist(i, i + 1)
                        )
                    )
                    rem_sl_act = menu.addAction("Remove from Setlist")
                    rem_sl_act.setEnabled(writable)
                    rem_sl_act.triggered.connect(
                        lambda _c=False, slid=active_sl.id, sid=song_id: (
                            self._on_remove_song_from_setlist(slid, sid)
                        )
                    )
            menu.addSeparator()
        if song_id:
            menu.addAction("Select").triggered.connect(
                lambda _checked=False, sid=song_id: self.toggle_song_mark(sid)
            )
        edit = menu.addAction("Edit...")
        edit.setEnabled(writable)
        edit.triggered.connect(
            lambda _checked=False, sid=song_id: self._on_edit_song(sid)
        )
        menu.exec(self._list.mapToGlobal(pos))

    def _on_edit_song(self, song_id: str | None = None) -> None:
        """Open the edit dialog for *song_id* or the currently selected song."""
        if song_id is None:
            current = self._list.currentItem()
            if current is None:
                return
            song_id = current.data(Qt.ItemDataRole.UserRole)
        if not song_id:
            return
        song = self._library.get_song(song_id)
        if song is None:
            return
        if not self._library_writable("Edit Song"):
            return
        dlg = EditSongDialog(song, parent=self)
        if dlg.exec():
            updates: dict[str, object] = {
                "title": dlg.title,
                "artist": dlg.artist,
            }
            if isinstance(getattr(dlg, "tags", None), list):
                updates["tags"] = dlg.tags
            if isinstance(getattr(dlg, "practice_status", None), str):
                updates["practice_status"] = dlg.practice_status
            try:
                self._library.update_song(song_id, **updates)
            except OSError as exc:
                QMessageBox.warning(
                    self, "Edit Song",
                    describe_error(exc, "Song edit failed"),
                )
                return
            self.refresh()
            self.song_edited.emit(song_id)

    def _library_writable(self, title: str) -> bool:
        """Say why nothing can change if the library is read-only."""
        if self._library.writable:
            return True
        QMessageBox.warning(self, title, LIBRARY_READ_ONLY_MESSAGE)
        return False

    def _on_remove_clicked(self) -> None:
        song_ids = self.marked_song_ids()
        if not song_ids:
            current = self._list.currentItem()
            song_id = (
                current.data(Qt.ItemDataRole.UserRole)
                if current is not None
                else None
            )
            if not song_id:
                return
            song_ids = [song_id]
        self._remove_songs(song_ids)

    def _remove_songs(self, song_ids: list[str]) -> None:
        """Ask once, then delete every song in *song_ids*."""
        # Checked first: the removal requests below close the song in the
        # player.
        if not self._library_writable("Remove Song"):
            return
        names = [
            item.text()
            for item in map(self._item_for_song, song_ids)
            if item is not None
        ]
        if len(song_ids) == 1:
            title = "Remove Song"
            question = (
                f"Are you sure you want to remove '{names[0]}' and delete "
                "its separated audio files?"
                if names
                else "Are you sure you want to remove this song?"
            )
        else:
            title = "Remove Songs"
            listed = "\n".join(names[:8])
            if len(names) > 8:
                listed += f"\nand {len(names) - 8} more"
            question = (
                f"Are you sure you want to remove these {len(song_ids)} "
                "songs and delete their separated audio files?\n\n"
                f"{listed}"
            )
        reply = QMessageBox.question(
            self,
            title,
            question,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        removed: list[str] = []
        failures: list[str] = []
        for song_id in song_ids:
            # A song removed, or sent to be separated, while the question
            # was open is left alone.
            if self.is_song_separating(song_id):
                continue
            # MainWindow must stop/drain readers before remove_song
            # deletes the directory they may still be reading.
            self.song_removal_requested.emit(song_id)
            try:
                self._library.remove_song(song_id)
            except KeyError:
                continue
            except (OSError, ValueError) as exc:
                failures.append(describe_error(exc, "Song removal failed"))
                continue
            removed.append(song_id)
        self.refresh()
        # Let the main window unload the player if the removed song is the
        # one currently loaded -- otherwise it keeps playing from RAM and
        # stopping would try to save recordings into the deleted directory.
        for song_id in removed:
            self.song_removed.emit(song_id)
        if failures:
            # The same cause usually repeats; say each one once.
            QMessageBox.warning(
                self, title, "\n\n".join(dict.fromkeys(failures)),
            )

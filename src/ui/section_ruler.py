"""Experimental song-section ruler above the stacked stem waveform (#133).

One row of blocks, one per detected section, on the stem lanes' own time
axis, labelled with letters that repeat for sections that sound alike.
Click a section to jump to its start; double-click it to loop it. Its
bounds are bar lines, so the loop keeps the song's time.

The blocks and letters are cached in a pixmap; each frame adds only the
current section's highlight, the looped section's outline, and the
playhead, and frames come from the existing 20 Hz position updates.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from src.song_sections import section_index_at
from src.ui.styles import DARK_COLORS
from src.ui.waveform_stack_widget import (
    _LABEL_PADDING,
    _LABEL_WIDTH,
    format_loop_time,
)

SECTION_RULER_HEIGHT = 24
_PAD = 2
_LABEL_SLACK = 4
_PLACEHOLDER_INSET = 6
# Loop points come back from the player in whole frames: a loop within
# this of a section's bounds is that section.
_LOOP_TOLERANCE_S = 0.01

PLACEHOLDER_DETECTING = "Finding sections..."
PLACEHOLDER_NONE = "No clear sections found"
PLACEHOLDER_FAILED = "Section detection failed"
_GUTTER_TITLE = "Sections"
_GUTTER_TAG = "beta"
_HELP = (
    "Song sections (experimental): guessed from the harmony, sound, and "
    "loudness of each bar. Sections with the same letter sound alike."
)


def _font(px: int, bold: bool = False) -> QFont:
    font = QFont()
    font.setPixelSize(px)
    font.setBold(bold)
    return font


class SectionRulerWidget(QWidget):
    """The detected song sections in time, above the stem lanes."""

    seek_requested = Signal(float)  # Emits seconds
    loop_requested = Signal(float, float)  # Emits (start_s, end_s)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sections: list[tuple[float, float, str]] = []
        self._total = 0.0
        self._position = 0.0
        self._loop: tuple[float, float] | None = None
        self._detecting = False
        self._failed = False
        self._frame_key: tuple | None = None
        self._blocks_cache: QPixmap | None = None
        self._blocks_cache_key: tuple | None = None
        self._label_metrics = QFontMetrics(_font(10, bold=True))
        self._apply_colors(DARK_COLORS)

        self.setFixedHeight(SECTION_RULER_HEIGHT)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.setMouseTracking(True)
        self.setAccessibleName("Song sections (experimental)")
        self.setToolTip(_HELP)

    # -- State --

    def _apply_colors(self, colors: dict[str, str]) -> None:
        self._bg = QColor(colors["base"])
        self._shades = (QColor(colors["surface0"]), QColor(colors["surface1"]))
        self._text = QColor(colors["text"])
        self._subtext = QColor(colors["subtext"])
        self._accent = QColor(colors["accent"])
        self._on_accent = QColor(colors["on_accent"])
        self._loop_color = QColor(colors["red"])
        self._colors_key = tuple(sorted(colors.items()))

    def set_theme_colors(self, colors: dict[str, str]) -> None:
        """Repaint in a new theme's colours."""
        self._apply_colors(colors)
        self._invalidate()

    def set_sections(self, sections) -> None:
        """Show the song's sections as ``(start_s, end_s, label)``."""
        self._sections = sorted(
            (float(s), float(e), str(label)) for s, e, label in sections
        )
        self._failed = False
        self._update_accessible_description()
        self._invalidate()

    def sections(self) -> list[tuple[float, float, str]]:
        return list(self._sections)

    def set_total_seconds(self, total: float) -> None:
        self._total = max(0.0, float(total))
        self._invalidate()

    def set_loop(self, a: float | None, b: float | None) -> None:
        """The A-B loop, to mark a section that is being looped."""
        loop = (float(a), float(b)) if a is not None and b is not None else None
        if loop != self._loop:
            self._loop = loop
            self._frame_key = None
            self.update()

    def set_detecting(self, detecting: bool) -> None:
        """Detection running: a quiet placeholder until sections arrive."""
        if detecting:
            self._failed = False
        if detecting != self._detecting:
            self._detecting = detecting
            self.update()

    def set_detection_failed(self) -> None:
        """Detection of the whole song failed: no sections, and say why."""
        self._sections = []
        self._detecting = False
        self._failed = True
        self._update_accessible_description()
        self._invalidate()

    def clear(self) -> None:
        """Forget the song."""
        self._sections = []
        self._total = 0.0
        self._position = 0.0
        self._loop = None
        self._detecting = False
        self._failed = False
        self._update_accessible_description()
        self._invalidate()

    def set_position(self, seconds: float) -> None:
        """Move the playhead; repaints only when something visible moved."""
        self._position = max(0.0, float(seconds))
        key = self._compute_frame_key()
        if key == self._frame_key:
            return
        index_changed = (
            self._frame_key is None or key[0] != self._frame_key[0]
        )
        self._frame_key = key
        if index_changed:
            self._update_accessible_description()
        self.update()

    def _compute_frame_key(self) -> tuple:
        """What a frame shows: the current section and the playhead pixel."""
        return (self.current_index(), round(self.x_for(self._position)))

    def _invalidate(self) -> None:
        self._blocks_cache = None
        self._blocks_cache_key = None
        self._frame_key = None
        self.update()

    def _update_accessible_description(self) -> None:
        index = self.current_index()
        text = "" if index is None else f"Section {self._sections[index][2]}"
        if self.accessibleDescription() != text:
            self.setAccessibleDescription(text)

    # -- Queries --

    def current_index(self) -> int | None:
        """Index of the section under the playhead, or None outside them."""
        return section_index_at(self._sections, self._position)

    def looped_index(self) -> int | None:
        """Index of the section the A-B loop covers exactly, if any."""
        if self._loop is None:
            return None
        a, b = self._loop
        for i, (start, end, _) in enumerate(self._sections):
            if (
                abs(start - a) <= _LOOP_TOLERANCE_S
                and abs(end - b) <= _LOOP_TOLERANCE_S
            ):
                return i
        return None

    def placeholder_text(self) -> str:
        """The quiet note shown instead of sections, or "" when none."""
        if self._sections or self._total <= 0:
            return ""
        if self._detecting:
            return PLACEHOLDER_DETECTING
        return PLACEHOLDER_FAILED if self._failed else PLACEHOLDER_NONE

    def gutter_texts(self) -> tuple[str, str]:
        """The gutter's title and its experimental tag."""
        return _GUTTER_TITLE, _GUTTER_TAG

    def _span(self) -> float:
        return max(1.0, float(self.width() - _LABEL_WIDTH))

    def x_for(self, seconds: float) -> float:
        """x of *seconds* on the stem lanes' axis (whole song)."""
        if self._total <= 0:
            return float(_LABEL_WIDTH)
        return _LABEL_WIDTH + seconds / self._total * self._span()

    def blocks(self) -> list[tuple[float, float, str]]:
        """(left, right, label) of each section."""
        return [
            (self.x_for(s), self.x_for(e), label)
            for s, e, label in self._sections
        ]

    def _block_rect(self, left: float, right: float) -> QRectF:
        rect = QRectF(
            left, float(_PAD), right - left,
            float(SECTION_RULER_HEIGHT - 2 * _PAD),
        )
        # One pixel short so neighbouring sections read as separate.
        return rect.adjusted(0, 0, -1, 0) if rect.width() > 2 else rect

    def _index_at(self, x: float) -> int | None:
        if (
            x < _LABEL_WIDTH or self._total <= 0 or not self._sections
            or self.placeholder_text()
        ):
            return None
        seconds = (x - _LABEL_WIDTH) / self._span() * self._total
        return section_index_at(self._sections, seconds)

    def _shade(self, label: str) -> QColor:
        """Alternate two shades by letter, so A and B read apart."""
        return self._shades[(ord(label[:1] or "A") - ord("A")) % 2]

    # -- Size --

    def sizeHint(self) -> QSize:
        return QSize(600, SECTION_RULER_HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return QSize(200, SECTION_RULER_HEIGHT)

    # -- Paint --

    def resizeEvent(self, event) -> None:
        self._frame_key = None
        super().resizeEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.fillRect(self.rect(), self._bg)
            if self._total <= 0:
                return
            self._draw_gutter(painter)
            placeholder = self.placeholder_text()
            if placeholder:
                self._draw_placeholder(painter, placeholder)
                return
            if not self._sections:
                return
            painter.drawPixmap(
                int(_LABEL_WIDTH), 0, self._blocks_pixmap(),
            )
            self._draw_current(painter)
            self._draw_loop_mark(painter)
            self._draw_playhead(painter)
        finally:
            painter.end()

    def _draw_gutter(self, painter: QPainter) -> None:
        width = float(_LABEL_WIDTH - 2 * _LABEL_PADDING)
        half = SECTION_RULER_HEIGHT / 2
        painter.setFont(_font(10))
        painter.setPen(self._text)
        painter.drawText(
            QRectF(_LABEL_PADDING, 0.0, width, half + 1),
            int(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft),
            _GUTTER_TITLE,
        )
        painter.setFont(_font(9))
        painter.setPen(self._subtext)
        painter.drawText(
            QRectF(_LABEL_PADDING, half - 1, width, half),
            int(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft),
            _GUTTER_TAG,
        )

    def _draw_placeholder(self, painter: QPainter, text: str) -> None:
        painter.setFont(_font(10))
        painter.setPen(self._subtext)
        painter.drawText(
            QRectF(
                float(_LABEL_WIDTH + _PLACEHOLDER_INSET), 0.0,
                self._span(), float(self.height()),
            ),
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            text,
        )

    def _draw_block(
        self, painter: QPainter, left: float, right: float, label: str,
        fill: QColor, ink: QColor,
    ) -> None:
        rect = self._block_rect(left, right)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, 3.0, 3.0)
        width = self._label_metrics.horizontalAdvance(label)
        if width + _LABEL_SLACK <= rect.width():
            painter.setFont(_font(10, bold=True))
            painter.setPen(ink)
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), label)

    def _blocks_pixmap(self) -> QPixmap:
        """Every section's block and letter, rebuilt only on change."""
        dpr = self.devicePixelRatioF()
        key = (
            self.width(), dpr, self._total, tuple(self._sections),
            self._colors_key,
        )
        if self._blocks_cache is not None and key == self._blocks_cache_key:
            return self._blocks_cache
        pixmap = QPixmap(
            max(1, round(self._span() * dpr)),
            max(1, round(SECTION_RULER_HEIGHT * dpr)),
        )
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.GlobalColor.transparent)
        p = QPainter(pixmap)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.translate(-float(_LABEL_WIDTH), 0.0)
            for left, right, label in self.blocks():
                self._draw_block(
                    p, left, right, label, self._shade(label), self._text,
                )
        finally:
            p.end()
        self._blocks_cache = pixmap
        self._blocks_cache_key = key
        return pixmap

    def _draw_current(self, painter: QPainter) -> None:
        index = self.current_index()
        if index is None:
            return
        left, right, label = self.blocks()[index]
        self._draw_block(
            painter, left, right, label, self._accent, self._on_accent,
        )

    def _draw_loop_mark(self, painter: QPainter) -> None:
        index = self.looped_index()
        if index is None:
            return
        left, right, _ = self.blocks()[index]
        rect = self._block_rect(left, right).adjusted(1.0, 1.0, -1.0, -1.0)
        painter.setPen(QPen(self._loop_color, 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, 3.0, 3.0)

    def _draw_playhead(self, painter: QPainter) -> None:
        x = min(self.x_for(self._position), float(self.width()) - 1.0)
        over_block = self.current_index() is not None
        pen = QPen(self._on_accent if over_block else self._text, 2.0)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.drawLine(x, float(_PAD), x, float(SECTION_RULER_HEIGHT - _PAD))

    # -- Mouse --

    def _section_tooltip(self, index: int) -> str:
        start, end, label = self._sections[index]
        return (
            f"Section {label}, {format_loop_time(start)}"
            f"–{format_loop_time(end)}\n"
            "Click to jump to it, double-click to loop it (Shift+L loops "
            "the section playing).\n"
            "Experimental: sections with the same letter sound alike."
        )

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        index = self._index_at(event.position().x())
        shape = (
            Qt.CursorShape.PointingHandCursor if index is not None
            else Qt.CursorShape.ArrowCursor
        )
        if self.cursor().shape() != shape:
            self.setCursor(shape)
        tip = _HELP if index is None else self._section_tooltip(index)
        if self.toolTip() != tip:
            self.setToolTip(tip)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        index = self._index_at(event.position().x())
        if index is not None:
            self.seek_requested.emit(self._sections[index][0])

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        index = self._index_at(event.position().x())
        if index is not None:
            start, end, _ = self._sections[index]
            self.loop_requested.emit(start, end)

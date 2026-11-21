"""Chord lane under the stacked stem waveform (#133).

Two rows share the stem lanes' label gutter:

* The overview row is the whole song on the stem lanes' own time axis,
  so a chord change lines up with the waveform above it. Labels show
  where a block is wide enough and are dropped where it is not.
* The read-ahead row is the bar being played and the next seven, drawn
  larger and scrolling with the playhead, so a player glancing up while
  playing can read what comes next. Over a whole song even a wide window
  gives one bar a dozen pixels, too small to read at a glance.

The overview's blocks and labels are cached in a pixmap; each frame only
adds the highlight, the playhead, and the read-ahead row, and frames come
from the existing position updates, never from the audio callback.
"""

from __future__ import annotations

import bisect
import statistics

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

from src.beat_detector import transpose_chord
from src.ui.styles import DARK_COLORS
from src.ui.waveform_stack_widget import _LABEL_PADDING, _LABEL_WIDTH

_PAD = 1
_OVERVIEW_HEIGHT = 16
_ROW_GAP = 4
_READ_AHEAD_HEIGHT = 28
CHORD_LANE_HEIGHT = (
    _PAD + _OVERVIEW_HEIGHT + _ROW_GAP + _READ_AHEAD_HEIGHT + _PAD
)

# The read-ahead row shows this many bars: the one being played and the
# next seven, with the playhead one bar in from the left.
READ_AHEAD_BARS = 8
_BARS_BEHIND = 1
# Without downbeats (the librosa fallback finds none) the window is a fixed
# span; with them it is still kept readable for very slow or fast songs.
_FALLBACK_WINDOW_S = 12.0
_MIN_WINDOW_S = 6.0
_MAX_WINDOW_S = 24.0
# Room a label needs beside its text before it is drawn in a block.
_LABEL_SLACK = 4
_READ_AHEAD_LABEL_INSET = 6
_PAST_OPACITY = 0.45

PLACEHOLDER_DETECTING = "Finding chords..."
PLACEHOLDER_NONE = "No chords detected"


def _small_font() -> QFont:
    font = QFont()
    font.setPixelSize(10)
    return font


def _large_font() -> QFont:
    font = QFont()
    font.setPixelSize(13)
    font.setBold(True)
    return font


def splice_chords(
    chords: list[tuple[float, str]],
    region: list[tuple[float, str]],
    start: float,
    end: float,
) -> list[tuple[float, str]]:
    """Put a loop re-detection's chords into the whole song's.

    Detection re-runs on the A-B region when a loop is set, and its result
    covers only the loop. Inside ``[start, end)`` the region's chords
    replace the song's; outside, the song's stay, and the chord that was
    playing at *end* resumes there. Nothing detected leaves *chords* as
    they were. Consecutive repeats of one chord are merged.
    """
    if not region:
        return list(chords)
    merged = [(t, c) for t, c in chords if t < start]
    merged.extend((t, c) for t, c in region if start <= t < end)
    onsets = [t for t, _ in chords]
    resume = bisect.bisect_right(onsets, end) - 1
    if resume >= 0 and chords[resume][0] < end:
        merged.append((end, chords[resume][1]))
    merged.extend((t, c) for t, c in chords if t >= end)
    result: list[tuple[float, str]] = []
    for time, chord in merged:
        if result and result[-1][1] == chord:
            continue
        result.append((time, chord))
    return result


class ChordLaneWidget(QWidget):
    """The detected chords in time, under the stem lanes."""

    seek_requested = Signal(float)  # Emits seconds

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._chords: list[tuple[float, str]] = []
        self._onsets: list[float] = []
        self._labels: list[str] = []
        self._downbeats: list[float] = []
        self._pitch = 0
        self._total = 0.0
        self._position = 0.0
        self._detecting = False
        self._frame_key: tuple | None = None
        self._overview_cache: QPixmap | None = None
        self._overview_cache_key: tuple | None = None
        self._small_metrics = QFontMetrics(_small_font())
        self._large_metrics = QFontMetrics(_large_font())
        self._apply_colors(DARK_COLORS)

        self.setFixedHeight(CHORD_LANE_HEIGHT)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName("Chord lane")
        self.setToolTip("Detected chords (suggestions). Click one to jump to it.")

    # -- State --

    def _apply_colors(self, colors: dict[str, str]) -> None:
        self._bg = QColor(colors["base"])
        self._block = QColor(colors["surface0"])
        self._text = QColor(colors["text"])
        self._subtext = QColor(colors["subtext"])
        self._accent = QColor(colors["accent"])
        self._on_accent = QColor(colors["on_accent"])
        self._bar_line = QColor(colors["subtext"])
        self._bar_line.setAlpha(110)
        self._window_outline = QColor(colors["subtext"])
        self._window_outline.setAlpha(170)
        self._colors_key = tuple(sorted(colors.items()))

    def set_theme_colors(self, colors: dict[str, str]) -> None:
        """Repaint in a new theme's colours."""
        self._apply_colors(colors)
        self._invalidate()

    def set_chords(self, chords: list[tuple[float, str]]) -> None:
        """Show a whole-song chord sequence of ``(onset_s, label)``."""
        self._chords = sorted((float(t), str(c)) for t, c in chords)
        self._onsets = [t for t, _ in self._chords]
        self._relabel()

    def splice_region(
        self, region: list[tuple[float, str]], start: float, end: float,
    ) -> None:
        """Merge a loop re-detection into the shown chords."""
        self.set_chords(splice_chords(self._chords, region, start, end))

    def set_downbeats(self, downbeats: list[float]) -> None:
        """Set the bar lines (and the read-ahead window's bar length)."""
        self._downbeats = sorted(float(t) for t in downbeats)
        self._invalidate()

    def downbeats(self) -> list[float]:
        return list(self._downbeats)

    def set_pitch(self, semitones: int) -> None:
        """Re-spell the labels for a transposition, as the badge does."""
        semitones = int(semitones)
        if semitones == self._pitch:
            return
        self._pitch = semitones
        self._relabel()

    def set_total_seconds(self, total: float) -> None:
        self._total = max(0.0, float(total))
        self._invalidate()

    def set_detecting(self, detecting: bool) -> None:
        """Detection running: a quiet placeholder until chords arrive."""
        if detecting != self._detecting:
            self._detecting = detecting
            self.update()

    def clear(self) -> None:
        """Forget the song."""
        self._chords = []
        self._onsets = []
        self._labels = []
        self._downbeats = []
        self._total = 0.0
        self._position = 0.0
        self._detecting = False
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
        """What a frame shows: the current chord and both playhead pixels."""
        start, end = self.read_ahead_window()
        span = self._span()
        scroll = round(self._position * span / max(end - start, 1e-6))
        overview = (
            round(self._position / self._total * span) if self._total else 0
        )
        return (self.current_index(), scroll, overview)

    def _relabel(self) -> None:
        self._labels = [transpose_chord(c, self._pitch) for _, c in self._chords]
        self._update_accessible_description()
        self._invalidate()

    def _invalidate(self) -> None:
        self._overview_cache = None
        self._overview_cache_key = None
        self._frame_key = None
        self.update()

    def _update_accessible_description(self) -> None:
        index = self.current_index()
        if index is None:
            text = ""
        else:
            text = f"Current chord {self._labels[index]}"
            if index + 1 < len(self._labels):
                text += f", next {self._labels[index + 1]}"
        if self.accessibleDescription() != text:
            self.setAccessibleDescription(text)

    # -- Queries --

    def labels(self) -> list[str]:
        """The chord labels as shown, transposition applied."""
        return list(self._labels)

    def current_index(self) -> int | None:
        """Index of the chord under the playhead, or None before the first."""
        index = bisect.bisect_right(self._onsets, self._position) - 1
        return index if index >= 0 else None

    def placeholder_text(self) -> str:
        """The quiet note shown instead of chords, or "" when none."""
        if self._chords or self._total <= 0:
            return ""
        return PLACEHOLDER_DETECTING if self._detecting else PLACEHOLDER_NONE

    def _span(self) -> float:
        return max(1.0, float(self.width() - _LABEL_WIDTH))

    def overview_rect(self) -> QRectF:
        return QRectF(
            float(_LABEL_WIDTH), float(_PAD), self._span(),
            float(_OVERVIEW_HEIGHT),
        )

    def read_ahead_rect(self) -> QRectF:
        top = _PAD + _OVERVIEW_HEIGHT + _ROW_GAP
        return QRectF(
            float(_LABEL_WIDTH), float(top), self._span(),
            float(_READ_AHEAD_HEIGHT),
        )

    def _segments(self) -> list[tuple[float, float, int]]:
        """(start_s, end_s, index) per chord; each runs to the next."""
        if not self._chords:
            return []
        last_end = max(self._total, self._onsets[-1] + 1.0)
        ends = self._onsets[1:] + [last_end]
        return [
            (start, end, i)
            for i, (start, end) in enumerate(zip(self._onsets, ends))
        ]

    def overview_x(self, seconds: float) -> float:
        """x of *seconds* on the stem lanes' axis (whole song)."""
        if self._total <= 0:
            return float(_LABEL_WIDTH)
        return _LABEL_WIDTH + seconds / self._total * self._span()

    def overview_blocks(self) -> list[tuple[float, float, str]]:
        """(left, right, label) of each chord on the overview row."""
        return [
            (self.overview_x(s), self.overview_x(e), self._labels[i])
            for s, e, i in self._segments()
        ]

    def overview_label_texts(self) -> list[str]:
        """Labels the overview draws: those whose block fits them."""
        return [
            label for left, right, label in self.overview_blocks()
            if self._fits(label, right - left, self._small_metrics)
        ]

    @staticmethod
    def _fits(label: str, width: float, metrics: QFontMetrics) -> bool:
        return metrics.horizontalAdvance(label) + _LABEL_SLACK <= width

    def bar_seconds(self) -> float | None:
        """Median bar length from the downbeats, or None without them."""
        gaps = [
            b - a for a, b in zip(self._downbeats, self._downbeats[1:])
            if b > a
        ]
        return statistics.median(gaps) if gaps else None

    def read_ahead_window(self) -> tuple[float, float]:
        """(start_s, end_s) shown in the read-ahead row."""
        bar = self.bar_seconds()
        if bar is None:
            window = _FALLBACK_WINDOW_S
        else:
            window = min(_MAX_WINDOW_S, max(_MIN_WINDOW_S, READ_AHEAD_BARS * bar))
        start = self._position - window * _BARS_BEHIND / READ_AHEAD_BARS
        return start, start + window

    def read_ahead_x(self, seconds: float) -> float:
        start, end = self.read_ahead_window()
        return _LABEL_WIDTH + (seconds - start) / (end - start) * self._span()

    def playhead_x(self) -> float:
        """x of the playhead in the read-ahead row."""
        return self.read_ahead_x(self._position)

    def read_ahead_blocks(self) -> list[tuple[float, float, str]]:
        """(left, right, label) of the chords in the read-ahead window."""
        start, end = self.read_ahead_window()
        return [
            (self.read_ahead_x(s), self.read_ahead_x(e), self._labels[i])
            for s, e, i in self._segments()
            if e > start and s < end
        ]

    # -- Size --

    def sizeHint(self) -> QSize:
        return QSize(600, CHORD_LANE_HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return QSize(200, CHORD_LANE_HEIGHT)

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
            self._draw_gutter_label(painter)
            placeholder = self.placeholder_text()
            if placeholder:
                self._draw_placeholder(painter, placeholder)
                return
            if not self._chords:
                return
            self._draw_overview(painter)
            self._draw_read_ahead(painter)
        finally:
            painter.end()

    def _draw_gutter_label(self, painter: QPainter) -> None:
        painter.setFont(_small_font())
        painter.setPen(self._text)
        painter.drawText(
            QRectF(
                _LABEL_PADDING, 0.0,
                float(_LABEL_WIDTH - 2 * _LABEL_PADDING),
                float(self.height()),
            ),
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            "Chords",
        )

    def _draw_placeholder(self, painter: QPainter, text: str) -> None:
        painter.setFont(_small_font())
        painter.setPen(self._subtext)
        rect = QRectF(
            float(_LABEL_WIDTH + _READ_AHEAD_LABEL_INSET), 0.0,
            self._span(), float(self.height()),
        )
        painter.drawText(
            rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            text,
        )

    def _overview_pixmap(self) -> QPixmap:
        """The overview row's blocks and labels, rebuilt only on change."""
        rect = self.overview_rect()
        dpr = self.devicePixelRatioF()
        key = (
            self.width(), dpr, self._total, self._pitch,
            tuple(self._chords), self._colors_key,
        )
        if self._overview_cache is not None and key == self._overview_cache_key:
            return self._overview_cache
        pixmap = QPixmap(
            max(1, round(rect.width() * dpr)),
            max(1, round(rect.height() * dpr)),
        )
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.GlobalColor.transparent)
        p = QPainter(pixmap)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.translate(-rect.left(), -rect.top())
            p.setFont(_small_font())
            for left, right, label in self.overview_blocks():
                self._draw_block(
                    p, QRectF(left, rect.top(), right - left, rect.height()),
                    self._block, 2.0,
                )
                if self._fits(label, right - left, self._small_metrics):
                    p.setPen(self._text)
                    p.drawText(
                        QRectF(left, rect.top(), right - left, rect.height()),
                        int(Qt.AlignmentFlag.AlignCenter), label,
                    )
        finally:
            p.end()
        self._overview_cache = pixmap
        self._overview_cache_key = key
        return pixmap

    @staticmethod
    def _draw_block(
        painter: QPainter, rect: QRectF, color: QColor, radius: float,
    ) -> None:
        """A chord block, one pixel short so neighbours read as separate."""
        if rect.width() > 2:
            rect = rect.adjusted(0, 0, -1, 0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(rect, radius, radius)

    def _draw_overview(self, painter: QPainter) -> None:
        rect = self.overview_rect()
        painter.drawPixmap(rect.topLeft(), self._overview_pixmap())

        index = self.current_index()
        if index is not None:
            left, right, label = self.overview_blocks()[index]
            block = QRectF(left, rect.top(), right - left, rect.height())
            self._draw_block(painter, block, self._accent, 2.0)
            if self._fits(label, right - left, self._small_metrics):
                painter.setFont(_small_font())
                painter.setPen(self._on_accent)
                painter.drawText(
                    block, int(Qt.AlignmentFlag.AlignCenter), label,
                )

        # Where the read-ahead row is in the song.
        start, end = self.read_ahead_window()
        x0 = max(rect.left(), self.overview_x(start))
        x1 = min(rect.right(), self.overview_x(end))
        painter.setPen(QPen(self._window_outline, 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(
            QRectF(x0, rect.top(), max(1.0, x1 - x0), rect.height()).adjusted(
                0.5, 0.5, -0.5, -0.5,
            ),
            2.0, 2.0,
        )

        x = min(self.overview_x(self._position), rect.right() - 1.0)
        pen = QPen(self._playhead_color(index), 2.0)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.drawLine(x, rect.top(), x, rect.bottom())

    def _draw_read_ahead(self, painter: QPainter) -> None:
        rect = self.read_ahead_rect()
        painter.save()
        painter.setClipRect(rect)
        index = self.current_index()
        blocks = []
        for start, end, i in self._segments():
            left = self.read_ahead_x(start)
            right = self.read_ahead_x(end)
            if right <= rect.left() or left >= rect.right():
                continue
            past = end <= self._position
            blocks.append((left, right, i, past))
            painter.setOpacity(_PAST_OPACITY if past else 1.0)
            self._draw_block(
                painter,
                QRectF(left, rect.top(), right - left, rect.height()),
                self._accent if i == index else self._block,
                3.0,
            )
        painter.setOpacity(1.0)

        # Subtle bar lines from the detected downbeats.
        window_start, window_end = self.read_ahead_window()
        painter.setPen(QPen(self._bar_line, 1.0))
        lo = bisect.bisect_left(self._downbeats, window_start)
        hi = bisect.bisect_right(self._downbeats, window_end)
        for downbeat in self._downbeats[lo:hi]:
            x = round(self.read_ahead_x(downbeat)) + 0.5
            painter.drawLine(x, rect.top() + 3, x, rect.bottom() - 3)

        # Labels stay in view while their block scrolls off to the left.
        painter.setFont(_large_font())
        inset = _READ_AHEAD_LABEL_INSET
        for left, right, i, past in blocks:
            label = self._labels[i]
            width = self._large_metrics.horizontalAdvance(label)
            visible_left = max(left, rect.left())
            if right - visible_left < width + inset + _LABEL_SLACK:
                continue
            painter.setOpacity(_PAST_OPACITY if past else 1.0)
            painter.setPen(self._on_accent if i == index else self._text)
            painter.drawText(
                QRectF(visible_left + inset, rect.top(), width + 2,
                       rect.height()),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                label,
            )
        painter.setOpacity(1.0)

        x = self.playhead_x()
        pen = QPen(self._playhead_color(index), 2.0)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.drawLine(x, rect.top(), x, rect.bottom())
        painter.restore()

    def _playhead_color(self, index: int | None) -> QColor:
        """The playhead sits on the accent block: draw it in that block's
        foreground, or the light theme's dark line vanishes on dark teal."""
        return self._on_accent if index is not None else self._text

    # -- Mouse --

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x()
        if x < _LABEL_WIDTH or self._total <= 0:
            return
        span = self._span()
        if event.position().y() < self.read_ahead_rect().top():
            seconds = (x - _LABEL_WIDTH) / span * self._total
        else:
            start, end = self.read_ahead_window()
            seconds = start + (x - _LABEL_WIDTH) / span * (end - start)
        seconds = max(0.0, min(self._total, seconds))
        index = bisect.bisect_right(self._onsets, seconds) - 1
        if index >= 0:
            seconds = self._onsets[index]
        self.seek_requested.emit(seconds)

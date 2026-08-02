"""Stacked multi-lane waveform widget for practice cockpit.

Each stem gets an equal vertical slice with colored bars, a small label on
the left, and shared playback cursor / loop markers across the full height.
Supports click-to-seek and drag-to-seek like WaveformWidget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt, Signal, QSize, QPointF, QRectF, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import QWidget, QSizePolicy

from src.ui.styles import DARK_COLORS

STACK_HEIGHT = 280
# The window may be as short as 600px, which cannot afford 280px of waveform
# on top of the transport, practice controls, and mixer. Below this floor the
# lanes stop being readable, so the stack shrinks to it and no further.
STACK_MIN_HEIGHT = 120
_BAR_WIDTH = 2
_BAR_GAP = 1
_BAR_STEP = _BAR_WIDTH + _BAR_GAP
_BAR_RADIUS = 1.0
_CURSOR_GLOW_WIDTH = 6
_LABEL_WIDTH = 52
_LABEL_PADDING = 4
_MUTED_LANE_OPACITY = 0.15
# A dimmed lane needs more opacity to stay legible against a light background;
# at 0.15 on white it disappears entirely and reads as "stem has no audio".
_MUTED_LANE_OPACITY_LIGHT = 0.3


@dataclass
class _StemLane:
    name: str
    peaks: np.ndarray
    color: str
    max_peak: float = 0.0
    cached_size: tuple[int, int] = (-1, -1)
    cached_path: QPainterPath | None = None


class WaveformStackWidget(QWidget):
    """Displays stacked stem waveforms with shared cursor and loop markers."""

    seek_requested = Signal(float)  # Emits seconds

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._lanes: list[_StemLane] = []
        self._muted: set[str] = set()
        self._soloed: set[str] = set()
        self._position_ratio: float = 0.0
        self._loop_a_ratio: float | None = None
        self._loop_b_ratio: float | None = None
        self._total_seconds: float = 0.0
        self._seeking: bool = False
        self._loading: bool = False
        self._shimmer_phase: float = 0.0

        self._shimmer_timer = QTimer(self)
        self._shimmer_timer.setInterval(30)  # ~33fps
        self._shimmer_timer.timeout.connect(self._tick_shimmer)

        self._apply_colors(DARK_COLORS)

        self.setMinimumHeight(STACK_MIN_HEIGHT)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.setMouseTracking(False)

    def _apply_colors(self, colors: dict[str, str]) -> None:
        self._bg_color = QColor(colors["base"])
        self._muted_opacity = (
            _MUTED_LANE_OPACITY_LIGHT
            if self._bg_color.lightness() > 127
            else _MUTED_LANE_OPACITY
        )
        self._cursor_color = QColor(colors["text"])
        self._loop_marker_color = QColor(colors["red"])
        self._label_color = QColor(colors["text"])
        accent = QColor(colors["accent"])
        self._accent_color = accent
        self._loop_region_color = QColor(
            accent.red(), accent.green(), accent.blue(), 38
        )
        self._cursor_glow_color = QColor(
            self._cursor_color.red(),
            self._cursor_color.green(),
            self._cursor_color.blue(),
            50,
        )

    def set_theme_colors(self, colors: dict[str, str]) -> None:
        """Update paint colors for a new theme and repaint."""
        self._apply_colors(colors)
        self._invalidate_lane_paths()
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(600, STACK_HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return QSize(200, STACK_MIN_HEIGHT)

    def lane_count(self) -> int:
        return len(self._lanes)

    def lane_opacity(self, stem_name: str) -> float:
        """Return paint opacity for *stem_name* based on mute/solo state."""
        if self._soloed:
            if stem_name in self._soloed:
                return 1.0
            return self._muted_opacity
        if stem_name in self._muted:
            return self._muted_opacity
        return 1.0

    def set_stem_lanes(
        self,
        stems: list[tuple[str, np.ndarray, str]],
        *,
        muted: set[str],
        soloed: set[str],
    ) -> None:
        """Set stem lanes as (name, peaks, color_hex) tuples."""
        self._lanes = []
        for name, peaks, color in stems:
            max_peak = float(np.max(peaks)) if len(peaks) > 0 else 0.0
            self._lanes.append(
                _StemLane(name=name, peaks=peaks, color=color, max_peak=max_peak)
            )
        self._muted = muted
        self._soloed = soloed
        if self._loading and self._lanes:
            self.set_loading(False)
        self.update()

    def update_lane_mix(
        self,
        *,
        muted: set[str],
        soloed: set[str],
    ) -> None:
        """Refresh mute/solo opacities without replacing peak data."""
        if muted == self._muted and soloed == self._soloed:
            return
        self._muted = muted
        self._soloed = soloed
        self.update()

    def set_loading(self, loading: bool) -> None:
        """Show or hide a shimmer animation while peaks are being computed."""
        if loading == self._loading:
            return
        self._loading = loading
        if loading:
            self._shimmer_phase = 0.0
            self._shimmer_timer.start()
        else:
            self._shimmer_timer.stop()
        self.update()

    def _tick_shimmer(self) -> None:
        """Advance the shimmer phase and repaint."""
        self._shimmer_phase = (self._shimmer_phase + 0.015) % 1.0
        self.update()

    def set_position(self, ratio: float) -> None:
        """Update the playback cursor position (0.0 to 1.0)."""
        new_ratio = max(0.0, min(1.0, ratio))
        if new_ratio == self._position_ratio:
            return
        old_px = int(self._x_for_ratio(self._position_ratio, self.width()))
        new_px = int(self._x_for_ratio(new_ratio, self.width()))
        self._position_ratio = new_ratio
        if not self._seeking and old_px != new_px:
            self.update()

    def set_loop_markers(
        self, a_ratio: float | None, b_ratio: float | None
    ) -> None:
        """Set loop marker positions as ratios (0.0 to 1.0), or None to clear."""
        self._loop_a_ratio = a_ratio
        self._loop_b_ratio = b_ratio
        self.update()

    def set_total_seconds(self, total: float) -> None:
        """Set the total duration for click-to-seek conversion."""
        self._total_seconds = total

    def _invalidate_lane_paths(self) -> None:
        for lane in self._lanes:
            lane.cached_size = (0, 0)
            lane.cached_path = None

    def _lane_rect(self, index: int, w: int, h: int) -> tuple[int, int, int, int]:
        """Return (x, y, width, height) for lane *index* waveform area."""
        count = max(len(self._lanes), 1)
        # Spread the integer remainder over the first lanes so the stack fills
        # its full height instead of leaving a dead strip at the bottom.
        base_h, extra = divmod(h, count)
        y = index * base_h + min(index, extra)
        lane_h = base_h + (1 if index < extra else 0)
        return _LABEL_WIDTH, y, w - _LABEL_WIDTH, lane_h

    # -- Time/pixel mapping --
    #
    # Lane waveforms are drawn inset by the label gutter, so the playhead,
    # loop markers, and seek handling must map ratios onto that same inset
    # span rather than onto the full widget width.

    def _plot_span(self, w: int) -> tuple[float, float]:
        """Return (x0, span) of the shared waveform plotting area."""
        return float(_LABEL_WIDTH), max(1.0, float(w - _LABEL_WIDTH))

    def _x_for_ratio(self, ratio: float, w: int) -> float:
        """Return the x pixel where *ratio* falls within the plotting area."""
        x0, span = self._plot_span(w)
        return x0 + ratio * span

    def _ratio_for_x(self, x: float) -> float:
        """Return the clamped 0..1 ratio for an x pixel in widget coords."""
        x0, span = self._plot_span(self.width())
        return max(0.0, min(1.0, (x - x0) / span))

    # -- Paint --

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            w = self.width()
            h = self.height()

            painter.fillRect(0, 0, w, h, self._bg_color)

            if self._loading:
                self._draw_shimmer(painter, w, h)
                return

            if self._lanes:
                self._draw_lanes(painter, w, h)

            if self._loop_a_ratio is not None and self._loop_b_ratio is not None:
                self._draw_loop_region(painter, w, h)

            self._draw_cursor(painter, w, h)
        finally:
            painter.end()

    def _draw_lanes(self, painter: QPainter, w: int, h: int) -> None:
        font = QFont()
        font.setPixelSize(10)
        painter.setFont(font)

        for i, lane in enumerate(self._lanes):
            _, y, lane_w, rect_h = self._lane_rect(i, w, h)

            if i > 0:
                divider = QColor(
                    self._label_color.red(),
                    self._label_color.green(),
                    self._label_color.blue(),
                    30,
                )
                painter.setPen(QPen(divider, 1.0))
                painter.drawLine(_LABEL_WIDTH, y, w, y)

            opacity = self.lane_opacity(lane.name)
            painter.save()
            painter.setOpacity(opacity)
            self._draw_lane_label(painter, lane.name, y, rect_h)
            self._draw_lane_waveform(painter, lane, lane_w, rect_h, y)
            painter.restore()

    def _draw_shimmer(self, painter: QPainter, w: int, h: int) -> None:
        """Draw a subtle animated shimmer bar sweeping left to right."""
        t = self._shimmer_phase
        pos = 0.5 + 0.5 * math.sin(t * 2 * math.pi - math.pi / 2)

        bar_w = w * 0.25
        bar_x = pos * (w + bar_w) - bar_w

        accent = self._accent_color
        grad = QLinearGradient(bar_x, 0, bar_x + bar_w, 0)
        grad.setColorAt(0.0, QColor(accent.red(), accent.green(), accent.blue(), 0))
        grad.setColorAt(0.4, QColor(accent.red(), accent.green(), accent.blue(), 50))
        grad.setColorAt(0.5, QColor(accent.red(), accent.green(), accent.blue(), 70))
        grad.setColorAt(0.6, QColor(accent.red(), accent.green(), accent.blue(), 50))
        grad.setColorAt(1.0, QColor(accent.red(), accent.green(), accent.blue(), 0))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(grad)
        painter.drawRect(QRectF(bar_x, 0, bar_w, h))

        center = h / 2.0
        line_color = QColor(accent.red(), accent.green(), accent.blue(), 40)
        painter.setPen(QPen(line_color, 1.0))
        painter.drawLine(QPointF(0, center), QPointF(w, center))

    def _draw_lane_label(
        self, painter: QPainter, name: str, y: int, lane_h: int
    ) -> None:
        painter.setPen(self._label_color)
        label = name[:6]
        text_rect = QRectF(
            _LABEL_PADDING,
            float(y),
            float(_LABEL_WIDTH - _LABEL_PADDING * 2),
            float(lane_h),
        )
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            label,
        )

    def _draw_lane_waveform(
        self,
        painter: QPainter,
        lane: _StemLane,
        lane_w: int,
        lane_h: int,
        y_offset: int,
    ) -> None:
        if lane_w <= 0 or lane.max_peak <= 0:
            return

        size_key = (lane_w, lane_h)
        if lane.cached_size != size_key:
            lane.cached_path = self._build_lane_path(lane, lane_w, lane_h)
            lane.cached_size = size_key

        if lane.cached_path is None:
            return

        color = QColor(lane.color)
        center = lane_h / 2.0
        gradient = QLinearGradient(0, y_offset + center, 0, y_offset)
        gradient.setColorAt(
            0.0, QColor(color.red(), color.green(), color.blue(), 220)
        )
        gradient.setColorAt(
            1.0, QColor(color.red(), color.green(), color.blue(), 80)
        )

        painter.save()
        painter.translate(_LABEL_WIDTH, y_offset)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawPath(lane.cached_path)
        painter.restore()

    def _build_lane_path(
        self, lane: _StemLane, lane_w: int, lane_h: int
    ) -> QPainterPath:
        num_peaks = len(lane.peaks)
        center_y = lane_h / 2.0
        max_amplitude = center_y - 2

        num_bars = max(1, lane_w // _BAR_STEP)
        path = QPainterPath()

        for i in range(num_bars):
            x = i * _BAR_STEP
            peak_idx = min(int(i * num_peaks / num_bars), num_peaks - 1)
            amplitude = lane.peaks[peak_idx] / lane.max_peak
            bar_h = amplitude * max_amplitude

            if bar_h < 0.5:
                continue

            top = center_y - bar_h
            full_h = bar_h * 2
            path.addRoundedRect(
                float(x),
                top,
                float(_BAR_WIDTH),
                full_h,
                _BAR_RADIUS,
                _BAR_RADIUS,
            )

        return path

    def _draw_loop_region(self, painter: QPainter, w: int, h: int) -> None:
        assert self._loop_a_ratio is not None
        assert self._loop_b_ratio is not None
        x_a = self._x_for_ratio(self._loop_a_ratio, w)
        x_b = self._x_for_ratio(self._loop_b_ratio, w)
        if x_b < x_a:
            x_a, x_b = x_b, x_a

        painter.fillRect(
            QRectF(x_a, 0.0, x_b - x_a, float(h)), self._loop_region_color
        )

        pen = QPen(self._loop_marker_color, 2.0)
        painter.setPen(pen)
        painter.drawLine(QPointF(x_a, 0.0), QPointF(x_a, float(h)))
        painter.drawLine(QPointF(x_b, 0.0), QPointF(x_b, float(h)))

    def _draw_cursor(self, painter: QPainter, w: int, h: int) -> None:
        if w <= 0:
            return
        x0, span = self._plot_span(w)
        x = self._x_for_ratio(self._position_ratio, w)
        x = max(x0, min(x, x0 + span - 1.0))

        glow_pen = QPen(self._cursor_glow_color, _CURSOR_GLOW_WIDTH)
        glow_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(glow_pen)
        painter.drawLine(QPointF(x, 0), QPointF(x, h))

        cursor_pen = QPen(self._cursor_color, 2.0)
        cursor_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(cursor_pen)
        painter.drawLine(QPointF(x, 0), QPointF(x, h))

    # -- Mouse interaction --

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if event.position().x() < _LABEL_WIDTH:
            # The label gutter is a track header, not part of the timeline.
            # Without this, clicking a stem label would jump to the start.
            return
        self._seeking = True
        self._seek_to_x(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._seeking:
            self._seek_to_x(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._seeking:
            self._seeking = False
            self._seek_to_x(event.position().x())

    def _seek_to_x(self, x: float) -> None:
        if self._total_seconds <= 0 or self.width() <= 0:
            return
        ratio = self._ratio_for_x(x)
        self._position_ratio = ratio
        self.update()
        self.seek_requested.emit(ratio * self._total_seconds)

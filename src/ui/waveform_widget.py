"""Waveform display widget with playback cursor and loop markers.

Custom QWidget using QPainter for rendering. Supports click-to-seek
and drag-to-seek. Respects the active theme via set_theme_colors().
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, Signal, QSize, QPointF, QTimer, QRectF
from PySide6.QtGui import (
    QColor,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import QWidget, QSizePolicy

from src.ui.styles import DARK_COLORS

_WAVEFORM_HEIGHT = 140
_BAR_WIDTH = 2
_BAR_GAP = 1
_BAR_STEP = _BAR_WIDTH + _BAR_GAP
_BAR_RADIUS = 1.0
_CURSOR_GLOW_WIDTH = 6


class WaveformWidget(QWidget):
    """Displays an audio waveform with playback cursor and loop markers."""

    seek_requested = Signal(float)  # Emits seconds

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._peaks: np.ndarray | None = None
        self._max_peak: float = 0.0
        self._position_ratio: float = 0.0
        self._loop_a_ratio: float | None = None
        self._loop_b_ratio: float | None = None
        self._total_seconds: float = 0.0
        self._seeking: bool = False
        self._cached_size: tuple[int, int] = (-1, -1)
        self._cached_path: QPainterPath | None = None
        self._loading: bool = False
        self._shimmer_phase: float = 0.0

        self._shimmer_timer = QTimer(self)
        self._shimmer_timer.setInterval(30)  # ~33fps
        self._shimmer_timer.timeout.connect(self._tick_shimmer)

        self._apply_colors(DARK_COLORS)

        self.setFixedHeight(_WAVEFORM_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(False)

    def _apply_colors(self, colors: dict[str, str]) -> None:
        """Derive paint colors from a theme color dict."""
        self._bg_color = QColor(colors["base"])
        self._waveform_color = QColor(colors["accent"])
        self._cursor_color = QColor(colors["text"])
        self._loop_marker_color = QColor(colors["red"])
        accent = QColor(colors["accent"])
        self._loop_region_color = QColor(
            accent.red(), accent.green(), accent.blue(), 38
        )
        self._cursor_glow_color = QColor(
            self._cursor_color.red(),
            self._cursor_color.green(),
            self._cursor_color.blue(),
            50,
        )
        self._build_gradients()

    def _build_gradients(self) -> None:
        """Pre-build the waveform gradient (bright at center, faded at peaks)."""
        h = self.height() if self.height() > 0 else _WAVEFORM_HEIGHT
        center = h / 2.0
        accent = self._waveform_color

        self._waveform_gradient = QLinearGradient(0, center, 0, 0)
        self._waveform_gradient.setColorAt(0.0, QColor(accent.red(), accent.green(), accent.blue(), 220))
        self._waveform_gradient.setColorAt(1.0, QColor(accent.red(), accent.green(), accent.blue(), 80))

    def set_theme_colors(self, colors: dict[str, str]) -> None:
        """Update paint colors for a new theme and repaint."""
        self._apply_colors(colors)
        self._cached_size = (0, 0)
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._build_gradients()
        self._cached_size = (0, 0)

    def minimumSizeHint(self) -> QSize:
        return QSize(200, _WAVEFORM_HEIGHT)

    def set_peaks(self, peaks: np.ndarray) -> None:
        """Set the pre-computed peak array and trigger repaint."""
        self._peaks = peaks
        self._max_peak = float(np.max(peaks)) if len(peaks) > 0 else 0.0
        self._cached_size = (0, 0)
        if self._loading:
            self.set_loading(False)
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
        old_px = int(self._position_ratio * self.width())
        new_px = int(new_ratio * self.width())
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

    # -- Paint --

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, self._bg_color)

        if self._loading:
            self._draw_shimmer(painter, w, h)
            painter.end()
            return

        if self._peaks is not None and len(self._peaks) > 0:
            self._draw_waveform(painter, w, h)

        if self._loop_a_ratio is not None and self._loop_b_ratio is not None:
            self._draw_loop_region(painter, w, h)

        self._draw_cursor(painter, w, h)

        painter.end()

    def _draw_shimmer(self, painter: QPainter, w: int, h: int) -> None:
        """Draw a subtle animated shimmer bar sweeping left to right."""
        # Smooth sinusoidal easing for the shimmer position
        t = self._shimmer_phase
        # Use sin for smooth back-and-forth feel
        pos = 0.5 + 0.5 * math.sin(t * 2 * math.pi - math.pi / 2)

        bar_w = w * 0.25  # shimmer highlight width
        bar_x = pos * (w + bar_w) - bar_w  # sweep from left to right

        # Build a horizontal gradient for the shimmer highlight
        accent = self._waveform_color
        grad = QLinearGradient(bar_x, 0, bar_x + bar_w, 0)
        grad.setColorAt(0.0, QColor(accent.red(), accent.green(), accent.blue(), 0))
        grad.setColorAt(0.4, QColor(accent.red(), accent.green(), accent.blue(), 50))
        grad.setColorAt(0.5, QColor(accent.red(), accent.green(), accent.blue(), 70))
        grad.setColorAt(0.6, QColor(accent.red(), accent.green(), accent.blue(), 50))
        grad.setColorAt(1.0, QColor(accent.red(), accent.green(), accent.blue(), 0))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(grad)
        painter.drawRect(QRectF(bar_x, 0, bar_w, h))

        # Draw a thin baseline to give the area some structure
        center = h / 2.0
        line_color = QColor(accent.red(), accent.green(), accent.blue(), 40)
        painter.setPen(QPen(line_color, 1.0))
        painter.drawLine(QPointF(0, center), QPointF(w, center))

    def _draw_waveform(self, painter: QPainter, w: int, h: int) -> None:
        assert self._peaks is not None
        if w <= 0 or self._max_peak <= 0:
            return

        if self._cached_size != (w, h):
            self._cached_path = self._build_waveform_path(w, h)
            self._cached_size = (w, h)

        if self._cached_path is None:
            return

        painter.setPen(Qt.PenStyle.NoPen)

        painter.setBrush(self._waveform_gradient)
        painter.drawPath(self._cached_path)

    def _build_waveform_path(self, w: int, h: int) -> QPainterPath:
        """Build a single QPainterPath of rounded bars for the waveform."""
        assert self._peaks is not None
        num_peaks = len(self._peaks)
        center_y = h / 2.0
        max_amplitude = center_y - 2

        num_bars = max(1, w // _BAR_STEP)
        path = QPainterPath()

        for i in range(num_bars):
            x = i * _BAR_STEP
            peak_idx = min(int(i * num_peaks / num_bars), num_peaks - 1)
            amplitude = self._peaks[peak_idx] / self._max_peak
            bar_h = amplitude * max_amplitude

            if bar_h < 0.5:
                continue

            top = center_y - bar_h
            full_h = bar_h * 2
            path.addRoundedRect(
                float(x), top, float(_BAR_WIDTH), full_h,
                _BAR_RADIUS, _BAR_RADIUS,
            )

        return path

    def _draw_loop_region(self, painter: QPainter, w: int, h: int) -> None:
        assert self._loop_a_ratio is not None
        assert self._loop_b_ratio is not None
        x_a = int(self._loop_a_ratio * w)
        x_b = int(self._loop_b_ratio * w)

        painter.fillRect(x_a, 0, x_b - x_a, h, self._loop_region_color)

        pen = QPen(self._loop_marker_color, 2.0)
        painter.setPen(pen)
        painter.drawLine(x_a, 0, x_a, h)
        painter.drawLine(x_b, 0, x_b, h)

    def _draw_cursor(self, painter: QPainter, w: int, h: int) -> None:
        if w <= 0:
            return
        x = self._position_ratio * w
        x = max(0.0, min(x, w - 1.0))

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
        if event.button() == Qt.MouseButton.LeftButton:
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
        ratio = max(0.0, min(1.0, x / self.width()))
        self._position_ratio = ratio
        self.update()
        self.seek_requested.emit(ratio * self._total_seconds)


_MINI_HEIGHT = 24
_MINI_BAR_WIDTH = 1
_MINI_BAR_GAP = 1
_MINI_BAR_STEP = _MINI_BAR_WIDTH + _MINI_BAR_GAP


class MiniWaveformWidget(QWidget):
    """Small inline waveform for a single stem, shown in mixer rows.

    Supports click-to-seek: clicking sets the playback position
    proportionally, mirroring the main waveform's seek behaviour.
    """

    seek_requested = Signal(float)

    def __init__(self, color: str, player=None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._peaks: np.ndarray | None = None
        self._max_peak: float = 0.0
        self._color = QColor(color)
        self._fill_color = QColor(
            self._color.red(), self._color.green(), self._color.blue(), 140
        )
        self._player = player
        self._cached_size: tuple[int, int] = (-1, -1)
        self._cached_path: QPainterPath | None = None

        self.setFixedHeight(_MINI_HEIGHT)
        self.setMinimumWidth(80)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_color(self, color: QColor) -> None:
        """Update the waveform fill color."""
        self._color = color
        self._fill_color = QColor(
            color.red(), color.green(), color.blue(), 140
        )

    def set_peaks(self, peaks: np.ndarray) -> None:
        """Set the peak array and repaint."""
        self._peaks = peaks
        self._max_peak = float(np.max(peaks)) if len(peaks) > 0 else 0.0
        self._cached_size = (0, 0)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._player:
            total = self._player.total_seconds
            if total > 0:
                ratio = max(0.0, min(1.0, event.position().x() / self.width()))
                self.seek_requested.emit(ratio * total)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = self.width()
        h = self.height()

        if self._peaks is not None and self._max_peak > 0:
            if self._cached_size != (w, h):
                self._cached_path = self._build_path(w, h)
                self._cached_size = (w, h)
            if self._cached_path is not None:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(self._fill_color)
                painter.drawPath(self._cached_path)

        painter.end()

    def _build_path(self, w: int, h: int) -> QPainterPath:
        assert self._peaks is not None
        num_peaks = len(self._peaks)
        center_y = h / 2.0
        max_amp = center_y - 1

        num_bars = max(1, w // _MINI_BAR_STEP)
        path = QPainterPath()

        for i in range(num_bars):
            x = i * _MINI_BAR_STEP
            idx = min(int(i * num_peaks / num_bars), num_peaks - 1)
            amplitude = self._peaks[idx] / self._max_peak
            bar_h = amplitude * max_amp
            if bar_h < 0.5:
                continue
            top = center_y - bar_h
            path.addRoundedRect(
                float(x), top, float(_MINI_BAR_WIDTH), bar_h * 2,
                0.5, 0.5,
            )

        return path

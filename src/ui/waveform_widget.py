"""Waveform display widget with playback cursor and loop markers.

Custom QWidget using QPainter for rendering. Supports click-to-seek
and drag-to-seek. Designed for the Catppuccin Mocha dark theme.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, QSize, QRect
from PySide6.QtGui import QColor, QPainter, QMouseEvent, QPaintEvent
from PySide6.QtWidgets import QWidget, QSizePolicy


# Catppuccin Mocha palette
_BG_COLOR = QColor("#1e1e2e")
_WAVEFORM_COLOR = QColor("#89b4fa")  # Accent blue
_CURSOR_COLOR = QColor("#cdd6f4")    # Text color
_LOOP_MARKER_COLOR = QColor("#f38ba8")  # Red
_LOOP_REGION_COLOR = QColor(137, 180, 250, 38)  # Accent at ~15% opacity

_WAVEFORM_HEIGHT = 80


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
        self._cached_size: tuple[int, int] = (0, 0)
        self._cached_rects: list[QRect] = []

        self.setFixedHeight(_WAVEFORM_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(False)

    def minimumSizeHint(self) -> QSize:
        return QSize(200, _WAVEFORM_HEIGHT)

    def set_peaks(self, peaks: np.ndarray) -> None:
        """Set the pre-computed peak array and trigger repaint."""
        self._peaks = peaks
        self._max_peak = float(np.max(peaks)) if len(peaks) > 0 else 0.0
        self._cached_size = (0, 0)  # Invalidate rect cache
        self.update()

    def set_position(self, ratio: float) -> None:
        """Update the playback cursor position (0.0 to 1.0)."""
        self._position_ratio = max(0.0, min(1.0, ratio))
        if not self._seeking:
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
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(0, 0, w, h, _BG_COLOR)

        if self._peaks is not None and len(self._peaks) > 0:
            self._draw_waveform(painter, w, h)

        # Loop region
        if self._loop_a_ratio is not None and self._loop_b_ratio is not None:
            self._draw_loop_region(painter, w, h)

        # Playback cursor
        self._draw_cursor(painter, w, h)

        painter.end()

    def _draw_waveform(self, painter: QPainter, w: int, h: int) -> None:
        assert self._peaks is not None
        if w <= 0 or self._max_peak <= 0:
            return

        # Rebuild rect cache only when size or peaks change.
        if self._cached_size != (w, h):
            self._cached_rects = self._build_waveform_rects(w, h)
            self._cached_size = (w, h)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_WAVEFORM_COLOR)
        painter.drawRects(self._cached_rects)

    def _build_waveform_rects(self, w: int, h: int) -> list[QRect]:
        """Pre-compute waveform bar rectangles for the given width."""
        assert self._peaks is not None
        num_peaks = len(self._peaks)
        center_y = h / 2

        xs = np.arange(w)
        indices = np.minimum((xs * num_peaks // w).astype(int), num_peaks - 1)
        amplitudes = self._peaks[indices] / self._max_peak
        bar_heights = amplitudes * (center_y - 2)

        mask = bar_heights >= 0.5
        visible_xs = xs[mask]
        visible_heights = bar_heights[mask]

        tops = (center_y - visible_heights).astype(int)
        heights = (visible_heights * 2).astype(int)

        return [
            QRect(int(visible_xs[i]), int(tops[i]), 1, int(heights[i]))
            for i in range(len(visible_xs))
        ]

    def _draw_loop_region(self, painter: QPainter, w: int, h: int) -> None:
        assert self._loop_a_ratio is not None
        assert self._loop_b_ratio is not None
        x_a = int(self._loop_a_ratio * w)
        x_b = int(self._loop_b_ratio * w)

        # Shaded region
        painter.fillRect(x_a, 0, x_b - x_a, h, _LOOP_REGION_COLOR)

        # Marker lines
        painter.setPen(_LOOP_MARKER_COLOR)
        painter.drawLine(x_a, 0, x_a, h)
        painter.drawLine(x_b, 0, x_b, h)

    def _draw_cursor(self, painter: QPainter, w: int, h: int) -> None:
        if w <= 0:
            return
        x = int(self._position_ratio * w)
        x = max(0, min(x, w - 1))
        painter.setPen(_CURSOR_COLOR)
        painter.drawLine(x, 0, x, h)
        x2 = min(x + 1, w - 1)
        if x2 != x:
            painter.drawLine(x2, 0, x2, h)  # 2px wide cursor

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

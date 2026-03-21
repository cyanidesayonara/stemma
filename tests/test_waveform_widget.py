"""Tests for the waveform display widget."""

import numpy as np
import pytest
from unittest.mock import patch
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPaintEvent
from PySide6.QtWidgets import QApplication

from src.ui.waveform_widget import WaveformWidget


@pytest.fixture(scope="module")
def app():
    """Ensure a QApplication exists for widget tests."""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


class TestWaveformWidget:
    """Tests for WaveformWidget."""

    def test_creates_without_peaks(self, app):
        """Widget instantiates with no peaks set."""
        widget = WaveformWidget()
        assert widget._peaks is None
        assert widget._position_ratio == 0.0

    def test_set_peaks_stores_data(self, app):
        """set_peaks stores the peak array."""
        widget = WaveformWidget()
        peaks = np.array([0.1, 0.5, 0.3], dtype=np.float32)
        widget.set_peaks(peaks)
        np.testing.assert_array_equal(widget._peaks, peaks)

    def test_set_position_clamps(self, app):
        """Position ratio is clamped to [0, 1]."""
        widget = WaveformWidget()
        widget.set_position(-0.5)
        assert widget._position_ratio == 0.0
        widget.set_position(1.5)
        assert widget._position_ratio == 1.0
        widget.set_position(0.5)
        assert widget._position_ratio == 0.5

    def test_set_loop_markers(self, app):
        """Loop marker ratios are stored."""
        widget = WaveformWidget()
        widget.set_loop_markers(0.2, 0.8)
        assert widget._loop_a_ratio == 0.2
        assert widget._loop_b_ratio == 0.8

    def test_clear_loop_markers(self, app):
        """Loop markers can be cleared by passing None."""
        widget = WaveformWidget()
        widget.set_loop_markers(0.2, 0.8)
        widget.set_loop_markers(None, None)
        assert widget._loop_a_ratio is None
        assert widget._loop_b_ratio is None

    def test_mouse_click_emits_seek(self, app):
        """Clicking the widget emits seek_requested with seconds."""
        widget = WaveformWidget()
        widget.set_total_seconds(120.0)
        widget.resize(500, 80)

        received = []
        widget.seek_requested.connect(lambda s: received.append(s))

        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QMouseEvent
        from PySide6.QtCore import QPointF

        event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(250, 40),
            QPointF(250, 40),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        widget.mousePressEvent(event)

        assert len(received) == 1
        # Click at x=250 of 500px wide → ratio 0.5 → 60.0 seconds
        assert received[0] == pytest.approx(60.0, abs=1.0)

    def test_paint_no_crash_without_peaks(self, app):
        """paintEvent does not crash when no peaks are set."""
        widget = WaveformWidget()
        widget.resize(200, 80)
        widget.repaint()  # Force synchronous paint

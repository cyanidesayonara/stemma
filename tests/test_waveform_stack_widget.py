"""Tests for the stacked stem lane waveform widget."""

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from src.ui.waveform_stack_widget import STACK_HEIGHT, WaveformStackWidget
from tests.widget_visual import assert_widget_snapshot


@pytest.fixture(scope="module")
def app():
    inst = QApplication.instance() or QApplication([])
    return inst


def test_stack_has_fixed_height(app):
    w = WaveformStackWidget()
    assert w.height() == STACK_HEIGHT


def test_update_lane_mix_refreshes_opacity(app):
    w = WaveformStackWidget()
    peaks = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    w.set_stem_lanes([("drums", peaks, "#00ff00")], muted=set(), soloed=set())
    assert w.lane_opacity("drums") == 1.0
    w.update_lane_mix(muted={"drums"}, soloed=set())
    assert w.lane_opacity("drums") == pytest.approx(0.15)


def test_set_stem_lanes_stores_order(app):
    w = WaveformStackWidget()
    peaks = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    w.set_stem_lanes([("vocals", peaks, "#ff0000")], muted=set(), soloed=set())
    assert w.lane_count() == 1


def test_seek_emits_seconds(app):
    w = WaveformStackWidget()
    w.set_total_seconds(100.0)
    w.resize(400, STACK_HEIGHT)
    got = []
    w.seek_requested.connect(got.append)
    w._emit_seek_at_ratio(0.25)
    assert got == [25.0]


def test_set_position_clamps(app):
    w = WaveformStackWidget()
    w.set_position(-0.5)
    assert w._position_ratio == 0.0
    w.set_position(1.5)
    assert w._position_ratio == 1.0
    w.set_position(0.5)
    assert w._position_ratio == 0.5


def test_set_loop_markers(app):
    w = WaveformStackWidget()
    w.set_loop_markers(0.2, 0.8)
    assert w._loop_a_ratio == 0.2
    assert w._loop_b_ratio == 0.8


def test_mouse_click_emits_seek(app):
    w = WaveformStackWidget()
    w.set_total_seconds(120.0)
    w.resize(500, STACK_HEIGHT)

    received = []
    w.seek_requested.connect(received.append)

    event = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(250, STACK_HEIGHT / 2),
        QPointF(250, STACK_HEIGHT / 2),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    w.mousePressEvent(event)

    assert len(received) == 1
    assert received[0] == pytest.approx(60.0, abs=1.0)


def test_paint_no_crash_without_lanes(app):
    w = WaveformStackWidget()
    w.resize(200, STACK_HEIGHT)
    w.repaint()


def test_set_loading_no_crash(app):
    w = WaveformStackWidget()
    w.resize(200, STACK_HEIGHT)
    w.set_loading(True)
    w.repaint()
    w.set_loading(False)
    w.repaint()


def test_muted_lane_low_opacity(app):
    w = WaveformStackWidget()
    peaks = np.array([1.0], dtype=np.float32)
    w.set_stem_lanes([("drums", peaks, "#00ff00")], muted={"drums"}, soloed=set())
    assert w.lane_opacity("drums") < 0.5


def test_solo_hides_non_solo_lanes(app):
    w = WaveformStackWidget()
    peaks = np.array([1.0], dtype=np.float32)
    w.set_stem_lanes(
        [("vocals", peaks, "#f00"), ("drums", peaks, "#0f0")],
        muted=set(),
        soloed={"vocals"},
    )
    assert w.lane_opacity("drums") < 0.5
    assert w.lane_opacity("vocals") == 1.0


def test_active_lane_full_opacity(app):
    w = WaveformStackWidget()
    peaks = np.array([1.0], dtype=np.float32)
    w.set_stem_lanes([("vocals", peaks, "#f00")], muted=set(), soloed=set())
    assert w.lane_opacity("vocals") == 1.0


def test_solo_overrides_mute_opacity(app):
    w = WaveformStackWidget()
    peaks = np.array([1.0], dtype=np.float32)
    w.set_stem_lanes(
        [("vocals", peaks, "#f00"), ("drums", peaks, "#0f0")],
        muted={"vocals"},
        soloed={"vocals"},
    )
    assert w.lane_opacity("vocals") == 1.0
    assert w.lane_opacity("drums") < 0.5


def test_two_lane_stack_snapshot(app):
    w = WaveformStackWidget()
    peaks = np.array([0.0, 0.8, 0.2, 0.9], dtype=np.float32)
    w.set_stem_lanes(
        [("vocals", peaks, "#e78284"), ("drums", peaks, "#a6e3a1")],
        muted=set(),
        soloed=set(),
    )
    w.set_total_seconds(60.0)
    assert_widget_snapshot(w, "waveform_stack_two_lanes", width=640, height=280)


def test_muted_drums_snapshot(app):
    w = WaveformStackWidget()
    peaks = np.array([0.0, 0.8, 0.2, 0.9], dtype=np.float32)
    w.set_stem_lanes(
        [("vocals", peaks, "#e78284"), ("drums", peaks, "#a6e3a1")],
        muted={"drums"},
        soloed=set(),
    )
    w.set_total_seconds(60.0)
    assert_widget_snapshot(w, "waveform_stack_muted_drums", width=640, height=280)


def test_solo_vocals_snapshot(app):
    w = WaveformStackWidget()
    peaks = np.array([0.0, 0.8, 0.2, 0.9], dtype=np.float32)
    w.set_stem_lanes(
        [("vocals", peaks, "#e78284"), ("drums", peaks, "#a6e3a1")],
        muted=set(),
        soloed={"vocals"},
    )
    w.set_total_seconds(60.0)
    assert_widget_snapshot(w, "waveform_stack_solo_vocals", width=640, height=280)

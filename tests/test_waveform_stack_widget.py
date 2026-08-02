"""Tests for the stacked stem lane waveform widget."""

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from src.ui.waveform_stack_widget import (
    _LABEL_WIDTH,
    STACK_HEIGHT,
    STACK_MIN_HEIGHT,
    WaveformStackWidget,
)
from tests.widget_visual import assert_widget_snapshot


@pytest.fixture(scope="module")
def app():
    inst = QApplication.instance() or QApplication([])
    return inst


def test_stack_prefers_full_height_but_can_shrink(app):
    """The stack asks for 280px yet yields down to a readable floor.

    It used to be a fixed 280px, which does not fit alongside the transport,
    practice controls, and mixer in a 600px-tall window -- the minimum the
    main window allows -- so the lowest lanes were clipped away entirely.
    """
    w = WaveformStackWidget()

    assert w.sizeHint().height() == STACK_HEIGHT
    assert w.minimumSizeHint().height() == STACK_MIN_HEIGHT
    assert STACK_MIN_HEIGHT < STACK_HEIGHT

    w.resize(400, STACK_MIN_HEIGHT)
    assert w.height() == STACK_MIN_HEIGHT


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
    _press_at(w, w._x_for_ratio(0.25, w.width()))
    assert got == [pytest.approx(25.0)]


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


def _press_at(w, x: float) -> None:
    event = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(x, STACK_HEIGHT / 2),
        QPointF(x, STACK_HEIGHT / 2),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    w.mousePressEvent(event)


def test_mouse_click_emits_seek(app):
    w = WaveformStackWidget()
    w.set_total_seconds(120.0)
    w.resize(500, STACK_HEIGHT)

    received = []
    w.seek_requested.connect(received.append)

    _press_at(w, 250)

    # x=250 is measured against the plotting area, which starts after the
    # label gutter: (250 - 52) / (500 - 52) * 120.
    assert len(received) == 1
    assert received[0] == pytest.approx(53.036, abs=0.01)


def test_click_on_label_gutter_does_not_seek(app):
    """Clicking a stem label is a header click, not a jump to the start."""
    w = WaveformStackWidget()
    w.set_total_seconds(120.0)
    w.resize(500, STACK_HEIGHT)

    received = []
    w.seek_requested.connect(received.append)

    _press_at(w, _LABEL_WIDTH / 2)

    assert received == []


@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_playhead_x_matches_lane_waveform_x(app, ratio):
    """The playhead must map onto the same span the lane waveform is drawn in.

    Regression test: the cursor, loop markers, and seek previously used the
    full widget width while lanes were inset by the label gutter, so the
    playhead pointed at audio up to _LABEL_WIDTH pixels away from itself.
    """
    w = WaveformStackWidget()
    width = 640
    w.resize(width, STACK_HEIGHT)
    peaks = np.ones(64, dtype=np.float32)
    w.set_stem_lanes([("vocals", peaks, "#e78284")], muted=set(), soloed=set())

    lane_x, _, lane_w, _ = w._lane_rect(0, width, STACK_HEIGHT)
    expected = lane_x + ratio * lane_w

    assert w._x_for_ratio(ratio, width) == pytest.approx(expected, abs=1.0)


@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_ratio_and_x_round_trip(app, ratio):
    """Seeking to where the playhead is drawn must not move it."""
    w = WaveformStackWidget()
    w.resize(640, STACK_HEIGHT)
    x = w._x_for_ratio(ratio, w.width())
    assert w._ratio_for_x(x) == pytest.approx(ratio, abs=1e-6)


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


def test_muted_lane_is_dimmed_more_gently_on_light_theme(app):
    """At the dark-theme dim level a muted lane vanishes on a light background."""
    from src.ui.styles import DARK_COLORS, LIGHT_COLORS

    w = WaveformStackWidget()
    peaks = np.array([1.0], dtype=np.float32)
    w.set_stem_lanes([("drums", peaks, "#00ff00")], muted={"drums"}, soloed=set())

    w.set_theme_colors(DARK_COLORS)
    dark_opacity = w.lane_opacity("drums")
    w.set_theme_colors(LIGHT_COLORS)
    light_opacity = w.lane_opacity("drums")

    assert light_opacity > dark_opacity
    # Still clearly reads as muted rather than active.
    assert light_opacity < 0.5


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

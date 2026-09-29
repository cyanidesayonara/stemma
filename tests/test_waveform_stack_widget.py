"""Tests for the stacked stem lane waveform widget."""

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from src.ui.waveform_stack_widget import (
    _LABEL_WIDTH,
    LANE_MAX_HEIGHT,
    STACK_HEIGHT,
    STACK_MAX_HEIGHT,
    STACK_MIN_HEIGHT,
    WaveformStackWidget,
    lane_label,
)
import src.ui.waveform_stack_widget as stack_module
from src.ui.styles import DARK_COLORS, LIGHT_COLORS
from tests.widget_visual import assert_widget_snapshot


@pytest.fixture(scope="module")
def app():
    inst = QApplication.instance() or QApplication([])
    return inst


@pytest.mark.parametrize("lanes, expected", [
    (0, STACK_HEIGHT),
    (2, STACK_HEIGHT),
    (4, 4 * LANE_MAX_HEIGHT),
    (6, STACK_MAX_HEIGHT),
    (9, STACK_MAX_HEIGHT),
])
def test_height_cap_follows_the_lane_count(app, lanes, expected):
    """Tall windows grow the stack, but never past a readable lane height.

    A single 520px cap gave a two-stem song two 260px lanes. The cap is now
    per lane, never below the stack's preferred height, and never above the
    overall ceiling.
    """
    w = WaveformStackWidget()
    w.set_lane_capacity(lanes)

    assert w.maximumHeight() == expected


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


def test_clear_loop_markers(app):
    w = WaveformStackWidget()
    w.set_loop_markers(0.2, 0.8)
    w.set_loop_markers(None, None)
    assert w._loop_a_ratio is None
    assert w._loop_b_ratio is None


def test_theme_change_invalidates_cached_lane_paths(app):
    """A theme switch drops the cached lane paths so the next paint rebuilds.

    The paths hold shape only; this guards the cache bookkeeping in
    set_theme_colors rather than a visible color.
    """
    w = WaveformStackWidget()
    w.set_stem_lanes(
        [("vocals", np.array([0.2, 0.8, 0.4], dtype=np.float32), "#bfa3dc")],
        muted=set(),
        soloed=set(),
    )
    w.resize(300, STACK_HEIGHT)
    # grab() really paints; repaint() on an unshown widget does nothing.
    w.grab()
    assert w._lanes[0].cached_path is not None

    w.set_theme_colors(LIGHT_COLORS)

    assert w._lanes[0].cached_size == (0, 0)
    assert w._lanes[0].cached_path is None


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


# The paint tests use grab(), which runs paintEvent for real; repaint() on
# an unshown widget is a no-op, so a crash in paintEvent would go unseen.

def test_paint_no_crash_without_lanes(app):
    w = WaveformStackWidget()
    w.resize(200, STACK_HEIGHT)
    assert not w.grab().isNull()


def test_paint_no_crash_narrower_than_the_label_gutter(app):
    """Below the label gutter the lane width goes negative."""
    w = WaveformStackWidget()
    w.set_stem_lanes(
        [("vocals", np.array([0.2, 0.8], dtype=np.float32), "#bfa3dc")],
        muted=set(),
        soloed=set(),
    )
    w.set_loop_markers(0.2, 0.8)
    w.resize(_LABEL_WIDTH - 22, STACK_HEIGHT)
    assert not w.grab().isNull()


def test_paint_no_crash_with_loop_in_light_theme(app):
    w = WaveformStackWidget()
    w.set_theme_colors(LIGHT_COLORS)
    w.set_stem_lanes(
        [("vocals", np.array([0.1, 0.5, 0.3, 0.8], dtype=np.float32), "#9878b8")],
        muted=set(),
        soloed=set(),
    )
    w.set_loop_markers(0.2, 0.8)
    w.resize(300, STACK_HEIGHT)
    assert not w.grab().isNull()


def test_set_loading_no_crash(app):
    w = WaveformStackWidget()
    w.resize(200, STACK_HEIGHT)
    w.set_loading(True)
    assert not w.grab().isNull()
    w.set_loading(False)
    assert not w.grab().isNull()


def test_muted_lane_low_opacity(app):
    w = WaveformStackWidget()
    peaks = np.array([1.0], dtype=np.float32)
    w.set_stem_lanes([("drums", peaks, "#00ff00")], muted={"drums"}, soloed=set())
    assert w.lane_opacity("drums") < 0.5


def test_muted_lane_is_dimmed_more_gently_on_light_theme(app):
    """At the dark-theme dim level a muted lane vanishes on a light background."""

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


@pytest.mark.parametrize("name, label", [
    ("vocals", "Vocals"),
    ("guitar", "Guitar"),
    ("recording_take1", "Take 1"),
    ("recording_take2", "Take 2"),
    ("recording_take01", "Take 1"),
    ("recording_take", "Record"),
])
def test_lane_labels_name_takes_like_the_mixer(name, label):
    """Take lanes showed the first six letters of the internal stem name,
    so both takes read "record" beside mixer rows labelled Take 1 and 2."""
    assert lane_label(name) == label


def test_lane_labels_are_painted_with_lane_label(app, monkeypatch):
    """The paint path must use lane_label, not its own truncation."""
    seen = []
    real = stack_module.lane_label

    def spy(name):
        seen.append(name)
        return real(name)

    monkeypatch.setattr(stack_module, "lane_label", spy)
    w = WaveformStackWidget()
    w.set_stem_lanes(
        [("recording_take1", np.array([0.2, 0.8], dtype=np.float32), "#d4849a")],
        muted=set(),
        soloed=set(),
    )
    w.resize(300, STACK_HEIGHT)
    w.grab()

    assert "recording_take1" in seen


def test_loop_markers_are_tagged_with_their_times(app):
    """The loop points are drawn on the markers ("A 0:12", "B 0:21")."""
    from src.ui.waveform_stack_widget import WaveformStackWidget

    widget = WaveformStackWidget()
    widget.resize(600, 120)
    widget.set_total_seconds(100.0)
    assert widget.loop_tag_texts() is None
    widget.set_loop_markers(0.21, 0.12)  # either order

    assert widget.loop_tag_texts() == ("A 0:12", "B 0:21")
    widget.set_total_seconds(4000.0)
    assert widget.loop_tag_texts() == ("A 8:00", "B 14:00")
    widget.set_total_seconds(20000.0)
    assert widget.loop_tag_texts() == ("A 40:00", "B 1:10:00")
    widget.grab()  # paints the tags without error


def test_a_lone_loop_point_is_tagged(app):
    """Set A before B: the only trace used to be the Loop tooltip."""
    from src.ui.waveform_stack_widget import WaveformStackWidget

    widget = WaveformStackWidget()
    widget.set_total_seconds(100.0)
    widget.set_loop_markers(0.12, None)
    assert widget.loop_tag_texts() == ("A 0:12",)
    widget.set_loop_markers(None, 0.21)
    assert widget.loop_tag_texts() == ("B 0:21",)


@pytest.mark.parametrize("a, b", [(0.985, 1.0), (0.0, 0.01), (0.5, 0.505)])
def test_loop_tags_stay_inside_the_lanes(app, a, b):
    """A narrow loop at an edge put its B tag past the right edge, or its
    A tag over the lane names; crowded tags stack instead of overlapping."""
    from src.ui.waveform_stack_widget import _LABEL_WIDTH, WaveformStackWidget

    width = 700
    widget = WaveformStackWidget()
    widget.resize(width, 120)
    widget.set_total_seconds(240.0)
    widget.set_loop_markers(a, b)
    xs = [widget._x_for_ratio(r, width) for r in (a, b)]
    tag_a, tag_b = widget.loop_tag_rects(xs, width)
    for rect in (tag_a, tag_b):
        assert rect.left() >= _LABEL_WIDTH
        assert rect.right() < width
    assert not tag_a.intersects(tag_b)
    widget.grab()


def test_a_lone_tag_sits_beside_its_marker(app):
    from src.ui.waveform_stack_widget import WaveformStackWidget

    widget = WaveformStackWidget()
    widget.resize(700, 120)
    widget.set_total_seconds(240.0)
    widget.set_loop_markers(0.5, None)
    x = widget._x_for_ratio(0.5, 700)
    (tag,) = widget.loop_tag_rects([x], 700)
    assert tag.left() > x
    widget.grab()


def test_model_labels_name_the_stems_the_app_shows():
    """The 2-stem option said "backing" for the stem the app calls Other."""
    from src.app_settings import SEPARATION_MODELS

    assert not any("backing" in label for _key, label in SEPARATION_MODELS)


def test_delete_take_prompt_names_the_take_like_the_mixer(tmp_path):
    from unittest.mock import MagicMock, patch

    from PySide6.QtWidgets import QMessageBox

    from src.ui.main_window import MainWindow

    stub = MagicMock()
    stub._current_song_id = "s1"
    stub._library.get_song.return_value.stems_path = str(tmp_path)
    with patch(
        "src.ui.main_window.QMessageBox.question",
        return_value=QMessageBox.StandardButton.No,
    ) as ask:
        MainWindow._on_delete_recording(stub, "recording_take1")
    assert ask.call_args.args[2].startswith("Delete Take 1?")


def test_every_lane_keeps_a_readable_height(app):
    """Six stems at the 120 px stack floor were 20 px lanes (#159)."""
    from src.ui.waveform_stack_widget import LANE_MIN_HEIGHT, STACK_MIN_HEIGHT

    widget = WaveformStackWidget()
    widget.set_lane_capacity(6)
    assert widget.minimumHeight() == 6 * LANE_MIN_HEIGHT
    widget.set_lane_capacity(2)
    assert widget.minimumHeight() == STACK_MIN_HEIGHT
    # Recording takes don't raise the floor.
    widget.set_lane_capacity(8, readable_lanes=6)
    assert widget.minimumHeight() == 6 * LANE_MIN_HEIGHT

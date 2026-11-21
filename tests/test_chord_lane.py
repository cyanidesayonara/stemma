"""Tests for the chord lane under the stacked stem waveform (#133)."""

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtWidgets import QApplication

from src.beat_detector import DetectionResult, transpose_chord
from src.ui.chord_lane import (
    CHORD_LANE_HEIGHT,
    ChordLaneWidget,
    splice_chords,
)
from src.ui.styles import DARK_COLORS, LIGHT_COLORS
from src.ui.waveform_stack_widget import WaveformStackWidget

CHORDS = [(0.0, "C"), (25.0, "G"), (50.0, "Am"), (75.0, "F")]
# One bar every 2 s: 8 bars of read-ahead window is 16 s.
DOWNBEATS = [2.0 * i for i in range(50)]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def lane(app):
    widget = ChordLaneWidget()
    widget.resize(852, CHORD_LANE_HEIGHT)
    widget.set_total_seconds(100.0)
    widget.set_chords(CHORDS)
    widget.set_downbeats(DOWNBEATS)
    yield widget
    widget.deleteLater()


def _click(widget, x: float, y: float) -> None:
    pos = QPointF(x, y)
    for kind in (
        QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease,
    ):
        event = QMouseEvent(
            kind, pos, widget.mapToGlobal(pos),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(widget, event)


# -- Time axis -------------------------------------------------------------


def test_overview_blocks_share_the_stem_lanes_x_mapping(lane):
    """A chord starts at the same x as that moment in the stem lanes.

    The overview row is the song on the stack's own axis, so a chord change
    lines up with the waveform above it.
    """
    stack = WaveformStackWidget()
    stack.resize(lane.width(), 200)

    blocks = lane.overview_blocks()

    assert [label for _, _, label in blocks] == ["C", "G", "Am", "F"]
    for (left, right, _), (start, _) in zip(blocks, CHORDS):
        assert left == pytest.approx(
            stack._x_for_ratio(start / 100.0, stack.width())
        )
    # Each block runs to the next chord, the last to the end of the song.
    assert blocks[0][1] == pytest.approx(blocks[1][0])
    assert blocks[-1][1] == pytest.approx(
        stack._x_for_ratio(1.0, stack.width())
    )
    stack.deleteLater()


def test_read_ahead_window_is_eight_bars_with_one_behind(lane):
    """The larger strip shows the bar being played and the next seven.

    Bars come from the detected downbeats, so a slow song shows the same
    amount of music as a fast one, just over more seconds.
    """
    lane.set_position(30.0)

    start, end = lane.read_ahead_window()

    assert end - start == pytest.approx(16.0)
    assert start == pytest.approx(28.0)


def test_read_ahead_window_without_downbeats_uses_a_fixed_span(lane):
    lane.set_downbeats([])
    lane.set_position(30.0)

    start, end = lane.read_ahead_window()

    assert 6.0 <= end - start <= 24.0
    assert start < 30.0 < end


def test_read_ahead_blocks_scroll_with_the_playhead(lane):
    lane.set_position(24.0)
    before = {label: left for left, _, label in lane.read_ahead_blocks()}
    lane.set_position(25.0)
    after = {label: left for left, _, label in lane.read_ahead_blocks()}

    # G starts at 25 s: it moves left by one second's worth of pixels.
    assert after["G"] < before["G"]
    assert after["G"] == pytest.approx(lane.playhead_x())


# -- Highlight -------------------------------------------------------------


@pytest.mark.parametrize("seconds, expected", [
    (0.0, 0), (24.9, 0), (25.0, 1), (60.0, 2), (99.0, 3),
])
def test_current_chord_follows_the_playhead(lane, seconds, expected):
    lane.set_position(seconds)

    assert lane.current_index() == expected


def test_no_current_chord_before_the_first_one(lane):
    lane.set_chords([(5.0, "C"), (10.0, "G")])
    lane.set_position(2.0)

    assert lane.current_index() is None


@pytest.mark.parametrize("colors", [DARK_COLORS, LIGHT_COLORS])
def test_current_chord_is_drawn_in_the_accent(lane, colors):
    """The block under the playhead stands out in both themes."""
    lane.set_theme_colors(colors)
    lane.set_position(60.0)  # Am, 50-75 s

    image = lane.grab().toImage()
    left, right, _ = lane.overview_blocks()[2]
    y = lane.overview_rect().center().y()
    # Sample away from the playhead line and the label text.
    x = int(left + 3)
    assert image.pixelColor(x, int(y)) == QColor(colors["accent"])
    other_left, _, _ = lane.overview_blocks()[0]
    assert image.pixelColor(int(other_left + 3), int(y)) != QColor(
        colors["accent"]
    )


# -- Labels ----------------------------------------------------------------


def test_labels_follow_the_transposition(lane):
    """Labels re-spell live with pitch, exactly as the chord badge does."""
    lane.set_pitch(2)

    assert lane.labels() == ["D", "A", "Bm", "G"]
    lane.set_pitch(-3)
    assert lane.labels() == [transpose_chord(c, -3) for _, c in CHORDS]
    lane.set_pitch(0)
    assert lane.labels() == ["C", "G", "Am", "F"]


def test_narrow_blocks_drop_their_labels_instead_of_overlapping(app):
    """A whole song squeezes a short chord into a few pixels: no label."""
    widget = ChordLaneWidget()
    widget.resize(400, CHORD_LANE_HEIGHT)
    widget.set_total_seconds(240.0)
    widget.set_chords([(0.0, "C"), (100.0, "F#m"), (100.5, "G"), (200.0, "C")])

    shown = widget.overview_label_texts()

    assert "F#m" not in shown
    assert "C" in shown
    widget.deleteLater()


# -- Clicks ----------------------------------------------------------------


def test_clicking_an_overview_block_seeks_to_its_chord(lane):
    seeks = []
    lane.seek_requested.connect(seeks.append)
    left, right, _ = lane.overview_blocks()[2]  # Am at 50 s

    _click(lane, (left + right) / 2, lane.overview_rect().center().y())

    assert seeks == [50.0]


def test_clicking_a_read_ahead_block_seeks_to_its_chord(lane):
    seeks = []
    lane.seek_requested.connect(seeks.append)
    lane.set_position(22.0)
    blocks = {label: (lo, hi) for lo, hi, label in lane.read_ahead_blocks()}
    left, right = blocks["G"]

    _click(lane, (left + right) / 2, lane.read_ahead_rect().center().y())

    assert seeks == [25.0]


def test_clicking_the_gutter_does_not_seek(lane):
    seeks = []
    lane.seek_requested.connect(seeks.append)

    _click(lane, 10, lane.overview_rect().center().y())

    assert seeks == []


# -- States ----------------------------------------------------------------


def test_lane_has_an_accessible_name(lane):
    assert lane.accessibleName() == "Chord lane"


def test_accessible_description_reads_the_current_and_next_chord(lane):
    lane.set_position(30.0)

    assert lane.accessibleDescription() == "Current chord G, next Am"


def test_no_song_shows_nothing(app):
    widget = ChordLaneWidget()

    assert widget.placeholder_text() == ""
    widget.deleteLater()


def test_detection_running_shows_a_quiet_placeholder(app):
    widget = ChordLaneWidget()
    widget.set_total_seconds(100.0)
    widget.set_detecting(True)

    assert widget.placeholder_text() == "Finding chords..."
    widget.deleteLater()


def test_no_chords_detected_says_so(app):
    widget = ChordLaneWidget()
    widget.set_total_seconds(100.0)
    widget.set_detecting(True)
    widget.set_chords([])
    widget.set_detecting(False)

    assert widget.placeholder_text() == "No chords detected"
    widget.grab()  # paints without chords
    widget.deleteLater()


def test_re_detection_keeps_the_chords_on_screen(lane):
    """A loop re-detect must not blank a lane that already has chords."""
    lane.set_detecting(True)

    assert lane.placeholder_text() == ""
    assert lane.labels() == ["C", "G", "Am", "F"]


def test_clear_forgets_the_song(lane):
    lane.clear()

    assert lane.labels() == []
    assert lane.placeholder_text() == ""


def test_still_position_does_not_repaint(lane, monkeypatch):
    """Paused, the 20 Hz position signal must not repaint the lane."""
    lane.set_position(30.0)
    calls = []
    monkeypatch.setattr(lane, "update", lambda *a: calls.append(a))

    lane.set_position(30.0)
    lane.set_position(30.0001)

    assert calls == []


def test_overview_layer_is_cached_between_frames(lane):
    lane.set_position(10.0)
    lane.grab()
    first = lane._overview_cache
    lane.set_position(11.0)
    lane.grab()

    assert lane._overview_cache is first
    lane.set_pitch(1)
    lane.grab()
    assert lane._overview_cache is not first


# -- Loop re-detection -----------------------------------------------------


def test_splice_replaces_chords_inside_the_loop_only():
    """A loop re-detect relabels the loop and keeps the rest of the song."""
    region = [(30.0, "Em"), (40.0, "D")]

    merged = splice_chords(CHORDS, region, 30.0, 45.0)

    assert merged == [
        (0.0, "C"), (25.0, "G"), (30.0, "Em"), (40.0, "D"),
        (45.0, "G"), (50.0, "Am"), (75.0, "F"),
    ]


def test_splice_with_nothing_detected_keeps_the_song():
    assert splice_chords(CHORDS, [], 30.0, 45.0) == CHORDS


def test_splice_does_not_repeat_a_chord_that_continues():
    merged = splice_chords(CHORDS, [(30.0, "G")], 30.0, 45.0)

    assert merged == CHORDS


# -- Wiring through PlayerControls ----------------------------------------


@pytest.fixture
def controls(app):
    from src.ui.player_controls import PlayerControls

    player = MagicMock()
    player.stems = {}
    player.muted_stems = set()
    player.soloed_stems = set()
    player.volumes = {}
    player.beat_times = []
    player.chord_sequence = []
    player.total_seconds = 100.0
    player.current_seconds = 0.0
    player.sample_rate = 44100
    player.loop_a = None
    player.loop_b = None
    player.looping = False
    player.is_playing = False
    player.speed = 1.0
    player.pitch_semitones = 0
    player.counting_in = False
    player.recording_armed = False
    result = PlayerControls(player)
    yield result
    result.shutdown()
    result.setParent(None)
    result.deleteLater()
    app.processEvents()


def _complete(controls, result, region=None):
    worker = MagicMock()
    worker.region = region
    controls._detection_worker = worker
    controls.sender = lambda: worker
    controls._is_active_detection_sender = lambda: True
    controls._on_detect_completed(result)


def test_detection_fills_the_lane(controls):
    lane = controls.waveform_panel.chord_lane
    _complete(controls, DetectionResult(
        chord_sequence=CHORDS, downbeat_times=DOWNBEATS,
    ))

    assert lane.labels() == ["C", "G", "Am", "F"]
    assert lane.downbeats() == DOWNBEATS


def test_loop_detection_splices_into_the_lane(controls):
    lane = controls.waveform_panel.chord_lane
    _complete(controls, DetectionResult(chord_sequence=CHORDS))
    _complete(
        controls,
        DetectionResult(chord_sequence=[(30.0, "Em")]),
        region=(30.0, 45.0),
    )

    assert lane.labels() == ["C", "G", "Em", "G", "Am", "F"]


def test_pitch_change_relabels_the_lane(controls):
    lane = controls.waveform_panel.chord_lane
    controls.restore_chord_sequence(CHORDS)
    controls._player.pitch_semitones = 2

    controls._on_pitch_applied(2)

    assert lane.labels() == ["D", "A", "Bm", "G"]


def test_lane_seek_reaches_the_player(controls):
    controls.waveform_panel.chord_lane.seek_requested.emit(25.0)

    controls._player.seek.assert_called_with(25.0)


def test_lane_sits_under_the_stem_lanes_in_the_waveform_card(controls):
    panel = controls.waveform_panel
    layout = panel.frame.layout()

    assert layout.indexOf(panel.chord_lane) == (
        layout.indexOf(panel.waveform) + 1
    )
    assert panel.chord_lane.height() == CHORD_LANE_HEIGHT or (
        panel.chord_lane.minimumHeight() == CHORD_LANE_HEIGHT
    )


def test_chord_lane_can_be_hidden(controls):
    controls.set_chord_lane_visible(False)
    assert controls.waveform_panel.chord_lane.isHidden()
    controls.set_chord_lane_visible(True)
    assert not controls.waveform_panel.chord_lane.isHidden()


# -- View menu toggle ------------------------------------------------------


def _main_window():
    from src.ui.main_window import MainWindow

    player = MagicMock()
    player.stems = {}
    player.muted_stems = set()
    player.soloed_stems = set()
    player.volumes = {}
    player.beat_times = []
    player.chord_sequence = []
    player.total_seconds = 0.0
    player.current_seconds = 0.0
    player.sample_rate = 44100
    player.loop_a = None
    player.loop_b = None
    player.is_playing = False
    player.has_stems = False
    player.pitch_semitones = 0
    library = MagicMock()
    library.songs = []
    return MainWindow(library, player, MagicMock())


def test_view_menu_toggle_is_on_by_default_and_persists(app):
    from src.app_settings import open_settings

    open_settings().remove("view/chord_lane")
    window = _main_window()
    action = window._chord_lane_action
    lane = window._player_controls.waveform_panel.chord_lane

    assert action.isCheckable()
    assert action.isChecked()
    assert not lane.isHidden()

    action.trigger()

    assert lane.isHidden()
    assert open_settings().value("view/chord_lane", True, type=bool) is False
    # Not close(): that saves the mock player's session into the store.
    window.deleteLater()

    reopened = _main_window()
    assert not reopened._chord_lane_action.isChecked()
    assert reopened._player_controls.waveform_panel.chord_lane.isHidden()
    reopened._chord_lane_action.trigger()
    assert open_settings().value("view/chord_lane", False, type=bool) is True
    reopened.deleteLater()

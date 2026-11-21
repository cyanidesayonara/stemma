"""Tests for the experimental song-section ruler (#133)."""

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtWidgets import QApplication

from src.beat_detector import DetectionResult
from src.ui.section_ruler import (
    PLACEHOLDER_DETECTING,
    PLACEHOLDER_FAILED,
    PLACEHOLDER_NONE,
    SECTION_RULER_HEIGHT,
    SectionRulerWidget,
)
from src.ui.styles import DARK_COLORS, LIGHT_COLORS
from src.ui.waveform_stack_widget import WaveformStackWidget

SECTIONS = [
    (2.0, 18.0, "A"), (18.0, 34.0, "B"), (34.0, 50.0, "A"),
    (50.0, 66.0, "B"), (66.0, 90.0, "C"),
]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def ruler(app):
    widget = SectionRulerWidget()
    widget.resize(852, SECTION_RULER_HEIGHT)
    widget.set_total_seconds(100.0)
    widget.set_sections(SECTIONS)
    yield widget
    widget.deleteLater()


def _mouse(widget, kind, x: float, y: float = None) -> None:
    pos = QPointF(x, SECTION_RULER_HEIGHT / 2 if y is None else y)
    event = QMouseEvent(
        kind, pos, widget.mapToGlobal(pos),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(widget, event)


def _click(widget, x: float) -> None:
    _mouse(widget, QEvent.Type.MouseButtonPress, x)
    _mouse(widget, QEvent.Type.MouseButtonRelease, x)


def _double_click(widget, x: float) -> None:
    _click(widget, x)
    _mouse(widget, QEvent.Type.MouseButtonDblClick, x)
    _mouse(widget, QEvent.Type.MouseButtonRelease, x)


def _middle(ruler, index: int) -> float:
    left, right, _ = ruler.blocks()[index]
    return (left + right) / 2


# -- Layout ----------------------------------------------------------------


def test_blocks_share_the_stem_lanes_x_mapping(ruler):
    stack = WaveformStackWidget()
    stack.resize(ruler.width(), 200)

    blocks = ruler.blocks()

    assert [label for _, _, label in blocks] == ["A", "B", "A", "B", "C"]
    for (left, right, _), (start, end, _) in zip(blocks, SECTIONS):
        assert left == pytest.approx(
            stack._x_for_ratio(start / 100.0, stack.width())
        )
        assert right == pytest.approx(
            stack._x_for_ratio(end / 100.0, stack.width())
        )
    stack.deleteLater()


@pytest.mark.parametrize("seconds, expected", [
    (0.0, None), (2.0, 0), (20.0, 1), (65.9, 3), (89.0, 4), (95.0, None),
])
def test_current_section_follows_the_playhead(ruler, seconds, expected):
    ruler.set_position(seconds)
    assert ruler.current_index() == expected


@pytest.mark.parametrize("colors", [DARK_COLORS, LIGHT_COLORS])
def test_current_section_is_drawn_in_the_accent(ruler, colors):
    ruler.set_theme_colors(colors)
    ruler.set_position(40.0)  # third section, 34-50 s
    image = ruler.grab().toImage()
    dpr = image.devicePixelRatio()
    left, right, _ = ruler.blocks()[2]
    # Away from the playhead and the centred letter.
    x = round((left + 6) * dpr)
    y = round(SECTION_RULER_HEIGHT / 2 * dpr)
    assert image.pixelColor(x, y) == QColor(colors["accent"])
    other_left, _, _ = ruler.blocks()[0]
    assert image.pixelColor(round((other_left + 6) * dpr), y) != QColor(
        colors["accent"]
    )


def test_looped_section_is_marked(ruler):
    ruler.set_loop(18.0, 34.0)
    assert ruler.looped_index() == 1
    ruler.set_loop(18.0, 30.0)  # not a whole section
    assert ruler.looped_index() is None
    ruler.set_loop(None, None)
    assert ruler.looped_index() is None


def test_looped_section_tolerates_sample_rounding(ruler):
    """The player keeps loop points in frames, so they come back a hair
    off the section's seconds."""
    ruler.set_loop(18.0 - 1e-5, 34.0 - 1e-5)
    assert ruler.looped_index() == 1


# -- Mouse -----------------------------------------------------------------


def test_click_seeks_to_the_section_start(ruler):
    seeks = []
    ruler.seek_requested.connect(seeks.append)
    _click(ruler, _middle(ruler, 1))
    assert seeks == [18.0]


def test_double_click_loops_the_section(ruler):
    loops = []
    ruler.loop_requested.connect(lambda a, b: loops.append((a, b)))
    _double_click(ruler, _middle(ruler, 3))
    assert loops == [(50.0, 66.0)]


def test_double_click_on_the_playing_section_does_not_restart_it(ruler):
    """The first click of a double-click must not jump back to the start
    of the section already playing (#246 review)."""
    seeks, loops = [], []
    ruler.seek_requested.connect(seeks.append)
    ruler.loop_requested.connect(lambda a, b: loops.append((a, b)))
    ruler.set_position(25.0)  # inside the second section, 18-34 s
    _double_click(ruler, _middle(ruler, 1))
    ruler._pending_seek_timer.timeout.emit()  # as if the interval passed
    assert loops == [(18.0, 34.0)]
    assert seeks == []


def test_single_click_on_the_playing_section_restarts_it(ruler):
    seeks = []
    ruler.seek_requested.connect(seeks.append)
    ruler.set_position(25.0)
    _click(ruler, _middle(ruler, 1))
    # Held back for the double-click interval, then sent.
    assert seeks == []
    assert ruler._pending_seek_timer.isActive()
    ruler._pending_seek_timer.timeout.emit()
    assert seeks == [18.0]


def test_clicks_outside_the_sections_do_nothing(ruler):
    seeks, loops = [], []
    ruler.seek_requested.connect(seeks.append)
    ruler.loop_requested.connect(lambda a, b: loops.append((a, b)))
    _double_click(ruler, 20.0)  # label gutter
    _double_click(ruler, ruler.width() - 2.0)  # after the last section
    assert seeks == [] and loops == []


def test_pointing_hand_only_over_sections(ruler):
    _mouse(ruler, QEvent.Type.MouseMove, _middle(ruler, 0))
    assert ruler.cursor().shape() == Qt.CursorShape.PointingHandCursor
    _mouse(ruler, QEvent.Type.MouseMove, 10.0)
    assert ruler.cursor().shape() == Qt.CursorShape.ArrowCursor


def test_hover_tooltip_names_the_section_and_says_experimental(ruler):
    _mouse(ruler, QEvent.Type.MouseMove, _middle(ruler, 1))
    tip = ruler.toolTip()
    assert "Section B" in tip
    assert "0:18" in tip and "0:34" in tip
    assert "loop" in tip.lower()
    assert "experimental" in tip.lower()


# -- States ----------------------------------------------------------------


def test_marked_experimental(ruler):
    assert "experimental" in ruler.accessibleName().lower()
    assert ruler.gutter_texts()[1].lower() in ("beta", "experimental")


def test_no_song_shows_nothing(app):
    widget = SectionRulerWidget()
    assert widget.placeholder_text() == ""
    widget.grab()
    widget.deleteLater()


def test_states_without_sections(app):
    widget = SectionRulerWidget()
    widget.resize(600, SECTION_RULER_HEIGHT)
    widget.set_total_seconds(100.0)
    widget.set_detecting(True)
    assert widget.placeholder_text() == PLACEHOLDER_DETECTING
    widget.grab()
    widget.set_detecting(False)
    assert widget.placeholder_text() == PLACEHOLDER_NONE
    widget.set_detection_failed()
    assert widget.placeholder_text() == PLACEHOLDER_FAILED
    widget.set_detecting(True)
    assert widget.placeholder_text() == PLACEHOLDER_DETECTING
    widget.deleteLater()


def test_re_detection_keeps_the_sections_on_screen(ruler):
    ruler.set_detecting(True)
    assert ruler.placeholder_text() == ""
    assert len(ruler.blocks()) == 5


def test_clear_forgets_the_song(ruler):
    ruler.set_loop(18.0, 34.0)
    ruler.clear()
    assert ruler.blocks() == []
    assert ruler.looped_index() is None
    assert ruler.placeholder_text() == ""


# -- Repaint cost ----------------------------------------------------------


def test_still_position_does_not_repaint(ruler, monkeypatch):
    ruler.set_position(30.0)
    calls = []
    monkeypatch.setattr(ruler, "update", lambda *a: calls.append(a))
    ruler.set_position(30.0)
    ruler.set_position(30.0001)
    assert calls == []


def test_block_layer_is_cached_between_frames(ruler):
    ruler.set_position(10.0)
    ruler.grab()
    cached = ruler._blocks_cache
    ruler.set_position(11.0)
    ruler.grab()
    assert ruler._blocks_cache is cached
    ruler.set_theme_colors(LIGHT_COLORS)
    ruler.grab()
    assert ruler._blocks_cache is not cached


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
    player.sections = []
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
    player.set_sections.side_effect = (
        lambda s: setattr(player, "sections", list(s))
    )

    def set_a(seconds):
        player.loop_a = seconds

    def set_b(seconds):
        player.loop_b = seconds

    def clear():
        player.loop_a = player.loop_b = None

    player.set_loop_a.side_effect = set_a
    player.set_loop_b.side_effect = set_b
    player.clear_loop.side_effect = clear
    result = PlayerControls(player)
    result.start_detection = MagicMock()
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


def test_ruler_sits_above_the_stem_lanes(controls):
    panel = controls.waveform_panel
    layout = panel.frame.layout()
    assert layout.indexOf(panel.section_ruler) == (
        layout.indexOf(panel.waveform) - 1
    )


def test_detection_fills_the_ruler_and_the_player(controls):
    _complete(controls, DetectionResult(sections=SECTIONS))
    assert controls.waveform_panel.section_ruler.sections() == SECTIONS
    assert controls._player.sections == SECTIONS


def test_loop_detection_keeps_the_songs_sections(controls):
    _complete(controls, DetectionResult(sections=SECTIONS))
    _complete(controls, DetectionResult(), region=(18.0, 34.0))
    assert controls._player.sections == SECTIONS
    assert controls.waveform_panel.section_ruler.sections() == SECTIONS


def test_detecting_state_reaches_the_ruler(controls):
    ruler = controls.waveform_panel.section_ruler
    ruler.set_total_seconds(100.0)
    controls._run_detection = MagicMock()
    controls._chord_lane.set_detecting(True)
    controls._section_ruler.set_detecting(True)
    assert ruler.placeholder_text() == PLACEHOLDER_DETECTING
    _complete(controls, DetectionResult())
    assert ruler.placeholder_text() == PLACEHOLDER_NONE


def test_failed_song_detection_is_shown_in_the_ruler(controls):
    worker = MagicMock()
    worker.region = None
    controls.sender = lambda: worker
    controls._is_active_detection_sender = lambda: True
    controls._section_ruler.set_total_seconds(100.0)
    controls._on_detect_error("boom")
    assert controls._section_ruler.placeholder_text() == PLACEHOLDER_FAILED


def test_restore_sections(controls):
    controls.restore_sections(SECTIONS)
    assert controls._player.sections == SECTIONS
    assert controls._section_ruler.sections() == SECTIONS


def test_loop_section_sets_a_whole_section_loop(controls):
    controls.restore_sections(SECTIONS)
    controls._player.current_seconds = 40.0  # inside the third, 34-50 s

    controls.loop_section_at(40.0)

    player = controls._player
    assert (player.loop_a, player.loop_b) == (34.0, 50.0)
    assert controls._loop_toggle_btn.isChecked()
    # Inside the section already: playback stays where it is.
    player.seek.assert_not_called()
    assert controls._section_ruler.looped_index() == 2
    # Chords re-detect for the loop as with any A-B loop.
    controls.start_detection.assert_called_with(start_sec=34.0, end_sec=50.0)


def test_loop_section_from_outside_jumps_to_its_start(controls):
    controls.restore_sections(SECTIONS)
    controls._player.current_seconds = 5.0
    controls._section_ruler.loop_requested.emit(50.0, 66.0)
    assert (controls._player.loop_a, controls._player.loop_b) == (50.0, 66.0)
    controls._player.seek.assert_called_with(50.0)


def test_loop_current_section_shortcut_target(controls):
    controls.restore_sections(SECTIONS)
    controls._player.current_seconds = 20.0
    controls.loop_current_section()
    assert (controls._player.loop_a, controls._player.loop_b) == (18.0, 34.0)


def test_loop_current_section_without_sections_does_nothing(controls):
    controls._player.current_seconds = 20.0
    controls.loop_current_section()
    controls._player.set_loop_a.assert_not_called()
    assert not controls._loop_toggle_btn.isChecked()


def test_manual_loop_change_updates_the_ruler_mark(controls):
    controls.restore_sections(SECTIONS)
    controls.loop_section_at(20.0)
    assert controls._section_ruler.looped_index() == 1
    controls._player.current_seconds = 30.0
    controls.set_loop_b()
    assert controls._section_ruler.looped_index() is None


def test_section_ruler_can_be_hidden(controls):
    controls.set_section_ruler_visible(False)
    assert controls.waveform_panel.section_ruler.isHidden()
    controls.set_section_ruler_visible(True)
    assert not controls.waveform_panel.section_ruler.isHidden()


def _main_window():
    from src.ui.main_window import MainWindow

    player = MagicMock()
    player.stems = {}
    player.muted_stems = set()
    player.soloed_stems = set()
    player.volumes = {}
    player.beat_times = []
    player.chord_sequence = []
    player.sections = []
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

    open_settings().remove("view/section_ruler")
    window = _main_window()
    action = window._section_ruler_action
    ruler = window._player_controls.waveform_panel.section_ruler

    assert action.isCheckable() and action.isChecked()
    assert "experimental" in action.text().lower()
    assert not ruler.isHidden()

    action.trigger()

    assert ruler.isHidden()
    assert open_settings().value("view/section_ruler", True, type=bool) is False
    window.deleteLater()

    reopened = _main_window()
    assert reopened._player_controls.waveform_panel.section_ruler.isHidden()
    reopened._section_ruler_action.trigger()
    assert open_settings().value("view/section_ruler", False, type=bool) is True
    reopened.deleteLater()


def test_shift_l_loops_the_section_playing(app):
    from PySide6.QtGui import QKeySequence, QShortcut

    window = _main_window()
    keys = [s.key() for s in window.findChildren(QShortcut)]
    assert QKeySequence(Qt.Modifier.SHIFT | Qt.Key.Key_L) in keys
    listed = [
        key for column in window._SHORTCUT_COLUMNS
        for _, rows in column for key, _ in rows
    ]
    assert "Shift+L" in listed
    window.deleteLater()


def test_new_song_starts_the_ruler_from_the_player(controls):
    controls._player.sections = SECTIONS
    controls._player.total_seconds = 100.0
    controls.set_stem_names([])
    assert controls._section_ruler.sections() == SECTIONS

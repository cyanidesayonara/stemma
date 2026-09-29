"""Characterization tests for the extracted PlayerControls facade."""

from importlib import import_module
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QScrollArea,
    QWidget,
)

from src.ui.main_window import MainWindow
from src.ui.player_controls import PlayerControls
from src.ui.practice_rack import PracticeRack
from src.ui.song_info_bar import SongInfoBar
from src.ui.styles import DARK_COLORS, LIGHT_COLORS, get_stylesheet
from src.ui.waveform_stack_widget import (
    LANE_MAX_HEIGHT,
    STACK_HEIGHT,
    STACK_MAX_HEIGHT,
)


@pytest.fixture(scope="module")
def qapp():
    """Keep one QApplication alive for all widget ownership checks."""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def player():
    """Provide the presentation state PlayerControls reads during setup."""
    result = MagicMock()
    result.stems = {}
    result.muted_stems = set()
    result.soloed_stems = set()
    result.volumes = {}
    result.beat_times = []
    result.chord_sequence = []
    result.total_seconds = 0.0
    result.current_seconds = 0.0
    result.sample_rate = 44100
    result.loop_a = None
    result.loop_b = None
    result.looping = False
    result.is_playing = False
    result.has_stems = False
    result.speed = 1.0
    result.pitch_semitones = 0
    result.counting_in = False
    result.count_in_current_beat = 0
    result.count_in_beats = 4
    result.recording_armed = False
    return result


@pytest.fixture
def controls(qapp, player):
    """Create and deterministically destroy the facade."""
    result = PlayerControls(player)
    yield result
    result.shutdown()
    result.setParent(None)
    result.deleteLater()
    qapp.processEvents()


def _component_types():
    """Import the intended component API with an actionable red failure."""
    try:
        transport = import_module("src.ui.transport_bar").TransportBar
        mixer = import_module("src.ui.stem_mixer").StemMixer
        practice = import_module("src.ui.practice_rack").PracticeRack
        song_info = import_module("src.ui.song_info_bar").SongInfoBar
    except (AttributeError, ModuleNotFoundError) as exc:
        pytest.fail(f"PlayerControls component extraction is missing: {exc}")
    return transport, mixer, practice, song_info


def _layout_widgets(widget: QWidget):
    """Yield widgets in visual layout order, descending into containers."""
    if isinstance(widget, QScrollArea):
        # A scroll area holds its content via setWidget rather than in a
        # layout item, so the walk would stop here without this.
        inner = widget.widget()
        if inner is not None:
            yield inner
            yield from _layout_widgets(inner)
        return
    layout = widget.layout()
    if layout is None:
        return
    for index in range(layout.count()):
        item = layout.itemAt(index)
        child = item.widget()
        child_layout = item.layout()
        if child is not None:
            yield child
            yield from _layout_widgets(child)
        elif child_layout is not None:
            for nested_index in range(child_layout.count()):
                nested_item = child_layout.itemAt(nested_index)
                nested_widget = nested_item.widget()
                if nested_widget is not None:
                    yield nested_widget
                    yield from _layout_widgets(nested_widget)


def test_facade_owns_all_four_component_types(controls):
    """The facade remains the lifetime owner of each cohesive widget."""
    component_types = _component_types()
    components = (
        controls.transport_bar,
        controls.stem_mixer,
        controls.practice_rack,
        controls.song_info_bar,
    )

    assert tuple(type(component) for component in components) == component_types
    assert all(controls.isAncestorOf(component) for component in components)


def test_facade_keeps_existing_widget_aliases(controls):
    """Existing MainWindow and test reach-through remains compatible."""
    _component_types()

    assert controls._play_btn is controls.transport_bar.play_button
    assert controls._record_btn is controls.transport_bar.record_button
    assert controls._waveform is controls.waveform_panel.waveform
    assert controls._waveform_frame is controls.waveform_panel.frame
    assert controls._stem_rows is controls.stem_mixer.stem_rows
    assert controls._recording_rows is controls.stem_mixer.recording_rows
    assert controls._speed_combo is controls.practice_rack.speed_combo
    assert controls._pitch_spin is controls.practice_rack.pitch_spin
    assert controls._key_label is controls.song_info_bar.key_label
    assert (
        controls._detected_bpm_label
        is controls.song_info_bar.detected_bpm_label
    )


def test_component_intent_signals_route_through_facade(controls, player):
    """Components expose intent while PlayerControls coordinates the player."""
    _component_types()

    controls.transport_bar.play_pause_requested.emit()
    player.play.assert_called_once_with()

    controls.practice_rack.metronome_toggled.emit(True)
    player.set_metronome_enabled.assert_called_once_with(True)

    controls.practice_rack.count_in_toggled.emit(True)
    player.set_count_in_enabled.assert_called_once_with(True)


def test_stem_and_recording_lifecycle_delegates_to_mixer(controls):
    """Facade row APIs delegate to the component that owns those rows."""
    _component_types()

    with patch.object(
        controls.stem_mixer,
        "set_stem_names",
        wraps=controls.stem_mixer.set_stem_names,
    ) as set_names, patch.object(
        controls.stem_mixer,
        "add_recording_row",
        wraps=controls.stem_mixer.add_recording_row,
    ) as add_recording, patch.object(
        controls.stem_mixer,
        "remove_recording_row",
        wraps=controls.stem_mixer.remove_recording_row,
    ) as remove_recording:
        controls.set_stem_names(["vocals", "drums"])
        row = controls.add_recording_row("recording_take1", "Take 1")
        controls.remove_recording_row("recording_take1")

    set_names.assert_called_once_with(["vocals", "drums"])
    add_recording.assert_called_once_with("recording_take1", "Take 1")
    remove_recording.assert_called_once_with("recording_take1")
    assert row.parent() is None


def test_practice_cards_compose_in_intended_order(controls):
    """Practice controls read as waveform, readout, three cards, mixer,
    then the anchored transport.

    This replaces the extraction-era order guard. That test pinned the
    pre-recomposition layout deliberately, so #131 slice 2 rewrites it rather
    than deleting it: every control that existed before must still be reachable
    from the layout, now grouped by purpose instead of by row.
    """
    _component_types()
    expected = [
        controls._waveform_frame,
        # Song readout strip: key, chord, and tempo together
        controls._key_label,
        controls._chord_label,
        controls._detected_bpm_label,
        # Card: Loop and Trainer (trainer progress beside the card title)
        controls._trainer_status,
        controls._loop_a_btn,
        controls._loop_b_btn,
        controls._loop_toggle_btn,
        controls._loop_clear_btn,
        # Loop points are tags on the waveform markers, not a card label.
        controls._trainer_check,
        controls._trainer_start_combo,
        # Card: Speed and Pitch
        controls._speed_label,
        controls._speed_combo,
        controls._pitch_label,
        controls._pitch_spin,
        # Card: Metronome and Count-in
        controls._metro_label,
        controls._metronome_toggle,
        controls._bpm_spin,
        controls._tap_btn,
        controls._beat_sync_btn,
        controls._beat_nudge_spin,
        controls._metronome_vol_slider,
        controls._metronome_vol_combo,
        controls._count_in_label,
        controls._ci_label,
        controls._count_in_toggle,
        controls._count_in_beats_spin,
        controls._count_in_repeats_cb,
        # Mixer
        controls._mixer_label,
        controls._stems_frame,
        controls._recordings_label,
        controls._recordings_frame,
        # Transport, anchored last: it sits below the scrolling content so it
        # stays put however far the practice controls are scrolled.
        controls._play_btn,
        controls._stop_btn,
        controls._record_btn,
        controls._time_label,
        controls._master_vol_label_prefix,
        controls._master_volume_slider,
        controls._master_volume_combo,
    ]
    markers = set(expected)
    actual = [
        widget for widget in _layout_widgets(controls) if widget in markers
    ]

    assert actual == expected


def test_count_in_sits_with_the_metronome_not_the_transport(controls):
    """Count-in moved out of the isolated transport corner (#131)."""
    transport = set(_layout_widgets(controls.transport_bar))
    rack = set(_layout_widgets(controls.practice_rack))

    assert controls._count_in_toggle not in transport
    for widget in (
        controls._count_in_toggle,
        controls._count_in_beats_spin,
        controls._count_in_repeats_cb,
        controls._ci_label,
    ):
        assert widget in rack


def test_song_readout_strip_holds_key_chord_and_tempo(controls):
    """Tempo reads beside key and chord instead of from the metronome row.

    The BPM label used to be re-parented into the metronome layout, so
    "detecting..." rendered twice in two different places while detection ran.
    """
    strip = set(_layout_widgets(controls.song_info_bar))

    assert controls._key_label in strip
    assert controls._chord_label in strip
    assert controls._detected_bpm_label in strip


def test_practice_controls_are_grouped_into_titled_cards(controls):
    """Three labeled cards replace the flat equal-weight rows."""
    titles = {
        label.text()
        for label in controls.practice_rack.findChildren(QLabel)
        if label.objectName() == "title-label"
    }

    assert titles == {
        "Loop and Trainer",
        "Speed and Pitch",
        "Metronome and Count-in",
    }


def test_practice_cards_wrap_when_the_rack_is_too_narrow(controls):
    """Three cards need ~1140px; the window's 900px minimum gives far less.

    Standing them in one row regardless clipped the metronome card's labels
    and buttons to fragments ("Metronome:" became "Metr", "Tap" became "aj").
    """
    # Standalone: inside PlayerControls the parent layout immediately resizes
    # the rack back, so an explicit resize would not stick. It must also be
    # shown, because Qt does not deliver resize events to unrealized widgets.
    rack = PracticeRack(SongInfoBar())
    # MainWindow sets an explicit 900x600 minimum, which is what lets the
    # layout squeeze the rack below the width three cards would need. Without
    # an explicit minimum here, resize() is clamped and never gets narrow.
    rack.setMinimumSize(200, 100)
    rack.show()
    needed = rack._required_card_width()

    def metronome_row() -> int:
        grid = rack._cards_grid
        return grid.getItemPosition(grid.indexOf(rack._metronome_card))[0]

    rack.resize(needed + 80, rack.height())
    QApplication.processEvents()
    assert rack.cards_side_by_side is True
    assert metronome_row() == 0

    rack.resize(needed - 300, rack.height())
    QApplication.processEvents()
    assert rack.cards_side_by_side is False
    assert metronome_row() == 1

    rack.close()
    rack.deleteLater()


def _hosted_rack():
    """A PracticeRack hosted the way PlayerControls hosts it.

    A resizable scroll area with the horizontal scrollbar off lets its
    content be no narrower than the content's minimum size, which is what
    held the side-by-side cards at three-card width. A plain layout squeezes
    the rack below its minimum anyway and would hide the bug.
    """
    host = QScrollArea()
    host.setWidgetResizable(True)
    host.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    rack = PracticeRack(SongInfoBar())
    host.setWidget(rack)
    host.setMinimumSize(200, 100)
    host.show()
    return host, rack


def test_cards_rewrap_when_the_available_width_shrinks(qapp):
    """Standing side by side must not lock the rack at three-card width.

    The cards' minimum width propagated up as the rack's own minimum, so once
    side by side the rack could never be given less room: when a scrollbar
    appeared it overflowed the scroll area and was clipped instead of
    wrapping.
    """
    host, rack = _hosted_rack()
    needed = rack._required_card_width()

    host.resize(needed + 80, 400)
    for _ in range(3):
        QApplication.processEvents()
    assert rack.cards_side_by_side is True

    host.resize(needed - 40, 400)
    for _ in range(3):
        QApplication.processEvents()
    assert rack.width() <= host.viewport().width()
    assert rack.cards_side_by_side is False

    host.close()
    host.deleteLater()


def test_cards_rewrap_when_their_content_grows(qapp):
    """Setting loop points widens the loop card with "A: 0:12  B: 0:21".

    At 1366px with six stems that pushed the cards past the rack's width,
    and nothing reflowed them because the window itself had not resized.
    """
    host, rack = _hosted_rack()
    host.resize(rack._required_card_width() + 10, 400)
    QApplication.processEvents()
    assert rack.cards_side_by_side is True

    # Any control in a card can grow; widen the trainer start combo.
    rack._trainer_start_combo.setMinimumWidth(900)
    # The size change travels label -> card frame -> card -> rack as a chain
    # of posted layout requests, which takes more than one event-loop pass.
    for _ in range(3):
        QApplication.processEvents()

    assert rack._required_card_width() > rack.width()
    assert rack.cards_side_by_side is False

    host.close()
    host.deleteLater()


def test_card_contents_do_not_paint_the_page_background(qapp):
    """Empty labels and row containers inside a card must be see-through.

    The global ``QWidget`` rule paints every widget in the page color, so
    each label, checkbox, and row container drew a darker box over the
    card's own background -- visible as stray tiles beside Clear and Speed,
    and as a tall dark band behind Count-in in a large window.
    """
    rack = PracticeRack(SongInfoBar())
    rack.setStyleSheet(get_stylesheet("dark"))
    rack.resize(1400, 400)
    rack.show()
    QApplication.processEvents()

    image = rack.grab().toImage()
    mantle = QColor(DARK_COLORS["mantle"])
    # The count-in beat label is fixed-width and empty until count-in runs,
    # so its center shows whatever sits behind it.
    label = rack._count_in_label
    center = label.mapTo(rack, label.rect().center())
    assert image.pixelColor(center).name() == mantle.name()

    rack.close()
    rack.deleteLater()


def test_spare_height_goes_to_the_waveform_not_the_cards(controls):
    """A tall window should grow the waveform, not stretch the cards.

    Expanding cards pulled the spare height into themselves and spread their
    rows apart, leaving the waveform at its preferred height with empty
    space both inside the cards and below the mixer.
    """
    controls.set_stem_names(["vocals", "drums", "bass", "other"])
    controls.resize(1800, 1100)
    controls.show()
    for _ in range(3):
        QApplication.processEvents()

    waveform = controls._waveform_panel.waveform
    assert STACK_HEIGHT < waveform.height() <= STACK_MAX_HEIGHT
    # The frame hugs the lanes; past the cap it must not grow on its own and
    # leave empty bands above and below them.
    assert controls._waveform_frame.height() <= waveform.height() + 12

    cards = (
        controls.practice_rack._loop_card,
        controls.practice_rack._speed_card,
        controls.practice_rack._metronome_card,
    )
    # Side by side, every card stretches to the tallest one so the row keeps
    # a shared bottom edge -- but no further than that.
    tallest = max(card.sizeHint().height() for card in cards)
    for card in cards:
        assert card.height() <= tallest + 2

    # A shorter card's extra height belongs to its frame, not its title:
    # titles and frame tops line up across the row.
    assert controls.practice_rack.cards_side_by_side
    frame_tops = {
        card.findChild(QFrame, "card-frame").y() for card in cards
    }
    title_heights = {
        card.findChild(QLabel, "title-label").height() for card in cards
    }
    assert len(frame_tops) == 1, frame_tops
    assert len(title_heights) == 1, title_heights

    controls.hide()


def test_two_stem_songs_do_not_get_giant_lanes(controls):
    """The waveform's height cap follows the mixer rows, takes included."""
    controls.set_stem_names(["vocals", "other"])
    controls.resize(1800, 1100)
    controls.show()
    # Layout settles over several posted-event passes.
    for _ in range(3):
        QApplication.processEvents()

    waveform = controls._waveform_panel.waveform
    assert waveform.height() <= STACK_HEIGHT
    assert controls._waveform_frame.height() <= waveform.height() + 12
    # With the waveform capped, the leftover height must collect below the
    # mixer rather than stretch the practice cards.
    cards = (
        controls.practice_rack._loop_card,
        controls.practice_rack._speed_card,
        controls.practice_rack._metronome_card,
    )
    tallest = max(card.sizeHint().height() for card in cards)
    assert all(card.height() <= tallest + 2 for card in cards)

    # The leftover opens between the cards and the mixer, so the mixer sits
    # on the anchored transport, where a player reaches between takes.
    rack = controls.practice_rack
    mixer = controls.stem_mixer
    column = controls._controls_widget
    assert mixer.geometry().bottom() >= column.height() - 12
    assert mixer.geometry().top() - rack.geometry().bottom() > 100

    controls.add_recording_row("recording_take1", "Take 1")
    controls.add_recording_row("recording_take2", "Take 2")
    QApplication.processEvents()
    assert waveform.maximumHeight() == 4 * LANE_MAX_HEIGHT

    # Deleting one take is the normal path (MainWindow's delete button).
    controls.remove_recording_row("recording_take1")
    QApplication.processEvents()
    assert waveform.maximumHeight() == 3 * LANE_MAX_HEIGHT

    controls.clear_recording_rows()
    QApplication.processEvents()
    assert waveform.maximumHeight() == STACK_HEIGHT

    controls.hide()


def test_transport_is_anchored_outside_the_scrolling_content(controls):
    """The transport must not scroll away with the practice controls.

    The waveform scrolls with everything else; play, stop, record, and the
    master volume stay put.
    """
    scrolled = set(_layout_widgets(controls._controls_widget))

    assert controls._waveform_frame in scrolled
    for widget in (
        controls._play_btn,
        controls._stop_btn,
        controls._record_btn,
        controls._master_volume_slider,
    ):
        assert widget not in scrolled

    assert controls._transport_bar.parent() is controls


def test_controls_column_scrolls_rather_than_overlapping(controls):
    """A short window must scroll the controls, not compress them past their
    minimums until the card rows draw on top of each other."""
    scroll = controls._controls_scroll

    assert scroll.widget() is controls._controls_widget
    assert scroll.widgetResizable() is True


def test_theme_and_session_state_survive_component_delegation(controls):
    """Practice and song-info state remains stable across a theme switch."""
    _component_types()

    controls.restore_trainer_state(True, 0.5)
    controls.restore_count_in_state(True, 6, True)
    controls.set_detected_key("A minor", "high")
    controls.set_detected_bpm_text("~120 BPM", "medium")

    controls.apply_theme("light", LIGHT_COLORS)
    controls.apply_theme("dark", DARK_COLORS)

    assert controls.trainer_enabled is True
    assert controls.trainer_start_speed == 0.5
    assert controls._count_in_toggle.isChecked()
    assert controls._count_in_beats_spin.value() == 6
    assert controls._count_in_repeats_cb.isChecked()
    assert controls.detected_key == "A minor"
    assert controls.detected_bpm_text == "~120 BPM"
    assert "A minor" in controls.song_info_bar.key_label.text()
    assert "~120 BPM" in controls.song_info_bar.detected_bpm_label.text()


def test_shutdown_is_idempotent_before_component_deletion(
    controls, qapp,
):
    """Repeated facade shutdown drains retained workers only once."""
    _component_types()
    worker = MagicMock()
    controls._orphaned_workers = [worker]

    controls.shutdown()
    controls.shutdown()
    controls.setParent(None)
    controls.deleteLater()
    qapp.processEvents()

    worker.wait.assert_called_once_with()


def test_speed_tooltip_names_the_real_shortcut(controls):
    """The Speed tooltip advertised [ / ], which are not bound; speed is on
    Shift+Up / Shift+Down (see MainWindow shortcuts and Help > Keyboard
    Shortcuts)."""
    tip = controls.practice_rack.speed_combo.toolTip()

    assert "Shift+Up" in tip
    assert "[" not in tip


def test_setting_a_loop_does_not_widen_the_loop_card():
    """Loop points used to be a label in the Loop card; setting a loop
    widened it and wrapped the practice cards at 1366 px (#186)."""
    host, rack = _hosted_rack()
    host.resize(rack._required_card_width() + 10, 400)
    QApplication.processEvents()
    before = rack._loop_card.minimumSizeHint().width()

    rack._loop_label.setText("A: 12:40  B: 13:00")
    for _ in range(3):
        QApplication.processEvents()

    assert rack._loop_card.minimumSizeHint().width() == before
    assert rack.cards_side_by_side is True


def test_trainer_progress_does_not_widen_the_loop_card():
    """"now 0.75x" in the card wrapped every card at 1366 px (#211 review);
    it sits beside the card title now, which has room to spare."""
    host, rack = _hosted_rack()
    host.resize(rack._required_card_width() + 10, 400)
    QApplication.processEvents()
    before = rack._loop_card.minimumSizeHint().width()

    for text in ("(set an A-B loop)", "now 0.75x", "at 1.0x"):
        rack._trainer_status.setText(text)
        for _ in range(3):
            QApplication.processEvents()
        assert rack._loop_card.minimumSizeHint().width() == before
        assert rack.cards_side_by_side is True

    host.close()
    host.deleteLater()


@pytest.mark.parametrize("restored", [True, False])
def test_key_and_tempo_badges_follow_a_theme_switch(controls, restored):
    """A restored (or detected) key and tempo kept the dark badge in the
    light theme: they were styled outside the info bar, which then had
    nothing to redraw from (#159)."""
    controls.apply_theme("dark", DARK_COLORS)
    if restored:
        controls.set_detected_key("B major", "high")
        controls.set_detected_bpm_text("~91 BPM", "low")
    else:
        result = MagicMock()
        result.bpm, result.bpm_confidence = 91.0, "low"
        result.key, result.key_confidence = "B major", "high"
        result.beat_times, result.downbeat_times = [], []
        result.chord_sequence = []
        with patch.object(
            controls, "_is_active_detection_sender", return_value=True,
        ):
            controls._on_detect_completed(result)

    controls.apply_theme("light", LIGHT_COLORS)

    bar = controls.song_info_bar
    for label in (bar.key_label, bar.detected_bpm_label):
        sheet = label.styleSheet()
        assert LIGHT_COLORS["surface0"] in sheet
        assert DARK_COLORS["surface0"] not in sheet
    assert "B major" in bar.key_label.text()


def test_key_only_detecting_status_follows_a_theme_switch(controls):
    """Drives the real re-detect path, with the worker stubbed out."""
    controls.apply_theme("dark", DARK_COLORS)
    controls.set_detected_bpm_text("~91 BPM", "low")
    with patch("src.ui.player_controls.DetectionWorker"):
        controls._redetect_key_only()

    controls.apply_theme("light", LIGHT_COLORS)

    bar = controls.song_info_bar
    assert bar.key_label.text() == "Key: detecting..."
    assert LIGHT_COLORS["surface0"] in bar.key_label.styleSheet()
    # The tempo badge keeps its value while only the key re-detects.
    assert "~91 BPM" in bar.detected_bpm_label.text()


def test_failed_detection_does_not_come_back_on_a_theme_switch(controls):
    """The error path styled the labels itself, so the stored
    "detecting..." status reappeared at the next theme switch."""
    controls.apply_theme("dark", DARK_COLORS)
    controls.set_detected_key("C major", "medium")
    controls.song_info_bar.show_detection_status(
        "Key: detecting...", "Tempo: detecting...",
    )
    with patch.object(
        controls, "_is_active_detection_sender", return_value=True,
    ):
        controls._on_detect_error("boom")

    controls.apply_theme("light", LIGHT_COLORS)

    bar = controls.song_info_bar
    assert bar.key_label.text() == ""
    assert bar.detected_bpm_label.text() == ""


def test_tempo_tooltip_keeps_the_precise_value(controls):
    result = MagicMock()
    result.bpm, result.bpm_confidence = 91.4, "low"
    result.key, result.key_confidence = "", ""
    result.beat_times, result.downbeat_times = [], []
    result.chord_sequence = []
    with patch.object(
        controls, "_is_active_detection_sender", return_value=True,
    ):
        controls._on_detect_completed(result)
    controls.apply_theme("light", LIGHT_COLORS)

    assert "91.4 BPM" in controls.song_info_bar.detected_bpm_label.toolTip()


def test_practice_card_controls_are_compact(qapp):
    """Shorter card controls leave the waveform more height (#159)."""
    rack = PracticeRack(SongInfoBar())
    rack.setStyleSheet(get_stylesheet("dark"))
    rack.resize(1200, 300)
    rack.show()
    QApplication.processEvents()
    assert rack._loop_a_button.height() <= 28
    assert rack._metronome_toggle.height() == 28
    rack.close()
    rack.deleteLater()


@pytest.mark.parametrize("available, expected", [
    ((1920, 1040), (1280, 820)),
    ((1366, 728), (1229, 655)),
    ((1000, 640), (900, 600)),
])
def test_first_run_window_size(qapp, available, expected):
    stub = MagicMock()
    stub.screen.return_value.availableGeometry.return_value = QRect(
        0, 0, *available,
    )
    stub.frameGeometry.return_value.height.return_value = expected[1] + 39
    stub.geometry.return_value.height.return_value = expected[1]
    MainWindow._apply_first_run_size(stub)
    stub.resize.assert_called_once_with(*expected)
    x, y = stub.move.call_args.args
    assert x >= 0 and y >= 0


def test_first_run_window_stays_on_a_short_screen():
    """A 1920x1080 screen at 200% is 516 px tall: the title bar went off."""
    stub = MagicMock()
    stub.screen.return_value.availableGeometry.return_value = QRect(
        0, 0, 960, 516,
    )
    stub.frameGeometry.return_value.height.return_value = 639
    stub.geometry.return_value.height.return_value = 600
    MainWindow._apply_first_run_size(stub)
    assert stub.move.call_args.args == (30, 0)


def test_practice_icon_buttons_stay_square(qapp):
    rack = PracticeRack(SongInfoBar())
    rack.setStyleSheet(get_stylesheet("light"))
    rack.resize(1200, 300)
    rack.show()
    QApplication.processEvents()
    for button in (rack._metronome_toggle, rack._count_in_toggle):
        assert button.width() == button.height() == 28
    rack.close()
    rack.deleteLater()

def test_loop_export_uses_song_time_at_any_speed():
    """Loop points are song time; export multiplied them by the speed."""
    from src.ui.main_window import loop_export_frames

    assert loop_export_frames(60.0, 70.0, 44100) == (60 * 44100, 70 * 44100)

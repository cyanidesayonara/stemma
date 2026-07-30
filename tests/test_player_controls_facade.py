"""Characterization tests for the extracted PlayerControls facade."""

from importlib import import_module
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from src.ui.player_controls import PlayerControls
from src.ui.styles import DARK_COLORS, LIGHT_COLORS


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
    assert controls._waveform is controls.transport_bar.waveform
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


def test_extraction_preserves_visual_control_order(controls):
    """The extraction boundary must not recompose the shipped interface."""
    _component_types()
    expected = [
        controls._play_btn,
        controls._stop_btn,
        controls._record_btn,
        controls._time_label,
        controls._master_vol_label_prefix,
        controls._master_volume_slider,
        controls._master_volume_label,
        controls._count_in_label,
        controls._ci_label,
        controls._count_in_toggle,
        controls._count_in_beats_spin,
        controls._count_in_repeats_cb,
        controls._waveform_frame,
        controls._loop_a_btn,
        controls._loop_b_btn,
        controls._loop_toggle_btn,
        controls._loop_clear_btn,
        controls._loop_label,
        controls._key_label,
        controls._chord_label,
        controls._speed_label,
        controls._speed_combo,
        controls._speed_status,
        controls._pitch_label,
        controls._pitch_spin,
        controls._trainer_check,
        controls._trainer_start_combo,
        controls._trainer_status,
        controls._metro_label,
        controls._metronome_toggle,
        controls._bpm_spin,
        controls._tap_btn,
        controls._beat_sync_btn,
        controls._beat_nudge_spin,
        controls._metronome_vol_slider,
        controls._metronome_vol_combo,
        controls._detected_bpm_label,
        controls._mixer_label,
        controls._stems_frame,
        controls._recordings_label,
        controls._recordings_frame,
    ]
    markers = set(expected)
    actual = [
        widget for widget in _layout_widgets(controls) if widget in markers
    ]

    assert actual == expected


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

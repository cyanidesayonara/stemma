"""Tab order must follow visual top-to-bottom layout (#188)."""

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    library = MagicMock()
    library.songs = []
    player = MagicMock()
    player.has_unsaved_take = False
    player.current_seconds = 0.0
    player.total_seconds = 120.0
    player.is_playing = False
    player.master_volume = 1.0
    player.speed = 1.0
    player.pitch_semitones = 0
    player.volumes = {}
    player.nudge_offsets = {}
    player.muted_stems = set()
    player.soloed_stems = set()
    player.loop_a = None
    player.loop_b = None
    player.looping = False
    player.metronome_bpm = 120
    player.metronome_enabled = False
    player.metronome_volume = 1.0
    player.count_in_enabled = False
    player.count_in_beats = 4
    player.count_in_on_repeats = False
    player.get_volume.return_value = 1.0
    win = MainWindow(library, player, MagicMock())
    win.resize(1200, 800)
    win.show()
    win.activateWindow()
    QApplication.processEvents()
    yield win
    win.close()
    win.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()


def _show_loaded_player(
    window: MainWindow,
    stem_names: tuple[str, ...] = ("vocals", "drums", "bass", "other"),
) -> None:
    pc = window.player_controls
    pc._empty_widget.setVisible(False)
    pc._controls_scroll.setVisible(True)
    pc._transport_bar.setVisible(True)
    pc.stem_mixer.set_stem_names(list(stem_names))
    QApplication.processEvents()


def _section_of(window: MainWindow, widget) -> str:
    pc = window.player_controls
    containers = (
        ("lib", window._library_panel),
        ("wave", pc.waveform_panel),
        ("rack", pc.practice_rack),
        ("mixer", pc.stem_mixer),
        ("transport", pc.transport_bar),
    )
    for name, container in containers:
        if widget is container or container.isAncestorOf(widget):
            return name
    return "other"


def _walk_tab_order(window: MainWindow, start_widget, limit: int = 200):
    """Return every widget visited by pressing Tab from *start_widget* once around."""
    start_widget.setFocus()
    QApplication.processEvents()
    first = QApplication.focusWidget()
    assert first is not None
    order = [first]
    for _ in range(limit):
        QTest.keyClick(window.windowHandle(), Qt.Key.Key_Tab)
        QApplication.processEvents()
        current = QApplication.focusWidget()
        if current is first:
            return order
        assert current not in order, f"Tab cycle skipped start at {current!r}"
        order.append(current)
    raise AssertionError("Tab order did not wrap back to start widget")


def _collapsed_sections(window: MainWindow, widgets) -> list[str]:
    sections: list[str] = []
    for widget in widgets:
        section = _section_of(window, widget)
        assert section != "other", f"Unexpected focus stop outside panels: {widget!r}"
        if not sections or sections[-1] != section:
            sections.append(section)
    return sections


def _widgets_in_section(window: MainWindow, widgets, section: str) -> list:
    return [w for w in widgets if _section_of(window, w) == section]


def _belongs_to(target, focus_widget) -> bool:
    return focus_widget is target or target.isAncestorOf(focus_widget)


def test_tab_order_visits_sections_top_to_bottom(window):
    _show_loaded_player(window)
    order = _walk_tab_order(window, window._library_panel._search_edit)

    assert _collapsed_sections(window, order) == [
        "lib", "wave", "rack", "mixer", "transport",
    ]

    transport = window.player_controls.transport_bar
    expected_transport = [
        transport.play_button,
        transport.stop_button,
        transport.record_button,
        transport.master_volume_slider,
        transport.master_volume_combo,
    ]
    actual_transport = _widgets_in_section(window, order, "transport")
    assert len(actual_transport) == len(expected_transport)
    for actual, expected in zip(actual_transport, expected_transport):
        assert _belongs_to(expected, actual)

    mixer_widgets = _widgets_in_section(window, order, "mixer")
    stem_rows = window.player_controls.stem_mixer.stem_rows
    row_positions = [
        next(
            i for i, w in enumerate(mixer_widgets)
            if _belongs_to(stem_rows[name], w)
        )
        for name in ("vocals", "drums", "bass", "other")
    ]
    assert row_positions == sorted(row_positions)


def test_tab_order_keeps_transport_last_after_adding_recording_rows(window):
    _show_loaded_player(window)
    mixer = window.player_controls.stem_mixer
    mixer.add_recording_row("recording_1", "Take 1")
    mixer.add_recording_row("recording_2", "Take 2")
    QApplication.processEvents()

    order = _walk_tab_order(window, window._library_panel._search_edit)
    assert _collapsed_sections(window, order) == [
        "lib", "wave", "rack", "mixer", "transport",
    ]

    mixer_widgets = _widgets_in_section(window, order, "mixer")
    rec1 = mixer.recording_rows["recording_1"]
    rec2 = mixer.recording_rows["recording_2"]
    last_source_idx = max(
        i for i, w in enumerate(mixer_widgets)
        if _belongs_to(mixer.stem_rows["other"], w)
    )
    rec1_indices = [
        i for i, w in enumerate(mixer_widgets) if _belongs_to(rec1, w)
    ]
    rec2_indices = [
        i for i, w in enumerate(mixer_widgets) if _belongs_to(rec2, w)
    ]
    assert rec1_indices and rec2_indices
    assert last_source_idx < min(rec1_indices)
    assert max(rec1_indices) < min(rec2_indices)
    assert _belongs_to(rec2._delete_btn, mixer_widgets[-1])


def test_tab_order_survives_song_switch_and_six_stems(window):
    _show_loaded_player(window, ("vocals", "drums"))
    mixer = window.player_controls.stem_mixer
    six_stems = ["vocals", "drums", "bass", "guitar", "piano", "other"]
    mixer.set_stem_names(six_stems)
    QApplication.processEvents()

    order = _walk_tab_order(window, window._library_panel._search_edit)
    assert _collapsed_sections(window, order) == [
        "lib", "wave", "rack", "mixer", "transport",
    ]

    mixer_widgets = _widgets_in_section(window, order, "mixer")
    row_positions = [
        next(
            i for i, w in enumerate(mixer_widgets)
            if _belongs_to(mixer.stem_rows[name], w)
        )
        for name in six_stems
    ]
    assert row_positions == sorted(row_positions)


def test_tab_order_after_removing_and_clearing_recording_rows(window):
    _show_loaded_player(window)
    mixer = window.player_controls.stem_mixer
    mixer.add_recording_row("recording_1", "Take 1")
    mixer.add_recording_row("recording_2", "Take 2")
    mixer.remove_recording_row("recording_1")
    QApplication.processEvents()

    order = _walk_tab_order(window, window._library_panel._search_edit)
    assert _collapsed_sections(window, order) == [
        "lib", "wave", "rack", "mixer", "transport",
    ]
    mixer_widgets = _widgets_in_section(window, order, "mixer")
    assert _belongs_to(
        mixer.recording_rows["recording_2"]._delete_btn, mixer_widgets[-1]
    )

    mixer.clear_recording_rows()
    QApplication.processEvents()
    order = _walk_tab_order(window, window._library_panel._search_edit)
    assert _collapsed_sections(window, order) == [
        "lib", "wave", "rack", "mixer", "transport",
    ]

    mixer.set_stem_names([])
    QApplication.processEvents()
    order = _walk_tab_order(window, window._library_panel._search_edit)
    assert _collapsed_sections(window, order) == [
        "lib", "wave", "rack", "transport",
    ]


def test_backtab_order_moves_backward_through_transport_mixer_and_rack(window):
    _show_loaded_player(window)
    pc = window.player_controls
    mixer = pc.stem_mixer
    transport = pc.transport_bar
    rec = mixer.add_recording_row("recording_1", "Take 1")
    QApplication.processEvents()

    window._library_panel._search_edit.setFocus()
    QApplication.processEvents()

    expected_reverse = [
        transport.master_volume_combo,
        transport.master_volume_slider,
        transport.record_button,
        transport.stop_button,
        transport.play_button,
        rec._delete_btn,
    ]
    for expected in expected_reverse:
        QTest.keyClick(window.windowHandle(), Qt.Key.Key_Backtab)
        QApplication.processEvents()
        assert _belongs_to(expected, QApplication.focusWidget())

    mixer.set_stem_names([])
    QApplication.processEvents()
    transport.play_button.setFocus()
    QApplication.processEvents()
    QTest.keyClick(window.windowHandle(), Qt.Key.Key_Backtab)
    QApplication.processEvents()
    assert _belongs_to(pc.practice_rack._count_in_repeats, QApplication.focusWidget())


def test_tab_order_at_minimum_window_size_with_wrapped_cards(window):
    window.resize(900, 600)
    _show_loaded_player(window)
    pc = window.player_controls
    assert not pc.practice_rack.cards_side_by_side

    order = _walk_tab_order(window, window._library_panel._search_edit)
    assert _collapsed_sections(window, order) == [
        "lib", "wave", "rack", "mixer", "transport",
    ]

    rack = pc.practice_rack
    rack_widgets = _widgets_in_section(window, order, "rack")
    loop_idx = next(
        i for i, w in enumerate(rack_widgets)
        if _belongs_to(rack._loop_card, w)
    )
    speed_idx = next(
        i for i, w in enumerate(rack_widgets)
        if _belongs_to(rack._speed_card, w)
    )
    metro_idx = next(
        i for i, w in enumerate(rack_widgets)
        if _belongs_to(rack._metronome_card, w)
    )
    assert loop_idx < speed_idx < metro_idx
    assert _belongs_to(rack._count_in_repeats, rack_widgets[-1])

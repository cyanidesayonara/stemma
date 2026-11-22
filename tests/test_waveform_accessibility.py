"""The waveform as a keyboard and screen-reader slider (#188)."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import QApplication

from src.ui.waveform_stack_widget import WaveformStackWidget


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def wave(app):
    w = WaveformStackWidget()
    w.resize(600, 200)
    w.set_total_seconds(296.0)  # 4:56
    w.set_position(83.0 / 296.0)  # 1:23
    yield w
    w.deleteLater()


def _iface(widget):
    iface = QAccessible.queryAccessibleInterface(widget)
    assert iface is not None
    return iface


def test_waveform_is_a_tab_stop(wave):
    assert wave.focusPolicy() & Qt.FocusPolicy.TabFocus


def test_reports_a_named_slider(wave):
    iface = _iface(wave)
    assert iface.role() == QAccessible.Role.Slider
    assert iface.text(QAccessible.Text.Name) == "Playback position"


def test_value_text_reads_position_of_length(wave):
    assert _iface(wave).text(QAccessible.Text.Value) == "1:23 of 4:56"


def test_no_value_interface_is_claimed(wave):
    """A Python value interface crashed Qt's UI Automation bridge (an
    invalid C++ pointer from interface_cast): Narrator or any UIA client
    took the app down (#253 review). Offscreen tests cannot see that
    crash, so guard against the interface coming back."""
    iface = _iface(wave)
    assert iface.valueInterface() is None
    assert "interface_cast" not in vars(type(iface))


def test_seek_to_seconds_clamps(wave):
    seeks = []
    wave.seek_requested.connect(seeks.append)
    wave.seek_to_seconds(120.0)
    wave.seek_to_seconds(999.0)
    wave.seek_to_seconds(-5.0)
    assert seeks == [
        pytest.approx(120.0), pytest.approx(296.0), pytest.approx(0.0),
    ]


def test_no_song_reads_as_empty(app):
    w = WaveformStackWidget()
    assert _iface(w).text(QAccessible.Text.Value) == "No song loaded"
    seeks = []
    w.seek_requested.connect(seeks.append)
    w.seek_to_seconds(10.0)
    assert seeks == []
    w.deleteLater()


def test_description_names_the_loop_points(wave):
    assert "loop" not in _iface(wave).text(QAccessible.Text.Description)
    wave.set_loop_markers(12.0 / 296.0, 21.0 / 296.0)
    text = _iface(wave).text(QAccessible.Text.Description)
    assert "Loop point A at 0:12" in text
    assert "loop point B at 0:21" in text
    # UI Automation's HelpText comes from Help.
    assert _iface(wave).text(QAccessible.Text.Help) == text


def test_lone_loop_point_is_named(wave):
    wave.set_loop_markers(12.0 / 296.0, None)
    text = _iface(wave).text(QAccessible.Text.Description)
    assert "Loop point A at 0:12" in text
    assert "point B" not in text


def test_focus_ring_paints_only_with_focus(wave):
    assert not wave.draws_focus_ring()
    wave.show()
    wave.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    if not wave.hasFocus():
        pytest.skip("offscreen platform did not grant focus")
    assert wave.draws_focus_ring()
    wave.grab()  # paints without error

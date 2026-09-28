"""A focused slider, list, spinbox, or combo keeps its navigation keys.

Left, Right, Home, End, Up, and Down are window shortcuts (seek and master
volume). Qt fires window shortcuts before the focused widget sees the key,
so without a mouse no slider, spinbox, list, or combo could be adjusted: a
stem volume slider with Right seeked +5 s, and the song list with Down
lowered the master volume (#184).
"""

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.ui.main_window import MainWindow

K = Qt.Key


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    library = MagicMock()
    library.songs = []
    player = MagicMock()
    player.current_seconds = 50.0
    player.total_seconds = 100.0
    player.master_volume = 1.0
    player.is_playing = False
    win = MainWindow(library, player, MagicMock())
    win.resize(1200, 800)
    win.show()
    win.activateWindow()
    QApplication.processEvents()
    # Spy on the volume shortcut; the lambdas look it up at call time.
    win._adjust_master_volume = MagicMock()
    yield win
    win.close()
    # Delete it: a closed window keeps its timers and shortcuts alive, and
    # 46 of them slowed later tests past their waits (#197 review).
    win.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()


def _host(window, widget):
    """Put *widget* in the window, visible and focused."""
    holder = QWidget(window.centralWidget())
    layout = QVBoxLayout(holder)
    layout.addWidget(widget)
    holder.setGeometry(10, 10, 300, 200)
    holder.show()
    widget.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    assert widget.hasFocus()
    return widget


def _press(window, key, modifier=Qt.KeyboardModifier.NoModifier) -> None:
    # Through the window, so the window's shortcuts get their chance first,
    # exactly as with real key input.
    QTest.keyClick(window.windowHandle(), key, modifier)
    QApplication.processEvents()


def _assert_no_shortcut(window) -> None:
    window._player.seek.assert_not_called()
    window._adjust_master_volume.assert_not_called()


def _slider():
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 100)
    slider.setPageStep(10)
    slider.setValue(50)
    return slider


def _spin():
    spin = QSpinBox()
    spin.setRange(0, 100)
    spin.setValue(50)
    return spin


def _combo():
    combo = QComboBox()
    combo.addItems([str(i) for i in range(10)])
    combo.setCurrentIndex(5)
    return combo


def _list():
    widget = QListWidget()
    widget.addItems([f"song {i}" for i in range(10)])
    widget.setCurrentRow(5)
    return widget


@pytest.mark.parametrize("key, expected", [
    (K.Key_Left, 49),
    (K.Key_Right, 51),
    (K.Key_Up, 51),
    (K.Key_Down, 49),
    (K.Key_Home, 0),
    (K.Key_End, 100),
    (K.Key_PageUp, 60),
    (K.Key_PageDown, 40),
])
def test_focused_slider_gets_the_key(window, key, expected):
    slider = _host(window, _slider())

    _press(window, key)

    assert slider.value() == expected
    _assert_no_shortcut(window)


@pytest.mark.parametrize("key, expected", [
    (K.Key_Up, 51),
    (K.Key_Down, 49),
    (K.Key_PageUp, 60),
    (K.Key_PageDown, 40),
])
def test_focused_spinbox_gets_the_key(window, key, expected):
    spin = _host(window, _spin())

    _press(window, key)

    assert spin.value() == expected
    _assert_no_shortcut(window)


@pytest.mark.parametrize("key", [K.Key_Left, K.Key_Right, K.Key_Home, K.Key_End])
def test_focused_spinbox_keeps_cursor_keys(window, key):
    _host(window, _spin())

    _press(window, key)

    _assert_no_shortcut(window)


@pytest.mark.parametrize("key, expected", [
    (K.Key_Up, 4),
    (K.Key_Down, 6),
    (K.Key_Home, 0),
    (K.Key_End, 9),
    (K.Key_PageUp, 4),
    (K.Key_PageDown, 6),
])
def test_focused_combo_gets_the_key(window, key, expected):
    combo = _host(window, _combo())

    _press(window, key)

    assert combo.currentIndex() == expected
    _assert_no_shortcut(window)


@pytest.mark.parametrize("key", [K.Key_Left, K.Key_Right])
def test_left_right_still_seek_from_a_combo_that_ignores_them(window, key):
    """A closed combo has no use for Left/Right; the key comes back up to
    the window and seeks, as it did before the override (#197 review)."""
    _host(window, _combo())

    _press(window, key)

    window._player.seek.assert_called_once_with(
        45.0 if key == K.Key_Left else 55.0
    )


@pytest.mark.parametrize("key", [K.Key_Up, K.Key_Down])
def test_focused_editable_combo_gets_the_key(window, key):
    combo = _combo()
    combo.setEditable(True)
    _host(window, combo)

    _press(window, key)

    assert combo.currentIndex() == (4 if key == K.Key_Up else 6)
    _assert_no_shortcut(window)


@pytest.mark.parametrize("key, expected", [
    (K.Key_Up, 4),
    (K.Key_Down, 6),
    (K.Key_Home, 0),
    (K.Key_End, 9),
])
def test_focused_list_gets_the_key(window, key, expected):
    widget = _host(window, _list())

    _press(window, key)

    assert widget.currentRow() == expected
    _assert_no_shortcut(window)


@pytest.mark.parametrize("key", [K.Key_Left, K.Key_Right, K.Key_Home, K.Key_End])
def test_focused_line_edit_keeps_cursor_keys(window, key):
    edit = _host(window, QLineEdit("some text"))
    edit.setCursorPosition(4)

    _press(window, key)

    _assert_no_shortcut(window)
    assert edit.cursorPosition() != 4


@pytest.mark.parametrize("key", [K.Key_Up, K.Key_Down])
def test_focused_line_edit_swallows_up_down(window, key):
    _host(window, QLineEdit("some text"))

    _press(window, key)

    _assert_no_shortcut(window)


def test_master_volume_slider_gets_right(window):
    """The real transport slider, not only a stand-in."""
    slider = window._player_controls._master_volume_slider
    slider.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    assert slider.hasFocus()
    before = slider.value()

    _press(window, K.Key_Right)

    assert slider.value() == before + 1
    window._player.seek.assert_not_called()


def test_song_list_gets_down(window):
    """The real library list: Down moves the selection, not the volume."""
    widget = window._library_panel._list
    widget.addItems(["a", "b", "c"])
    widget.setCurrentRow(0)
    widget.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    assert widget.hasFocus()

    _press(window, K.Key_Down)

    assert widget.currentRow() == 1
    window._adjust_master_volume.assert_not_called()


# -- Shortcuts still work elsewhere ------------------------------------------

@pytest.mark.parametrize("key, seconds", [
    (K.Key_Left, 45.0),
    (K.Key_Right, 55.0),
    (K.Key_Home, 0.0),
    (K.Key_End, 100.0),
])
def test_seek_shortcuts_fire_with_a_button_focused(window, key, seconds):
    _host(window, QPushButton("button"))

    _press(window, key)

    window._player.seek.assert_called_once_with(seconds)


@pytest.mark.parametrize("key, delta", [(K.Key_Up, 0.05), (K.Key_Down, -0.05)])
def test_volume_shortcuts_fire_with_a_button_focused(window, key, delta):
    _host(window, QPushButton("button"))

    _press(window, key)

    window._adjust_master_volume.assert_called_once_with(delta)


def test_seek_shortcut_fires_with_nothing_focused(window):
    focused = QApplication.focusWidget()
    if focused is not None:
        focused.clearFocus()
    QApplication.processEvents()

    _press(window, K.Key_Right)

    window._player.seek.assert_called_once_with(55.0)


def test_shift_arrow_still_bumps_pitch_from_a_slider(window):
    """Modifier combos are not the widget's keys; they stay global."""
    slider = _host(window, _slider())
    bump = MagicMock()
    window._player_controls.bump_pitch = bump

    _press(window, K.Key_Right, Qt.KeyboardModifier.ShiftModifier)

    bump.assert_called_once_with(1)
    assert slider.value() == 50


def test_right_on_the_song_list_still_seeks(window):
    """Click a song, then Right to skip ahead: the list ignores Right."""
    widget = window._library_panel._list
    widget.addItems(["a", "b"])
    widget.setCurrentRow(0)
    widget.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()

    _press(window, K.Key_Right)

    window._player.seek.assert_called_once_with(55.0)
    assert widget.currentRow() == 0

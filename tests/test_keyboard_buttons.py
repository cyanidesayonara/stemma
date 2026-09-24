"""The keyboard must be able to press a focused button.

Space was only the global play/pause shortcut, and Qt fires window
shortcuts before the focused widget sees the key, so a keyboard user could
tab to Repeat or Mute and see it focused (#165) but not press it. Enter
did nothing on these buttons either.
"""

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.ui.library_panel import REPEAT_ALL, REPEAT_OFF
from src.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    library = MagicMock()
    library.songs = []
    player = MagicMock()
    player.current_seconds = 0.0
    player.is_playing = False
    win = MainWindow(library, player, MagicMock())
    win.resize(1200, 800)
    win.show()
    win.activateWindow()
    QApplication.processEvents()
    yield win
    win.close()
    QApplication.processEvents()


def _focus(widget) -> None:
    widget.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    assert widget.hasFocus()


def _press(window, key) -> None:
    # Through the window, so the window's shortcuts get their chance first,
    # exactly as with real key input.
    QTest.keyClick(window.windowHandle(), key)
    QTest.qWait(250)  # animateClick releases the button after ~100 ms.


@pytest.mark.parametrize("key", [Qt.Key.Key_Space, Qt.Key.Key_Return])
def test_key_presses_the_focused_button(window, key):
    panel = window._library_panel
    _focus(panel._repeat_btn)
    assert panel._repeat_mode == REPEAT_OFF

    _press(window, key)

    assert panel._repeat_mode == REPEAT_ALL
    window._player.play.assert_not_called()


def test_space_on_a_focused_toggle_does_not_also_play(window):
    shuffle = window._library_panel._shuffle_btn
    _focus(shuffle)

    _press(window, Qt.Key.Key_Space)

    assert shuffle.isChecked()
    window._player.play.assert_not_called()
    window._player.pause.assert_not_called()


def test_space_still_plays_when_no_button_has_focus(window):
    _focus(window._library_panel._list)

    _press(window, Qt.Key.Key_Space)

    window._player.play.assert_called_once_with()

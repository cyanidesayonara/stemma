"""The keyboard must be able to press a focused button.

Space was only the global play/pause shortcut, and Qt fires window
shortcuts before the focused widget sees the key, so a keyboard user could
tab to Repeat or Mute and see it focused (#165) but not press it. Enter
did nothing on these buttons either.
"""

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from src.ui.library_panel import REPEAT_ALL, REPEAT_OFF
from src.ui.main_window import MainWindow
from src.ui.stem_mixer import StemRow


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


def _tab_to(window, widget, limit: int = 200) -> None:
    """Press real Tab keys until *widget* has focus.

    setFocus(TabFocusReason) would not do: the window tracks keyboard focus
    from Tab navigation itself.
    """
    for _ in range(limit):
        if widget.hasFocus():
            return
        QTest.keyClick(window.windowHandle(), Qt.Key.Key_Tab)
        QApplication.processEvents()
    raise AssertionError(f"could not tab to {widget.accessibleName()!r}")


def _press(window, key) -> None:
    # Through the window, so the window's shortcuts get their chance first,
    # exactly as with real key input.
    QTest.keyClick(window.windowHandle(), key)
    QTest.qWait(250)  # animateClick releases the button after ~100 ms.


@pytest.mark.parametrize("key", [Qt.Key.Key_Space, Qt.Key.Key_Return])
def test_key_presses_the_focused_button(window, key):
    panel = window._library_panel
    _tab_to(window, panel._repeat_btn)
    assert panel._repeat_mode == REPEAT_OFF

    _press(window, key)

    assert panel._repeat_mode == REPEAT_ALL
    window._player.play.assert_not_called()


def test_space_on_a_focused_toggle_does_not_also_play(window):
    shuffle = window._library_panel._shuffle_btn
    _tab_to(window, shuffle)

    _press(window, Qt.Key.Key_Space)

    assert shuffle.isChecked()
    window._player.play.assert_not_called()
    window._player.pause.assert_not_called()


def test_space_still_plays_when_no_button_has_focus(window):
    window._library_panel._list.setFocus()
    QApplication.processEvents()

    _press(window, Qt.Key.Key_Space)

    window._player.play.assert_called_once_with()


def test_space_after_a_mouse_click_still_plays(window):
    """Buttons take focus on click too. Pressing the clicked button again on
    Space broke the core loop of clicking a control, then Space to play."""
    shuffle = window._library_panel._shuffle_btn
    QTest.mouseClick(shuffle, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    assert shuffle.hasFocus() and shuffle.isChecked()

    _press(window, Qt.Key.Key_Space)

    window._player.play.assert_called_once_with()
    assert shuffle.isChecked(), "Space pressed the clicked button again"


def test_space_after_clicking_away_from_a_tabbed_button_plays(window):
    """Keyboard focus ends when focus moves on by mouse."""
    panel = window._library_panel
    _tab_to(window, panel._repeat_btn)
    QTest.mouseClick(panel._shuffle_btn, Qt.MouseButton.LeftButton)
    QApplication.processEvents()
    mode = panel._repeat_mode

    _press(window, Qt.Key.Key_Space)

    assert panel._repeat_mode == mode
    window._player.play.assert_called_once_with()


def test_space_after_window_reactivation_still_plays(window):
    """Qt restores focus when the window is activated again; that restored
    focus is not keyboard focus, even though no mouse button is held."""
    shuffle = window._library_panel._shuffle_btn
    QTest.mouseClick(shuffle, Qt.MouseButton.LeftButton)
    other = QWidget()
    other.show()
    other.activateWindow()
    QApplication.processEvents()
    window.activateWindow()
    QApplication.processEvents()
    shuffle.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
    QApplication.processEvents()

    _press(window, Qt.Key.Key_Space)

    assert shuffle.isChecked(), "Space re-pressed the restored button"
    window._player.play.assert_called_once_with()
    other.close()


def test_programmatic_focus_is_not_keyboard_focus(window):
    """Focus the app moves itself, such as after a dialog closes."""
    panel = window._library_panel
    panel._repeat_btn.setFocus(Qt.FocusReason.PopupFocusReason)
    QApplication.processEvents()
    mode = panel._repeat_mode

    _press(window, Qt.Key.Key_Space)

    assert panel._repeat_mode == mode
    window._player.play.assert_called_once_with()


def test_holding_space_does_not_auto_repeat(window):
    """An auto-repeating Space flipped a focused toggle ~30 times a second."""
    space = [
        shortcut for shortcut in window.findChildren(QShortcut)
        if shortcut.key() == QKeySequence(Qt.Key.Key_Space)
    ]
    assert space and not any(s.autoRepeat() for s in space)


def test_keyboard_pressed_mute_keeps_focus(app):
    """Mute and Solo dropped focus after every toggle, so the next Tab
    started over from the top and the next Space played instead."""
    player = MagicMock()
    player.muted_stems = set()
    player.soloed_stems = set()
    player.volumes = {}
    row = StemRow("drums", player)
    row.show()
    row._mute_btn.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()

    row._mute_btn.click()
    QApplication.processEvents()

    assert row._mute_btn.hasFocus()
    row.close()

"""Menu accelerators and the shortcuts dialog that lists them (#188)."""

import pytest
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication

from src.ui.styles import get_stylesheet
from tests.test_section_ruler import _main_window
from tests.widget_visual import deterministic_render_state


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    win = _main_window()
    yield win
    win.deleteLater()


def _menu_action(window, text: str):
    matches = [a for a in window.findChildren(QAction) if a.text() == text]
    assert len(matches) == 1, f"{text!r}: {len(matches)} actions"
    return matches[0]


@pytest.mark.parametrize(("text", "keys"), [
    ("&Import Song…", "Ctrl+O"),
    ("&Quit", "Ctrl+Q"),
    ("&Preferences…", "Ctrl+,"),
])
def test_menu_items_have_accelerators(window, text, keys):
    action = _menu_action(window, text)
    assert action.shortcut() == QKeySequence(keys)


def test_accelerators_are_listed_in_the_shortcuts_dialog(window):
    listed = {
        key for column in window._SHORTCUT_COLUMNS
        for _, rows in column for key, _ in rows
    }
    assert {"Ctrl+O", "Ctrl+,", "Ctrl+Q", "F1"} <= listed


def test_accelerators_do_not_clash_with_window_shortcuts(window):
    taken = {s.key().toString() for s in window.findChildren(QShortcut)}
    for keys in ("Ctrl+O", "Ctrl+,", "Ctrl+Q"):
        assert QKeySequence(keys).toString() not in taken


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_shortcuts_dialog_fits_a_1366x768_screen(app, window, theme):
    """About 728 px is left after the taskbar, title bar included (#186)."""
    with deterministic_render_state():
        app.setStyleSheet(get_stylesheet(theme))
        dialog = window._build_shortcuts_dialog()
        assert dialog.sizeHint().height() <= 690
        dialog.deleteLater()

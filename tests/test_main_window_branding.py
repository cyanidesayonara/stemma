"""Author credit, clickable logos, and the player's right-hand stop."""

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QPointingDevice
from PySide6.QtWidgets import QApplication, QLabel

from src.app_settings import SETTINGS_FILE_ENV
from src.ui import _wav_playback_impl
from src.ui.animated_arpeggio import AnimatedArpeggioWidget
from src.ui.main_window import MainWindow
from src.ui.styles import get_stylesheet


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    # A settings file of its own: closing saves the session, and a mock
    # player's values would otherwise land in the suite's shared store.
    monkeypatch.setenv(SETTINGS_FILE_ENV, str(tmp_path / "settings.ini"))
    player = MagicMock()
    player.has_unsaved_take = False
    player.stems = {}
    player.muted_stems = set()
    player.soloed_stems = set()
    player.volumes = {}
    player.total_seconds = 0.0
    player.current_seconds = 0.0
    player.loop_a = None
    player.loop_b = None
    player.is_playing = False
    player.has_stems = False
    library = MagicMock()
    library.songs = []
    win = MainWindow(library, player, MagicMock())
    yield win
    win._player_controls.shutdown()
    win.hide()
    win.deleteLater()


@pytest.fixture
def played(monkeypatch):
    """Paths handed to the platform sound player, which is mocked out."""
    paths = []
    monkeypatch.setattr(_wav_playback_impl, "play_impl", paths.append)
    return paths


def _click(widget) -> None:
    pos = QPointF(10, 10)
    widget.mousePressEvent(QMouseEvent(
        QEvent.Type.MouseButtonPress, pos, pos, pos,
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QPointingDevice.primaryPointingDevice(),
    ))
    widget._timer.stop()


def test_main_window_carries_no_author_line(window):
    texts = [label.text() for label in window.findChildren(QLabel)]
    assert not any("Nyk" in text or "©" in text for text in texts)


def test_about_dialog_credits_the_author(window):
    dlg = window._build_about_dialog()
    text = dlg.info_label.text().replace("&nbsp;", " ")
    assert "Made by Santtu Nykänen" in text
    assert text.count("Nykänen") == 1
    dlg.deleteLater()


def test_about_logo_click_plays_the_chord(window, played):
    dlg = window._build_about_dialog()
    _click(dlg.logo)
    assert [path.name for path in played] == ["chord.wav"]
    dlg.deleteLater()


def test_empty_player_logo_click_plays_the_chord(window, played):
    _click(window._player_controls._empty_logo)
    assert [path.name for path in played] == ["chord.wav"]


def test_transport_logo_click_plays_the_jingle(window, played):
    logo = window._player_controls.transport_bar.logo
    assert isinstance(logo, AnimatedArpeggioWidget)
    _click(logo)
    assert [path.name for path in played] == ["arpeggio.wav"]


def test_logo_closes_the_transport_row(window):
    """The logo is the last widget in the row, right-aligned by a stretch."""
    transport = window._player_controls.transport_bar
    row = transport.layout().itemAt(0).layout()
    last = row.itemAt(row.count() - 1).widget()
    before = row.itemAt(row.count() - 2)
    assert last is transport.logo
    assert before.spacerItem() is not None


def test_player_side_stops_at_the_transport_row(window, app):
    """Dragging the splitter right stops before the logo is squeezed."""
    window.setStyleSheet(get_stylesheet("dark"))
    controls = window._player_controls
    controls._empty_widget.setVisible(False)
    controls._controls_scroll.setVisible(True)
    controls._transport_bar.setVisible(True)
    window.resize(1360, 820)
    window.show()
    app.processEvents()
    splitter = window._splitter
    assert not splitter.isCollapsible(1)

    splitter.moveSplitter(window.width(), 1)
    app.processEvents()

    transport = controls.transport_bar
    assert splitter.sizes()[1] > 0
    assert transport.width() >= transport.minimumSizeHint().width()
    assert transport.logo.geometry().right() < transport.width()

"""The chord badge must follow the pitch shift, like the key badge (#174)."""

from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QApplication

from src.ui.player_controls import PlayerControls
from src.ui.styles import DARK_COLORS


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controls(qapp):
    player = MagicMock()
    player.stems = {}
    player.muted_stems = set()
    player.soloed_stems = set()
    player.volumes = {}
    player.beat_times = []
    player.chord_sequence = [(0.0, "C")]
    player.chord_at.return_value = "C"
    player.total_seconds = 10.0
    player.current_seconds = 1.0
    player.sample_rate = 44100
    player.loop_a = None
    player.loop_b = None
    player.is_playing = True
    player.speed = 1.0
    player.pitch_semitones = -2
    result = PlayerControls(player)
    yield result
    result.shutdown()
    result.setParent(None)
    result.deleteLater()
    qapp.processEvents()


def test_playing_chord_badge_is_transposed(controls):
    controls._update_chord_label()

    assert "Bb" in controls._chord_label.text()


def test_theme_change_keeps_the_chord_transposed(controls):
    controls.apply_theme("dark", DARK_COLORS)

    assert "Bb" in controls._chord_label.text()


def test_untransposed_chord_is_unchanged(controls):
    controls._player.pitch_semitones = 0

    controls._update_chord_label()

    assert "C" in controls._chord_label.text()
    assert "Bb" not in controls._chord_label.text()

"""Tests for per-song practice state and its library row badges."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from src import player as player_module
from src.app_settings import open_settings
from src.library import SongLibrary
from src.practice_state import PracticeState
from src.ui import main_window as main_window_module
from src.ui.library_panel import _PRACTICE_ROLE, _STATUS_ROLE

SAMPLE_RATE = 8000


class _Worker:
    """A finished stem load, as MainWindow's completion handler sees it."""

    def __init__(self, window, song_id):
        self.generation = window._stem_load_generation
        self.song_id = song_id
        self.source_stem_names = ("vocals",)


class _QuietMainWindow(main_window_module.MainWindow):
    """MainWindow without the startup prune."""

    def _prune_incomplete_songs(self) -> None:
        pass


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
    lib = SongLibrary(str(tmp_path / "data"))
    for i in range(2):
        wav = tmp_path / f"{i}.wav"
        sf.write(str(wav), np.zeros((64, 2), dtype=np.float32), SAMPLE_RATE)
        song = lib.add_song(f"Song {i}", f"Artist {i}", str(wav))
        sf.write(
            f"{song.stems_path}/vocals.wav",
            np.zeros((64, 2), dtype=np.float32),
            SAMPLE_RATE,
        )
    settings = open_settings()
    settings.clear()
    player = player_module.MultiTrackPlayer()
    with patch.object(main_window_module.QTimer, "singleShot"):
        win = _QuietMainWindow(lib, player, MagicMock())
    win._player_controls.start_detection = lambda *a, **k: None
    yield win
    win._shutdown_stem_loads()
    win._separation_queue.shutdown(0)
    win._player_controls.shutdown()
    player.shutdown(wait_ms=5000)
    win.setParent(None)
    win.deleteLater()
    qapp.processEvents()
    settings.clear()


def _open(window, song_id, *, autoplay=False):
    """Open *song_id* with ten seconds of audio, without a worker thread."""
    if window._current_song_id is not None:
        window._save_session()
    window._player.unload()
    window._invalidate_stem_load()
    window._current_song_id = None
    window._loading_song_id = song_id
    worker = _Worker(window, song_id)
    window._stem_load_worker = worker
    if autoplay:
        window._active_load_autoplay = (worker.generation, song_id)
    stems = {"vocals": np.zeros((SAMPLE_RATE * 10, 2), dtype=np.float32)}
    with patch.object(window._player, "play"):
        window._handle_stem_load_completed(worker, stems, SAMPLE_RATE)
    window._stem_load_worker = None
    assert window._current_song_id == song_id


def _row(window, song_id):
    return window._library_panel._item_for_song(song_id)


class TestPracticeState:
    def test_round_trip(self):
        state = PracticeState(1.5, 4.0, True, 0.75, -2)
        assert PracticeState.from_json(state.to_json()) == state

    @pytest.mark.parametrize("raw", [
        None, "", "not json", "[]", "3", json.dumps({
            "loop_a": -1, "loop_b": "x", "looping": "yes",
            "speed": 9, "pitch": 1.5,
        }), '{"loop_a": NaN, "loop_b": true, "speed": true, "pitch": true}',
    ])
    def test_malformed_falls_back_to_defaults(self, raw):
        assert PracticeState.from_json(raw).is_default

    def test_pitch_is_clamped(self):
        assert PracticeState.from_json('{"pitch": 99}').pitch == 7

    def test_badge_and_description(self):
        state = PracticeState(12.0, 75.0, False, 0.75, 1)
        assert state.badge_text() == "A-B 0.75x +1"
        assert state.description_lines() == [
            "Loop 0:12 to 1:15", "Speed 0.75x", "Pitch +1 semitone",
        ]
        assert PracticeState(loop_a=3.0).badge_text() == ""


class TestWindow:
    def test_each_song_keeps_its_own_state(self, window):
        first, second = [s.id for s in window._library.songs]
        _open(window, first)
        window._player.seek(2.0)
        window._player_controls.set_loop_a()
        window._player.seek(5.0)
        window._player_controls.set_loop_b()
        window._player_controls.toggle_looping()
        window._player.set_speed(0.75)
        window._player.set_pitch(2)
        assert _row(window, first).data(_PRACTICE_ROLE) == "A-B 0.75x +2"

        _open(window, second)
        player, controls = window._player, window._player_controls
        assert (player.loop_a, player.loop_b, player.looping) == (
            None, None, False,
        )
        assert (player.speed, player.pitch_semitones) == (1.0, 0)
        # The loop switch no longer shows the previous song's state.
        assert not controls._loop_toggle_btn.isChecked()
        assert _row(window, second).data(_PRACTICE_ROLE) == ""

        _open(window, first)
        assert (player.loop_a, player.loop_b, player.looping) == (
            2.0, 5.0, True,
        )
        assert (player.speed, player.pitch_semitones) == (0.75, 2)
        assert player.current_seconds == 2.0
        assert controls._loop_toggle_btn.isChecked()
        assert controls._speed_combo.currentText() == "0.75x"
        assert controls._pitch_spin.value() == 2

    def test_autoplay_restores_points_but_does_not_loop(self, window):
        first, second = [s.id for s in window._library.songs]
        stored = PracticeState(2.0, 5.0, True, 0.75, 0)
        window._settings.setValue(f"practice/{first}", stored.to_json())
        _open(window, first, autoplay=True)
        assert window._player.loop_a == 2.0
        assert window._player.looping is False
        assert window._player.current_seconds == 0.0
        assert window._player.speed == 0.75
        # Passing through in a setlist does not disarm the saved loop.
        _open(window, second)
        assert window._practice_state(first) == stored

    def test_loop_switch_matches_player_for_equal_consecutive_songs(
        self, window,
    ):
        for song in window._library.songs:
            window._settings.setValue(
                f"practice/{song.id}",
                PracticeState(1.0, 3.0, True).to_json(),
            )
        for song in window._library.songs:
            _open(window, song.id)
            assert window._player.looping is True

    def test_resetting_to_defaults_clears_the_stored_state(self, window):
        song_id = window._library.songs[0].id
        _open(window, song_id)
        window._player.set_speed(0.5)
        assert window._settings.contains(f"practice/{song_id}")
        window._player.set_speed(1.0)
        assert not window._settings.contains(f"practice/{song_id}")
        assert _row(window, song_id).data(_PRACTICE_ROLE) == ""

    def test_closing_a_song_keeps_its_state(self, window):
        song_id = window._library.songs[0].id
        _open(window, song_id)
        window._player.set_pitch(-3)
        window._on_close_song()
        assert window._practice_state(song_id).pitch == -3

    def test_removing_a_song_forgets_its_state(self, window):
        song_id = window._library.songs[0].id
        _open(window, song_id)
        window._player.set_pitch(-3)
        window._on_song_removal_requested(song_id)
        window._library.remove_song(song_id)
        window._on_song_removed(song_id)
        assert not window._settings.contains(f"practice/{song_id}")

    def test_status_badge_and_spoken_details(self, window):
        song_id = window._library.songs[0].id
        window._library.update_song(
            song_id, practice_status="learning", tags=["gig"],
        )
        window._settings.setValue(
            f"practice/{song_id}", PracticeState(speed=0.75).to_json(),
        )
        window._library_panel.refresh()
        item = _row(window, song_id)
        assert item.data(_STATUS_ROLE) == "Learning"
        details = "Status: Learning, Tags: gig, Speed 0.75x"
        assert (
            item.data(Qt.ItemDataRole.AccessibleDescriptionRole) == details
        )
        assert item.toolTip().splitlines()[1:] == details.split(", ")

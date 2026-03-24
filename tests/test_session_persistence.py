"""Tests for session persistence (save/restore player state)."""

import json
import os

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

from src.library import SongLibrary
from src.player import MultiTrackPlayer
from src.ui.library_panel import LibraryPanel
from src.ui.player_controls import PlayerControls


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def player():
    return MultiTrackPlayer()


@pytest.fixture
def library(tmp_path):
    return SongLibrary(str(tmp_path / "data"))


@pytest.fixture
def controls(qapp, player):
    return PlayerControls(player)


@pytest.fixture
def panel(qapp, library):
    return LibraryPanel(library)


def _make_song_with_stems(library, tmp_path, title="Test Song"):
    """Create a song with 4-stem WAV files and return it."""
    audio = tmp_path / f"{title}.wav"
    sf.write(str(audio), np.zeros((44100 * 5, 2), dtype=np.float32), 44100)
    song = library.add_song(title, "Artist", str(audio))
    os.makedirs(song.stems_path, exist_ok=True)
    for stem in ["vocals", "drums", "bass", "other"]:
        sf.write(
            os.path.join(song.stems_path, f"{stem}.wav"),
            np.zeros((44100, 2), dtype=np.float32),
            44100,
        )
    library.update_song(song.id, model_used="htdemucs")
    return song


# -----------------------------------------------------------------------
# LibraryPanel.select_song
# -----------------------------------------------------------------------


class TestLibraryPanelSelectSong:
    def test_select_existing_song(self, panel, library, tmp_path):
        song = _make_song_with_stems(library, tmp_path)
        panel.refresh()
        assert panel.select_song(song.id) is True

    def test_select_nonexistent_returns_false(self, panel):
        assert panel.select_song("nonexistent") is False

    def test_select_emits_signal(self, panel, library, tmp_path):
        song = _make_song_with_stems(library, tmp_path)
        panel.refresh()

        received = []
        panel.song_selected.connect(received.append)
        panel.select_song(song.id)
        assert received == [song.id]


# -----------------------------------------------------------------------
# StemRow helpers
# -----------------------------------------------------------------------


class TestStemRowSetters:
    def test_set_soloed(self, controls, player, tmp_path):
        audio = tmp_path / "stem.wav"
        sf.write(str(audio), np.zeros((44100 * 5, 2), dtype=np.float32), 44100)
        player.load_stems({"vocals": str(audio)})
        controls.set_stem_names(["vocals"])

        row = controls._stem_rows["vocals"]
        row.set_soloed(True)
        assert "vocals" in player.soloed_stems

    def test_set_volume_slider(self, controls, player, tmp_path):
        audio = tmp_path / "stem.wav"
        sf.write(str(audio), np.zeros((44100 * 5, 2), dtype=np.float32), 44100)
        player.load_stems({"vocals": str(audio)})
        controls.set_stem_names(["vocals"])

        row = controls._stem_rows["vocals"]
        row.set_volume_slider(150)
        assert player.get_volume("vocals") == pytest.approx(1.5, abs=0.05)


# -----------------------------------------------------------------------
# PlayerControls.restore_stem_state
# -----------------------------------------------------------------------


class TestRestoreStemState:
    def test_restores_mute_solo_volume(self, controls, player, tmp_path):
        audio = tmp_path / "stem.wav"
        sf.write(str(audio), np.zeros((44100 * 5, 2), dtype=np.float32), 44100)
        player.load_stems({"vocals": str(audio), "drums": str(audio)})
        controls.set_stem_names(["vocals", "drums"])

        controls.restore_stem_state(
            muted={"vocals"},
            soloed={"drums"},
            volumes={"vocals": 0.5, "drums": 1.5},
        )

        assert "vocals" in player.muted_stems
        assert "drums" in player.soloed_stems
        assert player.get_volume("vocals") == pytest.approx(0.5, abs=0.05)
        assert player.get_volume("drums") == pytest.approx(1.5, abs=0.05)

    def test_ignores_unknown_stems(self, controls, player, tmp_path):
        audio = tmp_path / "stem.wav"
        sf.write(str(audio), np.zeros((44100 * 5, 2), dtype=np.float32), 44100)
        player.load_stems({"vocals": str(audio)})
        controls.set_stem_names(["vocals"])

        # "guitar" is not in the loaded stems -- should not crash
        controls.restore_stem_state(
            muted={"guitar"},
            soloed=set(),
            volumes={"guitar": 0.8},
        )
        assert "vocals" not in player.muted_stems


# -----------------------------------------------------------------------
# PlayerControls.restore_loop_state
# -----------------------------------------------------------------------


class TestRestoreLoopState:
    def test_restores_loop_points(self, controls, player, tmp_path):
        audio = tmp_path / "stem.wav"
        sf.write(str(audio), np.zeros((44100 * 5, 2), dtype=np.float32), 44100)
        player.load_stems({"vocals": str(audio)})
        controls.set_stem_names(["vocals"])

        controls.restore_loop_state(loop_a=1.0, loop_b=2.0, looping=True)

        assert player.loop_a == pytest.approx(1.0, abs=0.05)
        assert player.loop_b == pytest.approx(2.0, abs=0.05)
        assert player.looping is True

    def test_none_loop_points_skipped(self, controls, player, tmp_path):
        audio = tmp_path / "stem.wav"
        sf.write(str(audio), np.zeros((44100 * 5, 2), dtype=np.float32), 44100)
        player.load_stems({"vocals": str(audio)})
        controls.set_stem_names(["vocals"])

        controls.restore_loop_state(loop_a=None, loop_b=None, looping=False)

        assert player.loop_a is None
        assert player.loop_b is None
        assert player.looping is False


# -----------------------------------------------------------------------
# Session round-trip: save then restore values
# -----------------------------------------------------------------------


class TestSessionRoundTrip:
    """Test that session values survive a QSettings write/read cycle."""

    def test_stem_state_roundtrip(self):
        muted = {"vocals", "drums"}
        soloed = {"bass"}
        volumes = {"vocals": 0.5, "drums": 1.5, "bass": 1.0}

        settings = QSettings("stemma", "stemma-test")
        settings.clear()
        settings.setValue("session/muted_stems", json.dumps(sorted(muted)))
        settings.setValue("session/soloed_stems", json.dumps(sorted(soloed)))
        settings.setValue("session/volumes", json.dumps(volumes))

        restored_muted = set(json.loads(settings.value("session/muted_stems")))
        restored_soloed = set(json.loads(settings.value("session/soloed_stems")))
        restored_volumes = json.loads(settings.value("session/volumes"))

        assert restored_muted == muted
        assert restored_soloed == soloed
        assert restored_volumes == volumes
        settings.clear()

    def test_loop_roundtrip(self):
        settings = QSettings("stemma", "stemma-test")
        settings.clear()
        settings.setValue("session/loop_a", 10.5)
        settings.setValue("session/loop_b", 20.3)
        settings.setValue("session/looping", True)

        a = float(settings.value("session/loop_a"))
        b = float(settings.value("session/loop_b"))
        assert a == pytest.approx(10.5)
        assert b == pytest.approx(20.3)
        settings.clear()

    def test_no_loop_roundtrip(self):
        settings = QSettings("stemma", "stemma-test")
        settings.clear()
        settings.setValue("session/loop_a", -1)
        settings.setValue("session/loop_b", -1)

        a = float(settings.value("session/loop_a"))
        b = float(settings.value("session/loop_b"))
        assert a == -1
        assert b == -1
        settings.clear()

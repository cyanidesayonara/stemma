"""The synced metronome's BPM box follows speed changes while paused.

Since speed applies live (#217), the synced BPM is tempo times speed. The
box refreshed only on a position change, and a paused speed change no
longer moves the position, so it kept the old value until play.
"""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from src.player import MultiTrackPlayer
from src.ui.player_controls import PlayerControls

SR = 44100


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def player(qapp):
    p = MultiTrackPlayer()
    stem = np.zeros((SR * 20, 2), dtype=np.float32)
    p.apply_loaded_stems({"vocals": stem}, SR)
    # 120 BPM: a beat every half second.
    beats = [i * 0.5 for i in range(40)]
    p.set_beat_times(beats, beats[::4])
    yield p
    p.shutdown(wait_ms=5000)


@pytest.fixture
def controls(qapp, player):
    ctrl = PlayerControls(player)
    yield ctrl
    ctrl._cleanup_peak_thread()
    ctrl.setParent(None)
    ctrl.deleteLater()
    qapp.processEvents()


def test_paused_speed_change_updates_synced_bpm(qapp, player, controls):
    controls._beat_sync_btn.setChecked(True)
    player.seek(5.0)
    qapp.processEvents()
    assert controls._bpm_spin.value() == 120

    player.set_speed(0.75)
    qapp.processEvents()

    assert controls._bpm_spin.value() == 90


def test_enabling_sync_while_paused_shows_live_bpm(qapp, player, controls):
    player.seek(5.0)
    player.set_speed(1.5)
    qapp.processEvents()

    controls._beat_sync_btn.setChecked(True)
    qapp.processEvents()

    assert controls._bpm_spin.value() == 180


def test_new_beats_while_synced_and_paused_update_bpm(
    qapp, player, controls,
):
    controls._beat_sync_btn.setChecked(True)
    player.seek(5.0)
    qapp.processEvents()
    assert controls._bpm_spin.value() == 120

    # A re-detection finds 100 BPM: a beat every 0.6 s.
    beats = [i * 0.6 for i in range(30)]
    controls.restore_beat_times(beats, beats[::4])
    qapp.processEvents()

    assert controls._bpm_spin.value() == 100


def test_turning_sync_off_shows_the_manual_tempo(qapp, player, controls):
    controls._bpm_spin.setValue(100)
    player.seek(5.0)
    controls._beat_sync_btn.setChecked(True)
    player.set_speed(0.5)
    qapp.processEvents()
    assert controls._bpm_spin.value() == 60

    controls._beat_sync_btn.setChecked(False)
    qapp.processEvents()

    assert controls._bpm_spin.value() == 100
    assert player.metronome_bpm == 100

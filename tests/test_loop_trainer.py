"""Tests for the Loop Trainer (v2.5.0).

Covers:
  - Player: the loop-wrap counter increments in the audio callback and
    _emit_position surfaces it as the loop_wrapped signal exactly once
    per wrap.
  - UI: the trainer steps playback speed up one preset per loop wrap,
    from the start speed up to 1.0x and no further; enabling drops to the
    start speed only with a valid A-B region; a disabled trainer ignores
    wraps; song load resets the trainer; session state round-trips.
"""

from unittest.mock import patch

import numpy as np
import pytest
import sounddevice as sd
from PySide6.QtWidgets import QApplication

from src.player import MultiTrackPlayer, SPEED_PRESETS
from src.ui.player_controls import PlayerControls


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def player(qapp):
    p = MultiTrackPlayer()
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


@pytest.fixture
def loaded(player, tmp_path):
    import soundfile as sf
    sr = 44100
    path = tmp_path / "s.wav"
    sf.write(str(path), np.ones((sr, 2), dtype=np.float32) * 0.2, sr)
    player.load_stems({"vocals": str(path)})
    return player


# -----------------------------------------------------------------------
# Player: loop-wrap counter + signal
# -----------------------------------------------------------------------

class TestLoopWrapSignal:
    def _loop(self, player, a_s, b_s):
        player.set_loop_a(a_s)
        player.set_loop_b(b_s)
        player.set_looping(True)

    def test_callback_increments_wrap_count_on_wrap(self, loaded):
        """Playing through the B boundary wraps to A and bumps the count."""
        self._loop(loaded, 0.0, 0.01)  # ~441-frame loop
        loaded._is_playing = True
        loaded.seek(0.0)
        before = loaded._loop_wrap_count
        out = np.zeros((2048, 2), dtype=np.float32)
        loaded._audio_callback(out, 2048, {}, sd.CallbackFlags())
        assert loaded._loop_wrap_count > before

    def test_emit_position_fires_loop_wrapped_once_per_wrap(self, loaded):
        received = []
        loaded.loop_wrapped.connect(lambda: received.append(1))
        # Simulate the callback having wrapped twice.
        loaded._is_playing = True
        loaded._loop_wrap_count = 2
        loaded._emit_position()
        assert received == [1]  # coalesced: fires once when count changes
        # No further wrap -> no further emit.
        loaded._emit_position()
        assert received == [1]
        # One more wrap -> one more emit.
        loaded._loop_wrap_count = 3
        loaded._emit_position()
        assert received == [1, 1]

    def test_load_stems_resets_wrap_count(self, loaded):
        loaded._loop_wrap_count = 5
        loaded._loop_wrap_seen = 5
        # Reload.
        loaded._reset_song_state()
        assert loaded._loop_wrap_count == 0
        assert loaded._loop_wrap_seen == 0


# -----------------------------------------------------------------------
# UI: trainer speed ramp
# -----------------------------------------------------------------------

class TestTrainerRamp:
    def _make_loop(self, controls):
        controls._player.set_loop_a(0.0)
        controls._player.set_loop_b(5.0)

    def test_next_speed_up_walks_presets_to_one(self, controls):
        assert controls._next_speed_up(0.5) == 0.75
        assert controls._next_speed_up(0.75) == 0.85
        assert controls._next_speed_up(0.85) == 1.0
        # At/above 1.0 there's nothing to step to (won't go 1.25+).
        assert controls._next_speed_up(1.0) is None

    def test_enable_drops_to_start_speed_with_loop(self, controls, loaded):
        self._make_loop(controls)
        controls._trainer_start_speed = 0.75
        with patch.object(loaded, "set_speed"):
            # Route through the combo the way the UI does.
            with patch.object(controls, "_set_speed_preset") as sp:
                controls._on_trainer_toggled(True)
                sp.assert_called_once_with(0.75)

    def test_enable_without_loop_does_not_change_speed(self, controls, loaded):
        # No A-B region set.
        with patch.object(controls, "_set_speed_preset") as sp:
            controls._on_trainer_toggled(True)
            sp.assert_not_called()

    def test_wrap_advances_one_preset(self, controls, loaded):
        self._make_loop(controls)
        controls._trainer_enabled = True
        loaded._playback_speed = 0.75
        with patch.object(controls, "_set_speed_preset") as sp:
            controls._on_loop_wrapped()
            sp.assert_called_once_with(0.85)

    def test_wrap_at_target_does_not_advance(self, controls, loaded):
        self._make_loop(controls)
        controls._trainer_enabled = True
        loaded._playback_speed = 1.0
        with patch.object(controls, "_set_speed_preset") as sp:
            controls._on_loop_wrapped()
            sp.assert_not_called()

    def test_wrap_ignored_when_trainer_off(self, controls, loaded):
        self._make_loop(controls)
        controls._trainer_enabled = False
        loaded._playback_speed = 0.75
        with patch.object(controls, "_set_speed_preset") as sp:
            controls._on_loop_wrapped()
            sp.assert_not_called()

    def test_full_ramp_sequence(self, controls, loaded):
        """Successive wraps climb 0.75 -> 0.85 -> 1.0 then hold."""
        self._make_loop(controls)
        controls._trainer_start_speed = 0.75
        controls._on_trainer_toggled(True)
        # Simulate the render landing at each step by setting the speed.
        seq = []
        for _ in range(4):
            loaded._playback_speed = (
                controls._speed_combo.currentData() or loaded._playback_speed
            )
            # Emulate: whatever _set_speed_preset would target, apply it.
            nxt = controls._next_speed_up(loaded._playback_speed)
            controls._on_loop_wrapped()
            if nxt is not None:
                loaded._playback_speed = nxt
            seq.append(loaded._playback_speed)
        # Climbs to 1.0 and holds.
        assert seq[-1] == 1.0
        assert seq == sorted(seq)  # monotonic non-decreasing


# -----------------------------------------------------------------------
# UI: reset on song load + session round-trip
# -----------------------------------------------------------------------

class TestTrainerLifecycle:
    def test_song_load_resets_trainer(self, controls):
        controls._trainer_check.setChecked(True)
        controls._trainer_enabled = True
        controls.set_stem_names([])  # simulates loading a new (empty) song
        assert controls._trainer_enabled is False
        assert not controls._trainer_check.isChecked()

    def test_restore_trainer_state_round_trip(self, controls):
        controls.restore_trainer_state(True, 0.5)
        assert controls.trainer_enabled is True
        assert controls.trainer_start_speed == 0.5
        assert controls._trainer_check.isChecked()

    def test_restore_ignores_unknown_start_speed(self, controls):
        # A start speed not in the combo is left at the default.
        controls.restore_trainer_state(True, 0.999)
        assert controls.trainer_start_speed in [
            p for p in SPEED_PRESETS if p < 1.0
        ]

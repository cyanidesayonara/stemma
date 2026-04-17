"""Tests for pitch transposition on MultiTrackPlayer + StretchWorker.

Covers:
  - Clamping of the semitone range
  - Fast path (speed=1.0 AND pitch=0 skips the worker)
  - ``pitch_changed`` signal fires on real changes and not on no-ops
  - Pitch + speed render in a single pass (not chained)
  - Recording-take stems skip pitch by default; sync toggle includes them
  - ``sync_recording_pitch`` re-renders only when pitch is active
  - Recording cannot be armed when pitch != 0
"""

from unittest.mock import patch

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from src.player import (
    MultiTrackPlayer,
    PITCH_MAX_SEMITONES,
    PITCH_MIN_SEMITONES,
    RECORDING_STEM_PREFIX,
    StretchWorker,
)


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


@pytest.fixture
def loaded_player(app):
    """Player with fake stems already loaded (no worker will spawn audio IO)."""
    p = MultiTrackPlayer()
    sr = 44100
    frames = sr  # 1 second
    p._stems = {"vocals": np.zeros((frames, 2), dtype=np.float32)}
    p._original_stems = dict(p._stems)
    p._sample_rate = sr
    p._total_frames = frames
    return p


# -----------------------------------------------------------------------
# Player API: clamping and no-op behaviour
# -----------------------------------------------------------------------

class TestPitchClamping:
    def test_default_pitch_is_zero(self, loaded_player):
        assert loaded_player.pitch_semitones == 0

    def test_set_pitch_clamps_high(self, loaded_player):
        loaded_player.set_pitch(99)
        assert loaded_player.pitch_semitones == PITCH_MAX_SEMITONES

    def test_set_pitch_clamps_low(self, loaded_player):
        loaded_player.set_pitch(-99)
        assert loaded_player.pitch_semitones == PITCH_MIN_SEMITONES

    def test_set_pitch_ignores_non_int(self, loaded_player):
        """Non-numeric input is silently rejected (no crash)."""
        loaded_player.set_pitch("bogus")  # type: ignore[arg-type]
        assert loaded_player.pitch_semitones == 0

    def test_set_pitch_coerces_float_to_int(self, loaded_player):
        loaded_player.set_pitch(2.7)
        assert loaded_player.pitch_semitones == 2


# -----------------------------------------------------------------------
# Fast path: speed=1.0 AND pitch=0 should never spawn a StretchWorker
# -----------------------------------------------------------------------

class TestPitchFastPath:
    def test_setting_pitch_zero_with_speed_one_skips_worker(
        self, loaded_player,
    ):
        """Identity state must not spawn a stretch worker."""
        with patch("src.player.StretchWorker") as worker_cls:
            loaded_player.set_pitch(0)  # Already 0 -- no-op entirely.
            worker_cls.assert_not_called()

    def test_setting_nonzero_pitch_spawns_worker(self, loaded_player):
        """A real pitch change with speed=1.0 still requires rendering."""
        with patch("src.player.StretchWorker") as worker_cls:
            worker_cls.return_value.isRunning.return_value = False
            loaded_player.set_pitch(3)
            assert worker_cls.called

    def test_returning_to_identity_restores_originals_fast(
        self, loaded_player,
    ):
        """pitch=+2 -> pitch=0 with speed=1.0 takes the fast path."""
        # Start from pitch=+2 WITHOUT actually rendering (skip the worker).
        loaded_player._pitch_semitones = 2

        with patch("src.player.StretchWorker") as worker_cls:
            loaded_player.set_pitch(0)
            # Fast path: no worker spawned when returning to identity.
            worker_cls.assert_not_called()

        # And the stems dict was swapped back to the originals object.
        assert loaded_player._stems is not loaded_player._original_stems
        # But their contents match.
        for k, v in loaded_player._original_stems.items():
            assert np.array_equal(loaded_player._stems[k], v)


# -----------------------------------------------------------------------
# pitch_changed signal
# -----------------------------------------------------------------------

class TestPitchSignal:
    def test_emits_on_change(self, loaded_player):
        received = []
        loaded_player.pitch_changed.connect(lambda n: received.append(n))

        # Skip the worker by patching it out.
        with patch("src.player.StretchWorker") as worker_cls:
            worker_cls.return_value.isRunning.return_value = False
            loaded_player.set_pitch(2)

        # Worker was spawned but we never fire `completed` manually, so the
        # signal only fires via the fast path. In this case, speed=1 AND
        # pitch=+2 is NOT the fast path, so nothing fires yet.
        # Emulate completion:
        loaded_player._on_stretch_ready(
            dict(loaded_player._original_stems), ("pitch",)
        )
        assert received == [2]

    def test_no_emit_on_noop(self, loaded_player):
        received = []
        loaded_player.pitch_changed.connect(lambda n: received.append(n))
        loaded_player.set_pitch(0)  # Already 0.
        assert received == []

    def test_fast_path_returning_to_zero_emits(self, loaded_player):
        """Going pitch=+2 -> pitch=0 via fast path emits pitch_changed."""
        loaded_player._pitch_semitones = 2  # Pretend.
        received = []
        loaded_player.pitch_changed.connect(lambda n: received.append(n))

        with patch("src.player.StretchWorker"):
            loaded_player.set_pitch(0)

        assert received == [0]


# -----------------------------------------------------------------------
# Combined speed+pitch rendering: one pass, not chained
# -----------------------------------------------------------------------

class TestStretchWorkerCombined:
    """The StretchWorker applies pitch_shift once, then time_stretch once."""

    def test_pitch_only_calls_pitch_shift(self, app):
        stems = {"vocals": np.random.randn(4410, 2).astype(np.float32)}
        worker = StretchWorker(stems, 44100, 1.0, 2)
        with patch("src.player.librosa.effects.pitch_shift",
                   side_effect=lambda y, sr, n_steps: y) as ps, \
             patch("src.player.librosa.effects.time_stretch") as ts:
            worker.run()
        assert ps.call_count == 2  # Once per channel.
        ts.assert_not_called()

    def test_speed_only_calls_time_stretch(self, app):
        stems = {"vocals": np.random.randn(4410, 2).astype(np.float32)}
        worker = StretchWorker(stems, 44100, 0.75, 0)
        with patch("src.player.librosa.effects.pitch_shift") as ps, \
             patch("src.player.librosa.effects.time_stretch",
                   side_effect=lambda y, rate: y) as ts:
            worker.run()
        ps.assert_not_called()
        assert ts.call_count == 2

    def test_both_calls_pitch_then_speed(self, app):
        """When both are non-identity, pitch runs first, then speed."""
        stems = {"vocals": np.random.randn(4410, 2).astype(np.float32)}
        worker = StretchWorker(stems, 44100, 0.75, 2)
        call_order: list[str] = []

        def fake_ps(y, sr, n_steps):
            call_order.append("pitch")
            return y

        def fake_ts(y, rate):
            call_order.append("speed")
            return y

        with patch("src.player.librosa.effects.pitch_shift", side_effect=fake_ps), \
             patch("src.player.librosa.effects.time_stretch", side_effect=fake_ts):
            worker.run()

        # Per-channel order: pitch, speed, pitch, speed
        assert call_order == ["pitch", "speed", "pitch", "speed"]

    def test_identity_reuses_input_buffers(self, app):
        """speed=1.0 AND pitch=0 means no work -- output IS input."""
        stems = {"vocals": np.random.randn(4410, 2).astype(np.float32)}
        worker = StretchWorker(stems, 44100, 1.0, 0)
        results: dict = {}
        worker.completed.connect(lambda d: results.update(d))
        worker.run()
        assert results["vocals"] is stems["vocals"]


# -----------------------------------------------------------------------
# Recording stems and sync_recording_pitch
# -----------------------------------------------------------------------

class TestRecordingPitchSync:
    def _make_stems(self):
        return {
            "vocals": np.random.randn(4410, 2).astype(np.float32),
            f"{RECORDING_STEM_PREFIX}1": (
                np.random.randn(4410, 2).astype(np.float32)
            ),
        }

    def test_recording_skips_pitch_by_default(self, app):
        """By default, recording stems are not pitch-shifted."""
        stems = self._make_stems()
        worker = StretchWorker(stems, 44100, 1.0, 3, sync_recording_pitch=False)
        shifted_names: list[str] = []

        def fake_ps(y, sr, n_steps):
            # Track which stem names the shift is applied to by reading
            # the worker's loop context indirectly: nothing in the call
            # reveals the name, so use call count instead.
            shifted_names.append("x")
            return y

        with patch("src.player.librosa.effects.pitch_shift", side_effect=fake_ps):
            worker.run()

        # Only the "vocals" stem (2 channels) should be shifted.
        # The recording stem reuses the original buffer.
        assert len(shifted_names) == 2

    def test_sync_true_shifts_recording_too(self, app):
        """With sync_recording_pitch=True, recording stems are shifted."""
        stems = self._make_stems()
        worker = StretchWorker(stems, 44100, 1.0, 3, sync_recording_pitch=True)
        shifts: list[int] = []

        def fake_ps(y, sr, n_steps):
            shifts.append(1)
            return y

        with patch("src.player.librosa.effects.pitch_shift", side_effect=fake_ps):
            worker.run()

        # Both stems, both channels: 4 calls.
        assert len(shifts) == 4

    def test_recording_always_gets_speed(self, app):
        """Speed applies to recordings regardless of sync flag (timing)."""
        stems = self._make_stems()
        worker = StretchWorker(stems, 44100, 0.5, 0, sync_recording_pitch=False)
        stretches: list[int] = []

        def fake_ts(y, rate):
            stretches.append(1)
            return y

        with patch("src.player.librosa.effects.time_stretch", side_effect=fake_ts):
            worker.run()

        # Both stems, both channels: 4 calls.
        assert len(stretches) == 4


class TestPlayerSyncRecordingPitch:
    def test_toggle_off_at_pitch_zero_is_noop(self, loaded_player):
        """Toggling sync without an active pitch shift does not render."""
        with patch("src.player.StretchWorker") as worker_cls:
            loaded_player.set_sync_recording_pitch(True)
            loaded_player.set_sync_recording_pitch(False)
            worker_cls.assert_not_called()

    def test_toggle_when_pitch_active_triggers_rerender(self, loaded_player):
        """Toggling the sync flag while pitch is active re-renders stems."""
        loaded_player._pitch_semitones = 2  # Pretend an active shift.
        with patch("src.player.StretchWorker") as worker_cls:
            worker_cls.return_value.isRunning.return_value = False
            loaded_player.set_sync_recording_pitch(True)
            assert worker_cls.called

    def test_default_is_false(self, loaded_player):
        assert loaded_player.sync_recording_pitch is False


# -----------------------------------------------------------------------
# Recording arming guarded by pitch state
# -----------------------------------------------------------------------

class TestRecordingArmGuard:
    def test_arm_refused_when_pitch_nonzero(self, loaded_player):
        loaded_player._pitch_semitones = 2
        loaded_player.arm_recording(True)
        assert loaded_player.recording_armed is False

    def test_arm_allowed_at_pitch_zero(self, loaded_player):
        loaded_player._pitch_semitones = 0
        loaded_player.arm_recording(True)
        assert loaded_player.recording_armed is True


# -----------------------------------------------------------------------
# load_stems resets pitch to 0
# -----------------------------------------------------------------------

class TestLoadStemsResetsPitch:
    def test_load_stems_resets_pitch(self, loaded_player, tmp_path):
        """Loading a new song resets the pitch to 0."""
        import soundfile as sf
        loaded_player._pitch_semitones = 4
        # Write a tiny WAV and load it.
        wav_path = tmp_path / "fake.wav"
        sf.write(str(wav_path), np.zeros((4410, 2), dtype=np.float32), 44100)
        loaded_player.load_stems({"vocals": str(wav_path)})
        assert loaded_player.pitch_semitones == 0

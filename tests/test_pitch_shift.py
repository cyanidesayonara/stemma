"""Tests for pitch transposition on MultiTrackPlayer + StretchWorker.

Covers:
  - Clamping of the semitone range
  - Fast path (speed=1.0 AND pitch=0 skips the worker)
  - ``pitch_changed`` signal fires on real changes and not on no-ops
  - Pitch + speed render in a single pass (not chained)
  - Recording-take stems skip pitch by default; sync toggle includes them
  - ``sync_recording_pitch`` re-renders only when pitch is active
  - Recording cannot be armed when pitch != 0
  - Stretch lifecycle signals (started / progress / finished) emit correctly
  - Detached workers are kept alive on ``_detached_workers`` until finished
  - ``_on_stretch_error`` recomputes beat frames after restoring originals
"""

from unittest.mock import MagicMock, patch

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
                   side_effect=lambda y, sr, n_steps, **kw: y) as ps, \
             patch("src.player.librosa.effects.time_stretch") as ts:
            worker.run()
        assert ps.call_count == 2  # Once per channel.
        ts.assert_not_called()

    def test_pitch_uses_fast_res_type(self, app):
        """The worker passes the fast resample kernel and a tuned hop_length
        to librosa so renders complete in a usable time during interactive
        use."""
        stems = {"vocals": np.random.randn(4410, 2).astype(np.float32)}
        worker = StretchWorker(stems, 44100, 1.0, 2)
        with patch(
            "src.player.librosa.effects.pitch_shift",
            side_effect=lambda y, sr, n_steps, **kw: y,
        ) as ps:
            worker.run()
        # All calls must have passed res_type=soxr_mq (~3-4x faster than
        # soxr_hq) and hop_length=_HOP_LENGTH (~2-3x faster per benchmark).
        for call in ps.call_args_list:
            assert call.kwargs.get("res_type") == "soxr_mq"
            assert call.kwargs.get("hop_length") == StretchWorker._HOP_LENGTH

    def test_speed_only_calls_time_stretch(self, app):
        stems = {"vocals": np.random.randn(4410, 2).astype(np.float32)}
        worker = StretchWorker(stems, 44100, 0.75, 0)
        with patch("src.player.librosa.effects.pitch_shift") as ps, \
             patch("src.player.librosa.effects.time_stretch",
                   side_effect=lambda y, rate, **kw: y) as ts:
            worker.run()
        ps.assert_not_called()
        assert ts.call_count == 2

    def test_both_calls_pitch_then_speed(self, app):
        """When both are non-identity, pitch runs first, then speed."""
        stems = {"vocals": np.random.randn(4410, 2).astype(np.float32)}
        worker = StretchWorker(stems, 44100, 0.75, 2)
        call_order: list[str] = []

        def fake_ps(y, sr, n_steps, **kw):
            call_order.append("pitch")
            return y

        def fake_ts(y, rate, **kw):
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

        def fake_ps(y, sr, n_steps, **kw):
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

        def fake_ps(y, sr, n_steps, **kw):
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

        def fake_ts(y, rate, **kw):
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


# -----------------------------------------------------------------------
# Stretch lifecycle signals (started / progress / finished)
# -----------------------------------------------------------------------

class TestStretchLifecycleSignals:
    """stretch_started/progress/finished frame the async render for the UI."""

    def test_started_emits_when_worker_spawns(self, loaded_player):
        """Spawning a worker emits stretch_started exactly once."""
        started: list[int] = []
        loaded_player.stretch_started.connect(lambda: started.append(1))
        with patch("src.player.StretchWorker") as worker_cls:
            worker_cls.return_value.isRunning.return_value = False
            loaded_player.set_pitch(2)
        assert started == [1]

    def test_started_does_not_emit_on_fast_path(self, loaded_player):
        """Identity no-op takes the fast path and must not emit."""
        started: list[int] = []
        loaded_player.stretch_started.connect(lambda: started.append(1))
        loaded_player.set_pitch(0)  # Already 0 -- fast path.
        assert started == []

    def test_finished_emits_on_success(self, loaded_player):
        """Successful render emits stretch_finished via _on_stretch_ready."""
        finished: list[int] = []
        loaded_player.stretch_finished.connect(lambda: finished.append(1))
        loaded_player._on_stretch_ready(
            dict(loaded_player._original_stems), ("pitch",)
        )
        assert finished == [1]

    def test_finished_emits_on_error(self, loaded_player):
        """Worker failure still emits stretch_finished so UI re-enables."""
        finished: list[int] = []
        loaded_player.stretch_finished.connect(lambda: finished.append(1))
        loaded_player._on_stretch_error("boom", ("pitch",))
        assert finished == [1]

    def test_progress_connected_to_worker(self, loaded_player):
        """Player wires the worker's per-stem progress through to its own
        ``stretch_progress`` Signal, so the UI can subscribe once and stay
        connected across successive renders."""
        with patch("src.player.StretchWorker") as worker_cls:
            fake_worker = MagicMock()
            fake_worker.isRunning.return_value = False
            worker_cls.return_value = fake_worker
            loaded_player.set_pitch(3)

        # The player must forward progress to its own signal.
        fake_worker.progress.connect.assert_called_once_with(
            loaded_player.stretch_progress
        )


# -----------------------------------------------------------------------
# Worker keepalive: QThread GC safety
# -----------------------------------------------------------------------

class TestWorkerKeepalive:
    """Detached-but-running workers must be held on _detached_workers.

    Without the keepalive list, the Python wrapper refcount can drop to
    zero while the QThread is still active, triggering the classic
    "QThread: Destroyed while thread is still running" crash.
    """

    def test_running_worker_is_detached_to_keepalive(self, loaded_player):
        """A running worker gets appended to _detached_workers."""
        fake_worker = MagicMock()
        fake_worker.isRunning.return_value = True
        loaded_player._stretch_worker = fake_worker

        assert loaded_player._detach_stretch_worker() is True

        assert fake_worker in loaded_player._detached_workers
        assert loaded_player._stretch_worker is None

    def test_running_worker_is_cancelled_on_detach(self, loaded_player):
        """Detach must call ``cancel()`` so the worker exits early
        instead of wasting CPU on a stale render target."""
        fake_worker = MagicMock()
        fake_worker.isRunning.return_value = True
        loaded_player._stretch_worker = fake_worker

        loaded_player._detach_stretch_worker()

        fake_worker.cancel.assert_called_once()

    def test_detached_worker_defers_setParent(self, loaded_player):
        """A running worker must not be setParent(None)'d mid-run."""
        fake_worker = MagicMock()
        fake_worker.isRunning.return_value = True
        loaded_player._stretch_worker = fake_worker

        loaded_player._detach_stretch_worker()

        # Qt parent ownership stays intact while the thread is running.
        fake_worker.setParent.assert_not_called()

    def test_finished_signal_is_connected_to_reaper(self, loaded_player):
        """The keepalive list is emptied when the worker finishes."""
        fake_worker = MagicMock()
        fake_worker.isRunning.return_value = True
        loaded_player._stretch_worker = fake_worker

        loaded_player._detach_stretch_worker()

        assert fake_worker.finished.connect.called

    def test_reaper_removes_from_keepalive(self, loaded_player):
        """_reap_detached_worker drops the worker from _detached_workers."""
        fake_worker = MagicMock()
        loaded_player._detached_workers.append(fake_worker)

        loaded_player._reap_detached_worker(fake_worker)

        assert fake_worker not in loaded_player._detached_workers
        fake_worker.setParent.assert_called_once_with(None)
        fake_worker.deleteLater.assert_called_once()

    def test_reaper_is_idempotent(self, loaded_player):
        """Calling the reaper twice does not raise on the second call."""
        fake_worker = MagicMock()
        loaded_player._detached_workers.append(fake_worker)
        loaded_player._reap_detached_worker(fake_worker)
        # Second call: the worker is no longer in the list but should not
        # raise (protects against double-connections or test re-entry).
        loaded_player._reap_detached_worker(fake_worker)

    def test_non_running_worker_is_released_immediately(self, loaded_player):
        """If the worker already stopped, skip the keepalive dance."""
        fake_worker = MagicMock()
        fake_worker.isRunning.return_value = False
        loaded_player._stretch_worker = fake_worker

        assert loaded_player._detach_stretch_worker() is False

        assert fake_worker not in loaded_player._detached_workers
        fake_worker.setParent.assert_called_once_with(None)
        fake_worker.deleteLater.assert_called_once()

    def test_multiple_rapid_detaches_accumulate(self, loaded_player):
        """Two rapid set_pitch calls keep both workers alive."""
        worker_a = MagicMock()
        worker_a.isRunning.return_value = True
        loaded_player._stretch_worker = worker_a
        loaded_player._detach_stretch_worker()

        worker_b = MagicMock()
        worker_b.isRunning.return_value = True
        loaded_player._stretch_worker = worker_b
        loaded_player._detach_stretch_worker()

        assert worker_a in loaded_player._detached_workers
        assert worker_b in loaded_player._detached_workers
        assert len(loaded_player._detached_workers) == 2

    def test_detach_returns_false_when_no_worker(self, loaded_player):
        """Detach is idempotent and reports 'no render was active'."""
        loaded_player._stretch_worker = None
        assert loaded_player._detach_stretch_worker() is False


# -----------------------------------------------------------------------
# StretchWorker.cancel() — early exit semantics
# -----------------------------------------------------------------------

class TestStretchWorkerCancel:
    """Cancellation must stop the worker without emitting completion."""

    def test_cancel_sets_flag(self, app):
        stems = {"vocals": np.random.randn(4410, 2).astype(np.float32)}
        worker = StretchWorker(stems, 44100, 1.0, 2)
        assert worker.cancelled is False
        worker.cancel()
        assert worker.cancelled is True

    def test_cancelled_worker_emits_no_completed(self, app):
        """If cancelled before run, no completed signal fires."""
        stems = {"vocals": np.random.randn(4410, 2).astype(np.float32)}
        worker = StretchWorker(stems, 44100, 1.0, 2)
        worker.cancel()
        completed: list = []
        worker.completed.connect(lambda d: completed.append(d))
        worker.run()
        assert completed == []

    def test_cancelled_worker_emits_no_progress(self, app):
        """Progress signals are suppressed once cancelled."""
        stems = {"vocals": np.random.randn(4410, 2).astype(np.float32)}
        worker = StretchWorker(stems, 44100, 1.0, 2)
        worker.cancel()
        progress: list = []
        worker.progress.connect(lambda c, t: progress.append((c, t)))
        worker.run()
        assert progress == []

    def test_cancelled_worker_swallows_errors(self, app):
        """Errors raised after cancellation are not surfaced (the caller
        no longer cares; surfacing would confuse the UI, which already
        moved on)."""
        stems = {"vocals": np.random.randn(4410, 2).astype(np.float32)}
        worker = StretchWorker(stems, 44100, 1.0, 2)
        errors: list = []
        worker.error.connect(lambda m: errors.append(m))
        worker.cancel()

        # Make librosa blow up; since we're cancelled, no error should
        # escape.
        with patch(
            "src.player.librosa.effects.pitch_shift",
            side_effect=RuntimeError("oops"),
        ):
            worker.run()
        assert errors == []


# -----------------------------------------------------------------------
# cancel_stretch() — player-level cancel without new render
# -----------------------------------------------------------------------

class TestCancelStretch:
    def test_cancel_without_active_worker_is_noop(self, loaded_player):
        """No worker running → no signal, no crash."""
        finished: list = []
        loaded_player.stretch_finished.connect(lambda: finished.append(1))
        loaded_player._stretch_worker = None
        loaded_player.cancel_stretch()
        assert finished == []

    def test_cancel_with_active_worker_emits_finished(self, loaded_player):
        """Cancelling a live render lets the UI clear its indicator."""
        fake_worker = MagicMock()
        fake_worker.isRunning.return_value = True
        loaded_player._stretch_worker = fake_worker

        finished: list = []
        loaded_player.stretch_finished.connect(lambda: finished.append(1))
        loaded_player.cancel_stretch()

        fake_worker.cancel.assert_called_once()
        assert finished == [1]

    def test_cancel_with_non_running_worker_still_releases(
        self, loaded_player,
    ):
        """If the worker exists but already finished, clean up quietly."""
        fake_worker = MagicMock()
        fake_worker.isRunning.return_value = False
        loaded_player._stretch_worker = fake_worker

        finished: list = []
        loaded_player.stretch_finished.connect(lambda: finished.append(1))
        loaded_player.cancel_stretch()

        # Non-running worker is not "rendering", so no finished signal.
        assert finished == []
        assert loaded_player._stretch_worker is None


# -----------------------------------------------------------------------
# Regression: _on_stretch_error must recompute beat frames
# -----------------------------------------------------------------------

class TestStretchErrorBeatsReset:
    """After a render error, beat frames must reflect the restored (original-
    length) stems — not the stretched indices that were active before the
    error."""

    def test_error_recomputes_beat_frames(self, loaded_player):
        """beat_frames are recalculated after an error restores identity.

        _recompute_beat_frames() uses _playback_speed as a divisor.  If
        speed was 0.5x before the error, beat_frames hold 2× the 1.0x
        frame indices.  The error handler resets _playback_speed to 1.0
        but, without the _recompute_beat_frames() call, the stale indices
        remain -- making the metronome click in the wrong places.
        """
        sr = loaded_player._sample_rate

        # Set up a beat grid at 0.5x speed (indices are 2× the 1.0x values).
        loaded_player._beat_times = [0.5, 1.0]
        loaded_player._playback_speed = 0.5
        loaded_player._recompute_beat_frames()
        slow_beat_frames = list(loaded_player._beat_frames)  # e.g. [44100, 88200]

        # Sanity: at 1.0x the indices should be half as large.
        loaded_player._playback_speed = 1.0
        loaded_player._recompute_beat_frames()
        normal_beat_frames = list(loaded_player._beat_frames)  # e.g. [22050, 44100]
        assert slow_beat_frames != normal_beat_frames, (
            "sanity: beat frames at 0.5x vs 1.0x must differ"
        )

        # Restore slow-speed indices (as they would be when a render is in
        # flight and _playback_speed is still 0.5).
        loaded_player._playback_speed = 0.5
        loaded_player._recompute_beat_frames()
        assert list(loaded_player._beat_frames) == slow_beat_frames

        # Trigger error recovery — should reset speed AND recompute frames.
        loaded_player._on_stretch_error("boom", ("speed",))

        assert loaded_player._playback_speed == 1.0
        assert np.array_equal(
            loaded_player._beat_frames, normal_beat_frames
        ), "_beat_frames not recomputed after error; metronome would misfire"

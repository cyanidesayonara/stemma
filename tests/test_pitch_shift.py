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
    """Player with fake stems already loaded (no worker will spawn audio IO).

    Teardown shuts the player down: tests that call set_pitch/set_speed
    without patching StretchWorker spawn a real QThread rendering via
    librosa, and letting the garbage collector delete the player (and
    its child QThread) while that render is still running corrupts the
    Qt heap -- the crash then surfaces in whichever later test touches
    Qt next.
    """
    p = MultiTrackPlayer()
    sr = 44100
    frames = sr  # 1 second
    p._stems = {"vocals": np.zeros((frames, 2), dtype=np.float32)}
    p._original_stems = dict(p._stems)
    p._sample_rate = sr
    p._total_frames = frames
    yield p
    p.shutdown(wait_ms=5000)


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

    def test_pitch_uses_high_quality_resampler(self, app):
        """The worker uses soxr_hq -- dropping to soxr_mq gives a big
        speedup but introduces an audible metallic timbre on transient
        material (drums, plucked strings), which defeats the point of a
        faithful-playback tool.  Speed wins come from parallel stem
        rendering instead.
        """
        stems = {"vocals": np.random.randn(4410, 2).astype(np.float32)}
        worker = StretchWorker(stems, 44100, 1.0, 2)
        with patch(
            "src.player.librosa.effects.pitch_shift",
            side_effect=lambda y, sr, n_steps, **kw: y,
        ) as ps:
            worker.run()
        assert ps.call_count > 0
        for call in ps.call_args_list:
            assert call.kwargs.get("res_type") == "soxr_hq"
            # hop_length must be librosa's default (i.e. not overridden);
            # enlarging it was tried and rejected -- it caused phase
            # smearing on drums.
            assert "hop_length" not in call.kwargs

    def test_time_stretch_uses_default_hop_length(self, app):
        """time_stretch also uses librosa's default hop.  Larger hops
        are faster but smear transients."""
        stems = {"vocals": np.random.randn(8820, 2).astype(np.float32)}
        worker = StretchWorker(stems, 44100, 0.75, 0)
        with patch(
            "src.player.librosa.effects.time_stretch",
            side_effect=lambda y, rate, **kw: y,
        ) as ts:
            worker.run()
        assert ts.call_count > 0
        for call in ts.call_args_list:
            assert "hop_length" not in call.kwargs

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


# -----------------------------------------------------------------------
# Render serialization — at most one live worker at a time
# -----------------------------------------------------------------------

class TestRenderSerialization:
    """A draining worker must block new spawns to bound peak memory use.

    The disk-fill bug: rapid speed/pitch knob twirls spawned a fresh
    ``StretchWorker`` every time.  Each worker runs a ``ThreadPoolExecutor``
    with up to 8 librosa calls holding float32 intermediates; stacking
    them serialized into ~10GB of pagefile swap on ~60s stems before the
    drain caught up.  The fix is a single queued-render slot that the
    reaper drains once the previous worker's cancellation settles.
    """

    def test_second_render_queues_instead_of_spawning(self, loaded_player):
        """While a worker drains, a new _render_stretch must not start a
        second thread."""
        # Simulate an in-flight worker that was just cancelled.
        draining = MagicMock()
        draining.isRunning.return_value = True
        loaded_player._detached_workers.append(draining)
        loaded_player._stretch_worker = None

        loaded_player._playback_speed = 0.75
        loaded_player._pitch_semitones = 0

        with patch.object(loaded_player, "_spawn_stretch_worker") as spawn:
            loaded_player._render_stretch(emit=("speed",))
            spawn.assert_not_called()

        assert loaded_player._pending_render_emit == ("speed",)

    def test_queued_render_supersedes_previous_queued(self, loaded_player):
        """A fresh queued render replaces the old emit tuple."""
        draining = MagicMock()
        draining.isRunning.return_value = True
        loaded_player._detached_workers.append(draining)
        loaded_player._pending_render_emit = ("pitch",)
        loaded_player._playback_speed = 0.75

        with patch.object(loaded_player, "_spawn_stretch_worker"):
            loaded_player._render_stretch(emit=("speed",))

        # The newer emit wins; the UI only needs to know the latest knob.
        assert loaded_player._pending_render_emit == ("speed",)

    def test_reaper_dispatches_queued_render(self, loaded_player):
        """When the last drain finishes, the queued render fires."""
        draining = MagicMock()
        loaded_player._detached_workers = [draining]
        loaded_player._pending_render_emit = ("speed",)
        loaded_player._playback_speed = 0.75
        loaded_player._pitch_semitones = 0

        with patch.object(loaded_player, "_spawn_stretch_worker") as spawn:
            loaded_player._reap_detached_worker(draining)
            spawn.assert_called_once_with(("speed",))

        assert loaded_player._pending_render_emit is None

    def test_reaper_dispatch_emits_stretch_started(self, loaded_player):
        """The queued render must re-light the UI indicator.

        Otherwise the render runs invisibly when cancel_stretch had
        previously cleared the indicator (common: every UI-driven
        debounce path calls cancel_stretch first).
        """
        draining = MagicMock()
        loaded_player._detached_workers = [draining]
        loaded_player._pending_render_emit = ("speed",)
        loaded_player._playback_speed = 0.75
        loaded_player._pitch_semitones = 0

        started: list[int] = []
        loaded_player.stretch_started.connect(lambda: started.append(1))

        # Patch StretchWorker construction so we exercise _spawn_stretch_worker
        # end-to-end without actually spinning up a QThread.
        with patch("src.player.StretchWorker") as mock_cls:
            mock_worker = MagicMock()
            mock_worker.isRunning.return_value = False
            mock_cls.return_value = mock_worker
            loaded_player._reap_detached_worker(draining)

        assert started == [1], (
            "queued render must re-emit stretch_started so the UI "
            "indicator lights back up"
        )

    def test_reaper_does_not_dispatch_while_others_draining(
        self, loaded_player,
    ):
        """If two workers are draining, only the last reap fires the queue."""
        drain_a = MagicMock()
        drain_b = MagicMock()
        loaded_player._detached_workers = [drain_a, drain_b]
        loaded_player._pending_render_emit = ("speed",)

        with patch.object(loaded_player, "_spawn_stretch_worker") as spawn:
            loaded_player._reap_detached_worker(drain_a)
            spawn.assert_not_called()

        # Queue is still armed for when drain_b reaps.
        assert loaded_player._pending_render_emit == ("speed",)

    def test_reaper_fast_path_when_queued_identity(self, loaded_player):
        """Queued render at identity settings uses fast path (no spawn)."""
        draining = MagicMock()
        loaded_player._detached_workers = [draining]
        loaded_player._pending_render_emit = ("speed",)
        loaded_player._playback_speed = 1.0
        loaded_player._pitch_semitones = 0

        finished: list = []
        loaded_player.stretch_finished.connect(lambda: finished.append(1))

        with patch.object(loaded_player, "_spawn_stretch_worker") as spawn:
            loaded_player._reap_detached_worker(draining)
            spawn.assert_not_called()

        assert finished == [1], "fast-path queued render must close lifecycle"
        assert loaded_player._pending_render_emit is None

    def test_cancel_stretch_clears_pending(self, loaded_player):
        """An explicit cancel supersedes a queued render."""
        loaded_player._pending_render_emit = ("speed",)
        loaded_player._stretch_worker = None

        finished: list = []
        loaded_player.stretch_finished.connect(lambda: finished.append(1))
        loaded_player.cancel_stretch()

        assert loaded_player._pending_render_emit is None
        # Pending alone (no active worker) still closes the lifecycle so
        # the UI indicator clears.
        assert finished == [1]

    def test_load_stems_clears_pending(self, loaded_player, tmp_path):
        """Switching songs must drop any queued render for the old stems."""
        loaded_player._pending_render_emit = ("speed",)

        # load_stems with an empty dict is a no-op load but still resets
        # the pending slot.
        with patch.object(loaded_player, "_render_stretch"):
            try:
                loaded_player.load_stems({})
            except Exception:  # noqa: BLE001
                pass

        assert loaded_player._pending_render_emit is None


# -----------------------------------------------------------------------
# Shutdown — cancel in-flight work and wait so atexit doesn't hang
# -----------------------------------------------------------------------

class TestPlayerShutdown:
    """``shutdown()`` is called from MainWindow.closeEvent.  Without it
    the Python atexit handler blocks on the ThreadPoolExecutor inside
    ``StretchWorker.run``, forcing Ctrl+C to quit the app."""

    def test_shutdown_cancels_active_worker(self, loaded_player):
        worker = MagicMock()
        worker.isRunning.return_value = True
        loaded_player._stretch_worker = worker

        loaded_player.shutdown(wait_ms=10)

        worker.cancel.assert_called_once()
        worker.wait.assert_called_once_with(10)

    def test_shutdown_cancels_detached_workers(self, loaded_player):
        a = MagicMock()
        a.isRunning.return_value = True
        b = MagicMock()
        b.isRunning.return_value = True
        loaded_player._detached_workers = [a, b]

        loaded_player.shutdown(wait_ms=5)

        a.cancel.assert_called_once()
        b.cancel.assert_called_once()
        a.wait.assert_called_once_with(5)
        b.wait.assert_called_once_with(5)

    def test_shutdown_clears_pending_render(self, loaded_player):
        loaded_player._pending_render_emit = ("speed",)
        loaded_player.shutdown(wait_ms=0)
        assert loaded_player._pending_render_emit is None

    def test_shutdown_stops_position_timer(self, loaded_player):
        loaded_player._timer.start()
        assert loaded_player._timer.isActive()

        loaded_player.shutdown(wait_ms=0)

        assert not loaded_player._timer.isActive()

    def test_shutdown_no_wait_for_already_stopped(self, loaded_player):
        """Workers that already finished must not get a .wait() call
        (Qt thread may be destroyed by then)."""
        done = MagicMock()
        done.isRunning.return_value = False
        loaded_player._stretch_worker = done

        loaded_player.shutdown(wait_ms=10)

        done.cancel.assert_called_once()
        done.wait.assert_not_called()

    def test_shutdown_is_safe_with_no_workers(self, loaded_player):
        """Idle-shutdown is a no-op but must not raise."""
        loaded_player._stretch_worker = None
        loaded_player._detached_workers = []
        loaded_player.shutdown(wait_ms=0)  # should not raise


# -----------------------------------------------------------------------
# Regression: a render discarded by cancel_stretch() must not be lost
# -----------------------------------------------------------------------

class TestCancelledRenderNotLost:
    """set_speed/set_pitch used to no-op purely on the requested value.

    The knobs are set optimistically before the render lands, and
    cancel_stretch() can discard that render -- after which "value
    unchanged" does not mean "nothing to do". The player now tracks the
    applied render state and re-renders when the stems are stale.
    """

    def test_set_pitch_same_value_rerenders_when_stems_stale(
        self, loaded_player,
    ):
        """Speed knob is ahead of the stems; a pitch no-op must render."""
        loaded_player._playback_speed = 0.75  # knob turned, render lost
        with patch("src.player.StretchWorker") as worker_cls:
            worker_cls.return_value.isRunning.return_value = False
            loaded_player.set_pitch(0)  # unchanged pitch value
            assert worker_cls.called
        args = worker_cls.call_args[0]
        assert args[2] == 0.75  # the owed speed render is included

    def test_set_speed_same_value_rerenders_when_stems_stale(
        self, loaded_player,
    ):
        """Pitch knob is ahead of the stems; a speed no-op must render."""
        loaded_player._pitch_semitones = 2
        with patch("src.player.StretchWorker") as worker_cls:
            worker_cls.return_value.isRunning.return_value = False
            loaded_player.set_speed(1.0)  # unchanged speed value
            assert worker_cls.called
        args = worker_cls.call_args[0]
        assert args[3] == 2  # the owed pitch render is included

    def test_scrub_back_during_speed_render_still_applies_speed(
        self, loaded_player,
    ):
        """Full UI flow: speed render in flight, pitch scrubbed away and
        back (the UI cancels on every tick), debounce flush lands on the
        current pitch value -- the speed render must still happen."""
        with patch("src.player.StretchWorker") as worker_cls:
            live = MagicMock()
            live.isRunning.return_value = True
            worker_cls.return_value = live
            loaded_player.set_speed(0.75)
        loaded_player.cancel_stretch()
        assert live in loaded_player._detached_workers

        with patch("src.player.StretchWorker") as worker_cls2:
            worker_cls2.return_value.isRunning.return_value = False
            loaded_player.set_pitch(0)
            # The cancelled worker is still draining: the render queues...
            assert loaded_player._pending_render_emit is not None
            # ...and dispatches once the drain finishes.
            loaded_player._reap_detached_worker(live)
            assert worker_cls2.called
            args = worker_cls2.call_args[0]
            assert args[2] == 0.75

    def test_true_noop_still_skips_render(self, loaded_player):
        """With stems current, unchanged values spawn nothing."""
        with patch("src.player.StretchWorker") as worker_cls:
            loaded_player.set_pitch(0)
            loaded_player.set_speed(1.0)
            worker_cls.assert_not_called()

    def test_sync_toggle_at_pitch_zero_does_not_mark_stale(
        self, loaded_player,
    ):
        """The sync flag is inaudible at pitch 0; toggling it must not
        make the next no-op set_speed/set_pitch re-render."""
        loaded_player.set_sync_recording_pitch(True)
        with patch("src.player.StretchWorker") as worker_cls:
            loaded_player.set_pitch(0)
            loaded_player.set_speed(1.0)
            worker_cls.assert_not_called()

    def test_apply_stretched_stems_records_applied_state(
        self, loaded_player,
    ):
        loaded_player._playback_speed = 0.75
        loaded_player._pitch_semitones = 2
        loaded_player._apply_stretched_stems(
            dict(loaded_player._original_stems)
        )
        assert loaded_player._applied_render_state == (0.75, 2, False)

    def test_load_stems_resets_applied_state(self, loaded_player, tmp_path):
        import soundfile as sf
        loaded_player._applied_render_state = (0.75, 2, False)
        wav_path = tmp_path / "fake.wav"
        sf.write(str(wav_path), np.zeros((4410, 2), dtype=np.float32), 44100)
        loaded_player.load_stems({"vocals": str(wav_path)})
        assert loaded_player._applied_render_state == (1.0, 0, False)


# -----------------------------------------------------------------------
# Regression: worker finishing between isRunning() and connect() must
# not wedge the render queue
# -----------------------------------------------------------------------

class TestDetachReapRace:
    def test_detach_reaps_worker_that_finished_before_connect(
        self, loaded_player,
    ):
        """isRunning flips False right after the keepalive connect: the
        finished signal fired unheard, so detach must reap directly --
        otherwise _detached_workers never drains and every queued render
        waits forever."""
        fake = MagicMock()
        fake.isRunning.side_effect = [True, False]
        loaded_player._stretch_worker = fake

        loaded_player._detach_stretch_worker()

        assert fake not in loaded_player._detached_workers
        fake.deleteLater.assert_called_once()

    def test_reaper_double_call_does_not_touch_worker_again(
        self, loaded_player,
    ):
        """Second delivery (manual reap plus the queued finished signal)
        must not poke a wrapper that may already be deleteLater'd."""
        fake = MagicMock()
        loaded_player._detached_workers.append(fake)
        loaded_player._reap_detached_worker(fake)
        fake.setParent.reset_mock()
        fake.deleteLater.reset_mock()

        loaded_player._reap_detached_worker(fake)

        fake.setParent.assert_not_called()
        fake.deleteLater.assert_not_called()

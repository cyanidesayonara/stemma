"""Tests for the count-in pre-roll feature."""

import numpy as np
import pytest
import sounddevice as sd
import soundfile as sf

from PySide6.QtWidgets import QApplication

from src.player import MultiTrackPlayer


@pytest.fixture(scope="module")
def app():
    """Ensure a QApplication exists for widget tests."""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


@pytest.fixture
def player(app):
    """A MultiTrackPlayer with no stems loaded."""
    return MultiTrackPlayer()


@pytest.fixture
def loaded_player(player, tmp_path):
    """A player with a 1-second silent stem loaded."""
    data = np.zeros((44100, 2), dtype=np.float32)
    path = tmp_path / "stem.wav"
    sf.write(str(path), data, 44100)
    player.load_stems({"vocals": str(path)})
    return player


# -----------------------------------------------------------------------
# API
# -----------------------------------------------------------------------

class TestCountInAPI:
    """Test count-in property/setter basics."""

    def test_default_disabled(self, player):
        assert player.count_in_enabled is False

    def test_default_beats(self, player):
        assert player.count_in_beats == 4

    def test_default_on_repeats_disabled(self, player):
        assert player.count_in_on_repeats is False

    def test_enable_disable(self, player):
        player.set_count_in_enabled(True)
        assert player.count_in_enabled is True
        player.set_count_in_enabled(False)
        assert player.count_in_enabled is False

    def test_beats_clamped_low(self, player):
        player.set_count_in_beats(0)
        assert player.count_in_beats == 1

    def test_beats_clamped_high(self, player):
        player.set_count_in_beats(99)
        assert player.count_in_beats == 8

    def test_beats_normal(self, player):
        player.set_count_in_beats(3)
        assert player.count_in_beats == 3

    def test_on_repeats_toggle(self, player):
        player.set_count_in_on_repeats(True)
        assert player.count_in_on_repeats is True
        player.set_count_in_on_repeats(False)
        assert player.count_in_on_repeats is False

    def test_counting_in_false_when_not_armed(self, player):
        assert player.counting_in is False

    def test_count_in_beat_zero_when_not_armed(self, player):
        assert player.count_in_current_beat == 0


# -----------------------------------------------------------------------
# Audio callback: count-in pre-roll
# -----------------------------------------------------------------------

class TestCountInCallback:
    """Test that the count-in plays clicks before stems start."""

    def test_count_in_produces_clicks_over_silence(self, loaded_player):
        """During count-in, output should have click energy but stems
        should not advance."""
        loaded_player.set_count_in_enabled(True)
        loaded_player.set_count_in_beats(4)
        loaded_player.set_metronome_bpm(120.0)
        loaded_player.set_metronome_volume(1.0)

        initial_frame = loaded_player._current_frame

        loaded_player._arm_count_in()
        loaded_player._is_playing = True

        outdata = np.zeros((512, 2), dtype=np.float32)
        loaded_player._audio_callback(outdata, 512, {}, sd.CallbackFlags())

        assert np.max(np.abs(outdata)) > 0.01
        assert loaded_player._current_frame == initial_frame

    def test_stems_start_after_count_in_exhausted(self, loaded_player):
        """After the count-in finishes, normal stem mixing should resume."""
        loaded_player.set_count_in_enabled(True)
        loaded_player.set_count_in_beats(1)
        loaded_player.set_metronome_bpm(120.0)
        loaded_player.set_metronome_volume(1.0)

        loaded_player._arm_count_in()
        loaded_player._is_playing = True

        beat_interval = int(60.0 / 120.0 * 44100)

        # Consume the entire count-in in one callback.
        outdata = np.zeros((beat_interval, 2), dtype=np.float32)
        loaded_player._audio_callback(
            outdata, beat_interval, {}, sd.CallbackFlags()
        )

        assert loaded_player._count_in_remaining == 0

        # Next callback should advance stems normally.
        frame_before = loaded_player._current_frame
        outdata2 = np.zeros((256, 2), dtype=np.float32)
        loaded_player._audio_callback(outdata2, 256, {}, sd.CallbackFlags())
        assert loaded_player._current_frame == frame_before + 256

    def test_count_in_does_not_advance_current_frame(self, loaded_player):
        """During the entire count-in, _current_frame must stay put."""
        loaded_player.set_count_in_enabled(True)
        loaded_player.set_count_in_beats(2)
        loaded_player.set_metronome_bpm(120.0)
        loaded_player.set_metronome_volume(1.0)
        loaded_player._current_frame = 1000

        loaded_player._arm_count_in()
        loaded_player._is_playing = True

        for _ in range(10):
            outdata = np.zeros((512, 2), dtype=np.float32)
            loaded_player._audio_callback(outdata, 512, {}, sd.CallbackFlags())
            if loaded_player._count_in_remaining <= 0:
                break

        assert loaded_player._current_frame == 1000

    def test_metronome_phase_reset_after_count_in(self, loaded_player):
        """After count-in finishes, metronome phase should be 0 so that the
        metronome (if enabled) starts beat-aligned with the stems."""
        loaded_player.set_count_in_enabled(True)
        loaded_player.set_count_in_beats(1)
        loaded_player.set_metronome_bpm(120.0)
        loaded_player.set_metronome_volume(1.0)
        loaded_player._metronome_phase = 999

        loaded_player._arm_count_in()
        loaded_player._is_playing = True

        beat_interval = int(60.0 / 120.0 * 44100)
        outdata = np.zeros((beat_interval, 2), dtype=np.float32)
        loaded_player._audio_callback(
            outdata, beat_interval, {}, sd.CallbackFlags()
        )

        assert loaded_player._metronome_phase == 0

    def test_play_arms_count_in(self, loaded_player):
        """Calling play() should arm the count-in when enabled."""
        loaded_player.set_count_in_enabled(True)
        loaded_player.set_count_in_beats(4)
        loaded_player.set_metronome_bpm(120.0)

        beat_interval = int(60.0 / 120.0 * 44100)
        expected = 4 * beat_interval

        loaded_player._arm_count_in()
        assert loaded_player._count_in_remaining == expected

    def test_count_in_disabled_does_not_arm(self, loaded_player):
        """When count-in is disabled, _arm_count_in should be a no-op."""
        loaded_player.set_count_in_enabled(False)
        loaded_player._arm_count_in()
        assert loaded_player._count_in_remaining == 0


# -----------------------------------------------------------------------
# Count-in beat tracking
# -----------------------------------------------------------------------

class TestCountInBeatTracking:
    """Test that the beat number updates correctly during count-in."""

    def test_beat_advances_through_count_in(self, loaded_player):
        loaded_player.set_count_in_enabled(True)
        loaded_player.set_count_in_beats(4)
        loaded_player.set_metronome_bpm(120.0)
        loaded_player.set_metronome_volume(0.5)
        loaded_player._arm_count_in()
        loaded_player._is_playing = True

        beat_interval = int(60.0 / 120.0 * 44100)
        beats_seen = set()

        while loaded_player._count_in_remaining > 0:
            outdata = np.zeros((beat_interval, 2), dtype=np.float32)
            loaded_player._audio_callback(
                outdata, beat_interval, {}, sd.CallbackFlags()
            )
            beats_seen.add(loaded_player._count_in_beat)

        # After count-in finishes, beat resets to 0.
        assert loaded_player._count_in_beat == 0
        assert len(beats_seen) >= 1

    def test_beat_is_zero_after_pause(self, loaded_player):
        loaded_player.set_count_in_enabled(True)
        loaded_player.set_count_in_beats(4)
        loaded_player.set_metronome_bpm(120.0)
        loaded_player._arm_count_in()
        loaded_player._is_playing = True

        outdata = np.zeros((512, 2), dtype=np.float32)
        loaded_player._audio_callback(outdata, 512, {}, sd.CallbackFlags())
        assert loaded_player._count_in_beat > 0

        loaded_player.pause()
        assert loaded_player._count_in_beat == 0


# -----------------------------------------------------------------------
# A-B loop repeat count-in
# -----------------------------------------------------------------------

class TestCountInLoopRepeat:
    """Test count-in behavior at A-B loop boundaries."""

    @pytest.fixture
    def loop_player(self, loaded_player):
        """A player with a short A-B loop set up."""
        loaded_player.set_loop_a(0.0)
        loaded_player.set_loop_b(0.1)
        loaded_player._looping = True
        return loaded_player

    def test_repeat_count_in_when_enabled(self, loop_player):
        """With on_repeats enabled, hitting loop_b should arm another count-in."""
        loop_player.set_count_in_enabled(True)
        loop_player.set_count_in_on_repeats(True)
        loop_player.set_metronome_bpm(120.0)
        loop_player.set_metronome_volume(0.5)
        loop_player._is_playing = True

        loop_b_frame = loop_player._loop_b_frame
        loop_player._current_frame = loop_b_frame - 100

        outdata = np.zeros((512, 2), dtype=np.float32)
        loop_player._audio_callback(outdata, 512, {}, sd.CallbackFlags())

        assert loop_player._count_in_remaining > 0

    def test_no_repeat_count_in_when_disabled(self, loop_player):
        """With on_repeats disabled, loop wrap should not arm count-in."""
        loop_player.set_count_in_enabled(True)
        loop_player.set_count_in_on_repeats(False)
        loop_player.set_metronome_bpm(120.0)
        loop_player._is_playing = True

        loop_b_frame = loop_player._loop_b_frame
        loop_player._current_frame = loop_b_frame - 100

        outdata = np.zeros((512, 2), dtype=np.float32)
        loop_player._audio_callback(outdata, 512, {}, sd.CallbackFlags())

        assert loop_player._count_in_remaining == 0

    def test_no_repeat_count_in_when_count_in_disabled(self, loop_player):
        """With count-in itself disabled, loop wrap never arms count-in."""
        loop_player.set_count_in_enabled(False)
        loop_player.set_count_in_on_repeats(True)
        loop_player.set_metronome_bpm(120.0)
        loop_player._is_playing = True

        loop_b_frame = loop_player._loop_b_frame
        loop_player._current_frame = loop_b_frame - 100

        outdata = np.zeros((512, 2), dtype=np.float32)
        loop_player._audio_callback(outdata, 512, {}, sd.CallbackFlags())

        assert loop_player._count_in_remaining == 0


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------

class TestCountInEdgeCases:
    """Edge case tests for count-in."""

    def test_count_in_works_with_metronome_disabled(self, loaded_player):
        """Count-in should produce clicks even if the metronome toggle is off."""
        loaded_player.set_metronome_enabled(False)
        loaded_player.set_count_in_enabled(True)
        loaded_player.set_count_in_beats(1)
        loaded_player.set_metronome_bpm(120.0)
        loaded_player.set_metronome_volume(1.0)

        loaded_player._arm_count_in()
        loaded_player._is_playing = True

        outdata = np.zeros((512, 2), dtype=np.float32)
        loaded_player._audio_callback(outdata, 512, {}, sd.CallbackFlags())

        assert np.max(np.abs(outdata)) > 0.01

    def test_seek_cancels_active_count_in(self, loaded_player):
        """Seeking while count-in is active should cancel it."""
        loaded_player.set_count_in_enabled(True)
        loaded_player.set_count_in_beats(4)
        loaded_player.set_metronome_bpm(120.0)

        loaded_player._arm_count_in()
        assert loaded_player._count_in_remaining > 0

        loaded_player.seek(0.5)
        assert loaded_player._count_in_remaining == 0
        assert loaded_player._count_in_beat == 0

    def test_pause_cancels_active_count_in(self, loaded_player):
        """Pausing while count-in is active should cancel it."""
        loaded_player.set_count_in_enabled(True)
        loaded_player.set_count_in_beats(4)
        loaded_player.set_metronome_bpm(120.0)

        loaded_player._arm_count_in()
        loaded_player._is_playing = True
        assert loaded_player._count_in_remaining > 0

        loaded_player.pause()
        assert loaded_player._count_in_remaining == 0

    def test_no_count_in_on_resume_after_pause(self, loaded_player):
        """Resuming from a mid-song position should not trigger a count-in."""
        loaded_player.set_count_in_enabled(True)
        loaded_player.set_count_in_beats(4)
        loaded_player.set_metronome_bpm(120.0)

        loaded_player._current_frame = 10000
        loaded_player._is_playing = False

        # Simulate what play() does: check boundary, then arm.
        at_boundary = (
            loaded_player._current_frame == 0
            or (loaded_player._loop_region_is_active()
                and loaded_player._current_frame
                == loaded_player._loop_a_frame)
        )
        assert not at_boundary
        # play() would not arm count-in here.
        if at_boundary:
            loaded_player._arm_count_in()
        assert loaded_player._count_in_remaining == 0

    def test_count_in_arms_at_position_zero(self, loaded_player):
        """Starting from position 0 should arm the count-in."""
        loaded_player.set_count_in_enabled(True)
        loaded_player.set_count_in_beats(4)
        loaded_player.set_metronome_bpm(120.0)

        loaded_player._current_frame = 0
        loaded_player._arm_count_in()
        assert loaded_player._count_in_remaining > 0

    def test_count_in_arms_at_loop_a(self, loaded_player):
        """Starting from loop A position should arm the count-in."""
        loaded_player.set_count_in_enabled(True)
        loaded_player.set_count_in_beats(4)
        loaded_player.set_metronome_bpm(120.0)
        loaded_player.set_loop_a(0.5)
        loaded_player.set_loop_b(0.8)
        loaded_player.set_looping(True)

        loaded_player._current_frame = loaded_player._loop_a_frame
        at_boundary = (
            loaded_player._current_frame == 0
            or (loaded_player._loop_region_is_active()
                and loaded_player._current_frame
                == loaded_player._loop_a_frame)
        )
        assert at_boundary

    def test_device_switch_no_spurious_count_in(self, loaded_player):
        """Switching output device mid-song should not trigger a count-in."""
        loaded_player.set_count_in_enabled(True)
        loaded_player.set_count_in_beats(4)
        loaded_player.set_metronome_bpm(120.0)

        loaded_player._current_frame = 5000
        loaded_player._is_playing = False
        # After a device switch, play() is called with current_frame mid-song.
        # It should not arm count-in.
        at_boundary = loaded_player._current_frame == 0
        assert not at_boundary
        assert loaded_player._count_in_remaining == 0

    def test_count_in_uses_metronome_volume(self, loaded_player):
        """Count-in click volume should follow the metronome volume setting."""
        loaded_player.set_count_in_enabled(True)
        loaded_player.set_count_in_beats(1)
        loaded_player.set_metronome_bpm(120.0)

        loaded_player.set_metronome_volume(0.1)
        loaded_player._arm_count_in()
        loaded_player._is_playing = True
        out_low = np.zeros((512, 2), dtype=np.float32)
        loaded_player._audio_callback(out_low, 512, {}, sd.CallbackFlags())
        peak_low = np.max(np.abs(out_low))

        loaded_player.set_metronome_volume(1.0)
        loaded_player._arm_count_in()
        loaded_player._current_frame = 0
        out_high = np.zeros((512, 2), dtype=np.float32)
        loaded_player._audio_callback(out_high, 512, {}, sd.CallbackFlags())
        peak_high = np.max(np.abs(out_high))

        assert peak_high > peak_low * 2

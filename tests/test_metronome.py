"""Tests for metronome click track and tap tempo."""

import numpy as np
import pytest
import sounddevice as sd

from PySide6.QtWidgets import QApplication

from src.metronome import tap_tempo
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


# -----------------------------------------------------------------------
# Click buffer generation
# -----------------------------------------------------------------------

class TestClickBuffer:
    """Verify the synthesized click sound is well-formed."""

    def test_click_shape_is_stereo(self, player):
        """Click buffer should be a 2-channel numpy array."""
        buf = player._click_buf
        assert buf.ndim == 2
        assert buf.shape[1] == 2

    def test_click_is_nonzero(self, player):
        """Click buffer should contain audible samples."""
        assert np.max(np.abs(player._click_buf)) > 0.1

    def test_click_decays(self, player):
        """The end of the click should be quieter than the start."""
        buf = player._click_buf
        first_quarter = np.max(np.abs(buf[: len(buf) // 4]))
        last_quarter = np.max(np.abs(buf[-(len(buf) // 4):]))
        assert last_quarter < first_quarter

    def test_click_duration_is_30ms(self, player):
        """Click buffer length should match 30ms at the sample rate."""
        expected = int(player._sample_rate * 0.03)
        assert len(player._click_buf) == expected

    def test_click_regenerated_on_load_stems(self, player, tmp_path):
        """Loading stems with a different sample rate regenerates the click."""
        import soundfile as sf
        sr = 22050
        data = np.zeros((sr, 2), dtype=np.float32)
        path = tmp_path / "stem.wav"
        sf.write(str(path), data, sr)
        player.load_stems({"vocals": str(path)})
        expected = int(sr * 0.03)
        assert len(player._click_buf) == expected


# -----------------------------------------------------------------------
# Metronome public API
# -----------------------------------------------------------------------

class TestMetronomeAPI:
    """Test metronome enable/disable, BPM, and volume setters."""

    def test_default_state(self, player):
        """Metronome should be disabled by default at 120 BPM."""
        assert player.metronome_enabled is False
        assert player.metronome_bpm == 120.0
        assert player.metronome_volume == 0.5

    def test_set_enabled(self, player):
        player.set_metronome_enabled(True)
        assert player.metronome_enabled is True

    def test_set_bpm(self, player):
        player.set_metronome_bpm(90.0)
        assert player.metronome_bpm == 90.0

    def test_bpm_clamped_low(self, player):
        player.set_metronome_bpm(5.0)
        assert player.metronome_bpm == 20.0

    def test_bpm_clamped_high(self, player):
        player.set_metronome_bpm(999.0)
        assert player.metronome_bpm == 300.0

    def test_volume_clamped_low(self, player):
        player.set_metronome_volume(-1.0)
        assert player.metronome_volume == 0.0

    def test_volume_clamped_high(self, player):
        player.set_metronome_volume(5.0)
        assert player.metronome_volume == 2.0

    def test_phase_resets_on_enable(self, player):
        """Enabling the metronome should reset the beat phase."""
        player._metronome_phase = 100
        player.set_metronome_enabled(True)
        assert player._metronome_phase == 0

    def test_phase_resets_on_bpm_change(self, player):
        """Changing BPM should reset the beat phase."""
        player._metronome_phase = 100
        player.set_metronome_bpm(100.0)
        assert player._metronome_phase == 0

    def test_phase_resets_on_seek(self, player, tmp_path):
        """Seeking should reset the metronome phase."""
        import soundfile as sf
        data = np.zeros((44100, 2), dtype=np.float32)
        path = tmp_path / "stem.wav"
        sf.write(str(path), data, 44100)
        player.load_stems({"vocals": str(path)})
        player._metronome_phase = 500
        player.seek(0.5)
        assert player._metronome_phase == 0


# -----------------------------------------------------------------------
# Audio callback integration
# -----------------------------------------------------------------------

class TestMetronomeCallback:
    """Test that the metronome click is mixed into the audio callback."""

    @pytest.fixture
    def loaded_player(self, player, tmp_path):
        """A player with a 1-second silent stem loaded."""
        import soundfile as sf
        data = np.zeros((44100, 2), dtype=np.float32)
        path = tmp_path / "stem.wav"
        sf.write(str(path), data, 44100)
        player.load_stems({"vocals": str(path)})
        return player

    def test_metronome_disabled_produces_silence(self, loaded_player):
        """With metronome off, silence in = silence out."""
        loaded_player.set_metronome_enabled(False)
        loaded_player._is_playing = True
        outdata = np.zeros((512, 2), dtype=np.float32)
        loaded_player._audio_callback(outdata, 512, {}, sd.CallbackFlags())
        assert np.max(np.abs(outdata)) == 0.0

    def test_metronome_enabled_produces_click_at_start(self, loaded_player):
        """With metronome on, the first buffer should contain a click."""
        loaded_player.set_metronome_enabled(True)
        loaded_player.set_metronome_bpm(120.0)
        loaded_player.set_metronome_volume(1.0)
        loaded_player._is_playing = True
        outdata = np.zeros((512, 2), dtype=np.float32)
        loaded_player._audio_callback(outdata, 512, {}, sd.CallbackFlags())
        # The click starts at phase 0, so the first samples should be nonzero.
        assert np.max(np.abs(outdata[:100])) > 0.01

    def test_metronome_click_repeats_at_bpm(self, loaded_player):
        """At 120 BPM (0.5s per beat), a click should appear every 22050 frames."""
        loaded_player.set_metronome_enabled(True)
        loaded_player.set_metronome_bpm(120.0)
        loaded_player.set_metronome_volume(1.0)
        loaded_player._is_playing = True

        # Process enough frames to pass one full beat interval (22050 at 44100Hz).
        beat_interval = int(60.0 / 120.0 * 44100)
        buf_size = beat_interval + 512
        outdata = np.zeros((buf_size, 2), dtype=np.float32)
        loaded_player._audio_callback(outdata, buf_size, {}, sd.CallbackFlags())

        # There should be a click at sample 0 and another at sample 22050.
        first_click_peak = np.max(np.abs(outdata[:100]))
        second_click_start = beat_interval
        second_click_peak = np.max(
            np.abs(outdata[second_click_start:second_click_start + 100])
        )
        assert first_click_peak > 0.01
        assert second_click_peak > 0.01

    def test_metronome_volume_scales_output(self, loaded_player):
        """Higher volume should produce louder clicks."""
        loaded_player.set_metronome_enabled(True)
        loaded_player.set_metronome_bpm(120.0)
        loaded_player._is_playing = True

        # Low volume
        loaded_player.set_metronome_volume(0.1)
        loaded_player._metronome_phase = 0
        loaded_player._current_frame = 0
        out_low = np.zeros((512, 2), dtype=np.float32)
        loaded_player._audio_callback(out_low, 512, {}, sd.CallbackFlags())
        peak_low = np.max(np.abs(out_low))

        # High volume
        loaded_player.set_metronome_volume(1.0)
        loaded_player._metronome_phase = 0
        loaded_player._current_frame = 0
        loaded_player._is_playing = True
        out_high = np.zeros((512, 2), dtype=np.float32)
        loaded_player._audio_callback(out_high, 512, {}, sd.CallbackFlags())
        peak_high = np.max(np.abs(out_high))

        assert peak_high > peak_low * 2

    def test_metronome_phase_continues_across_callbacks(self, loaded_player):
        """Phase should carry over between callbacks without clicks drifting."""
        loaded_player.set_metronome_enabled(True)
        loaded_player.set_metronome_bpm(120.0)
        loaded_player.set_metronome_volume(1.0)
        loaded_player._is_playing = True

        # First callback
        out1 = np.zeros((256, 2), dtype=np.float32)
        loaded_player._audio_callback(out1, 256, {}, sd.CallbackFlags())
        phase_after_1 = loaded_player._metronome_phase
        assert phase_after_1 == 256

        # Second callback: phase should continue from 256.
        out2 = np.zeros((256, 2), dtype=np.float32)
        loaded_player._audio_callback(out2, 256, {}, sd.CallbackFlags())
        phase_after_2 = loaded_player._metronome_phase
        assert phase_after_2 == 512

    def test_metronome_phase_advances_full_block_at_eof(self, loaded_player):
        """At EOF the phase must advance by the full block size, not just the
        stem-filled portion, so the metronome stays in sync with wall-clock
        time through partial buffers."""
        loaded_player.set_metronome_enabled(True)
        loaded_player.set_metronome_bpm(120.0)
        loaded_player.set_metronome_volume(1.0)
        loaded_player._is_playing = True

        # Seek close to the end so the next callback only partially fills.
        total = loaded_player._total_frames
        loaded_player._current_frame = total - 100

        block_size = 512
        outdata = np.zeros((block_size, 2), dtype=np.float32)
        try:
            loaded_player._audio_callback(
                outdata, block_size, {}, sd.CallbackFlags()
            )
        except sd.CallbackStop:
            pass

        # Phase should reflect the full block (512), not the partial fill (100).
        assert loaded_player._metronome_phase == block_size


# -----------------------------------------------------------------------
# NaN / non-finite guard
# -----------------------------------------------------------------------

class TestMetronomeNanGuard:
    """set_metronome_bpm must reject non-finite values."""

    @pytest.fixture
    def player(self, app):
        return MultiTrackPlayer()

    def test_nan_bpm_is_ignored(self, player):
        player.set_metronome_bpm(100.0)
        player.set_metronome_bpm(float("nan"))
        assert player.metronome_bpm == 100.0

    def test_inf_bpm_is_ignored(self, player):
        player.set_metronome_bpm(100.0)
        player.set_metronome_bpm(float("inf"))
        assert player.metronome_bpm == 100.0

    def test_negative_inf_bpm_is_ignored(self, player):
        player.set_metronome_bpm(100.0)
        player.set_metronome_bpm(float("-inf"))
        assert player.metronome_bpm == 100.0

    def test_valid_bpm_still_accepted(self, player):
        player.set_metronome_bpm(90.0)
        assert player.metronome_bpm == 90.0


# -----------------------------------------------------------------------
# Tap tempo
# -----------------------------------------------------------------------

class TestTapTempo:
    """Test the tap_tempo utility function."""

    def test_fewer_than_two_taps_returns_zero(self):
        assert tap_tempo([]) == 0.0
        assert tap_tempo([1.0]) == 0.0

    def test_two_taps_at_half_second(self):
        """Two taps 0.5s apart = 120 BPM."""
        bpm = tap_tempo([0.0, 0.5])
        assert abs(bpm - 120.0) < 0.1

    def test_four_taps_at_half_second(self):
        """Four taps at 0.5s intervals = 120 BPM."""
        bpm = tap_tempo([0.0, 0.5, 1.0, 1.5])
        assert abs(bpm - 120.0) < 0.1

    def test_max_taps_limits_history(self):
        """Only the last max_taps entries are used."""
        # First 4 taps at 1s intervals (60 BPM), last 3 at 0.5s (120 BPM).
        taps = [0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5]
        bpm = tap_tempo(taps, max_taps=3)
        assert abs(bpm - 120.0) < 0.1

    def test_uneven_intervals_averaged(self):
        """BPM should be based on average interval."""
        # Intervals: 0.4, 0.6 -> avg 0.5 -> 120 BPM
        bpm = tap_tempo([0.0, 0.4, 1.0])
        assert abs(bpm - 120.0) < 0.1


# -----------------------------------------------------------------------
# Beat-synced metronome
# -----------------------------------------------------------------------

class TestBeatSyncAPI:
    """Test the beat-synced metronome public API."""

    def test_default_disabled(self, player):
        assert player.beat_sync_enabled is False

    def test_set_beat_sync(self, player):
        player.set_beat_sync_enabled(True)
        assert player.beat_sync_enabled is True
        player.set_beat_sync_enabled(False)
        assert player.beat_sync_enabled is False

    def test_beat_frames_computed_from_beat_times(self, player):
        """set_beat_times should compute _beat_frames at the correct positions."""
        player._sample_rate = 44100
        player._playback_speed = 1.0
        player.set_beat_times([0.0, 0.5, 1.0], [])
        expected = np.array([0, 22050, 44100], dtype=np.int64)
        np.testing.assert_array_equal(player._beat_frames, expected)

    def test_beat_frames_scaled_by_speed(self, player):
        """At half speed, beat frames should be doubled (audio is stretched)."""
        player._sample_rate = 44100
        player._playback_speed = 0.5
        player.set_beat_times([0.0, 0.5, 1.0], [])
        expected = np.array([0, 44100, 88200], dtype=np.int64)
        np.testing.assert_array_equal(player._beat_frames, expected)

    def test_beat_frames_empty_when_no_beats(self, player):
        player._sample_rate = 44100
        player.set_beat_times([], [])
        assert len(player._beat_frames) == 0

    def test_instantaneous_bpm_mid_track(self, player):
        """BPM between two beats spaced 0.5s apart should be ~120."""
        player._sample_rate = 44100
        player._playback_speed = 1.0
        player.set_beat_times([0.0, 0.5, 1.0, 1.5], [])
        bpm = player.instantaneous_bpm_at(11025)  # 0.25s
        assert abs(bpm - 120.0) < 0.1

    def test_instantaneous_bpm_too_few_beats(self, player):
        player._sample_rate = 44100
        player.set_beat_times([0.5], [])
        assert player.instantaneous_bpm_at(0) == 0.0

    def test_instantaneous_bpm_at_end(self, player):
        """Past the last beat, should use the last interval."""
        player._sample_rate = 44100
        player._playback_speed = 1.0
        player.set_beat_times([0.0, 0.5, 1.0], [])
        bpm = player.instantaneous_bpm_at(55000)  # past 1.0s
        assert abs(bpm - 120.0) < 0.1


class TestBeatSyncCallback:
    """Test synced metronome clicks in the audio callback."""

    @pytest.fixture
    def loaded_player(self, player, tmp_path):
        """A player with a 2-second silent stem loaded."""
        import soundfile as sf
        data = np.zeros((88200, 2), dtype=np.float32)
        path = tmp_path / "stem.wav"
        sf.write(str(path), data, 44100)
        player.load_stems({"vocals": str(path)})
        return player

    def test_synced_click_at_beat_zero(self, loaded_player):
        """Beat at frame 0 should produce a click at the start of the buffer."""
        loaded_player.set_beat_times([0.0, 0.5, 1.0], [])
        loaded_player.set_beat_sync_enabled(True)
        loaded_player.set_metronome_enabled(True)
        loaded_player.set_metronome_volume(1.0)
        loaded_player._is_playing = True

        outdata = np.zeros((512, 2), dtype=np.float32)
        loaded_player._audio_callback(outdata, 512, {}, sd.CallbackFlags())
        assert np.max(np.abs(outdata[:100])) > 0.01

    def test_synced_no_click_between_beats(self, loaded_player):
        """Between beats, the synced metronome should be silent."""
        loaded_player.set_beat_times([0.0, 1.0], [])
        loaded_player.set_beat_sync_enabled(True)
        loaded_player.set_metronome_enabled(True)
        loaded_player.set_metronome_volume(1.0)
        loaded_player._is_playing = True

        # Advance past the first beat's click tail.
        click_len = len(loaded_player._click_buf)
        loaded_player._current_frame = click_len + 100

        outdata = np.zeros((512, 2), dtype=np.float32)
        loaded_player._audio_callback(outdata, 512, {}, sd.CallbackFlags())
        assert np.max(np.abs(outdata)) == 0.0

    def test_synced_click_mid_buffer(self, loaded_player):
        """A beat that falls in the middle of a buffer should produce a click there."""
        # Beat at 0.5s = frame 22050.
        loaded_player.set_beat_times([0.5], [])
        loaded_player.set_beat_sync_enabled(True)
        loaded_player.set_metronome_enabled(True)
        loaded_player.set_metronome_volume(1.0)
        loaded_player._is_playing = True

        # Start just before the beat so it falls in our buffer.
        loaded_player._current_frame = 22000
        buf_size = 512
        outdata = np.zeros((buf_size, 2), dtype=np.float32)
        loaded_player._audio_callback(outdata, buf_size, {}, sd.CallbackFlags())

        # Click should appear at offset 50 (22050 - 22000).
        offset = 50
        assert np.max(np.abs(outdata[offset:offset + 50])) > 0.01
        # Samples before the beat should be silent (from stems, which are zeros).
        assert np.max(np.abs(outdata[:offset])) == 0.0

    def test_synced_fallback_to_grid_when_no_beats(self, loaded_player):
        """With sync enabled but no beats, grid metronome should be used."""
        loaded_player.set_beat_times([], [])
        loaded_player.set_beat_sync_enabled(True)
        loaded_player.set_metronome_enabled(True)
        loaded_player.set_metronome_bpm(120.0)
        loaded_player.set_metronome_volume(1.0)
        loaded_player._is_playing = True

        outdata = np.zeros((512, 2), dtype=np.float32)
        loaded_player._audio_callback(outdata, 512, {}, sd.CallbackFlags())
        # Grid metronome starts with a click at phase 0.
        assert np.max(np.abs(outdata[:100])) > 0.01

    def test_synced_disabled_uses_grid(self, loaded_player):
        """With sync disabled, grid metronome should be used even with beats."""
        loaded_player.set_beat_times([0.0, 0.5, 1.0], [])
        loaded_player.set_beat_sync_enabled(False)
        loaded_player.set_metronome_enabled(True)
        loaded_player.set_metronome_bpm(120.0)
        loaded_player.set_metronome_volume(1.0)
        loaded_player._is_playing = True

        outdata = np.zeros((512, 2), dtype=np.float32)
        loaded_player._audio_callback(outdata, 512, {}, sd.CallbackFlags())
        # Grid metronome should produce a click.
        assert np.max(np.abs(outdata[:100])) > 0.01
        # Phase should advance (grid mode behaviour).
        assert loaded_player._metronome_phase == 512

    def test_synced_seek_then_click(self, loaded_player):
        """After seeking to just before a beat, the click should land correctly."""
        loaded_player.set_beat_times([0.0, 0.5, 1.0], [])
        loaded_player.set_beat_sync_enabled(True)
        loaded_player.set_metronome_enabled(True)
        loaded_player.set_metronome_volume(1.0)
        loaded_player._is_playing = True

        # Seek to 10 frames before beat at 0.5s (frame 22050).
        loaded_player.seek(22040.0 / 44100.0)
        outdata = np.zeros((512, 2), dtype=np.float32)
        loaded_player._audio_callback(outdata, 512, {}, sd.CallbackFlags())

        # Click at offset ~10.
        assert np.max(np.abs(outdata[10:60])) > 0.01

    def test_beat_sync_cleared_on_load(self, loaded_player, tmp_path):
        """Loading new stems should disable beat sync."""
        import soundfile as sf
        loaded_player.set_beat_sync_enabled(True)
        data = np.zeros((44100, 2), dtype=np.float32)
        path = tmp_path / "new_stem.wav"
        sf.write(str(path), data, 44100)
        loaded_player.load_stems({"vocals": str(path)})
        assert loaded_player.beat_sync_enabled is False
        assert len(loaded_player._beat_frames) == 0

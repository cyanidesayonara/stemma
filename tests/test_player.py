"""Tests for the MultiTrackPlayer."""

import os
from unittest.mock import patch

import numpy as np
import pytest
import sounddevice as sd
import soundfile as sf

from src.player import MultiTrackPlayer, next_take_number


@pytest.fixture
def mock_stems(tmp_path):
    """Create fake stereo WAV stems for testing."""
    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    paths = {}
    
    # 1 second of audio at 44.1kHz
    sr = 44100
    frames = sr * 1
    
    # Create distinguishable dummy audio for each stem
    signals = {
        "vocals": np.ones((frames, 2), dtype=np.float32) * 0.1,
        "drums": np.ones((frames, 2), dtype=np.float32) * 0.2,
        "bass": np.ones((frames, 2), dtype=np.float32) * 0.3,
    }
    
    for name, data in signals.items():
        path = stems_dir / f"{name}.wav"
        sf.write(str(path), data, sr)
        paths[name] = str(path)
        
    return paths


class TestMultiTrackPlayerLoad:
    
    def test_load_stems_sets_properties(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        
        assert player._sample_rate == 44100
        assert player.total_seconds == 1.0
        assert player._total_frames == 44100
        assert len(player._stems) == 3
        # Ensure it loaded as stereo
        assert player._stems["vocals"].shape == (44100, 2)

    def test_load_mono_to_stereo(self, tmp_path):
        mono_path = tmp_path / "mono.wav"
        # 1 sec mono
        sf.write(str(mono_path), np.ones((44100,), dtype=np.float32), 44100)
        
        player = MultiTrackPlayer()
        player.load_stems({"mono_stem": str(mono_path)})
        
        # It should automatically duplicate to shape (44100, 2)
        assert player._stems["mono_stem"].shape == (44100, 2)


class TestMultiTrackPlayerPositions:

    def test_seek_updates_current_frame(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        
        player.seek(0.5)
        assert player._current_frame == 22050

    def test_seek_out_of_bounds(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        
        player.seek(5.0) # Past 1.0 second EOF
        assert player._current_frame == 44100
        
        player.seek(-1.0) # Before start
        assert player._current_frame == 0


class TestPlaybackFailedSignal:

    def test_play_emits_playback_failed_on_portaudio_error(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        messages: list[str] = []
        player.playback_failed.connect(lambda m: messages.append(m))
        with patch(
            "src.player.sd.OutputStream",
            side_effect=sd.PortAudioError("no device"),
        ):
            player.play()
        assert not player.is_playing
        assert len(messages) == 1
        assert "Preferences" in messages[0]


class TestMultiTrackPlayerMixing:
    """Test the math inside the audio callback without playing sound."""
    
    def test_audio_callback_sums_stems(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        
        player._is_playing = True
        outdata = np.zeros((100, 2), dtype=np.float32)
        
        # Manually trigger the callback
        player._audio_callback(outdata, 100, {}, sd.CallbackFlags())
        
        # V = 0.1, D = 0.2, B = 0.3. Sum should be 0.6.
        assert np.allclose(outdata, 0.6, atol=1e-4)
        assert player._current_frame == 100

    def test_audio_callback_respects_mute(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._is_playing = True
        
        player.set_mute("vocals", True)
        
        outdata = np.zeros((100, 2), dtype=np.float32)
        player._audio_callback(outdata, 100, {}, sd.CallbackFlags())
        
        # sum of D (0.2) + B (0.3) = 0.5
        assert np.allclose(outdata, 0.5, atol=1e-4)

    def test_audio_callback_respects_solo(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._is_playing = True
        
        player.set_solo("drums", True)
        
        outdata = np.zeros((100, 2), dtype=np.float32)
        player._audio_callback(outdata, 100, {}, sd.CallbackFlags())
        
        # Only D = 0.2 should play
        assert np.allclose(outdata, 0.2, atol=1e-4)
        
    def test_solo_overrides_mute(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._is_playing = True
        
        # If a stem is both muted AND soloed, solo takes precedence in most DAWs
        # or at least solo shuts down everything else. In our logic:
        # "If any stem is soloed, play soloed stems (ignoring mute)"
        player.set_mute("vocals", True)
        player.set_solo("vocals", True)
        
        outdata = np.zeros((100, 2), dtype=np.float32)
        player._audio_callback(outdata, 100, {}, sd.CallbackFlags())
        
        # Only V = 0.1 should play
        assert np.allclose(outdata, 0.1, atol=1e-4)

    def test_audio_callback_eof_raises_stop(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._is_playing = True
        
        # Seek right near the end (10 frames left)
        player.seek(1.0 - (10 / 44100))
        outdata = np.zeros((100, 2), dtype=np.float32)
        
        # Requesting 100 frames, but only 10 available
        with pytest.raises(sd.CallbackStop):
            player._audio_callback(outdata, 100, {}, sd.CallbackFlags())
            
        # The first 10 frames should be summed (0.6), the rest zeros
        assert np.allclose(outdata[:10], 0.6, atol=1e-4)
        assert np.allclose(outdata[10:], 0.0)
        assert not player.is_playing # Flag should be flipped to false


class TestMultiTrackPlayerVolume:
    """Test per-stem volume control."""

    def test_default_volume_is_1(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)

        assert player.get_volume("vocals") == 1.0
        assert player.get_volume("drums") == 1.0

    def test_set_volume_changes_gain(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)

        player.set_volume("vocals", 0.5)
        assert player.get_volume("vocals") == 0.5

    def test_set_volume_clamps_range(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)

        player.set_volume("vocals", -0.5)
        assert player.get_volume("vocals") == 0.0

        player.set_volume("vocals", 2.5)
        assert player.get_volume("vocals") == 2.0

    def test_volume_applied_in_callback(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._is_playing = True

        player.set_volume("vocals", 0.5)  # 0.1 * 0.5 = 0.05
        player.set_volume("drums", 0.0)   # 0.2 * 0.0 = 0.0
        # bass stays at 1.0: 0.3 * 1.0 = 0.3

        outdata = np.zeros((100, 2), dtype=np.float32)
        player._audio_callback(outdata, 100, {}, sd.CallbackFlags())

        # 0.05 + 0.0 + 0.3 = 0.35
        assert np.allclose(outdata, 0.35, atol=1e-4)

    def test_volume_with_mute(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._is_playing = True

        player.set_volume("vocals", 0.5)
        player.set_mute("vocals", True)

        outdata = np.zeros((100, 2), dtype=np.float32)
        player._audio_callback(outdata, 100, {}, sd.CallbackFlags())

        # Muted vocals: 0.0. Drums 0.2 + Bass 0.3 = 0.5
        assert np.allclose(outdata, 0.5, atol=1e-4)

    def test_volume_with_solo(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._is_playing = True

        player.set_volume("drums", 0.5)
        player.set_solo("drums", True)

        outdata = np.zeros((100, 2), dtype=np.float32)
        player._audio_callback(outdata, 100, {}, sd.CallbackFlags())

        # Only drums at half volume: 0.2 * 0.5 = 0.1
        assert np.allclose(outdata, 0.1, atol=1e-4)

    def test_load_stems_resets_volumes(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.set_volume("vocals", 0.5)

        # Reload — volumes should reset to 1.0
        player.load_stems(mock_stems)
        assert player.get_volume("vocals") == 1.0


class TestABLoop:
    """Test A-B loop functionality."""

    def test_loop_disabled_by_default(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        assert player.loop_a is None
        assert player.loop_b is None
        assert not player.looping

    def test_set_loop_points(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)

        player.set_loop_a(0.2)
        player.set_loop_b(0.8)
        assert player.loop_a == pytest.approx(0.2, abs=0.001)
        assert player.loop_b == pytest.approx(0.8, abs=0.001)

    def test_set_loop_a_after_b_swaps(self, mock_stems):
        """If A is set after B and A > B, they should swap."""
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)

        player.set_loop_b(0.3)
        player.set_loop_a(0.7)
        assert player.loop_a == pytest.approx(0.3, abs=0.001)
        assert player.loop_b == pytest.approx(0.7, abs=0.001)

    def test_clear_loop(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)

        player.set_loop_a(0.2)
        player.set_loop_b(0.8)
        player.set_looping(True)
        player.clear_loop()

        assert player.loop_a is None
        assert player.loop_b is None
        assert not player.looping

    def test_callback_wraps_at_loop_b(self, mock_stems):
        """When looping, playback should wrap from B back to A."""
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._is_playing = True

        # Set loop region: 0.0s to 0.5s (22050 frames)
        player.set_loop_a(0.0)
        player.set_loop_b(0.5)
        player.set_looping(True)

        # Seek near end of loop region (100 frames before loop_b)
        player.seek(0.5 - 100 / 44100)

        outdata = np.zeros((200, 2), dtype=np.float32)
        # Should NOT raise CallbackStop — loop wraps instead of EOF
        player._audio_callback(outdata, 200, {}, sd.CallbackFlags())

        # Should have wrapped back into the loop region
        assert player._current_frame < 22050  # Still within loop

    def test_callback_no_wrap_when_not_looping(self, mock_stems):
        """Without looping enabled, playback at EOF raises CallbackStop."""
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._is_playing = True

        player.set_loop_a(0.0)
        player.set_loop_b(0.5)
        # looping NOT enabled

        # Seek near end of track
        player.seek(1.0 - 10 / 44100)

        outdata = np.zeros((100, 2), dtype=np.float32)
        with pytest.raises(sd.CallbackStop):
            player._audio_callback(outdata, 100, {}, sd.CallbackFlags())

    def test_load_stems_clears_loop(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)

        player.set_loop_a(0.2)
        player.set_loop_b(0.8)
        player.set_looping(True)

        player.load_stems(mock_stems)
        assert player.loop_a is None
        assert player.loop_b is None
        assert not player.looping

    def test_zero_width_loop_does_not_deadlock(self, mock_stems):
        """A == B must not cause an infinite loop in the callback."""
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._is_playing = True

        # Set A and B to the exact same position (zero-width loop)
        player.set_loop_a(0.5)
        player.set_loop_b(0.5)
        player.set_looping(True)

        # Seek to that position
        player.seek(0.5)

        outdata = np.zeros((200, 2), dtype=np.float32)
        # Should complete without hanging -- looping is effectively ignored
        player._audio_callback(outdata, 200, {}, sd.CallbackFlags())

        # Playback should have advanced past the zero-width "loop"
        assert player._current_frame > 22050

    def test_loop_fills_output_buffer_correctly(self, mock_stems):
        """Even when wrapping, the entire output buffer should be filled."""
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._is_playing = True

        # Small loop: 0.0s to 0.01s (~441 frames)
        player.set_loop_a(0.0)
        player.set_loop_b(441 / 44100)
        player.set_looping(True)

        # Seek to 400 frames in — only 41 frames before wrap
        player._current_frame = 400

        outdata = np.zeros((200, 2), dtype=np.float32)
        player._audio_callback(outdata, 200, {}, sd.CallbackFlags())

        # Entire buffer should be non-zero (filled with stem data)
        assert np.all(outdata != 0.0)

    def test_stop_seeks_to_loop_a_when_looping(self, mock_stems):
        """Stop should place the playhead at A, not at the start of the song."""
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.set_loop_a(0.2)
        player.set_loop_b(0.8)
        player.set_looping(True)
        player.seek(0.5)
        player._is_playing = True

        player.stop()

        assert player.current_seconds == pytest.approx(0.2, abs=0.002)
        assert not player.is_playing

    def test_stop_seeks_to_zero_without_loop(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.seek(0.5)
        player._is_playing = True
        player.stop()
        assert player.current_seconds == pytest.approx(0.0, abs=0.001)

    def test_seek_clamps_before_loop_a_when_looping(self, mock_stems):
        """Scrubbing before the loop region snaps to loop A."""
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.set_loop_a(0.3)
        player.set_loop_b(0.7)
        player.set_looping(True)
        player.seek(0.1)
        assert player.current_seconds == pytest.approx(0.3, abs=0.002)

    def test_seek_clamps_at_or_after_loop_b_when_looping(self, mock_stems):
        """Scrubbing at or past B snaps to loop A."""
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.set_loop_a(0.3)
        player.set_loop_b(0.7)
        player.set_looping(True)
        player.seek(0.75)
        assert player.current_seconds == pytest.approx(0.3, abs=0.002)

    def test_seek_within_loop_unchanged_when_looping(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.set_loop_a(0.2)
        player.set_loop_b(0.8)
        player.set_looping(True)
        player.seek(0.5)
        assert player.current_seconds == pytest.approx(0.5, abs=0.002)


class TestRecordingArm:
    """Test recording arm/disarm logic."""

    def test_arm_sets_flag(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.arm_recording(True)
        assert player.recording_armed

    def test_disarm_clears_flag(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.arm_recording(True)
        player.arm_recording(False)
        assert not player.recording_armed

    def test_arm_rejected_without_stems(self):
        player = MultiTrackPlayer()
        player.arm_recording(True)
        assert not player.recording_armed

    def test_arm_rejected_at_non_1x_speed(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._playback_speed = 0.75
        player.arm_recording(True)
        assert not player.recording_armed

    def test_load_stems_clears_armed(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.arm_recording(True)
        player.load_stems(mock_stems)
        assert not player.recording_armed


class TestRecordingBuffer:
    """Test recording buffer allocation and capture."""

    def test_buffer_allocated_with_correct_shape(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._allocate_recording_buffer()
        assert player._recording_buffer is not None
        assert player._recording_buffer.shape == (44100, 2)
        assert np.all(player._recording_buffer == 0.0)

    def test_duplex_callback_captures_input(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._allocate_recording_buffer()
        player._recording = True
        player._is_playing = True

        indata = np.full((100, 2), 0.42, dtype=np.float32)
        outdata = np.zeros((100, 2), dtype=np.float32)

        player._full_duplex_callback(
            indata, outdata, 100, {}, sd.CallbackFlags()
        )

        assert np.allclose(
            player._recording_buffer[:100], 0.42, atol=1e-6
        )
        assert player._indata_capture is None

    def test_duplex_callback_skips_input_during_count_in(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._allocate_recording_buffer()
        player._recording = True
        player._is_playing = True
        player._count_in_remaining = 44100
        player._metronome_bpm = 120.0

        indata = np.full((100, 2), 0.99, dtype=np.float32)
        outdata = np.zeros((100, 2), dtype=np.float32)

        player._full_duplex_callback(
            indata, outdata, 100, {}, sd.CallbackFlags()
        )

        assert np.allclose(player._recording_buffer[:100], 0.0)

    def test_mono_input_duplicated_to_stereo(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._allocate_recording_buffer()
        player._recording = True
        player._is_playing = True

        indata = np.full((100, 1), 0.33, dtype=np.float32)
        outdata = np.zeros((100, 2), dtype=np.float32)

        player._full_duplex_callback(
            indata, outdata, 100, {}, sd.CallbackFlags()
        )

        assert np.allclose(
            player._recording_buffer[:100, 0], 0.33, atol=1e-6
        )
        assert np.allclose(
            player._recording_buffer[:100, 1], 0.33, atol=1e-6
        )


class TestRecordingLoopCapture:
    """Test that recording follows loop wraps correctly."""

    def test_recording_overwrites_same_region_on_loop(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._allocate_recording_buffer()
        player._recording = True
        player._is_playing = True

        player.set_loop_a(0.0)
        player.set_loop_b(200 / 44100)
        player.set_looping(True)

        indata_pass1 = np.full((200, 2), 0.1, dtype=np.float32)
        outdata = np.zeros((200, 2), dtype=np.float32)
        player._full_duplex_callback(
            indata_pass1, outdata, 200, {}, sd.CallbackFlags()
        )
        assert np.allclose(
            player._recording_buffer[:200], 0.1, atol=1e-6
        )

        indata_pass2 = np.full((200, 2), 0.9, dtype=np.float32)
        outdata2 = np.zeros((200, 2), dtype=np.float32)
        player._full_duplex_callback(
            indata_pass2, outdata2, 200, {}, sd.CallbackFlags()
        )
        assert np.allclose(
            player._recording_buffer[:200], 0.9, atol=1e-6
        )


class TestRecordingSave:
    """Test writing recording to disk."""

    def test_save_creates_wav(self, mock_stems, tmp_path):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._recording_buffer = np.full(
            (44100, 2), 0.5, dtype=np.float32
        )

        song_dir = str(tmp_path / "song")
        os.makedirs(song_dir)
        path = player.save_recording(song_dir)

        assert path is not None
        assert os.path.isfile(path)
        assert "recording_take1.wav" in path

        data, sr = sf.read(path, dtype="float32")
        assert sr == 44100
        assert data.shape == (44100, 2)
        assert np.allclose(data, 0.5, atol=1e-4)

    def test_take_numbering_increments(self, tmp_path):
        song_dir = str(tmp_path / "song")
        os.makedirs(song_dir)
        sf.write(
            os.path.join(song_dir, "recording_take1.wav"),
            np.zeros((100, 2), dtype=np.float32),
            44100,
        )
        assert next_take_number(song_dir) == 2

    def test_save_returns_none_without_buffer(self, mock_stems, tmp_path):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        assert player.save_recording(str(tmp_path)) is None

    def test_latency_offset_shifts_buffer(self, mock_stems, tmp_path):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)

        buf = np.zeros((44100, 2), dtype=np.float32)
        buf[441:541] = 0.8
        player._recording_buffer = buf
        player.set_latency_offset_ms(10.0)

        song_dir = str(tmp_path / "song")
        os.makedirs(song_dir)
        path = player.save_recording(song_dir)

        data, _ = sf.read(path, dtype="float32")
        peak_pos = np.argmax(np.abs(data[:, 0]))
        assert peak_pos < 441

    def test_next_take_number_empty_dir(self, tmp_path):
        assert next_take_number(str(tmp_path)) == 1

    def test_next_take_number_multiple(self, tmp_path):
        for n in (1, 2, 5):
            sf.write(
                os.path.join(str(tmp_path), f"recording_take{n}.wav"),
                np.zeros((10, 2), dtype=np.float32),
                44100,
            )
        assert next_take_number(str(tmp_path)) == 6


class TestRecordingInputDevice:
    """Test input device configuration."""

    def test_set_input_device(self):
        player = MultiTrackPlayer()
        player.set_input_device(3)
        assert player._input_device == 3

    def test_set_input_device_none(self):
        player = MultiTrackPlayer()
        player.set_input_device(None)
        assert player._input_device is None

    def test_play_with_recording_creates_duplex_stream(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.arm_recording(True)

        with patch("src.player.sd.Stream") as mock_stream, \
             patch("src.player.sd.default") as mock_default, \
             patch("src.player.sd.query_devices") as mock_qd:
            mock_default.device = (0, 1)
            mock_qd.return_value = {"max_input_channels": 1}
            mock_instance = mock_stream.return_value
            mock_instance.start = lambda: None
            player.play()

        mock_stream.assert_called_once()
        call_kwargs = mock_stream.call_args[1]
        assert call_kwargs["channels"] == (1, 2)
        assert player._recording is True

    def test_play_without_recording_creates_output_stream(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)

        with patch("src.player.sd.OutputStream") as mock_stream:
            mock_instance = mock_stream.return_value
            mock_instance.start = lambda: None
            player.play()

        mock_stream.assert_called_once()
        assert player._recording is False


class TestMonoInputDevice:
    """Test mono microphone channel handling in duplex mode."""

    def test_mono_device_uses_1_input_channel(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.arm_recording(True)
        player.set_input_device(5)

        with patch("src.player.sd.Stream") as mock_stream, \
             patch("src.player.sd.default") as mock_default, \
             patch("src.player.sd.query_devices") as mock_qd:
            mock_default.device = (0, 1)
            mock_qd.return_value = {"max_input_channels": 1}
            mock_instance = mock_stream.return_value
            mock_instance.start = lambda: None
            player.play()

        call_kwargs = mock_stream.call_args[1]
        assert call_kwargs["channels"] == (1, 2)
        assert call_kwargs["device"][0] == 5

    def test_stereo_device_uses_2_input_channels(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.arm_recording(True)
        player.set_input_device(7)

        with patch("src.player.sd.Stream") as mock_stream, \
             patch("src.player.sd.default") as mock_default, \
             patch("src.player.sd.query_devices") as mock_qd:
            mock_default.device = (0, 1)
            mock_qd.return_value = {"max_input_channels": 4}
            mock_instance = mock_stream.return_value
            mock_instance.start = lambda: None
            player.play()

        call_kwargs = mock_stream.call_args[1]
        assert call_kwargs["channels"] == (2, 2)

    def test_query_devices_failure_defaults_to_1_channel(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.arm_recording(True)
        player.set_input_device(99)

        with patch("src.player.sd.Stream") as mock_stream, \
             patch("src.player.sd.default") as mock_default, \
             patch("src.player.sd.query_devices") as mock_qd:
            mock_default.device = (0, 1)
            mock_qd.side_effect = sd.PortAudioError("bad device")
            mock_instance = mock_stream.return_value
            mock_instance.start = lambda: None
            player.play()

        call_kwargs = mock_stream.call_args[1]
        assert call_kwargs["channels"] == (1, 2)


class TestStopOnlyFinalization:
    """Test that stop saves recordings but pause does not."""

    def test_pause_keeps_recording_buffer(self, mock_stems, tmp_path):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._recording_buffer = np.full(
            (44100, 2), 0.5, dtype=np.float32
        )
        player._recording = True
        player._is_playing = True
        player.set_recording_song_dir(str(tmp_path))

        saved = []
        player.recording_saved.connect(lambda p: saved.append(p))

        player.pause()

        assert len(saved) == 0
        assert player._recording_buffer is not None

    def test_stop_saves_recording(self, mock_stems, tmp_path):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._recording_buffer = np.full(
            (44100, 2), 0.5, dtype=np.float32
        )
        player._recording = True
        player._is_playing = True
        player.set_recording_song_dir(str(tmp_path))

        saved = []
        player.recording_saved.connect(lambda p: saved.append(p))

        player.stop()

        assert len(saved) == 1
        assert os.path.isfile(saved[0])
        assert player._recording_buffer is None

    def test_stop_without_buffer_is_safe(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player._is_playing = True
        player.stop()
        assert not player.is_playing


class TestRecordingStemMethods:
    """Test add/remove recording stem encapsulation."""

    def test_add_recording_stem(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        old_total = player._total_frames

        longer_data = np.ones((88200, 2), dtype=np.float32) * 0.1
        player.add_recording_stem("recording_take1", longer_data)

        assert "recording_take1" in player._stems
        assert "recording_take1" in player._original_stems
        assert player._total_frames == 88200

    def test_add_mono_recording_stem(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)

        mono_data = np.ones((44100, 1), dtype=np.float32) * 0.3
        player.add_recording_stem("recording_take1", mono_data)

        assert player._stems["recording_take1"].shape == (44100, 2)

    def test_remove_recording_stem_recalculates_total(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)

        longer_data = np.ones((88200, 2), dtype=np.float32)
        player.add_recording_stem("recording_take1", longer_data)
        assert player._total_frames == 88200

        player.remove_recording_stem("recording_take1")
        assert player._total_frames == 44100
        assert "recording_take1" not in player._stems

    def test_remove_clamps_current_frame(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)

        longer_data = np.ones((88200, 2), dtype=np.float32)
        player.add_recording_stem("recording_take1", longer_data)
        player._current_frame = 80000

        player.remove_recording_stem("recording_take1")
        assert player._current_frame <= player._total_frames

    def test_remove_nonexistent_stem_is_safe(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.remove_recording_stem("does_not_exist")
        assert player._total_frames == 44100


class TestCountInAtAnyPosition:
    """Test that count-in fires at any play position, not just boundaries."""

    def test_count_in_at_mid_song_position(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.set_count_in_enabled(True)
        player.set_count_in_beats(4)
        player.set_metronome_bpm(120.0)

        player.seek(0.5)
        assert player._current_frame > 0

        with patch("src.player.sd.OutputStream") as mock_stream:
            mock_instance = mock_stream.return_value
            mock_instance.start = lambda: None
            player.play()

        assert player._count_in_remaining > 0

    def test_no_count_in_on_device_switch(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.set_count_in_enabled(True)
        player.set_count_in_beats(4)
        player.set_metronome_bpm(120.0)

        with patch("src.player.sd.OutputStream") as mock_stream:
            mock_instance = mock_stream.return_value
            mock_instance.start = lambda: None
            mock_instance.stop = lambda: None
            mock_instance.close = lambda: None
            player.play()

        player._count_in_remaining = 0

        with patch("src.player.sd.OutputStream") as mock_stream:
            mock_instance = mock_stream.return_value
            mock_instance.start = lambda: None
            player.set_output_device(None)

        assert player._count_in_remaining == 0


class TestNudgeStem:
    """Tests for post-recording track nudge."""

    def test_nudge_positive_shifts_later(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        original = player._stems["vocals"].copy()

        player.nudge_stem("vocals", 10.0)  # +10ms

        offset_frames = int(10.0 / 1000.0 * 44100)
        np.testing.assert_array_equal(
            player._stems["vocals"][:offset_frames],
            np.zeros((offset_frames, 2), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            player._stems["vocals"][offset_frames:],
            original[:-offset_frames],
        )

    def test_nudge_negative_shifts_earlier(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        original = player._stems["vocals"].copy()

        player.nudge_stem("vocals", -10.0)  # -10ms

        offset_frames = int(10.0 / 1000.0 * 44100)
        np.testing.assert_array_equal(
            player._stems["vocals"][:-offset_frames],
            original[offset_frames:],
        )
        np.testing.assert_array_equal(
            player._stems["vocals"][-offset_frames:],
            np.zeros((offset_frames, 2), dtype=np.float32),
        )

    def test_nudge_zero_is_noop(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        original = player._stems["vocals"].copy()

        player.nudge_stem("vocals", 0.0)

        np.testing.assert_array_equal(player._stems["vocals"], original)

    def test_nudge_clamped_to_200ms(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)

        player.nudge_stem("vocals", 500.0)
        assert player.get_nudge_ms("vocals") == 200.0

        player.nudge_stem("vocals", -500.0)
        assert player.get_nudge_ms("vocals") == -200.0

    def test_nudge_updates_original_stems(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)

        player.nudge_stem("vocals", 10.0)

        np.testing.assert_array_equal(
            player._stems["vocals"], player._original_stems["vocals"]
        )

    def test_nudge_nonexistent_stem_is_noop(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)

        player.nudge_stem("nonexistent", 10.0)
        assert player.get_nudge_ms("nonexistent") == 0.0

    def test_get_nudge_default_zero(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        assert player.get_nudge_ms("vocals") == 0.0

    def test_nudge_cleared_on_load(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.nudge_stem("vocals", 10.0)

        player.load_stems(mock_stems)
        assert player.get_nudge_ms("vocals") == 0.0

    def test_nudge_incremental(self, mock_stems):
        """Nudging from 50ms to 100ms applies only the +50ms delta."""
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)

        player.nudge_stem("vocals", 50.0)
        after_50 = player._stems["vocals"].copy()

        player.nudge_stem("vocals", 100.0)
        after_100 = player._stems["vocals"].copy()

        delta_frames = int(50.0 / 1000.0 * 44100)
        total_zeroed = int(100.0 / 1000.0 * 44100)
        np.testing.assert_array_equal(
            after_100[:total_zeroed],
            np.zeros((total_zeroed, 2), dtype=np.float32),
        )
        assert player.get_nudge_ms("vocals") == 100.0

    def test_remove_recording_clears_nudge(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        rec = np.zeros((44100, 2), dtype=np.float32)
        player.add_recording_stem("take1", rec)
        player.nudge_stem("take1", 10.0)

        player.remove_recording_stem("take1")
        assert player.get_nudge_ms("take1") == 0.0


# ---------------------------------------------------------------------------
# Chord sequence
# ---------------------------------------------------------------------------

class TestChordSequence:
    def test_default_empty(self):
        player = MultiTrackPlayer()
        assert player.chord_sequence == []
        assert player.chord_at(0) == ""

    def test_set_chord_sequence(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        chords = [(0.0, "Am"), (2.3, "G"), (3.8, "C")]
        player.set_chord_sequence(chords)
        assert len(player.chord_sequence) == 3
        assert player.chord_sequence[0] == (0.0, "Am")

    def test_chord_at_returns_correct_chord(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        chords = [(0.0, "Am"), (0.5, "G")]
        player.set_chord_sequence(chords)
        sr = player.sample_rate
        # Frame at t=0.25s should be in "Am" region.
        assert player.chord_at(int(0.25 * sr)) == "Am"
        # Frame at t=0.75s should be in "G" region.
        assert player.chord_at(int(0.75 * sr)) == "G"

    def test_chord_at_before_first_onset(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        chords = [(1.0, "Am")]
        player.set_chord_sequence(chords)
        # Before the first chord onset, should return empty.
        assert player.chord_at(0) == ""

    def test_chord_at_empty_sequence(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.set_chord_sequence([])
        assert player.chord_at(22050) == ""

    def test_load_stems_clears_chords(self, mock_stems):
        player = MultiTrackPlayer()
        player.load_stems(mock_stems)
        player.set_chord_sequence([(0.0, "Am")])
        assert len(player.chord_sequence) == 1
        player.load_stems(mock_stems)
        assert player.chord_sequence == []

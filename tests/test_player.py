"""Tests for the MultiTrackPlayer."""

import os
from unittest.mock import patch

import numpy as np
import pytest
import sounddevice as sd
import soundfile as sf

from src.player import MultiTrackPlayer


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

"""Tests for the MultiTrackPlayer."""

import os

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

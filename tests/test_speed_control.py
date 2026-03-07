"""Tests for playback speed control (pitch-preserving time-stretch)."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from PySide6.QtWidgets import QApplication

from src.player import MultiTrackPlayer, SpeedWorker


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
# SpeedWorker
# -----------------------------------------------------------------------

class TestSpeedWorker:
    """SpeedWorker stretches stems via librosa in a background thread."""

    def test_stretch_produces_different_frame_count(self, app):
        """Stretching at 0.5x should roughly double the frame count."""
        stems = {"vocals": np.random.randn(4410, 2).astype(np.float32)}
        worker = SpeedWorker(stems, 0.5)
        results = {}
        worker.completed.connect(lambda d: results.update(d))
        worker.run()  # Run synchronously for testing.
        assert "vocals" in results
        # At 0.5x rate, output should be ~2x the input frames.
        assert results["vocals"].shape[0] > 4410 * 1.5
        assert results["vocals"].shape[1] == 2

    def test_stretch_at_1x_returns_same_length(self, app):
        """Stretching at 1.0x should return approximately the same length."""
        stems = {"vocals": np.random.randn(4410, 2).astype(np.float32)}
        worker = SpeedWorker(stems, 1.0)
        results = {}
        worker.completed.connect(lambda d: results.update(d))
        worker.run()
        # Should be very close to original length.
        assert abs(results["vocals"].shape[0] - 4410) < 100

    def test_stretch_at_2x_halves_frames(self, app):
        """Stretching at 2.0x should roughly halve the frame count."""
        stems = {"vocals": np.random.randn(8820, 2).astype(np.float32)}
        worker = SpeedWorker(stems, 2.0)
        results = {}
        worker.completed.connect(lambda d: results.update(d))
        worker.run()
        assert results["vocals"].shape[0] < 8820 * 0.75

    def test_stretch_multiple_stems(self, app):
        """All stems are stretched."""
        stems = {
            "vocals": np.random.randn(4410, 2).astype(np.float32),
            "drums": np.random.randn(4410, 2).astype(np.float32),
        }
        worker = SpeedWorker(stems, 0.75)
        results = {}
        worker.completed.connect(lambda d: results.update(d))
        worker.run()
        assert "vocals" in results
        assert "drums" in results

    def test_progress_emitted(self, app):
        """Progress signal emitted for each stem."""
        stems = {
            "vocals": np.random.randn(4410, 2).astype(np.float32),
            "drums": np.random.randn(4410, 2).astype(np.float32),
        }
        worker = SpeedWorker(stems, 0.75)
        progress = []
        worker.progress.connect(lambda cur, tot: progress.append((cur, tot)))
        worker.run()
        assert len(progress) == 2
        assert progress[-1] == (2, 2)


# -----------------------------------------------------------------------
# MultiTrackPlayer speed API
# -----------------------------------------------------------------------

class TestPlayerSpeedAPI:
    """Player exposes speed property and set_speed method."""

    def test_default_speed_is_1(self, player):
        assert player.speed == 1.0

    def test_set_speed_clamps_low(self, player):
        """Speed below 0.5 is clamped."""
        player.set_speed(0.1)
        assert player.speed == 0.5

    def test_set_speed_clamps_high(self, player):
        """Speed above 2.0 is clamped."""
        player.set_speed(5.0)
        assert player.speed == 2.0

    def test_set_speed_without_stems_is_noop(self, player):
        """Changing speed with no stems loaded does not crash."""
        player.set_speed(0.75)
        assert player.speed == 0.75


# -----------------------------------------------------------------------
# Frame index mapping on speed change
# -----------------------------------------------------------------------

class TestFrameMapping:
    """Frame indices are adjusted proportionally when stems are swapped."""

    def _load_fake_stems(self, player):
        """Load fake stems directly for testing without WAV files."""
        sr = 44100
        frames = sr * 2  # 2 seconds
        player._stems = {
            "vocals": np.zeros((frames, 2), dtype=np.float32),
        }
        player._original_stems = dict(player._stems)
        player._sample_rate = sr
        player._total_frames = frames
        player._current_frame = frames // 2  # Halfway
        player._loop_a_frame = frames // 4
        player._loop_b_frame = (frames * 3) // 4

    def test_apply_stretched_adjusts_current_frame(self, player):
        """Current frame is proportionally adjusted to new total."""
        self._load_fake_stems(player)
        old_total = player._total_frames
        old_pos = player._current_frame

        # Simulate stretched stems at 0.5x (double length).
        new_frames = old_total * 2
        stretched = {
            "vocals": np.zeros((new_frames, 2), dtype=np.float32),
        }
        player._apply_stretched_stems(stretched, 0.5)

        assert player._total_frames == new_frames
        expected_pos = int(old_pos / old_total * new_frames)
        assert player._current_frame == expected_pos

    def test_apply_stretched_adjusts_loop_points(self, player):
        """Loop A and B frames are proportionally adjusted."""
        self._load_fake_stems(player)
        old_total = player._total_frames
        old_a = player._loop_a_frame
        old_b = player._loop_b_frame

        new_frames = old_total * 2
        stretched = {
            "vocals": np.zeros((new_frames, 2), dtype=np.float32),
        }
        player._apply_stretched_stems(stretched, 0.5)

        expected_a = int(old_a / old_total * new_frames)
        expected_b = int(old_b / old_total * new_frames)
        assert player._loop_a_frame == expected_a
        assert player._loop_b_frame == expected_b

    def test_restore_originals_at_1x(self, player):
        """Setting speed to 1.0 restores original stems."""
        self._load_fake_stems(player)
        original_frames = player._total_frames

        # Apply stretched stems.
        stretched = {
            "vocals": np.zeros((original_frames * 2, 2), dtype=np.float32),
        }
        player._apply_stretched_stems(stretched, 0.5)
        assert player._total_frames == original_frames * 2

        # Restore to 1.0x.
        player._apply_stretched_stems(dict(player._original_stems), 1.0)
        assert player._total_frames == original_frames

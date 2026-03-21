"""Tests for waveform peak computation."""

import numpy as np
import pytest

from src.waveform import compute_peaks


class TestComputePeaks:
    """Tests for the compute_peaks function."""

    def _make_stems(self, values_per_stem: dict[str, float],
                    num_frames: int = 1000) -> dict[str, np.ndarray]:
        """Create stereo stem arrays filled with constant values."""
        stems = {}
        for name, value in values_per_stem.items():
            stems[name] = np.full((num_frames, 2), value, dtype=np.float32)
        return stems

    def test_basic_peaks(self):
        """Peak values reflect the audio amplitude."""
        stems = self._make_stems({"vocals": 0.5}, num_frames=100)
        peaks = compute_peaks(stems, muted=set(), soloed=set(),
                              volumes={}, num_bins=10)
        assert peaks.shape == (10,)
        np.testing.assert_allclose(peaks, 0.5, atol=1e-6)

    def test_respects_mute(self):
        """Muted stems are excluded from peak computation."""
        stems = self._make_stems({"vocals": 0.8, "drums": 0.3},
                                 num_frames=100)
        peaks = compute_peaks(stems, muted={"vocals"}, soloed=set(),
                              volumes={}, num_bins=10)
        np.testing.assert_allclose(peaks, 0.3, atol=1e-6)

    def test_respects_solo(self):
        """Only soloed stems appear in peaks when any stem is soloed."""
        stems = self._make_stems({"vocals": 0.8, "drums": 0.3, "bass": 0.5},
                                 num_frames=100)
        peaks = compute_peaks(stems, muted=set(), soloed={"drums"},
                              volumes={}, num_bins=10)
        np.testing.assert_allclose(peaks, 0.3, atol=1e-6)

    def test_respects_volume(self):
        """Volume scaling is applied to peaks."""
        stems = self._make_stems({"vocals": 1.0}, num_frames=100)
        peaks = compute_peaks(stems, muted=set(), soloed=set(),
                              volumes={"vocals": 0.5}, num_bins=10)
        np.testing.assert_allclose(peaks, 0.5, atol=1e-6)

    def test_no_active_stems_returns_zeros(self):
        """All stems muted produces zero peaks."""
        stems = self._make_stems({"vocals": 0.8, "drums": 0.3},
                                 num_frames=100)
        peaks = compute_peaks(stems, muted={"vocals", "drums"}, soloed=set(),
                              volumes={}, num_bins=10)
        np.testing.assert_allclose(peaks, 0.0)

    def test_single_bin(self):
        """Single bin returns the global peak."""
        data = np.zeros((100, 2), dtype=np.float32)
        data[50, 0] = 0.9  # spike at frame 50
        stems = {"vocals": data}
        peaks = compute_peaks(stems, muted=set(), soloed=set(),
                              volumes={}, num_bins=1)
        assert peaks.shape == (1,)
        assert peaks[0] == pytest.approx(0.9, abs=1e-6)

    def test_bins_greater_than_frames(self):
        """Gracefully handles more bins than frames."""
        stems = self._make_stems({"vocals": 0.5}, num_frames=5)
        peaks = compute_peaks(stems, muted=set(), soloed=set(),
                              volumes={}, num_bins=20)
        assert peaks.shape == (20,)
        # Bins that map to actual frames should have values; extras should be 0
        assert np.any(peaks > 0)

    def test_empty_stems_dict(self):
        """Empty stems dict returns zeros."""
        peaks = compute_peaks({}, muted=set(), soloed=set(),
                              volumes={}, num_bins=10)
        assert peaks.shape == (10,)
        np.testing.assert_allclose(peaks, 0.0)

    def test_multiple_stems_sum(self):
        """Multiple active stems are summed."""
        stems = self._make_stems({"vocals": 0.3, "drums": 0.2},
                                 num_frames=100)
        peaks = compute_peaks(stems, muted=set(), soloed=set(),
                              volumes={}, num_bins=10)
        # Sum of 0.3 + 0.2 = 0.5
        np.testing.assert_allclose(peaks, 0.5, atol=1e-6)

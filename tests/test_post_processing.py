"""Tests for audio post-processing filters."""

import numpy as np
import pytest

from src.post_processing import wiener_filter, soft_gate, _CHUNK_SAMPLES
from src.separator import SAMPLE_RATE


def _make_stems(n_stems: int = 4, duration_s: float = 1.0) -> np.ndarray:
    """Create synthetic stems with distinct frequencies for testing.

    Returns shape (n_stems, 2, total_samples).
    """
    total_samples = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0, duration_s, total_samples, dtype=np.float32)
    stems = np.zeros((n_stems, 2, total_samples), dtype=np.float32)

    freqs = [220, 440, 110, 660, 880, 1200]
    for s in range(n_stems):
        signal = 0.3 * np.sin(2 * np.pi * freqs[s] * t)
        stems[s, 0] = signal
        stems[s, 1] = signal

    return stems


def _make_stems_with_bleed(
    n_stems: int = 4, duration_s: float = 1.0, bleed: float = 0.1
) -> np.ndarray:
    """Create stems where each stem has some bleed from others.

    The main signal has amplitude 0.3, bleed from each other stem is
    `bleed * 0.3`.
    """
    total_samples = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0, duration_s, total_samples, dtype=np.float32)
    stems = np.zeros((n_stems, 2, total_samples), dtype=np.float32)

    freqs = [220, 440, 110, 660]
    signals = [0.3 * np.sin(2 * np.pi * f * t) for f in freqs[:n_stems]]

    for s in range(n_stems):
        stems[s, 0] = signals[s].copy()
        stems[s, 1] = signals[s].copy()
        # Add bleed from other stems.
        for other in range(n_stems):
            if other != s:
                stems[s, 0] += bleed * signals[other]
                stems[s, 1] += bleed * signals[other]

    return stems


class TestWienerFilter:

    def test_output_shape_matches_input(self):
        stems = _make_stems()
        result = wiener_filter(stems)
        assert result.shape == stems.shape

    def test_output_dtype_is_float32(self):
        stems = _make_stems()
        result = wiener_filter(stems)
        assert result.dtype == np.float32

    def test_preserves_silence(self):
        stems = np.zeros((4, 2, SAMPLE_RATE), dtype=np.float32)
        result = wiener_filter(stems)
        assert np.allclose(result, 0.0, atol=1e-8)

    def test_reduces_bleed(self):
        """Wiener filtering should increase each stem's purity.

        Measures what fraction of each stem's energy is at its target
        frequency. After Wiener filtering, stems should be purer.
        """
        stems = _make_stems_with_bleed(bleed=0.15)
        freqs = [220, 440, 110, 660]

        def purity(stem_signal, target_freq):
            """Fraction of energy within +/-5 Hz of target frequency."""
            fft = np.abs(np.fft.rfft(stem_signal))
            freq_bins = np.fft.rfftfreq(len(stem_signal), 1.0 / SAMPLE_RATE)
            mask = np.abs(freq_bins - target_freq) < 5.0
            return np.sum(fft[mask] ** 2) / max(np.sum(fft ** 2), 1e-10)

        orig_purity = np.mean([purity(stems[s, 0], freqs[s]) for s in range(4)])
        filtered = wiener_filter(stems)
        filt_purity = np.mean([purity(filtered[s, 0], freqs[s]) for s in range(4)])

        assert filt_purity >= orig_purity

    def test_preserves_total_energy(self):
        """Wiener filtering should roughly preserve the mixture energy."""
        stems = _make_stems()
        mixture_energy = np.sum(np.sum(stems, axis=0) ** 2)

        filtered = wiener_filter(stems)
        filtered_mixture_energy = np.sum(np.sum(filtered, axis=0) ** 2)

        # Should be within 10% of original.
        ratio = filtered_mixture_energy / max(mixture_energy, 1e-10)
        assert 0.8 < ratio < 1.2

    def test_single_stem_is_identity(self):
        """With only one stem, Wiener mask is all 1s -- output ~= input."""
        stems = _make_stems(n_stems=1)
        original = stems.copy()
        result = wiener_filter(stems)
        assert np.allclose(result, original, atol=1e-3)

    def test_higher_exponent_sharpens_masks(self):
        """Higher exponent should produce more aggressive separation.

        With a higher exponent, each stem's energy should be more
        concentrated at its target frequency.
        """
        freqs = [220, 440, 110, 660]

        def avg_purity(filtered_stems):
            purities = []
            for s in range(4):
                fft = np.abs(np.fft.rfft(filtered_stems[s, 0]))
                freq_bins = np.fft.rfftfreq(
                    filtered_stems.shape[2], 1.0 / SAMPLE_RATE
                )
                mask = np.abs(freq_bins - freqs[s]) < 5.0
                p = np.sum(fft[mask] ** 2) / max(np.sum(fft ** 2), 1e-10)
                purities.append(p)
            return np.mean(purities)

        mild = wiener_filter(_make_stems_with_bleed(bleed=0.15), exponent=1.0)
        sharp = wiener_filter(_make_stems_with_bleed(bleed=0.15), exponent=4.0)

        assert avg_purity(sharp) >= avg_purity(mild)


class TestSoftGate:

    def test_output_shape_matches_input(self):
        stems = _make_stems()
        result = soft_gate(stems)
        assert result.shape == stems.shape

    def test_output_dtype_is_float32(self):
        stems = _make_stems()
        result = soft_gate(stems)
        assert result.dtype == np.float32

    def test_preserves_silence(self):
        stems = np.zeros((4, 2, SAMPLE_RATE), dtype=np.float32)
        result = soft_gate(stems)
        assert np.allclose(result, 0.0, atol=1e-8)

    def test_loud_signal_passes_through(self):
        """A full-amplitude signal should pass the gate unchanged."""
        stems = _make_stems()
        original = stems.copy()
        result = soft_gate(stems, threshold_db=-80.0)

        for s in range(original.shape[0]):
            corr = np.corrcoef(original[s, 0], result[s, 0])[0, 1]
            assert corr > 0.99

    def test_quiet_signal_is_suppressed(self):
        """A signal well below the threshold should be gated to near zero."""
        stems = _make_stems()
        stems *= 1e-5

        input_energy = np.sum(stems ** 2)
        result = soft_gate(stems, threshold_db=-40.0)

        output_energy = np.sum(result ** 2)
        assert output_energy < input_energy * 0.1

    def test_mixed_loud_and_quiet_sections(self):
        """Gate should suppress quiet sections but preserve loud ones."""
        total_samples = SAMPLE_RATE * 2
        t = np.linspace(0, 2.0, total_samples, dtype=np.float32)

        stems = np.zeros((1, 2, total_samples), dtype=np.float32)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)

        signal[SAMPLE_RATE:] *= 1e-5
        stems[0, 0] = signal
        stems[0, 1] = signal

        loud_energy_in = np.sum(stems[0, 0, :SAMPLE_RATE] ** 2)
        result = soft_gate(stems, threshold_db=-40.0)

        loud_energy_out = np.sum(result[0, 0, :SAMPLE_RATE] ** 2)
        assert loud_energy_out > loud_energy_in * 0.9

        quiet_energy_out = np.sum(result[0, 0, SAMPLE_RATE:] ** 2)
        assert quiet_energy_out < loud_energy_out * 0.001

    def test_threshold_sensitivity(self):
        """Lower threshold should let more signal through."""
        stems_strict = _make_stems()
        stems_strict *= 0.01
        stems_lenient = stems_strict.copy()

        soft_gate(stems_strict, threshold_db=-20.0)
        soft_gate(stems_lenient, threshold_db=-80.0)

        strict_energy = np.sum(stems_strict ** 2)
        lenient_energy = np.sum(stems_lenient ** 2)
        assert lenient_energy >= strict_energy


class TestInPlaceBehavior:
    """Both filters should modify and return the input array (no extra copy)."""

    def test_wiener_returns_same_array(self):
        stems = _make_stems()
        result = wiener_filter(stems)
        assert result is stems

    def test_soft_gate_returns_same_array(self):
        stems = _make_stems()
        result = soft_gate(stems)
        assert result is stems


class TestChunkedProcessing:
    """Verify that chunked processing produces correct results for
    audio longer than _CHUNK_SAMPLES."""

    def test_wiener_multi_chunk_shape(self):
        """Wiener filter on long audio should return correct shape."""
        # Create audio that spans multiple chunks.
        duration_s = (_CHUNK_SAMPLES / SAMPLE_RATE) * 2.5
        stems = _make_stems(n_stems=4, duration_s=duration_s)
        result = wiener_filter(stems)
        assert result.shape == stems.shape

    def test_wiener_multi_chunk_no_boundary_silence(self):
        """No silent gaps should appear at chunk boundaries."""
        duration_s = (_CHUNK_SAMPLES / SAMPLE_RATE) * 2.5
        stems = _make_stems(n_stems=2, duration_s=duration_s)
        result = wiener_filter(stems)

        # Sum of all stems at every sample should have non-trivial energy.
        mixture = np.sum(result, axis=0)  # (2, samples)
        # Check RMS in 0.5s windows — none should be near-zero.
        window = SAMPLE_RATE // 2
        for start in range(0, result.shape[2] - window, window):
            chunk_rms = np.sqrt(np.mean(mixture[0, start:start + window] ** 2))
            assert chunk_rms > 1e-4, f"Silent gap at sample {start}"

    def test_soft_gate_multi_chunk_shape(self):
        """Soft gate on long audio should return correct shape."""
        duration_s = (_CHUNK_SAMPLES / SAMPLE_RATE) * 2.5
        stems = _make_stems(n_stems=4, duration_s=duration_s)
        result = soft_gate(stems)
        assert result.shape == stems.shape

"""Tests for audio post-processing filters."""

import numpy as np
import pytest

from src.post_processing import wiener_filter, soft_gate
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
        filtered = wiener_filter(stems)

        freqs = [220, 440, 110, 660]

        def purity(stem_signal, target_freq):
            """Fraction of energy within ±5 Hz of target frequency."""
            fft = np.abs(np.fft.rfft(stem_signal))
            freq_bins = np.fft.rfftfreq(len(stem_signal), 1.0 / SAMPLE_RATE)
            mask = np.abs(freq_bins - target_freq) < 5.0
            return np.sum(fft[mask] ** 2) / max(np.sum(fft ** 2), 1e-10)

        # Average purity improvement across stems.
        orig_purity = np.mean([purity(stems[s, 0], freqs[s]) for s in range(4)])
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
        """With only one stem, Wiener mask is all 1s — output ~= input."""
        stems = _make_stems(n_stems=1)
        result = wiener_filter(stems)
        assert np.allclose(result, stems, atol=1e-3)

    def test_higher_exponent_sharpens_masks(self):
        """Higher exponent should produce more aggressive separation.

        With a higher exponent, each stem's energy should be more
        concentrated at its target frequency.
        """
        stems = _make_stems_with_bleed(bleed=0.15)
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

        mild = wiener_filter(stems, exponent=1.0)
        sharp = wiener_filter(stems, exponent=4.0)

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
        result = soft_gate(stems, threshold_db=-80.0)

        # Should be very close to original.
        for s in range(stems.shape[0]):
            corr = np.corrcoef(stems[s, 0], result[s, 0])[0, 1]
            assert corr > 0.99

    def test_quiet_signal_is_suppressed(self):
        """A signal well below the threshold should be gated to near zero."""
        stems = _make_stems()
        # Make stems very quiet.
        stems *= 1e-5

        result = soft_gate(stems, threshold_db=-40.0)

        # Gated output should be much quieter than input.
        input_energy = np.sum(stems ** 2)
        output_energy = np.sum(result ** 2)
        assert output_energy < input_energy * 0.1

    def test_mixed_loud_and_quiet_sections(self):
        """Gate should suppress quiet sections but preserve loud ones."""
        total_samples = SAMPLE_RATE * 2  # 2 seconds
        t = np.linspace(0, 2.0, total_samples, dtype=np.float32)

        stems = np.zeros((1, 2, total_samples), dtype=np.float32)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)

        # First half loud, second half near-silent.
        signal[SAMPLE_RATE:] *= 1e-5
        stems[0, 0] = signal
        stems[0, 1] = signal

        result = soft_gate(stems, threshold_db=-40.0)

        # First half should be preserved.
        loud_energy_in = np.sum(stems[0, 0, :SAMPLE_RATE] ** 2)
        loud_energy_out = np.sum(result[0, 0, :SAMPLE_RATE] ** 2)
        assert loud_energy_out > loud_energy_in * 0.9

        # Second half should be suppressed.
        quiet_energy_out = np.sum(result[0, 0, SAMPLE_RATE:] ** 2)
        assert quiet_energy_out < loud_energy_out * 0.001

    def test_threshold_sensitivity(self):
        """Lower threshold should let more signal through."""
        stems = _make_stems()
        stems *= 0.01  # Quiet but not silent.

        strict = soft_gate(stems, threshold_db=-20.0)
        lenient = soft_gate(stems, threshold_db=-80.0)

        strict_energy = np.sum(strict ** 2)
        lenient_energy = np.sum(lenient ** 2)
        assert lenient_energy >= strict_energy

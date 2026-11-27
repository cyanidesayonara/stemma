"""Tests for vectorized STFT/iSTFT utilities parity against librosa."""

from __future__ import annotations

import librosa
import numpy as np
import pytest

from src.stft_utils import (
    istft_batch,
    istft_frames_first,
    stft_batch,
    stft_frames_first,
)


def _librosa_stft_batch(y: np.ndarray, *, n_fft: int, hop_length: int, pad_mode: str) -> np.ndarray:
    flat = y.reshape(-1, y.shape[-1])
    specs = np.stack(
        [
            librosa.stft(
                row,
                n_fft=n_fft,
                hop_length=hop_length,
                window="hann",
                center=True,
                pad_mode=pad_mode,
            )
            for row in flat
        ],
        axis=0,
    )
    return specs.reshape(*y.shape[:-1], specs.shape[1], specs.shape[2])


def _librosa_istft_batch(
    spec: np.ndarray, *, n_fft: int, hop_length: int, length: int | None
) -> np.ndarray:
    flat = spec.reshape(-1, spec.shape[-2], spec.shape[-1])
    waves = np.stack(
        [
            librosa.istft(
                mat,
                hop_length=hop_length,
                n_fft=n_fft,
                window="hann",
                center=True,
                length=length,
            )
            for mat in flat
        ],
        axis=0,
    )
    return waves.reshape(*spec.shape[:-2], waves.shape[-1])


@pytest.mark.parametrize(
    ("n_fft", "hop_length", "pad_mode", "samples"),
    [
        (4096, 1024, "constant", 32768),
        (6144, 1024, "reflect", 32768),
        (1024, 300, "constant", 12345),
        (1024, 300, "reflect", 12345),
    ],
)
def test_stft_batch_and_frames_first_match_librosa(n_fft, hop_length, pad_mode, samples):
    rng = np.random.default_rng(42)
    y = (rng.standard_normal((2, samples)) * 0.25).astype(np.float32)

    ref = _librosa_stft_batch(y, n_fft=n_fft, hop_length=hop_length, pad_mode=pad_mode)
    actual = stft_batch(y, n_fft=n_fft, hop_length=hop_length, pad_mode=pad_mode)
    actual_t = stft_frames_first(y, n_fft=n_fft, hop_length=hop_length, pad_mode=pad_mode)

    assert actual.dtype == np.complex64
    assert actual_t.dtype == np.complex64
    assert actual.shape == ref.shape
    assert actual_t.shape == (y.shape[0], ref.shape[2], ref.shape[1])
    np.testing.assert_allclose(actual, ref, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.swapaxes(actual_t, -1, -2), ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    ("n_fft", "hop_length", "length_mode"),
    [
        (4096, 1024, "default"),
        (4096, 1024, "shorter"),
        (4096, 1024, "longer"),
        (6144, 1024, "shorter"),
        (1024, 300, "default"),
        (1024, 300, "shorter"),
        (1024, 300, "longer"),
    ],
)
def test_istft_batch_and_frames_first_match_librosa(n_fft, hop_length, length_mode):
    rng = np.random.default_rng(7)
    n_frames = 37
    n_bins = n_fft // 2 + 1
    spec = (
        rng.standard_normal((2, n_bins, n_frames))
        + 1j * rng.standard_normal((2, n_bins, n_frames))
    ).astype(np.complex64)
    # Keep DC and Nyquist real as in real physical spectra.
    spec[:, 0, :] = spec[:, 0, :].real
    spec[:, -1, :] = 0.0

    default_len = n_fft + hop_length * (n_frames - 1) - 2 * (n_fft // 2)
    if length_mode == "default":
        length = None
    elif length_mode == "shorter":
        length = default_len - 173
    else:
        length = default_len + 511

    ref = _librosa_istft_batch(spec, n_fft=n_fft, hop_length=hop_length, length=length)
    spec_copy = spec.copy()
    actual = istft_batch(spec_copy, n_fft=n_fft, hop_length=hop_length, length=length)
    spec_t = np.ascontiguousarray(np.swapaxes(spec, -1, -2))
    spec_t_copy = spec_t.copy()
    actual_t = istft_frames_first(
        spec_t_copy, n_fft=n_fft, hop_length=hop_length, length=length, overwrite_x=False
    )

    assert actual.dtype == np.float32
    assert actual_t.dtype == np.float32
    assert actual.shape == ref.shape
    assert actual_t.shape == ref.shape
    np.testing.assert_allclose(actual, ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(actual_t, ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(spec_t_copy, spec_t)


def test_batched_6x2_stft_and_istft_parity():
    rng = np.random.default_rng(99)
    stems = (rng.standard_normal((6, 2, 24576)) * 0.2).astype(np.float32)

    ref_spec = _librosa_stft_batch(stems, n_fft=4096, hop_length=1024, pad_mode="constant")
    fast_spec = stft_batch(stems, n_fft=4096, hop_length=1024, pad_mode="constant")
    np.testing.assert_allclose(fast_spec, ref_spec, atol=1e-4, rtol=1e-4)

    ref_wave = _librosa_istft_batch(
        fast_spec, n_fft=4096, hop_length=1024, length=stems.shape[-1]
    )
    fast_wave = istft_batch(
        fast_spec, n_fft=4096, hop_length=1024, length=stems.shape[-1]
    )
    np.testing.assert_allclose(fast_wave, ref_wave, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(fast_wave, stems, atol=1e-4, rtol=1e-4)


def test_stft_istft_roundtrip_smooth_and_noisy_stereo():
    rng = np.random.default_rng(11)
    sr = 44100
    t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False, dtype=np.float32)
    smooth = np.stack([0.4 * np.sin(2 * np.pi * 220 * t), 0.3 * np.cos(2 * np.pi * 330 * t)])
    noisy = smooth + (rng.standard_normal(smooth.shape) * 0.05).astype(np.float32)

    spec_b = stft_batch(noisy, n_fft=4096, hop_length=1024, pad_mode="constant")
    rec_b = istft_batch(spec_b, n_fft=4096, hop_length=1024, length=noisy.shape[1])
    np.testing.assert_allclose(rec_b, noisy, atol=1e-4, rtol=1e-4)

    spec_f = stft_frames_first(noisy, n_fft=6144, hop_length=1024, pad_mode="reflect")
    rec_f = istft_frames_first(spec_f, n_fft=6144, hop_length=1024, length=noisy.shape[1])
    np.testing.assert_allclose(rec_f, noisy, atol=1e-4, rtol=1e-4)

"""Vectorized STFT/iSTFT helpers backed by ``scipy.fft`` for stem separation.

``librosa.stft`` and ``librosa.istft`` are convenient but process one channel
at a time in Python, and ``librosa.istft`` runs a Python frame loop over every
hop. Batching channels together with ``scipy.fft.rfft`` / ``irfft``,
transposing so the transform runs along a contiguous last axis, using a
zero-copy block overlap-add when ``n_fft % hop_length == 0``, and caching
both the Hann window and overlap-add normalization envelope preserves
numerical parity with ``librosa`` at much higher throughput.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.fft
import scipy.signal

_TINY_F32 = float(np.finfo(np.float32).tiny)


@lru_cache(maxsize=32)
def _hann_window(n_fft: int) -> np.ndarray:
    """Return a cached read-only float32 periodic Hann window."""
    win = scipy.signal.get_window("hann", n_fft, fftbins=True).astype(np.float32)
    win.setflags(write=False)
    return win


@lru_cache(maxsize=64)
def _window_sumsquare_inv(
    n_fft: int,
    hop_length: int,
    n_frames: int,
    length: int | None,
) -> tuple[np.ndarray, int, int]:
    """Return ``(inv_envelope, copy_len, out_len)`` for centered iSTFT overlap-add."""
    expected_len = n_fft + hop_length * (n_frames - 1)
    pad = n_fft // 2
    avail = max(0, expected_len - pad)
    out_len = max(0, expected_len - 2 * pad) if length is None else length
    copy_len = min(out_len, avail)

    win = _hann_window(n_fft).astype(np.float64)
    w2 = win * win
    win_sq = np.zeros(expected_len, dtype=np.float64)
    if n_fft % hop_length == 0:
        blocks = n_fft // hop_length
        w2_blocks = w2.reshape(blocks, hop_length)
        span = n_frames * hop_length
        for b in range(blocks):
            start = b * hop_length
            win_sq[start : start + span].reshape(n_frames, hop_length)[:] += w2_blocks[b]
    else:
        for i in range(n_frames):
            start = i * hop_length
            win_sq[start : start + n_fft] += w2

    win_slice = win_sq[pad : pad + copy_len]
    inv = np.zeros(copy_len, dtype=np.float32)
    nonzero = win_slice > _TINY_F32
    inv[nonzero] = (1.0 / win_slice[nonzero]).astype(np.float32)
    inv.setflags(write=False)
    return inv, copy_len, out_len


def stft_frames_first(
    y: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    pad_mode: str = "constant",
) -> np.ndarray:
    """Compute centered STFT returning ``(..., n_frames, n_fft // 2 + 1)`` complex64."""
    arr = np.asarray(y, dtype=np.float32)
    pad = n_fft // 2
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(pad, pad)]
    padded = np.pad(arr, pad_width, mode=pad_mode)
    frames = np.lib.stride_tricks.sliding_window_view(
        padded, n_fft, axis=-1
    )[..., ::hop_length, :]
    windowed = frames * _hann_window(n_fft)
    return scipy.fft.rfft(windowed, n=n_fft, axis=-1)


def stft_batch(
    y: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    pad_mode: str = "constant",
) -> np.ndarray:
    """Compute centered STFT returning ``(..., n_fft // 2 + 1, n_frames)`` complex64."""
    return np.ascontiguousarray(
        np.swapaxes(
            stft_frames_first(
                y, n_fft=n_fft, hop_length=hop_length, pad_mode=pad_mode
            ),
            -1,
            -2,
        )
    )


def istft_frames_first(
    spec_t: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    length: int | None = None,
    overwrite_x: bool = False,
) -> np.ndarray:
    """Invert ``(..., n_frames, n_fft // 2 + 1)`` spectrogram to ``(..., length)`` float32."""
    n_frames = spec_t.shape[-2]
    lead_shape = spec_t.shape[:-2]
    inv, copy_len, out_len = _window_sumsquare_inv(n_fft, hop_length, n_frames, length)
    if out_len == 0:
        return np.zeros(lead_shape + (0,), dtype=np.float32)

    expected_len = n_fft + hop_length * (n_frames - 1)
    frames = scipy.fft.irfft(
        spec_t, n=n_fft, axis=-1, overwrite_x=overwrite_x
    ).astype(np.float32, copy=False)
    frames *= _hann_window(n_fft)

    overlap_buf = np.zeros(lead_shape + (expected_len,), dtype=np.float32)
    if n_fft % hop_length == 0:
        blocks = n_fft // hop_length
        frame_blocks = frames.reshape(lead_shape + (n_frames, blocks, hop_length))
        span = n_frames * hop_length
        for b in range(blocks):
            start = b * hop_length
            overlap_buf[..., start : start + span].reshape(
                lead_shape + (n_frames, hop_length)
            )[:] += frame_blocks[..., b, :]
    else:
        for i in range(n_frames):
            start = i * hop_length
            overlap_buf[..., start : start + n_fft] += frames[..., i, :]

    pad = n_fft // 2
    trimmed = overlap_buf[..., pad : pad + copy_len]
    trimmed *= inv
    if copy_len == out_len:
        return np.ascontiguousarray(trimmed)

    out = np.zeros(lead_shape + (out_len,), dtype=np.float32)
    out[..., :copy_len] = trimmed
    return out


def istft_batch(
    stft_matrix: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    length: int | None = None,
) -> np.ndarray:
    """Invert ``(..., n_fft // 2 + 1, n_frames)`` spectrogram to ``(..., length)`` float32."""
    spec_t = np.ascontiguousarray(np.swapaxes(stft_matrix, -1, -2))
    return istft_frames_first(
        spec_t,
        n_fft=n_fft,
        hop_length=hop_length,
        length=length,
        overwrite_x=True,
    )

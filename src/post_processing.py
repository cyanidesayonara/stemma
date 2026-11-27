"""Audio post-processing filters for improving stem separation quality.

Applied after ONNX model inference to reduce bleed and suppress faint
ghost artifacts. Both filters operate **in-place** on the stems array to
avoid allocating full-track copies (saves ~1 GB on a 4-minute 6-stem song).
Chunked processing keeps peak per-chunk memory bounded regardless of
track length.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from src.separator import HOP_LENGTH, NFFT, SAMPLE_RATE
from src.stft_utils import istft_frames_first, stft_frames_first

# Default chunk size for frequency-domain processing (~10 seconds).
# Keeps peak memory per chunk under ~50 MB even with 6 stems.
_CHUNK_SAMPLES = SAMPLE_RATE * 10
_CHUNK_OVERLAP = NFFT  # Overlap to avoid boundary artifacts after iSTFT.


def wiener_filter(
    stems: np.ndarray,
    n_iterations: int = 1,
    exponent: float = 2.0,
    should_cancel: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Apply multi-channel Wiener filtering to reduce inter-stem bleed.

    Computes STFT masks per ~10-second chunk using stem-level threads,
    deriving the mixture spectrum as the sum of stem spectra in the
    frequency domain to avoid full-track copies.

    Operates **in-place**: the input ``stems`` array is modified and
    returned.
    """
    n_stems, _n_channels, total_samples = stems.shape
    if n_stems == 0 or total_samples == 0:
        return stems

    chunks = _chunk_boundaries(total_samples, _CHUNK_SAMPLES, _CHUNK_OVERLAP)
    pool = ThreadPoolExecutor(max_workers=min(n_stems, 6)) if n_stems > 1 else None

    def _chunk_stft(args: tuple[int, int, int]) -> np.ndarray:
        s, cs, ce = args
        return stft_frames_first(
            stems[s, :, cs:ce],
            n_fft=NFFT,
            hop_length=HOP_LENGTH,
            pad_mode="constant",
        )

    def _chunk_istft(args: tuple[np.ndarray, int]) -> np.ndarray:
        spec_t, clen = args
        return istft_frames_first(
            spec_t,
            n_fft=NFFT,
            hop_length=HOP_LENGTH,
            length=clen,
            overwrite_x=True,
        )

    try:
        for chunk_start, chunk_end, out_start, out_end in chunks:
            _check_cancel(should_cancel)
            chunk_len = chunk_end - chunk_start
            out_len = out_end - out_start
            trim_left = out_start - chunk_start

            stft_args = [(s, chunk_start, chunk_end) for s in range(n_stems)]
            if pool is not None:
                stem_stfts = list(pool.map(_chunk_stft, stft_args))
            else:
                stem_stfts = [_chunk_stft(stft_args[0])]

            current_stfts = np.stack(stem_stfts, axis=0)
            mix_stft = np.sum(current_stfts, axis=0)

            for _ in range(n_iterations):
                if exponent == 2.0:
                    mag_power = (
                        current_stfts.real * current_stfts.real
                        + current_stfts.imag * current_stfts.imag
                    )
                else:
                    mag_power = np.abs(current_stfts) ** exponent
                total_mag = np.maximum(np.sum(mag_power, axis=0), 1e-10)
                mag_power /= total_mag
                current_stfts = mag_power * mix_stft[np.newaxis, ...]

            istft_args = [(current_stfts[s], chunk_len) for s in range(n_stems)]
            if pool is not None:
                waves = list(pool.map(_chunk_istft, istft_args))
            else:
                waves = [_chunk_istft(istft_args[0])]

            for s in range(n_stems):
                stems[s, :, out_start:out_end] = (
                    waves[s][:, trim_left : trim_left + out_len]
                )

            del stem_stfts, current_stfts, mix_stft, waves, stft_args, istft_args
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    return stems


def soft_gate(
    stems: np.ndarray,
    threshold_db: float = -60.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    attack_frames: int = 2,
    release_frames: int = 4,
    should_cancel: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Apply per-stem soft gating to suppress faint ghost artifacts.

    Computes short-time RMS energy per stem via a prefix-sum sliding window
    and applies a smoothed piecewise-linear gain envelope in place.
    """
    n_stems, _n_channels, total_samples = stems.shape
    if n_stems == 0 or total_samples == 0:
        return stems
    _check_cancel(should_cancel)

    threshold_linear = 10.0 ** (threshold_db / 20.0)
    thresh_sq = threshold_linear * threshold_linear
    attack_coeff = 1.0 / max(attack_frames, 1)
    release_coeff = 1.0 / max(release_frames, 1)
    pad = frame_length // 2
    ramp = np.arange(hop_length, dtype=np.float32) / hop_length

    def _gate_stem(s: int) -> None:
        _check_cancel(should_cancel)
        mono = np.mean(stems[s], axis=0)
        padded = np.pad(mono, (pad, pad), mode="constant")
        sq = padded.astype(np.float64) ** 2
        csum = np.empty(len(sq) + 1, dtype=np.float64)
        csum[0] = 0.0
        np.cumsum(sq, out=csum[1:])
        starts = np.arange(0, len(padded) - frame_length + 1, hop_length)
        mean_sq = np.maximum(
            (csum[starts + frame_length] - csum[starts]) / frame_length,
            0.0,
        )

        gate = (mean_sq >= thresh_sq).astype(np.float32)
        n_frames = len(gate)
        if n_frames == 0:
            return

        smoothed = np.empty_like(gate)
        current = float(gate[0])
        for i in range(n_frames):
            val = float(gate[i])
            coeff = attack_coeff if val > current else release_coeff
            current += coeff * (val - current)
            smoothed[i] = current

        envelope = np.empty(total_samples, dtype=np.float32)
        full_hops = min((n_frames - 1) * hop_length, total_samples)
        n_full = full_hops // hop_length
        if n_full > 0:
            s0 = smoothed[:n_full, np.newaxis]
            s1 = smoothed[1 : n_full + 1, np.newaxis]
            envelope[: n_full * hop_length] = (
                s0 + (s1 - s0) * ramp[np.newaxis, :]
            ).ravel()

        rem = total_samples - n_full * hop_length
        if rem > 0:
            tail = envelope[n_full * hop_length :]
            if n_full < n_frames - 1:
                tail[:] = smoothed[n_full] + (
                    smoothed[n_full + 1] - smoothed[n_full]
                ) * ramp[:rem]
            else:
                tail[:] = smoothed[-1]

        stems[s] *= envelope[np.newaxis, :]

    if n_stems > 1:
        with ThreadPoolExecutor(max_workers=min(n_stems, 6)) as pool:
            list(pool.map(_gate_stem, range(n_stems)))
    else:
        _gate_stem(0)

    return stems


def _check_cancel(should_cancel: Callable[[], bool] | None) -> None:
    """Raise `InterruptedError` if the caller asked to stop."""
    if should_cancel is not None and should_cancel():
        raise InterruptedError("Separation cancelled by user.")


def _chunk_boundaries(
    total_samples: int,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    """Compute chunk read/write boundaries for overlap-safe processing.

    Returns a list of (chunk_start, chunk_end, out_start, out_end) tuples.
    Each chunk is read from [chunk_start, chunk_end) but only the
    [out_start, out_end) portion is written to the output, avoiding
    boundary artifacts from the STFT windowing.
    """
    if total_samples <= chunk_size:
        return [(0, total_samples, 0, total_samples)]

    chunks = []
    pos = 0
    while pos < total_samples:
        chunk_start = max(0, pos - overlap)
        chunk_end = min(total_samples, pos + chunk_size + overlap)
        out_start = pos
        out_end = min(pos + chunk_size, total_samples)
        chunks.append((chunk_start, chunk_end, out_start, out_end))
        pos += chunk_size

    return chunks

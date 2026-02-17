"""Audio post-processing filters for improving stem separation quality.

Applied after ONNX model inference to reduce bleed and suppress faint
ghost artifacts. Both filters process audio in chunks to avoid
allocating full-track STFT matrices that would consume multiple GB of
RAM on longer tracks.
"""

import numpy as np
import librosa

from src.separator import NFFT, HOP_LENGTH, SAMPLE_RATE

# Default chunk size for frequency-domain processing (~10 seconds).
# Keeps peak memory per chunk under ~50 MB even with 6 stems.
_CHUNK_SAMPLES = SAMPLE_RATE * 10
_CHUNK_OVERLAP = NFFT  # Overlap to avoid boundary artifacts after iSTFT.


def wiener_filter(
    stems: np.ndarray,
    n_iterations: int = 1,
    exponent: float = 2.0,
) -> np.ndarray:
    """Apply multi-channel Wiener filtering to reduce inter-stem bleed.

    Given the model's initial stem estimates, computes soft frequency masks
    based on relative magnitude and re-applies them to the mixture STFT.
    This reinforces the model's separation decisions and suppresses bleed.

    Processes audio in ~10-second chunks with overlap to keep peak memory
    usage bounded regardless of track length.

    Args:
        stems: Separated stems, shape (n_stems, 2, total_samples).
        n_iterations: Number of Wiener iterations (1 is usually enough).
        exponent: Power applied to magnitudes when computing masks.
            Higher values create sharper masks (2.0 = standard Wiener).

    Returns:
        Filtered stems array with same shape as input.
    """
    n_stems, n_channels, total_samples = stems.shape
    result = np.zeros_like(stems)

    # Reconstruct the mixture from stem estimates.
    mixture = np.sum(stems, axis=0)  # (2, total_samples)

    # Build chunk boundaries with overlap.
    chunks = _chunk_boundaries(total_samples, _CHUNK_SAMPLES, _CHUNK_OVERLAP)

    for chunk_start, chunk_end, out_start, out_end in chunks:
        for ch in range(n_channels):
            chunk_len = chunk_end - chunk_start
            out_len = out_end - out_start
            trim_left = out_start - chunk_start

            # STFT of the mixture chunk.
            mix_stft = librosa.stft(
                mixture[ch, chunk_start:chunk_end],
                n_fft=NFFT, hop_length=HOP_LENGTH, center=True,
            )

            # Compute magnitude sum across stems (denominator).
            total_mag = np.zeros_like(np.abs(mix_stft))
            stem_stfts = []
            for s in range(n_stems):
                stft = librosa.stft(
                    stems[s, ch, chunk_start:chunk_end],
                    n_fft=NFFT, hop_length=HOP_LENGTH, center=True,
                )
                stem_stfts.append(stft)
                total_mag += np.abs(stft) ** exponent

            total_mag = np.maximum(total_mag, 1e-10)

            # Iterative Wiener: apply masks and optionally refine.
            current_stfts = list(stem_stfts)
            for _ in range(n_iterations):
                if _ > 0:
                    # Recompute magnitudes from current estimates.
                    total_mag = np.zeros_like(np.abs(mix_stft))
                    for s in range(n_stems):
                        total_mag += np.abs(current_stfts[s]) ** exponent
                    total_mag = np.maximum(total_mag, 1e-10)

                for s in range(n_stems):
                    mask = np.abs(current_stfts[s]) ** exponent / total_mag
                    current_stfts[s] = mask * mix_stft

            # iSTFT back to time domain and write the non-overlapping portion.
            for s in range(n_stems):
                audio_chunk = librosa.istft(
                    current_stfts[s],
                    hop_length=HOP_LENGTH, n_fft=NFFT,
                    length=chunk_len, center=True,
                )
                result[s, ch, out_start:out_end] = (
                    audio_chunk[trim_left:trim_left + out_len]
                )

            # Free STFT arrays for this channel/chunk before next iteration.
            del stem_stfts, current_stfts, mix_stft, total_mag

    return result


def soft_gate(
    stems: np.ndarray,
    threshold_db: float = -60.0,
    frame_length: int = 2048,
    hop_length: int = 512,
    attack_frames: int = 2,
    release_frames: int = 4,
) -> np.ndarray:
    """Apply per-stem soft gating to suppress faint ghost artifacts.

    Computes short-time RMS energy for each stem. When energy drops
    below the threshold, a smooth gain envelope fades the signal to
    silence. This removes faint echoes of other instruments that the
    model leaked through.

    Processes one stem at a time to keep memory bounded.

    Args:
        stems: Separated stems, shape (n_stems, 2, total_samples).
        threshold_db: Gate threshold in dB below peak (default -60 dB).
        frame_length: RMS analysis window size in samples.
        hop_length: RMS analysis hop size in samples.
        attack_frames: Number of RMS frames for gate to open (smooth on).
        release_frames: Number of RMS frames for gate to close (smooth off).

    Returns:
        Gated stems array with same shape as input.
    """
    n_stems, n_channels, total_samples = stems.shape
    result = np.zeros_like(stems)

    threshold_linear = 10.0 ** (threshold_db / 20.0)
    attack_coeff = 1.0 / max(attack_frames, 1)
    release_coeff = 1.0 / max(release_frames, 1)

    for s in range(n_stems):
        # Compute RMS across channels for this stem.
        mono = np.mean(stems[s], axis=0)  # (total_samples,)

        # Frame-based RMS — this is lightweight (returns one float per frame).
        rms = librosa.feature.rms(
            y=mono, frame_length=frame_length, hop_length=hop_length,
        )[0]

        # Binary gate decision.
        gate = (rms >= threshold_linear).astype(np.float32)

        # Smooth attack/release via IIR filter.
        smoothed = np.empty_like(gate)
        current = gate[0]
        for i in range(len(gate)):
            if gate[i] > current:
                current += attack_coeff * (gate[i] - current)
            else:
                current += release_coeff * (gate[i] - current)
            smoothed[i] = current

        # Upsample envelope to sample rate via linear interpolation.
        frame_times = librosa.frames_to_samples(
            np.arange(len(smoothed)), hop_length=hop_length,
        )
        envelope = np.interp(
            np.arange(total_samples), frame_times, smoothed,
        ).astype(np.float32)

        # Apply envelope to all channels.
        for ch in range(n_channels):
            result[s, ch] = stems[s, ch] * envelope

        del mono, rms, gate, smoothed, envelope

    return result


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

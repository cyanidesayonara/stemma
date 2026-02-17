"""Audio post-processing filters for improving stem separation quality.

Applied after ONNX model inference to reduce bleed and suppress faint
ghost artifacts. Both filters operate in the frequency domain and are
designed to be non-destructive to the primary stem content.
"""

import numpy as np
import librosa

from src.separator import NFFT, HOP_LENGTH, SAMPLE_RATE


def wiener_filter(
    stems: np.ndarray,
    n_iterations: int = 1,
    exponent: float = 2.0,
    residual: bool = True,
) -> np.ndarray:
    """Apply multi-channel Wiener filtering to reduce inter-stem bleed.

    Given the model's initial stem estimates, computes soft frequency masks
    based on relative magnitude and re-applies them to the mixture STFT.
    This reinforces the model's separation decisions and suppresses bleed.

    Args:
        stems: Separated stems, shape (n_stems, 2, total_samples).
        n_iterations: Number of Wiener iterations (1 is usually enough).
        exponent: Power applied to magnitudes when computing masks.
            Higher values create sharper masks (2.0 = standard Wiener).
        residual: If True, redistribute residual energy (difference between
            mixture and sum of masked stems) proportionally across stems.

    Returns:
        Filtered stems array with same shape as input.
    """
    n_stems, n_channels, total_samples = stems.shape

    # Reconstruct the mixture from stem estimates.
    mixture = np.sum(stems, axis=0)  # (2, total_samples)

    result = np.zeros_like(stems)

    for ch in range(n_channels):
        # Compute STFT of the mixture.
        mix_stft = librosa.stft(
            mixture[ch], n_fft=NFFT, hop_length=HOP_LENGTH, center=True
        )  # (freq_bins, time_frames)

        # Compute STFT of each stem estimate.
        stem_stfts = []
        for s in range(n_stems):
            stft = librosa.stft(
                stems[s, ch], n_fft=NFFT, hop_length=HOP_LENGTH, center=True
            )
            stem_stfts.append(stft)

        # Iterative Wiener filtering.
        current_stfts = list(stem_stfts)
        for _ in range(n_iterations):
            # Compute magnitude-based soft masks.
            magnitudes = np.array(
                [np.abs(s) ** exponent for s in current_stfts]
            )  # (n_stems, freq, time)
            total_mag = np.sum(magnitudes, axis=0)
            total_mag = np.maximum(total_mag, 1e-10)

            masks = magnitudes / total_mag  # (n_stems, freq, time)

            # Apply masks to mixture STFT.
            current_stfts = [masks[s] * mix_stft for s in range(n_stems)]

        # Convert back to time domain.
        for s in range(n_stems):
            result[s, ch] = librosa.istft(
                current_stfts[s],
                hop_length=HOP_LENGTH,
                n_fft=NFFT,
                length=total_samples,
                center=True,
            )

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

    for s in range(n_stems):
        # Compute RMS across channels for this stem.
        mono = np.mean(stems[s], axis=0)  # (total_samples,)

        # Frame-based RMS.
        rms = librosa.feature.rms(
            y=mono,
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]  # (n_frames,)

        # Compute gate envelope: 1.0 where signal is above threshold, 0.0 below.
        gate = (rms >= threshold_linear).astype(np.float32)

        # Smooth attack and release using a simple IIR-style filter.
        smoothed = np.zeros_like(gate)
        attack_coeff = 1.0 / max(attack_frames, 1)
        release_coeff = 1.0 / max(release_frames, 1)

        current = gate[0]
        for i in range(len(gate)):
            if gate[i] > current:
                current += attack_coeff * (gate[i] - current)
            else:
                current += release_coeff * (gate[i] - current)
            smoothed[i] = current

        # Upsample the gate envelope to sample rate.
        n_frames = len(smoothed)
        frame_times = librosa.frames_to_samples(
            np.arange(n_frames), hop_length=hop_length
        )

        # Linear interpolation to full sample resolution.
        sample_indices = np.arange(total_samples)
        envelope = np.interp(sample_indices, frame_times, smoothed)
        envelope = envelope.astype(np.float32)

        # Apply envelope to all channels.
        for ch in range(n_channels):
            result[s, ch] = stems[s, ch] * envelope

    return result

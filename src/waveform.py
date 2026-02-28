"""Waveform peak computation for visualization.

Pure numpy module with no Qt dependency. Computes per-bin peak amplitudes
from stem audio data, respecting mute/solo/volume state.
"""

from __future__ import annotations

import numpy as np


def compute_peaks(
    stems: dict[str, np.ndarray],
    muted: set[str],
    soloed: set[str],
    volumes: dict[str, float],
    num_bins: int,
) -> np.ndarray:
    """Compute peak amplitude per bin from active stems.

    Args:
        stems: Mapping of stem name to audio array, shape (frames, 2).
        muted: Set of muted stem names.
        soloed: Set of soloed stem names.
        volumes: Mapping of stem name to gain (default 1.0 if absent).
        num_bins: Number of output bins (typically widget width or fixed 2000).

    Returns:
        1-D float32 array of shape (num_bins,) with peak amplitudes per bin.
    """
    if not stems or num_bins <= 0:
        return np.zeros(num_bins, dtype=np.float32)

    # Determine active stems (same logic as audio callback)
    if soloed:
        active = [name for name in stems if name in soloed]
    else:
        active = [name for name in stems if name not in muted]

    if not active:
        return np.zeros(num_bins, dtype=np.float32)

    # Find total frames across all active stems
    total_frames = max(stems[name].shape[0] for name in active)
    if total_frames == 0:
        return np.zeros(num_bins, dtype=np.float32)

    # Sum active stems weighted by volume
    mix = np.zeros(total_frames, dtype=np.float32)
    for name in active:
        data = stems[name]
        gain = volumes.get(name, 1.0)
        # Mono amplitude: max of abs across channels
        mono = np.max(np.abs(data), axis=1)
        frames = mono.shape[0]
        mix[:frames] += mono * gain

    # Compute peak per bin using reduceat (fully vectorized)
    bin_starts = np.linspace(0, total_frames, num_bins + 1, dtype=np.int64)[:-1]
    # reduceat takes the max of mix[bin_starts[i]:bin_starts[i+1]] for each i
    peaks = np.maximum.reduceat(mix, bin_starts).astype(np.float32)
    return peaks

"""Shared metronome click generation used by the player and exporter."""

import numpy as np


def generate_click(sample_rate: int) -> np.ndarray:
    """Generate a short click sound for the metronome.

    Returns a stereo numpy array (shape (N, 2), dtype float32) containing
    a 1000 Hz sine tone with an exponential decay envelope, lasting 30ms.
    """
    duration = 0.03  # 30 ms
    n_frames = int(sample_rate * duration)
    t = np.arange(n_frames, dtype=np.float32) / sample_rate
    tone = np.sin(2.0 * np.pi * 1000.0 * t)
    envelope = np.exp(-t / 0.006).astype(np.float32)
    click_mono = (tone * envelope).astype(np.float32)
    return np.column_stack((click_mono, click_mono))


def generate_count_in(
    beats: int,
    bpm: float,
    sample_rate: int,
    volume: float = 0.5,
) -> np.ndarray:
    """Generate a count-in audio buffer with metronome clicks.

    Args:
        beats: Number of count-in beats.
        bpm: Tempo in beats per minute (clamped to 20--300).
        sample_rate: Audio sample rate in Hz.
        volume: Click volume (0.0--2.0, default 0.5).

    Returns:
        Stereo float32 array of shape (total_frames, 2).
    """
    bpm = max(20.0, min(300.0, float(bpm)))
    beat_interval = int(60.0 / bpm * sample_rate)
    total_frames = beats * beat_interval
    buf = np.zeros((total_frames, 2), dtype=np.float32)
    click = generate_click(sample_rate)
    click_len = len(click)

    for beat in range(beats):
        start = beat * beat_interval
        end = min(start + click_len, total_frames)
        n = end - start
        buf[start:end] += click[:n] * volume

    np.clip(buf, -1.0, 1.0, out=buf)
    return buf

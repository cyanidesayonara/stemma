"""Metronome utilities for BPM calculation."""


def tap_tempo(tap_times: list[float], max_taps: int = 8) -> float:
    """Calculate BPM from a list of tap timestamps.

    Args:
        tap_times: Monotonic timestamps in seconds (e.g. from time.time()).
            Only the last *max_taps* entries are used.
        max_taps: Maximum number of recent taps to average over.

    Returns:
        Estimated BPM, or 0.0 if fewer than 2 taps are provided.
    """
    if len(tap_times) < 2:
        return 0.0

    recent = tap_times[-max_taps:]
    intervals = [
        recent[i] - recent[i - 1] for i in range(1, len(recent))
    ]
    avg_interval = sum(intervals) / len(intervals)
    if avg_interval <= 0:
        return 0.0
    return 60.0 / avg_interval

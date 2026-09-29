"""Metronome utilities for BPM calculation."""

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Beat intervals within this fraction of the median count towards the
# average. Wide enough for the beat model's 20 ms frame steps (0.44 s and
# 0.46 s at 133 BPM), narrow enough to drop the two halves of an interval
# split by an extra beat, or the double gap left by a missing one.
_INLIER_TOLERANCE = 0.1

# Beat intervals on each side of an interval that set its local tempo.
_LOCAL_HALF_WINDOW = 4

# Below this share of steady intervals, the beats around an interval are
# too scattered to give a tempo (a free-time ending, a count-off), and the
# previous tempo is kept.
_MIN_STEADY_SHARE = 0.5

# A local tempo this close to 2x or 4x (or 1/2, 1/4) of the tempo before
# it is an octave jump; one further than _MAX_TEMPO_RATIO from it, a leap.
_OCTAVE_TOLERANCE = 0.15
_MAX_OCTAVES = 2
_MAX_TEMPO_RATIO = 1.6

# A jump or leap that lasts at most this many beats (of the tempo before
# it) is the beat model briefly following eighth notes, every other beat,
# or noise: an octave jump is folded back and a leap shows the tempo
# before it. Anything longer is a section really played at that tempo (a
# slow intro, a double-time or half-time part) and is shown as detected.
_MAX_BRIEF_BEATS = 16


def tap_tempo(tap_times: list[float], max_taps: int = 8) -> float:
    """Calculate BPM from a list of tap timestamps.

    Args:
        tap_times: Monotonic timestamps in seconds (e.g. from
            ``time.monotonic()``).  Only the last *max_taps* entries are used.
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


def robust_beat_interval(intervals) -> float:
    """Return the typical gap between beats, ignoring outliers.

    The mean of the intervals within ``_INLIER_TOLERANCE`` of the median.
    The median alone snaps to the detector's frame grid (133 BPM reads as
    130.4 or 136.4); the mean of the inliers keeps the finer tempo, and
    leaving out the rest keeps an extra or missing beat from pulling it.
    Returns 0.0 when there is no positive interval.
    """
    values = np.asarray(intervals, dtype=np.float64)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    inliers = values[np.abs(values - median) <= _INLIER_TOLERANCE * median]
    if inliers.size == 0:
        return median
    return float(inliers.mean())


def _fold_octave(interval: float, reference: float) -> float:
    """Fold *interval* onto *reference* when it is near an octave of it."""
    ratio = interval / reference
    octave = round(math.log2(ratio))
    if octave == 0 or abs(octave) > _MAX_OCTAVES:
        return interval
    scale = 2.0 ** octave
    if abs(ratio / scale - 1.0) <= _OCTAVE_TOLERANCE:
        return interval / scale
    return interval


def _departs(interval: float, reference: float) -> bool:
    """True if *interval* is an octave jump or a leap from *reference*."""
    if _fold_octave(interval, reference) != interval:
        return True
    ratio = interval / reference
    return not 1.0 / _MAX_TEMPO_RATIO <= ratio <= _MAX_TEMPO_RATIO


def _windowed_intervals(intervals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust interval and steadiness for the window around each gap.

    Returns ``(local, steady)``: the mean of the inliers around each
    interval's window median, and whether at least ``_MIN_STEADY_SHARE``
    of the window's intervals are inliers.
    """
    half = _LOCAL_HALF_WINDOW
    padded = np.pad(intervals, half, constant_values=np.nan)
    windows = sliding_window_view(padded, 2 * half + 1)
    # Every window holds its own (finite) centre gap, so no row is all NaN.
    median = np.nanmedian(windows, axis=1)
    inlier = np.abs(windows - median[:, None]) <= (
        _INLIER_TOLERANCE * np.abs(median[:, None])
    )
    count = inlier.sum(axis=1)
    present = np.isfinite(windows).sum(axis=1)
    total = np.where(inlier, windows, 0.0).sum(axis=1)
    local = np.where(count > 0, total / np.maximum(count, 1), median)
    steady = (count >= _MIN_STEADY_SHARE * present) & (local > 0)
    return local, steady


def local_beat_tempi(positions, units_per_minute: float) -> np.ndarray:
    """Return the local tempo in BPM for each gap between adjacent beats.

    *positions* are ascending beat positions in any unit (seconds, sample
    frames); *units_per_minute* converts that unit to minutes (60 for
    seconds, ``60 * sample_rate`` for frames). Element ``i`` is the tempo
    around the gap from beat ``i`` to beat ``i + 1``.

    One gap is not a tempo: a busy riff tracked in eighth notes, a missed
    beat, or one extra beat turns a single gap into 2x or 0.5x the song
    tempo. Each gap instead takes the robust interval of the gaps around
    it (``_LOCAL_HALF_WINDOW`` on each side), so a real tempo change shows
    within a few beats while stray beats do not. Where the beats are too
    scattered to agree, the previous tempo is kept.

    A jump to about 2x or 0.5x the tempo before it (or a leap beyond
    ``_MAX_TEMPO_RATIO``) that lasts at most ``_MAX_BRIEF_BEATS`` beats is
    the model briefly following eighth notes or every other beat: it is
    folded back, or the tempo before it is kept. A longer one is a section
    played at that tempo, such as a slow intro or a half-time part, and
    reads as detected. The first tempo before any beat is the song's.
    """
    beats = np.asarray(positions, dtype=np.float64)
    if beats.size < 2:
        return np.zeros(0, dtype=np.float64)
    intervals = np.diff(beats)
    song_interval = robust_beat_interval(intervals)
    if song_interval <= 0:
        return np.zeros(intervals.size, dtype=np.float64)

    local, steady = _windowed_intervals(intervals)
    count = intervals.size
    result = np.empty(count, dtype=np.float64)
    previous = song_interval
    i = 0
    while i < count:
        if not steady[i]:
            result[i] = previous
            i += 1
            continue
        if not _departs(float(local[i]), previous):
            previous = result[i] = float(local[i])
            i += 1
            continue
        # A departure: find how long it lasts (scattered gaps inside it
        # do not end it), measured in beats of the tempo before it.
        end = i
        while end + 1 < count and (
            not steady[end + 1] or _departs(float(local[end + 1]), previous)
        ):
            end += 1
        while not steady[end]:
            end -= 1
        span = beats[end + 1] - beats[i]
        if span <= _MAX_BRIEF_BEATS * previous:
            for j in range(i, end + 1):
                result[j] = previous
                if steady[j]:
                    folded = _fold_octave(float(local[j]), previous)
                    if folded != float(local[j]):
                        result[j] = folded
        else:
            for j in range(i, end + 1):
                if steady[j]:
                    previous = float(local[j])
                result[j] = previous
        i = end + 1
    return units_per_minute / result

"""Experimental song-section detection on the bar grid (#133).

Finds where a song changes section (intro, verse, chorus, ...) from the
analysis stemma already runs: the chroma that chord detection computes,
plus a cheap timbre and loudness summary. Everything is measured per bar,
so every boundary is a detected downbeat and a loop set on a section
starts and ends on a bar line.

Method, kept to numpy and librosa:

1. Average the features over each bar of the beat grid, then over a
   few bars around it, so a section cycling through its progression
   reads as one steady thing; scale them by how much single bars vary.
2. Build a bar-by-bar self-similarity matrix and slide a checkerboard
   kernel along its diagonal (Foote novelty): the score peaks where the
   bars before and after a point are alike among themselves but unlike
   each other.
3. Keep novelty peaks at least ``MIN_SECTION_BARS`` apart.
4. Give sections that sound alike the same letter (A, B, A, B, C, ...).
   Naming them verse or chorus would need far more than this knows, so
   the letters only say which sections repeat.

Sections are ``(start_s, end_s, label)`` and tile the song from its
first bar line to the end of its last bar.
"""

from __future__ import annotations

import bisect
import statistics

import librosa
import numpy as np

# A section is at least this many bars; shorter changes are fills.
MIN_SECTION_BARS = 4
# Bars on each side of the checkerboard kernel's centre.
_KERNEL_BARS = 4
# Bars each bar's features are averaged over (a typical progression).
_CONTEXT_BARS = 4
# Songs (or bar grids) shorter than this are not split.
_MIN_SONG_BARS = 2 * MIN_SECTION_BARS
# A novelty peak counts when it stands this many standard deviations
# above the mean novelty, and above an absolute floor (novelty is 1 for a
# change between two halves with nothing in common, 0 for no change). The
# floor keeps a song that never changes from being split anyway: one 4-bar
# loop repeated scores at most about 0.2, while the changes in the demo
# songs and the test fixtures score 0.33 to 0.55.
_PEAK_STD = 0.25
_MIN_NOVELTY = 0.3
# Two sections share a letter when their mean features are at least this
# similar (``self_similarity``). A repeated verse or chorus scores 0.8 to
# 1.0; different sections of the demo songs at most about 0.6.
_SAME_LABEL_SIMILARITY = 0.7
# Without detected downbeats (the librosa fallback), a bar is this many
# beats.
_FALLBACK_BEATS_PER_BAR = 4
# Feature weights: harmony, timbre, loudness.
_CHROMA_WEIGHT = 1.0
_TIMBRE_WEIGHT = 1.0
_LOUDNESS_WEIGHT = 0.5
_N_MFCC = 13
# Squared feature distance that scores exp(-1) similarity. Distances are
# in units of single-bar spread, and context-averaged bars of one section
# sit well under a quarter of that apart.
_SIMILARITY_SCALE = 0.5


def bar_grid(
    beat_times: list[float],
    downbeat_times: list[float],
    duration: float,
) -> list[float]:
    """Bar lines to segment on, ending with the end of the last bar.

    Uses the downbeats, or every fourth beat when there are none. The
    last bar runs one median bar past the last bar line, clipped to the
    song. Returns ``[]`` when there is no usable grid.
    """
    lines = sorted(float(t) for t in downbeat_times if 0.0 <= t < duration)
    if len(lines) < 2:
        beats = sorted(float(t) for t in beat_times if 0.0 <= t < duration)
        lines = beats[::_FALLBACK_BEATS_PER_BAR]
    if len(lines) < 2:
        return []
    gaps = [b - a for a, b in zip(lines, lines[1:]) if b > a]
    if not gaps:
        return []
    end = min(duration, lines[-1] + statistics.median(gaps))
    if end > lines[-1]:
        lines.append(end)
    return lines


def _bar_means(features: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Mean of each feature row over each bar ``[frames[i], frames[i+1])``.

    Returns ``(n_bars, n_features)``. A bar shorter than a frame takes the
    frame it starts in.
    """
    n_frames = features.shape[1]
    rows = []
    for start, end in zip(frames[:-1], frames[1:]):
        start = int(min(max(start, 0), n_frames - 1))
        end = int(min(max(end, start + 1), n_frames))
        rows.append(features[:, start:end].mean(axis=1))
    return np.asarray(rows, dtype=np.float64)


def _smooth_bars(block: np.ndarray, width: int = _CONTEXT_BARS) -> np.ndarray:
    """Average each bar with its neighbours over a centred *width*-bar span.

    A section usually cycles through a chord progression a bar or two per
    chord, so single bars of one section differ as much as bars of two.
    Averaged over a progression's length, the section's bars look alike,
    and the window being centred keeps a change where it happens.
    """
    n = block.shape[0]
    if n == 0 or width <= 1:
        return block
    before = width // 2
    after = width - before
    padded = np.concatenate([
        np.repeat(block[:1], before, axis=0), block,
        np.repeat(block[-1:], after - 1, axis=0),
    ])
    sums = np.cumsum(np.vstack([np.zeros((1, block.shape[1])), padded]), axis=0)
    return (sums[width:width + n] - sums[:n]) / width


def _context_features(raw: np.ndarray, weight: float) -> np.ndarray:
    """Smooth per-bar features and scale them by the bar-to-bar spread.

    Each column is centred and divided by its spread across *single*
    bars, then by the square root of the column count, so a group's
    squared distances average over its columns and *weight* sets its
    share. Measured against single bars, a section's averaged features
    move only when the music really changes: a song that repeats one
    progression throughout stays near zero instead of having its noise
    blown up to full scale.
    """
    smoothed = _smooth_bars(raw)
    centred = smoothed - smoothed.mean(axis=0, keepdims=True)
    spread = raw.std(axis=0, keepdims=True)
    centred = centred / np.where(spread < 1e-9, 1.0, spread)
    return centred * np.sqrt(weight / raw.shape[1])


def bar_features(
    mono: np.ndarray,
    sr: int,
    bars: list[float],
    chroma: np.ndarray | None = None,
    hop_length: int = 4096,
) -> np.ndarray:
    """One row per bar: harmony, timbre, and loudness in context.

    Each bar is averaged with its neighbours (``_CONTEXT_BARS``) and
    scaled by how much single bars vary, so distances between rows are in
    units of ordinary bar-to-bar variation (see ``_context_features``).

    *chroma* is the ``(12, n_frames)`` chroma at *hop_length* that chord
    detection already computed; it is computed here when not given.
    """
    if chroma is None:
        chroma = librosa.feature.chroma_cqt(
            y=mono, sr=sr, hop_length=hop_length,
        )
    mfcc = librosa.feature.mfcc(
        y=mono, sr=sr, n_mfcc=_N_MFCC, hop_length=hop_length,
        n_fft=hop_length,
    )
    rms = librosa.feature.rms(
        y=mono, frame_length=hop_length, hop_length=hop_length,
    )
    loudness = np.log1p(100.0 * rms)
    n_frames = min(chroma.shape[1], mfcc.shape[1], loudness.shape[1])
    frames = librosa.time_to_frames(bars, sr=sr, hop_length=hop_length)
    frames = np.clip(frames, 0, n_frames)

    total = _CHROMA_WEIGHT + _TIMBRE_WEIGHT + _LOUDNESS_WEIGHT
    return np.hstack([
        _context_features(
            _bar_means(chroma[:, :n_frames], frames), _CHROMA_WEIGHT / total,
        ),
        # The first coefficient is overall level, which loudness covers.
        _context_features(
            _bar_means(mfcc[1:, :n_frames], frames), _TIMBRE_WEIGHT / total,
        ),
        _context_features(
            _bar_means(loudness[:, :n_frames], frames),
            _LOUDNESS_WEIGHT / total,
        ),
    ])


def self_similarity(features: np.ndarray) -> np.ndarray:
    """Gaussian similarity between every pair of bars, 1 for identical.

    A squared distance of ``_SIMILARITY_SCALE`` in ``bar_features``
    units scores ``exp(-1)``.
    """
    sq = np.sum(features ** 2, axis=1)
    dist = np.maximum(sq[:, None] + sq[None, :] - 2 * features @ features.T, 0)
    return np.exp(-dist / _SIMILARITY_SCALE)


def _checkerboard(half: int) -> np.ndarray:
    """A ``2*half`` square Gaussian-tapered checkerboard kernel."""
    signs = np.concatenate([-np.ones(half), np.ones(half)])
    kernel = np.outer(signs, signs)
    axis = (np.arange(2 * half) - half + 0.5) / half
    taper = np.exp(-2.0 * axis ** 2)
    return kernel * np.outer(taper, taper)


def novelty(ssm: np.ndarray, half: int = _KERNEL_BARS) -> np.ndarray:
    """Foote novelty at each bar line: high where the music changes.

    Entry ``i`` scores the boundary between bar ``i - 1`` and bar ``i``,
    scaled so two uniform halves with nothing in common score 1 and no
    change scores 0. The matrix is zero-padded, so scores near the ends
    see half a kernel.
    """
    n = ssm.shape[0]
    kernel = _checkerboard(half)
    padded = np.zeros((n + 2 * half, n + 2 * half))
    padded[half:half + n, half:half + n] = ssm
    scores = np.zeros(n + 1)
    for i in range(n + 1):
        window = padded[i:i + 2 * half, i:i + 2 * half]
        scores[i] = float(np.sum(window * kernel))
    return 2.0 * scores / float(np.abs(kernel).sum())


def pick_boundaries(
    scores: np.ndarray, min_bars: int = MIN_SECTION_BARS,
) -> list[int]:
    """Bar indices where a new section starts, strongest first wins.

    A peak must clear ``_MIN_NOVELTY`` and stand ``_PEAK_STD``
    deviations above the mean, sit at least *min_bars* from the song's
    first and last bar line, and at least *min_bars* from any stronger
    peak already chosen.
    """
    n_bars = len(scores) - 1
    inner = scores[min_bars:n_bars - min_bars + 1]
    if inner.size == 0:
        return []
    threshold = max(
        _MIN_NOVELTY, float(inner.mean() + _PEAK_STD * inner.std()),
    )
    candidates = [
        i for i in range(min_bars, n_bars - min_bars + 1)
        if scores[i] > threshold
        and scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1]
    ]
    chosen: list[int] = []
    for i in sorted(candidates, key=lambda j: -scores[j]):
        if all(abs(i - c) >= min_bars for c in chosen):
            chosen.append(i)
    return sorted(chosen)


def _label_name(index: int) -> str:
    if index < 26:
        return chr(ord("A") + index)
    return f"S{index + 1}"


def label_segments(
    features: np.ndarray, starts: list[int], ends: list[int],
) -> list[str]:
    """A letter per segment; segments that sound alike share one.

    Each segment is compared with the first segment of every letter so
    far, in song order, and takes the most similar letter at or above
    ``_SAME_LABEL_SIMILARITY``, or the next new letter.
    """
    means = [features[s:e].mean(axis=0) for s, e in zip(starts, ends)]
    representatives: list[np.ndarray] = []
    labels: list[str] = []
    for mean in means:
        best, best_sim = None, _SAME_LABEL_SIMILARITY
        for k, rep in enumerate(representatives):
            sim = float(self_similarity(np.vstack([mean, rep]))[0, 1])
            if sim >= best_sim:
                best, best_sim = k, sim
        if best is None:
            representatives.append(mean)
            best = len(representatives) - 1
        labels.append(_label_name(best))
    return labels


def detect_sections(
    mono: np.ndarray,
    sr: int,
    beat_times: list[float],
    downbeat_times: list[float],
    chroma: np.ndarray | None = None,
    hop_length: int = 4096,
) -> list[tuple[float, float, str]]:
    """Sections of a whole song as ``(start_s, end_s, label)``.

    Boundaries are bar lines from the beat grid. Returns ``[]`` when the
    song is too short, has no beat grid, or never clearly changes (one
    section would say nothing).
    """
    duration = len(mono) / sr if sr else 0.0
    bars = bar_grid(beat_times, downbeat_times, duration)
    n_bars = len(bars) - 1
    if n_bars < _MIN_SONG_BARS:
        return []
    features = bar_features(mono, sr, bars, chroma, hop_length)
    boundaries = pick_boundaries(novelty(self_similarity(features)))
    if not boundaries:
        return []
    starts = [0] + boundaries
    ends = starts[1:] + [n_bars]
    labels = label_segments(features, starts, ends)
    return [
        (bars[s], bars[e], label)
        for s, e, label in zip(starts, ends, labels)
    ]


def section_index_at(
    sections: list[tuple[float, float, str]], seconds: float,
) -> int | None:
    """Index of the section playing at *seconds*, or None outside them."""
    starts = [start for start, _, _ in sections]
    index = bisect.bisect_right(starts, seconds) - 1
    if index < 0 or seconds >= sections[index][1]:
        return None
    return index


def parse_sections(raw) -> list[tuple[float, float, str]]:
    """Sections from saved JSON data, dropping anything malformed."""
    sections: list[tuple[float, float, str]] = []
    if not isinstance(raw, list):
        return sections
    for item in raw:
        try:
            start, end, label = item
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if end > start >= 0.0:
            sections.append((start, end, str(label)))
    sections.sort()
    return sections

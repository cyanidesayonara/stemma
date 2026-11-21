"""Tests for experimental song-section detection (#133)."""

import numpy as np
import pytest

from src.song_sections import (
    MIN_SECTION_BARS,
    bar_grid,
    confirm_boundaries,
    detect_sections,
    label_segments,
    merge_repeats,
    novelty,
    parse_sections,
    pick_boundaries,
    section_index_at,
    self_similarity,
)

SR = 22050
BAR_S = 2.0  # 120 BPM in 4/4


def _tone(freqs, t, harmonics=1):
    out = np.zeros_like(t)
    for f in freqs:
        for h in range(1, harmonics + 1):
            out += np.sin(2 * np.pi * f * h * t) / h
    return out


def _section(kind: str, bars: int, rng) -> np.ndarray:
    """Eight bars of a quiet sine verse (A) or a loud bright chorus (B)."""
    n = int(bars * BAR_S * SR)
    t = np.arange(n) / SR
    bar = (t // BAR_S).astype(int) % 2
    if kind == "A":
        # C major and G major triads, soft and pure.
        chords = [(261.63, 329.63, 392.0), (196.0, 246.94, 293.66)]
        audio = np.where(
            bar == 0, _tone(chords[0], t), _tone(chords[1], t),
        ) * 0.2
    else:
        # F major and A minor, loud, with harmonics and noise hits.
        chords = [(174.61, 220.0, 261.63), (220.0, 261.63, 329.63)]
        audio = np.where(
            bar == 0, _tone(chords[0], t, 6), _tone(chords[1], t, 6),
        ) * 0.5
        beat = np.mod(t, BAR_S / 4)
        audio += 0.4 * np.exp(-40 * beat) * rng.standard_normal(n)
    return audio.astype(np.float32)


def _structured_song(layout: str = "ABAB", bars: int = 8):
    rng = np.random.default_rng(0)
    audio = np.concatenate([_section(k, bars, rng) for k in layout])
    n_bars = bars * len(layout)
    downbeats = [i * BAR_S for i in range(n_bars)]
    beats = [i * BAR_S / 4 for i in range(n_bars * 4)]
    return audio, beats, downbeats


def _uniform_loop(n_bars: int = 32):
    """One four-bar progression with a drum hit, repeated: no sections."""
    t = np.arange(int(n_bars * BAR_S * SR)) / SR
    bar = (t // BAR_S).astype(int) % 4
    roots = np.array([220.0, 174.61, 261.63, 196.0])[bar]
    audio = sum(
        np.sin(2 * np.pi * np.cumsum(roots * m) / SR) for m in (1, 1.26, 1.5)
    )
    audio += 0.3 * np.exp(-40 * np.mod(t, BAR_S / 4)) * (
        np.random.default_rng(0).standard_normal(t.size)
    )
    downbeats = [i * BAR_S for i in range(n_bars)]
    beats = [i * BAR_S / 4 for i in range(n_bars * 4)]
    return (audio / np.abs(audio).max()).astype(np.float32), beats, downbeats


@pytest.fixture(scope="module")
def abab():
    """(sections, downbeats) of the 32-bar verse/chorus fixture, once."""
    audio, beats, downbeats = _structured_song("ABAB")
    return detect_sections(audio, SR, beats, downbeats), downbeats


class TestBarGrid:
    def test_downbeats_plus_the_end_of_the_last_bar(self):
        assert bar_grid([], [0.5, 2.5, 4.5], 10.0) == [0.5, 2.5, 4.5, 6.5]

    def test_last_bar_clipped_to_the_song(self):
        assert bar_grid([], [0.0, 2.0, 4.0], 5.0) == [0.0, 2.0, 4.0, 5.0]

    def test_every_fourth_beat_without_downbeats(self):
        beats = [i * 0.5 for i in range(16)]
        assert bar_grid(beats, [], 100.0) == [0.0, 2.0, 4.0, 6.0, 8.0]

    def test_no_grid(self):
        assert bar_grid([], [], 10.0) == []
        assert bar_grid([1.0], [], 10.0) == []


class TestNovelty:
    def test_peaks_at_the_change(self):
        # Eight bars of one thing, then eight of another.
        features = np.vstack([
            np.tile([1.0, 0.0], (8, 1)), np.tile([0.0, 1.0], (8, 1)),
        ])
        scores = novelty(self_similarity(features))
        assert len(scores) == 17
        assert int(np.argmax(scores)) == 8

    def test_halves_with_nothing_in_common_score_one(self):
        features = np.vstack([
            np.tile([10.0, 0.0], (8, 1)), np.tile([-10.0, 0.0], (8, 1)),
        ])
        assert novelty(self_similarity(features))[8] == pytest.approx(1.0)

    def test_no_change_scores_zero(self):
        features = np.tile([1.0, 2.0], (16, 1))
        assert novelty(self_similarity(features))[8] == pytest.approx(0.0)

    def test_weak_peaks_are_not_boundaries(self):
        scores = np.full(33, 0.05)
        scores[16] = 0.25  # stands out, but under the absolute floor
        assert pick_boundaries(scores) == []

    def test_boundaries_keep_their_distance(self):
        scores = np.zeros(33)
        scores[8] = 5.0
        scores[10] = 4.0  # too close to the stronger peak at 8
        scores[16] = 3.0
        scores[24] = 4.0
        assert pick_boundaries(scores) == [8, 16, 24]

    def test_no_boundary_too_near_the_ends(self):
        scores = np.zeros(33)
        scores[2] = 9.0
        scores[31] = 9.0
        scores[16] = 1.0
        assert pick_boundaries(scores) == [16]


class TestConfirmBoundaries:
    def test_a_real_step_is_kept(self):
        rng = np.random.default_rng(0)
        rows = np.vstack([
            np.tile([1.0, 0.0], (16, 1)), np.tile([-1.0, 0.0], (16, 1)),
        ]) + 0.05 * rng.standard_normal((32, 2))
        assert confirm_boundaries(rows, [16]) == [16]

    def test_random_bars_are_not_boundaries(self):
        rows = np.random.default_rng(1).standard_normal((48, 6))
        assert confirm_boundaries(rows, [8, 16, 24, 32, 40]) == []


class TestLabels:
    def test_alike_segments_share_a_letter(self):
        a = np.tile([1.0, 0.0, 0.1], (8, 1))
        b = np.tile([0.0, 1.0, 0.1], (8, 1))
        c = np.tile([0.0, 0.0, 1.0], (8, 1))
        features = np.vstack([a, b, a, b, c])
        starts = [0, 8, 16, 24, 32]
        ends = [8, 16, 24, 32, 40]
        assert label_segments(features, starts, ends) == [
            "A", "B", "A", "B", "C",
        ]


def test_neighbours_with_one_letter_merge():
    sections = [
        (0.0, 8.0, "A"), (8.0, 16.0, "A"), (16.0, 24.0, "B"),
        (24.0, 32.0, "A"),
    ]
    assert merge_repeats(sections) == [
        (0.0, 16.0, "A"), (16.0, 24.0, "B"), (24.0, 32.0, "A"),
    ]
    assert merge_repeats([]) == []


class TestDetectSections:
    def test_finds_the_verse_chorus_structure(self, abab):
        sections, _ = abab
        starts = [round(s, 3) for s, _, _ in sections]
        assert starts == [0.0, 16.0, 32.0, 48.0]
        assert [label for _, _, label in sections] == ["A", "B", "A", "B"]
        assert sections[-1][1] == pytest.approx(64.0)

    def test_boundaries_are_bar_lines(self, abab):
        sections, downbeats = abab
        song_end = 64.0
        for start, end, _ in sections:
            assert start in downbeats
            assert end in downbeats or end == pytest.approx(song_end)
            assert end - start >= MIN_SECTION_BARS * BAR_S - 1e-6

    def test_sections_tile_the_bar_grid(self, abab):
        sections, _ = abab
        for (_, end, _), (start, _, _) in zip(sections, sections[1:]):
            assert end == start

    def test_sections_that_cycle_through_chords_each_bar(self):
        """Verse and chorus each change chord every bar; single bars of
        one section then differ as much as bars of two (render fixture:
        eight bars of verse, eight of chorus, four of verse, at 100 BPM)."""
        from scripts.render_ui_review import SAMPLE_RATE, synth_stems

        stems = synth_stems(4)
        mono = sum(s.mean(axis=1) for s in stems.values()).astype(np.float32)
        downbeats = [i * 2.4 for i in range(20)]
        sections = detect_sections(mono, SAMPLE_RATE, [], downbeats)
        assert [(round(s, 1), label) for s, _, label in sections] == [
            (0.0, "A"), (19.2, "B"), (38.4, "A"),
        ]

    @pytest.mark.parametrize("seed", range(6))
    def test_white_noise_has_no_sections(self, seed):
        """Bars that vary at random are not sections (#246 review)."""
        audio = (
            0.3 * np.random.default_rng(seed).standard_normal(SR * 120)
        ).astype(np.float32)
        downbeats = [i * BAR_S for i in range(60)]
        assert detect_sections(audio, SR, [], downbeats) == []

    def test_a_song_that_never_changes_has_none(self):
        audio, beats, downbeats = _uniform_loop()
        assert detect_sections(audio, SR, beats, downbeats) == []

    def test_too_short_or_no_grid_is_empty(self):
        audio, beats, downbeats = _structured_song("A", bars=6)
        assert detect_sections(audio, SR, beats, downbeats) == []
        long_audio, _, _ = _structured_song("AB")
        assert detect_sections(long_audio, SR, [], []) == []

    def test_reuses_given_chroma(self):
        audio, beats, downbeats = _structured_song("AB")
        chroma = np.zeros((12, 400))
        chroma[0] = 1.0
        # Flat harmony leaves timbre and loudness to find the change.
        sections = detect_sections(
            audio, SR, beats, downbeats, chroma=chroma,
            hop_length=int(SR * 32.0 / 400),
        )
        assert [round(s) for s, _, _ in sections] == [0, 16]


class TestQueries:
    SECTIONS = [(1.0, 9.0, "A"), (9.0, 17.0, "B"), (17.0, 25.0, "A")]

    @pytest.mark.parametrize("seconds, index", [
        (0.5, None), (1.0, 0), (8.99, 0), (9.0, 1), (24.9, 2), (25.0, None),
    ])
    def test_section_index_at(self, seconds, index):
        assert section_index_at(self.SECTIONS, seconds) == index

    def test_empty(self):
        assert section_index_at([], 3.0) is None

    def test_parse_sections_drops_malformed(self):
        raw = [[9, 17, "B"], [1, 9, "A"], ["x", 2, "C"], [5, 4, "D"], 7]
        assert parse_sections(raw) == [(1.0, 9.0, "A"), (9.0, 17.0, "B")]
        assert parse_sections("nope") == []

    def test_parse_sections_drops_overlaps(self):
        """Saved sections tile the song; one that starts inside another
        is corrupt and dropped (#246 review)."""
        raw = [[0, 1, "A"], [0.5, 3, "D"], [1, 2, "B"], [2, 4, "C"]]
        assert parse_sections(raw) == [
            (0.0, 1.0, "A"), (1.0, 2.0, "B"), (2.0, 4.0, "C"),
        ]

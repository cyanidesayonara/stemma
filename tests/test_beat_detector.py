"""Tests for src/beat_detector.py — BPM/key detection and confidence scoring."""

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from src.beat_detector import (
    DetectionResult,
    DetectionWorker,
    _bpm_confidence,
    _build_chord_templates,
    _detect_beats_librosa,
    _detect_chords,
    _detect_key,
    _detect_time_signature,
    _key_confidence,
    _peak_pick,
    _sigmoid,
    _viterbi_smooth,
    detect_bpm_and_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _click_track(bpm: float, duration: float, sr: int = 44100) -> np.ndarray:
    """Synthesise a mono click track at the given BPM."""
    n_samples = int(duration * sr)
    audio = np.zeros(n_samples, dtype=np.float32)
    interval = int(60.0 / bpm * sr)
    click_len = min(200, interval)
    t = np.arange(click_len, dtype=np.float32) / sr
    click = (np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 40)).astype(np.float32)
    pos = 0
    while pos + click_len <= n_samples:
        audio[pos:pos + click_len] += click
        pos += interval
    return audio


def _chord(freqs: list[float], duration: float, sr: int = 44100) -> np.ndarray:
    """Synthesise a chord from a list of frequencies."""
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    audio = np.zeros_like(t)
    for f in freqs:
        audio += np.sin(2 * np.pi * f * t)
    audio /= len(freqs)
    return audio


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------

class TestDetectionResult:
    def test_defaults(self):
        r = DetectionResult()
        assert r.bpm == 0.0
        assert r.key == ""
        assert r.beat_times == []
        assert r.time_signature == ""

    def test_fields(self):
        r = DetectionResult(bpm=120.0, key="C major", bpm_confidence="high")
        assert r.bpm == 120.0
        assert r.key == "C major"
        assert r.bpm_confidence == "high"


# ---------------------------------------------------------------------------
# Time signature detection
# ---------------------------------------------------------------------------

class TestDetectTimeSignature:
    def test_four_four(self):
        """4 beats per bar -> 4/4."""
        downbeats = [0.0, 2.0, 4.0, 6.0]
        beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
                 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
        assert _detect_time_signature(beats, downbeats) == "4/4"

    def test_three_four(self):
        """3 beats per bar -> 3/4."""
        downbeats = [0.0, 1.5, 3.0, 4.5]
        beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
        assert _detect_time_signature(beats, downbeats) == "3/4"

    def test_six_eight(self):
        """6 beats per bar -> 6/8."""
        downbeats = [0.0, 3.0, 6.0]
        beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
        assert _detect_time_signature(beats, downbeats) == "6/8"

    def test_no_downbeats(self):
        """No downbeats -> empty string (librosa fallback)."""
        assert _detect_time_signature([0.0, 0.5, 1.0], []) == ""

    def test_single_downbeat(self):
        """Only one downbeat -> not enough to measure a bar."""
        assert _detect_time_signature([0.0, 0.5, 1.0], [0.0]) == ""

    def test_no_beats(self):
        """No beats at all."""
        assert _detect_time_signature([], [0.0, 2.0]) == ""

    def test_unusual_meter(self):
        """5 beats per bar -> 5/4."""
        downbeats = [0.0, 2.5, 5.0]
        beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        assert _detect_time_signature(beats, downbeats) == "5/4"

    def test_unmapped_meter_fallback(self):
        """Meter not in the standard map falls back to N/4."""
        downbeats = [0.0, 4.0, 8.0]
        beats = [float(i) * 0.5 for i in range(16)]  # 8 beats per bar
        assert _detect_time_signature(beats, downbeats) == "8/4"

    def test_robust_to_outlier_bars(self):
        """Median ignores outlier bars (e.g. partial last bar)."""
        # 3 bars of 4/4 plus a partial bar of 2 beats
        downbeats = [0.0, 2.0, 4.0, 6.0, 7.0]
        beats = [0.0, 0.5, 1.0, 1.5,
                 2.0, 2.5, 3.0, 3.5,
                 4.0, 4.5, 5.0, 5.5,
                 6.0, 6.5,
                 7.0, 7.5]
        assert _detect_time_signature(beats, downbeats) == "4/4"


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------

class TestSigmoid:
    def test_zero(self):
        assert abs(_sigmoid(np.array([0.0]))[0] - 0.5) < 1e-6

    def test_large_positive(self):
        assert _sigmoid(np.array([100.0]))[0] > 0.99

    def test_large_negative(self):
        assert _sigmoid(np.array([-100.0]))[0] < 0.01


# ---------------------------------------------------------------------------
# Peak picker
# ---------------------------------------------------------------------------

class TestPeakPick:
    def test_simple_peaks(self):
        logits = np.array([0.0, 0.1, 0.8, 0.1, 0.0, 0.1, 0.9, 0.1, 0.0])
        peaks = _peak_pick(logits, threshold=0.5, min_distance=2)
        assert peaks == [2, 6]

    def test_min_distance(self):
        logits = np.array([0.0, 0.8, 0.1, 0.9, 0.0])
        peaks = _peak_pick(logits, threshold=0.5, min_distance=4)
        assert len(peaks) == 1

    def test_below_threshold(self):
        logits = np.array([0.0, 0.2, 0.3, 0.2, 0.0])
        peaks = _peak_pick(logits, threshold=0.5, min_distance=1)
        assert peaks == []

    def test_empty(self):
        peaks = _peak_pick(np.array([]), threshold=0.5, min_distance=1)
        assert peaks == []


# ---------------------------------------------------------------------------
# BPM confidence
# ---------------------------------------------------------------------------

class TestBpmConfidence:
    def test_high_confidence(self):
        # Perfectly regular beats at 120 BPM.
        times = [i * 0.5 for i in range(20)]
        assert _bpm_confidence(times) == "high"

    def test_medium_confidence(self):
        # Slightly irregular beats (CV ~0.1).
        rng = np.random.RandomState(42)
        base = np.arange(20) * 0.5
        jitter = rng.normal(0, 0.05, size=20)
        times = (base + jitter).tolist()
        assert _bpm_confidence(times) == "medium"

    def test_low_confidence_few_beats(self):
        assert _bpm_confidence([0.0, 0.5]) == "low"

    def test_low_confidence_irregular(self):
        # Highly irregular intervals.
        times = [0.0, 0.5, 1.5, 1.8, 3.5, 4.0]
        assert _bpm_confidence(times) == "low"


# ---------------------------------------------------------------------------
# Key confidence
# ---------------------------------------------------------------------------

class TestKeyConfidence:
    def test_high(self):
        assert _key_confidence(0.90) == "high"

    def test_medium(self):
        assert _key_confidence(0.78) == "medium"

    def test_low(self):
        assert _key_confidence(0.50) == "low"

    def test_boundary(self):
        assert _key_confidence(0.85) == "medium"
        assert _key_confidence(0.70) == "low"


# ---------------------------------------------------------------------------
# Key detection
# ---------------------------------------------------------------------------

class TestDetectKey:
    def test_c_major_chord(self):
        # C4 + E4 + G4 — should detect C major.
        audio = _chord([261.63, 329.63, 392.00], duration=5.0)
        key, corr = _detect_key(audio, sr=44100)
        assert "C" in key
        assert corr > 0.5

    def test_a_minor_chord(self):
        # A3 + C4 + E4 — should detect A minor.
        audio = _chord([220.00, 261.63, 329.63], duration=5.0)
        key, corr = _detect_key(audio, sr=44100)
        assert "A" in key or "minor" in key
        assert corr > 0.3

    def test_silence(self):
        audio = np.zeros(44100 * 3, dtype=np.float32)
        key, corr = _detect_key(audio, sr=44100)
        assert key == ""
        assert corr == 0.0


# ---------------------------------------------------------------------------
# Librosa beat detection
# ---------------------------------------------------------------------------

class TestDetectBeatsLibrosa:
    def test_120bpm_click(self):
        audio = _click_track(120.0, duration=10.0)
        beats, downbeats, bpm = _detect_beats_librosa(audio, sr=44100)
        # Allow ±15% tolerance for librosa.
        assert 100 < bpm < 140, f"Expected ~120 BPM, got {bpm}"
        assert len(beats) > 5

    def test_returns_no_downbeats(self):
        audio = _click_track(100.0, duration=8.0)
        _, downbeats, _ = _detect_beats_librosa(audio, sr=44100)
        assert downbeats == []


# ---------------------------------------------------------------------------
# High-level detect_bpm_and_key
# ---------------------------------------------------------------------------

class TestDetectBpmAndKey:
    def test_empty_stems(self):
        result = detect_bpm_and_key({}, 44100)
        assert result.bpm == 0.0

    def test_short_audio(self):
        # 1 second — below _MIN_DURATION.
        short = np.zeros((44100, 2), dtype=np.float32)
        result = detect_bpm_and_key({"stem": short}, 44100)
        assert result.bpm == 0.0

    def test_click_track_stereo(self):
        mono = _click_track(120.0, duration=10.0)
        stereo = np.column_stack([mono, mono])
        result = detect_bpm_and_key({"drums": stereo}, 44100)
        assert 80 < result.bpm < 160
        assert result.bpm_confidence in ("high", "medium", "low")

    def test_no_model_uses_librosa(self):
        mono = _click_track(100.0, duration=8.0)
        stereo = np.column_stack([mono, mono])
        result = detect_bpm_and_key(
            {"drums": stereo}, 44100, model_path="/nonexistent.onnx",
        )
        # Should still work via librosa fallback.
        assert result.bpm > 0

    def test_key_populated(self):
        chord = _chord([261.63, 329.63, 392.00], duration=8.0)
        stereo = np.column_stack([chord, chord])
        result = detect_bpm_and_key({"pad": stereo}, 44100)
        assert result.key != ""
        assert result.key_confidence in ("high", "medium", "low")

    def test_ab_region_slicing(self):
        """Detection with start_sec/end_sec should only analyse that region."""
        mono = _click_track(120.0, duration=15.0)
        stereo = np.column_stack([mono, mono])
        result = detect_bpm_and_key(
            {"drums": stereo}, 44100, start_sec=2.0, end_sec=12.0,
        )
        assert 80 < result.bpm < 160

    def test_ab_region_too_short(self):
        """A-B region shorter than _MIN_DURATION returns empty result."""
        mono = _click_track(120.0, duration=10.0)
        stereo = np.column_stack([mono, mono])
        result = detect_bpm_and_key(
            {"drums": stereo}, 44100, start_sec=0.0, end_sec=1.0,
        )
        assert result.bpm == 0.0


# ---------------------------------------------------------------------------
# DetectionWorker
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


class TestDetectionWorker:
    @pytest.fixture(autouse=True)
    def _app(self, app):
        """Ensure a QApplication exists for QThread."""

    def test_worker_completes(self):
        mono = _click_track(120.0, duration=8.0)
        stereo = np.column_stack([mono, mono])
        stems = {"drums": stereo}

        results = []
        worker = DetectionWorker(stems, 44100)
        worker.completed.connect(results.append)
        worker.start()
        worker.wait(30_000)
        QCoreApplication.processEvents()

        assert len(results) == 1
        assert results[0].bpm > 0

    def test_worker_error_on_empty(self):
        errors = []
        results = []
        worker = DetectionWorker({}, 44100)
        worker.completed.connect(results.append)
        worker.error.connect(errors.append)
        worker.start()
        worker.wait(10_000)
        QCoreApplication.processEvents()

        # Empty stems should return an empty result, not an error.
        assert len(results) == 1
        assert results[0].bpm == 0.0


# ---------------------------------------------------------------------------
# Chord templates
# ---------------------------------------------------------------------------

class TestBuildChordTemplates:
    def test_template_count(self):
        """12 roots x 5 qualities = 60 templates."""
        templates = _build_chord_templates()
        assert len(templates) == 60

    def test_templates_normalised(self):
        """Every template should be unit-normalised."""
        for label, vec in _build_chord_templates():
            norm = float(np.linalg.norm(vec))
            assert abs(norm - 1.0) < 1e-6, f"{label}: norm={norm}"

    def test_c_major_template(self):
        """C major template should have energy on C, E, G (indices 0, 4, 7)."""
        templates = _build_chord_templates()
        c_major = [v for l, v in templates if l == "C"][0]
        assert c_major[0] > 0   # C
        assert c_major[4] > 0   # E
        assert c_major[7] > 0   # G
        assert c_major[1] == 0  # C#


# ---------------------------------------------------------------------------
# Viterbi smoothing
# ---------------------------------------------------------------------------

class TestViterbiSmooth:
    def test_empty(self):
        assert _viterbi_smooth([], 10) == []

    def test_stable_sequence(self):
        """Already-stable sequence should remain unchanged."""
        labels = [0] * 10 + [1] * 10
        smoothed = _viterbi_smooth(labels, 5)
        assert smoothed == labels

    def test_removes_isolated_spike(self):
        """A single-frame spike should be smoothed away."""
        labels = [0] * 5 + [3] + [0] * 5
        smoothed = _viterbi_smooth(labels, 5)
        assert smoothed[5] == 0  # spike removed

    def test_preserves_real_change(self):
        """A sustained change should be preserved."""
        labels = [0] * 20 + [2] * 20
        smoothed = _viterbi_smooth(labels, 5)
        # The bulk of each segment should match.
        assert smoothed[5] == 0
        assert smoothed[35] == 2


# ---------------------------------------------------------------------------
# Chord detection
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestDetectChords:
    def test_c_major_chord(self):
        """A sustained C major chord should be detected as C or C-related."""
        audio = _chord([261.63, 329.63, 392.00], duration=5.0)
        chords = _detect_chords(audio, sr=44100)
        assert len(chords) >= 1
        # First chord should be C-something.
        assert chords[0][1].startswith("C")

    def test_chord_segments_sorted(self):
        """Chord onsets should be in ascending time order."""
        audio = _chord([261.63, 329.63, 392.00], duration=5.0)
        chords = _detect_chords(audio, sr=44100)
        times = [t for t, _ in chords]
        assert times == sorted(times)

    def test_silence_returns_chords(self):
        """Even silence should produce some output (possibly a single chord)."""
        audio = np.zeros(44100 * 4, dtype=np.float32)
        chords = _detect_chords(audio, sr=44100)
        # Should not crash; may return empty or a single dim chord.
        assert isinstance(chords, list)

    def test_two_chord_sequence(self):
        """Two distinct chords played sequentially should produce at least 2 segments."""
        sr = 44100
        c_major = _chord([261.63, 329.63, 392.00], duration=3.0, sr=sr)
        a_minor = _chord([220.00, 261.63, 329.63], duration=3.0, sr=sr)
        audio = np.concatenate([c_major, a_minor])
        chords = _detect_chords(audio, sr=sr)
        # Should detect at least a chord change somewhere.
        assert len(chords) >= 2
        labels = [c for _, c in chords]
        # The set of unique labels should have more than 1 entry.
        assert len(set(labels)) >= 2

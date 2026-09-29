"""Tests for src/beat_detector.py — BPM/key detection and confidence scoring."""

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from src.beat_detector import (
    DetectionResult,
    DetectionWorker,
    _bpm_confidence,
    _bt_chunked_inference,
    _bt_peaks,
    _bt_spectrogram,
    _build_chord_templates,
    _detect_beats_librosa,
    _detect_beats_onnx,
    _detect_chords,
    _detect_key,
    _detect_time_signature,
    _key_confidence,
    _snap_to_beats,
    _viterbi_smooth,
    detect_bpm_and_key,
    transpose_chord,
    transpose_key,
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
# beat_this input spectrogram
# ---------------------------------------------------------------------------

def _hz_to_slaney_mel(freq: np.ndarray) -> np.ndarray:
    """Slaney mel scale: linear below 1 kHz, logarithmic above."""
    freq = np.asarray(freq, dtype=np.float64)
    f_sp = 200.0 / 3
    min_log_mel = 1000.0 / f_sp
    logstep = np.log(6.4) / 27.0
    log_part = min_log_mel + np.log(np.maximum(freq, 1e-10) / 1000.0) / logstep
    return np.where(freq >= 1000.0, log_part, freq / f_sp)


def _slaney_mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    f_sp = 200.0 / 3
    min_log_mel = 1000.0 / f_sp
    logstep = np.log(6.4) / 27.0
    log_part = 1000.0 * np.exp(logstep * (mels - min_log_mel))
    return np.where(mels >= min_log_mel, log_part, mels * f_sp)


def _reference_log_mel(audio: np.ndarray) -> np.ndarray:
    """NumPy port of beat_this ``LogMelSpect`` (CPJKU/beat_this).

    That is torchaudio ``MelSpectrogram(sample_rate=22050, n_fft=1024,
    hop_length=441, f_min=30, f_max=11000, n_mels=128, mel_scale="slaney",
    normalized="frame_length", power=1)`` followed by ``log1p(1000 * x)``.
    torchaudio's defaults fill in the rest: a periodic Hann window,
    centred frames with reflect padding, and triangular filters with no
    area normalisation (``norm=None``).
    """
    sr, n_fft, hop, n_mels = 22050, 1024, 441, 128
    padded = np.pad(audio.astype(np.float64), n_fft // 2, mode="reflect")
    n_frames = 1 + (len(padded) - n_fft) // hop
    index = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    magnitude = np.abs(np.fft.rfft(padded[index] * window, axis=1))
    magnitude /= np.sqrt(n_fft)  # normalized="frame_length"

    f_pts = _slaney_mel_to_hz(np.linspace(
        _hz_to_slaney_mel(30.0), _hz_to_slaney_mel(11000.0), n_mels + 2,
    ))
    all_freqs = np.linspace(0, sr // 2, n_fft // 2 + 1)
    f_diff = np.diff(f_pts)
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    filters = np.maximum(0.0, np.minimum(down, up))  # (freqs, mels)
    return np.log1p(1000.0 * (magnitude @ filters))


def _test_signal(seconds: float = 1.5, sr: int = 22050) -> np.ndarray:
    """Deterministic test audio: two tones, a chirp, and clicks."""
    t = np.arange(int(seconds * sr)) / sr
    audio = 0.4 * np.sin(2 * np.pi * 220.0 * t)
    audio += 0.2 * np.sin(2 * np.pi * 3150.0 * t)
    audio += 0.1 * np.sin(2 * np.pi * (100.0 * t + 1500.0 * t ** 2))
    for start in np.arange(0.1, seconds, 0.25):
        tail = t[t >= start] - start
        audio[t >= start] += 0.5 * np.sin(2 * np.pi * 1000.0 * tail) * np.exp(
            -tail * 60.0,
        )
    return audio.astype(np.float32)


class TestBtSpectrogram:
    """The model only tracks beats well on the input it was trained on."""

    def test_matches_the_beat_this_preprocessing(self):
        audio = _test_signal()
        spec = _bt_spectrogram(audio)
        assert spec.dtype == np.float32
        assert spec.shape == (1 + len(audio) // 441, 128)
        np.testing.assert_allclose(
            spec, _reference_log_mel(audio), rtol=0, atol=1e-4,
        )

    def test_edges_use_reflect_padding(self):
        """torchaudio pads centred frames by reflection, not with zeros."""
        audio = _test_signal()
        ref = _reference_log_mel(audio)
        spec = _bt_spectrogram(audio)
        np.testing.assert_allclose(spec[0], ref[0], rtol=0, atol=1e-4)
        np.testing.assert_allclose(spec[-1], ref[-1], rtol=0, atol=1e-4)


# ---------------------------------------------------------------------------
# beat_this chunked inference
# ---------------------------------------------------------------------------

class _EchoSession:
    """Fake ONNX session: beat logits echo mel bin 0, downbeats bin 1."""

    def __init__(self):
        self.chunks: list[np.ndarray] = []

    def get_inputs(self):
        return [SimpleNamespace(name="spect")]

    def run(self, _output_names, feeds):
        chunk = feeds["spect"]
        self.chunks.append(chunk)
        return [chunk[:, :, 0].copy(), chunk[:, :, 1].copy()]


def _ramp_spec(n_frames: int) -> np.ndarray:
    """Spectrogram whose first two bins number the frames (0 = padding)."""
    spec = np.zeros((n_frames, 128), dtype=np.float32)
    spec[:, 0] = np.arange(1, n_frames + 1)
    spec[:, 1] = -np.arange(1, n_frames + 1)
    return spec


class TestBtChunkedInference:
    @pytest.mark.parametrize("n_frames", [150, 1488, 1500, 2977, 4000])
    def test_each_frame_gets_its_own_prediction(self, n_frames):
        session = _EchoSession()
        beat, downbeat = _bt_chunked_inference(_ramp_spec(n_frames), session)
        expected = np.arange(1, n_frames + 1)
        np.testing.assert_array_equal(beat, expected)
        np.testing.assert_array_equal(downbeat, -expected)

    @pytest.mark.parametrize("n_frames", [150, 4000])
    def test_chunks_have_the_model_size(self, n_frames):
        session = _EchoSession()
        _bt_chunked_inference(_ramp_spec(n_frames), session)
        assert all(c.shape == (1, 1500, 128) for c in session.chunks)

    @pytest.mark.parametrize("n_frames", [1488, 2977, 4000])
    def test_last_chunk_is_moved_back_instead_of_padded(self, n_frames):
        """Like beat_this, no chunk is mostly silence: the last one ends
        at the end of the song and only the 6-frame borders are padded."""
        session = _EchoSession()
        _bt_chunked_inference(_ramp_spec(n_frames), session)
        for chunk in session.chunks:
            padding = int(np.sum(chunk[0, :, 0] == 0))
            assert padding <= 2 * 6


# ---------------------------------------------------------------------------
# beat_this peak picking
# ---------------------------------------------------------------------------

def _logits(n_frames: int, peaks: dict[int, float]) -> np.ndarray:
    logits = np.full(n_frames, -5.0, dtype=np.float32)
    for frame, value in peaks.items():
        logits[frame] = value
    return logits


class TestBtPeaks:
    """beat_this's minimal postprocessing: logit above 0 (probability
    above 0.5) and the highest within +-3 frames (70 ms)."""

    def test_positive_local_maxima_are_beats(self):
        peaks = _bt_peaks(_logits(40, {5: 2.0, 15: 0.5, 30: 3.0}))
        assert peaks.tolist() == [5, 15, 30]

    def test_probability_half_or_below_is_not_a_beat(self):
        # -0.4 is probability 0.4, which the old 0.3 threshold accepted.
        assert _bt_peaks(_logits(20, {5: 0.0, 12: -0.4})).size == 0

    def test_only_the_highest_peak_within_three_frames(self):
        peaks = _bt_peaks(_logits(40, {10: 2.0, 13: 1.0, 20: 1.0, 24: 2.0}))
        assert peaks.tolist() == [10, 20, 24]

    def test_adjacent_equal_peaks_merge(self):
        peaks = _bt_peaks(_logits(20, {10: 2.0, 11: 2.0}))
        assert peaks.tolist() == [10.5]

    def test_empty(self):
        assert _bt_peaks(np.array([], dtype=np.float32)).size == 0


class TestSnapToBeats:
    def test_downbeats_move_to_the_nearest_beat(self):
        beats = [0.5, 1.0, 1.5, 2.0, 2.5]
        assert _snap_to_beats([0.52, 2.46], beats) == [0.5, 2.5]

    def test_two_downbeats_on_one_beat_are_kept_once(self):
        assert _snap_to_beats([0.98, 1.02], [0.5, 1.0, 1.5]) == [1.0]

    def test_no_beats_leaves_downbeats(self):
        assert _snap_to_beats([1.0, 3.0], []) == [1.0, 3.0]


class TestDetectBeatsOnnx:
    def test_tempo_is_finer_than_the_frame_grid(self, monkeypatch):
        """133 BPM on 20 ms frames alternates 22 and 23 frame gaps. The
        median gap read 136.4 BPM; the mean of the steady gaps is 133.3."""
        n_frames = 1000
        beat_frames = [round(i * 22.5) for i in range(40)]
        down_frames = [f + 1 for f in beat_frames[::4]]
        monkeypatch.setattr(
            "src.beat_detector.create_onnx_session", lambda _path: object(),
        )
        monkeypatch.setattr(
            "src.beat_detector._bt_chunked_inference",
            lambda _spec, _session: (
                _logits(n_frames, {f: 3.0 for f in beat_frames}),
                _logits(n_frames, {f: 3.0 for f in down_frames}),
            ),
        )
        audio = np.zeros(22050 * 20, dtype=np.float32)
        beats, downbeats, bpm = _detect_beats_onnx(audio, 22050, "model")

        assert beats == pytest.approx([f / 50 for f in beat_frames])
        assert downbeats == pytest.approx(beats[::4])
        assert bpm == pytest.approx(133.33, abs=0.3)


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

    def test_ab_region_timestamps_use_actual_absolute_slice_start(
        self, monkeypatch,
    ):
        """Region-relative detector times are shifted into song time."""
        sample_rate = 10
        stereo = np.ones((100, 2), dtype=np.float32)
        monkeypatch.setattr(
            "src.beat_detector._detect_beats_librosa",
            lambda *_args: (
                [0.25, 0.75, 1.25, 1.75],
                [0.25, 1.25],
                120.0,
            ),
        )
        monkeypatch.setattr(
            "src.beat_detector._detect_key",
            lambda *_args: ("C major", 0.9),
        )
        monkeypatch.setattr(
            "src.beat_detector._detect_chords",
            lambda *_args: [(0.5, "C"), (1.5, "G")],
        )

        result = detect_bpm_and_key(
            {"mix": stereo},
            sample_rate,
            start_sec=2.75,
            end_sec=8.0,
        )

        # int(2.75 * 10) selects frame 27, so the actual offset is 2.7 s.
        assert result.beat_times == pytest.approx([2.95, 3.45, 3.95, 4.45])
        assert result.downbeat_times == pytest.approx([2.95, 3.95])
        assert [time for time, _ in result.chord_sequence] == pytest.approx(
            [3.2, 4.2]
        )
        assert result.bpm == 120.0
        assert result.key == "C major"

    def test_ab_region_negative_start_clamps_timestamp_offset(
        self, monkeypatch,
    ):
        """A negative requested start uses the actual zero slice boundary."""
        stereo = np.ones((60, 2), dtype=np.float32)
        monkeypatch.setattr(
            "src.beat_detector._detect_beats_librosa",
            lambda *_args: ([0.5], [0.5], 100.0),
        )
        monkeypatch.setattr(
            "src.beat_detector._detect_key",
            lambda *_args: ("", 0.0),
        )
        monkeypatch.setattr(
            "src.beat_detector._detect_chords",
            lambda *_args: [(0.75, "Am")],
        )

        result = detect_bpm_and_key(
            {"mix": stereo},
            10,
            start_sec=-1.0,
            end_sec=4.0,
        )

        assert result.beat_times == [0.5]
        assert result.downbeat_times == [0.5]
        assert result.chord_sequence == [(0.75, "Am")]

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
        assert worker.wait(30_000), "detection worker did not finish"
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
        assert worker.wait(10_000), "detection worker did not finish"
        QCoreApplication.processEvents()

        # Empty stems should return an empty result, not an error.
        assert len(results) == 1
        assert results[0].bpm == 0.0


# ---------------------------------------------------------------------------
# Chord templates
# ---------------------------------------------------------------------------

class TestBuildChordTemplates:
    def test_template_count(self):
        """12 roots x 2 qualities = 24 templates."""
        templates = _build_chord_templates()
        assert len(templates) == 24

    def test_templates_normalised(self):
        """Every template should be unit-normalised."""
        for label, vec in _build_chord_templates():
            norm = float(np.linalg.norm(vec))
            assert abs(norm - 1.0) < 1e-6, f"{label}: norm={norm}"

    def test_c_major_template(self):
        """C major template should have energy on C, E, G (indices 0, 4, 7)."""
        templates = _build_chord_templates()
        c_major = [values for label, values in templates if label == "C"][0]
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


# ---------------------------------------------------------------------------
# transpose_key
# ---------------------------------------------------------------------------

class TestTransposeKey:
    def test_zero_steps_is_identity(self):
        assert transpose_key("A minor", 0) == "A minor"

    def test_up_two_semitones_major(self):
        assert transpose_key("C major", 2) == "D major"

    def test_up_one_semitone_minor(self):
        assert transpose_key("A minor", 1) == "Bb minor"

    def test_down_semitone(self):
        assert transpose_key("C major", -1) == "B major"

    def test_wrap_around_twelve(self):
        assert transpose_key("C major", 12) == "C major"
        assert transpose_key("C major", -12) == "C major"

    def test_flat_spelling_accepted(self):
        # Detector uses "Bb"; ensure aliases (e.g. A#) also parse.
        assert transpose_key("A# minor", 1) == "B minor"

    def test_empty_string_unchanged(self):
        assert transpose_key("", 3) == ""

    def test_unparseable_unchanged(self):
        assert transpose_key("nonsense", 2) == "nonsense"
        assert transpose_key("C", 2) == "C"
        assert transpose_key("Q major", 2) == "Q major"

    def test_mode_preserved(self):
        assert transpose_key("F# minor", 3).endswith("minor")
        assert transpose_key("F major", 5).endswith("major")

    def test_case_insensitive_mode(self):
        assert transpose_key("C MAJOR", 2) == "D major"


# ---------------------------------------------------------------------------
# transpose_chord (#174)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chord, steps, expected", [
    ("C", 0, "C"),
    ("C", -2, "Bb"),
    ("Am", 3, "Cm"),
    ("B", 1, "C"),
    ("F#m", -1, "Fm"),
    ("Eb", 12, "Eb"),
    ("", 2, ""),
    ("N.C.", 2, "N.C."),
])
def test_transpose_chord_follows_the_pitch_shift(chord, steps, expected):
    """Chord labels shift with pitch, spelled like the key badge."""
    assert transpose_chord(chord, steps) == expected

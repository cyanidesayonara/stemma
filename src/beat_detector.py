"""Automatic BPM, beat, and key detection.

Uses the beat_this ONNX model (MIT license, ISMIR 2024) for high-accuracy
beat tracking when available, with a librosa fallback. Key detection uses
chroma features and the Krumhansl-Schmuckler algorithm.

The ``DetectionWorker`` QThread runs analysis in the background, emitting
progress updates and a ``DetectionResult`` on completion.
"""

import dataclasses
import math
import os

import librosa
import numpy as np
from PySide6.QtCore import QThread, Signal


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DetectionResult:
    """Container for beat/key detection results."""

    bpm: float = 0.0
    bpm_confidence: str = ""        # "high" / "medium" / "low"
    key: str = ""                   # e.g. "C major", "A minor"
    key_confidence: str = ""        # "high" / "medium" / "low"
    beat_times: list[float] = dataclasses.field(default_factory=list)
    downbeat_times: list[float] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Beat detection — beat_this ONNX
# ---------------------------------------------------------------------------

# Preprocessing constants matching the beat_this model.
_BT_SR = 22050
_BT_N_FFT = 1024
_BT_HOP = 441          # 50 fps at 22050 Hz
_BT_N_MELS = 128
_BT_FMIN = 30.0
_BT_FMAX = 10000.0

# Peak-picking parameters.
_BEAT_THRESHOLD = 0.3
_BEAT_MIN_DIST = 6      # ~120 ms at 50 fps


def _create_onnx_session(model_path: str):
    """Create an ONNX Runtime session with DML-first / CPU fallback."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    available = set(ort.get_available_providers())

    if "DmlExecutionProvider" in available:
        try:
            return ort.InferenceSession(
                model_path,
                sess_options=opts,
                providers=["DmlExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception:
            pass

    return ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def _peak_pick(logits: np.ndarray, threshold: float,
               min_distance: int) -> list[int]:
    """Simple peak picker — no scipy dependency.

    Returns frame indices where *logits* exceed *threshold* and are local
    maxima with at least *min_distance* frames between peaks.
    """
    peaks: list[int] = []
    last = -min_distance
    n = len(logits)
    for i in range(1, n - 1):
        if logits[i] < threshold:
            continue
        if logits[i] <= logits[i - 1] or logits[i] <= logits[i + 1]:
            continue
        if i - last < min_distance:
            continue
        peaks.append(i)
        last = i
    return peaks


def _detect_beats_onnx(
    audio_mono: np.ndarray, sr: int, model_path: str,
) -> tuple[list[float], list[float], float]:
    """Run beat_this ONNX model and return (beat_times, downbeat_times, bpm).

    The model expects a log-mel spectrogram at 22050 Hz with the parameters
    defined by the ``_BT_*`` constants above.
    """
    # Resample to model sample rate.
    if sr != _BT_SR:
        audio_mono = librosa.resample(audio_mono, orig_sr=sr, target_sr=_BT_SR)

    # Compute mel spectrogram (power → log scale).
    mel = librosa.feature.melspectrogram(
        y=audio_mono, sr=_BT_SR,
        n_fft=_BT_N_FFT, hop_length=_BT_HOP,
        n_mels=_BT_N_MELS, fmin=_BT_FMIN, fmax=_BT_FMAX,
    )
    log_mel = np.log1p(mel).astype(np.float32)  # (n_mels, frames)

    # Model expects (batch, frames, n_mels).
    spec = log_mel.T[np.newaxis, :, :]  # (1, frames, 128)

    session = _create_onnx_session(model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: spec})

    # Outputs: beat_logits (1, frames), downbeat_logits (1, frames).
    beat_logits = _sigmoid(outputs[0][0])
    downbeat_logits = _sigmoid(outputs[1][0]) if len(outputs) > 1 else None

    # Peak-pick beats.
    beat_frames = _peak_pick(beat_logits, _BEAT_THRESHOLD, _BEAT_MIN_DIST)
    fps = _BT_SR / _BT_HOP
    beat_times = [f / fps for f in beat_frames]

    # Peak-pick downbeats.
    downbeat_times: list[float] = []
    if downbeat_logits is not None:
        db_frames = _peak_pick(downbeat_logits, _BEAT_THRESHOLD, _BEAT_MIN_DIST)
        downbeat_times = [f / fps for f in db_frames]

    # Derive BPM from median inter-beat interval.
    bpm = 0.0
    if len(beat_times) >= 2:
        intervals = np.diff(beat_times)
        bpm = 60.0 / float(np.median(intervals))

    return beat_times, downbeat_times, bpm


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# ---------------------------------------------------------------------------
# Beat detection — librosa fallback
# ---------------------------------------------------------------------------

def _detect_beats_librosa(
    audio_mono: np.ndarray, sr: int,
) -> tuple[list[float], list[float], float]:
    """Fallback beat tracker using librosa's dynamic programming method."""
    tempo, beat_frames = librosa.beat.beat_track(y=audio_mono, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # librosa may return tempo as an array.
    if hasattr(tempo, "__len__"):
        bpm = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        bpm = float(tempo)

    return beat_times, [], bpm


# ---------------------------------------------------------------------------
# Key detection — Krumhansl-Schmuckler
# ---------------------------------------------------------------------------

# Krumhansl key profiles (Krumhansl & Kessler 1982).
_MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
], dtype=np.float64)

_MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
], dtype=np.float64)

_KEY_NAMES = ["C", "C#", "D", "Eb", "E", "F",
              "F#", "G", "Ab", "A", "Bb", "B"]


def _detect_key(
    audio_mono: np.ndarray, sr: int,
) -> tuple[str, float]:
    """Detect musical key using chroma CQT and Krumhansl-Schmuckler.

    Returns (key_name, correlation) where key_name is e.g. "C major"
    and correlation is the Pearson r (0–1 range, higher = more confident).
    """
    chroma = librosa.feature.chroma_cqt(y=audio_mono, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)  # (12,)

    if np.max(chroma_avg) < 1e-9:
        return "", 0.0

    best_key = ""
    best_corr = -1.0

    for shift in range(12):
        rotated = np.roll(chroma_avg, -shift)
        for profile, mode in [(_MAJOR_PROFILE, "major"), (_MINOR_PROFILE, "minor")]:
            corr = float(np.corrcoef(rotated, profile)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_key = f"{_KEY_NAMES[shift]} {mode}"

    return best_key, max(0.0, best_corr)


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _bpm_confidence(beat_times: list[float]) -> str:
    """Rate BPM confidence from inter-beat interval consistency."""
    if len(beat_times) < 4:
        return "low"
    intervals = np.diff(beat_times)
    mean_iv = float(np.mean(intervals))
    if mean_iv <= 0:
        return "low"
    cv = float(np.std(intervals)) / mean_iv
    if cv < 0.05:
        return "high"
    if cv < 0.15:
        return "medium"
    return "low"


def _key_confidence(correlation: float) -> str:
    """Rate key confidence from Krumhansl correlation strength."""
    if correlation > 0.85:
        return "high"
    if correlation > 0.70:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# High-level detection function
# ---------------------------------------------------------------------------

_MIN_DURATION = 3.0  # seconds


def detect_bpm_and_key(
    stems: dict[str, np.ndarray],
    sample_rate: int,
    model_path: str | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> DetectionResult:
    """Analyse loaded stems and return tempo, key, and beat grid.

    Mixes all stems to mono, runs beat detection (ONNX if *model_path*
    exists, else librosa fallback), and detects key via chroma analysis.

    If *start_sec* and *end_sec* are given, only the audio within that
    time range is analysed (useful for A-B loop region detection).
    """
    if not stems:
        return DetectionResult()

    # Mix all stems to mono.
    arrays = list(stems.values())
    max_len = max(a.shape[0] for a in arrays)
    mono = np.zeros(max_len, dtype=np.float32)
    for arr in arrays:
        # Average stereo channels, accumulate.
        if arr.ndim == 2:
            mono[:arr.shape[0]] += arr.mean(axis=1)
        else:
            mono[:arr.shape[0]] += arr

    # Slice to A-B region if specified.
    if start_sec is not None and end_sec is not None and end_sec > start_sec:
        s = int(start_sec * sample_rate)
        e = int(end_sec * sample_rate)
        mono = mono[max(0, s):min(len(mono), e)]

    # Normalise to prevent clipping.
    peak = np.max(np.abs(mono))
    if peak > 0:
        mono /= peak

    duration = len(mono) / sample_rate
    if duration < _MIN_DURATION:
        return DetectionResult()

    # Beat detection.
    if model_path and os.path.isfile(model_path):
        beat_times, downbeat_times, bpm = _detect_beats_onnx(
            mono, sample_rate, model_path,
        )
    else:
        beat_times, downbeat_times, bpm = _detect_beats_librosa(
            mono, sample_rate,
        )

    # Key detection.
    key_name, key_corr = _detect_key(mono, sample_rate)

    # Clamp BPM to sensible range.
    if bpm > 0:
        bpm = max(20.0, min(300.0, bpm))

    return DetectionResult(
        bpm=bpm,
        bpm_confidence=_bpm_confidence(beat_times),
        key=key_name if _key_confidence(key_corr) != "low" else key_name,
        key_confidence=_key_confidence(key_corr),
        beat_times=beat_times,
        downbeat_times=downbeat_times,
    )


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class DetectionWorker(QThread):
    """Background thread for BPM/key detection.

    Signals:
        completed(object): A ``DetectionResult`` instance.
        progress(str): Status message (e.g. "Analyzing beats...").
        error(str): Error description if detection fails.
    """

    completed = Signal(object)
    progress = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        stems: dict[str, np.ndarray],
        sample_rate: int,
        model_path: str | None = None,
        start_sec: float | None = None,
        end_sec: float | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._stems = stems
        self._sr = sample_rate
        self._model_path = model_path
        self._start_sec = start_sec
        self._end_sec = end_sec

    def run(self) -> None:
        try:
            self.progress.emit("Detecting...")
            result = detect_bpm_and_key(
                self._stems, self._sr, self._model_path,
                self._start_sec, self._end_sec,
            )
            self.completed.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))

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
    time_signature: str = ""        # e.g. "4/4", "3/4", "" if unknown
    chord_sequence: list = dataclasses.field(default_factory=list)  # list[tuple[float, str]]


# ---------------------------------------------------------------------------
# Beat detection — beat_this ONNX
# ---------------------------------------------------------------------------

# Preprocessing constants matching the beat_this model (CPJKU/beat_this).
_BT_SR = 22050
_BT_N_FFT = 1024
_BT_HOP = 441          # 50 fps at 22050 Hz
_BT_N_MELS = 128
_BT_FMIN = 30.0
_BT_FMAX = 11000.0     # official: 11 kHz upper band
_BT_LOG_MUL = 1000.0   # ln(1 + 1000 * mel)

# Chunked inference — rotary embeddings require fixed-size chunks.
_BT_CHUNK_SIZE = 1500   # frames (30 s at 50 fps)
_BT_BORDER = 6          # overlap frames discarded at chunk boundaries

# Peak-picking parameters.
_BEAT_THRESHOLD = 0.3
_DOWNBEAT_THRESHOLD = 0.15  # Lower: downbeat activations are weaker
_BEAT_MIN_DIST = 6          # ~120 ms at 50 fps


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


def _bt_spectrogram(audio_mono: np.ndarray) -> np.ndarray:
    """Compute the log-mel spectrogram expected by beat_this.

    Returns array of shape ``(frames, 128)`` in float32.  Matches the
    official ``LogMelSpect`` preprocessing: magnitude mel spectrogram
    (power=1), frame-length normalisation, and ``ln(1 + 1000 * x)``.
    """
    mel = librosa.feature.melspectrogram(
        y=audio_mono, sr=_BT_SR,
        n_fft=_BT_N_FFT, hop_length=_BT_HOP,
        n_mels=_BT_N_MELS, fmin=_BT_FMIN, fmax=_BT_FMAX,
        power=1.0,           # magnitude, not power
    )
    # Frame-length normalisation (torchaudio normalized="frame_length").
    mel = mel / math.sqrt(_BT_N_FFT)
    log_mel = np.log1p(_BT_LOG_MUL * mel).astype(np.float32)
    return log_mel.T  # (frames, 128)


def _bt_chunked_inference(
    spec: np.ndarray, session,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run beat_this on *spec* in 1500-frame chunks with border overlap.

    Returns ``(beat_logits, downbeat_logits)`` arrays covering the full
    spectrogram, or ``(beat_logits, None)`` if the model has only one
    output.
    """
    n_frames = spec.shape[0]
    input_name = session.get_inputs()[0].name
    stride = _BT_CHUNK_SIZE - 2 * _BT_BORDER

    beat_parts: list[np.ndarray] = []
    db_parts: list[np.ndarray] = []

    offset = 0
    while offset < n_frames:
        end = min(offset + _BT_CHUNK_SIZE, n_frames)
        chunk = spec[offset:end]

        # Pad to exactly _BT_CHUNK_SIZE frames.
        if chunk.shape[0] < _BT_CHUNK_SIZE:
            pad_width = _BT_CHUNK_SIZE - chunk.shape[0]
            chunk = np.pad(chunk, ((0, pad_width), (0, 0)))

        outputs = session.run(
            None, {input_name: chunk[np.newaxis, :, :]},
        )
        beat_out = outputs[0][0]  # (chunk_size,)
        db_out = outputs[1][0] if len(outputs) > 1 else None

        # Trim borders (keep full extent at first/last chunk).
        actual_len = min(end - offset, _BT_CHUNK_SIZE)
        lo = 0 if offset == 0 else _BT_BORDER
        hi = actual_len if end >= n_frames else actual_len - _BT_BORDER
        beat_parts.append(beat_out[lo:hi])
        if db_out is not None:
            db_parts.append(db_out[lo:hi])

        offset += stride

    beat_logits = np.concatenate(beat_parts)[:n_frames]
    db_logits = np.concatenate(db_parts)[:n_frames] if db_parts else None
    return beat_logits, db_logits


def _detect_beats_onnx(
    audio_mono: np.ndarray, sr: int, model_path: str,
) -> tuple[list[float], list[float], float]:
    """Run beat_this ONNX model and return (beat_times, downbeat_times, bpm).

    Uses chunked inference (1500-frame segments with 6-frame overlap) to
    match the official beat_this inference pipeline and avoid shape errors
    in the rotary-embedding attention layers.
    """
    # Resample to model sample rate.
    if sr != _BT_SR:
        audio_mono = librosa.resample(audio_mono, orig_sr=sr, target_sr=_BT_SR)

    spec = _bt_spectrogram(audio_mono)  # (frames, 128)
    session = _create_onnx_session(model_path)
    beat_logits, db_logits = _bt_chunked_inference(spec, session)

    beat_probs = _sigmoid(beat_logits)
    beat_frames = _peak_pick(beat_probs, _BEAT_THRESHOLD, _BEAT_MIN_DIST)
    fps = _BT_SR / _BT_HOP
    beat_times = [f / fps for f in beat_frames]

    downbeat_times: list[float] = []
    if db_logits is not None:
        db_probs = _sigmoid(db_logits)
        db_frames = _peak_pick(db_probs, _DOWNBEAT_THRESHOLD, _BEAT_MIN_DIST)
        downbeat_times = [f / fps for f in db_frames]

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
# Time signature detection
# ---------------------------------------------------------------------------

_TIME_SIG_MAP = {2: "2/4", 3: "3/4", 4: "4/4", 5: "5/4", 6: "6/8", 7: "7/8"}


def _detect_time_signature(
    beat_times: list[float], downbeat_times: list[float],
) -> str:
    """Infer time signature from beats between consecutive downbeats.

    Requires at least two downbeats (one complete bar) produced by the
    beat_this ONNX model.  Returns e.g. ``"4/4"`` or ``""`` if unknown.
    """
    if len(downbeat_times) < 2 or len(beat_times) < 2:
        return ""

    bt = np.asarray(beat_times)
    beats_per_bar: list[int] = []
    for i in range(len(downbeat_times) - 1):
        count = int(np.sum((bt >= downbeat_times[i]) & (bt < downbeat_times[i + 1])))
        if count > 0:
            beats_per_bar.append(count)

    if not beats_per_bar:
        return ""

    median_bpb = int(round(float(np.median(beats_per_bar))))
    return _TIME_SIG_MAP.get(median_bpb, f"{median_bpb}/4")


# ---------------------------------------------------------------------------
# Chord detection — chromagram template matching + HMM smoothing
# ---------------------------------------------------------------------------

_CHORD_NAMES = ["C", "C#", "D", "Eb", "E", "F",
                "F#", "G", "Ab", "A", "Bb", "B"]

# Chord qualities — major, minor, plus the most chromagram-distinguishable
# extended chords.  7th chords with tritones (dom7) and wide voicings
# (maj7, dim) are reliable; m7 is too close to minor in chroma space.
_QUALITY_INTERVALS: dict[str, list[int]] = {
    "":     [0, 4, 7],       # major triad
    "m":    [0, 3, 7],       # minor triad
    "7":    [0, 4, 7, 10],   # dominant 7th (tritone makes it distinct)
    "maj7": [0, 4, 7, 11],   # major 7th (wide voicing, distinct from major)
    "dim":  [0, 3, 6],       # diminished (tritone interval, very distinct)
}


def _build_chord_templates() -> tuple[tuple[str, np.ndarray], ...]:
    """Build chord templates for all roots and quality types.

    Returns a tuple of (label, 12-dim_chroma_vector) tuples.
    Built once at module load — thread-safe and immutable.
    """
    templates: list[tuple[str, np.ndarray]] = []
    for root_idx, root_name in enumerate(_CHORD_NAMES):
        for suffix, intervals in _QUALITY_INTERVALS.items():
            template = np.zeros(12, dtype=np.float64)
            for iv in intervals:
                template[(root_idx + iv) % 12] = 1.0
            norm = np.linalg.norm(template)
            if norm > 0:
                template /= norm
            template.flags.writeable = False
            templates.append((f"{root_name}{suffix}", template))
    return tuple(templates)


# Built eagerly at import time — 84 templates, ~8 KB, thread-safe.
_CHORD_TEMPLATES = _build_chord_templates()


def _viterbi_smooth(
    frame_labels: list[int], n_states: int,
    self_prob: float = 0.96,
) -> list[int]:
    """Viterbi smoothing with a simple self-transition-biased HMM.

    Encourages staying in the same chord state across consecutive frames,
    reducing noisy frame-by-frame flickering.

    Vectorised: the inner state loop uses NumPy broadcasting so the
    Python-level complexity is O(T × N) instead of O(T × N²).
    """
    n_frames = len(frame_labels)
    if n_frames == 0:
        return []

    # Uniform initial probability.
    log_pi = np.full(n_states, -math.log(n_states))

    # Transition matrix: log_trans[prev, cur].
    other_prob = (1.0 - self_prob) / max(1, n_states - 1)
    log_other = math.log(other_prob) if other_prob > 0 else -1e12
    log_trans = np.full((n_states, n_states), log_other)
    np.fill_diagonal(log_trans, math.log(self_prob))

    # Emission log-probabilities.
    emit_match = math.log(0.8)
    emit_miss = math.log(0.2 / max(1, n_states - 1))

    # Viterbi forward pass — fully vectorised inner loop.
    viterbi = np.full((n_frames, n_states), -np.inf)
    backptr = np.zeros((n_frames, n_states), dtype=np.int32)

    # Initialise t=0.
    emit_vec = np.full(n_states, emit_miss)
    emit_vec[frame_labels[0]] = emit_match
    viterbi[0] = log_pi + emit_vec

    for t in range(1, n_frames):
        # scores[prev, cur] = viterbi[t-1, prev] + log_trans[prev, cur]
        scores = viterbi[t - 1, :, np.newaxis] + log_trans  # (N, N)
        backptr[t] = np.argmax(scores, axis=0)               # (N,)
        best_scores = scores[backptr[t], np.arange(n_states)] # (N,)

        emit_vec = np.full(n_states, emit_miss)
        emit_vec[frame_labels[t]] = emit_match
        viterbi[t] = best_scores + emit_vec

    # Backtrack.
    path = [0] * n_frames
    path[-1] = int(np.argmax(viterbi[-1]))
    for t in range(n_frames - 2, -1, -1):
        path[t] = int(backptr[t + 1, path[t + 1]])

    return path


def _detect_chords(
    audio_mono: np.ndarray, sr: int,
    hop_length: int = 4096,
) -> list[tuple[float, str]]:
    """Detect chords using chromagram template matching + HMM smoothing.

    Returns a list of (onset_seconds, chord_label) tuples representing
    consecutive chord segments.
    """
    chroma = librosa.feature.chroma_cqt(
        y=audio_mono, sr=sr, hop_length=hop_length,
    )  # (12, n_frames)
    n_frames = chroma.shape[1]
    if n_frames == 0:
        return []

    templates = _build_chord_templates()
    template_matrix = np.array([t[1] for t in templates])  # (n_templates, 12)

    # Per-frame energy for silence gating.
    frame_energy = np.sum(chroma, axis=0)  # (n_frames,)
    nonzero = frame_energy[frame_energy > 0]
    energy_thresh = float(np.percentile(nonzero, 10)) * 0.5 if len(nonzero) else 0.0

    # Normalise each chroma frame.
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    chroma_norm = chroma / norms  # (12, n_frames)

    # Cosine similarity: (n_templates, n_frames).
    similarities = template_matrix @ chroma_norm
    frame_labels = np.argmax(similarities, axis=0).tolist()

    # HMM smoothing — high self-transition to reduce flickering.
    n_states = len(templates)
    smoothed = _viterbi_smooth(frame_labels, n_states, self_prob=0.99)

    # Merge consecutive same-chord frames, holding previous chord in silence.
    fps = sr / hop_length
    segments: list[tuple[float, str]] = []
    prev_label = -1
    for i, label in enumerate(smoothed):
        if frame_energy[i] < energy_thresh:
            continue  # hold previous chord during silence
        if label != prev_label:
            segments.append((i / fps, templates[label][0]))
            prev_label = label

    return segments


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

    # Beat detection — try ONNX model, fall back to librosa on any error.
    beat_times: list[float] = []
    downbeat_times: list[float] = []
    bpm = 0.0
    if model_path and os.path.isfile(model_path):
        try:
            beat_times, downbeat_times, bpm = _detect_beats_onnx(
                mono, sample_rate, model_path,
            )
        except Exception:
            beat_times, downbeat_times, bpm = _detect_beats_librosa(
                mono, sample_rate,
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

    # Time signature from beat_this downbeats (empty when using librosa
    # fallback, which does not produce downbeats).
    time_sig = _detect_time_signature(beat_times, downbeat_times)

    # Chord detection.
    chord_sequence = _detect_chords(mono, sample_rate)

    return DetectionResult(
        bpm=bpm,
        bpm_confidence=_bpm_confidence(beat_times),
        key=key_name if _key_confidence(key_corr) != "low" else key_name,
        key_confidence=_key_confidence(key_corr),
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        time_signature=time_sig,
        chord_sequence=chord_sequence,
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

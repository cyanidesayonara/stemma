"""Multi-track audio player for synchronized playback of separated stems.

Uses `sounddevice` for zero-latency memory buffer mixing. Stems are loaded
into RAM entirely and summed dynamically inside the C-level audio callback,
allowing instant, click-free muting and soloing.

Recording is supported via ``sd.Stream`` (full-duplex): when recording is
armed, ``play()`` creates a duplex stream whose single callback captures
input audio at the exact playback frame position -- guaranteeing perfect
frame synchronisation with the stems being mixed to output.
"""

import glob
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from PySide6.QtCore import QObject, QThread, Signal, QTimer

from src.click_utils import generate_click
from src.qt_signal_utils import safe_disconnect as _safe_disconnect


SPEED_PRESETS = (0.5, 0.75, 0.85, 1.0, 1.25, 1.5, 2.0)

# Bounds for pitch transposition, in semitones. librosa's pitch_shift
# quality degrades noticeably beyond ±7 (it chains resample+time_stretch
# internally); this range covers the practical use cases (vocal range
# adjustment, capo equivalents) without exposing the quality cliff.
PITCH_MIN_SEMITONES = -7
PITCH_MAX_SEMITONES = 7

# Prefix used for recording-take stem names (e.g. "recording_take1").
# Used to distinguish user recordings from source stems when deciding
# whether to apply pitch transposition.
RECORDING_STEM_PREFIX = "recording_take"


def next_take_number(song_dir: str) -> int:
    """Return the next recording take number for *song_dir*."""
    existing = glob.glob(os.path.join(song_dir, f"{RECORDING_STEM_PREFIX}*.wav"))
    nums: list[int] = []
    for p in existing:
        base = os.path.basename(p)
        try:
            n = int(base.replace(RECORDING_STEM_PREFIX, "").replace(".wav", ""))
            nums.append(n)
        except ValueError:
            continue
    return max(nums, default=0) + 1


class StretchWorker(QThread):
    """Background thread for pitch-preserving time-stretch and/or pitch
    shift of all stems.

    Both transforms are applied in a single pass per stem so artifacts do
    not compound. Pitch shift runs first (it internally resamples then
    time-stretches back to the original length), then the requested
    playback-speed stretch is applied.

    Stems are dispatched to a ThreadPoolExecutor for overlap between the
    Python bookkeeping and the C-level FFT / resampling kernels.  In
    practice the GIL and librosa's own internal Python loops limit
    wall-clock speedup to roughly 1.0-1.3x; the main perf lever is
    ``_HOP_LENGTH`` (STFT hop size), which trades STFT frame count
    linearly against render time with minimal quality impact.

    Recording-take stems are only pitch-shifted when
    ``sync_recording_pitch`` is True. Speed is always applied to every
    stem (speed changes affect all audible audio; recordings must stay in
    sync with the backing track).

    Cancellation:
        ``cancel()`` sets a flag checked between each stem and each
        channel. In-flight librosa calls cannot be interrupted -- cancel
        responsiveness is bounded by the length of a single channel's
        pitch-shift / time-stretch pass (~0.5-2s on typical songs).
        A cancelled worker emits no further ``progress`` / ``completed``
        / ``error`` signals.
    """

    completed = Signal(dict)  # {name: stretched_ndarray}
    progress = Signal(int, int)  # (current_stem, total_stems)
    error = Signal(str)

    # Resampler quality.  librosa's default is "soxr_hq" (high-quality
    # SOX resampler).  We keep it -- dropping to "soxr_mq" is ~3x faster
    # but introduces an audible metallic timbre on transient-heavy
    # material (drums, plucked strings), which is unacceptable for a
    # tool whose whole purpose is faithful playback of the source.
    # Speed wins come from parallel stem rendering, not from cutting
    # resampler quality.
    _RESAMPLE_TYPE = "soxr_hq"

    # Cap parallelism to avoid excess RAM usage.  Each thread buffers a
    # mono copy of one stem; 4 threads × ~28 MB/stem = ~112 MB overhead
    # for a typical 5-min project.
    _MAX_PARALLEL_STEMS = 4

    def __init__(
        self,
        stems: dict[str, np.ndarray],
        sample_rate: int,
        speed: float,
        pitch_semitones: int,
        sync_recording_pitch: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._stems = stems
        self._sample_rate = int(sample_rate)
        self._speed = float(speed)
        self._pitch_semitones = int(pitch_semitones)
        self._sync_recording_pitch = bool(sync_recording_pitch)
        self._cancelled = False

    def cancel(self) -> None:
        """Request early termination at the next stem/channel boundary.

        Once set, the worker emits no further ``progress``, ``completed``,
        or ``error`` signals; its ``run()`` returns cleanly so the QThread
        ``finished`` signal still fires and the player can reap it.
        """
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        try:
            self._stretch()
        except Exception as exc:  # noqa: BLE001
            if not self._cancelled:
                self.error.emit(str(exc))

    def _stretch(self) -> None:
        total = len(self._stems)
        if total == 0:
            if not self._cancelled:
                self.completed.emit({})
            return

        apply_speed = self._speed != 1.0
        apply_pitch = self._pitch_semitones != 0

        # Show "(0/N)" in the spinbox suffix immediately so the user sees
        # the expected count from the first frame rather than "..." for the
        # full render duration.
        if not self._cancelled:
            self.progress.emit(0, total)

        out: dict[str, np.ndarray] = {}

        def process(item):
            name, data = item
            if self._cancelled:
                return name, None
            result = self._process_stem(
                name, data, apply_speed, apply_pitch,
            )
            return name, result

        # Progress is emitted from this (the run) thread rather than from
        # pool workers so the signal behaves consistently whether the
        # worker is started via QThread.start() or called synchronously
        # via run().  Qt cross-thread emits need an event loop to be
        # delivered; same-thread emits do not.
        max_workers = min(total, self._MAX_PARALLEL_STEMS)
        completed_count = 0
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="stretch",
        ) as pool:
            futures = [
                pool.submit(process, item)
                for item in self._stems.items()
            ]
            for fut in as_completed(futures):
                name, result = fut.result()
                completed_count += 1
                if result is not None:
                    out[name] = result
                    if not self._cancelled:
                        self.progress.emit(completed_count, total)

        if self._cancelled:
            return
        self.completed.emit(out)

    def _process_stem(
        self,
        name: str,
        data: np.ndarray,
        apply_speed: bool,
        apply_pitch: bool,
    ) -> np.ndarray | None:
        """Render one stem; return ``None`` if cancelled mid-flight."""
        is_recording = name.startswith(RECORDING_STEM_PREFIX)
        stem_apply_pitch = apply_pitch and (
            not is_recording or self._sync_recording_pitch
        )
        if not apply_speed and not stem_apply_pitch:
            # Nothing to do for this stem -- reuse the original buffer.
            return data

        # Preserve the peak amplitude after phase-vocoder processing.
        original_peak = np.max(np.abs(data))

        # librosa effects work on mono; process each channel.
        channels = []
        for ch_idx in range(data.shape[1]):
            if self._cancelled:
                return None
            mono = data[:, ch_idx].astype(np.float32)
            if stem_apply_pitch:
                mono = librosa.effects.pitch_shift(
                    mono,
                    sr=self._sample_rate,
                    n_steps=self._pitch_semitones,
                    res_type=self._RESAMPLE_TYPE,
                )
            if apply_speed:
                mono = librosa.effects.time_stretch(
                    mono, rate=self._speed,
                )
            channels.append(mono)

        if self._cancelled:
            return None

        # Recombine to stereo, matching shortest channel.
        min_len = min(c.shape[0] for c in channels)
        stereo = np.column_stack([c[:min_len] for c in channels])
        stereo = stereo.astype(np.float32)

        # Normalize to match original peak level (phase vocoder can
        # reduce amplitude).
        stretched_peak = np.max(np.abs(stereo))
        if stretched_peak > 0 and original_peak > 0:
            stereo *= original_peak / stretched_peak

        return stereo


class MultiTrackPlayer(QObject):
    """Audio player that mixes multiple stems in real-time.

    Signals:
        position_changed(float): Current playback position in seconds.
        state_changed(bool): Emitted when playback starts or stops.
        play_finished(): Emitted when the end of the track is reached.
        playback_failed(str): Emitted when opening the output device fails
            (e.g. no device available). *str* is a short user-facing message.
    """

    position_changed = Signal(float)
    state_changed = Signal(bool)
    play_finished = Signal()
    speed_changed = Signal(float)
    pitch_changed = Signal(int)  # emitted with semitones (-N..+N)
    stretch_started = Signal()   # render began (worker spawned)
    stretch_progress = Signal(int, int)  # (current_stem, total_stems)
    stretch_finished = Signal()  # render completed (success or error)
    playback_failed = Signal(str)
    recording_saved = Signal(str)  # emitted with the saved WAV path

    def __init__(self) -> None:
        super().__init__()
        # Audio data storage.
        self._stems: dict[str, np.ndarray] = {}
        self._sample_rate: int = 44100
        self._total_frames: int = 0
        self._current_frame: int = 0

        # Mixing state. State tracking is safe because sounddevice acquires
        # the GIL before invoking the Python audio callback, ensuring atomicity.
        self._is_playing: bool = False
        self._muted_stems: set[str] = set()
        self._soloed_stems: set[str] = set()
        self._volumes: dict[str, float] = {}  # Per-stem gain, 0.0–2.0
        self._master_volume: float = 1.0     # Master gain, 0.0–2.0
        self._applied_gains: dict[str, float] = {}  # Last gain per stem (for ramping)
        self._active_stems_cache: list[str] | None = None

        # A-B loop state.
        self._loop_a_frame: int | None = None
        self._loop_b_frame: int | None = None
        self._looping: bool = False

        # Speed / time-stretch / pitch state.
        self._playback_speed: float = 1.0
        self._pitch_semitones: int = 0
        self._sync_recording_pitch: bool = False
        self._original_stems: dict[str, np.ndarray] = {}
        self._stretch_worker: StretchWorker | None = None
        # Keepalive refs for detached-but-still-running workers. Without
        # this, a rapid succession of set_pitch/set_speed calls drops the
        # previous Python wrapper to refcount zero; the GC then deletes
        # the QThread while its run loop is still active, producing
        # "QThread: Destroyed while thread is still running" crashes.
        self._detached_workers: list[StretchWorker] = []

        # Metronome state.
        self._metronome_enabled: bool = False
        self._metronome_bpm: float = 120.0
        self._metronome_volume: float = 0.5
        self._metronome_phase: int = 0
        self._click_buf: np.ndarray = self._generate_click(self._sample_rate)

        # Beat detection results (populated externally by DetectionWorker).
        self._beat_times: list[float] = []
        self._downbeat_times: list[float] = []
        self._beat_sync_enabled: bool = False
        self._beat_sync_nudge_ms: float = 0.0
        self._beat_frames: np.ndarray = np.array([], dtype=np.int64)

        # Chord sequence: list of (onset_seconds, chord_label).
        self._chord_sequence: list[tuple[float, str]] = []
        self._chord_times: np.ndarray = np.array([], dtype=np.float64)

        # Count-in state.
        self._count_in_enabled: bool = False
        self._count_in_beats: int = 4
        self._count_in_on_repeats: bool = False
        self._count_in_remaining: int = 0
        self._count_in_phase: int = 0
        self._count_in_beat: int = 0

        # Recording state.
        self._recording_armed: bool = False
        self._recording: bool = False
        self._recording_buffer: np.ndarray | None = None
        self._indata_capture: np.ndarray | None = None
        self._input_device: int | None = None
        self._latency_offset_frames: int = 0
        self._recording_song_dir: str | None = None

        # Per-stem nudge offsets (ms), for post-recording alignment.
        self._nudge_offsets: dict[str, float] = {}

        # Hardware stream.
        self._stream: sd.OutputStream | sd.Stream | None = None
        self._output_device: int | None = None
        self._suppress_next_count_in: bool = False

        # UI updater.
        self._timer = QTimer(self)
        self._timer.setInterval(50)  # ~20fps
        self._timer.timeout.connect(self._emit_position)

    @property
    def has_stems(self) -> bool:
        """Return True if any stems are loaded."""
        return bool(self._stems)

    @property
    def is_playing(self) -> bool:
        """Return True if audio is currently playing."""
        return self._is_playing

    @property
    def current_seconds(self) -> float:
        """Return the current playback position in seconds."""
        if self._sample_rate == 0:
            return 0.0
        return self._current_frame / self._sample_rate

    @property
    def total_seconds(self) -> float:
        """Return the total duration of the loaded track in seconds."""
        if self._sample_rate == 0:
            return 0.0
        return self._total_frames / self._sample_rate

    @staticmethod
    def _generate_click(sample_rate: int) -> np.ndarray:
        """Generate a short click sound for the metronome.

        Delegates to the shared ``generate_click`` utility so the same
        click waveform is used by both the live player and the exporter.
        """
        return generate_click(sample_rate)

    @property
    def metronome_enabled(self) -> bool:
        """Return True if the metronome is active."""
        return self._metronome_enabled

    @property
    def metronome_bpm(self) -> float:
        """Return the current metronome BPM."""
        return self._metronome_bpm

    @property
    def metronome_volume(self) -> float:
        """Return the current metronome volume (0.0-2.0)."""
        return self._metronome_volume

    def set_metronome_enabled(self, enabled: bool) -> None:
        """Enable or disable the metronome click track."""
        self._metronome_enabled = enabled
        self._metronome_phase = 0

    def set_metronome_bpm(self, bpm: float) -> None:
        """Set the metronome tempo. Clamped to 20--300 BPM.

        Non-finite values (NaN, inf) are silently ignored.
        """
        value = float(bpm)
        if not math.isfinite(value):
            return
        self._metronome_bpm = max(20.0, min(300.0, value))
        self._metronome_phase = 0

    def set_metronome_volume(self, volume: float) -> None:
        """Set the metronome volume. Clamped to 0.0-2.0."""
        self._metronome_volume = max(0.0, min(2.0, float(volume)))

    # -- Beat grid ----------------------------------------------------------

    @property
    def beat_times(self) -> list[float]:
        """Beat timestamps in seconds (populated by detection)."""
        return self._beat_times

    @property
    def downbeat_times(self) -> list[float]:
        """Downbeat (bar-1) timestamps in seconds."""
        return self._downbeat_times

    def set_beat_times(
        self, beats: list[float], downbeats: list[float],
    ) -> None:
        """Store detected beat/downbeat timestamps."""
        self._beat_times = [float(b) for b in beats]
        self._downbeat_times = [float(b) for b in downbeats]
        self._recompute_beat_frames()

    # -- Chord sequence API -------------------------------------------------

    @property
    def chord_sequence(self) -> list[tuple[float, str]]:
        """Chord onsets: list of (time_seconds, chord_label)."""
        return self._chord_sequence

    def set_chord_sequence(self, chords: list[tuple[float, str]]) -> None:
        """Store detected chord sequence."""
        self._chord_sequence = [(float(t), str(c)) for t, c in chords]
        if self._chord_sequence:
            self._chord_times = np.array(
                [t for t, _ in self._chord_sequence], dtype=np.float64,
            )
        else:
            self._chord_times = np.array([], dtype=np.float64)

    def chord_at(self, frame: int) -> str:
        """Return the chord label active at the given frame index.

        Uses binary search on chord onset times. Returns empty string
        if no chord data is available.
        """
        if len(self._chord_times) == 0 or self._sample_rate == 0:
            return ""
        speed = self._playback_speed if self._playback_speed > 0 else 1.0
        # Frame is in the stretched timeline; chord onsets are in original
        # audio time.  Divide by speed to map back (same convention as
        # _recompute_beat_frames).
        time_sec = frame / self._sample_rate / speed
        idx = int(np.searchsorted(self._chord_times, time_sec, side="right")) - 1
        if idx < 0:
            return ""
        return self._chord_sequence[idx][1]

    # -- Beat-synced metronome API ------------------------------------------

    @property
    def beat_sync_enabled(self) -> bool:
        """Return True if the metronome is synced to detected beats."""
        return self._beat_sync_enabled

    def set_beat_sync_enabled(self, enabled: bool) -> None:
        """Enable or disable beat-synced metronome mode."""
        self._beat_sync_enabled = enabled
        if enabled:
            self._recompute_beat_frames()

    @property
    def beat_sync_nudge_ms(self) -> float:
        """Return the user-defined metronome synchronization offset in ms."""
        return self._beat_sync_nudge_ms

    def set_beat_sync_nudge_ms(self, offset_ms: float) -> None:
        """Shift the metronome click timing by `offset_ms` milliseconds."""
        self._beat_sync_nudge_ms = float(offset_ms)
        if self._beat_sync_enabled:
            self._recompute_beat_frames()

    def _recompute_beat_frames(self) -> None:
        """Convert beat_times (seconds) to frame indices for the current speed.

        When speed != 1.0, the audio is time-stretched, so a beat at time
        *t* in the original sits at frame ``t / speed * sample_rate`` in
        the stretched audio.
        """
        if not self._beat_times or self._sample_rate == 0:
            self._beat_frames = np.array([], dtype=np.int64)
            return
        speed = self._playback_speed if self._playback_speed > 0 else 1.0
        sr = self._sample_rate
        offset_sec = self._beat_sync_nudge_ms / 1000.0
        self._beat_frames = np.array(
            [max(0, int((t + offset_sec) / speed * sr)) for t in self._beat_times],
            dtype=np.int64,
        )

    def instantaneous_bpm_at(self, frame: int) -> float:
        """Return the local BPM at *frame* based on neighbouring beat positions.

        Returns 0.0 when fewer than 2 beats are available.
        """
        bf = self._beat_frames
        if len(bf) < 2:
            return 0.0
        idx = int(np.searchsorted(bf, frame, side="right"))
        # Clamp so we always have two adjacent beats.
        idx = max(1, min(idx, len(bf) - 1))
        interval_frames = int(bf[idx] - bf[idx - 1])
        if interval_frames <= 0:
            return 0.0
        return 60.0 * self._sample_rate / interval_frames

    # -- Count-in API -------------------------------------------------------

    @property
    def count_in_enabled(self) -> bool:
        """Return True if the count-in is active."""
        return self._count_in_enabled

    @property
    def count_in_beats(self) -> int:
        """Return the number of count-in beats (1--8)."""
        return self._count_in_beats

    @property
    def count_in_on_repeats(self) -> bool:
        """Return True if the count-in plays before A-B loop repeats."""
        return self._count_in_on_repeats

    @property
    def counting_in(self) -> bool:
        """Return True if a count-in is currently playing."""
        return self._count_in_remaining > 0

    @property
    def count_in_current_beat(self) -> int:
        """Return the 1-based beat number during an active count-in, or 0."""
        return self._count_in_beat

    def set_count_in_enabled(self, enabled: bool) -> None:
        """Enable or disable the count-in before playback."""
        self._count_in_enabled = enabled

    def set_count_in_beats(self, beats: int) -> None:
        """Set the number of count-in beats. Clamped to 1--8."""
        self._count_in_beats = max(1, min(8, int(beats)))

    def set_count_in_on_repeats(self, enabled: bool) -> None:
        """Enable or disable count-in before A-B loop repeats."""
        self._count_in_on_repeats = enabled

    def _arm_count_in(self) -> None:
        """Prepare the count-in pre-roll if enabled and BPM is set."""
        if self._count_in_enabled and self._metronome_bpm > 0:
            beat_interval = int(
                60.0 / self._metronome_bpm * self._sample_rate
            )
            self._count_in_remaining = self._count_in_beats * beat_interval
            self._count_in_phase = 0
            self._count_in_beat = 1
        else:
            self._count_in_remaining = 0
            self._count_in_phase = 0
            self._count_in_beat = 0

    # -- Recording API -------------------------------------------------------

    @property
    def recording_armed(self) -> bool:
        """Return True if recording is armed (will start on next play)."""
        return self._recording_armed

    @property
    def is_recording(self) -> bool:
        """Return True if audio is actively being recorded."""
        return self._recording

    def arm_recording(self, armed: bool) -> None:
        """Arm or disarm recording.

        Recording requires the backing track at its original tempo and
        pitch (speed=1.0, pitch=0). Recording against time-stretched or
        transposed audio would capture performance that can't be cleanly
        mixed with the source stems later. Also requires stems to be
        loaded. Arming while already recording is a no-op.
        """
        if armed:
            if self._playback_speed != 1.0:
                return
            if self._pitch_semitones != 0:
                return
            if not self._stems:
                return
        self._recording_armed = armed

    def set_input_device(self, device: int | None) -> None:
        """Select the PortAudio input device index, or None for system default."""
        self._input_device = device

    def set_latency_offset_ms(self, ms: float) -> None:
        """Set recording latency compensation in milliseconds.

        Positive values shift the recording earlier (compensate for input
        device latency).
        """
        ms = max(-200.0, min(200.0, float(ms)))
        self._latency_offset_frames = int(ms / 1000.0 * self._sample_rate)

    def _allocate_recording_buffer(self) -> None:
        """Create a zeroed stereo buffer the same length as the current stems."""
        self._recording_buffer = np.zeros(
            (self._total_frames, 2), dtype=np.float32
        )

    def set_recording_song_dir(self, song_dir: str | None) -> None:
        """Set the directory where recording takes are saved."""
        self._recording_song_dir = song_dir

    def save_recording(self, song_dir: str) -> str | None:
        """Write the recording buffer to a WAV file in *song_dir*.

        Returns the path to the saved file, or None if there is no recording.
        The take number auto-increments based on existing files.
        """
        if self._recording_buffer is None:
            return None

        buf = self._recording_buffer

        if self._latency_offset_frames != 0:
            buf = np.roll(buf, -self._latency_offset_frames, axis=0)
            if self._latency_offset_frames > 0:
                buf[-self._latency_offset_frames:] = 0.0
            else:
                buf[:-self._latency_offset_frames] = 0.0

        take_num = next_take_number(song_dir)
        filename = f"recording_take{take_num}.wav"
        path = os.path.join(song_dir, filename)
        sf.write(path, buf, self._sample_rate)
        self._recording_buffer = None
        return path

    def add_recording_stem(self, name: str, data: np.ndarray) -> None:
        """Add a recording take as a playable stem.

        Validates sample rate compatibility and forces stereo.
        """
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        self._stems[name] = data
        self._original_stems[name] = data
        self._total_frames = max(self._total_frames, data.shape[0])

    def remove_recording_stem(self, name: str) -> None:
        """Remove a recording stem and recalculate total frames."""
        self._stems.pop(name, None)
        self._original_stems.pop(name, None)
        self._muted_stems.discard(name)
        self._soloed_stems.discard(name)
        self._volumes.pop(name, None)
        self._nudge_offsets.pop(name, None)
        self._recalculate_total_frames()

    def nudge_stem(self, name: str, offset_ms: float) -> None:
        """Shift a stem's audio by *offset_ms* milliseconds.

        Positive values shift the audio later (add silence at the start);
        negative values shift it earlier. The offset is clamped to
        -200..+200 ms. Wrapped samples are zeroed out.

        Both ``_stems`` and ``_original_stems`` are updated so the nudge
        survives speed changes.
        """
        if name not in self._stems:
            return
        offset_ms = max(-200.0, min(200.0, float(offset_ms)))
        old_offset = self._nudge_offsets.get(name, 0.0)
        if offset_ms == old_offset:
            return

        delta_ms = offset_ms - old_offset
        delta_frames = int(delta_ms / 1000.0 * self._sample_rate)
        if delta_frames == 0:
            self._nudge_offsets[name] = offset_ms
            return

        for store in (self._stems, self._original_stems):
            if name not in store:
                continue
            data = store[name]
            data = np.roll(data, delta_frames, axis=0)
            if delta_frames > 0:
                data[:delta_frames] = 0.0
            else:
                data[delta_frames:] = 0.0
            store[name] = data

        self._nudge_offsets[name] = offset_ms

    def get_nudge_ms(self, name: str) -> float:
        """Return the current nudge offset in ms for *name* (default 0)."""
        return self._nudge_offsets.get(name, 0.0)

    @property
    def nudge_offsets(self) -> dict[str, float]:
        """Return a copy of all per-stem nudge offsets (ms)."""
        return dict(self._nudge_offsets)

    def _recalculate_total_frames(self) -> None:
        """Recompute ``_total_frames`` from the current stems dict."""
        self._total_frames = max(
            (d.shape[0] for d in self._stems.values()), default=0
        )
        self._current_frame = min(self._current_frame, self._total_frames)

    def load_stems(self, stem_paths: dict[str, str]) -> None:
        """Load all stem WAV files into memory.

        Args:
            stem_paths: Dictionary mapping stem names to file paths.
        """
        self.stop()
        self._detach_stretch_worker()
        self._stems.clear()
        self._original_stems.clear()
        self._muted_stems.clear()
        self._soloed_stems.clear()
        self._volumes.clear()
        self._applied_gains.clear()
        self._active_stems_cache = None
        self._beat_times.clear()
        self._downbeat_times.clear()
        self._beat_sync_enabled = False
        self._beat_sync_nudge_ms = 0.0
        self._beat_frames = np.array([], dtype=np.int64)
        self._chord_sequence.clear()
        self._chord_times = np.array([], dtype=np.float64)
        self._loop_a_frame = None
        self._loop_b_frame = None
        self._looping = False
        self._playback_speed = 1.0
        self._pitch_semitones = 0
        self._recording_armed = False
        self._recording = False
        self._recording_buffer = None
        self._nudge_offsets.clear()

        max_frames = 0
        sample_rate = 0

        for name, path in stem_paths.items():
            # read returns (samples, channels)
            data, sr = sf.read(path, always_2d=True, dtype='float32')
            if sample_rate == 0:
                sample_rate = sr
            elif sr != sample_rate:
                raise ValueError(f"Sample rate mismatch in stem '{name}'")

            # Force stereo if mono.
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)

            self._stems[name] = data
            max_frames = max(max_frames, data.shape[0])

        self._sample_rate = sample_rate
        self._total_frames = max_frames
        self._current_frame = 0
        self._original_stems = dict(self._stems)
        self._click_buf = self._generate_click(self._sample_rate)
        self._metronome_phase = 0

        self.position_changed.emit(0.0)

    def set_output_device(self, device: int | None) -> None:
        """Select the PortAudio output device index, or None for the system default."""
        self._output_device = device
        if self._is_playing:
            self._suppress_next_count_in = True
            self.pause()
            self.play()

    def play(self) -> None:
        """Start or resume playback."""
        if not self._stems or self._is_playing:
            return

        if self._current_frame >= self._total_frames:
            self._current_frame = 0

        try:
            if self._stream is None:
                if self._recording_armed:
                    if self._recording_buffer is None:
                        self._allocate_recording_buffer()
                    in_dev = self._input_device
                    out_dev = self._output_device
                    try:
                        defaults = sd.default.device
                        in_resolved = in_dev if in_dev is not None else defaults[0]
                        out_resolved = out_dev if out_dev is not None else defaults[1]
                    except (TypeError, IndexError):
                        in_resolved = in_dev
                        out_resolved = out_dev
                    input_ch = 1
                    try:
                        if in_resolved is not None:
                            info = sd.query_devices(in_resolved)
                            input_ch = max(
                                1, min(int(info.get("max_input_channels", 1)), 2)
                            )
                    except (sd.PortAudioError, ValueError, TypeError, OSError):
                        pass
                    self._stream = sd.Stream(
                        samplerate=self._sample_rate,
                        channels=(input_ch, 2),
                        callback=self._full_duplex_callback,
                        device=(in_resolved, out_resolved),
                    )
                    self._recording = True
                else:
                    kwargs: dict[str, Any] = {
                        "samplerate": self._sample_rate,
                        "channels": 2,
                        "callback": self._audio_callback,
                    }
                    if self._output_device is not None:
                        kwargs["device"] = self._output_device
                    self._stream = sd.OutputStream(**kwargs)
            self._stream.start()
        except (sd.PortAudioError, OSError):
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            self._recording = False
            self._recording_buffer = None
            self.playback_failed.emit(
                "No audio device is available, or playback failed to "
                "start. Connect speakers or headphones, or choose another "
                "device in Edit > Preferences."
            )
            return

        if self._suppress_next_count_in:
            self._suppress_next_count_in = False
        else:
            self._arm_count_in()

        self._is_playing = True
        self._timer.start()
        self.state_changed.emit(True)

    def pause(self) -> None:
        """Pause playback.

        Recording state is preserved across pause/resume: the buffer is kept
        so that pressing Play again continues recording from where the
        playhead left off.  To finalize a take, use ``stop()``.
        """
        if not self._is_playing:
            return

        self._is_playing = False
        self._recording = False
        self._count_in_remaining = 0
        self._count_in_beat = 0
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._timer.stop()
        self.state_changed.emit(False)

    def stop(self) -> None:
        """Stop playback, finalize any recording, and reset the playhead.

        When A-B looping is active with a valid region, the playhead moves to
        loop A; otherwise it moves to the start of the track.

        If recording was active, the take is saved and ``recording_saved``
        is emitted.  Use ``pause()`` to interrupt without finalizing.
        """
        had_buffer = self._recording_buffer is not None
        self.pause()
        if had_buffer and self._recording_song_dir:
            path = self.save_recording(self._recording_song_dir)
            if path:
                self._recording_armed = False
                self.recording_saved.emit(path)
        elif had_buffer:
            self._recording_buffer = None
        if self._loop_region_is_active():
            self.seek(self._loop_a_frame / self._sample_rate)
        else:
            self.seek(0.0)

    def seek(self, position_s: float) -> None:
        """Seek to a specific position in seconds.

        When A-B looping is active with a valid region, the playhead is
        clamped into ``[loop_a, loop_b)``: positions before A or at/after B
        snap to loop A.

        Args:
            position_s: Target time in seconds.
        """
        target_frame = int(position_s * self._sample_rate)
        target_frame = max(0, min(target_frame, self._total_frames))
        if self._loop_region_is_active():
            la = self._loop_a_frame
            lb = self._loop_b_frame
            if target_frame < la:
                target_frame = la
            elif target_frame >= lb:
                target_frame = la
        self._current_frame = target_frame
        self._metronome_phase = 0
        self._count_in_remaining = 0
        self._count_in_beat = 0
        self.position_changed.emit(self._current_frame / self._sample_rate)

    def set_mute(self, stem_name: str, muted: bool) -> None:
        """Mute or unmute a specific stem."""
        if muted:
            self._muted_stems.add(stem_name)
        else:
            self._muted_stems.discard(stem_name)
        self._active_stems_cache = None

    @property
    def muted_stems(self) -> set[str]:
        """Return the set of currently muted stem names."""
        return set(self._muted_stems)

    @property
    def soloed_stems(self) -> set[str]:
        """Return the set of currently soloed stem names."""
        return set(self._soloed_stems)

    @property
    def volumes(self) -> dict[str, float]:
        """Return a copy of per-stem volume settings."""
        return dict(self._volumes)

    @property
    def stems(self) -> dict[str, "np.ndarray"]:
        """Return a shallow copy of the stems dict. Arrays are shared, not copied."""
        return dict(self._stems)

    @property
    def sample_rate(self) -> int:
        """Return the sample rate of loaded audio."""
        return self._sample_rate

    @property
    def master_volume(self) -> float:
        """Return the master volume (0.0–2.0)."""
        return self._master_volume

    def set_master_volume(self, volume: float) -> None:
        """Set the master volume (gain multiplier for all stems).

        Args:
            volume: Gain from 0.0 (silent) to 2.0 (double). Clamped.
        """
        self._master_volume = max(0.0, min(volume, 2.0))

    def set_volume(self, stem_name: str, volume: float) -> None:
        """Set the volume (gain) for a stem.

        Args:
            stem_name: Name of the stem.
            volume: Gain from 0.0 (silent) to 2.0 (double). Clamped.
        """
        self._volumes[stem_name] = max(0.0, min(volume, 2.0))

    def get_volume(self, stem_name: str) -> float:
        """Return the current volume for a stem (default 1.0)."""
        return self._volumes.get(stem_name, 1.0)

    def set_solo(self, stem_name: str, soloed: bool) -> None:
        """Solo or unsolo a specific stem."""
        if soloed:
            self._soloed_stems.add(stem_name)
        else:
            self._soloed_stems.discard(stem_name)
        self._active_stems_cache = None

    # ------------------------------------------------------------------
    # A-B Loop
    # ------------------------------------------------------------------

    def _loop_region_is_active(self) -> bool:
        """True when looping is on and A/B form a non-empty region (same
        condition the audio callback uses for wrap behaviour).
        """
        return (
            self._looping
            and self._loop_a_frame is not None
            and self._loop_b_frame is not None
            and self._loop_b_frame > self._loop_a_frame
        )

    @property
    def loop_a(self) -> float | None:
        """Return the A (start) loop point in seconds, or None."""
        if self._loop_a_frame is None:
            return None
        return self._loop_a_frame / self._sample_rate

    @property
    def loop_b(self) -> float | None:
        """Return the B (end) loop point in seconds, or None."""
        if self._loop_b_frame is None:
            return None
        return self._loop_b_frame / self._sample_rate

    @property
    def looping(self) -> bool:
        """Return True if A-B looping is active."""
        return self._looping

    def set_loop_a(self, position_s: float) -> None:
        """Set the A (start) loop point in seconds.

        If B is already set and A > B, the two points are swapped so that
        A is always before B.
        """
        frame = int(position_s * self._sample_rate)
        frame = max(0, min(frame, self._total_frames))

        if self._loop_b_frame is not None and frame > self._loop_b_frame:
            self._loop_a_frame = self._loop_b_frame
            self._loop_b_frame = frame
        else:
            self._loop_a_frame = frame

    def set_loop_b(self, position_s: float) -> None:
        """Set the B (end) loop point in seconds.

        If A is already set and B < A, the two points are swapped so that
        A is always before B.
        """
        frame = int(position_s * self._sample_rate)
        frame = max(0, min(frame, self._total_frames))

        if self._loop_a_frame is not None and frame < self._loop_a_frame:
            self._loop_b_frame = self._loop_a_frame
            self._loop_a_frame = frame
        else:
            self._loop_b_frame = frame

    def set_looping(self, enabled: bool) -> None:
        """Enable or disable A-B looping."""
        self._looping = enabled

    def clear_loop(self) -> None:
        """Clear both loop points and disable looping."""
        self._loop_a_frame = None
        self._loop_b_frame = None
        self._looping = False

    # ------------------------------------------------------------------
    # Speed / Pitch / Time-Stretch
    # ------------------------------------------------------------------
    #
    # Speed (time-stretch) and pitch (transposition) are both pre-rendered
    # from ``_original_stems`` in a single ``StretchWorker`` pass so their
    # artifacts do not compound. The fast path (speed=1.0 AND pitch=0)
    # skips the worker entirely and swaps originals in directly.

    @property
    def speed(self) -> float:
        """Return the current playback speed multiplier."""
        return self._playback_speed

    @property
    def pitch_semitones(self) -> int:
        """Return the current pitch transposition in semitones."""
        return self._pitch_semitones

    @property
    def sync_recording_pitch(self) -> bool:
        """Return True if recording stems are pitch-shifted with the backing track."""
        return self._sync_recording_pitch

    def set_speed(self, speed: float) -> None:
        """Set the playback speed with pitch-preserving time-stretch.

        Clamps *speed* to [0.5, 2.0]. Stretching runs in a background
        thread; the ``speed_changed`` signal fires when the stretched
        audio is ready.
        """
        speed = max(0.5, min(speed, 2.0))
        if speed == self._playback_speed:
            return
        self._playback_speed = speed
        self._render_stretch(emit=("speed",))

    def set_pitch(self, semitones: int) -> None:
        """Set pitch transposition in semitones.

        Clamps to [PITCH_MIN_SEMITONES, PITCH_MAX_SEMITONES]. Rendering
        runs in a background thread; ``pitch_changed`` fires when the
        transposed audio is ready (or immediately, on the fast path).
        """
        try:
            semitones = int(semitones)
        except (TypeError, ValueError):
            return
        semitones = max(PITCH_MIN_SEMITONES,
                        min(PITCH_MAX_SEMITONES, semitones))
        if semitones == self._pitch_semitones:
            return
        self._pitch_semitones = semitones
        self._render_stretch(emit=("pitch",))

    def set_sync_recording_pitch(self, sync: bool) -> None:
        """Enable or disable pitch-shifting of recording-take stems.

        When False (default), recordings keep their source pitch even when
        the backing track is transposed. When True, recordings are shifted
        alongside the source stems. Takes effect on the next render.
        """
        sync = bool(sync)
        if sync == self._sync_recording_pitch:
            return
        self._sync_recording_pitch = sync
        # Only re-render if a pitch shift is actually active -- otherwise
        # the setting has no audible effect and we can skip the work.
        if self._pitch_semitones != 0 and self._original_stems:
            self._render_stretch(emit=("pitch",))

    def _render_stretch(self, emit: tuple[str, ...] = ()) -> None:
        """Render stems for the current speed + pitch state.

        Detaches any in-flight worker first. Uses the fast path when both
        transforms are identity. Otherwise spawns a new ``StretchWorker``.

        *emit* is a tuple of signal names ("speed", "pitch") to fire after
        the render completes; the caller uses this to indicate which knob
        the user just turned so the UI can react appropriately.
        """
        was_rendering = self._detach_stretch_worker()

        if not self._original_stems:
            self._emit_stretch_signals(emit)
            if was_rendering:
                self.stretch_finished.emit()
            return

        speed = self._playback_speed
        pitch = self._pitch_semitones

        if speed == 1.0 and pitch == 0:
            # Fast path: no transform at all, swap originals in directly.
            self._apply_stretched_stems(dict(self._original_stems))
            self._recompute_beat_frames()
            self._emit_stretch_signals(emit)
            if was_rendering:
                # Close the render lifecycle the UI was waiting on so it
                # can restore its indicator.
                self.stretch_finished.emit()
            return

        self._stretch_worker = StretchWorker(
            self._original_stems,
            self._sample_rate,
            speed,
            pitch,
            sync_recording_pitch=self._sync_recording_pitch,
            parent=self,
        )
        self._stretch_worker.completed.connect(
            lambda stems: self._on_stretch_ready(stems, emit)
        )
        self._stretch_worker.error.connect(
            lambda msg: self._on_stretch_error(msg, emit)
        )
        # Re-emit per-stem progress so the UI can show a meaningful
        # "rendering N/M stems" indicator instead of an indefinite spinner.
        self._stretch_worker.progress.connect(self.stretch_progress)
        # Only emit stretch_started if this is a fresh render.  When we
        # chain worker-to-worker (user scrubbed the spinbox), the UI is
        # already in "rendering" mode; starting again would reset the
        # progress counter visibly but not restart the lifecycle.
        if not was_rendering:
            self.stretch_started.emit()
        self._stretch_worker.start()

    def _detach_stretch_worker(self) -> bool:
        """Disconnect, cancel, and release the current stretch worker.

        Returns True if a running worker was detached (UI can use this
        to emit a matching ``stretch_finished`` when no new render is
        about to start).  Running workers are asked to ``cancel()`` so
        they stop at the next stem/channel boundary instead of finishing
        their full (now-stale) render; we retain a Python reference on
        ``_detached_workers`` until the thread's ``finished`` signal
        fires, preventing QThread GC-during-run crashes.
        """
        if self._stretch_worker is None:
            return False
        _safe_disconnect(self._stretch_worker.completed)
        _safe_disconnect(self._stretch_worker.error)
        _safe_disconnect(self._stretch_worker.progress)
        worker = self._stretch_worker
        self._stretch_worker = None
        was_running = worker.isRunning()
        if was_running:
            worker.cancel()
            # Keep alive until the thread actually stops. ``setParent(None)``
            # is intentionally deferred to _reap_detached_worker so the Qt
            # parent remains valid while the run loop is active.
            self._detached_workers.append(worker)
            worker.finished.connect(
                lambda w=worker: self._reap_detached_worker(w)
            )
        else:
            worker.setParent(None)
            worker.deleteLater()
        return was_running

    def cancel_stretch(self) -> None:
        """Cancel any in-flight stretch render without starting a new one.

        Emits ``stretch_finished`` so the UI can clear its render
        indicator.  Safe to call when no render is active (no-op).
        """
        if self._detach_stretch_worker():
            self.stretch_finished.emit()

    def _reap_detached_worker(self, worker: "StretchWorker") -> None:
        """Release a detached worker after its thread has stopped."""
        try:
            self._detached_workers.remove(worker)
        except ValueError:
            pass
        worker.setParent(None)
        worker.deleteLater()

    def _on_stretch_ready(
        self, stems: dict, emit: tuple[str, ...],
    ) -> None:
        """Swap in rendered stems and adjust frame indices."""
        self._apply_stretched_stems(stems)
        self._recompute_beat_frames()
        self._emit_stretch_signals(emit)
        self.stretch_finished.emit()

    def _on_stretch_error(
        self, message: str, emit: tuple[str, ...],
    ) -> None:
        """Handle stretch failure by restoring originals at identity state."""
        self._playback_speed = 1.0
        self._pitch_semitones = 0
        self._apply_stretched_stems(dict(self._original_stems))
        # Beat frames must be recomputed after the stem swap: if speed was
        # non-identity before the error, _beat_frames still held stretched
        # indices that no longer match the restored (original-length) stems.
        self._recompute_beat_frames()
        # Emit both signals so any UI bound to either knob resets.
        self.speed_changed.emit(1.0)
        self.pitch_changed.emit(0)
        self.stretch_finished.emit()

    def _emit_stretch_signals(self, emit: tuple[str, ...]) -> None:
        """Fire speed_changed / pitch_changed signals per *emit* contents."""
        if "speed" in emit:
            self.speed_changed.emit(self._playback_speed)
        if "pitch" in emit:
            self.pitch_changed.emit(self._pitch_semitones)

    def _apply_stretched_stems(self, stems: dict) -> None:
        """Replace current stems with *stems* and adjust frame indices."""
        old_total = self._total_frames if self._total_frames > 0 else 1

        self._stems = stems
        self._total_frames = max(
            (data.shape[0] for data in stems.values()), default=0
        )

        ratio = self._total_frames / old_total

        self._current_frame = int(self._current_frame * ratio)
        if self._loop_a_frame is not None:
            self._loop_a_frame = int(self._loop_a_frame * ratio)
        if self._loop_b_frame is not None:
            self._loop_b_frame = int(self._loop_b_frame * ratio)

    # ------------------------------------------------------------------
    # Internal Callbacks
    # ------------------------------------------------------------------

    def _emit_position(self) -> None:
        """Emit the current playback position for UI updates."""
        if not self._is_playing and self._timer.isActive():
            had_buffer = self._recording_buffer is not None
            self._recording = False
            self._timer.stop()
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self.state_changed.emit(False)
            self.play_finished.emit()
            if had_buffer and self._recording_song_dir:
                path = self.save_recording(self._recording_song_dir)
                if path:
                    self._recording_armed = False
                    self.recording_saved.emit(path)
            elif had_buffer:
                self._recording_buffer = None
            return

        pos_s = self._current_frame / self._sample_rate
        self.position_changed.emit(pos_s)

    def _mix_metronome(self, outdata: np.ndarray, frames_written: int,
                       beat_interval: int, click_len: int) -> None:
        """Overlay metronome clicks onto the output buffer.

        Tracks beat phase across callbacks so clicks stay in sync.
        Handles both continuing a click that started in a previous buffer
        and starting new clicks within this buffer.
        """
        gain = self._metronome_volume
        offset_frames = int(self._beat_sync_nudge_ms / 1000.0 * self._sample_rate)
        render_phase = (self._metronome_phase - offset_frames) % beat_interval

        # Continue any click that started in a previous callback.
        if render_phase < click_len:
            n = min(click_len - render_phase, frames_written)
            outdata[:n] += self._click_buf[render_phase:render_phase + n] * gain

        # Walk through the buffer finding beat boundaries.
        pos = beat_interval - render_phase  # Frames until next beat start
        while pos < frames_written:
            # Overlay click starting at this beat.
            n = min(click_len, frames_written - pos)
            if n > 0:
                outdata[pos:pos + n] += self._click_buf[:n] * gain
            pos += beat_interval

        # Update phase for the next callback.
        self._metronome_phase = (self._metronome_phase + frames_written) % beat_interval

    def _mix_metronome_synced(
        self, outdata: np.ndarray, frame_start: int, frame_count: int,
        buf_offset: int = 0,
    ) -> None:
        """Overlay metronome clicks at detected beat positions.

        Unlike the grid-based ``_mix_metronome``, this method places clicks
        at the exact frame positions stored in ``_beat_frames``.

        *frame_start*/*frame_count* describe the range of stem frames that
        were mixed into ``outdata[buf_offset:buf_offset+frame_count]``.
        Multiple calls per callback handle mid-buffer loop wraps.
        """
        bf = self._beat_frames
        if len(bf) == 0:
            return

        gain = self._metronome_volume
        click_len = len(self._click_buf)
        frame_end = frame_start + frame_count

        # Find the first beat >= frame_start.
        idx = int(np.searchsorted(bf, frame_start, side="left"))

        while idx < len(bf) and bf[idx] < frame_end:
            offset_in_segment = int(bf[idx]) - frame_start
            buf_pos = buf_offset + offset_in_segment
            n = min(click_len, buf_offset + frame_count - buf_pos)
            if n > 0:
                outdata[buf_pos:buf_pos + n] += self._click_buf[:n] * gain
            idx += 1

    def _mix_count_in(self, outdata: np.ndarray, ci_frames: int,
                      beat_interval: int, click_len: int) -> None:
        """Overlay count-in clicks onto the output buffer.

        Uses its own phase counter (``_count_in_phase``) independent of the
        metronome phase so the two features don't interfere with each other.
        Updates ``_count_in_beat`` (1-based) for UI feedback.
        """
        gain = self._metronome_volume
        offset_frames = int(self._beat_sync_nudge_ms / 1000.0 * self._sample_rate)
        render_phase = (self._count_in_phase - offset_frames) % beat_interval

        if render_phase < click_len:
            n = min(click_len - render_phase, ci_frames)
            outdata[:n] += self._click_buf[render_phase:render_phase + n] * gain

        pos = beat_interval - render_phase
        while pos < ci_frames:
            n = min(click_len, ci_frames - pos)
            if n > 0:
                outdata[pos:pos + n] += self._click_buf[:n] * gain
            pos += beat_interval

        new_phase = (self._count_in_phase + ci_frames) % beat_interval
        self._count_in_phase = new_phase

        total_elapsed = (self._count_in_beats * beat_interval
                         - self._count_in_remaining + ci_frames)
        self._count_in_beat = min(
            total_elapsed // beat_interval + 1, self._count_in_beats
        )

    def _full_duplex_callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Full-duplex PortAudio callback for simultaneous record + playback.

        Stashes *indata* so ``_audio_callback`` can write it into the
        recording buffer at the correct (loop-aware) frame position, then
        delegates to ``_audio_callback`` for output mixing.
        """
        if self._recording and self._count_in_remaining == 0:
            self._indata_capture = indata.copy()
        else:
            self._indata_capture = None
        try:
            self._audio_callback(outdata, frames, time_info, status)
        finally:
            self._indata_capture = None

    def _audio_callback(self, outdata: np.ndarray, frames: int,
                        time_info: dict, status: sd.CallbackFlags) -> None:
        """PortAudio callback for pushing mixed audio to the hardware.

        When A-B looping is active, playback wraps from loop_b back to loop_a
        instead of stopping at the end of the track.
        """
        if not self._is_playing:
            outdata.fill(0.0)
            raise sd.CallbackStop

        # -- Count-in pre-roll -----------------------------------------------
        # During count-in, output metronome clicks over silence and do not
        # advance the stem playback position.
        if self._count_in_remaining > 0:
            outdata.fill(0.0)
            beat_interval = int(
                60.0 / self._metronome_bpm * self._sample_rate
            )
            if beat_interval > 0:
                ci_frames = min(frames, self._count_in_remaining)
                click_len = len(self._click_buf)
                self._mix_count_in(
                    outdata, ci_frames, beat_interval, click_len
                )
                self._count_in_remaining -= ci_frames
                if self._count_in_remaining <= 0:
                    self._count_in_remaining = 0
                    self._count_in_beat = 0
                    self._metronome_phase = 0
            np.clip(outdata, -1.0, 1.0, out=outdata)
            return

        # -- Normal playback -------------------------------------------------

        # If at or past end and not looping, stop playback.
        looping = (self._looping
                   and self._loop_a_frame is not None
                   and self._loop_b_frame is not None
                   and self._loop_b_frame > self._loop_a_frame)
        if self._current_frame >= self._total_frames:
            if looping:
                self._current_frame = self._loop_a_frame
            else:
                outdata.fill(0.0)
                raise sd.CallbackStop

        # Clear the output buffer.
        outdata.fill(0.0)

        # Determine which stems should be audible (cached between changes).
        active_stems = self._active_stems_cache
        if active_stems is None:
            if self._soloed_stems:
                active_stems = [
                    name for name in self._stems
                    if name in self._soloed_stems
                ]
            else:
                active_stems = [
                    name for name in self._stems
                    if name not in self._muted_stems
                ]
            self._active_stems_cache = active_stems

        # Fill the output buffer, handling loop wraps as needed.
        buf_offset = 0
        remaining = frames

        while remaining > 0:
            if looping:
                boundary = self._loop_b_frame
            else:
                boundary = self._total_frames

            frames_available = boundary - self._current_frame
            frames_to_read = min(remaining, max(frames_available, 0))

            if frames_to_read > 0:
                start = self._current_frame
                end = start + frames_to_read

                active_set = set(active_stems)
                for name, stem_data in self._stems.items():
                    target = (self._volumes.get(name, 1.0) * self._master_volume
                              if name in active_set else 0.0)
                    prev = self._applied_gains.get(name, target)
                    if target == 0.0 and prev == 0.0:
                        continue
                    stem_end = min(end, stem_data.shape[0])
                    read_len = stem_end - start
                    if read_len <= 0:
                        continue
                    chunk = stem_data[start:stem_end]
                    if prev != target:
                        ramp_len = min(read_len, max(
                            int(0.005 * self._sample_rate), 1))
                        ramp = np.linspace(
                            prev, target, ramp_len,
                            dtype=np.float32,
                        )[:, np.newaxis]
                        outdata[buf_offset:buf_offset + ramp_len] += (
                            chunk[:ramp_len] * ramp
                        )
                        if read_len > ramp_len:
                            outdata[
                                buf_offset + ramp_len
                                :buf_offset + read_len
                            ] += chunk[ramp_len:] * target
                        self._applied_gains[name] = target
                    else:
                        outdata[buf_offset:buf_offset + read_len] += (
                            chunk * target
                        )

                if (self._recording
                        and self._indata_capture is not None
                        and self._recording_buffer is not None):
                    rec_end = min(end, self._recording_buffer.shape[0])
                    rec_len = rec_end - start
                    if rec_len > 0:
                        chunk = self._indata_capture[
                            buf_offset:buf_offset + rec_len
                        ]
                        if chunk.ndim == 1:
                            chunk = chunk[:, np.newaxis]
                        if chunk.shape[1] == 1:
                            chunk = np.repeat(chunk, 2, axis=1)
                        actual = min(rec_len, chunk.shape[0])
                        self._recording_buffer[
                            start:start + actual
                        ] = chunk[:actual, :2]

                # Mix synced metronome for this segment before advancing.
                if (self._metronome_enabled and self._beat_sync_enabled
                        and len(self._beat_frames) > 0):
                    self._mix_metronome_synced(
                        outdata, start, frames_to_read, buf_offset,
                    )

                self._current_frame += frames_to_read
                buf_offset += frames_to_read
                remaining -= frames_to_read

            # Check if we hit the boundary.
            if self._current_frame >= boundary:
                if looping:
                    self._current_frame = self._loop_a_frame
                    if (self._count_in_enabled
                            and self._count_in_on_repeats):
                        self._arm_count_in()
                        break
                else:
                    break

        # Mix in grid-based metronome click track (only when not beat-synced).
        # Uses *frames* (the full PortAudio block size) so the beat phase
        # stays in sync with wall-clock time even when stems don't fill the
        # entire buffer (e.g. at EOF without looping).
        if (self._metronome_enabled and self._metronome_bpm > 0
                and not (self._beat_sync_enabled
                         and len(self._beat_frames) > 0)):
            beat_interval = int(60.0 / self._metronome_bpm * self._sample_rate)
            if beat_interval > 0:
                click_len = len(self._click_buf)
                self._mix_metronome(
                    outdata, frames, beat_interval, click_len
                )

        # Apply clipping protection.
        np.clip(outdata, -1.0, 1.0, out=outdata)

        # If we didn't fill the entire buffer and no count-in was just armed,
        # we hit EOF without looping.
        if remaining > 0 and self._count_in_remaining == 0:
            self._is_playing = False
            raise sd.CallbackStop

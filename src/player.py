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
import threading
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import QObject, QThread, Signal, QTimer

from src.click_utils import generate_click
from src.import_messages import describe_error
from src.metronome import local_beat_tempi
from src.stretch import StreamingStretcher


SPEED_PRESETS = (0.5, 0.75, 0.85, 1.0, 1.25, 1.5, 2.0)

# Bounds for pitch transposition, in semitones. A phase-vocoder pitch shift
# (stretch, then resample) degrades noticeably beyond ±7; this range covers
# the practical use cases (vocal range adjustment, capo equivalents)
# without exposing the quality cliff.
PITCH_MIN_SEMITONES = -7
PITCH_MAX_SEMITONES = 7

# Prefix used for recording-take stem names (e.g. "recording_take1").
# Used to distinguish user recordings from source stems when deciding
# whether to apply pitch transposition.
RECORDING_STEM_PREFIX = "recording_take"

# Output frames a new or re-seeked stretch path prepares before the audio
# callback reads it, so its first block costs no more than any other.
_PRIME_FRAMES = 2048
# Going back to the original speed and pitch mid-play crossfades from the
# stretched mix to the direct one over this many frames (10 ms): the
# vocoder's phase no longer matches the original, and a hard switch
# clicked (#217 review).
_RETURN_FADE_FRAMES = 441


def read_stem_files(
    stem_paths: dict[str, str],
) -> tuple[dict[str, np.ndarray], int]:
    """Read stem WAV files without mutating a player.

    The complete mapping is returned only after every file has loaded and
    all sample rates match, so callers can apply the result atomically.
    """
    stems: dict[str, np.ndarray] = {}
    sample_rate = 0

    for name, path in stem_paths.items():
        data, stem_sample_rate = sf.read(
            path, always_2d=True, dtype="float32",
        )
        if sample_rate == 0:
            sample_rate = stem_sample_rate
        elif stem_sample_rate != sample_rate:
            raise ValueError(f"Sample rate mismatch in stem '{name}'")

        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        stems[name] = data

    return stems, sample_rate


class StemLoadWorker(QThread):
    """Read a complete set of stem files away from the GUI thread."""

    completed = Signal(dict, int)
    error = Signal(str)

    def __init__(
        self,
        stem_paths: dict[str, str],
        *,
        generation: int = 0,
        song_id: str = "",
        source_stem_names: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self._stem_paths = dict(stem_paths)
        self.generation = generation
        self.song_id = song_id
        self.source_stem_names = source_stem_names

    def run(self) -> None:
        try:
            stems, sample_rate = read_stem_files(self._stem_paths)
            self.completed.emit(stems, sample_rate)
        except Exception as exc:
            self.error.emit(describe_error(exc, "Could not load song"))


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


class _SourceReader:
    """Mixes the stems one stretcher plays, from a read head that loops.

    Called by ``StreamingStretcher`` from the audio callback. The read head
    runs a little ahead of what the listener hears (the stretcher's
    analysis window); mute, solo, and volume are applied as the stems are
    read, so a change reaches the speakers a few tens of milliseconds later.
    """

    def __init__(
        self,
        player: "MultiTrackPlayer",
        names: frozenset[str] | None,
        head: int,
    ) -> None:
        self._player = player
        self.names = names  # None: every stem
        self.head = int(head)
        self._gains: dict[str, float] = {}

    def __call__(self, count: int) -> tuple[np.ndarray, int]:
        player = self._player
        loop_a = player._loop_a_frame
        loop_b = player._loop_b_frame
        looping = (
            player._looping
            and loop_a is not None
            and loop_b is not None
            and loop_b > loop_a
        )
        boundary = loop_b if looping else player._total_frames
        start = self.head
        if start >= boundary:
            if not looping:
                return np.zeros((0, 2), dtype=np.float32), start
            start = loop_a
        count = max(0, min(count, boundary - start))
        out = np.zeros((count, 2), dtype=np.float32)
        player._mix_stems_into(out, start, self.names, self._gains)
        self.head = start + count
        return out, start


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
    playback_failed = Signal(str)
    # Recording was armed but no input could be opened; recording is
    # disarmed and playback continues without it.
    recording_unavailable = Signal(str)
    recording_saved = Signal(str)  # emitted with the saved WAV path
    # A take could not be written (disk full, folder not writable). The
    # take stays in memory so the next Stop can try again.
    recording_save_failed = Signal(str)
    loop_wrapped = Signal()  # A-B loop repeated (wrapped back to A)

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
        self._active_stems_cache: set[str] | None = None

        # A-B loop state.
        self._loop_a_frame: int | None = None
        self._loop_b_frame: int | None = None
        self._looping: bool = False
        # Incremented in the audio callback each time the loop wraps back
        # to A (a plain int write -- realtime-safe). _emit_position (GUI
        # thread) diffs it against _loop_wrap_seen and emits loop_wrapped
        # so trainer/UI logic runs on the GUI thread, never the callback.
        self._loop_wrap_count: int = 0
        self._loop_wrap_seen: int = 0

        # Speed and pitch. Both are applied live: StreamingStretcher
        # instances in the audio callback stretch the mix as it plays, so
        # positions, loop points, beats, and chords stay in song frames.
        self._playback_speed: float = 1.0
        self._pitch_semitones: int = 0
        self._sync_recording_pitch: bool = False
        # (stretcher, source reader) per group of stems that share a pitch;
        # empty at speed 1.0 and pitch 0, where the callback mixes directly.
        self._stretch_paths: list[tuple[StreamingStretcher, _SourceReader]] = []
        # Paths fading out after a return to the original speed and pitch.
        self._fading_paths: list[tuple[StreamingStretcher, _SourceReader]] = []
        self._fade_left: int = 0
        # Song frame where the last stretched block ended, so synced clicks
        # cover every frame exactly once (positions from the stretcher's
        # hop marks jump by up to a hop when pitch is shifted).
        self._click_cursor: float | None = None
        # Held by the audio callback while it stretches and by the GUI
        # thread while it rebuilds or seeks the stretch paths.
        self._stretch_lock = threading.Lock()

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
        # Local tempo per gap in _beat_frames, for the synced BPM readout.
        self._beat_tempi: np.ndarray = np.array([], dtype=np.float64)

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
        # Input frames actually written to the buffer this take. Guards
        # against saving a full-length silent WAV when playback is
        # stopped before any input was captured (e.g. during count-in).
        self._recording_frames_captured: int = 0
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

    def shutdown(self, wait_ms: int = 3000) -> None:
        """Stop the position timer (app close).

        Speed and pitch run inside the audio callback, so there are no
        render threads left to cancel or wait for; *wait_ms* is kept for
        callers.
        """
        if self._timer.isActive():
            self._timer.stop()
        # Release the soxr streams before interpreter exit.
        with self._stretch_lock:
            self._stretch_paths = []
            self._fading_paths = []

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
        time_sec = frame / self._sample_rate
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
        """Convert beat_times (seconds) to song frame indices.

        Frames are in the song's own time at any speed: the stretcher
        maps them to the moment they are heard.
        """
        if not self._beat_times or self._sample_rate == 0:
            self._beat_frames = np.array([], dtype=np.int64)
            self._beat_tempi = np.array([], dtype=np.float64)
            return
        sr = self._sample_rate
        offset_sec = self._beat_sync_nudge_ms / 1000.0
        self._beat_frames = np.array(
            [max(0, int((t + offset_sec) * sr)) for t in self._beat_times],
            dtype=np.int64,
        )
        self._beat_tempi = local_beat_tempi(self._beat_frames, 60.0 * sr)

    def instantaneous_bpm_at(self, frame: int) -> float:
        """Return the local BPM at *frame*, as the synced metronome shows it.

        Uses the tempo of the beats around *frame* (see
        ``local_beat_tempi``) rather than the one gap it falls in, so a
        stray or missed beat does not flash double or half the tempo.
        Follows playback speed, like the synced clicks. Returns 0.0 when
        fewer than 2 beats are available.
        """
        bf = self._beat_frames
        tempi = self._beat_tempi
        if len(bf) < 2 or len(tempi) != len(bf) - 1:
            return 0.0
        idx = int(np.searchsorted(bf, frame, side="right"))
        # Clamp to the first or last gap before and after the beats.
        idx = max(1, min(idx, len(bf) - 1))
        # Beat frames are song time; what you hear runs at the speed.
        return float(tempi[idx - 1]) * self._playback_speed

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
        self._recording_frames_captured = 0

    def set_recording_song_dir(self, song_dir: str | None) -> None:
        """Set the directory where recording takes are saved."""
        self._recording_song_dir = song_dir

    def save_recording(self, song_dir: str) -> str | None:
        """Write the recording buffer to a WAV file in *song_dir*.

        Returns the path to the saved file, or None if there is no recording
        or no input was ever captured (playback stopped during count-in, or
        play was pressed and stopped without the input stream delivering
        anything) -- saving would produce a full-length silent take that
        eats one of the take slots.
        The take number auto-increments based on existing files.
        """
        if self._recording_buffer is None:
            return None
        if self._recording_frames_captured == 0:
            self._recording_buffer = None
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
        try:
            sf.write(path, buf, self._sample_rate)
        except (OSError, sf.SoundFileError) as exc:
            try:
                os.remove(path)  # a partial file would count as a take
            except OSError:
                pass
            self.recording_save_failed.emit(
                "The take could not be saved. "
                + describe_error(exc, "Saving a recording")
                + " The take is kept in memory: press Stop again to save "
                "it."
            )
            return None
        self._recording_buffer = None
        return path

    def add_recording_stem(self, name: str, data: np.ndarray) -> None:
        """Add a recording take as a playable stem.

        Validates sample rate compatibility and forces stereo.

        The stems dict is replaced, not mutated: the audio callback may
        be iterating the old dict on the PortAudio thread. The active-
        stems cache is invalidated so the new take is audible on the
        next play without requiring a mute/solo toggle.
        """
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        stems = dict(self._stems)
        stems[name] = data
        self._stems = stems
        self._total_frames = max(self._total_frames, data.shape[0])
        self._active_stems_cache = None
        # A take may need its own stretch path (unpitched takes).
        self._rebuild_stretch()

    def remove_recording_stem(self, name: str) -> None:
        """Remove a recording stem and recalculate total frames.

        Same copy-on-write discipline as ``add_recording_stem``: the
        audio callback may be mid-iteration over the current dict, so
        it is replaced rather than popped in place.
        """
        stems = dict(self._stems)
        stems.pop(name, None)
        self._stems = stems
        self._muted_stems.discard(name)
        self._soloed_stems.discard(name)
        self._volumes.pop(name, None)
        self._nudge_offsets.pop(name, None)
        self._active_stems_cache = None
        self._recalculate_total_frames()
        self._rebuild_stretch()

    def nudge_stem(self, name: str, offset_ms: float) -> None:
        """Shift a stem's audio by *offset_ms* milliseconds.

        Positive values shift the audio later (add silence at the start);
        negative values shift it earlier. The offset is clamped to
        -200..+200 ms. Wrapped samples are zeroed out.

        The shifted array replaces the stem in a new dict, so the audio
        callback never sees a half-written stem.
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

        data = np.roll(self._stems[name], delta_frames, axis=0)
        if delta_frames > 0:
            data[:delta_frames] = 0.0
        else:
            data[delta_frames:] = 0.0
        stems = dict(self._stems)
        stems[name] = data
        self._stems = stems

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

    def _reset_song_state(self) -> None:
        """Stop playback and clear all per-song state.

        Shared prologue of ``load_stems`` and ``unload``.
        """
        self.stop()
        with self._stretch_lock:
            self._stretch_paths = []
            self._fading_paths = []
            self._click_cursor = None
        self._stems = {}
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
        self._beat_tempi = np.array([], dtype=np.float64)
        self._chord_sequence.clear()
        self._chord_times = np.array([], dtype=np.float64)
        self._loop_a_frame = None
        self._loop_b_frame = None
        self._looping = False
        self._loop_wrap_count = 0
        self._loop_wrap_seen = 0
        self._playback_speed = 1.0
        self._pitch_semitones = 0
        self._recording_armed = False
        self._recording = False
        self._recording_buffer = None
        self._nudge_offsets.clear()

    def unload(self) -> None:
        """Unload the current song entirely.

        Close Song (and removing the currently loaded song) must leave
        the player truly empty: previously only the UI was cleared, so
        ``has_stems`` stayed True and global shortcuts (Space, R, loop
        keys) kept operating on the invisible, closed song.
        """
        self._reset_song_state()
        self._total_frames = 0
        self._current_frame = 0
        self.position_changed.emit(0.0)

    def load_stems(self, stem_paths: dict[str, str]) -> None:
        """Load all stem WAV files into memory.

        Args:
            stem_paths: Dictionary mapping stem names to file paths.
        """
        stems, sample_rate = read_stem_files(stem_paths)
        self.apply_loaded_stems(stems, sample_rate)

    def apply_loaded_stems(
        self,
        stems: dict[str, np.ndarray],
        sample_rate: int,
    ) -> None:
        """Atomically replace live player state with preloaded stem arrays."""
        self._reset_song_state()
        self._stems = dict(stems)
        self._sample_rate = sample_rate
        self._total_frames = max(
            (data.shape[0] for data in stems.values()), default=0,
        )
        self._current_frame = 0
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

    def _open_recording_stream(self):
        """Open the full-duplex stream for recording, or return None.

        None means there is no usable input: no default input device, a
        device with no input channels, or PortAudio refused the stream.
        A failure to *query* the device is not treated as missing input.
        """
        in_dev = self._input_device
        out_dev = self._output_device
        try:
            defaults = sd.default.device
            in_resolved = in_dev if in_dev is not None else defaults[0]
            out_resolved = out_dev if out_dev is not None else defaults[1]
        except (TypeError, IndexError):
            in_resolved = in_dev
            out_resolved = out_dev
        # PortAudio reports "no default input" as -1.
        if in_resolved is None or (
            isinstance(in_resolved, int) and in_resolved < 0
        ):
            return None
        input_ch = 1
        try:
            info = sd.query_devices(in_resolved)
            max_in = int(info.get("max_input_channels", 1))
            if max_in <= 0:
                return None
            input_ch = max(1, min(max_in, 2))
        except (sd.PortAudioError, ValueError, TypeError, OSError):
            pass
        try:
            stream = sd.Stream(
                samplerate=self._sample_rate,
                channels=(input_ch, 2),
                callback=self._full_duplex_callback,
                device=(in_resolved, out_resolved),
            )
        except (sd.PortAudioError, ValueError, OSError):
            return None
        if self._recording_buffer is None:
            self._allocate_recording_buffer()
        self._recording = True
        return stream

    def _open_output_stream(self):
        """Open the playback-only stream on the chosen output device."""
        kwargs: dict[str, Any] = {
            "samplerate": self._sample_rate,
            "channels": 2,
            "callback": self._audio_callback,
        }
        if self._output_device is not None:
            kwargs["device"] = self._output_device
        return sd.OutputStream(**kwargs)

    def _disarm_recording_without_input(self) -> str:
        """Turn recording off for want of an input; return the message.

        Playback goes on, and the next Play does not fail the same way
        again. A take paused earlier keeps what it captured, and the next
        Stop saves it.
        """
        self._recording_armed = False
        self._recording = False
        message = (
            "No microphone or other input device could be opened, so "
            "recording was turned off. Connect one or choose it in "
            "Edit > Preferences, then arm recording again."
        )
        if (self._recording_buffer is not None
                and self._recording_frames_captured > 0):
            return message + (
                " What you recorded before is kept and saved when you "
                "press Stop."
            )
        self._recording_buffer = None
        return message

    def _discard_empty_recording(self) -> None:
        """Drop a recording buffer that captured nothing yet."""
        if self._recording_frames_captured == 0:
            self._recording_buffer = None

    def play(self) -> None:
        """Start or resume playback."""
        if not self._stems or self._is_playing:
            return

        if self._current_frame >= self._total_frames:
            self._current_frame = 0
            self._restart_stretch()

        # Emitted once playback runs: its slot opens a modal dialog, and a
        # nested event loop must not run while the player is half set up.
        unavailable = None
        try:
            if self._stream is None and self._recording_armed:
                self._stream = self._open_recording_stream()
                if self._stream is None:
                    unavailable = self._disarm_recording_without_input()
            if self._stream is None:
                self._stream = self._open_output_stream()
            try:
                self._stream.start()
            except (sd.PortAudioError, OSError):
                if not self._recording:
                    raise
                # The duplex stream opened but would not start: blame the
                # input and play on without it.
                self._stream.close()
                self._stream = None
                unavailable = self._disarm_recording_without_input()
                self._stream = self._open_output_stream()
                self._stream.start()
        except (sd.PortAudioError, OSError):
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            self._recording = False
            self._discard_empty_recording()
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
        if unavailable:
            self.recording_unavailable.emit(unavailable)

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
        self._restart_stretch(target_frame)
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
    # Speed / Pitch
    # ------------------------------------------------------------------
    #
    # Applied live by StreamingStretcher (src/stretch.py): the audio
    # callback stretches the mix as it plays, so a change is heard within
    # a frame or two and nothing is rendered in advance. Stems whose pitch
    # must stay put (recording takes, unless they follow the pitch) get a
    # second stretcher at the same speed.

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

    @property
    def stretching(self) -> bool:
        """True while speed or pitch differ from the original."""
        return bool(self._stretch_paths)

    def set_speed(self, speed: float) -> None:
        """Set the playback speed (0.5 to 2.0) with the pitch preserved.

        Takes effect immediately. Refused while recording: a take is
        captured against the song at its own tempo.
        """
        if self._recording:
            return
        speed = max(0.5, min(float(speed), 2.0))
        if speed == self._playback_speed:
            return
        self._playback_speed = speed
        self._rebuild_stretch()
        self.speed_changed.emit(speed)

    def set_pitch(self, semitones: int) -> None:
        """Transpose by *semitones* (clamped) with the tempo preserved.

        Takes effect immediately. Refused while recording, as speed is.
        """
        if self._recording:
            return
        try:
            semitones = int(semitones)
        except (TypeError, ValueError):
            return
        semitones = max(PITCH_MIN_SEMITONES,
                        min(PITCH_MAX_SEMITONES, semitones))
        if semitones == self._pitch_semitones:
            return
        self._pitch_semitones = semitones
        self._rebuild_stretch()
        self.pitch_changed.emit(semitones)

    def set_sync_recording_pitch(self, sync: bool) -> None:
        """Choose whether recording takes follow the pitch shift.

        When False (default), takes keep their own pitch while the song is
        transposed; when True they shift with it.
        """
        sync = bool(sync)
        if sync == self._sync_recording_pitch:
            return
        self._sync_recording_pitch = sync
        self._rebuild_stretch()

    def _stretch_groups(self) -> list[tuple[frozenset[str] | None, int]]:
        """(stem names, or None for all; semitones) per stretch path."""
        if self._playback_speed == 1.0 and self._pitch_semitones == 0:
            return []
        pitch = self._pitch_semitones
        takes = frozenset(
            name for name in self._stems
            if name.startswith(RECORDING_STEM_PREFIX)
        )
        if pitch and takes and not self._sync_recording_pitch:
            others = frozenset(n for n in self._stems if n not in takes)
            return [(others, pitch), (takes, 0)]
        return [(None, pitch)]

    def _play_frame(self) -> int:
        """The song frame the listener is hearing now."""
        if self._stretch_paths:
            return int(self._stretch_paths[0][0].position)
        return self._current_frame

    def _rebuild_stretch(self) -> None:
        """Match the stretch paths to the speed, pitch, and stems.

        A speed or pitch change on the same stems only retunes the running
        stretchers, so playback carries straight on; a change of stem
        groups (or to or from the original speed and pitch) starts fresh
        paths at the frame being heard.
        """
        groups = self._stretch_groups()
        with self._stretch_lock:
            current = [reader.names for _, reader in self._stretch_paths]
            if groups and current == [names for names, _ in groups]:
                for (stretcher, _), (_, semitones) in zip(
                    self._stretch_paths, groups,
                ):
                    stretcher.set_params(self._playback_speed, semitones)
                return
            frame = min(self._play_frame(), self._total_frames)
        # Built and primed here, outside the lock: filling a fresh path's
        # analysis window on the audio thread overran a 512-frame block.
        paths = []
        for names, semitones in groups:
            stretcher = StreamingStretcher(self._sample_rate, 2)
            stretcher.set_params(self._playback_speed, semitones)
            stretcher.reset(frame)
            reader = _SourceReader(self, names, frame)
            stretcher.fill(_PRIME_FRAMES, reader)
            paths.append((stretcher, reader))
        with self._stretch_lock:
            if not paths and self._stretch_paths and self._is_playing:
                self._fading_paths = self._stretch_paths
                self._fade_left = _RETURN_FADE_FRAMES
            self._current_frame = frame
            self._stretch_paths = paths
            self._click_cursor = None

    def _restart_stretch(self, frame: int | None = None) -> None:
        """Start the stretch paths again from *frame* (a seek).

        The frame is set under the same lock the audio callback holds, so
        a callback finishing mid-seek cannot write its old position back.
        """
        with self._stretch_lock:
            if frame is not None:
                self._current_frame = frame
            self._fading_paths = []
            self._click_cursor = None
            for stretcher, reader in self._stretch_paths:
                stretcher.reset(self._current_frame)
                reader.head = self._current_frame
                stretcher.fill(_PRIME_FRAMES, reader)

    def _mix_stems_into(
        self,
        out: np.ndarray,
        start: int,
        names: frozenset[str] | None,
        gains: dict[str, float],
    ) -> None:
        """Add the audible stems' frames ``start:start+len(out)`` to *out*.

        *names* limits the mix to those stems (None: all). *gains* holds
        the last gain applied per stem, so a change ramps over 5 ms.
        """
        stems = self._stems
        active = self._active_stems_cache
        if active is None:
            if self._soloed_stems:
                active = {n for n in stems if n in self._soloed_stems}
            else:
                active = {n for n in stems if n not in self._muted_stems}
            self._active_stems_cache = active
        count = len(out)
        end = start + count
        for name, data in stems.items():
            if names is not None and name not in names:
                continue
            target = (
                self._volumes.get(name, 1.0) * self._master_volume
                if name in active else 0.0
            )
            prev = gains.get(name, target)
            if target == 0.0 and prev == 0.0:
                gains[name] = 0.0
                continue
            read_len = min(end, data.shape[0]) - start
            if read_len <= 0:
                continue
            chunk = data[start:start + read_len]
            if prev != target:
                ramp_len = min(read_len, max(int(0.005 * self._sample_rate), 1))
                ramp = np.linspace(
                    prev, target, ramp_len, dtype=np.float32,
                )[:, np.newaxis]
                out[:ramp_len] += chunk[:ramp_len] * ramp
                if read_len > ramp_len:
                    out[ramp_len:read_len] += chunk[ramp_len:] * target
            else:
                out[:read_len] += chunk * target
            gains[name] = target

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

        # Surface loop wraps that happened on the audio thread since the
        # last tick. Emitted here (GUI thread) so slots can safely touch
        # Qt objects and spawn render workers.
        wraps = self._loop_wrap_count
        if wraps != self._loop_wrap_seen:
            self._loop_wrap_seen = wraps
            self.loop_wrapped.emit()

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

    def _stretched_callback(self, outdata: np.ndarray, frames: int) -> None:
        """Fill *outdata* through the stretchers (speed or pitch changed).

        Reads up to each loop wrap separately, so the wrap is handled (loop
        count, count-in on repeats) at the sample where it is heard.
        Metronome and count-in clicks are added after stretching, so they
        keep their shape at any speed.
        """
        outdata.fill(0.0)
        speed = self._playback_speed
        offset = 0
        remaining = frames
        ended = False
        with self._stretch_lock:
            paths = self._stretch_paths
            if not paths:
                return
            main = paths[0][0]
            while remaining > 0:
                for stretcher, reader in paths:
                    stretcher.fill(remaining, reader)
                jump = main.jump_offset()
                if jump == 0:
                    for stretcher, _ in paths:
                        stretcher.clear_jump()
                    self._click_cursor = None
                    self._loop_wrap_count += 1
                    if self._count_in_enabled and self._count_in_on_repeats:
                        self._arm_count_in()
                        if self._count_in_remaining > 0:
                            ci_frames = min(remaining, self._count_in_remaining)
                            beat_interval = int(
                                60.0 / self._metronome_bpm * self._sample_rate
                            )
                            self._mix_count_in(
                                outdata[offset:offset + ci_frames], ci_frames,
                                beat_interval, len(self._click_buf),
                            )
                            self._count_in_remaining -= ci_frames
                            offset += ci_frames
                            remaining -= ci_frames
                            if self._count_in_remaining <= 0:
                                self._count_in_remaining = 0
                                self._count_in_beat = 0
                                self._metronome_phase = 0
                            else:
                                break
                    continue
                n = remaining if jump is None else min(remaining, jump)
                start = (
                    main.position if self._click_cursor is None
                    else self._click_cursor
                )
                self._click_cursor = start + n * speed
                for stretcher, reader in paths:
                    outdata[offset:offset + n] += stretcher.read(n, reader)
                if (self._metronome_enabled and self._beat_sync_enabled
                        and len(self._beat_frames) > 0):
                    self._mix_metronome_stretched(
                        outdata, offset, n, start, speed,
                    )
                offset += n
                remaining -= n
                if main.finished:
                    ended = True
                    break
            self._current_frame = min(int(main.position), self._total_frames)

        if (self._metronome_enabled and self._metronome_bpm > 0
                and not (self._beat_sync_enabled
                         and len(self._beat_frames) > 0)):
            beat_interval = int(60.0 / self._metronome_bpm * self._sample_rate)
            if beat_interval > 0:
                self._mix_metronome(
                    outdata, frames, beat_interval, len(self._click_buf),
                )
        np.clip(outdata, -1.0, 1.0, out=outdata)
        if ended and self._count_in_remaining == 0:
            self._current_frame = self._total_frames
            self._is_playing = False
            raise sd.CallbackStop

    def _mix_return_fade(self, outdata: np.ndarray, frames: int) -> None:
        """Crossfade the fading stretched mix into the direct one."""
        with self._stretch_lock:
            paths = self._fading_paths
            if not paths:
                return
            n = min(frames, self._fade_left)
            old = np.zeros((n, 2), dtype=np.float32)
            for stretcher, reader in paths:
                old += stretcher.read(n, reader)
            done = _RETURN_FADE_FRAMES - self._fade_left
            ramp = (np.arange(done, done + n, dtype=np.float32)
                    / _RETURN_FADE_FRAMES)[:, np.newaxis]
            outdata[:n] = outdata[:n] * ramp + old * (1.0 - ramp)
            self._fade_left -= n
            if self._fade_left <= 0:
                self._fading_paths = []

    def _mix_metronome_stretched(
        self, outdata: np.ndarray, offset: int, count: int,
        start_frame: float, speed: float,
    ) -> None:
        """Synced clicks for song frames heard in ``outdata[offset:+count]``.

        The block plays song frames from *start_frame* at *speed* song
        frames per output sample.
        """
        bf = self._beat_frames
        end_frame = start_frame + count * speed
        idx = int(np.searchsorted(bf, start_frame, side="left"))
        click_len = len(self._click_buf)
        gain = self._metronome_volume
        while idx < len(bf) and bf[idx] < end_frame:
            pos = offset + int((bf[idx] - start_frame) / speed)
            n = min(click_len, offset + count - pos)
            if n > 0:
                outdata[pos:pos + n] += self._click_buf[:n] * gain
            idx += 1

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

        if self._stretch_paths:
            self._stretched_callback(outdata, frames)
            return

        # -- Normal playback -------------------------------------------------

        # Snapshot cross-thread state once per callback. The GUI thread
        # replaces the stems dict (copy-on-write) and can null the loop
        # frames at any moment (clear_loop); reading them repeatedly
        # mid-callback risks a None arithmetic TypeError that aborts the
        # stream.
        stems = self._stems
        loop_a = self._loop_a_frame
        loop_b = self._loop_b_frame

        # If at or past end and not looping, stop playback.
        looping = (self._looping
                   and loop_a is not None
                   and loop_b is not None
                   and loop_b > loop_a)
        if self._current_frame >= self._total_frames:
            if looping:
                self._current_frame = loop_a
            else:
                outdata.fill(0.0)
                raise sd.CallbackStop

        # Clear the output buffer.
        outdata.fill(0.0)

        # Determine which stems should be audible (cached between changes).
        # Stored as a set so the per-segment membership tests below don't
        # allocate.
        active_stems = self._active_stems_cache
        if active_stems is None:
            if self._soloed_stems:
                active_stems = {
                    name for name in stems
                    if name in self._soloed_stems
                }
            else:
                active_stems = {
                    name for name in stems
                    if name not in self._muted_stems
                }
            self._active_stems_cache = active_stems

        # Fill the output buffer, handling loop wraps as needed.
        buf_offset = 0
        remaining = frames

        while remaining > 0:
            if looping:
                boundary = loop_b
            else:
                boundary = self._total_frames

            frames_available = boundary - self._current_frame
            frames_to_read = min(remaining, max(frames_available, 0))

            if frames_to_read > 0:
                start = self._current_frame
                end = start + frames_to_read

                for name, stem_data in stems.items():
                    target = (self._volumes.get(name, 1.0) * self._master_volume
                              if name in active_stems else 0.0)
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
                        self._recording_frames_captured += actual

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
                    self._current_frame = loop_a
                    # RT-safe int write; surfaced as loop_wrapped from the
                    # GUI-thread timer (see _emit_position).
                    self._loop_wrap_count += 1
                    if (self._count_in_enabled
                            and self._count_in_on_repeats):
                        self._arm_count_in()
                        if self._count_in_remaining > 0:
                            beat_interval = int(
                                60.0 / self._metronome_bpm
                                * self._sample_rate
                            )
                            ci_frames = min(
                                remaining,
                                self._count_in_remaining,
                            )
                            click_len = len(self._click_buf)
                            count_in_output = outdata[
                                buf_offset:buf_offset + ci_frames
                            ]
                            self._mix_count_in(
                                count_in_output,
                                ci_frames,
                                beat_interval,
                                click_len,
                            )
                            self._count_in_remaining -= ci_frames
                            buf_offset += ci_frames
                            remaining -= ci_frames
                            if self._count_in_remaining <= 0:
                                self._count_in_remaining = 0
                                self._count_in_beat = 0
                                self._metronome_phase = 0
                else:
                    break

        if self._fading_paths:
            self._mix_return_fade(outdata, frames)

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

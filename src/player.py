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
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from PySide6.QtCore import QObject, QThread, Signal, QTimer


SPEED_PRESETS = (0.5, 0.75, 0.85, 1.0, 1.25, 1.5, 2.0)


def _safe_disconnect(signal) -> None:
    """Disconnect all slots from *signal*, ignoring RuntimeError."""
    try:
        signal.disconnect()
    except RuntimeError:
        pass


def next_take_number(song_dir: str) -> int:
    """Return the next recording take number for *song_dir*."""
    existing = glob.glob(os.path.join(song_dir, "recording_take*.wav"))
    nums: list[int] = []
    for p in existing:
        base = os.path.basename(p)
        try:
            n = int(base.replace("recording_take", "").replace(".wav", ""))
            nums.append(n)
        except ValueError:
            continue
    return max(nums, default=0) + 1


class SpeedWorker(QThread):
    """Background thread for pitch-preserving time-stretch of all stems."""

    completed = Signal(dict)  # {name: stretched_ndarray}
    progress = Signal(int, int)  # (current_stem, total_stems)
    error = Signal(str)

    def __init__(self, stems: dict[str, np.ndarray], speed: float,
                 parent=None) -> None:
        super().__init__(parent)
        self._stems = stems
        self._speed = speed

    def run(self) -> None:
        try:
            self._stretch()
        except Exception as exc:
            self.error.emit(str(exc))

    def _stretch(self) -> None:
        total = len(self._stems)
        stretched = {}
        for i, (name, data) in enumerate(self._stems.items()):
            # Preserve the peak amplitude after phase-vocoder stretch.
            original_peak = np.max(np.abs(data))

            # librosa.effects.time_stretch works on mono; process each channel.
            channels = []
            for ch_idx in range(data.shape[1]):
                mono = data[:, ch_idx].astype(np.float32)
                stretched_ch = librosa.effects.time_stretch(
                    mono, rate=self._speed
                )
                channels.append(stretched_ch)
            # Recombine to stereo, matching shortest channel.
            min_len = min(c.shape[0] for c in channels)
            stereo = np.column_stack([c[:min_len] for c in channels])
            stereo = stereo.astype(np.float32)

            # Normalize to match original peak level (phase vocoder can
            # reduce amplitude).
            stretched_peak = np.max(np.abs(stereo))
            if stretched_peak > 0 and original_peak > 0:
                stereo *= original_peak / stretched_peak

            stretched[name] = stereo
            self.progress.emit(i + 1, total)
        self.completed.emit(stretched)


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

        # A-B loop state.
        self._loop_a_frame: int | None = None
        self._loop_b_frame: int | None = None
        self._looping: bool = False

        # Speed / time-stretch state.
        self._playback_speed: float = 1.0
        self._original_stems: dict[str, np.ndarray] = {}
        self._speed_worker: SpeedWorker | None = None

        # Metronome state.
        self._metronome_enabled: bool = False
        self._metronome_bpm: float = 120.0
        self._metronome_volume: float = 0.5
        self._metronome_phase: int = 0
        self._click_buf: np.ndarray = self._generate_click(self._sample_rate)

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

        Returns a stereo numpy array (shape (N, 2), dtype float32) containing
        a 1000 Hz sine tone with an exponential decay envelope, lasting 30ms.
        """
        duration = 0.03  # 30 ms
        n_frames = int(sample_rate * duration)
        t = np.arange(n_frames, dtype=np.float32) / sample_rate
        tone = np.sin(2.0 * np.pi * 1000.0 * t)
        envelope = np.exp(-t / 0.006).astype(np.float32)  # ~6ms decay constant
        click_mono = (tone * envelope).astype(np.float32)
        return np.column_stack((click_mono, click_mono))

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

        Recording cannot be armed when playback speed is not 1.0x or when
        no stems are loaded.  Arming while already recording is a no-op.
        """
        if armed:
            if self._playback_speed != 1.0:
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
        self._recalculate_total_frames()

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
        self._detach_speed_worker()
        self._stems.clear()
        self._original_stems.clear()
        self._muted_stems.clear()
        self._soloed_stems.clear()
        self._volumes.clear()
        self._loop_a_frame = None
        self._loop_b_frame = None
        self._looping = False
        self._playback_speed = 1.0
        self._recording_armed = False
        self._recording = False
        self._recording_buffer = None

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
    # Speed / Time-Stretch
    # ------------------------------------------------------------------

    @property
    def speed(self) -> float:
        """Return the current playback speed multiplier."""
        return self._playback_speed

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

        if not self._original_stems:
            return

        # Detach any in-flight worker so its signals don't fire stale data.
        self._detach_speed_worker()

        if speed == 1.0:
            self._apply_stretched_stems(dict(self._original_stems))
            self.speed_changed.emit(speed)
            return

        self._speed_worker = SpeedWorker(
            self._original_stems, speed, parent=self
        )
        self._speed_worker.completed.connect(self._on_speed_ready)
        self._speed_worker.error.connect(self._on_speed_error)
        self._speed_worker.start()

    def _detach_speed_worker(self) -> None:
        """Disconnect and release the current speed worker, if any."""
        if self._speed_worker is not None:
            _safe_disconnect(self._speed_worker.completed)
            _safe_disconnect(self._speed_worker.error)
            _safe_disconnect(self._speed_worker.progress)
            # Let it finish in the background; don't block the UI.
            worker = self._speed_worker
            self._speed_worker = None
            worker.setParent(None)
            if worker.isRunning():
                worker.finished.connect(worker.deleteLater)
            else:
                worker.deleteLater()

    def _on_speed_ready(self, stretched: dict) -> None:
        """Swap in stretched stems and adjust frame indices."""
        self._apply_stretched_stems(stretched)
        self.speed_changed.emit(self._playback_speed)

    def _on_speed_error(self, message: str) -> None:
        """Handle stretch failure by restoring original speed."""
        self._playback_speed = 1.0
        self._apply_stretched_stems(dict(self._original_stems))
        self.speed_changed.emit(1.0)

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
        phase = self._metronome_phase

        # Continue any click that started in a previous callback.
        if phase < click_len:
            n = min(click_len - phase, frames_written)
            outdata[:n] += self._click_buf[phase:phase + n] * gain

        # Walk through the buffer finding beat boundaries.
        pos = beat_interval - phase  # Frames until next beat start
        while pos < frames_written:
            # Overlay click starting at this beat.
            n = min(click_len, frames_written - pos)
            if n > 0:
                outdata[pos:pos + n] += self._click_buf[:n] * gain
            pos += beat_interval

        # Update phase for the next callback.
        self._metronome_phase = (phase + frames_written) % beat_interval

    def _mix_count_in(self, outdata: np.ndarray, ci_frames: int,
                      beat_interval: int, click_len: int) -> None:
        """Overlay count-in clicks onto the output buffer.

        Uses its own phase counter (``_count_in_phase``) independent of the
        metronome phase so the two features don't interfere with each other.
        Updates ``_count_in_beat`` (1-based) for UI feedback.
        """
        gain = self._metronome_volume
        phase = self._count_in_phase

        if phase < click_len:
            n = min(click_len - phase, ci_frames)
            outdata[:n] += self._click_buf[phase:phase + n] * gain

        pos = beat_interval - phase
        while pos < ci_frames:
            n = min(click_len, ci_frames - pos)
            if n > 0:
                outdata[pos:pos + n] += self._click_buf[:n] * gain
            pos += beat_interval

        new_phase = (phase + ci_frames) % beat_interval
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

        # Determine which stems should be audible.
        if self._soloed_stems:
            active_stems = [
                name for name in self._stems if name in self._soloed_stems
            ]
        else:
            active_stems = [
                name for name in self._stems if name not in self._muted_stems
            ]

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

                for name in active_stems:
                    stem_data = self._stems[name]
                    stem_end = min(end, stem_data.shape[0])
                    read_len = stem_end - start
                    if read_len > 0:
                        gain = self._volumes.get(name, 1.0)
                        outdata[buf_offset:buf_offset + read_len] += (
                            stem_data[start:stem_end] * gain
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

        # Mix in metronome click track.  Use *frames* (the full PortAudio
        # block size) rather than buf_offset so the beat phase stays in sync
        # with wall-clock time even when stems don't fill the entire buffer
        # (e.g. at EOF without looping).
        if self._metronome_enabled and self._metronome_bpm > 0:
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

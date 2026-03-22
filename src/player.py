"""Multi-track audio player for synchronized playback of separated stems.

Uses `sounddevice` for zero-latency memory buffer mixing. Stems are loaded
into RAM entirely and summed dynamically inside the C-level audio callback,
allowing instant, click-free muting and soloing.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from PySide6.QtCore import QObject, QThread, Signal, QTimer


SPEED_PRESETS = (0.5, 0.75, 0.85, 1.0, 1.25, 1.5, 2.0)


class SpeedWorker(QThread):
    """Background thread for pitch-preserving time-stretch of all stems."""

    completed = Signal(dict)  # {name: stretched_ndarray}
    progress = Signal(int, int)  # (current_stem, total_stems)

    def __init__(self, stems: dict[str, np.ndarray], speed: float,
                 parent=None) -> None:
        super().__init__(parent)
        self._stems = stems
        self._speed = speed

    def run(self) -> None:
        total = len(self._stems)
        stretched = {}
        for i, (name, data) in enumerate(self._stems.items()):
            # librosa.effects.time_stretch works on mono; process each channel.
            channels = []
            for ch in range(data.shape[1]):
                mono = data[:, ch].astype(np.float32)
                stretched_ch = librosa.effects.time_stretch(
                    mono, rate=self._speed
                )
                channels.append(stretched_ch)
            # Recombine to stereo, matching shortest channel.
            min_len = min(ch.shape[0] for ch in channels)
            stereo = np.column_stack([ch[:min_len] for ch in channels])
            stretched[name] = stereo.astype(np.float32)
            self.progress.emit(i + 1, total)
        self.completed.emit(stretched)


class MultiTrackPlayer(QObject):
    """Audio player that mixes multiple stems in real-time.

    Signals:
        position_changed(float): Current playback position in seconds.
        state_changed(bool): Emitted when playback starts or stops.
        play_finished(): Emitted when the end of the track is reached.
    """

    position_changed = Signal(float)
    state_changed = Signal(bool)
    play_finished = Signal()
    speed_changed = Signal(float)

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

        # Hardware stream.
        self._stream: sd.OutputStream | None = None

        # UI updater.
        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30fps
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

    def load_stems(self, stem_paths: dict[str, str]) -> None:
        """Load all stem WAV files into memory.

        Args:
            stem_paths: Dictionary mapping stem names to file paths.
        """
        self.stop()
        self._stems.clear()
        self._original_stems.clear()
        self._muted_stems.clear()
        self._soloed_stems.clear()
        self._volumes.clear()
        self._loop_a_frame = None
        self._loop_b_frame = None
        self._looping = False
        self._playback_speed = 1.0

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

        self.position_changed.emit(0.0)

    def play(self) -> None:
        """Start or resume playback."""
        if not self._stems or self._is_playing:
            return

        if self._current_frame >= self._total_frames:
            self._current_frame = 0

        try:
            if self._stream is None:
                self._stream = sd.OutputStream(
                    samplerate=self._sample_rate,
                    channels=2,
                    callback=self._audio_callback,
                )
            self._stream.start()
        except sd.PortAudioError:
            # No audio device or device error — clean up and bail.
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            return

        self._is_playing = True
        self._timer.start()
        self.state_changed.emit(True)

    def pause(self) -> None:
        """Pause playback."""
        if not self._is_playing:
            return

        self._is_playing = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._timer.stop()
        self.state_changed.emit(False)

    def stop(self) -> None:
        """Stop playback and reset position to 0."""
        self.pause()
        self.seek(0.0)

    def seek(self, position_s: float) -> None:
        """Seek to a specific position in seconds.

        Args:
            position_s: Target time in seconds.
        """
        target_frame = int(position_s * self._sample_rate)
        target_frame = max(0, min(target_frame, self._total_frames))
        self._current_frame = target_frame
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
        self._playback_speed = speed

        if not self._original_stems:
            return

        # Cancel any in-flight stretch.
        if self._speed_worker is not None and self._speed_worker.isRunning():
            self._speed_worker.wait(5000)

        if speed == 1.0:
            self._apply_stretched_stems(dict(self._original_stems), 1.0)
            self.speed_changed.emit(speed)
            return

        self._speed_worker = SpeedWorker(
            self._original_stems, speed, parent=self
        )
        self._speed_worker.completed.connect(
            lambda stretched: self._on_speed_ready(stretched, speed)
        )
        self._speed_worker.start()

    def _on_speed_ready(self, stretched: dict, speed: float) -> None:
        """Swap in stretched stems and adjust frame indices."""
        self._apply_stretched_stems(stretched, speed)
        self.speed_changed.emit(speed)

    def _apply_stretched_stems(self, stems: dict, speed: float) -> None:
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
            self._timer.stop()
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self.state_changed.emit(False)
            self.play_finished.emit()
            return

        pos_s = self._current_frame / self._sample_rate
        self.position_changed.emit(pos_s)

    def _audio_callback(self, outdata: np.ndarray, frames: int,
                        time_info: dict, status: sd.CallbackFlags) -> None:
        """PortAudio callback for pushing mixed audio to the hardware.

        When A-B looping is active, playback wraps from loop_b back to loop_a
        instead of stopping at the end of the track.
        """
        if not self._is_playing or self._current_frame >= self._total_frames:
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

        # Determine the effective playback boundary.
        looping = (self._looping
                   and self._loop_a_frame is not None
                   and self._loop_b_frame is not None
                   and self._loop_b_frame > self._loop_a_frame)

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

                self._current_frame += frames_to_read
                buf_offset += frames_to_read
                remaining -= frames_to_read

            # Check if we hit the boundary.
            if self._current_frame >= boundary:
                if looping:
                    # Wrap back to loop A.
                    self._current_frame = self._loop_a_frame
                else:
                    # End of track.
                    break

        # Apply clipping protection.
        np.clip(outdata, -1.0, 1.0, out=outdata)

        # If we didn't fill the entire buffer, we hit EOF without looping.
        if remaining > 0:
            self._is_playing = False
            raise sd.CallbackStop

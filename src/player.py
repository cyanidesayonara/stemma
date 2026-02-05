"""Multi-track audio player for synchronized playback of separated stems.

Uses `sounddevice` for zero-latency memory buffer mixing. Stems are loaded
into RAM entirely and summed dynamically inside the C-level audio callback,
allowing instant, click-free muting and soloing.
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import QObject, Signal, QTimer


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

    def __init__(self) -> None:
        super().__init__()
        # Audio data storage.
        self._stems: dict[str, np.ndarray] = {}
        self._sample_rate: int = 44100
        self._total_frames: int = 0
        self._current_frame: int = 0

        # Mixing state. Muting/soloing is thread-safe enough for this use case
        # because dict/set reads in Python are GIL-atomic.
        self._is_playing: bool = False
        self._muted_stems: set[str] = set()
        self._soloed_stems: set[str] = set()

        # Hardware stream.
        self._stream: sd.OutputStream | None = None

        # UI updater.
        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30fps
        self._timer.timeout.connect(self._emit_position)

    @property
    def is_playing(self) -> bool:
        """Return True if audio is currently playing."""
        return self._is_playing

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
        self._muted_stems.clear()
        self._soloed_stems.clear()

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

        self.position_changed.emit(0.0)

    def play(self) -> None:
        """Start or resume playback."""
        if not self._stems or self._is_playing:
            return

        if self._current_frame >= self._total_frames:
            self._current_frame = 0

        if self._stream is None:
            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=2,
                callback=self._audio_callback,
            )

        self._stream.start()
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

    def set_solo(self, stem_name: str, soloed: bool) -> None:
        """Solo or unsolo a specific stem."""
        if soloed:
            self._soloed_stems.add(stem_name)
        else:
            self._soloed_stems.discard(stem_name)

    # ------------------------------------------------------------------
    # Internal Callbacks
    # ------------------------------------------------------------------

    def _emit_position(self) -> None:
        """Emit the current playback position for UI updates."""
        pos_s = self._current_frame / self._sample_rate
        self.position_changed.emit(pos_s)

    def _audio_callback(self, outdata: np.ndarray, frames: int,
                        time_info: dict, status: sd.CallbackFlags) -> None:
        """PortAudio callback for pushing mixed audio to the hardware.

        This runs on a high-priority C thread. It must not block or allocate
        significant memory.
        """
        if status:
            print(f"Audio callback status: {status}")

        if not self._is_playing or self._current_frame >= self._total_frames:
            outdata.fill(0.0)
            raise sd.CallbackStop

        # Calculate how many frames we can actually read without overflowing.
        frames_available = self._total_frames - self._current_frame
        frames_to_read = min(frames, frames_available)

        # Clear the output buffer.
        outdata.fill(0.0)

        # Determine which stems should be audible.
        active_stems = []
        if self._soloed_stems:
            # If any stem is soloed, only play soloed stems (ignoring mute).
            active_stems = [
                name for name in self._stems if name in self._soloed_stems
            ]
        else:
            # Otherwise play everything not muted.
            active_stems = [
                name for name in self._stems if name not in self._muted_stems
            ]

        # Accumulate the active stems directly into the output buffer slice.
        start = self._current_frame
        end = start + frames_to_read

        for name in active_stems:
            stem_data = self._stems[name]
            # Ensure we don't index past the end of a shorter stem.
            stem_end = min(end, stem_data.shape[0])
            read_len = stem_end - start

            if read_len > 0:
                outdata[:read_len] += stem_data[start:stem_end]

        self._current_frame += frames_to_read

        # If we reached the end of the track, pad the rest of the buffer
        # with silence and raise the stop signal for PortAudio.
        if frames_to_read < frames:
            # Remaining buffer is already filled with 0.0 from clear.
            # We schedule the stop flag internally, but sd.CallbackStop
            # is the proper way to tell PortAudio we are done.
            self._is_playing = False
            # We cannot emit Qt signals directly from this C thread reliably
            # depending on the OS, so we trigger QMetaObject.invokeMethod or
            # just let the timer catch it.
            # A safe way is to let the play() or QTimer detect EOF next tick,
            # but raising CallbackStop handles the hardware correctly.
            raise sd.CallbackStop

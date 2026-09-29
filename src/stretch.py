"""Real-time time-stretch and pitch-shift for the Speed and Pitch controls.

The player used to render every stem at the new speed or pitch before it
could play them: librosa's ``time_stretch``/``pitch_shift`` build the whole
song's spectrum at once (about 3.5 GB per channel of a ten-minute stem), so
a change took a minute or more, swapped, and made playback stutter.

``StreamingStretcher`` instead processes the mix as it plays, the way
practice apps and DAWs do: a phase vocoder, one frame at a time, followed
by a streaming resampler when the pitch moves. A speed or pitch change
takes effect within a frame or two, and nothing is rendered in advance.

The phase vocoder is the one librosa uses (same Hann window, frame size,
hop, linearly interpolated magnitudes, and accumulated phase advance), so
it sounds the same; a pitch shift is a stretch followed by a resample, so
speed and pitch share one pass. The phase is carried as unit phasors
(a complex multiply per frame, no trigonometry) and renormalised each
frame, so it stays exact however long the song plays.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import numpy as np
import soxr

N_FFT = 2048
HOP = 512
_PAD = N_FFT // 2
_WINDOW = np.hanning(N_FFT + 1)[:-1].astype(np.float32)  # periodic Hann
_WINDOW_SQ = _WINDOW ** 2
_TINY = np.float32(1e-10)
# Samples pulled from the source at a time when the analysis runs dry.
_PULL = 2048
# Output samples over which a pitch change crossfades from the old
# resampler to the new one (10 ms): a hard switch clicked (#217 review).
_PITCH_FADE = 441

# A source read: (audio of shape (count, channels), song frame of its first
# sample). It may return fewer samples than asked (a loop boundary); an
# empty read means the song has ended.
SourceRead = Callable[[int], tuple[np.ndarray, int]]


def pitch_rate(semitones: float) -> float:
    """Stretch factor that, followed by a resample, shifts *semitones*."""
    return float(2.0 ** (-float(semitones) / 12.0))


class StreamingStretcher:
    """Stretch and pitch-shift a pulled audio stream in real time.

    Call ``read(count, source)`` from the audio callback for each block of
    output; the stretcher pulls as much source audio as it needs through
    *source*. ``position`` is the source frame of the audio most recently
    returned, which is what the listener is hearing. ``reset`` starts over
    (a seek), and ``set_params`` changes speed and pitch between reads.
    """

    def __init__(self, sample_rate: int, channels: int = 2) -> None:
        self._sr = int(sample_rate)
        self._channels = int(channels)
        self._speed = 1.0
        self._semitones = 0.0
        self._resampler: soxr.ResampleStream | None = None
        self.reset()

    # -- public API --------------------------------------------------------

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def semitones(self) -> float:
        return self._semitones

    @property
    def position(self) -> float:
        """Source frame of the last sample returned by ``read``."""
        return self._position

    def reset(self, position: float = 0.0) -> None:
        """Forget all state and start again at source frame *position*."""
        ch = self._channels
        # Input, in padded coordinates: sample 0 is _PAD zeros before the
        # first source sample, as librosa's centred STFT pads.
        self._input = np.zeros((_PAD, ch), dtype=np.float32)
        self._input_start = 0  # padded index of self._input[0]
        # (padded index, source frame, count) for each pulled run, so an
        # analysis position maps back to where it came from in the song.
        self._runs: deque[tuple[int, int, int]] = deque()
        self._source_done = False
        self._tail_flushed = False
        self._t = 0.0  # analysis position, in frames
        self._frame = 0  # output frames synthesised since reset
        self._phase: np.ndarray | None = None  # unit phasors, (ch, bins)
        self._advance: np.ndarray | None = None
        self._spectra: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._ola = np.zeros((N_FFT, ch), dtype=np.float32)
        self._norm = np.zeros(N_FFT, dtype=np.float32)
        self._recent_t: deque[float] = deque(maxlen=3)
        self._drop = _PAD  # leading output samples that are padding
        self._fifo = np.zeros((0, ch), dtype=np.float32)
        self._fifo_start = 0  # output samples consumed from the fifo
        # (output sample index, song frame, jumped back) at each finished
        # hop; "jumped back" marks the first hop after a loop wrap.
        self._marks: deque[tuple[int, float, bool]] = deque()
        self._last_mark_frame: float | None = None
        self._position = float(position)
        self._last_mark_frame = None
        self._old_resampler: soxr.ResampleStream | None = None
        self._fading = False
        self._new_resampler()

    def set_params(self, speed: float, semitones: float) -> None:
        """Change speed and pitch; takes effect with the next frame."""
        speed = float(speed)
        semitones = float(semitones)
        if semitones != self._semitones:
            self._start_pitch_fade()
            self._semitones = semitones
            self._new_resampler()
        self._speed = speed

    def read(self, count: int, source: SourceRead) -> np.ndarray:
        """Return *count* output samples, shape (count, channels).

        Past the end of the source the output runs out into silence;
        ``finished`` then turns True.
        """
        self.fill(count, source)
        out = self._fifo[:count]
        if len(out) < count:
            out = np.concatenate([
                out,
                np.zeros((count - len(out), self._channels), np.float32),
            ])
        self._fifo = self._fifo[count:]
        self._fifo_start += count
        self._update_position()
        return out

    def fill(self, count: int, source: SourceRead) -> None:
        """Make at least *count* output samples ready (unless it ends)."""
        while len(self._fifo) < count and not self._exhausted:
            self._synthesise_frame(source)
        if self._exhausted and not self._tail_flushed:
            self._tail_flushed = True
            self._finish_pitch_fade()
            self._flush_resampler()

    def jump_offset(self) -> int | None:
        """Output samples before the next loop wrap, or None if none is due.

        The audio callback reads up to the wrap, so a count-in or the
        Loop Trainer's next step lands exactly where the listener hears
        the loop start again.
        """
        start = self._fifo_start
        for index, _, jumped in self._marks:
            if jumped and index >= start:
                return index - start
        return None

    def clear_jump(self) -> None:
        """Mark the wrap at the read position as handled."""
        start = self._fifo_start
        for i, (index, frame, jumped) in enumerate(self._marks):
            if jumped and index >= start:
                self._marks[i] = (index, frame, False)
                return

    @property
    def _exhausted(self) -> bool:
        """True once the analysis has passed the end of the source."""
        return (
            self._source_done
            and self._t * HOP >= self._input_end() - _PAD
        )

    @property
    def finished(self) -> bool:
        """True once the source has ended and all its audio is out."""
        return self._exhausted and self._tail_flushed and not len(self._fifo)

    # -- internals ---------------------------------------------------------

    def _rate(self) -> float:
        """Analysis frames advanced per output frame."""
        return self._speed * pitch_rate(self._semitones)

    def _input_end(self) -> int:
        return self._input_start + len(self._input)

    def _pull(self, source: SourceRead) -> None:
        audio, frame = source(_PULL)
        audio = np.asarray(audio, dtype=np.float32).reshape(
            -1, self._channels,
        )
        if len(audio):
            self._runs.append((self._input_end(), int(frame), len(audio)))
            self._input = np.concatenate([self._input, audio])
        else:
            self._source_done = True
            # Trailing padding, as the centred STFT pads the end.
            self._input = np.concatenate([
                self._input, np.zeros((N_FFT, self._channels), np.float32),
            ])

    def _spectrum(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        """(magnitude, unit phasor) of analysis frame *j*, (ch, bins)."""
        cached = self._spectra.get(j)
        if cached is not None:
            return cached
        start = j * HOP - self._input_start
        frame = self._input[start:start + N_FFT]
        if len(frame) < N_FFT:
            frame = np.concatenate([
                frame,
                np.zeros((N_FFT - len(frame), self._channels), np.float32),
            ])
        spectrum = np.fft.rfft(frame.T * _WINDOW, axis=1)
        magnitude = np.abs(spectrum)
        unit = spectrum / np.maximum(magnitude, _TINY)
        unit[magnitude <= _TINY] = 1.0
        self._spectra[j] = (magnitude, unit)
        return magnitude, unit

    def _synthesise_frame(self, source: SourceRead) -> None:
        t = self._t
        i0 = int(np.floor(t))
        i1 = i0 + 1
        need = i1 * HOP + N_FFT
        while self._input_end() < need and not self._source_done:
            self._pull(source)

        mag0, unit0 = self._spectrum(i0)
        mag1, unit1 = self._spectrum(i1)
        a = np.float32(t - i0)
        magnitude = mag0 + a * (mag1 - mag0)
        if self._phase is None:
            self._phase = unit0.copy()
        else:
            self._phase = self._phase * self._advance
            self._phase /= np.maximum(np.abs(self._phase), _TINY)
        self._advance = unit1 * np.conj(unit0)

        frame = np.fft.irfft(magnitude * self._phase, n=N_FFT, axis=1)
        self._ola += (frame * _WINDOW).T
        self._norm += _WINDOW_SQ
        self._recent_t.append(t)

        # The first hop can take no more frames: normalise and emit it.
        done = self._ola[:HOP] / np.maximum(self._norm[:HOP, None], _TINY)
        self._ola = np.roll(self._ola, -HOP, axis=0)
        self._ola[-HOP:] = 0.0
        self._norm = np.roll(self._norm, -HOP)
        self._norm[-HOP:] = 0.0
        self._emit(done, self._recent_t[0])

        self._frame += 1
        self._t = t + self._rate()
        # Drop spectra and input the analysis has moved past.
        floor_now = int(np.floor(self._t))
        for key in [k for k in self._spectra if k < floor_now]:
            del self._spectra[key]
        keep_from = floor_now * HOP
        if keep_from > self._input_start + _PULL:
            cut = keep_from - self._input_start
            self._input = self._input[cut:]
            self._input_start = keep_from
            while self._runs and (
                self._runs[0][0] + self._runs[0][2] <= self._input_start
            ):
                self._runs.popleft()

    def _emit(self, samples: np.ndarray, t: float) -> None:
        """Queue one hop of stretched audio (resampled for pitch)."""
        if self._drop:
            skip = min(self._drop, len(samples))
            samples = samples[skip:]
            self._drop -= skip
            if len(samples) == 0:
                return
        samples = self._resample(samples)
        if len(samples) == 0:
            return
        frame = self._source_frame(t)
        jumped = (
            self._last_mark_frame is not None
            and frame < self._last_mark_frame - HOP
        )
        self._last_mark_frame = frame
        self._marks.append((
            self._fifo_start + len(self._fifo), frame, jumped,
        ))
        self._fifo = np.concatenate([self._fifo, samples])

    def _source_frame(self, t: float) -> float:
        """Song frame at analysis position *t* (frames)."""
        padded = t * HOP + _PAD  # centre of analysis frame t
        for start, frame, count in reversed(self._runs):
            if padded >= start:
                return frame + min(padded - start, count)
        return self._position

    def _update_position(self) -> None:
        consumed = self._fifo_start
        while len(self._marks) > 1 and self._marks[1][0] <= consumed:
            self._marks.popleft()
        if self._marks and self._marks[0][0] <= consumed:
            index, frame, _ = self._marks[0]
            self._position = frame + (consumed - index) * self._speed

    def _new_resampler(self) -> None:
        if self._semitones == 0:
            self._resampler = None
            return
        self._resampler = soxr.ResampleStream(
            self._sr / pitch_rate(self._semitones), self._sr,
            self._channels, dtype="float32", quality="HQ",
        )

    def _resample(self, samples: np.ndarray) -> np.ndarray:
        """Pass one hop through the resampler, crossfading after a change."""
        new = (
            self._resampler.resample_chunk(samples)
            if self._resampler is not None else samples
        )
        if not self._fading:
            return new
        old = (
            self._old_resampler.resample_chunk(samples)
            if self._old_resampler is not None else samples
        )
        # First, what the old resampler still held from before the change
        # plays on; then both carry the same audio from the switch point,
        # and the crossfade blends one pitch into the other.
        held = min(self._old_held, len(old))
        out = [old[:held]]
        self._old_held -= held
        self._old_tail = np.concatenate([self._old_tail, old[held:]])
        self._new_head = np.concatenate([self._new_head, new])
        if self._old_held == 0:
            n = min(len(self._old_tail), len(self._new_head), self._fade_left)
            if n:
                done = _PITCH_FADE - self._fade_left
                ramp = (np.arange(done, done + n, dtype=np.float32)
                        / _PITCH_FADE)[:, np.newaxis]
                out.append(self._old_tail[:n] * (1.0 - ramp)
                           + self._new_head[:n] * ramp)
                self._old_tail = self._old_tail[n:]
                self._new_head = self._new_head[n:]
                self._fade_left -= n
            if self._fade_left == 0:
                out.append(self._new_head)
                self._end_pitch_fade()
        return np.concatenate(out)

    def _start_pitch_fade(self) -> None:
        """Keep the current resampler for a crossfade into the next one."""
        if self._fading:
            self._finish_pitch_fade()
        empty = np.zeros((0, self._channels), np.float32)
        self._old_resampler = self._resampler
        self._old_held = (
            int(round(self._resampler.delay()))
            if self._resampler is not None else 0
        )
        self._old_tail = empty
        self._new_head = empty
        self._fade_left = _PITCH_FADE
        self._fading = True

    def _finish_pitch_fade(self) -> None:
        """Cut an unfinished crossfade short (another change, or the end)."""
        if not self._fading:
            return
        if len(self._new_head):
            self._fifo = np.concatenate([self._fifo, self._new_head])
        self._end_pitch_fade()

    def _end_pitch_fade(self) -> None:
        empty = np.zeros((0, self._channels), np.float32)
        self._old_resampler = None
        self._old_tail = empty
        self._new_head = empty
        self._fading = False

    def _flush_resampler(self) -> None:
        """Emit what the resampler still holds (the end of the song)."""
        if self._resampler is None:
            return
        tail = self._resampler.resample_chunk(
            np.zeros((0, self._channels), np.float32), last=True,
        )
        if len(tail):
            self._fifo = np.concatenate([self._fifo, tail])

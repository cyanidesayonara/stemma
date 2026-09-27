"""Export individual stems or custom mixes as WAV or MP3 files.

Reads stem WAV files from disk and writes either a single stem copy
or a mixed-down combination of selected stems to a new file.
Output format is determined by the file extension (.wav or .mp3).
"""

import os

import numpy as np
import soundfile as sf
from PySide6.QtCore import QThread, Signal

from src.click_utils import generate_count_in
from src.separator import SAMPLE_RATE

# Supported output formats by extension.
SUPPORTED_FORMATS = (".wav", ".mp3")


def _write_audio(
    path: str,
    audio: np.ndarray,
    sample_rate: int,
    mp3_bitrate: int = 320,
) -> None:
    """Write audio data to a file, choosing format by extension.

    Args:
        path: Output file path (.wav or .mp3).
        audio: Audio array of shape (frames, channels), float32 in [-1, 1].
        sample_rate: Sample rate in Hz.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".wav":
        sf.write(path, audio, sample_rate)
    elif ext == ".mp3":
        _write_mp3(path, audio, sample_rate, bitrate=mp3_bitrate)
    else:
        raise ValueError(f"Unsupported format '{ext}'. Use .wav or .mp3.")


def _write_mp3(
    path: str,
    audio: np.ndarray,
    sample_rate: int,
    bitrate: int = 320,
) -> None:
    """Encode float32 audio to MP3 using lameenc.

    Args:
        path: Output file path.
        audio: Float32 array of shape (frames, channels) in [-1, 1].
        sample_rate: Sample rate in Hz.
        bitrate: MP3 bitrate in kbps (default 320).
    """
    import lameenc

    # Ensure stereo.
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])

    n_channels = audio.shape[1]

    # Convert float32 [-1, 1] to int16 PCM.
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(n_channels)
    encoder.set_quality(2)  # 2 = high quality

    mp3_data = encoder.encode(pcm.tobytes())
    mp3_data += encoder.flush()

    with open(path, "wb") as f:
        f.write(mp3_data)


def audible_stems(
    stem_names: list[str],
    muted_stems: set[str] | None = None,
    soloed_stems: set[str] | None = None,
    volumes: dict[str, float] | None = None,
    master_volume: float = 1.0,
) -> list[str]:
    """Return the stems the user actually hears, in *stem_names* order.

    Mirrors the player's audio callback: when any stem is soloed, only
    soloed stems play (mute is ignored for them); otherwise every
    unmuted stem plays. A stem whose effective gain (stem volume times
    master volume) is zero is silent and left out.

    Args:
        stem_names: Candidate stem names.
        muted_stems: Muted stem names.
        soloed_stems: Soloed stem names.
        volumes: Per-stem gain levels. Missing stems default to 1.0.
        master_volume: Master gain applied to every stem.
    """
    muted = muted_stems or set()
    soloed = soloed_stems or set()
    volumes = volumes or {}
    if soloed:
        active = [s for s in stem_names if s in soloed]
    else:
        active = [s for s in stem_names if s not in muted]
    return [
        s for s in active if volumes.get(s, 1.0) * master_volume > 0.0
    ]


def _shift_frames(audio: np.ndarray, offset_frames: int) -> np.ndarray:
    """Shift *audio* in time, keeping its length, padding with silence.

    Positive offsets move the audio later; negative ones earlier. Matches
    ``AudioPlayer.nudge_stem``.
    """
    if offset_frames == 0:
        return audio
    shifted = np.zeros_like(audio)
    n = audio.shape[0]
    if abs(offset_frames) >= n:
        return shifted
    if offset_frames > 0:
        shifted[offset_frames:] = audio[:n - offset_frames]
    else:
        shifted[:offset_frames] = audio[-offset_frames:]
    return shifted


class StemExporter:
    """Exports stems or custom mixes from separated audio.

    Args:
        stem_paths: Mapping of stem name to WAV file path on disk.
    """

    def __init__(self, stem_paths: dict[str, str]) -> None:
        self.stem_paths = dict(stem_paths)

    @property
    def available_stems(self) -> list[str]:
        """Return the list of available stem names."""
        return list(self.stem_paths.keys())

    def export_stem(self, stem_name: str, output_path: str) -> None:
        """Export a single stem to a WAV or MP3 file.

        Args:
            stem_name: Name of the stem to export.
            output_path: Destination file path (.wav or .mp3).

        Raises:
            KeyError: If *stem_name* is not available.
        """
        if stem_name not in self.stem_paths:
            raise KeyError(f"Stem '{stem_name}' not found")

        audio, sr = sf.read(self.stem_paths[stem_name], dtype="float32")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        _write_audio(output_path, audio, sr)

    def export_mix(
        self,
        output_path: str,
        stem_names: list[str] | None = None,
        muted_stems: set[str] | None = None,
        volumes: dict[str, float] | None = None,
        mp3_bitrate: int = 320,
        start_frame: int | None = None,
        end_frame: int | None = None,
        count_in_beats: int = 0,
        count_in_bpm: float = 120.0,
        count_in_volume: float = 0.5,
        soloed_stems: set[str] | None = None,
        master_volume: float = 1.0,
        nudge_offsets: dict[str, float] | None = None,
    ) -> None:
        """Export a mix of selected stems to a WAV or MP3 file.

        The mix follows playback: mute/solo decide which stems are heard
        (see :func:`audible_stems`), each stem is scaled by its volume
        and the master volume, nudged stems are shifted in time, and the
        mix runs to the end of the longest stem.

        Args:
            output_path: Destination file path (.wav or .mp3).
            stem_names: Stems to include. Defaults to all available stems.
            muted_stems: Muted stems. Ignored while any stem is soloed.
            volumes: Per-stem gain levels (0.0--2.0). Missing stems
                default to 1.0.
            mp3_bitrate: Bitrate for MP3 output (default 320 kbps).
            start_frame: First frame to include (for loop-region export).
                When ``None`` (default), starts from the beginning.
            end_frame: Frame after the last included frame. When ``None``
                (default), goes to the end of the audio.
            count_in_beats: Number of metronome beats to prepend (0 = none).
            count_in_bpm: Tempo for the prepended count-in (default 120).
            count_in_volume: Volume for count-in clicks (0.0--2.0).
            soloed_stems: Soloed stems. When non-empty, only these play.
            master_volume: Master gain applied to every stem.
            nudge_offsets: Per-stem time offsets in milliseconds
                (positive = later), as set by ``AudioPlayer.nudge_stem``.

        Raises:
            ValueError: If no stem would be audible.
        """
        if stem_names is None:
            stem_names = self.available_stems

        if volumes is None:
            volumes = {}
        if nudge_offsets is None:
            nudge_offsets = {}

        stem_names = audible_stems(
            stem_names, muted_stems, soloed_stems, volumes, master_volume
        )
        if not stem_names:
            raise ValueError("No stems selected for export")

        tracks = []
        for name in stem_names:
            audio, sr = sf.read(self.stem_paths[name], dtype="float32")
            offset_ms = nudge_offsets.get(name, 0.0)
            if offset_ms:
                audio = _shift_frames(audio, int(offset_ms / 1000.0 * sr))
            if start_frame is not None or end_frame is not None:
                sf_start = start_frame if start_frame is not None else 0
                sf_end = end_frame if end_frame is not None else audio.shape[0]
                audio = audio[sf_start:sf_end]
            tracks.append(audio * (volumes.get(name, 1.0) * master_volume))

        # Playback runs to the end of the longest stem, so shorter stems
        # simply fall silent rather than truncating the mix.
        # Mono tracks are broadcast across the channels of a stereo mix.
        length = max(t.shape[0] for t in tracks)
        channels = max((t.shape[1] for t in tracks if t.ndim == 2), default=0)
        shape = (length, channels) if channels else (length,)
        mixed = np.zeros(shape, dtype=np.float32)
        for audio in tracks:
            if audio.ndim == 1 and channels:
                audio = audio[:, np.newaxis]
            mixed[:audio.shape[0]] += audio

        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed /= peak
        else:
            np.clip(mixed, -1.0, 1.0, out=mixed)

        if count_in_beats > 0:
            ci_audio = generate_count_in(
                beats=count_in_beats,
                bpm=count_in_bpm,
                sample_rate=SAMPLE_RATE,
                volume=count_in_volume,
            )
            if mixed.ndim == 1:
                ci_audio = ci_audio[:, 0]
            mixed = np.concatenate([ci_audio, mixed], axis=0)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        _write_audio(output_path, mixed, SAMPLE_RATE, mp3_bitrate=mp3_bitrate)


class ExportWorker(QThread):
    """Background thread for exporting a custom mix without freezing the UI.

    Signals:
        finished(str): Emitted with the final output path on success.
        error(str): Emitted with error traceback on failure.
    """

    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        exporter: StemExporter,
        output_path: str,
        muted_stems: set[str],
        volumes: dict[str, float],
        mp3_bitrate: int = 320,
        start_frame: int | None = None,
        end_frame: int | None = None,
        count_in_beats: int = 0,
        count_in_bpm: float = 120.0,
        count_in_volume: float = 0.5,
        soloed_stems: set[str] | None = None,
        master_volume: float = 1.0,
        nudge_offsets: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.exporter = exporter
        self.output_path = output_path
        self.muted_stems = muted_stems
        self.soloed_stems = soloed_stems or set()
        self.master_volume = master_volume
        self.nudge_offsets = nudge_offsets or {}
        self.volumes = volumes
        self.mp3_bitrate = mp3_bitrate
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.count_in_beats = count_in_beats
        self.count_in_bpm = count_in_bpm
        self.count_in_volume = count_in_volume

    def run(self) -> None:
        try:
            self.exporter.export_mix(
                self.output_path,
                muted_stems=self.muted_stems,
                soloed_stems=self.soloed_stems,
                volumes=self.volumes,
                master_volume=self.master_volume,
                nudge_offsets=self.nudge_offsets,
                mp3_bitrate=self.mp3_bitrate,
                start_frame=self.start_frame,
                end_frame=self.end_frame,
                count_in_beats=self.count_in_beats,
                count_in_bpm=self.count_in_bpm,
                count_in_volume=self.count_in_volume,
            )
            self.finished.emit(self.output_path)
        except Exception as exc:
            self.error.emit(str(exc))

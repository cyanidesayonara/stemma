"""Export individual stems or custom mixes as WAV or MP3 files.

Reads stem WAV files from disk and writes either a single stem copy
or a mixed-down combination of selected stems to a new file.
Output format is determined by the file extension (.wav or .mp3).
"""

import os

import numpy as np
import soundfile as sf
from PySide6.QtCore import QThread, Signal

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
    ) -> None:
        """Export a mix of selected stems to a WAV file.

        Args:
            output_path: Destination file path.
            stem_names: Stems to include. Defaults to all available stems.
            muted_stems: Stems to exclude from the mix. Applied after
                *stem_names* filtering.
            volumes: Per-stem gain levels (0.0–2.0). Missing stems
                default to 1.0.

        Raises:
            ValueError: If the resulting stem list is empty.
        """
        if stem_names is None:
            stem_names = self.available_stems

        if muted_stems:
            stem_names = [s for s in stem_names if s not in muted_stems]

        if not stem_names:
            raise ValueError("No stems selected for export")

        if volumes is None:
            volumes = {}

        # Load and sum the selected stems with volume applied.
        mixed = None
        for name in stem_names:
            audio, sr = sf.read(self.stem_paths[name], dtype="float32")
            gain = volumes.get(name, 1.0)
            audio = audio * gain
            if mixed is None:
                mixed = audio.copy()
            else:
                min_len = min(mixed.shape[0], audio.shape[0])
                mixed[:min_len] += audio[:min_len]

        # Prevent hard digital clipping distortion by auto-normalizing 
        # the master bus down if it exceeds 0dBFS (1.0).
        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed /= peak
        else:
            np.clip(mixed, -1.0, 1.0, out=mixed)

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
    ) -> None:
        super().__init__()
        self.exporter = exporter
        self.output_path = output_path
        self.muted_stems = muted_stems
        self.volumes = volumes
        self.mp3_bitrate = mp3_bitrate

    def run(self) -> None:
        try:
            self.exporter.export_mix(
                self.output_path,
                muted_stems=self.muted_stems,
                volumes=self.volumes,
                mp3_bitrate=self.mp3_bitrate,
            )
            self.finished.emit(self.output_path)
        except Exception as exc:
            self.error.emit(str(exc))

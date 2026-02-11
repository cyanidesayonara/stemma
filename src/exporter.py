"""Export individual stems or custom mixes as WAV files.

Reads stem WAV files from disk and writes either a single stem copy
or a mixed-down combination of selected stems to a new file.
"""

import os

import numpy as np
import soundfile as sf

from src.separator import SAMPLE_RATE


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
        """Export a single stem to a WAV file.

        Args:
            stem_name: Name of the stem to export.
            output_path: Destination file path.

        Raises:
            KeyError: If *stem_name* is not available.
        """
        if stem_name not in self.stem_paths:
            raise KeyError(f"Stem '{stem_name}' not found")

        audio, sr = sf.read(self.stem_paths[stem_name], dtype="float32")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, audio, sr)

    def export_mix(
        self,
        output_path: str,
        stem_names: list[str] | None = None,
        muted_stems: set[str] | None = None,
        volumes: dict[str, float] | None = None,
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

        np.clip(mixed, -1.0, 1.0, out=mixed)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, mixed, SAMPLE_RATE)

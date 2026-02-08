"""Tests for the stem exporter."""

import os

import numpy as np
import pytest
import soundfile as sf

from src.exporter import StemExporter
from src.separator import SAMPLE_RATE


@pytest.fixture
def stem_dir(tmp_path):
    """Create a directory with fake stem WAV files."""
    stems = tmp_path / "stems"
    stems.mkdir()

    sr = SAMPLE_RATE
    duration_frames = sr * 2  # 2 seconds

    for name, freq in (("vocals", 220), ("drums", 440), ("bass", 110), ("other", 660)):
        t = np.linspace(0, 2.0, duration_frames, dtype=np.float32)
        signal = 0.3 * np.sin(2 * np.pi * freq * t)
        stereo = np.column_stack([signal, signal])
        sf.write(str(stems / f"{name}.wav"), stereo, sr)

    return str(stems)


@pytest.fixture
def exporter(stem_dir):
    """Create a StemExporter pointed at the fake stems."""
    stem_paths = {}
    for name in ("vocals", "drums", "bass", "other"):
        stem_paths[name] = os.path.join(stem_dir, f"{name}.wav")
    return StemExporter(stem_paths)


class TestStemExporterInit:

    def test_init_stores_stem_paths(self, exporter):
        assert len(exporter.stem_paths) == 4
        assert "vocals" in exporter.stem_paths

    def test_available_stems(self, exporter):
        assert set(exporter.available_stems) == {"vocals", "drums", "bass", "other"}


class TestExportSingleStem:

    def test_export_single_stem_wav(self, exporter, tmp_path):
        out = str(tmp_path / "vocals_export.wav")
        exporter.export_stem("vocals", out)

        assert os.path.isfile(out)
        audio, sr = sf.read(out)
        assert sr == SAMPLE_RATE
        assert audio.ndim == 2
        assert audio.shape[1] == 2
        assert audio.shape[0] == SAMPLE_RATE * 2

    def test_export_stem_content_matches_source(self, exporter, tmp_path, stem_dir):
        out = str(tmp_path / "bass_export.wav")
        exporter.export_stem("bass", out)

        original, _ = sf.read(os.path.join(stem_dir, "bass.wav"), dtype="float32")
        exported, _ = sf.read(out, dtype="float32")
        assert np.allclose(original, exported, atol=1e-5)

    def test_export_nonexistent_stem_raises(self, exporter, tmp_path):
        out = str(tmp_path / "nope.wav")
        with pytest.raises(KeyError):
            exporter.export_stem("guitar", out)

    def test_export_creates_parent_directory(self, exporter, tmp_path):
        out = str(tmp_path / "nested" / "deep" / "vocals.wav")
        exporter.export_stem("vocals", out)
        assert os.path.isfile(out)


class TestExportMix:

    def test_export_mix_all_stems(self, exporter, tmp_path):
        out = str(tmp_path / "full_mix.wav")
        exporter.export_mix(out)

        assert os.path.isfile(out)
        audio, sr = sf.read(out)
        assert sr == SAMPLE_RATE
        assert audio.ndim == 2
        assert audio.shape[0] == SAMPLE_RATE * 2

    def test_export_mix_subset(self, exporter, tmp_path):
        out = str(tmp_path / "no_vocals.wav")
        exporter.export_mix(out, stem_names=["drums", "bass", "other"])

        audio, sr = sf.read(out, dtype="float32")
        assert sr == SAMPLE_RATE

        # Should not contain the vocals frequency component.
        # Just verify it has audio content.
        assert np.max(np.abs(audio)) > 0.1

    def test_export_mix_with_muted(self, exporter, tmp_path):
        out = str(tmp_path / "muted_mix.wav")
        exporter.export_mix(out, muted_stems={"vocals"})

        full_out = str(tmp_path / "full.wav")
        exporter.export_mix(full_out)

        muted, _ = sf.read(out, dtype="float32")
        full, _ = sf.read(full_out, dtype="float32")

        # Muted mix should have lower energy.
        assert np.sum(muted ** 2) < np.sum(full ** 2)

    def test_export_mix_empty_raises(self, exporter, tmp_path):
        out = str(tmp_path / "empty.wav")
        with pytest.raises(ValueError):
            exporter.export_mix(out, stem_names=[])

    def test_export_mix_clips_output(self, exporter, tmp_path):
        out = str(tmp_path / "mix.wav")
        exporter.export_mix(out)

        audio, _ = sf.read(out, dtype="float32")
        assert np.max(audio) <= 1.0
        assert np.min(audio) >= -1.0

    def test_export_mix_creates_parent_directory(self, exporter, tmp_path):
        out = str(tmp_path / "nested" / "mix.wav")
        exporter.export_mix(out)
        assert os.path.isfile(out)

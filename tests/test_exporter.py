"""Tests for the stem exporter."""

import os

import numpy as np
import pytest
import soundfile as sf

from src.exporter import StemExporter, _write_audio
from src.separator import SAMPLE_RATE
from src.click_utils import generate_click, generate_count_in


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

    def test_export_mix_with_volumes(self, exporter, tmp_path):
        full_out = str(tmp_path / "full.wav")
        exporter.export_mix(full_out)

        half_out = str(tmp_path / "half.wav")
        volumes = {"vocals": 0.5, "drums": 0.5, "bass": 0.5, "other": 0.5}
        exporter.export_mix(half_out, volumes=volumes)

        full, _ = sf.read(full_out, dtype="float32")
        half, _ = sf.read(half_out, dtype="float32")

        # Half-volume mix should have ~25% the energy of full mix
        full_energy = np.sum(full ** 2)
        half_energy = np.sum(half ** 2)
        assert half_energy < full_energy * 0.5


class TestMP3Export:

    def test_export_stem_as_mp3(self, exporter, tmp_path):
        out = str(tmp_path / "vocals.mp3")
        exporter.export_stem("vocals", out)
        assert os.path.isfile(out)
        # MP3 files start with ID3 tag or MPEG sync word (0xff 0xfb/f3/f2).
        with open(out, "rb") as f:
            header = f.read(2)
        assert header == b"ID3" [:2] or header[0:1] == b"\xff"

    def test_export_mix_as_mp3(self, exporter, tmp_path):
        out = str(tmp_path / "mix.mp3")
        exporter.export_mix(out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 1000  # Not trivially empty.

    def test_mp3_file_is_smaller_than_wav(self, exporter, tmp_path):
        wav_out = str(tmp_path / "mix.wav")
        mp3_out = str(tmp_path / "mix.mp3")
        exporter.export_mix(wav_out)
        exporter.export_mix(mp3_out)
        assert os.path.getsize(mp3_out) < os.path.getsize(wav_out)

    def test_write_audio_unsupported_format(self, tmp_path):
        audio = np.zeros((1000, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="Unsupported format"):
            _write_audio(str(tmp_path / "out.ogg"), audio, SAMPLE_RATE)

    def test_export_stem_mp3_creates_parent_directory(self, exporter, tmp_path):
        out = str(tmp_path / "nested" / "deep" / "vocals.mp3")
        exporter.export_stem("vocals", out)
        assert os.path.isfile(out)


class TestExportLoopRegion:
    """Tests for exporting a sub-region (A-B loop) of the mix."""

    def test_export_region_output_length(self, exporter, tmp_path):
        start = SAMPLE_RATE // 2  # 0.5s
        end = SAMPLE_RATE + SAMPLE_RATE // 2  # 1.5s
        out = str(tmp_path / "region.wav")
        exporter.export_mix(out, start_frame=start, end_frame=end)

        audio, sr = sf.read(out, dtype="float32")
        assert audio.shape[0] == end - start

    def test_export_region_start_only(self, exporter, tmp_path):
        start = SAMPLE_RATE  # 1.0s
        out = str(tmp_path / "from_1s.wav")
        exporter.export_mix(out, start_frame=start)

        audio, sr = sf.read(out, dtype="float32")
        expected_len = SAMPLE_RATE * 2 - start  # 2s total - 1s offset
        assert audio.shape[0] == expected_len

    def test_export_region_end_only(self, exporter, tmp_path):
        end = SAMPLE_RATE  # first 1.0s
        out = str(tmp_path / "first_1s.wav")
        exporter.export_mix(out, end_frame=end)

        audio, sr = sf.read(out, dtype="float32")
        assert audio.shape[0] == end

    def test_export_region_no_frames_gives_full(self, exporter, tmp_path):
        out = str(tmp_path / "full.wav")
        exporter.export_mix(out)

        audio, sr = sf.read(out, dtype="float32")
        assert audio.shape[0] == SAMPLE_RATE * 2

    def test_export_region_content_correct(self, exporter, tmp_path, stem_dir):
        full_out = str(tmp_path / "full.wav")
        exporter.export_mix(full_out)
        full, _ = sf.read(full_out, dtype="float32")

        start = SAMPLE_RATE // 4
        end = SAMPLE_RATE
        region_out = str(tmp_path / "region.wav")
        exporter.export_mix(region_out, start_frame=start, end_frame=end)
        region, _ = sf.read(region_out, dtype="float32")

        assert np.allclose(full[start:end], region, atol=1e-5)

    def test_export_region_with_muted(self, exporter, tmp_path):
        start = 0
        end = SAMPLE_RATE
        out = str(tmp_path / "muted_region.wav")
        exporter.export_mix(
            out, muted_stems={"vocals"}, start_frame=start, end_frame=end
        )

        audio, sr = sf.read(out, dtype="float32")
        assert audio.shape[0] == end - start
        assert np.max(np.abs(audio)) > 0.01

    def test_export_region_as_mp3(self, exporter, tmp_path):
        start = 0
        end = SAMPLE_RATE
        out = str(tmp_path / "region.mp3")
        exporter.export_mix(out, start_frame=start, end_frame=end)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 100


class TestExportCountIn:
    """Tests for exporting with a count-in prepended."""

    def test_count_in_prepends_audio(self, exporter, tmp_path):
        out = str(tmp_path / "with_ci.wav")
        exporter.export_mix(
            out, count_in_beats=4, count_in_bpm=120.0, count_in_volume=0.5
        )

        audio, sr = sf.read(out, dtype="float32")
        beat_interval = int(60.0 / 120.0 * SAMPLE_RATE)
        ci_frames = 4 * beat_interval
        expected_total = SAMPLE_RATE * 2 + ci_frames
        assert audio.shape[0] == expected_total

    def test_count_in_zero_beats_no_change(self, exporter, tmp_path):
        out_no_ci = str(tmp_path / "no_ci.wav")
        exporter.export_mix(out_no_ci, count_in_beats=0)

        out_plain = str(tmp_path / "plain.wav")
        exporter.export_mix(out_plain)

        a, _ = sf.read(out_no_ci, dtype="float32")
        b, _ = sf.read(out_plain, dtype="float32")
        assert a.shape[0] == b.shape[0]
        assert np.allclose(a, b, atol=1e-6)

    def test_count_in_has_clicks(self, exporter, tmp_path):
        out = str(tmp_path / "ci.wav")
        exporter.export_mix(
            out, count_in_beats=2, count_in_bpm=120.0, count_in_volume=1.0
        )

        audio, sr = sf.read(out, dtype="float32")
        beat_interval = int(60.0 / 120.0 * SAMPLE_RATE)
        ci_region = audio[:2 * beat_interval]
        assert np.max(np.abs(ci_region)) > 0.1

    def test_count_in_combined_with_region(self, exporter, tmp_path):
        start = 0
        end = SAMPLE_RATE
        out = str(tmp_path / "ci_region.wav")
        exporter.export_mix(
            out,
            start_frame=start,
            end_frame=end,
            count_in_beats=4,
            count_in_bpm=120.0,
        )

        audio, sr = sf.read(out, dtype="float32")
        beat_interval = int(60.0 / 120.0 * SAMPLE_RATE)
        ci_frames = 4 * beat_interval
        region_frames = end - start
        assert audio.shape[0] == ci_frames + region_frames

    def test_count_in_as_mp3(self, exporter, tmp_path):
        out = str(tmp_path / "ci.mp3")
        exporter.export_mix(
            out, count_in_beats=4, count_in_bpm=120.0, count_in_volume=0.5
        )
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 100


class TestClickUtils:
    """Tests for the shared click generation utilities."""

    def test_generate_click_shape(self):
        click = generate_click(SAMPLE_RATE)
        assert click.ndim == 2
        assert click.shape[1] == 2
        expected_len = int(SAMPLE_RATE * 0.03)
        assert click.shape[0] == expected_len

    def test_generate_click_has_content(self):
        click = generate_click(SAMPLE_RATE)
        assert np.max(np.abs(click)) > 0.1

    def test_generate_count_in_length(self):
        ci = generate_count_in(4, 120.0, SAMPLE_RATE)
        beat_interval = int(60.0 / 120.0 * SAMPLE_RATE)
        assert ci.shape[0] == 4 * beat_interval

    def test_generate_count_in_has_clicks_at_beat_positions(self):
        ci = generate_count_in(4, 120.0, SAMPLE_RATE, volume=1.0)
        beat_interval = int(60.0 / 120.0 * SAMPLE_RATE)
        for beat in range(4):
            start = beat * beat_interval
            end = start + int(SAMPLE_RATE * 0.03)
            assert np.max(np.abs(ci[start:end])) > 0.1

    def test_generate_count_in_stereo(self):
        ci = generate_count_in(2, 120.0, SAMPLE_RATE)
        assert ci.ndim == 2
        assert ci.shape[1] == 2

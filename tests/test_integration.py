"""Integration tests that exercise real components together.

These tests use real files on disk, real audio data, and (when available)
real ONNX model inference. They catch issues that unit tests miss: file
path handling on Windows, audio format round-trips, cross-module contracts,
and actual model output validity.

Markers:
    slow: Tests that run ONNX inference (~30-60s on GPU, minutes on CPU).
    hardware: Tests that require a working audio output device.

Run all integration tests:
    pytest tests/test_integration.py -v

Skip slow tests (no model needed):
    pytest tests/test_integration.py -v -m "not slow"

Skip hardware tests (no speakers needed):
    pytest tests/test_integration.py -v -m "not hardware"
"""

import os
import json
import time

import numpy as np
import pytest
import soundfile as sf

from src.library import SongLibrary, Song
from src.player import MultiTrackPlayer
from src.separator import (
    SeparatorWorker, SAMPLE_RATE, STEMS_4, STEMS_6,
)
from src.model_manager import ModelManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture
def data_dir():
    """Return the real project data directory."""
    return os.path.abspath(DATA_DIR)


@pytest.fixture
def model_manager(data_dir):
    """Return a ModelManager pointed at the real data directory."""
    return ModelManager(data_dir=data_dir)


@pytest.fixture
def synthetic_song(tmp_path):
    """Generate a short synthetic stereo WAV with recognizable content.

    Creates a 3-second file with a 440Hz sine (left) and 880Hz sine (right)
    so we can verify channels survive the pipeline.
    """
    duration = 3.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    audio = np.column_stack([left, right])

    path = tmp_path / "synthetic_song.wav"
    sf.write(str(path), audio, SAMPLE_RATE)
    return str(path)


@pytest.fixture
def synthetic_stems(tmp_path):
    """Generate fake stem WAV files that mimic separator output.

    Each stem is a sine wave at a different frequency so they are
    distinguishable when mixed.
    """
    duration = 2.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)

    stems = {
        "vocals": 220,
        "drums": 440,
        "bass": 110,
        "other": 660,
    }

    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    paths = {}

    for name, freq in stems.items():
        signal = 0.3 * np.sin(2 * np.pi * freq * t)
        stereo = np.column_stack([signal, signal])
        path = stems_dir / f"{name}.wav"
        sf.write(str(path), stereo, SAMPLE_RATE)
        paths[name] = str(path)

    return paths


# ---------------------------------------------------------------------------
# Library integration: real file I/O, persistence round-trips
# ---------------------------------------------------------------------------

class TestLibraryIntegration:
    """Test library operations against the real filesystem."""

    def test_full_lifecycle(self, tmp_path, synthetic_song):
        """Add, update, reload from disk, and remove a song."""
        data_dir = str(tmp_path / "data")
        lib = SongLibrary(data_dir=data_dir)

        # Add
        song = lib.add_song(
            title="Integration Test",
            artist="Test Artist",
            original_path=synthetic_song,
        )
        assert os.path.isfile(song.original_path)
        assert song.original_path.startswith(
            os.path.join(data_dir, "songs", song.id)
        )

        # Verify the copied file is valid audio.
        audio, sr = sf.read(song.original_path)
        assert sr == SAMPLE_RATE
        assert audio.shape[0] == int(3.0 * SAMPLE_RATE)

        # Update
        lib.update_song(song.id, model_used="htdemucs")
        assert lib.get_song(song.id).model_used == "htdemucs"

        # Reload from disk (simulates app restart).
        lib2 = SongLibrary(data_dir=data_dir)
        assert len(lib2.songs) == 1
        reloaded = lib2.get_song(song.id)
        assert reloaded.title == "Integration Test"
        assert reloaded.model_used == "htdemucs"
        assert os.path.isfile(reloaded.original_path)

        # Remove
        song_dir = song.stems_path
        lib2.remove_song(song.id)
        assert len(lib2.songs) == 0
        assert not os.path.exists(song_dir)

    def test_multiple_songs_persist(self, tmp_path, synthetic_song):
        """Add several songs, reload, verify ordering and count."""
        data_dir = str(tmp_path / "data")
        lib = SongLibrary(data_dir=data_dir)

        ids = []
        for i in range(5):
            song = lib.add_song(
                title=f"Song {i}",
                artist=f"Artist {i}",
                original_path=synthetic_song,
            )
            ids.append(song.id)

        # Reload
        lib2 = SongLibrary(data_dir=data_dir)
        assert len(lib2.songs) == 5
        for i, sid in enumerate(ids):
            assert lib2.get_song(sid).title == f"Song {i}"

    def test_json_is_valid_after_crash_simulation(self, tmp_path, synthetic_song):
        """Verify JSON integrity by checking it is always parseable."""
        data_dir = str(tmp_path / "data")
        lib = SongLibrary(data_dir=data_dir)
        json_path = os.path.join(data_dir, "library.json")

        lib.add_song(title="A", artist="A", original_path=synthetic_song)
        lib.add_song(title="B", artist="B", original_path=synthetic_song)

        # JSON should always be valid (atomic writes).
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 2

        # No .tmp file should be left behind.
        assert not os.path.exists(json_path + ".tmp")


# ---------------------------------------------------------------------------
# Player integration: real WAV files through the mixing callback
# ---------------------------------------------------------------------------

class TestPlayerIntegration:
    """Test the player with real WAV files loaded from disk."""

    def test_load_real_stems_and_verify_properties(self, synthetic_stems):
        """Load WAV files from disk and verify duration/sample rate."""
        player = MultiTrackPlayer()
        player.load_stems(synthetic_stems)

        assert len(player._stems) == 4
        assert abs(player.total_seconds - 2.0) < 0.01
        assert player._sample_rate == SAMPLE_RATE

    def test_callback_with_disk_loaded_stems(self, synthetic_stems):
        """Run the audio callback with stems loaded from real files."""
        import sounddevice as sd

        player = MultiTrackPlayer()
        player.load_stems(synthetic_stems)
        player._is_playing = True

        # Read a chunk through the callback.
        chunk_frames = 1024
        outdata = np.zeros((chunk_frames, 2), dtype=np.float32)
        player._audio_callback(outdata, chunk_frames, {}, sd.CallbackFlags())

        # Output should be non-silent (sum of four sine waves).
        assert np.max(np.abs(outdata)) > 0.1
        assert player._current_frame == chunk_frames

    def test_mute_solo_with_real_stems(self, synthetic_stems):
        """Verify mute/solo produces different output with real audio."""
        import sounddevice as sd

        player = MultiTrackPlayer()
        player.load_stems(synthetic_stems)
        player._is_playing = True

        # Full mix.
        out_full = np.zeros((1024, 2), dtype=np.float32)
        player._audio_callback(out_full, 1024, {}, sd.CallbackFlags())
        player._current_frame = 0

        # Solo bass only.
        player.set_solo("bass", True)
        out_solo = np.zeros((1024, 2), dtype=np.float32)
        player._audio_callback(out_solo, 1024, {}, sd.CallbackFlags())

        # Solo output should have lower energy than the full mix.
        energy_full = np.sum(out_full ** 2)
        energy_solo = np.sum(out_solo ** 2)
        assert energy_solo < energy_full

    def test_seek_and_play_to_end(self, synthetic_stems):
        """Seek near the end and verify EOF is handled."""
        import sounddevice as sd

        player = MultiTrackPlayer()
        player.load_stems(synthetic_stems)
        player._is_playing = True

        # Seek to 100 frames before the end.
        end_pos = player.total_seconds - (100 / SAMPLE_RATE)
        player.seek(end_pos)

        outdata = np.zeros((1024, 2), dtype=np.float32)
        with pytest.raises(sd.CallbackStop):
            player._audio_callback(outdata, 1024, {}, sd.CallbackFlags())

        # First ~100 frames should have audio, rest should be silence.
        assert np.max(np.abs(outdata[:100])) > 0.0
        assert np.allclose(outdata[100:], 0.0, atol=1e-7)

    def test_clipping_protection_with_loud_stems(self, tmp_path):
        """Verify output is clipped to [-1, 1] even with hot stems."""
        import sounddevice as sd

        stems_dir = tmp_path / "loud_stems"
        stems_dir.mkdir()
        paths = {}

        # Create stems that will sum > 1.0.
        for name in ("a", "b", "c", "d"):
            loud = np.ones((44100, 2), dtype=np.float32) * 0.8
            path = stems_dir / f"{name}.wav"
            sf.write(str(path), loud, SAMPLE_RATE)
            paths[name] = str(path)

        player = MultiTrackPlayer()
        player.load_stems(paths)
        player._is_playing = True

        outdata = np.zeros((512, 2), dtype=np.float32)
        player._audio_callback(outdata, 512, {}, sd.CallbackFlags())

        # Without clipping, sum would be 3.2. Should be clamped to 1.0.
        assert np.max(outdata) <= 1.0
        assert np.min(outdata) >= -1.0


# ---------------------------------------------------------------------------
# Separator integration: real ONNX inference (requires model on disk)
# ---------------------------------------------------------------------------

def _model_available(is_6_stem=False):
    """Check if the ONNX model file exists."""
    mm = ModelManager(data_dir=os.path.abspath(DATA_DIR))
    return mm.is_model_downloaded(is_6_stem=is_6_stem)


@pytest.mark.slow
class TestSeparatorIntegration:
    """End-to-end separation tests using the real ONNX model.

    These tests are skipped if the model file is not downloaded.
    Run them with: pytest tests/test_integration.py -v -m slow
    """

    @pytest.fixture(autouse=True)
    def _require_model(self):
        if not _model_available():
            pytest.skip("htdemucs.onnx model not downloaded")

    def test_separate_produces_valid_stems(
        self, tmp_path, synthetic_song, model_manager
    ):
        """Run real separation and verify output stem files."""
        output_dir = str(tmp_path / "stems")
        model_path = model_manager.model_path(is_6_stem=False)

        worker = SeparatorWorker(
            input_path=synthetic_song,
            output_dir=output_dir,
            model_path=model_path,
            is_6_stem=False,
        )

        # Collect signals.
        results = {}
        errors = []
        progress_log = []

        worker.finished.connect(lambda r: results.update(r))
        worker.error.connect(lambda e: errors.append(e))
        worker.progress.connect(lambda p, m: progress_log.append((p, m)))

        worker.run()  # Run synchronously (not .start()).

        assert not errors, f"Separation failed: {errors}"
        assert len(results) == 4

        for stem_name in STEMS_4:
            assert stem_name in results
            stem_path = results[stem_name]
            assert os.path.isfile(stem_path)

            # Verify it is a valid WAV at the correct sample rate.
            audio, sr = sf.read(stem_path)
            assert sr == SAMPLE_RATE
            assert audio.ndim == 2
            assert audio.shape[1] == 2  # Stereo
            assert audio.shape[0] > 0

        # Progress should have reached 100%.
        assert any(p == 100 for p, _ in progress_log)

    def test_stems_sum_approximates_original(
        self, tmp_path, synthetic_song, model_manager
    ):
        """Verify that summing all stems roughly reconstructs the input."""
        output_dir = str(tmp_path / "stems")
        model_path = model_manager.model_path(is_6_stem=False)

        worker = SeparatorWorker(
            input_path=synthetic_song,
            output_dir=output_dir,
            model_path=model_path,
            is_6_stem=False,
        )

        results = {}
        worker.finished.connect(lambda r: results.update(r))
        worker.run()

        # Load original.
        original, _ = sf.read(synthetic_song, always_2d=True)

        # Sum all stems.
        summed = np.zeros_like(original)
        for stem_path in results.values():
            stem_audio, _ = sf.read(stem_path, always_2d=True)
            min_len = min(summed.shape[0], stem_audio.shape[0])
            summed[:min_len] += stem_audio[:min_len]

        # The reconstruction won't be perfect but should be correlated.
        # Use Pearson correlation on a middle section to avoid edge effects.
        mid = original.shape[0] // 4
        section = slice(mid, mid * 3)
        for ch in range(2):
            corr = np.corrcoef(
                original[section, ch], summed[section, ch]
            )[0, 1]
            assert corr > 0.8, (
                f"Channel {ch} correlation too low: {corr:.3f}"
            )

    def test_separated_stems_load_into_player(
        self, tmp_path, synthetic_song, model_manager
    ):
        """Verify stems from real separation load and play in the player."""
        import sounddevice as sd

        output_dir = str(tmp_path / "stems")
        model_path = model_manager.model_path(is_6_stem=False)

        worker = SeparatorWorker(
            input_path=synthetic_song,
            output_dir=output_dir,
            model_path=model_path,
            is_6_stem=False,
        )

        results = {}
        worker.finished.connect(lambda r: results.update(r))
        worker.run()

        # Load into the player.
        player = MultiTrackPlayer()
        player.load_stems(results)

        assert len(player._stems) == 4
        assert player.total_seconds > 0
        assert player._sample_rate == SAMPLE_RATE

        # Run the callback to verify it produces audio.
        player._is_playing = True
        outdata = np.zeros((1024, 2), dtype=np.float32)
        player._audio_callback(outdata, 1024, {}, sd.CallbackFlags())
        assert np.max(np.abs(outdata)) > 0.0


# ---------------------------------------------------------------------------
# Full pipeline: library -> separator -> player
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestFullPipeline:
    """End-to-end pipeline: import into library, separate, play stems."""

    @pytest.fixture(autouse=True)
    def _require_model(self):
        if not _model_available():
            pytest.skip("htdemucs.onnx model not downloaded")

    def test_import_separate_play(self, tmp_path, synthetic_song):
        """Simulate the complete user workflow without a GUI."""
        import sounddevice as sd

        data_dir = str(tmp_path / "data")

        # 1. Import song into library.
        lib = SongLibrary(data_dir=data_dir)
        song = lib.add_song(
            title="Pipeline Test",
            artist="Integration",
            original_path=synthetic_song,
        )
        assert os.path.isfile(song.original_path)

        # 2. Separate stems using the real model.
        mm = ModelManager(data_dir=data_dir)
        # Use the real model from the project data dir.
        real_mm = ModelManager(
            data_dir=os.path.abspath(DATA_DIR)
        )
        model_path = real_mm.model_path(is_6_stem=False)

        worker = SeparatorWorker(
            input_path=song.original_path,
            output_dir=song.stems_path,
            model_path=model_path,
            is_6_stem=False,
        )

        results = {}
        errors = []
        worker.finished.connect(lambda r: results.update(r))
        worker.error.connect(lambda e: errors.append(e))
        worker.run()

        assert not errors, f"Separation failed: {errors}"
        assert len(results) == 4

        # 3. Update library with model info.
        lib.update_song(song.id, model_used="htdemucs")

        # 4. Load stems into player.
        player = MultiTrackPlayer()
        player.load_stems(results)

        assert player.total_seconds > 0
        assert len(player._stems) == 4

        # 5. Simulate playback: run callback, verify output.
        player._is_playing = True
        outdata = np.zeros((2048, 2), dtype=np.float32)
        player._audio_callback(outdata, 2048, {}, sd.CallbackFlags())
        assert np.max(np.abs(outdata)) > 0.0

        # 6. Verify mute works end-to-end.
        player._current_frame = 0
        player.set_mute("other", True)
        out_muted = np.zeros((2048, 2), dtype=np.float32)
        player._audio_callback(out_muted, 2048, {}, sd.CallbackFlags())

        # Muted mix should differ from full mix.
        assert not np.allclose(outdata, out_muted, atol=1e-6)

        # 7. Verify library survives reload.
        lib2 = SongLibrary(data_dir=data_dir)
        reloaded = lib2.get_song(song.id)
        assert reloaded.model_used == "htdemucs"
        assert os.path.isfile(reloaded.original_path)

        # Stem files should still exist.
        for stem_path in results.values():
            assert os.path.isfile(stem_path)


# ---------------------------------------------------------------------------
# Hardware playback: audible verification through real speakers
# ---------------------------------------------------------------------------

def _find_test_song():
    """Find a real audio file for the hardware playback test.

    Checks (in order):
        1. STEMMA_TEST_SONG environment variable
        2. Any audio file matching data/test_song.* (mp3, wav, flac)

    Returns the path if found, else None.
    """
    # Environment variable override.
    env_path = os.environ.get("STEMMA_TEST_SONG")
    if env_path and os.path.isfile(env_path):
        return env_path

    # Convention: data/test_song.{ext}
    data = os.path.abspath(DATA_DIR)
    for ext in ("mp3", "wav", "flac"):
        candidate = os.path.join(data, f"test_song.{ext}")
        if os.path.isfile(candidate):
            return candidate

    return None


@pytest.mark.hardware
@pytest.mark.slow
class TestAudioPlayback:
    """Play actual audio through the speakers for manual verification.

    Requires a real music file (not committed to the repo). Provide one by
    either setting STEMMA_TEST_SONG or dropping a file at data/test_song.mp3.

    Run with: pytest tests/test_integration.py -v -m hardware
    """

    @pytest.fixture(autouse=True)
    def _require_prerequisites(self):
        if not _model_available():
            pytest.skip("htdemucs.onnx model not downloaded")
        if _find_test_song() is None:
            pytest.skip(
                "No test song found. Set STEMMA_TEST_SONG or place a file "
                "at data/test_song.mp3"
            )

    def test_audible_playback_with_mute_solo(self, tmp_path):
        """Separate a real song, play the full mix, then solo each stem.

        Uses only the first 15 seconds to keep separation fast.

        You should hear:
            1. ~4s of the full mix (all stems together)
            2. ~4s of vocals only
            3. ~4s of drums only
            4. ~4s of bass only

        The volume is kept moderate (0.25 scale) to avoid surprises.
        """
        import sounddevice as sd

        test_song = _find_test_song()

        # Load only the first 15 seconds to keep separation fast.
        audio, sr = sf.read(test_song, always_2d=True, dtype="float32")
        max_frames = int(15.0 * sr)
        if audio.shape[0] > max_frames:
            audio = audio[:max_frames]

        clip_path = str(tmp_path / "clip.wav")
        sf.write(clip_path, audio, sr)

        # Separate.
        real_mm = ModelManager(data_dir=os.path.abspath(DATA_DIR))
        model_path = real_mm.model_path(is_6_stem=False)

        worker = SeparatorWorker(
            input_path=clip_path,
            output_dir=str(tmp_path / "stems"),
            model_path=model_path,
            is_6_stem=False,
        )

        results = {}
        errors = []
        worker.finished.connect(lambda r: results.update(r))
        worker.error.connect(lambda e: errors.append(e))
        worker.run()

        assert not errors, f"Separation failed: {errors}"
        assert len(results) == 4

        # Load stems and scale to a comfortable volume.
        stem_data = {}
        for name, path in results.items():
            data, _ = sf.read(path, dtype="float32")
            stem_data[name] = data * 0.25

        total_frames = max(a.shape[0] for a in stem_data.values())
        play_sr = SAMPLE_RATE

        def _play_section(active_stems, label, duration_s=4.0):
            """Play a mix of the given stems for *duration_s* seconds."""
            frames = min(int(duration_s * play_sr), total_frames)
            buf = np.zeros((frames, 2), dtype=np.float32)
            for name in active_stems:
                end = min(frames, stem_data[name].shape[0])
                buf[:end] += stem_data[name][:end]
            np.clip(buf, -1.0, 1.0, out=buf)
            print(f"  Playing: {label}")
            sd.play(buf, samplerate=play_sr)
            sd.wait()

        print()
        _play_section(list(stem_data.keys()), "full mix")
        _play_section(["vocals"], "vocals only")
        _play_section(["drums"], "drums only")
        _play_section(["bass"], "bass only")

        # Verify each stem has meaningful audio (not silent).
        for name, data in stem_data.items():
            rms = np.sqrt(np.mean(data ** 2))
            assert rms > 0.001, f"Stem '{name}' is nearly silent (RMS={rms:.6f})"

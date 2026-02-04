"""Tests for the stem separation engine."""

import os

import numpy as np
import pytest
import soundfile as sf

from src.separator import SAMPLE_RATE, STEMS_4, STEMS_6, SeparatorWorker


class TestSeparatorConstants:
    """Verify module-level constants are correctly defined."""

    def test_sample_rate(self):
        assert SAMPLE_RATE == 44100

    def test_stems_4(self):
        assert STEMS_4 == ("vocals", "drums", "bass", "other")

    def test_stems_6(self):
        assert STEMS_6 == ("vocals", "drums", "bass", "other", "guitar", "piano")


class TestSeparatorWorkerInit:
    """Verify SeparatorWorker initialisation and properties."""

    def test_default_is_4_stem(self, tmp_dir, sample_audio_path):
        worker = SeparatorWorker(
            input_path=sample_audio_path,
            output_dir=tmp_dir,
            model_path="fake_model.onnx",
        )
        assert worker.stems == STEMS_4

    def test_6_stem_flag(self, tmp_dir, sample_audio_path):
        worker = SeparatorWorker(
            input_path=sample_audio_path,
            output_dir=tmp_dir,
            model_path="fake_model.onnx",
            is_6_stem=True,
        )
        assert worker.stems == STEMS_6

    def test_cancel_sets_flag(self, tmp_dir, sample_audio_path):
        worker = SeparatorWorker(
            input_path=sample_audio_path,
            output_dir=tmp_dir,
            model_path="fake_model.onnx",
        )
        assert not worker._is_cancelled
        worker.cancel()
        assert worker._is_cancelled


class TestSeparatorWorkerLoadAudio:
    """Verify audio loading logic."""

    def test_load_valid_file(self, tmp_dir, sample_audio_path):
        worker = SeparatorWorker(
            input_path=sample_audio_path,
            output_dir=tmp_dir,
            model_path="fake_model.onnx",
        )
        audio, sr = worker._load_audio()
        assert sr == 44100
        assert audio.shape[0] == 2  # stereo
        assert audio.shape[1] == 44100  # 1 second

    def test_load_missing_file_raises(self, tmp_dir):
        worker = SeparatorWorker(
            input_path="/nonexistent/audio.wav",
            output_dir=tmp_dir,
            model_path="fake_model.onnx",
        )
        with pytest.raises(FileNotFoundError, match="Input audio file not found"):
            worker._load_audio()

    def test_load_mono_becomes_2d(self, tmp_dir):
        """Mono files should still produce a (channels, samples) array."""
        mono_path = os.path.join(tmp_dir, "mono.wav")
        mono = np.zeros((44100,), dtype=np.float32)
        sf.write(mono_path, mono, 44100)

        worker = SeparatorWorker(
            input_path=mono_path,
            output_dir=tmp_dir,
            model_path="fake_model.onnx",
        )
        audio, sr = worker._load_audio()
        assert audio.ndim == 2
        assert audio.shape[0] == 1  # mono -> 1 channel


class TestSeparatorWorkerCreateSession:
    """Verify ONNX session creation."""

    def test_missing_model_raises(self, tmp_dir, sample_audio_path):
        worker = SeparatorWorker(
            input_path=sample_audio_path,
            output_dir=tmp_dir,
            model_path="/nonexistent/model.onnx",
        )
        with pytest.raises(FileNotFoundError, match="ONNX model file not found"):
            worker._create_session()

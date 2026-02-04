"""Tests for the stem separation engine."""

import os

import numpy as np
import pytest
import soundfile as sf

from src.separator import (
    HOP_LENGTH,
    NFFT,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    STEMS_4,
    STEMS_6,
    SeparatorWorker,
)


class TestSeparatorConstants:
    """Verify module-level constants are correctly defined."""

    def test_sample_rate(self):
        assert SAMPLE_RATE == 44100

    def test_nfft(self):
        assert NFFT == 4096

    def test_hop_length(self):
        assert HOP_LENGTH == 1024

    def test_segment_samples(self):
        assert SEGMENT_SAMPLES == 343980

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
        assert audio.dtype == np.float32

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


class TestSeparatorResample:
    """Verify audio resampling logic."""

    def test_no_resample_when_same_rate(self, tmp_dir, sample_audio_path):
        worker = SeparatorWorker(
            input_path=sample_audio_path,
            output_dir=tmp_dir,
            model_path="fake_model.onnx",
        )
        audio = np.random.randn(2, 44100).astype(np.float32)
        result = worker._resample(audio, 44100)
        np.testing.assert_array_equal(result, audio)

    def test_resample_from_48khz(self, tmp_dir, sample_audio_path):
        worker = SeparatorWorker(
            input_path=sample_audio_path,
            output_dir=tmp_dir,
            model_path="fake_model.onnx",
        )
        # 1 second of audio at 48kHz.
        audio = np.random.randn(2, 48000).astype(np.float32)
        result = worker._resample(audio, 48000)
        assert result.shape[0] == 2
        # 48000 samples at 48kHz -> 44100 samples at 44.1kHz.
        assert result.shape[1] == 44100

    def test_resample_preserves_channels(self, tmp_dir, sample_audio_path):
        worker = SeparatorWorker(
            input_path=sample_audio_path,
            output_dir=tmp_dir,
            model_path="fake_model.onnx",
        )
        audio = np.random.randn(2, 22050).astype(np.float32)
        result = worker._resample(audio, 22050)
        assert result.shape[0] == 2


class TestSeparatorCreateSession:
    """Verify ONNX session creation."""

    def test_missing_model_raises(self, tmp_dir, sample_audio_path):
        worker = SeparatorWorker(
            input_path=sample_audio_path,
            output_dir=tmp_dir,
            model_path="/nonexistent/model.onnx",
        )
        with pytest.raises(FileNotFoundError, match="ONNX model file not found"):
            worker._create_session()


class TestSeparatorSTFT:
    """Verify STFT complex-as-channels computation."""

    def test_stft_cac_output_shape(self, tmp_dir, sample_audio_path):
        worker = SeparatorWorker(
            input_path=sample_audio_path,
            output_dir=tmp_dir,
            model_path="fake_model.onnx",
        )
        # Create a stereo signal of SEGMENT_SAMPLES length.
        audio = np.random.randn(2, SEGMENT_SAMPLES).astype(np.float32)
        result = worker._compute_stft_cac(audio)

        # Should have 4 channels: left_real, left_imag, right_real, right_imag.
        assert result.shape[0] == 4
        # Frequency bins should be NFFT // 2 = 2048.
        assert result.shape[1] == NFFT // 2
        # Time frames: for the model's fixed segment.
        assert result.shape[2] > 0
        assert result.dtype == np.float32

    def test_stft_cac_real_imag_split(self, tmp_dir, sample_audio_path):
        """Verify that the CaC channels contain real and imaginary parts."""
        worker = SeparatorWorker(
            input_path=sample_audio_path,
            output_dir=tmp_dir,
            model_path="fake_model.onnx",
        )
        # Use a simple sine wave so we can verify non-zero values.
        t = np.linspace(0, 1, SAMPLE_RATE, dtype=np.float32)
        sine = np.sin(2 * np.pi * 440 * t)
        audio = np.stack([sine, sine])  # stereo

        # Pad to SEGMENT_SAMPLES.
        padded = np.pad(audio, ((0, 0), (0, SEGMENT_SAMPLES - SAMPLE_RATE)))
        result = worker._compute_stft_cac(padded)

        # Real and imaginary parts should not all be zero for a sine wave.
        assert np.any(result[0] != 0), "Left real should have non-zero values"
        assert np.any(result[1] != 0), "Left imag should have non-zero values"


class TestSeparatorISTFT:
    """Verify iSTFT reconstruction from CaC spectrogram."""

    def test_istft_output_shape(self, tmp_dir, sample_audio_path):
        worker = SeparatorWorker(
            input_path=sample_audio_path,
            output_dir=tmp_dir,
            model_path="fake_model.onnx",
        )
        # Create a dummy CaC spectrogram with the right dimensions.
        time_frames = 336  # Matches model output shape.
        cac_spec = np.random.randn(4, NFFT // 2, time_frames).astype(np.float32)

        result = worker._compute_istft_cac(cac_spec, length=SEGMENT_SAMPLES)
        assert result.shape == (2, SEGMENT_SAMPLES)
        assert result.dtype == np.float32

    def test_stft_istft_roundtrip(self, tmp_dir, sample_audio_path):
        """Verify that STFT -> iSTFT approximately reconstructs the input."""
        worker = SeparatorWorker(
            input_path=sample_audio_path,
            output_dir=tmp_dir,
            model_path="fake_model.onnx",
        )
        t = np.linspace(0, 1, SAMPLE_RATE, dtype=np.float32)
        sine = np.sin(2 * np.pi * 440 * t)
        audio = np.stack([sine, sine])

        # Pad to segment length.
        padded = np.pad(audio, ((0, 0), (0, SEGMENT_SAMPLES - SAMPLE_RATE)))

        # Forward STFT.
        cac = worker._compute_stft_cac(padded)

        # Inverse STFT.
        reconstructed = worker._compute_istft_cac(cac, length=SEGMENT_SAMPLES)

        # The roundtrip should be approximately equal (within floating point).
        # Only check the non-padded region for meaningful signal.
        np.testing.assert_allclose(
            reconstructed[:, :SAMPLE_RATE],
            padded[:, :SAMPLE_RATE],
            atol=1e-4,
            rtol=1e-4,
        )


class TestSeparatorSaveStems:
    """Verify stem file writing."""

    def test_saves_4_stems(self, tmp_dir, sample_audio_path):
        worker = SeparatorWorker(
            input_path=sample_audio_path,
            output_dir=os.path.join(tmp_dir, "output"),
            model_path="fake_model.onnx",
        )
        # Create dummy separated audio.
        separated = np.random.randn(4, 2, 44100).astype(np.float32)
        result = worker._save_stems(separated)

        assert len(result) == 4
        for stem_name in STEMS_4:
            assert stem_name in result
            assert os.path.isfile(result[stem_name])

            # Verify the file is valid audio.
            audio, sr = sf.read(result[stem_name])
            assert sr == 44100
            assert audio.shape == (44100, 2)

    def test_saves_6_stems(self, tmp_dir, sample_audio_path):
        worker = SeparatorWorker(
            input_path=sample_audio_path,
            output_dir=os.path.join(tmp_dir, "output"),
            model_path="fake_model.onnx",
            is_6_stem=True,
        )
        separated = np.random.randn(6, 2, 44100).astype(np.float32)
        result = worker._save_stems(separated)

        assert len(result) == 6
        for stem_name in STEMS_6:
            assert stem_name in result
            assert os.path.isfile(result[stem_name])

"""Tests for the MDX-Net 2-stem separation engine.

The heavy path (real ONNX inference) is exercised by the gated
integration test at the bottom; everything else runs against a fake
session so the suite stays fast and offline.

The key correctness property: with an identity model (output equals
input spectrogram), the whole STFT -> chunk -> inference -> iSTFT ->
overlap pipeline must reconstruct the input waveform almost exactly.
That validates the packing/windowing math independently of any model.
"""

import os

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtWidgets import QApplication

from src.mdx_separator import (
    MDX_MODELS,
    MdxSeparatorWorker,
    PRIMARY_TO_FILES,
    hash_model_file,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _IdentitySession:
    """Fake ORT session: returns the input spectrogram unchanged."""

    class _Input:
        name = "input"

    def get_inputs(self):
        return [self._Input()]

    def run(self, _outputs, feeds):
        return [feeds["input"]]


class _HalfSession(_IdentitySession):
    """Fake ORT session: returns the input at half amplitude."""

    def run(self, _outputs, feeds):
        return [feeds["input"] * 0.5]


def _make_worker(tmp_path, session, seconds=3.0, sr=44100):
    """Build a worker over a synthetic stereo tone file."""
    t = np.arange(int(seconds * sr)) / sr
    left = 0.4 * np.sin(2 * np.pi * 220 * t)
    right = 0.4 * np.sin(2 * np.pi * 330 * t)
    audio = np.stack([left, right], axis=1).astype(np.float32)
    src = tmp_path / "tone.wav"
    sf.write(str(src), audio, sr)

    worker = MdxSeparatorWorker(
        input_path=str(src),
        output_dir=str(tmp_path / "out"),
        model_path="unused.onnx",
    )
    worker._create_session = lambda: session
    return worker, audio.T


class TestRegistry:
    def test_registry_entries_are_complete(self):
        for key, info in MDX_MODELS.items():
            for field in ("file", "url", "md5_tail", "n_fft", "hop",
                          "dim_f", "dim_t", "compensate", "primary_stem"):
                assert field in info, f"{key} missing {field}"
            assert info["primary_stem"] in PRIMARY_TO_FILES

    def test_hash_model_file_small_file(self, tmp_path):
        """Files under 10,000 KiB hash from the start (seek fallback)."""
        p = tmp_path / "m.onnx"
        p.write_bytes(b"model-bytes")
        import hashlib
        assert hash_model_file(str(p)) == hashlib.md5(b"model-bytes").hexdigest()


class TestPipelineReconstruction:
    def test_identity_model_reconstructs_input(self, app, tmp_path):
        """Identity 'model' -> primary stem equals the mix.

        This validates STFT packing, bin cropping/padding, window trim
        and reassembly in one shot: any mismatch in the math shows up as
        reconstruction error.
        """
        worker, audio = _make_worker(tmp_path, _IdentitySession())
        session = worker._create_session()
        primary = worker._demix(audio, session)

        compensate = worker.params["compensate"]
        expected = audio * compensate
        err = np.max(np.abs(primary - expected))
        # The model's dim_f crop discards the top ~1.5% of the spectrum,
        # so reconstruction is near-exact for band-limited content.
        assert err < 1e-3, f"reconstruction error {err}"

    def test_half_model_scales_output(self, app, tmp_path):
        worker, audio = _make_worker(tmp_path, _HalfSession())
        session = worker._create_session()
        primary = worker._demix(audio, session)
        expected = audio * 0.5 * worker.params["compensate"]
        assert np.max(np.abs(primary - expected)) < 1e-3

    def test_demix_preserves_length(self, app, tmp_path):
        """Output length must equal input length for any duration,
        including ones that don't divide evenly into windows."""
        for seconds in (0.7, 3.0, 6.2):
            worker, audio = _make_worker(
                tmp_path, _IdentitySession(), seconds=seconds,
            )
            session = worker._create_session()
            primary = worker._demix(audio, session)
            assert primary.shape == audio.shape


class TestFullRun:
    def test_run_writes_both_stems_and_emits_finished(self, app, tmp_path):
        worker, audio = _make_worker(tmp_path, _IdentitySession())
        results = {}
        worker.finished.connect(lambda d: results.update(d))
        errors = []
        worker.error.connect(lambda m: errors.append(m))

        worker.run()

        assert not errors
        assert set(results) == {"vocals", "other"}
        for path in results.values():
            assert os.path.isfile(path)
        # Identity model: primary (Instrumental -> other.wav) carries the
        # mix; secondary (vocals) = mix - primary*compensate is small.
        other, _ = sf.read(results["other"], always_2d=True)
        vocals, _ = sf.read(results["vocals"], always_2d=True)
        assert np.sqrt(np.mean(other ** 2)) > 0.1
        assert np.sqrt(np.mean(vocals ** 2)) < 0.05

    def test_cancel_mid_run_emits_no_finished(self, app, tmp_path):
        worker, audio = _make_worker(tmp_path, _IdentitySession())
        finished = []
        worker.finished.connect(lambda d: finished.append(d))
        errors = []
        worker.error.connect(lambda m: errors.append(m))
        # Cancel as soon as the first progress tick arrives.
        worker.progress.connect(lambda p, m: worker.cancel())

        worker.run()

        assert finished == []
        assert errors and "cancelled" in errors[0].lower()

    def test_missing_input_emits_error(self, app, tmp_path):
        worker = MdxSeparatorWorker(
            input_path=str(tmp_path / "nope.mp3"),
            output_dir=str(tmp_path / "out"),
            model_path="unused.onnx",
        )
        errors = []
        worker.error.connect(lambda m: errors.append(m))
        worker.run()
        assert errors and "not found" in errors[0]


class TestDownloadVerification:
    def test_md5_tail_mismatch_removes_file_and_errors(self, tmp_path):
        """A downloaded MDX model failing the integrity check must be
        deleted and surfaced as an error, never kept as 'cached'."""
        from src.model_manager import ModelDownloader
        from unittest.mock import patch
        import io

        class _Resp:
            headers = {"Content-Length": "9"}
            def read(self, n=-1):
                return self._buf.read(n)
            def __enter__(self):
                self._buf = io.BytesIO(b"bad-bytes")
                return self
            def __exit__(self, *a):
                return False

        dl = ModelDownloader(
            "mdx_inst_hq3", str(tmp_path),
            url="http://x/m.onnx", file_name="m.onnx",
            expected_md5_tail="0" * 32,
        )
        errors = []
        dl.error.connect(lambda m: errors.append(m))
        with patch("src.model_manager.urllib.request.urlopen",
                   return_value=_Resp()):
            dl.run()

        assert errors and "integrity" in errors[0]
        assert not os.path.exists(tmp_path / "m.onnx")


# ---------------------------------------------------------------------------
# Gated integration test: real model, real inference (slow marker).
# Runs only when the MDX model file is already cached locally.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_model_separates_tone(app, tmp_path):
    from src.data_paths import platform_user_data_dir

    model = os.path.join(
        platform_user_data_dir(), "models",
        MDX_MODELS["mdx_inst_hq3"]["file"],
    )
    if not os.path.isfile(model):
        pytest.skip("MDX model not cached locally")

    t = np.arange(44100 * 4) / 44100
    audio = np.stack([np.sin(2 * np.pi * 220 * t)] * 2, axis=1) * 0.4
    src = tmp_path / "tone.wav"
    sf.write(str(src), audio.astype(np.float32), 44100)

    worker = MdxSeparatorWorker(
        input_path=str(src), output_dir=str(tmp_path / "out"),
        model_path=model,
    )
    results = {}
    worker.finished.connect(lambda d: results.update(d))
    worker.run()

    assert set(results) == {"vocals", "other"}
    # A pure tone is not vocals: the instrumental stem should carry
    # nearly all the energy.
    other, _ = sf.read(results["other"], always_2d=True)
    vocals, _ = sf.read(results["vocals"], always_2d=True)
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    assert rms(other) > 5 * rms(vocals)

"""Tests for the model download and cache manager."""

import hashlib
import io
import os
from unittest.mock import patch

import pytest

import src.model_manager as model_manager
from src.model_manager import ModelDownloader, ModelManager, _MODEL_FILES


class _FakeResponse:
    """Minimal stand-in for the urlopen context manager."""

    def __init__(self, body: bytes, content_length: int | None = None):
        self._buf = io.BytesIO(body)
        length = len(body) if content_length is None else content_length
        self.headers = {"Content-Length": str(length)} if length is not None else {}

    def read(self, size: int = -1) -> bytes:
        return self._buf.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestModelManager:
    """Verify ModelManager path resolution and state checks."""

    def test_model_path_4_stem(self, tmp_dir):
        manager = ModelManager(data_dir=tmp_dir)
        path = manager.model_path(is_6_stem=False)
        assert path.endswith("htdemucs.onnx")

    def test_model_path_6_stem(self, tmp_dir):
        manager = ModelManager(data_dir=tmp_dir)
        path = manager.model_path(is_6_stem=True)
        assert path.endswith("htdemucs_6s.onnx")

    def test_is_model_downloaded_false_when_missing(self, tmp_dir):
        manager = ModelManager(data_dir=tmp_dir)
        assert not manager.is_model_downloaded(is_6_stem=False)

    def test_is_model_downloaded_true_when_exists(self, tmp_dir):
        manager = ModelManager(data_dir=tmp_dir)
        models_dir = os.path.join(tmp_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        for name in ("htdemucs.onnx", "htdemucs.onnx.data"):
            with open(os.path.join(models_dir, name), "wb") as f:
                f.write(b"dummy")
        assert manager.is_model_downloaded(is_6_stem=False)

    def test_is_model_downloaded_6_stem_requires_both_artifacts(self, tmp_dir):
        manager = ModelManager(data_dir=tmp_dir)
        models_dir = os.path.join(tmp_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, "htdemucs_6s.onnx"), "wb") as f:
            f.write(b"stub")
        assert not manager.is_model_downloaded(is_6_stem=True)
        with open(os.path.join(models_dir, "htdemucs_6s.onnx.data"), "wb") as f:
            f.write(b"weights")
        assert manager.is_model_downloaded(is_6_stem=True)

    def test_download_model_returns_downloader(self, tmp_dir):
        manager = ModelManager(data_dir=tmp_dir)
        downloader = manager.download_model(is_6_stem=False)
        assert isinstance(downloader, ModelDownloader)

    def test_beat_model_path(self, tmp_dir):
        manager = ModelManager(data_dir=tmp_dir)
        assert manager.beat_model_path().endswith("beat_this.onnx")

    def test_is_beat_model_downloaded_false(self, tmp_dir):
        manager = ModelManager(data_dir=tmp_dir)
        assert not manager.is_beat_model_downloaded()

    def test_is_beat_model_downloaded_true(self, tmp_dir):
        manager = ModelManager(data_dir=tmp_dir)
        models_dir = os.path.join(tmp_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, "beat_this.onnx"), "wb") as f:
            f.write(b"dummy")
        assert manager.is_beat_model_downloaded()

    def test_download_beat_model_returns_downloader(self, tmp_dir):
        manager = ModelManager(data_dir=tmp_dir)
        downloader = manager.download_beat_model()
        assert isinstance(downloader, ModelDownloader)


class TestModelDownloader:
    """Verify ModelDownloader initialization and cancellation."""

    def test_init_sets_attributes(self, tmp_dir):
        downloader = ModelDownloader("htdemucs", tmp_dir)
        assert downloader.model_name == "htdemucs"
        assert downloader.models_dir == tmp_dir

    def test_cancel_sets_flag(self, tmp_dir):
        downloader = ModelDownloader("htdemucs", tmp_dir)
        assert not downloader._is_cancelled
        downloader.cancel()
        assert downloader._is_cancelled


class TestModelFiles:
    """Verify the model file name constants."""

    def test_4_stem_artifacts(self):
        assert _MODEL_FILES["htdemucs"][0] == "htdemucs.onnx"
        assert _MODEL_FILES["htdemucs"][1] == "htdemucs.onnx.data"

    def test_6_stem_artifacts(self):
        assert _MODEL_FILES["htdemucs_6s"][0] == "htdemucs_6s.onnx"
        assert _MODEL_FILES["htdemucs_6s"][1] == "htdemucs_6s.onnx.data"

    def test_htdemucs_urls_and_sha256_are_commit_pinned(self):
        assert (
            getattr(model_manager, "_REPO_REVISION", None)
            == "ee08c547c91ef9f20ba19cf6ac2ed059ec9dcca0"
        )
        assert getattr(model_manager, "_MODEL_SHA256", None) == {
            "htdemucs.onnx": (
                "be6fa125c457bc4fcdba43b0506270b5e"
                "d2113872748e8163de817f418db17bb"
            ),
            "htdemucs.onnx.data": (
                "e523708037d55151ac03feae48c9dbea"
                "b9908c086ed8e655e40b70dfaa66a3b8"
            ),
            "htdemucs_6s.onnx": (
                "cd881678a816731121d476c83305663a"
                "343b40b5e2c4e12b200ed220ba19808e"
            ),
            "htdemucs_6s.onnx.data": (
                "3eae380175adb9112c8ea8d105730770"
                "2dca09a82c2ede230897a0976c9a5461"
            ),
        }

    def test_beat_model_url_and_sha256_are_commit_pinned(self):
        url = getattr(model_manager, "_BEAT_THIS_URL", "")
        assert "07ab790a9ec2eda8093d52d249e3ec4f0510ee72" in url
        assert "refs/heads" not in url
        assert getattr(model_manager, "_BEAT_THIS_SHA256", None) == (
            "c5c1466e08abdb03fdeb50668a06f244"
            "b787d564c212490482231a9cfbe9ccbd"
        )


class TestDownloadFile:
    """Exercise the atomic streaming download (previously untested)."""

    def test_downloads_to_final_path_after_sha256_verification(self, tmp_dir):
        body = b"onnx-bytes" * 5000
        expected_sha256 = hashlib.sha256(body).hexdigest()
        dl = ModelDownloader(
            "beat_this",
            tmp_dir,
            url="http://x/m.onnx",
            file_name="m.onnx",
            expected_sha256=expected_sha256,
        )
        os.makedirs(tmp_dir, exist_ok=True)
        dest = os.path.join(tmp_dir, "m.onnx")
        real_replace = os.replace
        publications = []

        def publish(part, final):
            assert part == dest + ".part"
            assert not os.path.exists(final)
            assert open(part, "rb").read() == body
            publications.append((part, final))
            real_replace(part, final)

        with (
            patch(
                "src.model_manager.urllib.request.urlopen",
                return_value=_FakeResponse(body),
            ),
            patch("src.model_manager.os.replace", side_effect=publish),
        ):
            dl._download_file(
                "http://x/m.onnx",
                dest,
                lambda d, t: None,
                expected_sha256=expected_sha256,
            )

        assert publications == [(dest + ".part", dest)]
        assert os.path.isfile(dest)
        assert open(dest, "rb").read() == body
        assert not os.path.exists(dest + ".part")
        assert dl._current_partial_path is None

    def test_sha256_mismatch_removes_part_and_never_publishes(self, tmp_dir):
        body = b"corrupt-model"
        dl = ModelDownloader(
            "beat_this",
            tmp_dir,
            url="http://x/m.onnx",
            file_name="m.onnx",
            expected_sha256="0" * 64,
        )
        dest = os.path.join(tmp_dir, "m.onnx")
        errors = []
        completed = []
        dl.error.connect(errors.append)
        dl.download_complete.connect(completed.append)

        with patch(
            "src.model_manager.urllib.request.urlopen",
            return_value=_FakeResponse(body),
        ):
            dl.run()

        assert errors and "SHA-256" in errors[0]
        assert hashlib.sha256(body).hexdigest() in errors[0]
        assert completed == []
        assert not os.path.exists(dest)
        assert not os.path.exists(dest + ".part")

    def test_incomplete_download_raises_and_leaves_no_final_file(self, tmp_dir):
        """Server promises more bytes than it delivers -> error, and the
        final path stays empty so it isn't mistaken for a cached model."""
        dl = ModelDownloader("beat_this", tmp_dir,
                             url="http://x/m.onnx", file_name="m.onnx")
        os.makedirs(tmp_dir, exist_ok=True)
        dest = os.path.join(tmp_dir, "m.onnx")
        truncated = _FakeResponse(b"only-half", content_length=1000)

        with patch("src.model_manager.urllib.request.urlopen",
                   return_value=truncated):
            with pytest.raises(OSError, match="Incomplete download"):
                dl._download_file("http://x/m.onnx", dest, lambda d, t: None)

        assert not os.path.exists(dest)

    def test_run_cleans_up_partial_on_error(self, tmp_dir):
        """A mid-stream failure leaves neither final nor .part file."""
        dl = ModelDownloader("beat_this", tmp_dir,
                             url="http://x/m.onnx", file_name="m.onnx")
        dest = os.path.join(tmp_dir, "models", "m.onnx")
        dl.models_dir = os.path.join(tmp_dir, "models")

        errors = []
        dl.error.connect(lambda m: errors.append(m))

        with patch("src.model_manager.urllib.request.urlopen",
                   side_effect=OSError("connection reset")):
            dl.run()

        assert errors and "connection reset" in errors[0]
        assert not os.path.exists(dest)
        assert not os.path.exists(dest + ".part")

    def test_cancel_mid_stream_stops_and_leaves_no_final_file(self, tmp_dir):
        dl = ModelDownloader("beat_this", tmp_dir,
                             url="http://x/m.onnx", file_name="m.onnx")
        os.makedirs(tmp_dir, exist_ok=True)
        dest = os.path.join(tmp_dir, "m.onnx")

        def _cancel_after_first_chunk(downloaded, total):
            dl.cancel()

        with patch("src.model_manager.urllib.request.urlopen",
                   return_value=_FakeResponse(b"x" * (1 << 18))):
            with pytest.raises(InterruptedError):
                dl._download_file("http://x/m.onnx", dest,
                                  _cancel_after_first_chunk)

        assert not os.path.exists(dest)
        assert not os.path.exists(dest + ".part")

    def test_stale_partial_is_removed_before_new_download(self, tmp_dir):
        dl = ModelDownloader("beat_this", tmp_dir,
                             url="http://x/m.onnx", file_name="m.onnx")
        os.makedirs(tmp_dir, exist_ok=True)
        dest = os.path.join(tmp_dir, "m.onnx")
        # Leave a stale partial from a hypothetical previous run.
        with open(dest + ".part", "wb") as f:
            f.write(b"garbage-prefix")

        with patch("src.model_manager.urllib.request.urlopen",
                   return_value=_FakeResponse(b"fresh-bytes")):
            dl._download_file("http://x/m.onnx", dest, lambda d, t: None)

        assert open(dest, "rb").read() == b"fresh-bytes"
        assert not os.path.exists(dest + ".part")

    def test_full_multi_artifact_download_completes(self, tmp_dir):
        """The two-artifact htdemucs path renames both files and emits
        download_complete with the graph path."""
        manager_dir = os.path.join(tmp_dir, "models")
        dl = ModelDownloader("htdemucs", manager_dir)
        done = []
        dl.download_complete.connect(lambda p: done.append(p))
        body = b"data" * 100
        fake_hashes = {
            name: hashlib.sha256(body).hexdigest()
            for name in _MODEL_FILES["htdemucs"]
        }

        with (
            patch(
                "src.model_manager.urllib.request.urlopen",
                side_effect=lambda *a, **k: _FakeResponse(body),
            ),
            patch.dict(
                "src.model_manager._MODEL_SHA256",
                fake_hashes,
            ),
        ):
            dl.run()

        assert done and done[0].endswith("htdemucs.onnx")
        for fname in _MODEL_FILES["htdemucs"]:
            assert os.path.isfile(os.path.join(manager_dir, fname))
            assert not os.path.exists(
                os.path.join(manager_dir, fname + ".part")
            )

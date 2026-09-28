"""Tests for the model download and cache manager."""

import hashlib
import io
import os
from unittest.mock import patch

import pytest

import src.model_manager as model_manager
from src.import_messages import MSG_DOWNLOAD_DAMAGED, MSG_RESET
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


@pytest.fixture
def tiny_sizes(monkeypatch):
    """Shrink the expected artifact sizes so tests can write real files."""
    sizes = {
        name: 100 + i * 10
        for i, name in enumerate(model_manager.MODEL_FILE_SIZES)
    }
    monkeypatch.setattr(model_manager, "MODEL_FILE_SIZES", sizes)
    return sizes


def _write(data_dir: str, name: str, size: int) -> str:
    """Write *size* bytes to ``<data_dir>/models/<name>``; return the path."""
    models_dir = os.path.join(data_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, name)
    with open(path, "wb") as f:
        f.write(b"x" * size)
    return path


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

    def test_is_model_downloaded_true_when_exists(self, tmp_dir, tiny_sizes):
        manager = ModelManager(data_dir=tmp_dir)
        for name in ("htdemucs.onnx", "htdemucs.onnx.data"):
            _write(tmp_dir, name, tiny_sizes[name])
        assert manager.is_model_downloaded(is_6_stem=False)

    def test_is_model_downloaded_6_stem_requires_both_artifacts(
        self, tmp_dir, tiny_sizes,
    ):
        manager = ModelManager(data_dir=tmp_dir)
        _write(tmp_dir, "htdemucs_6s.onnx", tiny_sizes["htdemucs_6s.onnx"])
        assert not manager.is_model_downloaded(is_6_stem=True)
        _write(
            tmp_dir, "htdemucs_6s.onnx.data",
            tiny_sizes["htdemucs_6s.onnx.data"],
        )
        assert manager.is_model_downloaded(is_6_stem=True)

    def test_wrong_size_file_is_not_downloaded(self, tmp_dir, tiny_sizes):
        """A truncated weights file used to count as cached and then
        failed to load (INVALID_PROTOBUF) on every import."""
        manager = ModelManager(data_dir=tmp_dir)
        _write(tmp_dir, "htdemucs.onnx", tiny_sizes["htdemucs.onnx"])
        _write(tmp_dir, "htdemucs.onnx.data", 3)
        assert not manager.is_model_downloaded(is_6_stem=False)
        _write(tmp_dir, "UVR-MDX-NET-Inst_HQ_3.onnx", 7)
        assert not manager.is_mdx_model_downloaded("mdx_inst_hq3")

    def test_download_size_counts_only_missing_files(self, tmp_dir):
        manager = ModelManager(data_dir=tmp_dir)
        assert manager.download_size_mb("htdemucs") == 172
        assert manager.download_size_mb("htdemucs_6s") == 114
        assert manager.download_size_mb("mdx_inst_hq3") == 64
        assert manager.download_size_mb("beat_this") == 79

    def test_download_size_zero_when_cached(self, tmp_dir, tiny_sizes):
        manager = ModelManager(data_dir=tmp_dir)
        _write(tmp_dir, "htdemucs.onnx", tiny_sizes["htdemucs.onnx"])
        assert manager.download_size_bytes("htdemucs") == (
            tiny_sizes["htdemucs.onnx.data"]
        )
        _write(tmp_dir, "htdemucs.onnx.data", tiny_sizes["htdemucs.onnx.data"])
        assert manager.download_size_bytes("htdemucs") == 0

    def test_labels_name_models_not_files(self):
        assert model_manager.model_label("htdemucs") == "4-stem model"
        assert model_manager.model_label("htdemucs_6s") == "6-stem model"
        assert model_manager.model_label("mdx_inst_hq3") == "2-stem model"

    def test_discard_removes_graph_and_weights(self, tmp_dir):
        graph = _write(tmp_dir, "htdemucs.onnx", 4)
        weights = _write(tmp_dir, "htdemucs.onnx.data", 4)
        model_manager.discard_model_files(graph)
        assert not os.path.exists(graph)
        assert not os.path.exists(weights)
        model_manager.discard_model_files(graph)  # already gone: no error

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

    def test_is_beat_model_downloaded_true(self, tmp_dir, tiny_sizes):
        manager = ModelManager(data_dir=tmp_dir)
        _write(tmp_dir, "beat_this.onnx", tiny_sizes["beat_this.onnx"])
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

    def test_mdx_asset_request_uses_id_url_and_octet_stream_accept(self, tmp_dir):
        from src.mdx_separator import MDX_MODELS

        body = b"mdx-model"
        manager = ModelManager(data_dir=tmp_dir)
        downloader = manager.download_mdx_model()
        downloader._expected_sha256 = hashlib.sha256(body).hexdigest()

        with patch(
            "src.model_manager.urllib.request.urlopen",
            return_value=_FakeResponse(body),
        ) as urlopen:
            downloader.run()

        request = urlopen.call_args.args[0]
        assert request.full_url == MDX_MODELS["mdx_inst_hq3"]["url"]
        assert request.get_header("Accept") == "application/octet-stream"

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

        # The user sees readable text; the checksums go to the log.
        assert errors == [MSG_DOWNLOAD_DAMAGED]
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
                   side_effect=ConnectionResetError("connection reset")):
            dl.run()

        assert errors == [MSG_RESET]
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


class TestDownloadProgress:
    """Progress is weighted by bytes and names the model, not files."""

    def _run(self, tmp_dir, sizes, cached=()):
        manager_dir = os.path.join(tmp_dir, "models")
        os.makedirs(manager_dir, exist_ok=True)
        bodies = {
            name: bytes([i + 1]) * sizes[name]
            for i, name in enumerate(_MODEL_FILES["htdemucs"])
        }
        for name in cached:
            with open(os.path.join(manager_dir, name), "wb") as f:
                f.write(bodies[name])
        hashes = {n: hashlib.sha256(b).hexdigest() for n, b in bodies.items()}

        def respond(request, **_kwargs):
            name = request.full_url.rsplit("/", 1)[-1]
            return _FakeResponse(bodies[name])

        dl = ModelDownloader("htdemucs", manager_dir)
        progress = []
        dl.progress.connect(lambda p, m: progress.append((p, m)))
        with (
            patch(
                "src.model_manager.urllib.request.urlopen",
                side_effect=respond,
            ),
            patch.dict("src.model_manager._MODEL_SHA256", hashes),
        ):
            dl.run()
        return progress

    def test_small_graph_file_is_a_small_share(self, tmp_dir, monkeypatch):
        sizes = {"htdemucs.onnx": 1 << 16, "htdemucs.onnx.data": 9 << 16}
        monkeypatch.setattr(model_manager, "MODEL_FILE_SIZES", sizes)
        percents = [p for p, _m in self._run(tmp_dir, sizes)]
        # The graph is 10% of the bytes, so the bar sits at 10% after it,
        # not at 50% as it did when each file counted as half.
        assert 10 in percents
        assert 50 not in percents[:3]
        assert percents == sorted(percents)
        assert percents[-1] == 100

    def test_messages_name_the_model_and_megabytes(self, tmp_dir, monkeypatch):
        sizes = {"htdemucs.onnx": 1 << 20, "htdemucs.onnx.data": 3 << 20}
        monkeypatch.setattr(model_manager, "MODEL_FILE_SIZES", sizes)
        messages = [m for _p, m in self._run(tmp_dir, sizes)]
        assert any("4-stem model" in m and "of 4 MB" in m for m in messages)
        assert not any(".onnx" in m for m in messages)

    def test_wrong_size_cached_file_is_downloaded_again(
        self, tmp_dir, monkeypatch,
    ):
        sizes = {"htdemucs.onnx": 1 << 16, "htdemucs.onnx.data": 2 << 16}
        monkeypatch.setattr(model_manager, "MODEL_FILE_SIZES", sizes)
        manager_dir = os.path.join(tmp_dir, "models")
        os.makedirs(manager_dir, exist_ok=True)
        stale = os.path.join(manager_dir, "htdemucs.onnx.data")
        with open(stale, "wb") as f:
            f.write(b"truncated")
        self._run(tmp_dir, sizes, cached=("htdemucs.onnx",))
        assert os.path.getsize(stale) == sizes["htdemucs.onnx.data"]

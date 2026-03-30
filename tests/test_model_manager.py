"""Tests for the model download and cache manager."""

import os

import pytest

from src.model_manager import ModelDownloader, ModelManager, _MODEL_FILES


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

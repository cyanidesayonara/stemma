"""Tests for scheduled slow-test model cache validation."""

import hashlib
import os

import pytest

import scripts.cache_slow_test_models as cache_script
import src.model_manager as model_manager
from src.mdx_separator import MDX_MODELS


@pytest.fixture
def cache_layout(tmp_path, monkeypatch):
    repository_data = tmp_path / "repository-data"
    user_data = tmp_path / "user-data"
    bodies = {
        "htdemucs.onnx": b"valid-htdemucs-graph",
        "htdemucs.onnx.data": b"valid-htdemucs-data",
        MDX_MODELS["mdx_inst_hq3"]["file"]: b"valid-mdx-model",
    }
    hashes = {
        name: hashlib.sha256(body).hexdigest()
        for name, body in bodies.items()
    }
    ht_hashes = {
        name: hashes[name]
        for name in model_manager._MODEL_FILES["htdemucs"]
    }
    monkeypatch.setattr(
        cache_script,
        "_MODEL_SHA256",
        ht_hashes,
        raising=False,
    )
    monkeypatch.setattr(model_manager, "_MODEL_SHA256", ht_hashes)
    monkeypatch.setitem(
        MDX_MODELS["mdx_inst_hq3"],
        "sha256",
        hashes[MDX_MODELS["mdx_inst_hq3"]["file"]],
    )
    paths = {
        "htdemucs.onnx": (
            repository_data / "models" / "htdemucs.onnx"
        ),
        "htdemucs.onnx.data": (
            repository_data / "models" / "htdemucs.onnx.data"
        ),
        MDX_MODELS["mdx_inst_hq3"]["file"]: (
            user_data / "models" / MDX_MODELS["mdx_inst_hq3"]["file"]
        ),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    return repository_data, user_data, bodies, paths


def test_valid_cached_models_are_preserved_and_partials_removed(
    cache_layout, monkeypatch
):
    repository_data, user_data, bodies, paths = cache_layout
    for name, path in paths.items():
        path.write_bytes(bodies[name])
        (path.parent / f"{path.name}.part").write_bytes(b"stale")
        (path.parent / f"{path.name}.partial").write_bytes(b"legacy")
    downloads = []

    def record_download(downloader):
        downloads.append(downloader.model_name)
        return downloader._file_name or "htdemucs.onnx"

    monkeypatch.setattr(cache_script, "_run_download", record_download)

    cache_script.cache_required_models(
        str(repository_data),
        str(user_data),
    )

    assert downloads == ["htdemucs", "mdx_inst_hq3"]
    for name, path in paths.items():
        assert path.read_bytes() == bodies[name]
        assert not os.path.exists(f"{path}.part")
        assert not os.path.exists(f"{path}.partial")


def test_hash_mismatch_is_removed_and_replaced(
    cache_layout, monkeypatch
):
    repository_data, user_data, bodies, paths = cache_layout
    paths["htdemucs.onnx"].write_bytes(bodies["htdemucs.onnx"])
    paths["htdemucs.onnx.data"].write_bytes(b"corrupt")
    mdx_name = MDX_MODELS["mdx_inst_hq3"]["file"]
    paths[mdx_name].write_bytes(b"corrupt")
    downloads = []

    def replace_missing(downloader):
        downloads.append(downloader.model_name)
        if downloader.model_name == "htdemucs":
            names = model_manager._MODEL_FILES["htdemucs"]
        else:
            names = (mdx_name,)
        for name in names:
            if not paths[name].exists():
                paths[name].write_bytes(bodies[name])
        return str(paths[names[0]])

    monkeypatch.setattr(cache_script, "_run_download", replace_missing)

    cache_script.cache_required_models(
        str(repository_data),
        str(user_data),
    )

    assert downloads == ["htdemucs", "mdx_inst_hq3"]
    for name, path in paths.items():
        assert path.read_bytes() == bodies[name]


def test_corrupt_cache_is_never_accepted(cache_layout, monkeypatch):
    repository_data, user_data, _bodies, paths = cache_layout
    for path in paths.values():
        path.write_bytes(b"corrupt")
    monkeypatch.setattr(
        cache_script,
        "_run_download",
        lambda downloader: downloader.model_name,
    )

    with pytest.raises(RuntimeError, match="integrity-verified"):
        cache_script.cache_required_models(
            str(repository_data),
            str(user_data),
        )

    assert all(not path.exists() for path in paths.values())

"""Populate the integrity-checked model cache used by scheduled slow tests."""

import hashlib
import os

from src.data_paths import platform_user_data_dir
from src.mdx_separator import MDX_MODELS
from src.model_manager import (
    ModelDownloader,
    ModelManager,
    _MODEL_FILES,
    _MODEL_SHA256,
)


def _run_download(downloader: ModelDownloader) -> str:
    errors: list[str] = []
    completed: list[str] = []
    downloader.error.connect(errors.append)
    downloader.download_complete.connect(completed.append)
    downloader.progress.connect(
        lambda percent, message: print(f"{percent:3d}% {message}", flush=True)
    )
    downloader.run()
    if errors:
        raise RuntimeError(errors[0])
    if len(completed) != 1:
        raise RuntimeError(
            f"Model download did not complete: {downloader.model_name}"
        )
    return completed[0]


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as model:
        for chunk in iter(lambda: model.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _remove_exact_file(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError(f"Could not remove invalid model cache file: {path}") from exc


def _purge_invalid_cache(path: str, expected_sha256: str) -> None:
    """Remove stale partials and any cached artifact with the wrong hash."""
    _remove_exact_file(f"{path}.part")
    _remove_exact_file(f"{path}.partial")
    if not os.path.exists(path):
        return
    if not os.path.isfile(path):
        raise RuntimeError(f"Model cache path is not a file: {path}")
    if _sha256_file(path) != expected_sha256:
        _remove_exact_file(path)


def _assert_integrity_verified(path: str, expected_sha256: str) -> None:
    if not os.path.isfile(path):
        raise RuntimeError(f"Model cache was not integrity-verified: {path}")
    actual = _sha256_file(path)
    if actual != expected_sha256:
        _remove_exact_file(path)
        raise RuntimeError(
            f"Model cache was not integrity-verified: {path} "
            f"(got {actual}, expected {expected_sha256})"
        )


def cache_required_models(
    repository_data_dir: str,
    user_data_dir: str,
) -> list[str]:
    """Validate and populate every model artifact used by slow tests."""
    repository_manager = ModelManager(data_dir=repository_data_dir)
    user_manager = ModelManager(data_dir=user_data_dir)
    artifacts = {
        os.path.join(repository_manager.models_dir, file_name): (
            _MODEL_SHA256[file_name]
        )
        for file_name in _MODEL_FILES["htdemucs"]
    }
    mdx_info = MDX_MODELS["mdx_inst_hq3"]
    artifacts[user_manager.mdx_model_path()] = mdx_info["sha256"]

    for path, expected_sha256 in artifacts.items():
        _purge_invalid_cache(path, expected_sha256)

    paths = [
        _run_download(repository_manager.download_model(is_6_stem=False)),
        _run_download(user_manager.download_mdx_model()),
    ]

    for path, expected_sha256 in artifacts.items():
        _assert_integrity_verified(path, expected_sha256)
    return paths


def main() -> int:
    """Cache the HTDemucs and MDX models exercised by real slow tests."""
    paths = cache_required_models(
        os.path.abspath("data"),
        platform_user_data_dir(),
    )
    for path in paths:
        print(f"Ready: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

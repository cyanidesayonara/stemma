"""Populate the integrity-checked model cache used by scheduled slow tests."""

import os

from src.data_paths import platform_user_data_dir
from src.model_manager import ModelDownloader, ModelManager


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


def main() -> int:
    """Cache the HTDemucs and MDX models exercised by real slow tests."""
    repository_manager = ModelManager(data_dir=os.path.abspath("data"))
    user_manager = ModelManager(data_dir=platform_user_data_dir())
    paths = [
        _run_download(repository_manager.download_model(is_6_stem=False)),
        _run_download(user_manager.download_mdx_model()),
    ]
    for path in paths:
        print(f"Ready: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

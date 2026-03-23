"""ONNX model download and cache management.

Manages the HTDemucs v4 ONNX model files that are required for stem
separation. Models are downloaded from HuggingFace on first run and
cached locally in the data/models/ directory.

Supported models:
    - htdemucs (4-stem): vocals, drums, bass, other (~80-300MB)
    - htdemucs_6s (6-stem): adds guitar + piano (~80-300MB)
"""

import os
import urllib.request

from PySide6.QtCore import QObject, QThread, Signal


# HuggingFace repository hosting the pre-converted ONNX models.
_REPO_URL = "https://huggingface.co/rysertio/Demucs-onnx/resolve/main"

# Model filename mapping.
_MODEL_FILES = {
    "htdemucs": "htdemucs.onnx",
    "htdemucs_6s": "htdemucs_6s.onnx",
}


class ModelDownloader(QThread):
    """Background thread for downloading an ONNX model file.

    Signals:
        progress(int, str): Download percentage (0-100) and status message.
        download_complete(str): Absolute path to the downloaded model file.
        error(str): Error description if download fails.

    ``download_complete`` is named to avoid shadowing ``QThread.finished``.
    """

    progress = Signal(int, str)
    download_complete = Signal(str)
    error = Signal(str)

    def __init__(self, model_name: str, models_dir: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.models_dir = models_dir
        self._is_cancelled = False

    def cancel(self) -> None:
        """Request cancellation of the active download."""
        self._is_cancelled = True

    def run(self) -> None:
        """Download the model file, emitting progress along the way."""
        try:
            self._download()
        except Exception as exc:
            # Clean up partial downloads on failure.
            dest = os.path.join(self.models_dir, _MODEL_FILES[self.model_name])
            if os.path.exists(dest):
                os.remove(dest)
            self.error.emit(str(exc))

    def _download(self) -> None:
        """Core download logic."""
        os.makedirs(self.models_dir, exist_ok=True)

        file_name = _MODEL_FILES[self.model_name]
        url = f"{_REPO_URL}/{file_name}"
        dest_path = os.path.join(self.models_dir, file_name)

        if os.path.exists(dest_path):
            self.progress.emit(100, f"{file_name} already cached.")
            self.download_complete.emit(dest_path)
            return

        self.progress.emit(0, f"Downloading {file_name}...")

        def _report_hook(block_num: int, block_size: int,
                         total_size: int) -> None:
            if self._is_cancelled:
                raise InterruptedError("Download cancelled by user.")
            if total_size > 0:
                percent = min(100, int(block_num * block_size * 100
                                       / total_size))
                self.progress.emit(
                    percent, f"Downloading {file_name}... {percent}%"
                )

        urllib.request.urlretrieve(url, dest_path, reporthook=_report_hook)

        self.progress.emit(100, "Download complete.")
        self.download_complete.emit(dest_path)


class ModelManager(QObject):
    """High-level interface for checking and downloading ONNX models.

    Usage:
        manager = ModelManager(data_dir="data")
        if not manager.is_model_downloaded(is_6_stem=False):
            downloader = manager.download_model(is_6_stem=False)
            downloader.progress.connect(on_progress)
            downloader.download_complete.connect(on_done)
            downloader.start()
    """

    def __init__(self, data_dir: str = "data") -> None:
        super().__init__()
        self.models_dir = os.path.join(data_dir, "models")
        self._active_downloader: ModelDownloader | None = None

    def model_path(self, is_6_stem: bool = False) -> str:
        """Return the expected local file path for the given model variant."""
        name = "htdemucs_6s" if is_6_stem else "htdemucs"
        return os.path.join(self.models_dir, _MODEL_FILES[name])

    def is_model_downloaded(self, is_6_stem: bool = False) -> bool:
        """Check whether the model file exists on disk."""
        return os.path.isfile(self.model_path(is_6_stem))

    def download_model(self, is_6_stem: bool = False) -> ModelDownloader:
        """Create and return a ModelDownloader thread (not yet started).

        The caller is responsible for connecting signals and calling start().
        """
        name = "htdemucs_6s" if is_6_stem else "htdemucs"
        self._active_downloader = ModelDownloader(name, self.models_dir)
        return self._active_downloader

"""ONNX model download and cache management.

Manages the HTDemucs v4 ONNX model files that are required for stem
separation and the beat_this model for beat/downbeat tracking. Models
are downloaded on first use and cached locally in the data/models/
directory.

Supported models:
    - htdemucs (4-stem): vocals, drums, bass, other (HuggingFace)
    - htdemucs_6s (6-stem): adds guitar + piano (HuggingFace)
    - beat_this: beat + downbeat detection (GitHub, MIT license)
"""

import os
import urllib.request

from PySide6.QtCore import QObject, QThread, Signal


# HuggingFace repository hosting the pre-converted ONNX models.
_REPO_URL = "https://huggingface.co/rysertio/Demucs-onnx/resolve/main"

# HuggingFace ships ONNX with external weights: small .onnx graph + large .onnx.data.
_MODEL_FILES = {
    "htdemucs": ("htdemucs.onnx", "htdemucs.onnx.data"),
    "htdemucs_6s": ("htdemucs_6s.onnx", "htdemucs_6s.onnx.data"),
}

# beat_this ONNX model for beat + downbeat tracking (ISMIR 2024, MIT license).
# Pre-exported by https://github.com/mosynthkey/beat_this_cpp
_BEAT_THIS_URL = (
    "https://github.com/mosynthkey/beat_this_cpp/raw/refs/heads/main/onnx/beat_this.onnx"
)
_BEAT_THIS_FILE = "beat_this.onnx"


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

    def __init__(
        self,
        model_name: str,
        models_dir: str,
        *,
        url: str | None = None,
        file_name: str | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.models_dir = models_dir
        self._is_cancelled = False
        self._url = url
        self._file_name = file_name

    def cancel(self) -> None:
        """Request cancellation of the active download."""
        self._is_cancelled = True

    def run(self) -> None:
        """Download the model file, emitting progress along the way."""
        try:
            self._download()
        except Exception as exc:
            dest = getattr(self, "_current_dest_path", None)
            if dest and os.path.exists(dest):
                os.remove(dest)
            self.error.emit(str(exc))

    def _download(self) -> None:
        """Core download logic."""
        os.makedirs(self.models_dir, exist_ok=True)

        # Single-file direct-URL mode (used for beat_this.onnx).
        if self._url and self._file_name:
            dest = os.path.join(self.models_dir, self._file_name)
            if os.path.exists(dest):
                self.progress.emit(100, f"{self._file_name} already cached.")
                self.download_complete.emit(dest)
                return
            self._current_dest_path = dest
            self.progress.emit(0, f"Downloading {self._file_name}...")

            def _hook(block: int, block_size: int, total: int) -> None:
                if self._is_cancelled:
                    raise InterruptedError("Download cancelled by user.")
                if total > 0:
                    pct = min(99, int(block * block_size * 100.0 / total))
                    self.progress.emit(pct, f"Downloading {self._file_name}... {pct}%")

            urllib.request.urlretrieve(self._url, dest, reporthook=_hook)
            self._current_dest_path = None
            self.progress.emit(100, "Download complete.")
            self.download_complete.emit(dest)
            return

        artifacts = _MODEL_FILES[self.model_name]
        n = len(artifacts)
        primary_path = os.path.join(self.models_dir, artifacts[0])

        for i, file_name in enumerate(artifacts):
            dest_path = os.path.join(self.models_dir, file_name)
            if os.path.exists(dest_path):
                self.progress.emit(
                    int((i + 1) / n * 100),
                    f"{file_name} already cached.",
                )
                continue

            url = f"{_REPO_URL}/{file_name}"
            self._current_dest_path = dest_path
            self.progress.emit(
                int(i / n * 100),
                f"Downloading {file_name}...",
            )

            def _make_hook(
                idx: int,
                name: str,
            ):
                def _report_hook(
                    block_num: int, block_size: int, total_size: int
                ) -> None:
                    if self._is_cancelled:
                        raise InterruptedError("Download cancelled by user.")
                    if total_size > 0:
                        file_pct = min(
                            100.0,
                            (block_num * block_size * 100.0) / total_size,
                        )
                        overall = int(((idx + file_pct / 100.0) / n) * 100)
                        overall = min(99, overall)
                        self.progress.emit(
                            overall,
                            f"Downloading {name}... {int(file_pct)}%",
                        )
                    else:
                        self.progress.emit(
                            int(idx / n * 100),
                            f"Downloading {name}...",
                        )

                return _report_hook

            urllib.request.urlretrieve(
                url, dest_path, reporthook=_make_hook(i, file_name)
            )
            self._current_dest_path = None

        self._current_dest_path = None
        self.progress.emit(100, "Download complete.")
        self.download_complete.emit(primary_path)


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
        """Return the expected local path to the ONNX graph (``.onnx``) file."""
        name = "htdemucs_6s" if is_6_stem else "htdemucs"
        return os.path.join(self.models_dir, _MODEL_FILES[name][0])

    def is_model_downloaded(self, is_6_stem: bool = False) -> bool:
        """Check whether all ONNX artifacts (graph + external data) exist."""
        name = "htdemucs_6s" if is_6_stem else "htdemucs"
        return all(
            os.path.isfile(os.path.join(self.models_dir, f))
            for f in _MODEL_FILES[name]
        )

    def download_model(self, is_6_stem: bool = False) -> ModelDownloader:
        """Create and return a ModelDownloader thread (not yet started).

        The caller is responsible for connecting signals and calling start().
        """
        name = "htdemucs_6s" if is_6_stem else "htdemucs"
        self._active_downloader = ModelDownloader(name, self.models_dir)
        return self._active_downloader

    def beat_model_path(self) -> str:
        """Return the expected local path to the beat_this ONNX model."""
        return os.path.join(self.models_dir, _BEAT_THIS_FILE)

    def is_beat_model_downloaded(self) -> bool:
        """Check whether the beat_this ONNX model exists on disk."""
        return os.path.isfile(self.beat_model_path())

    def download_beat_model(self) -> ModelDownloader:
        """Create a downloader for the beat_this ONNX model (not started)."""
        self._active_downloader = ModelDownloader(
            "beat_this", self.models_dir,
            url=_BEAT_THIS_URL, file_name=_BEAT_THIS_FILE,
        )
        return self._active_downloader

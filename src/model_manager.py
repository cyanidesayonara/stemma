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

# Connection/read timeout for model downloads, in seconds. A stalled TCP
# connection would otherwise hang the download thread forever -- the
# cancel flag is only checked between chunks, so a dead socket that
# never delivers another chunk can't be interrupted without a timeout.
_DOWNLOAD_TIMEOUT_S = 30

# Streaming read size. 64 KiB balances progress-update granularity
# against per-chunk Python overhead on ~180 MB model files.
_CHUNK_SIZE = 1 << 16


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
        expected_md5_tail: str | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.models_dir = models_dir
        self._is_cancelled = False
        self._url = url
        self._file_name = file_name
        # Optional integrity check: md5 of the file's last 10,000 KiB
        # (UVR's model-hashing convention). Verified after the download
        # completes; a mismatch removes the file and raises, so a swapped
        # or corrupted upstream file can never be treated as cached.
        self._expected_md5_tail = expected_md5_tail
        self._current_partial_path: str | None = None

    def cancel(self) -> None:
        """Request cancellation of the active download."""
        self._is_cancelled = True

    def run(self) -> None:
        """Download the model file, emitting progress along the way."""
        try:
            self._download()
        except Exception as exc:
            # Clean up the in-progress ``.partial`` file so a later run
            # doesn't resume from a corrupt prefix. Guard the removal:
            # if it fails (AV lock, permissions) we still want to surface
            # the original error rather than mask it with a second one.
            partial = getattr(self, "_current_partial_path", None)
            if partial and os.path.exists(partial):
                try:
                    os.remove(partial)
                except OSError:
                    pass
            self.error.emit(str(exc))

    def _verify_md5_tail(self, dest: str) -> None:
        """Verify the downloaded file against the expected tail hash.

        No-op when the downloader was created without an expected hash.
        On mismatch the file is removed and an OSError raised so the
        run() handler surfaces it via the error signal.
        """
        if not self._expected_md5_tail:
            return
        from src.mdx_separator import hash_model_file

        actual = hash_model_file(dest)
        if actual != self._expected_md5_tail:
            try:
                os.remove(dest)
            except OSError:
                pass
            raise OSError(
                f"Downloaded model failed integrity check "
                f"(md5 {actual}, expected {self._expected_md5_tail}). "
                "The upstream file may have changed; try again later."
            )

    def _download_file(self, url: str, dest: str, on_progress) -> None:
        """Download *url* to *dest* atomically.

        Streams the body into ``dest + '.partial'`` and renames it into
        place only after the whole response arrives and (when the server
        advertises a length) the byte count matches. A partial or stalled
        download therefore never leaves a file at the final path, so
        ``is_model_downloaded`` cannot mistake an interrupted transfer
        for a complete, usable model.

        *on_progress* is called as ``on_progress(downloaded, total)`` with
        byte counts (``total`` is 0 when the server sends no
        Content-Length).
        """
        partial = dest + ".partial"
        self._current_partial_path = partial
        # Drop any stale partial from a previously aborted attempt.
        if os.path.exists(partial):
            os.remove(partial)

        req = urllib.request.Request(url, headers={"User-Agent": "stemma"})
        with urllib.request.urlopen(
            req, timeout=_DOWNLOAD_TIMEOUT_S,
        ) as resp:
            total = int(resp.headers.get("Content-Length", 0) or 0)
            downloaded = 0
            with open(partial, "wb") as fh:
                while True:
                    if self._is_cancelled:
                        raise InterruptedError("Download cancelled by user.")
                    chunk = resp.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    on_progress(downloaded, total)

        if total > 0 and downloaded != total:
            raise OSError(
                f"Incomplete download: received {downloaded} of {total} "
                f"bytes for {os.path.basename(dest)}."
            )

        os.replace(partial, dest)
        self._current_partial_path = None

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
            self.progress.emit(0, f"Downloading {self._file_name}...")

            def _on_progress(downloaded: int, total: int) -> None:
                if total > 0:
                    pct = min(99, int(downloaded * 100 / total))
                    self.progress.emit(
                        pct, f"Downloading {self._file_name}... {pct}%",
                    )

            self._download_file(self._url, dest, _on_progress)
            self._verify_md5_tail(dest)
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
            self.progress.emit(
                int(i / n * 100),
                f"Downloading {file_name}...",
            )

            def _on_progress(
                downloaded: int, total: int, idx: int = i, name: str = file_name,
            ) -> None:
                if total > 0:
                    file_pct = min(100.0, downloaded * 100.0 / total)
                    overall = min(99, int(((idx + file_pct / 100.0) / n) * 100))
                    self.progress.emit(
                        overall, f"Downloading {name}... {int(file_pct)}%",
                    )
                else:
                    self.progress.emit(
                        int(idx / n * 100), f"Downloading {name}...",
                    )

            self._download_file(url, dest_path, _on_progress)

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

    def mdx_model_path(self, model_key: str = "mdx_inst_hq3") -> str:
        """Return the expected local path to an MDX-Net ONNX model."""
        from src.mdx_separator import MDX_MODELS

        return os.path.join(self.models_dir, MDX_MODELS[model_key]["file"])

    def is_mdx_model_downloaded(self, model_key: str = "mdx_inst_hq3") -> bool:
        """Check whether the MDX-Net model exists on disk."""
        return os.path.isfile(self.mdx_model_path(model_key))

    def download_mdx_model(
        self, model_key: str = "mdx_inst_hq3",
    ) -> ModelDownloader:
        """Create a downloader for an MDX-Net model (not started).

        The downloader verifies the file against UVR's published tail
        hash after the transfer.
        """
        from src.mdx_separator import MDX_MODELS

        info = MDX_MODELS[model_key]
        self._active_downloader = ModelDownloader(
            model_key, self.models_dir,
            url=info["url"], file_name=info["file"],
            expected_md5_tail=info["md5_tail"],
        )
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

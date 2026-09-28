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

import hashlib
import os
import urllib.request

from PySide6.QtCore import QObject, QThread, Signal

from src.import_messages import DownloadIntegrityError, describe_error

# Connection/read timeout for model downloads, in seconds. A stalled TCP
# connection would otherwise hang the download thread forever -- the
# cancel flag is only checked between chunks, so a dead socket that
# never delivers another chunk can't be interrupted without a timeout.
_DOWNLOAD_TIMEOUT_S = 30

# Streaming read size. 64 KiB balances progress-update granularity
# against per-chunk Python overhead on ~180 MB model files.
_CHUNK_SIZE = 1 << 16


# Immutable HuggingFace revision hosting the pre-converted ONNX models.
_REPO_REVISION = "ee08c547c91ef9f20ba19cf6ac2ed059ec9dcca0"
_REPO_URL = (
    "https://huggingface.co/rysertio/Demucs-onnx/resolve/"
    f"{_REPO_REVISION}"
)

# HuggingFace ships ONNX with external weights: small .onnx graph + large .onnx.data.
_MODEL_FILES = {
    "htdemucs": ("htdemucs.onnx", "htdemucs.onnx.data"),
    "htdemucs_6s": ("htdemucs_6s.onnx", "htdemucs_6s.onnx.data"),
}
# Exact byte sizes of every artifact, from the pinned upstream revisions.
# They feed the download consent prompt, byte-weighted progress, and the
# cache check: a file of the wrong size is treated as not downloaded.
MODEL_FILE_SIZES = {
    "htdemucs.onnx": 2_399_845,
    "htdemucs.onnx.data": 177_995_776,
    "htdemucs_6s.onnx": 2_390_444,
    "htdemucs_6s.onnx.data": 117_178_368,
    "UVR-MDX-NET-Inst_HQ_3.onnx": 66_759_214,
    "beat_this.onnx": 83_077_778,
}
# What the user sees instead of file names.
MODEL_LABELS = {
    "htdemucs": "4-stem model",
    "htdemucs_6s": "6-stem model",
    "mdx_inst_hq3": "2-stem model",
    "beat_this": "beat detection model",
}
_MODEL_SHA256 = {
    "htdemucs.onnx": (
        "be6fa125c457bc4fcdba43b0506270b5ed2113872748e8163de817f418db17bb"
    ),
    "htdemucs.onnx.data": (
        "e523708037d55151ac03feae48c9dbeab9908c086ed8e655e40b70dfaa66a3b8"
    ),
    "htdemucs_6s.onnx": (
        "cd881678a816731121d476c83305663a343b40b5e2c4e12b200ed220ba19808e"
    ),
    "htdemucs_6s.onnx.data": (
        "3eae380175adb9112c8ea8d1057307702dca09a82c2ede230897a0976c9a5461"
    ),
}

# beat_this ONNX model for beat + downbeat tracking (ISMIR 2024, MIT license).
# Pre-exported by https://github.com/mosynthkey/beat_this_cpp
_BEAT_THIS_REVISION = "07ab790a9ec2eda8093d52d249e3ec4f0510ee72"
_BEAT_THIS_URL = (
    "https://raw.githubusercontent.com/mosynthkey/beat_this_cpp/"
    f"{_BEAT_THIS_REVISION}/onnx/beat_this.onnx"
)
_BEAT_THIS_FILE = "beat_this.onnx"
_BEAT_THIS_SHA256 = (
    "c5c1466e08abdb03fdeb50668a06f244b787d564c212490482231a9cfbe9ccbd"
)


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
        expected_sha256: str | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.models_dir = models_dir
        self._is_cancelled = False
        self._url = url
        self._file_name = file_name
        self._expected_sha256 = expected_sha256
        self._current_partial_path: str | None = None

    def cancel(self) -> None:
        """Request cancellation of the active download."""
        self._is_cancelled = True

    def run(self) -> None:
        """Download the model file, emitting progress along the way."""
        try:
            self._download()
        except Exception as exc:
            # Clean up the in-progress ``.part`` file so a later run
            # doesn't resume from a corrupt prefix. Guard the removal:
            # if it fails (AV lock, permissions) we still want to surface
            # the original error rather than mask it with a second one.
            partial = getattr(self, "_current_partial_path", None)
            if partial and os.path.exists(partial):
                try:
                    os.remove(partial)
                except OSError:
                    pass
            self.error.emit(describe_error(exc, "Model download failed"))

    def _download_file(
        self,
        url: str,
        dest: str,
        on_progress,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        """Download *url* to *dest* atomically.

        Streams the body into ``dest + '.part'``, hashes it while writing,
        and renames it into place only after the byte count and expected
        SHA-256 match. A partial, corrupt, or stalled download therefore
        never reaches the final path.

        *on_progress* is called as ``on_progress(downloaded, total)`` with
        byte counts (``total`` is 0 when the server sends no
        Content-Length).
        """
        partial = dest + ".part"
        self._current_partial_path = partial
        # Drop any stale partial from a previously aborted attempt.
        if os.path.exists(partial):
            os.remove(partial)
        # Remove the scratch suffix used by releases before integrity
        # verification was introduced.
        legacy_partial = dest + ".partial"
        if os.path.exists(legacy_partial):
            os.remove(legacy_partial)

        hasher = hashlib.sha256()
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "stemma",
                    "Accept": "application/octet-stream",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            with urllib.request.urlopen(
                req,
                timeout=_DOWNLOAD_TIMEOUT_S,
            ) as resp:
                total = int(resp.headers.get("Content-Length", 0) or 0)
                downloaded = 0
                with open(partial, "wb") as fh:
                    while True:
                        if self._is_cancelled:
                            raise InterruptedError(
                                "Download cancelled by user."
                            )
                        chunk = resp.read(_CHUNK_SIZE)
                        if not chunk:
                            break
                        fh.write(chunk)
                        hasher.update(chunk)
                        downloaded += len(chunk)
                        on_progress(downloaded, total)

            if total > 0 and downloaded != total:
                raise DownloadIntegrityError(
                    f"Incomplete download: received {downloaded} of {total} "
                    f"bytes for {os.path.basename(dest)}."
                )

            expected = expected_sha256 or self._expected_sha256
            actual = hasher.hexdigest()
            if expected and actual != expected:
                raise DownloadIntegrityError(
                    "Downloaded model failed SHA-256 integrity check for "
                    f"{os.path.basename(dest)} (got {actual}, expected "
                    f"{expected}). The upstream file may have changed; "
                    "try again later."
                )

            os.replace(partial, dest)
            self._current_partial_path = None
        except Exception:
            if os.path.exists(partial):
                try:
                    os.remove(partial)
                except OSError:
                    pass
            self._current_partial_path = None
            raise

    def _plan(self) -> list[tuple[str, str, str | None, int]]:
        """Return ``(file_name, url, sha256, size)`` for each artifact."""
        if self._url and self._file_name:
            return [(
                self._file_name, self._url, self._expected_sha256,
                MODEL_FILE_SIZES.get(self._file_name, 0),
            )]
        return [
            (
                name, f"{_REPO_URL}/{name}", _MODEL_SHA256[name],
                MODEL_FILE_SIZES.get(name, 0),
            )
            for name in _MODEL_FILES[self.model_name]
        ]

    def _download(self) -> None:
        """Download every missing artifact, reporting progress by bytes."""
        os.makedirs(self.models_dir, exist_ok=True)
        plan = self._plan()
        primary_path = os.path.join(self.models_dir, plan[0][0])
        label = model_label(self.model_name)

        pending = []
        for name, url, sha, size in plan:
            dest = os.path.join(self.models_dir, name)
            if _is_complete(dest, size):
                continue
            if os.path.exists(dest):
                # A truncated or stale copy would never load; replace it.
                os.remove(dest)
            pending.append((dest, url, sha, size))

        if not pending:
            self.progress.emit(100, f"The {label} is already downloaded.")
            self.download_complete.emit(primary_path)
            return

        # Weight progress by bytes so the tiny graph file does not count
        # as half of a two-file model.
        known_total = sum(size for _dest, _url, _sha, size in pending)
        done_before = 0
        self.progress.emit(0, f"Downloading the {label}...")

        for dest, url, sha, size in pending:
            def _on_progress(
                downloaded: int, total: int, before: int = done_before,
            ) -> None:
                overall_total = known_total or total
                if overall_total <= 0:
                    self.progress.emit(0, f"Downloading the {label}...")
                    return
                done = before + downloaded
                pct = min(99, int(done * 100 / overall_total))
                self.progress.emit(
                    pct,
                    f"Downloading the {label}... "
                    f"{_mb(done)} of {_mb(overall_total)} MB",
                )

            self._download_file(url, dest, _on_progress, expected_sha256=sha)
            done_before += size or os.path.getsize(dest)

        self.progress.emit(100, "Download complete.")
        self.download_complete.emit(primary_path)


def _mb(num_bytes: int) -> int:
    """Whole megabytes (MiB), rounded, for progress and consent text."""
    return int(round(num_bytes / (1024 * 1024)))


def _is_complete(path: str, expected_size: int) -> bool:
    """True if *path* exists and, when its size is known, matches it."""
    try:
        actual = os.path.getsize(path)
    except OSError:
        return False
    return expected_size <= 0 or actual == expected_size


def model_label(model_key: str) -> str:
    """Return the user-facing name of a model, such as "4-stem model"."""
    return MODEL_LABELS.get(model_key, "separation model")


def discard_model_files(model_path: str) -> None:
    """Delete a damaged model and its external weights, if present.

    HTDemucs keeps its weights next to the graph as ``<name>.data``; both
    go so the next import downloads a fresh, verified copy.
    """
    for path in (model_path, model_path + ".data"):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


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

    def _files_for(self, model_key: str) -> tuple[str, ...]:
        """Return the artifact file names that make up *model_key*."""
        if model_key in _MODEL_FILES:
            return _MODEL_FILES[model_key]
        if model_key == "beat_this":
            return (_BEAT_THIS_FILE,)
        from src.mdx_separator import MDX_MODELS

        return (MDX_MODELS[model_key]["file"],)

    def _is_key_downloaded(self, model_key: str) -> bool:
        return all(
            _is_complete(
                os.path.join(self.models_dir, name),
                MODEL_FILE_SIZES.get(name, 0),
            )
            for name in self._files_for(model_key)
        )

    def download_size_bytes(self, model_key: str) -> int:
        """Bytes still to download for *model_key* (0 when cached)."""
        return sum(
            MODEL_FILE_SIZES.get(name, 0)
            for name in self._files_for(model_key)
            if not _is_complete(
                os.path.join(self.models_dir, name),
                MODEL_FILE_SIZES.get(name, 0),
            )
        )

    def download_size_mb(self, model_key: str) -> int:
        """Megabytes still to download for *model_key*, rounded."""
        return _mb(self.download_size_bytes(model_key))

    def model_path(self, is_6_stem: bool = False) -> str:
        """Return the expected local path to the ONNX graph (``.onnx``) file."""
        name = "htdemucs_6s" if is_6_stem else "htdemucs"
        return os.path.join(self.models_dir, _MODEL_FILES[name][0])

    def is_model_downloaded(self, is_6_stem: bool = False) -> bool:
        """Check that every ONNX artifact exists with its expected size."""
        return self._is_key_downloaded("htdemucs_6s" if is_6_stem else "htdemucs")

    def download_model(self, is_6_stem: bool = False) -> ModelDownloader:
        """Create and return a ModelDownloader thread (not yet started).

        The caller is responsible for connecting signals and calling start().
        """
        name = "htdemucs_6s" if is_6_stem else "htdemucs"
        self._active_downloader = ModelDownloader(name, self.models_dir)
        return self._active_downloader

    def mdx_model_path(self, model_key: str = "mdx_inst_hq3") -> str:
        """Return the expected local path to an MDX-Net ONNX model."""
        return os.path.join(self.models_dir, self._files_for(model_key)[0])

    def is_mdx_model_downloaded(self, model_key: str = "mdx_inst_hq3") -> bool:
        """Check that the MDX-Net model exists with its expected size."""
        return self._is_key_downloaded(model_key)

    def download_mdx_model(
        self, model_key: str = "mdx_inst_hq3",
    ) -> ModelDownloader:
        """Create a downloader for an MDX-Net model (not started).

        The downloader verifies the file against the reviewed SHA-256
        before atomically publishing it.
        """
        from src.mdx_separator import MDX_MODELS

        info = MDX_MODELS[model_key]
        self._active_downloader = ModelDownloader(
            model_key, self.models_dir,
            url=info["url"], file_name=info["file"],
            expected_sha256=info["sha256"],
        )
        return self._active_downloader

    def beat_model_path(self) -> str:
        """Return the expected local path to the beat_this ONNX model."""
        return os.path.join(self.models_dir, _BEAT_THIS_FILE)

    def is_beat_model_downloaded(self) -> bool:
        """Check that the beat_this model exists with its expected size."""
        return self._is_key_downloaded("beat_this")

    def download_beat_model(self) -> ModelDownloader:
        """Create a downloader for the beat_this ONNX model (not started)."""
        self._active_downloader = ModelDownloader(
            "beat_this", self.models_dir,
            url=_BEAT_THIS_URL, file_name=_BEAT_THIS_FILE,
            expected_sha256=_BEAT_THIS_SHA256,
        )
        return self._active_downloader

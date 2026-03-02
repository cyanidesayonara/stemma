"""YouTube audio downloader using yt-dlp.

Downloads the audio track from a YouTube URL and saves it as an audio file
for subsequent import into the stemma library.
"""

import os
import re
import shutil
from typing import Callable

import imageio_ffmpeg
import yt_dlp


class DownloadError(Exception):
    """Raised when a download or metadata extraction fails."""


_YOUTUBE_PATTERN = re.compile(
    r"^(https?://)?(www\.)?"
    r"(youtube\.com/watch\?v=|youtu\.be/|music\.youtube\.com/watch\?v=)"
)


def _get_ffmpeg_exe() -> str | None:
    """Return a path to an ffmpeg executable, or None if unavailable.

    Prefers the binary bundled with imageio-ffmpeg so the app works
    without the user installing ffmpeg separately. Falls back to
    whatever is on PATH.
    """
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError:
        return shutil.which("ffmpeg")


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available (bundled or on PATH)."""
    return _get_ffmpeg_exe() is not None


def is_supported_url(text: str) -> bool:
    """Return True if *text* looks like a supported YouTube URL."""
    return bool(_YOUTUBE_PATTERN.search(text))


def extract_metadata(url: str) -> tuple[str, str]:
    """Extract title and artist from a YouTube URL without downloading.

    Returns:
        A ``(title, artist)`` tuple. Falls back to ``"Untitled"`` and
        ``"Unknown Artist"`` when metadata is unavailable.

    Raises:
        DownloadError: If the metadata extraction fails.
    """
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "skip_download": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise DownloadError(str(exc)) from exc

    if info is None:
        raise DownloadError("Could not retrieve video metadata")

    title = info.get("title") or "Untitled"
    artist = info.get("artist") or info.get("uploader") or "Unknown Artist"
    return title, artist


def download_audio(
    url: str,
    output_path: str,
    progress_callback: Callable[[dict], None] | None = None,
) -> str:
    """Download the audio from *url* and save it to *output_path*.

    Args:
        url: YouTube video URL.
        output_path: Destination file path (e.g. ``/tmp/audio.mp3``).
        progress_callback: Optional callable invoked with yt-dlp progress
            dictionaries (keys: ``status``, ``downloaded_bytes``,
            ``total_bytes``, etc.).

    Returns:
        The *output_path* on success.

    Raises:
        DownloadError: If the download fails.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Strip extension from outtmpl because FFmpegExtractAudio appends the
    # codec extension itself (e.g. ".mp3"). Passing "audio.mp3" would
    # produce "audio.mp3.mp3".
    stem = os.path.splitext(output_path)[0]

    opts = {
        "format": "bestaudio/best",
        "outtmpl": stem,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "320",
            }
        ],
        "progress_hooks": [],
    }

    ffmpeg_exe = _get_ffmpeg_exe()
    if ffmpeg_exe:
        opts["ffmpeg_location"] = ffmpeg_exe

    if progress_callback is not None:
        opts["progress_hooks"].append(progress_callback)

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        raise DownloadError(str(exc)) from exc

    if not os.path.isfile(output_path):
        raise DownloadError(
            f"Download finished but output file not found: {output_path}"
        )

    return output_path

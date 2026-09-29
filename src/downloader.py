"""YouTube audio downloader using yt-dlp.

Downloads the audio track from a YouTube URL and saves it as an audio file
for subsequent import into the stemma library.
"""

import os
import re
import shutil
import sys
import time
from typing import Callable

import imageio_ffmpeg
import yt_dlp

from src.crash_log import logger
from src.import_messages import strip_ansi


class DownloadError(Exception):
    """Raised when a download or metadata extraction fails."""


class DownloadCancelled(DownloadError):
    """Raised when the caller cancelled before a retry could start."""


# YouTube often refuses the first media URL it hands out (HTTP 403) while a
# fresh extraction a moment later works, so a download is tried up to this
# many times, waiting _BACKOFF_S[n] seconds before retry n + 1.
_MAX_ATTEMPTS = 3
_BACKOFF_S = (1.5, 3.0)
_CANCEL_POLL_S = 0.1
# Indirection so tests can record the backoff instead of sleeping.
_sleep = time.sleep

# Worth a fresh extraction: a refused or expired media URL, a server
# hiccup, or a connection dropped mid-stream. Windows words socket
# failures its own way ("forcibly closed", WinError 10053/10054/10060).
_TRANSIENT = re.compile(
    r"http error (?:403|408|500|502|503|504)\b"
    r"|timed out|connection reset|connection (?:was )?aborted"
    r"|forcibly closed|did not properly respond"
    r"|winerror 100(?:53|54|60)\b"
    r"|incomplete ?read|remote end closed",
    re.IGNORECASE,
)
# Never worth retrying, even when a 403 appears alongside: the video is
# private, removed, blocked, gated behind a sign-in, or not a video.
_PERMANENT = re.compile(
    r"private video|video is private|video (?:is )?unavailable"
    r"|has been removed"
    r"|no longer available|not available|unsupported url|sign in"
    r"|not a bot|confirm your age|age-restricted|members-only|copyright"
    r"|premieres in|live event",
    re.IGNORECASE,
)


def is_transient_error(message: str) -> bool:
    """Return True if a yt-dlp failure *message* may pass on a fresh try."""
    text = strip_ansi(message)
    return bool(_TRANSIENT.search(text)) and not _PERMANENT.search(text)


def _clean_message(exc: BaseException) -> str:
    """A yt-dlp failure's text without colour codes, for log and UI."""
    return strip_ansi(str(exc)).strip()


def _remove_leftovers(stem: str) -> None:
    """Delete what a failed attempt left at *stem* (``.part``, ``.ytdl``).

    A retry may pick a different format; resuming the old partial file
    would append its bytes to the wrong stream.
    """
    folder = os.path.dirname(stem) or "."
    prefix = os.path.basename(stem)
    try:
        names = os.listdir(folder)
    except OSError:
        return
    for name in names:
        if name == prefix or name.startswith(prefix + "."):
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                logger.warning("could not remove download leftover %s", name)


def _wait_unless_cancelled(
    seconds: float, should_cancel: Callable[[], bool] | None,
) -> bool:
    """Sleep up to *seconds*; return False early if cancelled."""
    remaining = seconds
    while remaining > 0:
        if should_cancel is not None and should_cancel():
            return False
        step = min(_CANCEL_POLL_S, remaining)
        _sleep(step)
        remaining -= step
    return not (should_cancel is not None and should_cancel())


_YOUTUBE_PATTERN = re.compile(
    r"^(https?://)?"
    # Optional youtube subdomains (www/m/music), in any combination, so
    # m.youtube.com and www.music.youtube.com both parse. The chain is
    # restricted to these labels so lookalikes (fakeyoutube.com) are
    # still rejected.
    r"(?:(?:www|m|music)\.)*"
    r"(youtube\.com/(watch\?v=|shorts/|live/|embed/)|youtu\.be/)"
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


def ffmpeg_missing_message() -> str:
    """Explain a missing ffmpeg in terms that fit how stemma was installed.

    Installed builds (Store, installer) ship ffmpeg inside the app, so
    telling their users to edit PATH is wrong: the fix is a reinstall.
    """
    if getattr(sys, "frozen", False):
        return (
            "YouTube import needs ffmpeg, which comes with stemma, but it "
            "could not be found. Reinstall stemma to restore it."
        )
    return (
        "YouTube import needs ffmpeg. Install the imageio-ffmpeg package "
        "or put ffmpeg on your PATH."
    )


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
        "no_color": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise DownloadError(_clean_message(exc)) from exc

    if info is None:
        raise DownloadError("Could not retrieve video metadata")

    title = info.get("title") or "Untitled"
    artist = info.get("artist") or info.get("uploader") or "Unknown Artist"
    return title, artist


def _download_options(
    stem: str, progress_callback: Callable[[dict], None] | None,
) -> dict:
    """Fresh yt-dlp options for one download attempt."""
    opts = {
        "format": "bestaudio/best",
        "outtmpl": stem,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Plain text: the message ends up in the log and the import dialog.
        "no_color": True,
        # Never resume a failed attempt's partial file: a retry may pick
        # a different format (leftovers are also removed before retrying).
        "continuedl": False,
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
    return opts


def download_audio(
    url: str,
    output_path: str,
    progress_callback: Callable[[dict], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_retry: Callable[[int, float], None] | None = None,
) -> str:
    """Download the audio from *url* and save it to *output_path*.

    A transient failure (see ``is_transient_error``), typically YouTube
    refusing the first media URL with HTTP 403, is retried with a fresh
    extraction after a short backoff, up to ``_MAX_ATTEMPTS`` in all.
    Permanent failures (private or removed video, unsupported URL) are
    raised at once.

    Args:
        url: YouTube video URL.
        output_path: Destination file path (e.g. ``/tmp/audio.mp3``).
        progress_callback: Optional callable invoked with yt-dlp progress
            dictionaries (keys: ``status``, ``downloaded_bytes``,
            ``total_bytes``, etc.). It receives every attempt's progress.
        should_cancel: Optional callable polled before each retry and
            during the backoff; when it returns True, no further attempt
            is made.
        on_retry: Optional callable invoked with the number of the next
            attempt and the backoff in seconds, before each retry waits.

    Returns:
        The *output_path* on success.

    Raises:
        DownloadError: If the download fails.
        DownloadCancelled: If *should_cancel* stopped a retry.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Strip extension from outtmpl because FFmpegExtractAudio appends the
    # codec extension itself (e.g. ".mp3"). Passing "audio.mp3" would
    # produce "audio.mp3.mp3".
    stem = os.path.splitext(output_path)[0]

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            # A new YoutubeDL per attempt re-extracts the video, so a
            # retry gets a fresh media URL instead of the refused one.
            with yt_dlp.YoutubeDL(
                _download_options(stem, progress_callback)
            ) as ydl:
                ydl.download([url])
            break
        except Exception as exc:
            message = _clean_message(exc)
            if attempt == _MAX_ATTEMPTS or not is_transient_error(message):
                raise DownloadError(message) from exc
            delay = _BACKOFF_S[min(attempt, len(_BACKOFF_S)) - 1]
            logger.info(
                "YouTube download attempt %d of %d failed (%s); "
                "retrying in %.1f s",
                attempt, _MAX_ATTEMPTS, message, delay,
            )
            if on_retry is not None:
                on_retry(attempt + 1, delay)
            if not _wait_unless_cancelled(delay, should_cancel):
                raise DownloadCancelled(
                    f"Download cancelled after: {message}"
                ) from exc
            _remove_leftovers(stem)

    if not os.path.isfile(output_path):
        raise DownloadError(
            f"Download finished but output file not found: {output_path}"
        )

    return output_path

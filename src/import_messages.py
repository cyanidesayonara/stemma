"""User-facing text for import, download, separation, and export failures.

Two entry points:

- ``describe_error(exc, context)`` is called where an exception is caught.
  It maps the exception by type and Windows/POSIX error code (walking the
  ``__cause__`` chain, ``URLError.reason``, and yt-dlp's ``exc_info``),
  logs the raw text and traceback to the stemma log, and returns short
  readable text for the UI.
- ``format_import_error(message)`` is the pure string fallback for places
  that only receive a message. It understands the same error codes when
  they appear in text (``[WinError 10061]``, ``[Errno 11001]``), plus the
  yt-dlp messages that have no exception type of their own.

Both return the readable messages below unchanged, so a worker can map an
exception once and the dialog that shows the result can format it again
safely.
"""

from __future__ import annotations

import errno
import re
import socket
import ssl
import urllib.error

import soundfile as sf

from src.crash_log import logger


class DownloadIntegrityError(OSError):
    """A downloaded file was incomplete or failed its checksum."""


class ModelDamagedError(RuntimeError):
    """A cached model file exists but ONNX Runtime cannot load it."""


MSG_GENERIC = "Something went wrong. Try again."
MSG_CANCELLED = "The operation was cancelled."
MSG_OFFLINE = (
    "Can't connect to the internet. Check your connection and try again."
)
MSG_TIMEOUT = "The request timed out. Check your network and try again."
MSG_REFUSED = "The server refused the connection. Try again later."
MSG_RESET = (
    "The connection was interrupted. Check your connection and try again."
)
MSG_SSL = "Secure connection failed. Check your network or system date."
MSG_DISK_FULL = (
    "Not enough disk space to finish. Free some space and try again."
)
MSG_PERMISSION = "Permission denied. Check that the folder is writable."
MSG_IN_USE = (
    "The file is in use by another program. Close it and try again."
)
MSG_PATH_TOO_LONG = (
    "The file path is too long. Move the file to a folder with a shorter "
    "path and try again."
)
MSG_NOT_FOUND = "File not found. It may have been moved or deleted."
MSG_UNREADABLE_AUDIO = (
    "stemma can't read this audio file. It may be damaged or in an "
    "unsupported format. Supported: MP3, WAV, FLAC."
)
# libsndfile reports a file it cannot open for writing (read-only, or held
# by another program) as "System error", the same exception type it uses
# for unreadable audio.
MSG_CANNOT_OPEN_FILE = (
    "stemma couldn't open the file. Check that it isn't open in another "
    "program and that you can write to its folder."
)
MSG_DOWNLOAD_DAMAGED = (
    "The download was incomplete or damaged. Try again."
)
MSG_HTTP_404 = "Download failed: file not found on the server."
MSG_HTTP = "Download failed: the server returned an error. Try again later."
MSG_MEMORY = (
    "Not enough memory for stem separation. "
    "Close other apps or try a shorter audio file."
)
MSG_ONNX_INIT = (
    "Stem separation failed to initialize. "
    "Try closing other apps to free memory, then retry."
)
MSG_YT_PRIVATE = "This video is private, so stemma can't download it."
MSG_YT_AGE = "This video is age-restricted, so stemma can't download it."
MSG_YT_UNAVAILABLE = (
    "This video is unavailable. It may have been removed or blocked "
    "in your region."
)
MSG_YT_BOT_CHECK = (
    "YouTube is asking to confirm you're not a bot. "
    "Wait a while and try again."
)
MSG_YT_LIVE = "Live streams can't be imported. Try again after it ends."
MSG_YT_UPCOMING = "This video hasn't premiered yet. Try again after it does."
MSG_YT_FORMAT = (
    "YouTube changed how this video is served. Try again later, or update "
    "stemma."
)
MSG_YT_ONLY = "Only YouTube links are supported."
MSG_MODEL_DAMAGED = (
    "The separation model on this PC was damaged, so stemma removed it. "
    "Import the song again to download a fresh copy."
)

_READABLE = frozenset(
    value for name, value in globals().items()
    if name.startswith("MSG_") and isinstance(value, str)
)

# Windows error codes (OSError.winerror, and the WSA socket codes that
# also appear as errno on Windows) mapped to readable text.
_WINERROR = {
    3: MSG_NOT_FOUND,          # ERROR_PATH_NOT_FOUND
    2: MSG_NOT_FOUND,          # ERROR_FILE_NOT_FOUND
    5: MSG_PERMISSION,         # ERROR_ACCESS_DENIED
    32: MSG_IN_USE,            # ERROR_SHARING_VIOLATION
    33: MSG_IN_USE,            # ERROR_LOCK_VIOLATION
    39: MSG_DISK_FULL,         # ERROR_HANDLE_DISK_FULL
    112: MSG_DISK_FULL,        # ERROR_DISK_FULL
    206: MSG_PATH_TOO_LONG,    # ERROR_FILENAME_EXCED_RANGE
    10050: MSG_OFFLINE,        # WSAENETDOWN
    10051: MSG_OFFLINE,        # WSAENETUNREACH
    10053: MSG_RESET,          # WSAECONNABORTED
    10054: MSG_RESET,          # WSAECONNRESET
    10060: MSG_TIMEOUT,        # WSAETIMEDOUT
    10061: MSG_REFUSED,        # WSAECONNREFUSED
    10065: MSG_OFFLINE,        # WSAEHOSTUNREACH
    11001: MSG_OFFLINE,        # WSAHOST_NOT_FOUND (getaddrinfo failed)
    11002: MSG_OFFLINE,        # WSATRY_AGAIN
    11004: MSG_OFFLINE,        # WSANO_DATA
}

_ERRNO = {
    errno.ENOSPC: MSG_DISK_FULL,
    errno.EACCES: MSG_PERMISSION,
    errno.EPERM: MSG_PERMISSION,
    errno.ENAMETOOLONG: MSG_PATH_TOO_LONG,
    errno.ENOENT: MSG_NOT_FOUND,
    errno.ECONNREFUSED: MSG_REFUSED,
    errno.ECONNRESET: MSG_RESET,
    errno.ECONNABORTED: MSG_RESET,
    errno.ETIMEDOUT: MSG_TIMEOUT,
    errno.ENETUNREACH: MSG_OFFLINE,
    errno.EHOSTUNREACH: MSG_OFFLINE,
}

_CODE_IN_TEXT = re.compile(r"\[(winerror|errno)\s+(-?\d+)\]", re.IGNORECASE)

_OOM_PHRASES = (
    "out of memory",
    "ran out of memory",
    "bad_alloc",
    "bad alloc",
    "failed to allocate",
)


def _code_message(winerror: int | None, err: int | None) -> str | None:
    """Map a winerror or errno code to readable text, if known."""
    if winerror is not None and winerror in _WINERROR:
        return _WINERROR[winerror]
    if err is not None:
        if err in _ERRNO:
            return _ERRNO[err]
        # Windows reports socket failures as errno with a WSA code.
        if err >= 10000 and err in _WINERROR:
            return _WINERROR[err]
    return None


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """Return *exc* and the exceptions it wraps, outermost first."""
    chain: list[BaseException] = []
    pending: list[BaseException | None] = [exc]
    while pending and len(chain) < 12:
        current = pending.pop(0)
        if current is None or any(current is seen for seen in chain):
            continue
        chain.append(current)
        # urllib wraps socket errors in URLError.reason.
        reason = getattr(current, "reason", None)
        if isinstance(reason, BaseException):
            pending.append(reason)
        # yt-dlp's DownloadError keeps the original in exc_info.
        exc_info = getattr(current, "exc_info", None)
        if isinstance(exc_info, tuple) and len(exc_info) >= 2:
            if isinstance(exc_info[1], BaseException):
                pending.append(exc_info[1])
        # Explicit causes only: an unrelated error raised while another was
        # being handled (__context__) must not take on that one's meaning.
        pending.append(current.__cause__)
    return chain


def _typed_message(exc: BaseException) -> str | None:
    """Map one exception by its type and error codes."""
    if isinstance(exc, InterruptedError):
        return MSG_CANCELLED
    if isinstance(exc, MemoryError):
        return MSG_MEMORY
    if isinstance(exc, DownloadIntegrityError):
        return MSG_DOWNLOAD_DAMAGED
    if isinstance(exc, ModelDamagedError):
        return MSG_MODEL_DAMAGED
    if isinstance(exc, urllib.error.HTTPError):
        return MSG_HTTP_404 if exc.code == 404 else MSG_HTTP
    if isinstance(exc, sf.SoundFileError):
        if "system error" in str(exc).lower():
            return MSG_CANNOT_OPEN_FILE
        return MSG_UNREADABLE_AUDIO
    if isinstance(exc, socket.gaierror):
        return MSG_OFFLINE
    if isinstance(exc, ssl.SSLError):
        return MSG_SSL
    if isinstance(exc, OSError):
        by_code = _code_message(getattr(exc, "winerror", None), exc.errno)
        if by_code is not None:
            return by_code
        if isinstance(exc, TimeoutError):
            return MSG_TIMEOUT
        if isinstance(exc, ConnectionRefusedError):
            return MSG_REFUSED
        if isinstance(exc, ConnectionError):
            return MSG_RESET
        if isinstance(exc, PermissionError):
            return MSG_PERMISSION
        if isinstance(exc, FileNotFoundError):
            return MSG_NOT_FOUND
    return None


def _youtube_message(low: str) -> str | None:
    """Map yt-dlp's text-only failures."""
    if "youtube" not in low and "video" not in low:
        return None
    if "private video" in low or "video is private" in low:
        return MSG_YT_PRIVATE
    if "confirm your age" in low or "age-restricted" in low:
        return MSG_YT_AGE
    if (
        "not a bot" in low
        or "cookies-from-browser" in low
        or "sign in to confirm" in low
    ):
        return MSG_YT_BOT_CHECK
    if "premieres in" in low or "premiere will begin" in low:
        return MSG_YT_UPCOMING
    if "live event" in low or "is live" in low:
        return MSG_YT_LIVE
    if "requested format is not available" in low:
        return MSG_YT_FORMAT
    if (
        "video unavailable" in low
        or "has been removed" in low
        or "no longer available" in low
        or "account associated with this video has been terminated" in low
        or "not available in your country" in low
        or "not made this video available in your country" in low
        or "is not available" in low
    ):
        return MSG_YT_UNAVAILABLE
    return None


def format_import_error(message: str, max_len: int = 400) -> str:
    """Turn a raw exception or library message into short, readable text.

    Pure: it never logs. Use ``describe_error`` where the exception object
    is available.
    """
    raw = (message or "").strip()
    if not raw:
        return MSG_GENERIC
    if raw in _READABLE:
        return raw

    low = raw.lower()

    youtube = _youtube_message(low)
    if youtube is not None:
        return youtube

    for kind, code_text in _CODE_IN_TEXT.findall(raw):
        code = int(code_text)
        if kind.lower() == "winerror":
            found = _code_message(code, None)
        else:
            found = _code_message(None, code)
        if found is not None:
            return found

    if "getaddrinfo failed" in low or "name or service not known" in low:
        return MSG_OFFLINE
    if "network is unreachable" in low:
        return MSG_OFFLINE
    if "disk full" in low or "no space left" in low:
        return MSG_DISK_FULL
    if "permission denied" in low:
        return MSG_PERMISSION
    if "timed out" in low or "timeout" in low:
        return MSG_TIMEOUT
    if "connection refused" in low:
        return MSG_REFUSED
    if "connection reset" in low or "connection aborted" in low:
        return MSG_RESET
    if "ssl" in low or "certificate" in low:
        return MSG_SSL
    if "interruptederror" in low or "cancelled" in low:
        return MSG_CANCELLED
    if "onnxruntimeerror" in low or "runtime_exception" in low:
        if any(p in low for p in _OOM_PHRASES):
            return MSG_MEMORY
        return MSG_ONNX_INIT
    if any(p in low for p in _OOM_PHRASES):
        return MSG_MEMORY
    if "http error 404" in low or ("404" in low and "not found" in low):
        return MSG_HTTP_404
    if "http error" in low:
        return MSG_HTTP
    if "possibly a pipe" in low or "format not recognised" in low:
        return MSG_UNREADABLE_AUDIO

    if len(raw) > max_len:
        return raw[:max_len].rstrip() + "..."
    return raw


def describe_error(
    error: BaseException | str,
    context: str = "Operation failed",
    max_len: int = 400,
) -> str:
    """Return readable text for *error* and log its raw detail.

    *error* is normally the caught exception; the raw message and
    traceback go to the stemma log under *context*, and the user sees
    the mapped text instead. A string is formatted with
    ``format_import_error`` and logged unless it is already readable.
    """
    if isinstance(error, BaseException):
        logger.warning(
            "%s: %s: %s", context, type(error).__name__, error,
            exc_info=(type(error), error, error.__traceback__),
        )
        for exc in _exception_chain(error):
            found = _typed_message(exc)
            if found is not None:
                return found
        return format_import_error(str(error), max_len=max_len)

    text = (error or "").strip()
    if text and text not in _READABLE:
        logger.warning("%s: %s", context, text)
    return format_import_error(text, max_len=max_len)


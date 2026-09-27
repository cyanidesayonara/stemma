"""Crash and error log for stemma.

The frozen build has no console, so an exception that escapes (at startup,
in a Qt slot, or on a worker thread) used to vanish. ``install`` routes
every uncaught exception into a small rotating log file under the per-user
data folder, falling back to the temp folder when that is not writable.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import tempfile
import threading

from src.data_paths import platform_user_data_dir

LOG_NAME = "stemma.log"
_MAX_BYTES = 512 * 1024
_BACKUPS = 2

logger = logging.getLogger("stemma")
_installed_path: str | None = None


def _candidate_dirs() -> list[str]:
    return [
        os.path.join(platform_user_data_dir(), "logs"),
        os.path.join(tempfile.gettempdir(), "stemma"),
    ]


def install() -> str | None:
    """Start logging to a file and hook uncaught exceptions.

    Returns the log file path, or None if no folder was writable. Safe to
    call more than once.
    """
    global _installed_path
    if _installed_path is not None:
        return _installed_path
    for folder in _candidate_dirs():
        try:
            os.makedirs(folder, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                os.path.join(folder, LOG_NAME),
                maxBytes=_MAX_BYTES,
                backupCount=_BACKUPS,
                encoding="utf-8",
            )
        except OSError:
            continue
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        _installed_path = handler.baseFilename
        break
    sys.excepthook = _log_uncaught
    threading.excepthook = _log_thread_exception
    return _installed_path


def log_path() -> str | None:
    """Return the active log file, or None before ``install``."""
    return _installed_path


def real_log_path() -> str | None:
    """Return the log file's path as Explorer sees it.

    A Store (MSIX) build writes %LOCALAPPDATA% into its package's
    LocalCache. The app sees the ordinary path, but a user who pastes it
    into Explorer finds nothing, so resolve the final path of the open file.
    """
    path = _installed_path
    if path is None or sys.platform != "win32":
        return path
    try:
        # Deferred: Windows-only API.
        import ctypes
        from ctypes import wintypes

        # A private WinDLL, so these signatures stay local to this module.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.CreateFileW.argtypes = (
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
            wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
        )
        kernel32.GetFinalPathNameByHandleW.restype = wintypes.DWORD
        kernel32.GetFinalPathNameByHandleW.argtypes = (
            wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD,
        )
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.CreateFileW(
            path, 0, 0x7, None, 3, 0x02000000, None,
        )  # any access, share all, OPEN_EXISTING, BACKUP_SEMANTICS
        if handle in (None, wintypes.HANDLE(-1).value):
            return path
        try:
            buffer = ctypes.create_unicode_buffer(1024)
            length = kernel32.GetFinalPathNameByHandleW(
                handle, buffer, len(buffer), 0,
            )
        finally:
            kernel32.CloseHandle(handle)
        if not 0 < length < len(buffer):
            return path
        resolved = buffer.value
        if resolved.startswith("\\\\?\\UNC\\"):
            return "\\\\" + resolved[8:]
        return resolved[4:] if resolved.startswith("\\\\?\\") else resolved
    except Exception:  # noqa: BLE001 - a nicer path is optional
        return path


def _log_uncaught(exc_type, exc, tb) -> None:
    logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
    if sys.__excepthook__ is not None and sys.stderr is not None:
        sys.__excepthook__(exc_type, exc, tb)


def _log_thread_exception(args: threading.ExceptHookArgs) -> None:
    if args.exc_type is SystemExit:
        return
    logger.critical(
        "Uncaught exception in thread %s",
        getattr(args.thread, "name", "?"),
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )
    if sys.stderr is not None:  # source runs keep their console traceback
        threading.__excepthook__(args)

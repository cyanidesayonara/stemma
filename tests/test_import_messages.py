"""Tests for user-facing import error formatting."""

import errno
import logging
import socket
import urllib.error

import pytest

import soundfile as sf

from src.import_messages import (
    _READABLE as READABLE,
    MSG_DISK_FULL,
    MSG_DOWNLOAD_DAMAGED,
    MSG_HTTP_404,
    MSG_OFFLINE,
    MSG_PATH_TOO_LONG,
    MSG_PERMISSION,
    MSG_REFUSED,
    MSG_RESET,
    MSG_TIMEOUT,
    MSG_UNREADABLE_AUDIO,
    DownloadIntegrityError,
    describe_error,
    format_import_error,
)


class TestFormatImportError:
    def test_empty(self):
        assert "try again" in format_import_error("").lower()

    def test_disk_full(self):
        out = format_import_error("OSError: [Errno 28] No space left on device")
        assert "disk" in out.lower() or "space" in out.lower()

    def test_permission(self):
        out = format_import_error("PermissionError: [Errno 13] Permission denied")
        assert "permission" in out.lower()

    def test_timeout(self):
        out = format_import_error("socket.timeout: timed out")
        assert "timed out" in out.lower() or "network" in out.lower()

    def test_truncates_long_messages(self):
        long_msg = "x" * 500
        out = format_import_error(long_msg, max_len=100)
        assert len(out) <= 104
        assert out.endswith("...")

    def test_short_passthrough(self):
        assert format_import_error("Custom failure") == "Custom failure"

    def test_onnx_runtime_exception(self):
        raw = (
            "ONNXRuntimeError: 6 : RUNTIME_EXCEPTION : Exception during initialization"
        )
        out = format_import_error(raw)
        assert "stem separation" in out.lower()
        assert "retry" in out.lower()

    def test_onnx_runtime_exception_with_oom(self):
        raw = (
            "ONNXRuntimeError: 6 : RUNTIME_EXCEPTION : "
            "failed to allocate 4294967296 bytes"
        )
        out = format_import_error(raw)
        assert "memory" in out.lower()
        assert "shorter" in out.lower() or "close" in out.lower()

    def test_out_of_memory_phrases(self):
        for raw in (
            "RuntimeError: out of memory",
            "std::bad_alloc",
            "failed to allocate 9000000000 bytes",
        ):
            out = format_import_error(raw)
            assert "memory" in out.lower()

    def test_arbitrary_alloc_substring_not_oom(self):
        """Avoid matching unrelated text that only contains 'alloc'."""
        out = format_import_error("reallocation of vector failed: invalid state")
        assert out == "reallocation of vector failed: invalid state"


class TestWindowsErrorCodesInText:
    """Raw strings that reached users during the pre-release audit."""

    def test_getaddrinfo_is_offline(self):
        out = format_import_error(
            "<urlopen error [Errno 11001] getaddrinfo failed>"
        )
        assert "internet" in out.lower()
        assert "errno" not in out.lower()

    def test_connection_refused(self):
        out = format_import_error(
            "[WinError 10061] No connection could be made because the "
            "target machine actively refused it"
        )
        assert "refused" in out.lower()
        assert "winerror" not in out.lower()

    def test_connection_reset(self):
        out = format_import_error(
            "[WinError 10054] An existing connection was forcibly closed "
            "by the remote host"
        )
        assert "interrupted" in out.lower()

    def test_timed_out_winerror(self):
        out = format_import_error("<urlopen error [WinError 10060] ...>")
        assert "timed out" in out.lower()

    def test_disk_full_winerror(self):
        out = format_import_error(
            "[WinError 112] There is not enough space on the disk"
        )
        assert "disk space" in out.lower()

    def test_path_too_long(self):
        out = format_import_error(
            "[WinError 206] The filename or extension is too long"
        )
        assert "too long" in out.lower()

    def test_corrupt_audio_pipe_message(self):
        out = format_import_error(
            "Error opening 'x.mp3': File does not exist or is not a regular "
            "file (possibly a pipe?)."
        )
        assert "can't read" in out.lower()
        assert "mp3, wav, flac" in out.lower()


class TestYouTubeMessages:
    def test_private(self):
        out = format_import_error(
            "ERROR: [youtube] abc: Private video. Sign in if you've been "
            "granted access to this video"
        )
        assert "private" in out.lower()
        assert "sign in" not in out.lower()

    def test_removed(self):
        out = format_import_error(
            "ERROR: [youtube] abc: Video unavailable. This video has been "
            "removed by the uploader"
        )
        assert "unavailable" in out.lower()
        assert "error:" not in out.lower()

    def test_bot_check(self):
        out = format_import_error(
            "ERROR: [youtube] abc: Sign in to confirm you're not a bot. Use "
            "--cookies-from-browser or --cookies for the authentication."
        )
        assert "bot" in out.lower()
        assert "--cookies" not in out

    def test_network(self):
        out = format_import_error(
            "ERROR: [youtube] abc: Unable to download webpage: <urlopen "
            "error [Errno 11001] getaddrinfo failed> (caused by "
            "TransportError(...))"
        )
        assert "internet" in out.lower()

    def test_unavailable_needs_video_context(self):
        """A plain 'is not available' message is not a YouTube message."""
        raw = "Selected output device is not available"
        assert format_import_error(raw) == raw


def _oserror(winerror: int, message: str = "raw detail") -> OSError:
    """Build an OSError carrying a Windows error code on any platform."""
    exc = OSError(message)
    exc.winerror = winerror
    return exc


class TestDescribeError:
    """Map exceptions by type and code; log the raw text."""

    def test_offline_via_urlerror_reason(self):
        exc = urllib.error.URLError(
            socket.gaierror(11001, "getaddrinfo failed")
        )
        assert describe_error(exc) == MSG_OFFLINE

    def test_winerror_codes(self):
        assert describe_error(_oserror(10061)) == MSG_REFUSED
        assert describe_error(_oserror(10054)) == MSG_RESET
        assert describe_error(_oserror(10060)) == MSG_TIMEOUT
        assert describe_error(_oserror(112)) == MSG_DISK_FULL
        assert describe_error(_oserror(206)) == MSG_PATH_TOO_LONG

    def test_errno_disk_full(self):
        exc = OSError(errno.ENOSPC, "No space left on device")
        assert describe_error(exc) == MSG_DISK_FULL

    def test_permission_error_type(self):
        exc = PermissionError(13, "Permission denied", "C:/out/mix.wav")
        assert describe_error(exc) == MSG_PERMISSION

    def test_connection_types_without_codes(self):
        assert describe_error(ConnectionResetError("reset")) == MSG_RESET
        assert describe_error(ConnectionRefusedError("no")) == MSG_REFUSED
        assert describe_error(TimeoutError("slow")) == MSG_TIMEOUT

    def test_cause_chain(self):
        try:
            try:
                raise _oserror(112)
            except OSError as inner:
                raise RuntimeError("wrapped") from inner
        except RuntimeError as outer:
            assert describe_error(outer) == MSG_DISK_FULL

    def test_yt_dlp_exc_info(self):
        class FakeYtDlpError(Exception):
            def __init__(self, msg, exc_info):
                super().__init__(msg)
                self.exc_info = exc_info

        inner = ConnectionResetError(10054, "forcibly closed")
        exc = FakeYtDlpError(
            "ERROR: Unable to download", (type(inner), inner, None),
        )
        assert describe_error(exc) == MSG_RESET

    def test_soundfile_error(self):
        exc = sf.LibsndfileError(0, prefix="Error opening 'x.mp3': ")
        assert describe_error(exc) == MSG_UNREADABLE_AUDIO

    def test_integrity_error(self):
        exc = DownloadIntegrityError("SHA-256 got abc expected def")
        out = describe_error(exc)
        assert out == MSG_DOWNLOAD_DAMAGED
        assert "SHA-256" not in out

    def test_http_error(self):
        exc = urllib.error.HTTPError("http://x", 404, "Not Found", {}, None)
        assert describe_error(exc) == MSG_HTTP_404

    def test_unknown_exception_falls_back_to_text(self):
        assert describe_error(ValueError("Sample rate mismatch")) == (
            "Sample rate mismatch"
        )

    def test_logs_raw_text_and_traceback(self, caplog):
        exc = _oserror(112, "There is not enough space on the disk")
        with caplog.at_level(logging.WARNING, logger="stemma"):
            describe_error(exc, "Export failed")
        record = caplog.records[-1]
        assert "Export failed" in record.getMessage()
        assert "not enough space" in record.getMessage()
        assert record.exc_info is not None

    def test_readable_message_is_not_logged_again(self, caplog):
        with caplog.at_level(logging.WARNING, logger="stemma"):
            assert describe_error(MSG_DISK_FULL) == MSG_DISK_FULL
        assert caplog.records == []

    def test_readable_messages_are_stable(self):
        """Formatting a readable message again returns it unchanged."""
        assert READABLE
        for message in READABLE:
            assert format_import_error(message) == message


# -- #199 review follow-ups ------------------------------------------------

def test_a_failed_write_is_not_called_unreadable_audio(tmp_path):
    """libsndfile raises the same SoundFileError for a file it cannot open
    for writing; exporting over a read-only or locked file said the audio
    was unreadable."""
    import os
    import stat

    import numpy as np
    import soundfile as sf

    from src.import_messages import MSG_CANNOT_OPEN_FILE, describe_error

    target = tmp_path / "mix.wav"
    target.write_bytes(b"")
    os.chmod(target, stat.S_IREAD)
    try:
        with pytest.raises(sf.SoundFileError) as caught:
            sf.write(str(target), np.zeros((10, 2)), 44100)
    finally:
        os.chmod(target, stat.S_IWRITE | stat.S_IREAD)

    assert describe_error(caught.value, "export") == MSG_CANNOT_OPEN_FILE


@pytest.mark.parametrize("text, expected_name", [
    ("ERROR: [youtube] abc: Requested format is not available. Use "
     "--list-formats for a list of available formats", "MSG_YT_FORMAT"),
    ("ERROR: [youtube] abc: The uploader has not made this video "
     "available in your country", "MSG_YT_UNAVAILABLE"),
    ("ERROR: [youtube] abc: Premieres in 2 days", "MSG_YT_UPCOMING"),
])
def test_youtube_texts(text, expected_name):
    import src.import_messages as messages

    assert format_import_error(text) == getattr(messages, expected_name)


def test_an_implicit_context_does_not_decide_the_message():
    """An unrelated error raised while handling another one used to take
    that one's meaning ("File not found", or even a silent "cancelled")."""
    from src.import_messages import MSG_CANCELLED, describe_error

    try:
        try:
            raise InterruptedError("cancelled")
        except InterruptedError:
            raise ValueError("Unsupported sample rate")
    except ValueError as exc:
        assert describe_error(exc, "import") != MSG_CANCELLED

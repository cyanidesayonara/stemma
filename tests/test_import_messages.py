"""Tests for user-facing import error formatting."""

from src.import_messages import format_import_error


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

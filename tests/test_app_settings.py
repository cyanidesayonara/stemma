"""Tests for QSettings read helpers."""

import os
import re
import sys
import tempfile
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from src.app_settings import (
    SETTINGS_FILE_ENV,
    normalize_output_device_setting,
    open_settings,
    parse_stored_output_device_index,
    read_default_export_format,
    read_default_import_6_stem,
    read_default_mp3_bitrate,
    read_output_device_index,
)


@pytest.fixture
def settings_ini(tmp_path):
    return QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)


class TestReadOutputDeviceIndex:
    def test_default_none(self, settings_ini, monkeypatch):
        monkeypatch.setattr(
            "src.app_settings.output_device_indices_with_output",
            lambda: frozenset({0, 1, 2, 3}),
        )
        assert read_output_device_index(settings_ini) is None

    def test_negative_none(self, settings_ini, monkeypatch):
        monkeypatch.setattr(
            "src.app_settings.output_device_indices_with_output",
            lambda: frozenset({0, 1, 2, 3}),
        )
        settings_ini.setValue("audio/output_device", -1)
        assert read_output_device_index(settings_ini) is None

    def test_positive_index(self, settings_ini, monkeypatch):
        monkeypatch.setattr(
            "src.app_settings.output_device_indices_with_output",
            lambda: frozenset({0, 1, 2, 3}),
        )
        settings_ini.setValue("audio/output_device", 3)
        assert read_output_device_index(settings_ini) == 3

    def test_stale_index_clears_setting(self, settings_ini, monkeypatch):
        monkeypatch.setattr(
            "src.app_settings.output_device_indices_with_output",
            lambda: frozenset({0, 1, 2}),
        )
        settings_ini.setValue("audio/output_device", 99)
        assert read_output_device_index(settings_ini) is None
        assert int(settings_ini.value("audio/output_device")) == -1

    def test_query_failure_keeps_stored_index(self, settings_ini, monkeypatch):
        monkeypatch.setattr(
            "src.app_settings.output_device_indices_with_output",
            lambda: None,
        )
        settings_ini.setValue("audio/output_device", 7)
        assert read_output_device_index(settings_ini) == 7


class TestParseVsNormalizeOutputDevice:
    def test_parse_does_not_clear_stale_index(self, settings_ini, monkeypatch):
        monkeypatch.setattr(
            "src.app_settings.output_device_indices_with_output",
            lambda: frozenset({0, 1}),
        )
        settings_ini.setValue("audio/output_device", 99)
        assert parse_stored_output_device_index(settings_ini) == 99
        assert int(settings_ini.value("audio/output_device")) == 99

    def test_normalize_matches_read_output_device_index(
        self, settings_ini, monkeypatch
    ):
        monkeypatch.setattr(
            "src.app_settings.output_device_indices_with_output",
            lambda: frozenset({0, 1, 2}),
        )
        settings_ini.setValue("audio/output_device", 2)
        assert normalize_output_device_setting(settings_ini) == 2
        assert read_output_device_index(settings_ini) == 2


class TestReadDefaultMp3Bitrate:
    def test_default_320(self, settings_ini):
        assert read_default_mp3_bitrate(settings_ini) == 320

    def test_valid_values(self, settings_ini):
        for b in (192, 256, 320):
            settings_ini.setValue("export/mp3_bitrate", b)
            assert read_default_mp3_bitrate(settings_ini) == b

    def test_invalid_falls_back(self, settings_ini):
        settings_ini.setValue("export/mp3_bitrate", 999)
        assert read_default_mp3_bitrate(settings_ini) == 320


class TestReadDefaultExportFormat:
    def test_default_wav(self, settings_ini):
        assert read_default_export_format(settings_ini) == "wav"

    def test_mp3(self, settings_ini):
        settings_ini.setValue("export/default_format", "mp3")
        assert read_default_export_format(settings_ini) == "mp3"


class TestReadDefaultImport6Stem:
    def test_default_false(self, settings_ini):
        assert read_default_import_6_stem(settings_ini) is False

    def test_true(self, settings_ini):
        settings_ini.setValue("import/default_6_stem", True)
        assert read_default_import_6_stem(settings_ini) is True


class TestOpenSettings:
    """The one place the app opens its settings store."""

    def test_override_writes_an_ini_file(self, tmp_path, monkeypatch):
        path = tmp_path / "isolated.ini"
        monkeypatch.setenv(SETTINGS_FILE_ENV, str(path))

        settings = open_settings()
        settings.setValue("session/last_song_id", "abc")
        settings.sync()

        assert settings.format() == QSettings.Format.IniFormat
        assert path.is_file()
        assert "abc" in path.read_text(encoding="utf-8")

    def test_default_is_the_native_user_store(self, monkeypatch):
        monkeypatch.delenv(SETTINGS_FILE_ENV, raising=False)

        settings = open_settings()

        assert settings.format() == QSettings.Format.NativeFormat
        assert settings.organizationName() == "stemma"
        assert settings.applicationName() == "stemma"

    def test_frozen_builds_ignore_the_override(self, tmp_path, monkeypatch):
        """A stray variable must never redirect a shipped app's settings."""
        monkeypatch.setenv(SETTINGS_FILE_ENV, str(tmp_path / "stray.ini"))
        monkeypatch.setattr(sys, "frozen", True, raising=False)

        settings = open_settings()

        assert settings.format() == QSettings.Format.NativeFormat

    def test_the_suite_uses_a_throwaway_settings_file(self):
        """conftest points every test at a file under pytest's temp dir.

        Without it, any test that builds a MainWindow overwrote a real user's
        session and window state in the registry.
        """
        path = Path(os.environ[SETTINGS_FILE_ENV]).resolve()

        assert path.is_relative_to(Path(tempfile.gettempdir()).resolve())
        assert Path(open_settings().fileName()).resolve() == path

    def test_app_code_opens_settings_only_through_the_helper(self):
        """No entry point may open the native store behind the helper's back.

        main.py did, so theme, data folder, and output device were still
        read from the registry while everything else used the override.
        """
        root = Path(__file__).resolve().parents[1]
        direct = re.compile(r"QSettings\(\s*[\"']stemma[\"']\s*,\s*[\"']stemma[\"']")
        sources = [root / "main.py", *(root / "src").rglob("*.py"),
                   *(root / "scripts").rglob("*.py")]
        offenders = [
            str(path.relative_to(root))
            for path in sources
            if path.name != "settings_store.py"
            and direct.search(path.read_text(encoding="utf-8"))
        ]

        assert offenders == []

"""Tests for QSettings read helpers."""

import pytest
from PySide6.QtCore import QSettings

from src.app_settings import (
    normalize_output_device_setting,
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

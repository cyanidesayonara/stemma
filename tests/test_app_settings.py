"""Tests for QSettings read helpers."""

import pytest
from PySide6.QtCore import QSettings

from src.app_settings import (
    read_default_export_format,
    read_default_import_6_stem,
    read_default_mp3_bitrate,
    read_output_device_index,
)


@pytest.fixture
def settings_ini(tmp_path):
    return QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)


class TestReadOutputDeviceIndex:
    def test_default_none(self, settings_ini):
        assert read_output_device_index(settings_ini) is None

    def test_negative_none(self, settings_ini):
        settings_ini.setValue("audio/output_device", -1)
        assert read_output_device_index(settings_ini) is None

    def test_positive_index(self, settings_ini):
        settings_ini.setValue("audio/output_device", 3)
        assert read_output_device_index(settings_ini) == 3


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

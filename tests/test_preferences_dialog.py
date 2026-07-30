"""Focused persistence coverage for the Preferences dialog."""

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
import pytest

from src.ui.preferences_dialog import PreferencesDialog


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings_ini(tmp_path):
    return QSettings(
        str(tmp_path / "preferences.ini"),
        QSettings.Format.IniFormat,
    )


@pytest.fixture
def audio_devices(monkeypatch):
    devices = [
        {
            "name": "Speakers",
            "max_output_channels": 2,
            "max_input_channels": 0,
        },
        {
            "name": "USB Interface",
            "max_output_channels": 2,
            "max_input_channels": 1,
        },
    ]
    monkeypatch.setattr(
        "src.app_settings.sd.query_devices",
        lambda: devices,
    )
    return devices


def _dialog(monkeypatch, settings):
    monkeypatch.setattr(
        "src.ui.preferences_dialog.QSettings",
        lambda *_args: settings,
    )
    return PreferencesDialog()


def test_preferences_loads_persisted_typed_values(
    app, settings_ini, audio_devices, monkeypatch
):
    settings_ini.setValue("audio/output_device", 1)
    settings_ini.setValue("audio/input_device", 1)
    settings_ini.setValue("audio/latency_offset_ms", 42.5)
    settings_ini.setValue("import/default_6_stem", True)
    settings_ini.setValue("export/default_format", "mp3")
    settings_ini.setValue("export/mp3_bitrate", 256)
    settings_ini.setValue("startup/play_sound", False)
    settings_ini.setValue("playback/sync_recording_pitch", True)

    dialog = _dialog(monkeypatch, settings_ini)

    assert dialog._device_combo.currentData() == 1
    assert dialog._input_device_combo.currentData() == 1
    assert dialog._latency_spin.value() == 42.5
    assert dialog._model_combo.currentData() is True
    assert dialog._export_combo.currentData() == "mp3"
    assert dialog._bitrate_combo.currentData() == 256
    assert not dialog._startup_sound_cb.isChecked()
    assert dialog._sync_rec_pitch_cb.isChecked()


def test_preferences_defaults_are_safe_and_typed(
    app, settings_ini, audio_devices, monkeypatch
):
    dialog = _dialog(monkeypatch, settings_ini)

    assert dialog._device_combo.currentData() == -1
    assert dialog._input_device_combo.currentData() == -1
    assert dialog._latency_spin.value() == 0.0
    assert dialog._model_combo.currentData() is False
    assert dialog._export_combo.currentData() == "wav"
    assert dialog._bitrate_combo.currentData() == 320
    assert dialog._startup_sound_cb.isChecked()
    assert not dialog._sync_rec_pitch_cb.isChecked()


def test_preferences_accept_persists_exposed_settings(
    app, settings_ini, audio_devices, monkeypatch
):
    dialog = _dialog(monkeypatch, settings_ini)
    dialog._device_combo.setCurrentIndex(
        dialog._device_combo.findData(0)
    )
    dialog._input_device_combo.setCurrentIndex(
        dialog._input_device_combo.findData(1)
    )
    dialog._latency_spin.setValue(-17.5)
    dialog._model_combo.setCurrentIndex(dialog._model_combo.findData(True))
    dialog._export_combo.setCurrentIndex(dialog._export_combo.findData("mp3"))
    dialog._bitrate_combo.setCurrentIndex(
        dialog._bitrate_combo.findData(192)
    )
    dialog._startup_sound_cb.setChecked(False)
    dialog._sync_rec_pitch_cb.setChecked(True)

    dialog._on_accept()

    assert int(settings_ini.value("audio/output_device")) == 0
    assert int(settings_ini.value("audio/input_device")) == 1
    assert float(settings_ini.value("audio/latency_offset_ms")) == -17.5
    assert settings_ini.value("import/default_6_stem", type=bool) is True
    assert settings_ini.value("export/default_format") == "mp3"
    assert int(settings_ini.value("export/mp3_bitrate")) == 192
    assert settings_ini.value("startup/play_sound", type=bool) is False
    assert (
        settings_ini.value("playback/sync_recording_pitch", type=bool)
        is True
    )

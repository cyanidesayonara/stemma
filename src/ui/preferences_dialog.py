"""Edit > Preferences: data directory, audio device, import/export defaults."""

from __future__ import annotations

import os

import sounddevice as sd
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.app_settings import (
    input_device_indices_with_input,
    output_device_indices_with_output,
    parse_stored_input_device_index,
    parse_stored_output_device_index,
    read_default_export_format,
    read_default_import_6_stem,
    read_default_mp3_bitrate,
    read_latency_offset_ms,
    read_startup_play_sound,
    read_sync_recording_pitch,
)
from src.data_paths import platform_user_data_dir
from src.version import __version__


def _combo_int_data(combo: QComboBox, default: int = -1) -> int:
    """Return ``currentData()`` as int, or *default* if missing or invalid."""
    d = combo.currentData()
    if d is None:
        return default
    try:
        return int(d)
    except (TypeError, ValueError):
        return default


class PreferencesDialog(QDialog):
    """Modal preferences editor backed by ``QSettings``."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(480)

        self._settings = QSettings("stemma", "stemma")

        self._data_dir_edit = QLineEdit()
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_data_dir)

        data_row = QHBoxLayout()
        data_row.addWidget(self._data_dir_edit)
        data_row.addWidget(browse_btn)

        data_hint = QLabel(
            "Leave empty to use the default user folder. "
            "Changing the data directory requires restarting stemma."
        )
        data_hint.setWordWrap(True)
        data_hint.setObjectName("subtle-label")

        data_box = QGroupBox("Data")
        dform = QFormLayout(data_box)
        dform.addRow("Data directory:", data_row)
        dform.addRow(data_hint)

        self._device_combo = QComboBox()
        self._device_combo.addItem("System default", -1)
        self._populate_output_devices()

        self._input_device_combo = QComboBox()
        self._input_device_combo.addItem("System default", -1)
        self._populate_input_devices()

        self._latency_spin = QDoubleSpinBox()
        self._latency_spin.setRange(-200.0, 200.0)
        self._latency_spin.setValue(0.0)
        self._latency_spin.setSuffix(" ms")
        self._latency_spin.setSingleStep(1.0)
        self._latency_spin.setToolTip(
            "Shift recording earlier (positive) or later (negative) "
            "to compensate for audio interface latency"
        )

        audio_box = QGroupBox("Audio")
        aform = QFormLayout(audio_box)
        aform.addRow("Output device:", self._device_combo)
        aform.addRow("Input device:", self._input_device_combo)
        aform.addRow("Recording latency offset:", self._latency_spin)

        self._model_combo = QComboBox()
        self._model_combo.addItem("4-stem (vocals, drums, bass, other)", False)
        self._model_combo.addItem("6-stem (+ guitar, piano)", True)

        import_box = QGroupBox("Import")
        iform = QFormLayout(import_box)
        iform.addRow("Default separation model:", self._model_combo)

        self._export_combo = QComboBox()
        self._export_combo.addItem("WAV", "wav")
        self._export_combo.addItem("MP3", "mp3")

        self._bitrate_combo = QComboBox()
        for b in (192, 256, 320):
            self._bitrate_combo.addItem(f"{b} kbps", b)

        export_box = QGroupBox("Export")
        eform = QFormLayout(export_box)
        eform.addRow("Default mix format:", self._export_combo)
        eform.addRow("MP3 bitrate:", self._bitrate_combo)

        self._startup_sound_cb = QCheckBox("Play startup sound")
        startup_box = QGroupBox("Startup")
        sform = QFormLayout(startup_box)
        sform.addRow(self._startup_sound_cb)

        self._sync_rec_pitch_cb = QCheckBox(
            "Pitch-shift recording takes with the stems"
        )
        self._sync_rec_pitch_cb.setToolTip(
            "When off (default), recordings play back at their original "
            "pitch even when the stems are transposed."
        )
        playback_box = QGroupBox("Playback")
        pform = QFormLayout(playback_box)
        pform.addRow(self._sync_rec_pitch_cb)

        self._load_from_settings()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(data_box)
        layout.addWidget(audio_box)
        layout.addWidget(import_box)
        layout.addWidget(export_box)
        layout.addWidget(playback_box)
        layout.addWidget(startup_box)

        ver = QLabel(f"stemma {__version__}")
        ver.setObjectName("subtle-label")
        layout.addWidget(ver)

        layout.addWidget(buttons)

    def _populate_output_devices(self) -> None:
        try:
            devices = sd.query_devices()
        except (OSError, ValueError, RuntimeError):
            return
        for i, dev in enumerate(devices):
            ch = int(dev.get("max_output_channels", 0) or 0)
            if ch <= 0:
                continue
            name = str(dev.get("name", f"Device {i}"))[:120]
            self._device_combo.addItem(name, i)

    def _populate_input_devices(self) -> None:
        try:
            devices = sd.query_devices()
        except (OSError, ValueError, RuntimeError):
            return
        for i, dev in enumerate(devices):
            ch = int(dev.get("max_input_channels", 0) or 0)
            if ch <= 0:
                continue
            name = str(dev.get("name", f"Device {i}"))[:120]
            self._input_device_combo.addItem(name, i)

    def _load_from_settings(self) -> None:
        raw = self._settings.value("paths/data_dir", "")
        if isinstance(raw, str):
            self._data_dir_edit.setText(raw.strip())

        dev = parse_stored_output_device_index(self._settings)
        valid = output_device_indices_with_output()
        if dev is not None and valid is not None and dev not in valid:
            dev = None
        target = -1 if dev is None else dev
        for i in range(self._device_combo.count()):
            raw = self._device_combo.itemData(i)
            if raw is None:
                continue
            try:
                if int(raw) == target:
                    self._device_combo.setCurrentIndex(i)
                    break
            except (TypeError, ValueError):
                continue

        in_dev = parse_stored_input_device_index(self._settings)
        valid_in = input_device_indices_with_input()
        if in_dev is not None and valid_in is not None and in_dev not in valid_in:
            in_dev = None
        in_target = -1 if in_dev is None else in_dev
        for i in range(self._input_device_combo.count()):
            raw = self._input_device_combo.itemData(i)
            if raw is None:
                continue
            try:
                if int(raw) == in_target:
                    self._input_device_combo.setCurrentIndex(i)
                    break
            except (TypeError, ValueError):
                continue

        self._latency_spin.setValue(
            read_latency_offset_ms(self._settings)
        )

        self._startup_sound_cb.setChecked(
            read_startup_play_sound(self._settings)
        )

        self._sync_rec_pitch_cb.setChecked(
            read_sync_recording_pitch(self._settings)
        )

        self._model_combo.setCurrentIndex(
            1 if read_default_import_6_stem(self._settings) else 0
        )

        fmt = read_default_export_format(self._settings)
        self._export_combo.setCurrentIndex(1 if fmt == "mp3" else 0)

        br = read_default_mp3_bitrate(self._settings)
        for i in range(self._bitrate_combo.count()):
            raw = self._bitrate_combo.itemData(i)
            if raw is None:
                continue
            try:
                if int(raw) == br:
                    self._bitrate_combo.setCurrentIndex(i)
                    break
            except (TypeError, ValueError):
                continue

    def _on_browse_data_dir(self) -> None:
        start = self._data_dir_edit.text().strip() or platform_user_data_dir()
        path = QFileDialog.getExistingDirectory(self, "Data directory", start)
        if path:
            self._data_dir_edit.setText(os.path.normpath(path))

    def _on_accept(self) -> None:
        new_dir = self._data_dir_edit.text().strip()
        if new_dir:
            try:
                os.makedirs(new_dir, exist_ok=True)
                probe = os.path.join(new_dir, ".stemma_write_test")
                with open(probe, "w", encoding="utf-8") as f:
                    f.write("")
                os.remove(probe)
            except OSError as exc:
                QMessageBox.warning(
                    self,
                    "Preferences",
                    f"Cannot use this data folder (create or write failed):\n{exc}",
                )
                return

        old_raw = self._settings.value("paths/data_dir", "")
        old_dir = old_raw.strip() if isinstance(old_raw, str) else ""

        self._settings.setValue("paths/data_dir", new_dir)
        if new_dir != old_dir:
            QMessageBox.information(
                self,
                "Preferences",
                "Restart stemma for the new data directory to take effect.",
            )

        self._settings.setValue(
            "audio/output_device",
            _combo_int_data(self._device_combo, -1),
        )

        self._settings.setValue(
            "audio/input_device",
            _combo_int_data(self._input_device_combo, -1),
        )

        self._settings.setValue(
            "audio/latency_offset_ms",
            self._latency_spin.value(),
        )

        mdata = self._model_combo.currentData()
        self._settings.setValue(
            "import/default_6_stem",
            bool(mdata) if mdata is not None else False,
        )

        fmt_data = self._export_combo.currentData()
        self._settings.setValue(
            "export/default_format",
            str(fmt_data) if fmt_data is not None else "wav",
        )

        self._settings.setValue(
            "export/mp3_bitrate",
            _combo_int_data(self._bitrate_combo, 320),
        )

        self._settings.setValue(
            "startup/play_sound",
            self._startup_sound_cb.isChecked(),
        )

        self._settings.setValue(
            "playback/sync_recording_pitch",
            self._sync_rec_pitch_cb.isChecked(),
        )
        self.accept()

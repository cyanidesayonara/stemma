"""Typed reads for stemma QSettings keys used across the app."""

from __future__ import annotations

import sounddevice as sd
from PySide6.QtCore import QSettings


def output_device_indices_with_output() -> frozenset[int] | None:
    """Return indices with at least one output channel, or None if query fails."""
    try:
        devices = sd.query_devices()
        out: list[int] = []
        for i, dev in enumerate(devices):
            ch = int(dev.get("max_output_channels", 0) or 0)
            if ch > 0:
                out.append(i)
        return frozenset(out)
    except (OSError, ValueError, RuntimeError):
        return None


def parse_stored_output_device_index(settings: QSettings) -> int | None:
    """Read the stored PortAudio output device index without mutating settings.

    Returns ``None`` for system default (unset, empty, negative, or non-numeric).
    Does not check whether the index is still valid for the current host API.
    """
    v = settings.value("audio/output_device", -1)
    if v in (None, ""):
        return None
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None
    if i < 0:
        return None
    return i


def normalize_output_device_setting(settings: QSettings) -> int | None:
    """Resolve the output device for playback: valid index, or None for default.

    If the stored index is not a current output device, it is cleared in
    *settings* (set to -1) and None is returned. Call at startup and after
    saving preferences, not when merely loading the preferences dialog.
    """
    i = parse_stored_output_device_index(settings)
    if i is None:
        return None
    valid = output_device_indices_with_output()
    if valid is not None and i not in valid:
        settings.setValue("audio/output_device", -1)
        return None
    return i


def read_output_device_index(settings: QSettings) -> int | None:
    """Same as :func:`normalize_output_device_setting` (backward-compatible name)."""
    return normalize_output_device_setting(settings)


def read_default_mp3_bitrate(settings: QSettings) -> int:
    """Return preferred MP3 export bitrate (192, 256, or 320 kbps)."""
    v = settings.value("export/mp3_bitrate", 320)
    try:
        b = int(v)
    except (TypeError, ValueError):
        return 320
    if b not in (192, 256, 320):
        return 320
    return b


def read_default_export_format(settings: QSettings) -> str:
    """Return ``wav`` or ``mp3`` for the export mix dialog default."""
    v = settings.value("export/default_format", "wav")
    if isinstance(v, str) and v.lower() in ("wav", "mp3"):
        return v.lower()
    return "wav"


def read_default_import_6_stem(settings: QSettings) -> bool:
    """Return True if the import dialog should default to the 6-stem model."""
    return bool(settings.value("import/default_6_stem", False, type=bool))

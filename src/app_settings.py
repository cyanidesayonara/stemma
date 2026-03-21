"""Typed reads for stemma QSettings keys used across the app."""

from __future__ import annotations

from PySide6.QtCore import QSettings


def read_output_device_index(settings: QSettings) -> int | None:
    """Return PortAudio output device index, or None for the system default."""
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

"""Offscreen Qt widget PNG snapshot helpers for visual regression tests."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from PySide6.QtCore import QBuffer, QIODevice
from PySide6.QtWidgets import QApplication

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "widget_snapshots"


def _snapshot_path(name: str) -> Path:
    return _FIXTURES_DIR / f"{name}.png"


def render_widget_png(widget, *, width: int, height: int) -> bytes:
    """Resize *widget*, process pending events, and return PNG bytes."""
    widget.resize(width, height)
    QApplication.processEvents()
    pixmap = widget.grab()
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    pixmap.save(buffer, "PNG")
    return bytes(buffer.data())


def assert_widget_snapshot(
    widget,
    name: str,
    *,
    width: int,
    height: int,
) -> None:
    """Compare widget PNG bytes to a golden snapshot, or update when requested."""
    actual_bytes = render_widget_png(widget, width=width, height=height)
    expected_path = _snapshot_path(name)

    if os.environ.get("UPDATE_WIDGET_SNAPSHOTS") == "1":
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_bytes(actual_bytes)
        return

    if not expected_path.is_file():
        raise AssertionError(
            f"Missing widget snapshot: {expected_path}\n"
            f"Run with UPDATE_WIDGET_SNAPSHOTS=1 to create it."
        )

    expected_bytes = expected_path.read_bytes()
    if actual_bytes == expected_bytes:
        return

    actual_path = expected_path.with_name(f"{name}.actual.png")
    actual_path.write_bytes(actual_bytes)
    actual_hash = hashlib.sha256(actual_bytes).hexdigest()
    expected_hash = hashlib.sha256(expected_bytes).hexdigest()
    raise AssertionError(
        f"Widget snapshot mismatch for {name!r}.\n"
        f"Expected: {expected_path} (sha256: {expected_hash})\n"
        f"Actual:   {actual_path} (sha256: {actual_hash})\n"
        f"Run with UPDATE_WIDGET_SNAPSHOTS=1 to update the golden file."
    )

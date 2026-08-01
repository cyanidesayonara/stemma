"""Offscreen Qt widget PNG snapshot helpers for visual regression tests.

Snapshots are compared with a small per-pixel tolerance rather than by raw
bytes. Offscreen Qt registers no system fonts, so the harness loads a real UI
font before rendering (otherwise every label draws as tofu boxes and the
snapshots cover nothing but the background). That font is rasterized slightly
differently across Windows builds and Qt point releases, so an exact byte
compare would fail for reasons unrelated to the widget under test.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PySide6.QtCore import QBuffer, QIODevice
from PySide6.QtGui import QFont, QFontDatabase, QImage
from PySide6.QtWidgets import QApplication

from src.ui.styles import get_stylesheet

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "widget_snapshots"

# Same list scripts/generate_screenshots.py uses.
_UI_FONTS = (
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
)

# A pixel counts as changed when any channel moves by more than this much.
_CHANNEL_TOLERANCE = 16
# Fraction of changed pixels tolerated before a snapshot is called a mismatch.
_MAX_CHANGED_FRACTION = 0.02

_ui_font_family: str | None = None
_ui_font_resolved = False


def ensure_ui_font(app: QApplication) -> str | None:
    """Apply a real UI font so offscreen text is not drawn as tofu boxes.

    The font file is registered once, but the application font is set on every
    call: other tests mutate it, and snapshots must not depend on which ran
    first.
    """
    global _ui_font_family, _ui_font_resolved
    if not _ui_font_resolved:
        _ui_font_resolved = True
        for path in _UI_FONTS:
            if not os.path.isfile(path):
                continue
            font_id = QFontDatabase.addApplicationFont(path)
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                _ui_font_family = families[0]
                break
    if _ui_font_family is not None:
        app.setFont(QFont(_ui_font_family, 9))
    return _ui_font_family


@contextmanager
def deterministic_render_state():
    """Pin process-global Qt state that would otherwise leak between tests.

    The application stylesheet and font are global, and `main_window` applies a
    theme stylesheet app-wide. Without pinning, a snapshot taken after those
    tests renders differently from the same snapshot taken in isolation.
    """
    app = QApplication.instance()
    if app is None:
        yield
        return
    previous_sheet = app.styleSheet()
    previous_font = app.font()
    try:
        app.setStyleSheet(get_stylesheet("dark"))
        ensure_ui_font(app)
        yield
    finally:
        app.setStyleSheet(previous_sheet)
        app.setFont(previous_font)


def _snapshot_path(name: str) -> Path:
    return _FIXTURES_DIR / f"{name}.png"


def render_widget_png(widget, *, width: int, height: int) -> bytes:
    """Resize *widget*, process pending events, and return PNG bytes."""
    with deterministic_render_state():
        widget.resize(width, height)
        QApplication.processEvents()
        pixmap = widget.grab()
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    pixmap.save(buffer, "PNG")
    return bytes(buffer.data())


def _to_array(png_bytes: bytes) -> np.ndarray:
    """Decode PNG bytes into an (h, w, 4) uint8 array."""
    image = QImage.fromData(png_bytes, "PNG")
    if image.isNull():
        raise AssertionError("Could not decode snapshot PNG")
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    height = image.height()
    stride = image.bytesPerLine()
    raw = np.frombuffer(memoryview(image.constBits()), dtype=np.uint8)
    return raw.reshape(height, stride // 4, 4)[:, : image.width(), :].copy()


def compare_snapshots(actual: bytes, expected: bytes) -> tuple[bool, str]:
    """Return (matches, human readable detail) for two PNG payloads."""
    actual_arr = _to_array(actual)
    expected_arr = _to_array(expected)

    if actual_arr.shape != expected_arr.shape:
        return False, (
            f"size differs: expected {expected_arr.shape[1]}x"
            f"{expected_arr.shape[0]}, got {actual_arr.shape[1]}x"
            f"{actual_arr.shape[0]}"
        )

    delta = np.abs(actual_arr.astype(np.int16) - expected_arr.astype(np.int16))
    changed = (delta > _CHANNEL_TOLERANCE).any(axis=2)
    changed_fraction = float(changed.mean())
    detail = (
        f"{changed_fraction:.4%} of pixels changed by more than "
        f"{_CHANNEL_TOLERANCE}/255 (tolerance {_MAX_CHANGED_FRACTION:.2%}), "
        f"max channel delta {int(delta.max())}"
    )
    return changed_fraction <= _MAX_CHANGED_FRACTION, detail


def assert_widget_snapshot(
    widget,
    name: str,
    *,
    width: int,
    height: int,
) -> None:
    """Compare a widget render to a golden snapshot within tolerance."""
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

    matches, detail = compare_snapshots(actual_bytes, expected_path.read_bytes())
    if matches:
        return

    actual_path = expected_path.with_name(f"{name}.actual.png")
    actual_path.write_bytes(actual_bytes)
    raise AssertionError(
        f"Widget snapshot mismatch for {name!r}: {detail}\n"
        f"Expected: {expected_path}\n"
        f"Actual:   {actual_path}\n"
        f"Run with UPDATE_WIDGET_SNAPSHOTS=1 to update the golden file."
    )

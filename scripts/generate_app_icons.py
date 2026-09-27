"""Generate every app icon from the SVG drawings in assets/icons/.

    python scripts/generate_app_icons.py

One drawing cannot serve every size: the full mark (four notes on a stem,
each flowing out as a wave) turns into a striped smudge when scaled down,
and a small drawing scaled to another size blurs every edge. So:

- ``icon_{16,20,24,32}px.svg`` are pixel-placed drawings, each rendered at
  exactly its own size (24 px is the Windows taskbar at 100% scaling).
  Sizes between them (30 px, for example) use the next larger one.
- ``icon_large.svg`` (256 grid) is used from 33 px up.

Writes:

- ``assets/icons/icon_{16..256}.png`` and ``assets/icons/stemma.ico`` (every
  size as its own PNG frame, for the window, taskbar, and the zip build)
- ``assets/msix/``: the MSIX logos at every scale, plus the
  ``Square44x44Logo.targetsize-*`` set (plated and unplated) that the
  taskbar, Start, and Explorer pick from. ``scripts/build_msix.ps1`` indexes
  them in resources.pri; without it Windows only uses the base images.

Uses Qt only (QtSvg + QImage), no Pillow.
"""

from __future__ import annotations

import os
import struct
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QImage, QPainter  # noqa: E402
from PySide6.QtSvg import QSvgRenderer  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICONS_DIR = os.path.join(_ROOT, "assets", "icons")
MSIX_DIR = os.path.join(_ROOT, "assets", "msix")

PIXEL_SIZES = (16, 20, 24, 32)
APP_PNG_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
# Windows' own list for Square44x44Logo target sizes.
TARGET_SIZES = (16, 20, 24, 30, 32, 36, 40, 48, 60, 64, 72, 80, 96, 256)
SCALES = (100, 125, 150, 200, 400)
MSIX_SQUARE = {
    "StoreLogo": 50,
    "Square44x44Logo": 44,
    "Square150x150Logo": 150,
}
WIDE = ("Wide310x150Logo", 310, 150)


def drawing_for(size: int) -> str:
    """Return the SVG path of the drawing meant for *size* pixels."""
    for grid in PIXEL_SIZES:
        if size <= grid:
            return os.path.join(ICONS_DIR, f"icon_{grid}px.svg")
    return os.path.join(ICONS_DIR, "icon_large.svg")


def render(size: int) -> QImage:
    """Render the icon at *size* x *size* px from the drawing for that size."""
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    renderer = QSvgRenderer(drawing_for(size))
    if not renderer.isValid():
        raise RuntimeError(f"Invalid SVG: {drawing_for(size)}")
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return image


def png_bytes(image: QImage) -> bytes:
    """Encode *image* as PNG (deterministic for the same pixels)."""
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    buffer.close()
    return bytes(data)


def ico_bytes(sizes=ICO_SIZES) -> bytes:
    """Return an .ico whose frames are PNGs rendered per size."""
    frames = [(size, png_bytes(render(size))) for size in sizes]
    header = struct.pack("<HHH", 0, 1, len(frames))
    offset = len(header) + 16 * len(frames)
    entries, blobs = b"", b""
    for size, blob in frames:
        dim = 0 if size >= 256 else size  # 0 means 256 in the ICO header
        entries += struct.pack(
            "<BBBBHHII", dim, dim, 0, 0, 1, 32, len(blob), offset,
        )
        blobs += blob
        offset += len(blob)
    return header + entries + blobs


def msix_assets() -> dict[str, QImage]:
    """Return {file name: image} for every MSIX logo variant."""
    assets: dict[str, QImage] = {}
    for base, logical in MSIX_SQUARE.items():
        for scale in SCALES:
            size = round(logical * scale / 100)
            assets[f"{base}.scale-{scale}.png"] = render(size)
    for size in TARGET_SIZES:
        image = render(size)
        assets[f"Square44x44Logo.targetsize-{size}.png"] = image
        # The icon draws its own tile, so the unplated taskbar form is the
        # same image; without it Windows puts a plate behind the tile.
        assets[f"Square44x44Logo.targetsize-{size}_altform-unplated.png"] = image
    base, width, height = WIDE
    for scale in SCALES:
        w, h = round(width * scale / 100), round(height * scale / 100)
        assets[f"{base}.scale-{scale}.png"] = _wide(w, h)
    return assets


def _wide(width: int, height: int) -> QImage:
    """The mark centred on a transparent wide tile, at 72% of its height."""
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    mark = render(round(height * 0.72))
    painter = QPainter(image)
    painter.drawImage(
        (width - mark.width()) // 2, (height - mark.height()) // 2, mark,
    )
    painter.end()
    return image


def generated_files() -> dict[str, bytes]:
    """Return {absolute path: content} for every file this script writes."""
    files = {
        os.path.join(ICONS_DIR, f"icon_{size}.png"): png_bytes(render(size))
        for size in APP_PNG_SIZES
    }
    files[os.path.join(ICONS_DIR, "stemma.ico")] = ico_bytes()
    for name, image in msix_assets().items():
        files[os.path.join(MSIX_DIR, name)] = png_bytes(image)
    return files


def main() -> int:
    # Qt needs an application object for QImage/QSvgRenderer painting.
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # noqa: F841
    files = generated_files()
    os.makedirs(MSIX_DIR, exist_ok=True)
    for name in os.listdir(MSIX_DIR):
        if name.endswith(".png"):
            os.remove(os.path.join(MSIX_DIR, name))
    for path, content in files.items():
        with open(path, "wb") as handle:
            handle.write(content)
    print(f"wrote {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

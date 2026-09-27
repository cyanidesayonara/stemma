"""The icon generator renders each size from the drawing made for it."""

import importlib.util
import os
import struct

import pytest
from PySide6.QtGui import QGuiApplication, QImageReader

_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "generate_app_icons.py",
)
_spec = importlib.util.spec_from_file_location("generate_app_icons", _SCRIPT)
icons = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(icons)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QGuiApplication.instance() or QGuiApplication([])


@pytest.mark.parametrize("size, drawing", [
    (16, "icon_16px.svg"), (20, "icon_20px.svg"), (24, "icon_24px.svg"),
    (30, "icon_32px.svg"), (32, "icon_32px.svg"),
    (36, "icon_large.svg"), (48, "icon_large.svg"), (256, "icon_large.svg"),
])
def test_each_size_uses_its_own_drawing(size, drawing):
    assert os.path.basename(icons.drawing_for(size)) == drawing
    image = icons.render(size)
    assert (image.width(), image.height()) == (size, size)
    # The tile is opaque in the middle: the drawing actually rendered.
    assert image.pixelColor(size // 2, size // 2).alpha() == 255


def test_ico_holds_one_png_frame_per_size(tmp_path):
    path = str(tmp_path / "test.ico")
    data = icons.ico_bytes(sizes=(16, 32, 256))
    with open(path, "wb") as handle:
        handle.write(data)

    reserved, kind, count = struct.unpack_from("<HHH", data)
    assert (reserved, kind, count) == (0, 1, 3)
    widths = [data[6 + 16 * i] for i in range(count)]
    assert widths == [16, 32, 0]  # 0 encodes 256
    reader = QImageReader(path)
    assert reader.imageCount() == 3


def test_msix_set_covers_every_taskbar_target_size():
    assets = icons.msix_assets()

    for size in icons.TARGET_SIZES:
        plated = assets[f"Square44x44Logo.targetsize-{size}.png"]
        assert plated.width() == size
        assert f"Square44x44Logo.targetsize-{size}_altform-unplated.png" in assets
    assert assets["Square44x44Logo.scale-200.png"].width() == 88
    wide = assets["Wide310x150Logo.scale-100.png"]
    assert (wide.width(), wide.height()) == (310, 150)


def _interior_colors(image):
    """Colours inside the rim, leaving out the tile's rounded corners
    (antialiased by design)."""
    size = image.width()
    corner = size // 4

    def in_corner(x, y):
        return ((x < corner or x >= size - corner)
                and (y < corner or y >= size - corner))

    return {
        image.pixel(x, y)
        for y in range(2, size - 2) for x in range(2, size - 2)
        if not in_corner(x, y)
    }


@pytest.mark.parametrize("size", icons.PIXEL_SIZES)
def test_pixel_drawings_render_without_resampling(size):
    """At its own size a pixel drawing uses only its palette: the tile,
    the stem, four note colours and their dimmed corners. Resampling (the
    old 24 px icon was the 32 px drawing scaled) blends dozens more."""
    assert len(_interior_colors(icons.render(size))) <= 10


def test_resampled_drawing_is_what_the_palette_check_catches():
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    image = QImage(24, 24, QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(os.path.join(icons.ICONS_DIR, "icon_large.svg")).render(
        painter, QRectF(0, 0, 24, 24),
    )
    painter.end()

    assert len(_interior_colors(image)) > 10


def test_committed_icons_are_up_to_date():
    """build_msix.ps1 and the app ship these files as committed. After an
    SVG change, regenerate: a stale or missing file fails here."""
    expected = icons.generated_files()
    committed_msix = {
        os.path.join(icons.MSIX_DIR, name)
        for name in os.listdir(icons.MSIX_DIR) if name.endswith(".png")
    }
    assert committed_msix == {
        path for path in expected if path.startswith(icons.MSIX_DIR)
    }
    for path, content in expected.items():
        with open(path, "rb") as handle:
            assert handle.read() == content, os.path.relpath(path)

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
    (16, "icon_small.svg"), (20, "icon_small.svg"),
    (24, "icon_medium.svg"), (40, "icon_medium.svg"),
    (48, "icon_large.svg"), (256, "icon_large.svg"),
])
def test_each_size_uses_its_own_drawing(size, drawing):
    assert os.path.basename(icons.drawing_for(size)) == drawing
    image = icons.render(size)
    assert (image.width(), image.height()) == (size, size)
    # The tile is opaque in the middle: the drawing actually rendered.
    assert image.pixelColor(size // 2, size // 2).alpha() == 255


def test_ico_holds_one_png_frame_per_size(tmp_path):
    path = str(tmp_path / "test.ico")

    icons.write_ico(path, sizes=(16, 32, 256))

    data = open(path, "rb").read()
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


def test_committed_msix_images_are_up_to_date():
    """build_msix.ps1 copies assets/msix as-is; regenerate after changing
    the SVGs so the package does not ship stale or missing sizes."""
    committed = {
        name for name in os.listdir(icons.MSIX_DIR) if name.endswith(".png")
    }
    assert committed == set(icons.msix_assets())

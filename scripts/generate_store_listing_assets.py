"""Generate the Microsoft Partner Center store listing images.

    python scripts/generate_store_listing_assets.py

Outputs (PNG, under assets/store_listing/):
  - poster_720x1080.png   -- 2:3 poster art
  - box_1080x1080.png     -- 1:1 box art
  - tile_300x300.png      -- 1:1 Store logo
  - logo_150x150.png      -- 1:1 Store logo
  - logo_71x71.png        -- 1:1 Store logo

The poster and box are the brand art: the main logo (the stemma wordmark
over a clef and a chord whose notes flow out as the stem waves), centred
on the dark theme's base colour. It is centred on its visible ink, not
its viewBox, which has uneven padding. The arpeggio strip that used to sit under
it repeated the wordmark and left a dead band in the middle of the poster.

The three Store logos are the app icon itself, so the Store shows the same
mark as the taskbar and Start. Each is rendered by
scripts/generate_app_icons.py from the drawing made for its size.

Uses Qt only (QtSvg + QImage), no Pillow. Re-run after changing
assets/icons/logo_main_dark.svg or the app icon drawings.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF  # noqa: E402
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter  # noqa: E402
from PySide6.QtSvg import QSvgRenderer  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_SVG = os.path.join(_ROOT, "assets", "icons", "logo_main_dark.svg")
OUT_DIR = os.path.join(_ROOT, "assets", "store_listing")
BACKGROUND = QColor("#1e1e2e")  # Catppuccin Mocha base, the app's dark theme

# Share of the canvas width the logo spans on the poster and box.
_LOGO_WIDTH = {"poster": 0.84, "box": 0.72}
# The Store may overlay text on the bottom third of the poster and box, so
# the logo ends at least this far above it.
SAFE_MARGIN = 24
STORE_LOGO_SIZES = {"tile_300x300.png": 300, "logo_150x150.png": 150,
                    "logo_71x71.png": 71}


def _app_icons():
    """The app icon generator, loaded from its script file."""
    path = os.path.join(_ROOT, "scripts", "generate_app_icons.py")
    spec = importlib.util.spec_from_file_location("generate_app_icons", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ink_bounds(image: QImage) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the pixels that differ from the
    brand background, inclusive."""
    argb = image.convertToFormat(QImage.Format.Format_ARGB32)
    pixels = np.frombuffer(argb.constBits(), dtype=np.uint32).reshape(
        argb.height(), argb.bytesPerLine() // 4
    )[:, :argb.width()]
    ys, xs = np.nonzero(pixels != np.uint32(BACKGROUND.rgba()))
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _paint_logo(width: int, height: int, rect: QRectF,
                renderer: QSvgRenderer) -> QImage:
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(BACKGROUND)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter, rect)
    painter.end()
    return image


def brand_art(width: int, height: int, logo_share: float) -> QImage:
    """The main logo centred on the brand background by its visible ink."""
    renderer = QSvgRenderer(MAIN_SVG)
    if not renderer.isValid():
        raise RuntimeError(f"Invalid SVG: {MAIN_SVG}")
    view = renderer.viewBoxF()
    logo_w = width * logo_share
    logo_h = logo_w * view.height() / view.width()
    rect = QRectF((width - logo_w) / 2, (height - logo_h) / 2, logo_w, logo_h)
    left, top, right, bottom = ink_bounds(
        _paint_logo(width, height, rect, renderer)
    )
    # Move the ink's centre to the canvas centre, then a little above it:
    # the optical centre sits higher than the geometric one.
    dx = width / 2 - (left + right + 1) / 2
    dy = height / 2 - (top + bottom + 1) / 2 - height * 0.03
    safe_bottom = height * 2 / 3 - SAFE_MARGIN
    dy = min(dy, safe_bottom - bottom)
    return _paint_logo(width, height, rect.translated(dx, dy), renderer)


def generated_images() -> dict[str, QImage]:
    """Return {file name: image} for every Store listing image."""
    icons = _app_icons()
    images = {
        "poster_720x1080.png": brand_art(720, 1080, _LOGO_WIDTH["poster"]),
        "box_1080x1080.png": brand_art(1080, 1080, _LOGO_WIDTH["box"]),
    }
    for name, size in STORE_LOGO_SIZES.items():
        images[name] = icons.render(size)
    return images


def main() -> int:
    # Qt needs an application object for QImage/QSvgRenderer painting.
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)  # noqa: F841
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, image in generated_images().items():
        image.save(os.path.join(OUT_DIR, name))
        print("wrote", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

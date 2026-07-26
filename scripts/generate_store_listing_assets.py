"""Generate Microsoft Partner Center store listing images.

Each format gets a composition built for its aspect ratio, rather than
one stacked layout reused everywhere:

  poster_720x1080.png   9:16  -- portrait lockup (mark over wordmark)
  box_1080x1080.png     1:1   -- horizontal lockup, generous margins
  tile_300x300.png      1:1   -- mark only (wordmark is unreadable this small)
  logo_150x150.png      1:1   -- app icon tile
  logo_71x71.png        1:1   -- app icon tile

Brand geometry lives in scripts/generate_brand.py (single source of
truth); this script only places and rasterizes it. Run that first if the
mark changed:

    python scripts/generate_brand.py
    python scripts/generate_store_listing_assets.py

Requires: PySide6 (SVG rasterization).
"""

from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT)

_ICONS = os.path.join(_ROOT, "assets", "icons")
_OUT_DIR = os.path.join(_ROOT, "assets", "store_listing")

# Catppuccin Mocha base -- matches the app's dark theme, so the store art
# and the first launch look like the same product.
_BG = "#1e1e2e"

# (output name, canvas size, source svg, fraction of canvas the art spans)
_JOBS = (
    ("poster_720x1080.png", (720, 1080), "lockup_stacked_dark.svg", 0.74),
    ("box_1080x1080.png", (1080, 1080), "lockup_dark.svg", 0.82),
    ("tile_300x300.png", (300, 300), "mark_dark.svg", 0.80),
    ("logo_150x150.png", (150, 150), "icon_square.svg", 1.00),
    ("logo_71x71.png", (71, 71), "icon_square.svg", 1.00),
)


def _render(svg_path: str, out_path: str, size: tuple[int, int],
            span: float) -> None:
    """Center *svg_path* on a brand-colored canvas of *size*.

    *span* is the fraction of the canvas the artwork occupies on its
    limiting axis. The icon tiles pass 1.0 because they already include
    their own rounded-tile background.
    """
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtWidgets import QApplication

    _ = QApplication.instance() or QApplication([])

    w, h = size
    renderer = QSvgRenderer(svg_path)
    vb = renderer.viewBoxF()
    if vb.width() <= 0 or vb.height() <= 0:
        raise ValueError(f"{svg_path}: missing or empty viewBox")

    img = QImage(w, h, QImage.Format.Format_ARGB32)
    img.fill(QColor(_BG))
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    # Fit inside the span box, preserving aspect ratio.
    scale = min(w * span / vb.width(), h * span / vb.height())
    dw, dh = vb.width() * scale, vb.height() * scale
    renderer.render(painter, QRectF((w - dw) / 2.0, (h - dh) / 2.0, dw, dh))
    painter.end()

    img.save(out_path)
    print(f"  {os.path.basename(out_path)} ({w}x{h}) "
          f"from {os.path.basename(svg_path)}")


def main() -> None:
    os.makedirs(_OUT_DIR, exist_ok=True)
    for name, size, svg, span in _JOBS:
        src = os.path.join(_ICONS, svg)
        if not os.path.isfile(src):
            raise FileNotFoundError(
                f"{src} missing -- run scripts/generate_brand.py first"
            )
        _render(src, os.path.join(_OUT_DIR, name), size, span)
    print(f"\nWrote assets to {_OUT_DIR}")


if __name__ == "__main__":
    main()

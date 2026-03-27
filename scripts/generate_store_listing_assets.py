"""Generate Microsoft Partner Center store listing images.

Large assets (poster, box) use the main (chord) + arpeggio SVG logos on brand
background. Content is trimmed to ink pixels so centering matches the visible
logo. Small tiles use the main logo (trimmed, larger fill); 71px uses icon.

Requires: Pillow, PySide6 (for SVG rasterization).

Outputs (PNG, under assets/store_listing/):
  - poster_720x1080.png   -- 9:16 poster art
  - box_1080x1080.png     -- 1:1 box art
  - tile_300x300.png      -- 1:1 app tile (main logo)
  - logo_150x150.png      -- 1:1 logo (main logo)
  - logo_71x71.png        -- 1:1 logo (square icon)

Re-run after changing assets/icons/*.svg or icon_256.png:
  python scripts/generate_store_listing_assets.py
"""

from __future__ import annotations

import io
import os
import sys

from PIL import Image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
_ICON_SRC = os.path.join(_ROOT, "assets", "icons", "icon_256.png")
_MAIN_SVG = os.path.join(_ROOT, "assets", "icons", "logo_main_dark.svg")
_ARP_SVG = os.path.join(_ROOT, "assets", "icons", "logo_arpeggio_dark.svg")
_OUT_DIR = os.path.join(_ROOT, "assets", "store_listing")

# Catppuccin Mocha base (matches app dark theme)
_BG = (30, 30, 46, 255)


def _ensure_qt() -> None:
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:
        QApplication(sys.argv)


def _svg_to_pil(svg_path: str, max_w: int, max_h: int) -> Image.Image:
    """Rasterize SVG to RGBA PIL Image, scaled to fit inside max_w x max_h."""
    from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QRectF
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    _ensure_qt()
    renderer = QSvgRenderer(svg_path)
    if not renderer.isValid():
        raise RuntimeError(f"Invalid SVG: {svg_path}")

    dw = renderer.defaultSize().width()
    dh = renderer.defaultSize().height()
    if dw <= 0 or dh <= 0:
        raise RuntimeError(f"Bad defaultSize for {svg_path}")

    scale = min(max_w / dw, max_h / dh)
    out_w = max(1, int(dw * scale))
    out_h = max(1, int(dh * scale))

    img = QImage(out_w, out_h, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(p, QRectF(0, 0, out_w, out_h))
    p.end()

    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return Image.open(io.BytesIO(ba.data())).convert("RGBA")


def _trim_alpha(im: Image.Image) -> Image.Image:
    """Crop to bounding box of non-transparent pixels (visual ink)."""
    bbox = im.getbbox()
    if bbox is None:
        return im
    return im.crop(bbox)


def _fit_rgba(src: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = src.size
    scale = min(max_w / w, max_h / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return src.resize((nw, nh), Image.Resampling.LANCZOS)


def _svg_trim_fit(svg_path: str, max_w: int, max_h: int, supersample: int = 2) -> Image.Image:
    """Rasterize SVG, trim empty margins, scale to fit max_w x max_h.

    Supersample > 1 renders larger first for cleaner edges after downscale.
    """
    sw = max_w * supersample
    sh = max_h * supersample
    pil = _svg_to_pil(svg_path, sw, sh)
    pil = _trim_alpha(pil)
    return _fit_rgba(pil, max_w, max_h)


def _make_canvas(w: int, h: int) -> Image.Image:
    return Image.new("RGBA", (w, h), _BG)


def _paste_xy(canvas: Image.Image, layer: Image.Image, x: int, y: int) -> None:
    canvas.paste(layer, (x, y), layer)


def main() -> None:
    os.makedirs(_OUT_DIR, exist_ok=True)

    # -- Poster: chord above arpeggio; both horizontally centered; stack vertically centered --
    poster_w, poster_h = 720, 1080
    pad_x = 40
    main_max_w = poster_w - 2 * pad_x
    main_max_h = int(poster_h * 0.50)
    arp_max_w = poster_w - 2 * pad_x
    arp_max_h = int(poster_h * 0.26)
    gap = int(poster_h * 0.035)

    main_p = _svg_trim_fit(_MAIN_SVG, main_max_w, main_max_h, supersample=2)
    arp_p = _svg_trim_fit(_ARP_SVG, arp_max_w, arp_max_h, supersample=2)

    mw, mh = main_p.size
    aw, ah = arp_p.size
    total_h = mh + gap + ah
    y0 = (poster_h - total_h) // 2

    c_poster = _make_canvas(poster_w, poster_h)
    _paste_xy(c_poster, main_p, (poster_w - mw) // 2, y0)
    _paste_xy(c_poster, arp_p, (poster_w - aw) // 2, y0 + mh + gap)
    c_poster.save(os.path.join(_OUT_DIR, "poster_720x1080.png"), optimize=True)
    print(f"  poster_720x1080.png ({poster_w}x{poster_h}) [main + arpeggio, centered stack]")

    # -- Box: same layout logic in 1080 square --
    box_s = 1080
    pad_x_b = 48
    main_max_w_b = box_s - 2 * pad_x_b
    main_max_h_b = int(box_s * 0.48)
    arp_max_w_b = box_s - 2 * pad_x_b
    arp_max_h_b = int(box_s * 0.24)
    gap_b = int(box_s * 0.032)

    main_b = _svg_trim_fit(_MAIN_SVG, main_max_w_b, main_max_h_b, supersample=2)
    arp_b = _svg_trim_fit(_ARP_SVG, arp_max_w_b, arp_max_h_b, supersample=2)

    mbw, mbh = main_b.size
    abw, abh = arp_b.size
    total_h_b = mbh + gap_b + abh
    y0_b = (box_s - total_h_b) // 2

    c_box = _make_canvas(box_s, box_s)
    _paste_xy(c_box, main_b, (box_s - mbw) // 2, y0_b)
    _paste_xy(c_box, arp_b, (box_s - abw) // 2, y0_b + mbh + gap_b)
    c_box.save(os.path.join(_OUT_DIR, "box_1080x1080.png"), optimize=True)
    print(f"  box_1080x1080.png ({box_s}x{box_s}) [main + arpeggio, centered stack]")

    # -- Tiles: larger fill, trimmed chord logo, dead center --
    for name, size, margin_pct in (
        ("tile_300x300.png", 300, 0.04),
        ("logo_150x150.png", 150, 0.04),
    ):
        m = max(4, int(size * margin_pct))
        max_inner = size - 2 * m
        chord = _svg_trim_fit(_MAIN_SVG, max_inner, max_inner, supersample=3)
        cw, ch = chord.size
        c = _make_canvas(size, size)
        _paste_xy(c, chord, (size - cw) // 2, (size - ch) // 2)
        c.save(os.path.join(_OUT_DIR, name), optimize=True)
        print(f"  {name} ({size}x{size}) [main logo, centered]")

    # -- Tiny: square icon --
    icon = Image.open(_ICON_SRC).convert("RGBA")
    for name, size, margin_pct in (("logo_71x71.png", 71, 0.08),):
        m = max(2, int(size * margin_pct))
        max_inner = size - 2 * m
        icon_i = _fit_rgba(_trim_alpha(icon), max_inner, max_inner)
        iw, ih = icon_i.size
        c = _make_canvas(size, size)
        _paste_xy(c, icon_i, (size - iw) // 2, (size - ih) // 2)
        c.save(os.path.join(_OUT_DIR, name), optimize=True)
        print(f"  {name} ({size}x{size}) [square icon]")

    print(f"\nWrote assets to {_OUT_DIR}")


if __name__ == "__main__":
    main()

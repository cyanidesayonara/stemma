"""Generate Microsoft Partner Center store listing images from icon_256.png.

Outputs (PNG, under assets/store_listing/):
  - poster_720x1080.png   -- 9:16 poster art
  - box_1080x1080.png     -- 1:1 box art
  - tile_300x300.png      -- 1:1 app tile icon
  - logo_150x150.png      -- 1:1 logo (150)
  - logo_71x71.png        -- 1:1 logo (71)

Background uses app dark base (#1e1e2e) for consistency with stemma branding.
"""

from __future__ import annotations

import os

from PIL import Image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC = os.path.join(_ROOT, "assets", "icons", "icon_256.png")
_OUT_DIR = os.path.join(_ROOT, "assets", "store_listing")

# Catppuccin Mocha base (matches app dark theme)
_BG = (30, 30, 46, 255)


def _fit_icon(src: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Scale icon to fit inside max_w x max_h, preserving aspect ratio."""
    w, h = src.size
    scale = min(max_w / w, max_h / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return src.resize((nw, nh), Image.Resampling.LANCZOS)


def _make_canvas(w: int, h: int) -> Image.Image:
    return Image.new("RGBA", (w, h), _BG)


def _paste_center(canvas: Image.Image, icon: Image.Image) -> None:
    cw, ch = canvas.size
    iw, ih = icon.size
    x = (cw - iw) // 2
    y = (ch - ih) // 2
    canvas.paste(icon, (x, y), icon)


def main() -> None:
    os.makedirs(_OUT_DIR, exist_ok=True)
    src = Image.open(_SRC).convert("RGBA")

    # 9:16 poster -- leave margin; icon ~55% of canvas height
    poster_w, poster_h = 720, 1080
    icon_p = _fit_icon(src, int(poster_w * 0.85), int(poster_h * 0.55))
    c_poster = _make_canvas(poster_w, poster_h)
    _paste_center(c_poster, icon_p)
    c_poster.save(os.path.join(_OUT_DIR, "poster_720x1080.png"), optimize=True)
    print(f"  poster_720x1080.png ({poster_w}x{poster_h})")

    # 1:1 box art
    box_s = 1080
    icon_b = _fit_icon(src, int(box_s * 0.72), int(box_s * 0.72))
    c_box = _make_canvas(box_s, box_s)
    _paste_center(c_box, icon_b)
    c_box.save(os.path.join(_OUT_DIR, "box_1080x1080.png"), optimize=True)
    print(f"  box_1080x1080.png ({box_s}x{box_s})")

    # Square tiles (fill most of frame)
    for name, size, margin in (
        ("tile_300x300.png", 300, 0.14),
        ("logo_150x150.png", 150, 0.12),
        ("logo_71x71.png", 71, 0.10),
    ):
        m = int(size * margin)
        max_inner = size - 2 * m
        icon_i = _fit_icon(src, max_inner, max_inner)
        c = _make_canvas(size, size)
        _paste_center(c, icon_i)
        c.save(os.path.join(_OUT_DIR, name), optimize=True)
        print(f"  {name} ({size}x{size})")

    print(f"\nWrote assets to {_OUT_DIR}")


if __name__ == "__main__":
    main()

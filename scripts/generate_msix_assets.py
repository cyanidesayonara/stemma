"""Generate MSIX visual assets from the existing app icon.

Produces PNG files at the sizes required by AppxManifest.xml:
  - StoreLogo.scale-{100,200}.png      (50x50, 100x100)
  - Square44x44Logo.scale-{100,200}.png (44x44, 88x88)
  - Square44x44Logo.targetsize-{24,32,48}.png (unplated taskbar icons)
  - Square150x150Logo.scale-{100,200}.png (150x150, 300x300)
  - Wide310x150Logo.scale-100.png       (310x150, icon centered)

Source: assets/icons/icon_256.png
Output: assets/msix/
"""

import os
import sys

from PIL import Image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC = os.path.join(_ROOT, "assets", "icons", "icon_256.png")
_OUT = os.path.join(_ROOT, "assets", "msix")

_SQUARE_ASSETS = [
    ("StoreLogo.scale-100.png", 50),
    ("StoreLogo.scale-200.png", 100),
    ("Square44x44Logo.scale-100.png", 44),
    ("Square44x44Logo.scale-200.png", 88),
    ("Square44x44Logo.targetsize-24.png", 24),
    ("Square44x44Logo.targetsize-32.png", 32),
    ("Square44x44Logo.targetsize-48.png", 48),
    ("Square150x150Logo.scale-100.png", 150),
    ("Square150x150Logo.scale-200.png", 300),
]


def _make_square(src: Image.Image, size: int) -> Image.Image:
    return src.resize((size, size), Image.Resampling.LANCZOS)


def _make_wide(src: Image.Image, w: int, h: int) -> Image.Image:
    """Centre the icon on a transparent wide canvas."""
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    icon_h = h - 20
    icon = src.resize((icon_h, icon_h), Image.Resampling.LANCZOS)
    x = (w - icon_h) // 2
    y = (h - icon_h) // 2
    canvas.paste(icon, (x, y), icon)
    return canvas


def main() -> None:
    os.makedirs(_OUT, exist_ok=True)
    src = Image.open(_SRC).convert("RGBA")

    for name, size in _SQUARE_ASSETS:
        img = _make_square(src, size)
        path = os.path.join(_OUT, name)
        img.save(path)
        print(f"  {name} ({size}x{size})")

    wide = _make_wide(src, 310, 150)
    wide_path = os.path.join(_OUT, "Wide310x150Logo.scale-100.png")
    wide.save(wide_path)
    print(f"  Wide310x150Logo.scale-100.png (310x150)")

    print(f"\nGenerated {len(_SQUARE_ASSETS) + 1} assets in {_OUT}")


if __name__ == "__main__":
    main()

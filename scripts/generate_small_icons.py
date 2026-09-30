"""Generate the small app icon drawings, assets/icons/icon_{16..32}px.svg.

    python scripts/generate_small_icons.py
    python scripts/generate_app_icons.py

At taskbar and title-bar sizes the full mark (four notes flowing into
four waves) became unreadable stripes (#159). The small drawings are its
short form: one note whose flag is three coloured waves. Each size is a
pixel map sampled from simple shapes (a tilted notehead, a stem, three
sine waves) with no antialiasing, written as whole-pixel rects, so it is
rendered at exactly its own size with a small palette. Then run
generate_app_icons.py to write the PNG, ICO, and MSIX files.
"""
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILE, RIM, STEM = "#1e1e2e", "#45475a", "#cdd6f4"
TEAL, PURPLE, ORANGE, PINK = "#4fb8b8", "#bfa3dc", "#e4ad6e", "#d4849a"

# Per size: stem left column, width, first and last row; notehead centre
# and radii; flag x range, first wave centre row, whole-row step between
# the waves, amplitude, thickness, and cycles. The rows are whole numbers so
# all three waves rasterize to the same shape.
SPEC = {
    16: dict(sx=7, sw=1, st=3, sb=12, hx=5.5, hy=12.2, rx=2.8, ry=1.95,
             flag16=True, fy=3, step=2),
    20: dict(sx=9, sw=1, st=4, sb=15, hx=6.8, hy=15.0, rx=3.4, ry=2.3,
             fx0=10, fx1=16, fy=5, step=3, amp=0.9, th=1.2, cyc=1.0),
    24: dict(sx=11, sw=2, st=4, sb=18, hx=8.4, hy=18.0, rx=4.1, ry=2.9,
             fx0=13, fx1=20, fy=5, step=3, amp=1.0, th=1.5, cyc=1.0),
    30: dict(sx=13, sw=2, st=5, sb=22, hx=10.4, hy=21.8, rx=5.1, ry=3.5,
             fx0=15, fx1=25, fy=6, step=4, amp=1.3, th=2.0, cyc=1.25),
    32: dict(sx=14, sw=2, st=5, sb=24, hx=11.2, hy=23.6, rx=5.4, ry=3.7,
             fx0=16, fx1=27, fy=6, step=5, amp=1.4, th=2.0, cyc=1.25),
    36: dict(sx=16, sw=2, st=5, sb=27, hx=12.6, hy=26.6, rx=6.1, ry=4.2,
             fx0=18, fx1=31, fy=7, step=5, amp=1.6, th=2.2, cyc=1.25),
}
WAVES = (PURPLE, ORANGE, PINK)


def _wave_cells(p):
    """(x, dy) cells of one wave, dy relative to its centre row."""
    if p.get("flag16"):
        # Crest, trough, crest: at 16 px a sampled sine is a one-row step.
        return [(8, 0), (9, 0), (10, 1), (11, 1), (12, 0), (13, 0)]
    cells = []
    for x in range(p["fx0"], p["fx1"] + 1):
        t = (x + 0.5 - p["fx0"]) / (p["fx1"] + 1 - p["fx0"])
        yc = 0.5 - p["amp"] * math.sin(t * p["cyc"] * 2 * math.pi)
        for dy in range(-4, 5):
            if abs(dy + 0.5 - yc) <= p["th"] / 2 + 0.01:
                cells.append((x, dy))
    return cells


def _prune(g, color):
    """Drop *color* pixels with fewer than two same-colour neighbours."""
    size = len(g)
    for y in range(size):
        for x in range(size):
            if g[y][x] != color:
                continue
            neighbours = sum(
                1 for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= x + dx < size and 0 <= y + dy < size
                and g[y + dy][x + dx] in (color, STEM)
            )
            if neighbours < 2:
                g[y][x] = None


def grid_for(size):
    """The pixel map of the *size* drawing: rows of colours or None."""
    p = SPEC[size]
    g = [[None] * size for _ in range(size)]
    # Notehead: a rotated ellipse, tested at pixel centres.
    ang = math.radians(-20)
    ca, sa = math.cos(ang), math.sin(ang)
    for y in range(size):
        for x in range(size):
            dx, dy = x + 0.5 - p["hx"], y + 0.5 - p["hy"]
            u = dx * ca + dy * sa
            v = -dx * sa + dy * ca
            if (u / p["rx"]) ** 2 + (v / p["ry"]) ** 2 <= 1.0:
                g[y][x] = TEAL
    for y in range(p["st"], p["sb"]):
        for x in range(p["sx"], p["sx"] + p["sw"]):
            g[y][x] = STEM
    _prune(g, TEAL)
    # The flag: one wave, stamped a whole number of rows apart.
    cells = _wave_cells(p)
    for i, color in enumerate(WAVES):
        row = p["fy"] + i * p["step"]
        for x, dy in cells:
            g[row + dy][x] = color
    return g


def svg_for(size, g):
    """SVG text for pixel map *g*: one rect per run of equal pixels."""
    r = size * 0.21
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" '
        f'height="{size}" viewBox="0 0 {size} {size}" '
        'shape-rendering="crispEdges">',
        f"  <!-- stemma app icon, {size} px drawing. Pixel-placed (whole-pixel"
        " rects, crispEdges), so it is rendered only at its own size and "
        "never resampled. The small form of the logo: one note whose flag "
        "is three coloured waves. Generated from a pixel map; rendered by "
        "scripts/generate_app_icons.py. -->",
        f'  <rect width="{size}" height="{size}" rx="{r:g}" fill="{TILE}"/>'
        f'<rect x="0.5" y="0.5" width="{size - 1}" height="{size - 1}" '
        f'rx="{r - 0.5:g}" fill="none" stroke="{RIM}" stroke-width="1"/>',
    ]
    rects = []
    for y in range(size):
        x = 0
        while x < size:
            c = g[y][x]
            if c is None:
                x += 1
                continue
            x0 = x
            while x < size and g[y][x] == c:
                x += 1
            rects.append(
                f'<rect x="{x0}" y="{y}" width="{x - x0}" height="1" '
                f'fill="{c}"/>'
            )
    parts.append("  " + "".join(rects) + "</svg>")
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    for size in SPEC:
        path = os.path.join(ROOT, "assets", "icons", f"icon_{size}px.svg")
        # Platform line endings, as git checks the SVGs out.
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(svg_for(size, grid_for(size)))
    print("ok")

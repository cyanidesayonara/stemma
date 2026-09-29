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

# Per size: stem x (left col), stem width, stem rows (top, bottom),
# head centre, radii; flag: x range, first wave y, gap, amplitude, thickness,
# cycles.
SPEC = {
    16: dict(sx=7, sw=1, st=2, sb=11, hx=5.5, hy=11.2, rx=2.8, ry=1.95,
             fx0=8, fx1=13, fy=3.0, gap=2.2, amp=0.75, th=1.0, cyc=1.0),
    20: dict(sx=9, sw=1, st=3, sb=14, hx=6.8, hy=14.0, rx=3.4, ry=2.4,
             fx0=10, fx1=16, fy=3.9, gap=2.6, amp=0.9, th=1.2, cyc=1.0),
    24: dict(sx=11, sw=2, st=3, sb=17, hx=8.4, hy=17.0, rx=4.1, ry=2.9,
             fx0=13, fx1=20, fy=4.4, gap=3.2, amp=1.0, th=1.5, cyc=1.0),
    32: dict(sx=14, sw=2, st=4, sb=23, hx=11.2, hy=22.6, rx=5.4, ry=3.7,
             fx0=16, fx1=27, fy=5.8, gap=4.2, amp=1.4, th=2.0, cyc=1.25),
}


def grid_for(size):
    p = SPEC[size]
    g = [[None] * size for _ in range(size)]
    # notehead: rotated ellipse, tested at pixel centres
    ang = math.radians(-20)
    ca, sa = math.cos(ang), math.sin(ang)
    for y in range(size):
        for x in range(size):
            dx, dy = x + 0.5 - p["hx"], y + 0.5 - p["hy"]
            u = dx * ca + dy * sa
            v = -dx * sa + dy * ca
            if (u / p["rx"]) ** 2 + (v / p["ry"]) ** 2 <= 1.0:
                g[y][x] = TEAL
    # stem
    for y in range(p["st"], p["sb"]):
        for x in range(p["sx"], p["sx"] + p["sw"]):
            g[y][x] = STEM
    # flag: three waves
    for i, color in enumerate((PURPLE, ORANGE, PINK)):
        cy = p["fy"] + i * p["gap"]
        for x in range(p["fx0"], p["fx1"] + 1):
            t = (x + 0.5 - p["fx0"]) / (p["fx1"] + 1 - p["fx0"])
            yc = cy - p["amp"] * math.sin(t * p["cyc"] * 2 * math.pi)
            for y in range(size):
                if abs(y + 0.5 - yc) <= p["th"] / 2 + 0.01:
                    g[y][x] = color
    return g


def svg_for(size, g):
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
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(svg_for(size, grid_for(size)))
    print("ok")

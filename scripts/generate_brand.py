"""Generate the stemma brand system (mark, icon, wordmark lockups).

One source of truth for every logo asset. Re-run after changing any
constant here:

    python scripts/generate_brand.py

Concept
-------
A root-position triad whose note stems continue into audio waveforms:
one song, separated into stems. The pun (note stem / audio stem) is the
brand idea, so the geometry has to make it literal -- each wave is the
*same stroke* as its stem, leaving the notehead and running out in time.

Execution rules (what makes it read as music rather than clip art):
  * Engraving proportions. With staff space S: notehead is 1.18S wide,
    1.0S tall, rotated -20 degrees; stems and staff lines take their
    thickness from S. Noteheads sit exactly on lines/in spaces.
  * Real triad spacing. Three noteheads a third apart = 1.0S vertically,
    so the chord stacks the way an engraver would set it.
  * Harmonic wave frequencies. The low notehead gets the long wavelength
    and the high notehead the short one, in 1 : 3/2 : 2 ratios -- the
    harmonic series, so a musician reads it as one chord, not stripes.
  * Audio envelope. Amplitude swells from zero at the notehead (the wave
    has to *start* as a flat stem) and decays as it runs right, like a
    struck note. Uniform sine waves are what made the old mark read as
    clip art.

Outputs (assets/icons/):
    mark_dark.svg / mark_light.svg        -- chord + waves, no text
    lockup_dark.svg / lockup_light.svg    -- mark + "stemma" wordmark
    lockup_stacked_{dark,light}.svg       -- portrait lockup (9:16 art)
    icon_square.svg                       -- app icon (rounded tile)
    icon_{16,32,48,64,128,256}.png, stemma.ico
"""

from __future__ import annotations

import math
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT)

_OUT = os.path.join(_ROOT, "assets", "icons")

# -- Palette ---------------------------------------------------------------
# Stem colors are the app's own (src/ui/styles.py) so the logo and the
# mixer speak the same language.
INK_DARK = "#cdd6f4"     # staff / notehead on dark
INK_LIGHT = "#4c4f69"    # staff / notehead on light
BG_DARK = "#1e1e2e"
BG_LIGHT = "#eff1f5"

WAVE_DARK = ("#4fb8b8", "#e4ad6e", "#bfa3dc")   # teal, gold, purple
WAVE_LIGHT = ("#3da8a8", "#c89040", "#9878b8")

# -- Engraving geometry ----------------------------------------------------
S = 100.0                 # staff space
NOTE_RX = 0.66 * S        # ellipse semi-major
NOTE_RY = 0.335 * S       # semi-minor: noteheads are flat, not round
                          # (rotated, the ink spans ~1.18S x 1.0S)
NOTE_ANGLE = -20.0        # engraved noteheads tilt up to the right
STEM_W = 0.13 * S         # stem / wave stroke weight
STAFF_W = 0.09 * S        # staff rule weight

WAVE_LEN = 9.2 * S        # how far the stems run out in "time"
WAVE_AMP = 0.62 * S       # peak amplitude of the lowest wave
BASE_CYCLES = 1.55        # cycles across WAVE_LEN for the lowest wave
HARMONICS = (1.0, 1.5, 2.0)   # root : fifth : octave


def _wave_path(x0: float, y0: float, cycles: float, amp: float,
               length: float = WAVE_LEN, steps: int = 240) -> str:
    """Polyline path for one stem-turned-waveform.

    Starts flat at (x0, y0) -- the notehead edge -- so the stroke reads
    as a stem before it becomes audio. Amplitude follows a swell/decay
    envelope: it ramps in over the first fifth of the run, then falls
    away like a struck note.
    """
    pts = []
    for i in range(steps + 1):
        t = i / steps
        attack = min(1.0, t / 0.12)          # flat at the notehead, swells
        decay = math.exp(-1.05 * t)          # then rings out (to ~0.35)
        env = attack * decay
        x = x0 + t * length
        y = y0 - math.sin(2 * math.pi * cycles * t) * amp * env
        pts.append((x, y))
    head = f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"
    rest = " ".join(f"L {x:.2f} {y:.2f}" for x, y in pts[1:])
    return f"{head} {rest}"


def _notehead(cx: float, cy: float, fill: str) -> str:
    return (
        f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" '
        f'rx="{NOTE_RX:.2f}" ry="{NOTE_RY:.2f}" fill="{fill}" '
        f'transform="rotate({NOTE_ANGLE} {cx:.2f} {cy:.2f})"/>'
    )


def _smoothstep(a: float, b: float, t: float) -> float:
    if t <= a:
        return 0.0
    if t >= b:
        return 1.0
    u = (t - a) / (b - a)
    return u * u * (3.0 - 2.0 * u)


def _strand_path(sx: float, sy: float, lane_y: float, cycles: float,
                 amp: float, length: float, steps: int = 260) -> str:
    """One waveform strand fanning out of the shared stem.

    All strands leave the same point (the stem tip), ease into their own
    lane over the first third of the run, and oscillate with a swell/
    decay envelope. Emerging from one point is the whole idea: a single
    stem becoming several.
    """
    pts = []
    for i in range(steps + 1):
        t = i / steps
        centre = sy + (lane_y - sy) * _smoothstep(0.0, 0.34, t)
        env = _smoothstep(0.04, 0.38, t) * math.exp(-0.75 * t)
        x = sx + t * length
        y = centre - math.sin(2 * math.pi * cycles * t) * amp * env
        pts.append((x, y))
    head = f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"
    return head + " " + " ".join(f"L {x:.2f} {y:.2f}" for x, y in pts[1:])


def build_mark(dark: bool = True, with_staff: bool = True) -> tuple[str, float, float]:
    """Return (svg_body, width, height) for the chord-to-waves mark.

    A root-position triad written the way an engraver would: three
    noteheads a third apart sharing ONE stem. The stem rises from the
    top notehead, then turns and unravels into three colored waveforms.
    Correct notation and the exact metaphor in one move -- one stem
    becomes several.
    """
    ink = INK_DARK if dark else INK_LIGHT
    waves = WAVE_DARK if dark else WAVE_LIGHT

    pad = 0.85 * S
    note_cx = pad + NOTE_RX
    stem_x = note_cx + NOTE_RX * math.cos(math.radians(abs(NOTE_ANGLE)))

    wave_len = 7.4 * S
    lane_gap = 1.05 * S
    max_amp = 0.46 * S
    stem_len = 2.15 * S

    # Lay the geometry out from the top down so nothing can fall outside
    # the viewBox: the highest ink is the top strand's crest.
    stem_top = pad + lane_gap + max_amp
    # Bottom-to-top, matching the notehead order below, so each
    # strand leaves in line with the notehead sharing its color.
    lanes = [stem_top + lane_gap, stem_top, stem_top - lane_gap]

    top_note_y = stem_top + stem_len
    # Triad: 1.0S between notehead centers (a third apart on the staff).
    ys = [top_note_y + 2.0 * S, top_note_y + 1.0 * S, top_note_y]  # low->high
    staff_ys = ys

    width = stem_x + wave_len + pad
    height = ys[0] + NOTE_RY + pad

    parts: list[str] = []

    if with_staff:
        # Just the three rules the chord sits on: enough staff to read as
        # notation without a full five-line stave competing with the waves.
        for y in staff_ys:
            parts.append(
                f'<line x1="{pad * 0.42:.2f}" y1="{y:.2f}" '
                f'x2="{stem_x + NOTE_RX * 0.55:.2f}" y2="{y:.2f}" '
                f'stroke="{ink}" stroke-width="{STAFF_W:.2f}" '
                f'stroke-opacity="0.30" stroke-linecap="round"/>'
            )

    # The shared stem, from the bottom notehead up to the fan point.
    parts.append(
        f'<line x1="{stem_x:.2f}" y1="{ys[0]:.2f}" '
        f'x2="{stem_x:.2f}" y2="{stem_top:.2f}" '
        f'stroke="{ink}" stroke-width="{STEM_W:.2f}" stroke-linecap="round"/>'
    )

    # Strands, low harmonic to high, each in a stem color.
    for lane, harmonic, color in zip(lanes, HARMONICS, waves):
        amp = (0.46 * S) / (harmonic ** 0.35)
        parts.append(
            f'<path d="{_strand_path(stem_x, stem_top, lane, BASE_CYCLES * harmonic, amp, wave_len)}" '
            f'fill="none" stroke="{color}" stroke-width="{STEM_W:.2f}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )

    # Noteheads last so they sit cleanly over the stem's foot.
    for y, color in zip(ys, waves):
        parts.append(_notehead(note_cx, y, color))

    return "\n  ".join(parts), width, height


def _svg(body: str, width: float, height: float) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width:.2f} {height:.2f}" '
        f'width="{width:.0f}" height="{height:.0f}">\n  {body}\n</svg>\n'
    )


# -- Wordmark --------------------------------------------------------------
# Palatino Linotype Italic: a humanist chancery italic (Zapf), named for
# the 16th-century Italian calligrapher Palatino. Italic is how musical
# terms are set, and "stemma" is itself Italian -- so the face carries
# the idea rather than just spelling the name. Converted to outlines so
# the SVG renders identically without the font installed.

def wordmark_path(text: str = "stemma", px: float = 400.0) -> tuple[str, float, float]:
    """Return (path_d, width, height) of *text* as outlines."""
    from PySide6.QtGui import QFont, QFontDatabase, QPainterPath
    from PySide6.QtWidgets import QApplication

    _ = QApplication.instance() or QApplication([])

    # Load the face from its file rather than by family name: on the
    # offscreen platform a name lookup silently substitutes a fallback
    # (which renders as boxes), and asset generation must be
    # reproducible in CI.
    family = "Palatino Linotype"
    for candidate in (r"C:\Windows\Fonts\palai.ttf",):
        if os.path.isfile(candidate):
            fid = QFontDatabase.addApplicationFont(candidate)
            fams = QFontDatabase.applicationFontFamilies(fid)
            if fams:
                family = fams[0]
            break

    font = QFont(family)
    font.setItalic(True)
    font.setPixelSize(int(px))
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 102.0)

    path = QPainterPath()
    path.addText(0.0, 0.0, font, text)

    d: list[str] = []
    i = 0
    n = path.elementCount()
    while i < n:
        e = path.elementAt(i)
        if e.isMoveTo():
            d.append(f"M {e.x:.2f} {e.y:.2f}")
            i += 1
        elif e.isLineTo():
            d.append(f"L {e.x:.2f} {e.y:.2f}")
            i += 1
        elif e.isCurveTo():
            c1 = path.elementAt(i)
            c2 = path.elementAt(i + 1)
            ep = path.elementAt(i + 2)
            d.append(
                f"C {c1.x:.2f} {c1.y:.2f} {c2.x:.2f} {c2.y:.2f} "
                f"{ep.x:.2f} {ep.y:.2f}"
            )
            i += 3
        else:
            i += 1
    rect = path.boundingRect()
    return " ".join(d) + " Z", rect.width(), rect.height()


def build_lockup(dark: bool = True) -> str:
    """Mark on the left, wordmark on the right, optically aligned."""
    ink = INK_DARK if dark else INK_LIGHT
    mark_body, mw, mh = build_mark(dark=dark, with_staff=True)

    d, ww, wh = wordmark_path("stemma", px=400.0)
    from PySide6.QtGui import QPainterPath  # noqa: PLC0415  (path bounds)
    # Scale the wordmark so its x-height era matches the chord height.
    target_h = mh * 0.30
    scale = target_h / wh
    gap = mw * 0.045
    # The path is drawn from a baseline at y=0 with negative-up glyphs;
    # shift so its bounding box starts at the origin before scaling.
    from PySide6.QtGui import QFont, QPainterPath as _P  # noqa: F401
    # Recompute bounds to place the glyphs.
    import re as _re
    xs = [float(v) for v in _re.findall(r"-?\d+\.\d+", d)][0::2]
    ys = [float(v) for v in _re.findall(r"-?\d+\.\d+", d)][1::2]
    minx, miny = min(xs), min(ys)

    wx = mw + gap
    wy = mh * 0.50 - target_h / 2  # optical centering against the chord
    width = wx + ww * scale + mw * 0.06
    height = mh

    return _svg(
        f"{mark_body}\n"
        f'  <g transform="translate({wx:.2f} {wy:.2f}) scale({scale:.5f}) '
        f'translate({-minx:.2f} {-miny:.2f})">\n'
        f'    <path d="{d}" fill="{ink}"/>\n'
        f"  </g>",
        width,
        height,
    )


def build_lockup_stacked(dark: bool = True) -> str:
    """Portrait lockup: mark above, wordmark centred below.

    For 9:16 store art, where the horizontal lockup would either sit in a
    sea of empty space or have to be scaled down to illegibility.
    """
    ink = INK_DARK if dark else INK_LIGHT
    mark_body, mw, mh = build_mark(dark=dark, with_staff=True)

    d, ww, wh = wordmark_path("stemma", px=400.0)
    import re as _re
    nums = [float(v) for v in _re.findall(r"-?\d+\.\d+", d)]
    minx, miny = min(nums[0::2]), min(nums[1::2])

    target_w = mw * 0.72
    scale = target_w / ww
    gap = mh * 0.16
    width = mw
    height = mh + gap + wh * scale
    wx = (width - target_w) / 2.0

    return _svg(
        f"{mark_body}\n"
        f'  <g transform="translate({wx:.2f} {mh + gap:.2f}) scale({scale:.5f}) '
        f'translate({-minx:.2f} {-miny:.2f})">\n'
        f'    <path d="{d}" fill="{ink}"/>\n'
        f"  </g>",
        width,
        height,
    )


def build_icon(dark: bool = True) -> str:
    """App icon: a single note whose stem frays into colored strands.

    The triad loses its spacing below ~48px, so the icon reduces to one
    quarter note -- which reads as "music" even at 16px -- with the stem
    tip fraying into three colored strands to carry the brand idea. The
    note itself stays ink-colored so it reads as notation; the color
    lives in the strands, matching the mixer.
    """
    bg = BG_DARK if dark else BG_LIGHT
    ink = INK_DARK if dark else INK_LIGHT
    waves = WAVE_DARK if dark else WAVE_LIGHT
    size = 512.0
    r = size * 0.215          # tile corner radius

    # Proper engraved notehead proportions, scaled to the tile.
    nrx, nry = size * 0.150, size * 0.076
    cx, cy = size * 0.35, size * 0.70
    sw = size * 0.050         # stroke weight: holds together at 16px
    stem_x = cx + nrx * math.cos(math.radians(abs(NOTE_ANGLE)))
    stem_top = size * 0.30
    x_end = size * 0.83
    spread = size * 0.150

    parts = [f'<rect width="{size}" height="{size}" rx="{r:.1f}" fill="{bg}"/>']

    # Stem.
    parts.append(
        f'<line x1="{stem_x:.1f}" y1="{cy:.1f}" x2="{stem_x:.1f}" '
        f'y2="{stem_top:.1f}" stroke="{ink}" stroke-width="{sw:.1f}" '
        f'stroke-linecap="round"/>'
    )
    # Three strands fraying off the stem tip: one becomes many.
    for i, color in enumerate(waves):
        dy = (i - 1) * spread
        parts.append(
            f'<path d="M {stem_x:.1f} {stem_top:.1f} '
            f'C {stem_x + size * 0.16:.1f} {stem_top:.1f}, '
            f'{stem_x + size * 0.16:.1f} {stem_top + dy:.1f}, '
            f'{x_end:.1f} {stem_top + dy:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="{sw:.1f}" '
            f'stroke-linecap="round"/>'
        )
    parts.append(
        f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{nrx:.1f}" ry="{nry:.1f}" '
        f'fill="{ink}" transform="rotate({NOTE_ANGLE} {cx:.1f} {cy:.1f})"/>'
    )
    return _svg("\n  ".join(parts), size, size)


def _rasterize(svg_path: str, out_path: str, size: int) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtWidgets import QApplication

    _ = QApplication.instance() or QApplication([])
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    QSvgRenderer(svg_path).render(p)
    p.end()
    img.save(out_path)


def main() -> None:
    os.makedirs(_OUT, exist_ok=True)
    written: list[str] = []

    for dark, suffix in ((True, "dark"), (False, "light")):
        body, w, h = build_mark(dark=dark)
        p = os.path.join(_OUT, f"mark_{suffix}.svg")
        with open(p, "w", encoding="utf-8") as f:
            f.write(_svg(body, w, h))
        written.append(p)

        p = os.path.join(_OUT, f"lockup_{suffix}.svg")
        with open(p, "w", encoding="utf-8") as f:
            f.write(build_lockup(dark=dark))
        written.append(p)

        p = os.path.join(_OUT, f"lockup_stacked_{suffix}.svg")
        with open(p, "w", encoding="utf-8") as f:
            f.write(build_lockup_stacked(dark=dark))
        written.append(p)

    icon_svg = os.path.join(_OUT, "icon_square.svg")
    with open(icon_svg, "w", encoding="utf-8") as f:
        f.write(build_icon(dark=True))
    written.append(icon_svg)

    for size in (16, 32, 48, 64, 128, 256):
        p = os.path.join(_OUT, f"icon_{size}.png")
        _rasterize(icon_svg, p, size)
        written.append(p)

    # Multi-resolution .ico for the exe / window icon.
    from PIL import Image

    frames = [
        Image.open(os.path.join(_OUT, f"icon_{s}.png"))
        for s in (16, 32, 48, 64, 128, 256)
    ]
    ico = os.path.join(_OUT, "stemma.ico")
    frames[-1].save(ico, format="ICO",
                    sizes=[(s, s) for s in (16, 32, 48, 64, 128, 256)])
    written.append(ico)

    for p in written:
        print("wrote", os.path.relpath(p, _ROOT))


if __name__ == "__main__":
    main()

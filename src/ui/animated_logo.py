"""Animated main logo widget for stemma.

Renders the main logo with a short procedural intro: the four Cmaj7
noteheads light up near-simultaneously, each wave is drawn out of its
note, and the waves breathe once or twice before settling. Clickable as an
Easter egg to replay with the chord sound.

The brand SVG (``assets/icons/logo_main_{dark,light}.svg``) is the single
source of the artwork. The noteheads and waves are read from it and
animated; every other element is rendered by QSvgRenderer, in document
order, so the stem still crosses the noteheads and the resting frame is
the brand logo itself.
"""

import logging
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from xml.etree.ElementTree import Element

from PySide6.QtCore import QByteArray, QElapsedTimer, QPointF, QRectF, Qt
from PySide6.QtCore import QTimer
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QTransform,
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QWidget

from src.paths import app_root
from src.ui.audio_sync import LOGO_AUDIO_VISUAL_LAG_MS
from src.ui.svg_source import (
    local_name,
    read_svg,
    to_svg_text,
    with_children,
)
from src.ui.wav_playback import play_wav_async

_ROOT = app_root()
_AUDIO_PATH = os.path.join(_ROOT, "assets", "audio", "chord.wav")

# Widget width in pixels. The height follows the SVG's aspect ratio, so the
# logo is scaled uniformly and never stretched.
_W = 300

_NOTE_SPACING_MS = 40
_FADE_MS = 150
_BOUNCE_MS = 150
_BOUNCE_HEIGHT = 4.0  # SVG units; the staff spacing is 15
_WAVE_GROW_MS = 250
_UNDULATE_START_MS = 300
_UNDULATE_RAMP_MS = 150
_DAMPEN_START_MS = 800
_ANIM_END_MS = 1500
_FRAME_MS = 33

# The undulation scales each wave's height about its staff line by up to
# this fraction, out of phase between neighbouring waves.
_UNDULATE_DEPTH = 0.25
_UNDULATE_PERIOD_MS = 600
_UNDULATE_PHASE_STEP = 1.5  # radians

_CAPS = {
    "butt": Qt.PenCapStyle.FlatCap,
    "round": Qt.PenCapStyle.RoundCap,
    "square": Qt.PenCapStyle.SquareCap,
}
_JOINS = {
    "miter": Qt.PenJoinStyle.SvgMiterJoin,
    "round": Qt.PenJoinStyle.RoundJoin,
    "bevel": Qt.PenJoinStyle.BevelJoin,
}
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_PATH_TOKEN = re.compile(rf"[A-Za-z]|{_NUMBER}")
_TRANSFORM = re.compile(r"(\w+)\s*\(([^)]*)\)")


@dataclass(frozen=True)
class _Note:
    """One notehead: a filled ellipse with its SVG transform."""

    cx: float
    cy: float
    rx: float
    ry: float
    transform: QTransform
    color: str
    opacity: float = 1.0


@dataclass(frozen=True)
class _Wave:
    """One wave: a stroked polyline starting on its note's staff line."""

    points: tuple[tuple[float, float], ...]
    lengths: tuple[float, ...]  # cumulative arc length at each point
    color: str
    width: float
    cap: Qt.PenCapStyle
    join: Qt.PenJoinStyle
    opacity: float = 1.0


@dataclass(frozen=True)
class _Logo:
    """The main logo split into animated parts and static SVG layers.

    ``notes[i]`` and ``waves[i]`` are one voice, bottom (root) note first.
    ``paint_order`` lists ``("static", layer)``, ``("note", voice)`` and
    ``("wave", voice)`` in SVG document order.
    """

    view_box: tuple[float, float, float, float]
    notes: tuple[_Note, ...]
    waves: tuple[_Wave, ...]
    static_layers: tuple[str, ...]
    paint_order: tuple[tuple[str, int], ...]


def _logo_path(theme: str) -> str:
    variant = "dark" if theme == "dark" else "light"
    return os.path.join(_ROOT, "assets", "icons", f"logo_main_{variant}.svg")


def _is_note(element: Element) -> bool:
    return local_name(element) == "ellipse"


def _is_wave(element: Element) -> bool:
    """A stroked, unfilled path: a wave, never a glyph of the wordmark."""
    return (
        local_name(element) == "path"
        and element.get("fill") == "none"
        and element.get("stroke") not in (None, "none")
    )


def _parse_polyline(d: str) -> list[tuple[float, float]]:
    """Points of an SVG path made of move and line commands only.

    Raises:
        ValueError: If the path uses any other command (a curve, say):
            the waves are animated point by point, so they must be
            polylines.
    """
    points: list[tuple[float, float]] = []
    command = ""
    numbers: list[float] = []

    def flush() -> None:
        nonlocal command
        if len(numbers) % 2:
            raise ValueError(f"odd coordinate count in path: {d[:40]!r}")
        for i in range(0, len(numbers), 2):
            x, y = numbers[i], numbers[i + 1]
            if command.islower() and points:
                x, y = points[-1][0] + x, points[-1][1] + y
            points.append((x, y))
        numbers.clear()

    for token in _PATH_TOKEN.findall(d):
        if token.isalpha():
            flush()
            if token not in "MmLl":
                raise ValueError(
                    f"wave paths must be a polyline (M/L only), got {token!r}"
                )
            command = token
        else:
            if not command:
                raise ValueError(f"path does not start with M: {d[:40]!r}")
            numbers.append(float(token))
    flush()
    return points


def _parse_transform(text: str) -> QTransform:
    """A QTransform for an SVG ``transform`` attribute.

    Supports ``translate``, ``scale``, ``rotate`` (optionally about a
    point) and ``matrix``, which covers what the brand SVGs use.

    Raises:
        ValueError: For any other transform function.
    """
    result = QTransform()
    for name, raw in _TRANSFORM.findall(text or ""):
        args = [float(v) for v in re.findall(_NUMBER, raw)]
        step = QTransform()
        if name == "translate":
            step.translate(args[0], args[1] if len(args) > 1 else 0.0)
        elif name == "scale":
            step.scale(args[0], args[1] if len(args) > 1 else args[0])
        elif name == "rotate":
            if len(args) == 3:
                step.translate(args[1], args[2])
                step.rotate(args[0])
                step.translate(-args[1], -args[2])
            else:
                step.rotate(args[0])
        elif name == "matrix":
            step = QTransform(*args)
        else:
            raise ValueError(f"unsupported SVG transform: {name}")
        # SVG applies the rightmost function first; Qt composes row
        # vectors, so the step goes on the left.
        result = step * result
    return result


def _opacity(element: Element, paint_opacity: str) -> float:
    """Element ``opacity`` times its fill or stroke opacity, 0 to 1."""
    value = 1.0
    for name in ("opacity", paint_opacity):
        raw = element.get(name)
        if raw is not None:
            value *= min(1.0, max(0.0, float(raw)))
    return value


def _parse_note(element: Element) -> _Note:
    return _Note(
        cx=float(element.get("cx", 0)),
        cy=float(element.get("cy", 0)),
        rx=float(element.get("rx", 0)),
        ry=float(element.get("ry", 0)),
        transform=_parse_transform(element.get("transform", "")),
        color=element.get("fill", "#000000"),
        opacity=_opacity(element, "fill-opacity"),
    )


def _parse_wave(element: Element) -> _Wave:
    points = _parse_polyline(element.get("d", ""))
    if len(points) < 2:
        raise ValueError("a wave needs at least two points")
    lengths = [0.0]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.hypot(x1 - x0, y1 - y0))
    return _Wave(
        points=tuple(points),
        lengths=tuple(lengths),
        color=element.get("stroke", "#000000"),
        width=float(element.get("stroke-width", 1)),
        # A value Qt has no match for ("inherit" at the top level, say)
        # falls back to the SVG default, as QSvgRenderer does.
        cap=_CAPS.get(element.get("stroke-linecap"), _CAPS["butt"]),
        join=_JOINS.get(element.get("stroke-linejoin"), _JOINS["miter"]),
        opacity=_opacity(element, "stroke-opacity"),
    )


@lru_cache(maxsize=None)
def _load_logo(theme: str) -> _Logo:
    """Split the theme's logo SVG into animated parts and static layers.

    Top-level noteheads (ellipses) and waves (stroked, unfilled paths) are
    parsed for animation. Runs of other elements become static layers,
    each a full SVG document, painted between the parts in document order.
    Each wave is paired with the note on whose staff line it starts.
    """
    root = read_svg(_logo_path(theme))
    view_box = tuple(float(v) for v in root.get("viewBox", "").split())
    if len(view_box) != 4:
        raise ValueError(f"logo SVG needs a viewBox: {_logo_path(theme)}")

    # Walk the top level in document order, keeping animated parts in
    # place and merging each run of other elements into one layer.
    layers: list[list[Element]] = []
    sequence: list[tuple[str, object]] = []
    for child in root:
        if _is_note(child):
            sequence.append(("note", _parse_note(child)))
        elif _is_wave(child):
            sequence.append(("wave", _parse_wave(child)))
        else:
            if not sequence or sequence[-1][0] != "static":
                layers.append([])
                sequence.append(("static", len(layers) - 1))
            layers[-1].append(child)

    if not any(kind == "note" for kind, _ in sequence):
        # Nested notes (say, wrapped in a <g>) would otherwise leave a
        # logo that silently never animates.
        raise ValueError(
            "no top-level noteheads (<ellipse>) in the logo SVG: "
            f"{_logo_path(theme)}"
        )

    # Voices run by pitch, bottom note first; each wave belongs to the
    # note whose staff line it starts on.
    notes = sorted(
        (part for kind, part in sequence if kind == "note"),
        key=lambda note: -note.cy,
    )
    waves: list[_Wave | None] = [None] * len(notes)
    for kind, wave in sequence:
        if kind != "wave":
            continue
        if not notes:
            raise ValueError("logo SVG has waves but no noteheads")
        start_y = wave.points[0][1]
        index = min(
            range(len(notes)), key=lambda i: abs(notes[i].cy - start_y),
        )
        if waves[index] is not None:
            raise ValueError(f"two waves start on the line y={start_y}")
        waves[index] = wave
    if None in waves:
        raise ValueError("every notehead in the logo SVG needs a wave")

    def voice(part: object) -> int:
        group = notes if isinstance(part, _Note) else waves
        return next(i for i, item in enumerate(group) if item is part)

    return _Logo(
        view_box=view_box,
        notes=tuple(notes),
        waves=tuple(waves),
        static_layers=tuple(
            to_svg_text(with_children(root, layer)) for layer in layers
        ),
        paint_order=tuple(
            (kind, part if kind == "static" else voice(part))
            for kind, part in sequence
        ),
    )


def _note_alpha(elapsed: int, onset: int) -> float:
    """Opacity ramp for a note appearing at *onset*."""
    if elapsed < onset:
        return 0.0
    return min(1.0, (elapsed - onset) / _FADE_MS)


def _bounce_y(elapsed: int, onset: int) -> float:
    """Vertical bounce offset in SVG units (ease-out, returns to 0)."""
    age = elapsed - onset
    if age < 0 or age > _BOUNCE_MS:
        return 0.0
    return -_BOUNCE_HEIGHT * (1.0 - age / _BOUNCE_MS) ** 2


def _wave_reveal(elapsed: int, onset: int) -> float:
    """Share of the wave drawn so far (ease-out), 0 to 1."""
    progress = (elapsed - onset) / _WAVE_GROW_MS
    if progress <= 0.0:
        return 0.0
    if progress >= 1.0:
        return 1.0
    return 1.0 - (1.0 - progress) ** 3


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _wave_swell(elapsed: int, index: int) -> float:
    """Height scale of wave *index*: 1 at rest, breathing in between.

    Eases in from the undulation start (no jump), holds, then dampens to
    exactly 1 at the end of the animation, so the last frame is the SVG.
    """
    if elapsed <= _UNDULATE_START_MS or elapsed >= _ANIM_END_MS:
        return 1.0
    age = elapsed - _UNDULATE_START_MS
    envelope = _smoothstep(age / _UNDULATE_RAMP_MS) * (
        1.0
        - _smoothstep(
            (elapsed - _DAMPEN_START_MS) / (_ANIM_END_MS - _DAMPEN_START_MS)
        )
    )
    phase = 2.0 * math.pi * age / _UNDULATE_PERIOD_MS
    return 1.0 + _UNDULATE_DEPTH * envelope * math.sin(
        phase + index * _UNDULATE_PHASE_STEP
    )


def _wave_path(wave: _Wave, reveal: float, swell: float) -> QPainterPath:
    """The wave drawn up to *reveal* of its length, height scaled by *swell*.

    The scaling is about the wave's staff line (its first point), so the
    wave stays anchored to its note.
    """
    base_y = wave.points[0][1]

    def shaped(x: float, y: float) -> QPointF:
        return QPointF(x, base_y + (y - base_y) * swell)

    limit = reveal * wave.lengths[-1]
    path = QPainterPath(shaped(*wave.points[0]))
    for i in range(1, len(wave.points)):
        x, y = wave.points[i]
        if wave.lengths[i] <= limit:
            path.lineTo(shaped(x, y))
            continue
        px, py = wave.points[i - 1]
        span = wave.lengths[i] - wave.lengths[i - 1]
        f = (limit - wave.lengths[i - 1]) / span if span > 0 else 0.0
        path.lineTo(shaped(px + (x - px) * f, py + (y - py) * f))
        break
    return path


def _fit(view_box: tuple[float, float, float, float], width: int,
         height: int) -> tuple[QRectF, QTransform]:
    """Where the view box lands in a widget, and the matching transform.

    The logo is scaled uniformly and centred, like the SVG's default
    ``preserveAspectRatio``.
    """
    vx, vy, vw, vh = view_box
    scale = min(width / vw, height / vh)
    left = (width - vw * scale) / 2.0
    top = (height - vh * scale) / 2.0
    transform = QTransform()
    transform.translate(left, top)
    transform.scale(scale, scale)
    transform.translate(-vx, -vy)
    return QRectF(left, top, vw * scale, vh * scale), transform


class AnimatedLogoWidget(QWidget):
    """Main stemma logo with animated notes and waves."""

    def __init__(self, theme: str = "dark", play_sound: bool = True) -> None:
        super().__init__()
        self._is_dark = theme == "dark"
        self._play_sound = play_sound
        self._logo = _load_logo(theme)
        _, _, view_w, view_h = self._logo.view_box
        self.setFixedSize(_W, round(_W * view_h / view_w))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._layer_pixmaps: list[QPixmap] = []
        self._layer_dpr = 0.0
        # The resting frame, cached: repaints at rest (hover, resize,
        # dialog moves) then cost one pixmap blit, not four restrokes.
        self._rest_pixmap: QPixmap | None = None

        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self.update)
        self._animating = False
        self._static_t = _ANIM_END_MS

    # -- public API ----------------------------------------------------------

    def play_intro(self, with_sound: bool = True) -> None:
        """Start (or restart) the note/wave animation."""
        self._static_t = _ANIM_END_MS
        self._clock.restart()
        self._animating = True
        self._timer.start(_FRAME_MS)
        self.update()
        if with_sound and self._play_sound:
            self._do_play_sound()

    def set_theme(self, theme: str) -> None:
        """Switch to the logo artwork and colours for *theme*."""
        self._is_dark = theme == "dark"
        self._logo = _load_logo(theme)
        self._layer_pixmaps = []
        self._rest_pixmap = None
        self.update()

    def set_play_sound(self, enabled: bool) -> None:
        """Toggle whether click-to-replay produces sound."""
        self._play_sound = enabled

    # -- events --------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.play_intro(with_sound=True)
        else:
            super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        t = self._clock.elapsed() if self._clock.isValid() else self._static_t
        if self._animating:
            t_draw = max(0, t - LOGO_AUDIO_VISUAL_LAG_MS)
        else:
            t_draw = t
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint_frame(p, t_draw)
        p.end()

        if self._animating and t_draw >= _ANIM_END_MS:
            self._animating = False
            self._timer.stop()
            self.update()

    # -- internal helpers ----------------------------------------------------

    def _paint_frame(self, p: QPainter, t: int) -> None:
        """Paint the logo as it looks *t* ms into the intro."""
        if t >= _ANIM_END_MS:
            p.drawPixmap(0, 0, self._rest_frame())
        else:
            self._paint_parts(p, t)

    def _rest_frame(self) -> QPixmap:
        """The final frame (the brand logo) for the current theme and DPR."""
        dpr = self.devicePixelRatioF()
        pix = self._rest_pixmap
        if pix is not None and pix.devicePixelRatio() == dpr:
            return pix
        pix = self._blank_pixmap(dpr)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint_parts(painter, _ANIM_END_MS)
        painter.end()
        self._rest_pixmap = pix
        return pix

    def _blank_pixmap(self, dpr: float) -> QPixmap:
        pix = QPixmap(
            math.ceil(self.width() * dpr), math.ceil(self.height() * dpr),
        )
        pix.setDevicePixelRatio(dpr)
        pix.fill(Qt.GlobalColor.transparent)
        return pix

    def _paint_parts(self, p: QPainter, t: int) -> None:
        """Paint static layers, notes and waves as they are at *t* ms."""
        _, view = _fit(self._logo.view_box, self.width(), self.height())
        pixmaps = self._static_pixmaps()
        for kind, index in self._logo.paint_order:
            if kind == "static":
                p.drawPixmap(0, 0, pixmaps[index])
                continue
            onset = index * _NOTE_SPACING_MS
            alpha = _note_alpha(t, onset)
            if alpha <= 0.0:
                continue
            p.save()
            p.setWorldTransform(view)
            if kind == "note":
                self._paint_note(p, self._logo.notes[index], alpha,
                                 _bounce_y(t, onset))
            else:
                self._paint_wave(p, self._logo.waves[index], alpha,
                                 _wave_reveal(t, onset),
                                 _wave_swell(t, index))
            p.restore()

    @staticmethod
    def _paint_note(p: QPainter, note: _Note, alpha: float,
                    lift: float) -> None:
        color = QColor(note.color)
        color.setAlphaF(alpha * note.opacity)
        p.translate(0.0, lift)
        p.setTransform(note.transform, True)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawEllipse(QPointF(note.cx, note.cy), note.rx, note.ry)

    @staticmethod
    def _paint_wave(p: QPainter, wave: _Wave, alpha: float, reveal: float,
                    swell: float) -> None:
        if reveal <= 0.0:
            return
        color = QColor(wave.color)
        color.setAlphaF(alpha * wave.opacity)
        pen = QPen(color, wave.width)
        pen.setCapStyle(wave.cap)
        pen.setJoinStyle(wave.join)
        pen.setMiterLimit(4.0)  # the SVG default
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(_wave_path(wave, reveal, swell))

    def _static_pixmaps(self) -> list[QPixmap]:
        """The static layers rendered for the current theme and DPR."""
        dpr = self.devicePixelRatioF()
        if self._layer_pixmaps and self._layer_dpr == dpr:
            return self._layer_pixmaps
        target, _ = _fit(self._logo.view_box, self.width(), self.height())
        pixmaps = []
        for layer in self._logo.static_layers:
            renderer = QSvgRenderer(QByteArray(layer.encode("utf-8")))
            pix = self._blank_pixmap(dpr)
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            renderer.render(painter, target)
            painter.end()
            pixmaps.append(pix)
        self._layer_pixmaps = pixmaps
        self._layer_dpr = dpr
        return pixmaps

    def _do_play_sound(self) -> None:
        if not os.path.isfile(_AUDIO_PATH):
            logging.getLogger(__name__).warning(
                "logo sound missing: %s", _AUDIO_PATH
            )
            return
        try:
            play_wav_async(_AUDIO_PATH)
        except Exception:
            # Never let a sound failure break the animation, but do not
            # swallow it silently either: a bare pass here hid the
            # click-plays-no-sound reports, because the Easter egg looked
            # like it worked while playback had failed outright.
            logging.getLogger(__name__).exception(
                "logo sound playback failed for %s", _AUDIO_PATH
            )

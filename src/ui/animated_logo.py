"""Animated main logo widget for stemma.

Renders the stemma logo with procedural animation: all four Cmaj7 chord
notes light up near-simultaneously, waves grow and gently undulate.
Clickable as an Easter egg to replay with chord sound.
"""

import math
import os

from PySide6.QtCore import QByteArray, QElapsedTimer, QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QWidget

from src.paths import app_root
from src.ui.audio_sync import LOGO_AUDIO_VISUAL_LAG_MS
from src.ui.wav_playback import play_wav_async

_ROOT = app_root()
_AUDIO_PATH = os.path.join(_ROOT, "assets", "audio", "chord.wav")

_W, _H = 300, 185

_NOTE_SPACING_MS = 40
_FADE_MS = 150
_BOUNCE_MS = 150
_WAVE_GROW_MS = 250
_UNDULATE_START_MS = 300
_DAMPEN_START_MS = 800
_ANIM_END_MS = 1500
_FRAME_MS = 33

# Notes in animation order (bottom-to-top: C, E, G, B).
# (cy, dark_color, light_color, wave_amplitude, wave_segments, wave_stroke_w)
_NOTES = [
    (123, "#4fb8b8", "#3da8a8", 17, 4, 2.2),
    (108, "#d4849a", "#c0707e", 11, 8, 1.8),
    (93, "#e4ad6e", "#c89040", 13, 6, 1.8),
    (78, "#bfa3dc", "#9878b8", 9, 10, 1.8),
]

_NOTE_CX = 94.0
_NOTE_RX, _NOTE_RY = 10.5, 6.5
_NOTE_ANGLE = -20.0
_WAVE_X0 = 105.0
_WAVE_LEN = 120.0


def _load_base_svg(theme: str) -> str:
    """Read the main logo SVG, stripping note ellipses and wave paths."""
    variant = "dark" if theme == "dark" else "light"
    path = os.path.join(_ROOT, "assets", "icons", f"logo_main_{variant}.svg")
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    return "".join(
        ln
        for ln in lines
        if not ln.strip().startswith("<ellipse")
        and not ln.strip().startswith('<path d="M105,')
    )


def _note_alpha(elapsed: int, onset: int) -> float:
    """Opacity ramp for a note appearing at *onset*."""
    if elapsed < onset:
        return 0.0
    return min(1.0, (elapsed - onset) / _FADE_MS)


def _bounce_y(elapsed: int, onset: int) -> float:
    """Vertical bounce offset (ease-out, returns to 0)."""
    age = elapsed - onset
    if age < 0 or age > _BOUNCE_MS:
        return 0.0
    return -4.0 * (1.0 - age / _BOUNCE_MS) ** 2


class AnimatedLogoWidget(QWidget):
    """Main stemma logo with animated notes and waves."""

    def __init__(self, theme: str = "dark", play_sound: bool = True) -> None:
        super().__init__()
        self.setFixedSize(_W, _H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._is_dark = theme == "dark"
        self._play_sound = play_sound
        self._base_pixmap = self._render_base(theme)
        self._colors = self._resolve_colors()

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
        """Rebuild the base pixmap and note colors for *theme*."""
        self._is_dark = theme == "dark"
        self._base_pixmap = self._render_base(theme)
        self._colors = self._resolve_colors()
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
        p.drawPixmap(0, 0, self._base_pixmap)

        for i, (cy, _, _, amp, segs, sw) in enumerate(_NOTES):
            onset = i * _NOTE_SPACING_MS
            alpha = _note_alpha(t_draw, onset)
            if alpha <= 0.0:
                continue

            color = QColor(self._colors[i])
            color.setAlphaF(min(alpha, 1.0))

            by = _bounce_y(t_draw, onset)

            _draw_note(p, cy + by, color)
            _draw_wave(p, i, cy, color, t_draw, onset, amp, segs, sw)

        if self._animating and t_draw >= _ANIM_END_MS:
            self._animating = False
            self._timer.stop()
            self.update()

        p.end()

    # -- internal helpers ----------------------------------------------------

    def _do_play_sound(self) -> None:
        if not os.path.isfile(_AUDIO_PATH):
            return
        try:
            play_wav_async(_AUDIO_PATH)
        except Exception:
            pass

    @staticmethod
    def _render_base(theme: str) -> QPixmap:
        svg = _load_base_svg(theme)
        renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
        dpr = 1.0
        screen = QApplication.primaryScreen()
        if screen:
            dpr = screen.devicePixelRatio()
        pw, ph = int(_W * dpr), int(_H * dpr)
        pix = QPixmap(pw, ph)
        pix.setDevicePixelRatio(dpr)
        pix.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pix)
        renderer.render(painter)
        painter.end()
        return pix

    def _resolve_colors(self) -> list[str]:
        return [dk if self._is_dark else lt for _, dk, lt, *_ in _NOTES]


# -- free drawing helpers (no self needed) -----------------------------------


def _draw_note(p: QPainter, cy: float, color: QColor) -> None:
    p.save()
    p.translate(_NOTE_CX, cy)
    p.rotate(_NOTE_ANGLE)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    p.drawEllipse(QPointF(0, 0), _NOTE_RX, _NOTE_RY)
    p.restore()


def _draw_wave(
    p: QPainter,
    idx: int,
    cy: float,
    color: QColor,
    t: int,
    onset: int,
    amp: float,
    segs: int,
    sw: float,
) -> None:
    grow = min(1.0, max(0.0, (t - onset) / _WAVE_GROW_MS))
    if grow <= 0.0:
        return

    phase = 0.0
    if t > _UNDULATE_START_MS:
        age = t - _UNDULATE_START_MS
        dampen = 1.0
        if t > _DAMPEN_START_MS:
            dampen = max(
                0.0,
                1.0
                - (t - _DAMPEN_START_MS) / (_ANIM_END_MS - _DAMPEN_START_MS),
            )
        phase = math.sin(age * 0.004 + idx * 1.5) * amp * 0.25 * dampen

    eff_amp = amp * grow
    half_w = _WAVE_LEN / segs

    path = QPainterPath()
    path.moveTo(_WAVE_X0, cy)
    for s in range(segs):
        sign = -1.0 if s % 2 == 0 else 1.0
        cx = _WAVE_X0 + s * half_w + half_w / 2.0
        c_y = cy + sign * eff_amp + phase
        ex = _WAVE_X0 + (s + 1) * half_w
        path.quadTo(cx, c_y, ex, cy)

    pen = QPen(color, sw)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)

"""Animated footer arpeggio logo widget for stemma.

Renders the arpeggio logo (staff, bass clef, coloured "stemma" letters)
with animation: letters glow sequentially matching the Am7 arpeggio.
Clickable as an Easter egg to replay with sound.
"""

import os

from PySide6.QtCore import QByteArray, QElapsedTimer, QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QWidget

from src.paths import app_root
from src.ui.audio_sync import LOGO_AUDIO_VISUAL_LAG_MS
from src.ui.wav_playback import play_wav_async

_ROOT = app_root()
_AUDIO_PATH = os.path.join(_ROOT, "assets", "audio", "arpeggio.wav")

_W, _H = 220, 36

_SVG_W, _SVG_H = 420, 75
_SVG_Y0 = 8
_SX = _W / _SVG_W
_SY = _H / _SVG_H

_NOTE_SPACING_MS = 280
_FADE_MS = 200
_GLOW_DUR_MS = 300
_ANIM_END_MS = 5 * _NOTE_SPACING_MS + _FADE_MS + _GLOW_DUR_MS
_FRAME_MS = 33

# Qt SFX output lags the animation clock slightly; pull visuals back to match.
_AUDIO_VISUAL_LAG_MS = 48

# (char, svg_x, svg_y, dark_color, light_color)
_LETTERS = [
    ("s", 95, 82, "#4fb8b8", "#3da8a8"),
    ("t", 155, 67, "#d4849a", "#c0707e"),
    ("e", 215, 52, "#e4ad6e", "#c89040"),
    ("m", 275, 37, "#bfa3dc", "#9878b8"),
    ("m", 335, 52, "#e4ad6e", "#c89040"),
    ("a", 395, 67, "#d4849a", "#c0707e"),
]


def _load_base_svg(theme: str) -> str:
    """Read the arpeggio SVG, stripping text elements (animated)."""
    variant = "dark" if theme == "dark" else "light"
    path = os.path.join(
        _ROOT, "assets", "icons", f"logo_arpeggio_{variant}.svg"
    )
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    return "".join(ln for ln in lines if not ln.strip().startswith("<text"))


def _brightness(elapsed: int, onset: int) -> float:
    """0 to 1 ramp starting at *onset* over _FADE_MS."""
    if elapsed < onset:
        return 0.0
    return min(1.0, (elapsed - onset) / _FADE_MS)


def _glow(elapsed: int, onset: int) -> float:
    """Glow pulse: 1 to 0 decay after the letter fully appears."""
    age = elapsed - onset - _FADE_MS
    if age < 0 or age > _GLOW_DUR_MS:
        return 0.0
    return 1.0 - age / _GLOW_DUR_MS


class AnimatedArpeggioWidget(QWidget):
    """Footer arpeggio logo with animated letter glow."""

    def __init__(self, theme: str = "dark", play_sound: bool = True) -> None:
        super().__init__()
        self.setFixedSize(_W, _H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("border: none;")

        self._is_dark = theme == "dark"
        self._play_sound = play_sound
        self._base_pixmap = self._render_base(theme)
        self._font = self._build_font()

        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self.update)
        self._animating = False

    # -- public API ----------------------------------------------------------

    def play_intro(self, with_sound: bool = True) -> None:
        """Start (or restart) the letter glow animation."""
        if with_sound and self._play_sound:
            self._do_play_sound()
        self._clock.restart()
        self._animating = True
        self._timer.start(_FRAME_MS)
        self.update()

    def set_theme(self, theme: str) -> None:
        """Rebuild base pixmap for *theme*."""
        self._is_dark = theme == "dark"
        self._base_pixmap = self._render_base(theme)
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
        t = self._clock.elapsed() if self._clock.isValid() else _ANIM_END_MS
        t_anim = max(0, t - LOGO_AUDIO_VISUAL_LAG_MS)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        p.drawPixmap(0, 0, self._base_pixmap)

        in_anim = self._animating or (
            self._clock.isValid() and t_anim < _ANIM_END_MS
        )

        p.save()
        p.scale(_SX, _SY)
        p.translate(0, -_SVG_Y0)
        p.setFont(self._font)

        for i, (char, sx, sy, dark_c, light_c) in enumerate(_LETTERS):
            onset = i * _NOTE_SPACING_MS
            b = _brightness(t_anim, onset)
            g = _glow(t_anim, onset)
            alpha = (0.4 + 0.6 * b) if in_anim else 1.0

            hex_c = dark_c if self._is_dark else light_c
            color = QColor(hex_c)
            color.setAlphaF(min(alpha, 1.0))

            tw = p.fontMetrics().horizontalAdvance(char)

            if g > 0.0:
                p.save()
                p.translate(sx, sy)
                p.scale(1.0 + 0.15 * g, 1.0 + 0.15 * g)
                gc = QColor(hex_c)
                gc.setAlphaF(0.3 * g)
                p.setPen(gc)
                p.drawText(QPointF(-tw / 2.0, 0), char)
                p.restore()

            p.setPen(color)
            p.drawText(QPointF(sx - tw / 2.0, sy), char)

        p.restore()

        if self._animating and t_anim >= _ANIM_END_MS:
            self._animating = False
            self._timer.stop()
            self.update()

        p.end()

    # -- internal helpers ----------------------------------------------------

    def _do_play_sound(self) -> None:
        if not os.path.isfile(_AUDIO_PATH):
            return
        play_wav_async(_AUDIO_PATH)

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

    @staticmethod
    def _build_font() -> QFont:
        font = QFont()
        font.setFamilies(
            ["Palatino Linotype", "Palatino", "Georgia", "serif"]
        )
        font.setPixelSize(30)
        font.setItalic(True)
        font.setWeight(QFont.Weight.DemiBold)
        return font

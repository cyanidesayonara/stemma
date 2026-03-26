"""Animated splash screen with arpeggio logo for stemma startup.

Shows the stemma arpeggio logo (staff lines, bass clef, coloured letters)
with each letter fading in sequentially while a Cmaj7 arpeggio plays.
Displayed before heavy module imports so the user sees immediate feedback.
"""

import math
import os

from PySide6.QtCore import (
    QByteArray,
    QElapsedTimer,
    QPointF,
    QPropertyAnimation,
    QRectF,
    Qt,
    QTimer,
)
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QWidget

from src.ui.styles import DARK_COLORS, LIGHT_COLORS

try:
    import winsound

    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False

_BASS_CLEF_PATH = (
    "M252 262Q122 257 61 187Q-1 116 0 39Q0 -23 31 -65Q62 -108 123 -110"
    "Q171 -109 200 -80Q228 -51 229 -4Q228 45 199 72Q171 100 133 100"
    "Q113 100 103 97Q92 93 83 93Q73 93 70 98Q67 103 67 111Q68 144 110 183"
    "Q152 221 229 224Q308 222 345 152Q381 82 381 -37Q380 -243 283 -376"
    "Q186 -508 10 -605Q3 -609 -1 -613Q-5 -617 -5 -623Q-5 -628 -2 -631"
    "Q1 -635 8 -635Q16 -635 25 -630Q218 -542 370 -398Q522 -254 531 -28"
    "Q530 104 456 182Q383 260 252 262ZM629 180Q605 179 590 164Q575 149"
    " 574 125Q575 101 590 86Q605 71 629 70Q653 71 668 86Q683 101 684 125"
    "Q683 149 668 164Q653 179 629 180ZM630 -71Q606 -71 591 -86Q576 -101"
    " 576 -125Q576 -149 591 -164Q606 -179 630 -179Q654 -179 669 -164"
    "Q684 -149 684 -125Q684 -101 669 -86Q654 -71 630 -71Z"
)

_LETTERS = [
    ("s", 95, 82, "#4fb8b8", "#3da8a8"),
    ("t", 155, 67, "#d4849a", "#c0707e"),
    ("e", 215, 52, "#e4ad6e", "#c89040"),
    ("m", 275, 37, "#bfa3dc", "#9878b8"),
    ("m", 335, 52, "#e4ad6e", "#c89040"),
    ("a", 395, 67, "#d4849a", "#c0707e"),
]

_SPLASH_W = 600
_SPLASH_H = 240
_SCALE = 1.333
_VB_Y = 8
_LOGO_X = (_SPLASH_W - 420 * _SCALE) / 2
_LOGO_Y = 55.0

_CLEF_DELAY_MS = 200
_NOTE_SPACING_MS = 280
_FADE_MS = 200
_MIN_DISPLAY_MS = 1800
_ANIM_END_MS = _CLEF_DELAY_MS + 5 * _NOTE_SPACING_MS + _FADE_MS
_FRAME_MS = 33


def _make_base_svg(line_color: str, clef_color: str) -> str:
    lines = "\n".join(
        f'  <line x1="12" y1="{y}" x2="408" y2="{y}" '
        f'stroke="{line_color}" stroke-width="1.0" opacity="0.35"/>'
        for y in (20, 35, 50, 65, 80)
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="420" height="75" viewBox="0 8 420 75">\n'
        f"{lines}\n"
        f'  <g transform="translate(16,35) scale(0.048,-0.048)" '
        f'fill="{clef_color}">\n'
        f'    <path d="{_BASS_CLEF_PATH}"/>\n'
        "  </g>\n"
        "</svg>"
    )


class SplashScreen(QWidget):
    """Animated splash shown during application startup."""

    def __init__(
        self,
        theme: str = "dark",
        play_sound: bool = True,
        audio_path: str | None = None,
    ) -> None:
        super().__init__(None)
        flags = (
            Qt.WindowType.SplashScreen
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setWindowFlags(flags)
        self.setFixedSize(_SPLASH_W, _SPLASH_H)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(
                geo.x() + (geo.width() - _SPLASH_W) // 2,
                geo.y() + (geo.height() - _SPLASH_H) // 2,
            )

        self._is_dark = theme == "dark"
        self._play_sound = play_sound and _HAS_WINSOUND
        self._audio_path = audio_path

        c = DARK_COLORS if self._is_dark else LIGHT_COLORS
        self._bg_color = QColor(c["base"])
        self._text_color = QColor(c["text"])
        self._border_color = QColor(c["surface0"])

        self._base_pixmap = self._render_base()
        self._font = self._build_logo_font()

        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self.update)

        self._main_window: QWidget | None = None
        self._fade_anim: QPropertyAnimation | None = None
        self._finishing = False

    def start(self) -> None:
        """Show the splash and begin the animation + optional sound."""
        self._clock.start()
        self._timer.start(_FRAME_MS)
        self.show()
        QApplication.processEvents()

        if (
            self._play_sound
            and self._audio_path
            and os.path.isfile(self._audio_path)
        ):
            winsound.PlaySound(
                self._audio_path,
                winsound.SND_ASYNC | winsound.SND_FILENAME,
            )

    def finish(self, main_window: QWidget) -> None:
        """Transition from splash to *main_window*.

        Waits for the minimum display time so the animation plays through,
        then fades out the splash and shows the main window.  Safe to call
        more than once; subsequent calls are ignored.
        """
        if self._finishing:
            return
        self._finishing = True
        self._main_window = main_window

        if not self.isVisible():
            main_window.show()
            return

        elapsed = self._clock.elapsed() if self._clock.isValid() else _MIN_DISPLAY_MS
        remaining = max(0, _MIN_DISPLAY_MS - elapsed)
        if remaining > 0:
            QTimer.singleShot(remaining, self._begin_fade_out)
        else:
            self._begin_fade_out()

    def _begin_fade_out(self) -> None:
        if self._fade_anim is not None:
            return
        self._timer.stop()
        anim = QPropertyAnimation(self, b"windowOpacity")
        anim.setDuration(250)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.finished.connect(self._on_fade_done)
        self._fade_anim = anim
        anim.start()

    def _on_fade_done(self) -> None:
        if self._main_window is not None:
            self._main_window.show()
        self.close()

    # -- painting -------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        elapsed = self._clock.elapsed() if self._clock.isValid() else 0
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        p.fillRect(self.rect(), self._bg_color)
        p.setPen(self._border_color)
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))

        p.drawPixmap(int(_LOGO_X), int(_LOGO_Y), self._base_pixmap)

        p.setFont(self._font)
        for i, (char, sx, sy, dark_c, light_c) in enumerate(_LETTERS):
            alpha = self._letter_alpha(i, elapsed)
            if alpha <= 0.0:
                continue
            color = QColor(dark_c if self._is_dark else light_c)
            color.setAlphaF(min(alpha, 1.0))
            p.setPen(color)
            px = _LOGO_X + sx * _SCALE
            py = _LOGO_Y + (sy - _VB_Y) * _SCALE
            tw = p.fontMetrics().horizontalAdvance(char)
            p.drawText(QPointF(px - tw / 2.0, py), char)

        if elapsed > _ANIM_END_MS:
            pulse = 0.35 + 0.15 * math.sin(elapsed / 400.0)
            lc = QColor(self._text_color)
            lc.setAlphaF(pulse)
            p.setPen(lc)
            p.setFont(QFont("Segoe UI", 10))
            text_rect = QRectF(
                0,
                _LOGO_Y + 75 * _SCALE + 25,
                _SPLASH_W,
                30,
            )
            p.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                "Loading\u2026",
            )

        p.end()

    @staticmethod
    def _letter_alpha(index: int, elapsed_ms: int) -> float:
        onset = _CLEF_DELAY_MS + index * _NOTE_SPACING_MS
        if elapsed_ms < onset:
            return 0.0
        return min(1.0, (elapsed_ms - onset) / _FADE_MS)

    # -- helpers --------------------------------------------------------

    def _render_base(self) -> QPixmap:
        line_color = "#b4bcd0" if self._is_dark else "#555555"
        clef_color = line_color
        svg_xml = _make_base_svg(line_color, clef_color)

        renderer = QSvgRenderer(QByteArray(svg_xml.encode("utf-8")))
        dpr = 1.0
        screen = QApplication.primaryScreen()
        if screen:
            dpr = screen.devicePixelRatio()
        pw = int(420 * _SCALE * dpr)
        ph = int(75 * _SCALE * dpr)
        pixmap = QPixmap(pw, ph)
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        return pixmap

    @staticmethod
    def _build_logo_font() -> QFont:
        font = QFont()
        font.setFamilies(
            ["Palatino Linotype", "Palatino", "Georgia", "serif"]
        )
        font.setPixelSize(int(30 * _SCALE))
        font.setItalic(True)
        font.setWeight(QFont.Weight.DemiBold)
        return font

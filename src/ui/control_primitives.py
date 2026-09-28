"""Shared drawing and sizing primitives for player-control widgets."""

import math
import re

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QComboBox,
    QPushButton,
    QWidget,
    QSpinBox,
    QStyle,
    QStyleOptionSpinBox,
)

from src.ui.styles import ICON_BUTTON_SIZES, ON_ACCENT

ICON_SIZE = 24
STEM_ICON_SIZE = 18

# Three button sizes, one per kind of row (#186): the transport's primary
# buttons, controls that sit beside 30 px fields in the practice rack, and
# the compact rows of the mixer and library.
TRANSPORT_BUTTON = 36
ROW_BUTTON = 30
ROW_ICON_SIZE = 20
COMPACT_BUTTON = 28

_CHECKED_ICON_COLOR = QColor(ON_ACCENT)
# A disabled glyph is its normal color at this opacity, so it recedes toward
# whatever sits behind it in either theme. Qt's generated disabled pixmap
# blends toward white instead, which made light dark-theme glyphs brighter.
DISABLED_ICON_OPACITY = 0.35


def fix_button_size(button: QPushButton, width: int, height: int) -> None:
    """Give *button* a fixed size the stylesheet cannot override.

    Square sizes listed in ``ICON_BUTTON_SIZES`` are also pinned by a
    stylesheet rule on the button's "squareSize" property, so no polish (theme
    switch, or a dynamic-property repolish) can bring back the 30 px
    minimum of the global QPushButton rule.
    """
    if width == height and width in ICON_BUTTON_SIZES:
        button.setProperty("squareSize", str(width))
    button.setFixedSize(width, height)


def make_display_combo(combo: QComboBox) -> None:
    """Make an editable combo act as a read-only display."""
    combo.setEditable(True)
    line_edit = combo.lineEdit()
    line_edit.setReadOnly(True)
    line_edit.installEventFilter(combo)
    original_mouse = line_edit.mousePressEvent

    def open_on_click(event):  # noqa: ANN001
        if event.button() == Qt.MouseButton.LeftButton:
            combo.showPopup()
        else:
            original_mouse(event)

    line_edit.mousePressEvent = open_on_click


def repolish(widget: QWidget) -> None:
    """Re-apply the stylesheet after a dynamic property changed."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def add_volume_presets(combo: QComboBox) -> None:
    """Fill a volume display combo with 200% down to 0%, showing 100%.

    Highest first, like a fader, so Down lowers the volume: Down selects
    the next item in the list.
    """
    for value in range(200, -1, -20):
        combo.addItem(f"{value}%", value)
    show_preset_value(combo, 100)


def show_preset_value(combo: QComboBox, value: int) -> None:
    """Show *value* in a preset display combo, with the index in step.

    Setting only the edit text left the index at whatever was last picked
    (the first item, at startup), so Up, Down, or the wheel jumped from
    that stale preset: 100% plus Down gave 20% (#209 review). Between
    presets the nearest one is selected, so a step goes to a neighbour.
    """
    presets = [combo.itemData(i) for i in range(combo.count())]
    if presets:
        nearest = min(range(len(presets)),
                      key=lambda i: abs(presets[i] - value))
        combo.blockSignals(True)
        combo.setCurrentIndex(nearest)
        combo.blockSignals(False)
    combo.setEditText(f"{value}%")


def fit_spinbox_width(spin: QSpinBox, sample: str | None = None) -> None:
    """Size a spinbox to its widest expected display value."""
    if sample is None:
        sample = f"{spin.prefix()}{spin.maximum()}{spin.suffix()}"
    metrics = spin.fontMetrics()
    content = QSize(metrics.horizontalAdvance(sample) + 8, metrics.height())
    option = QStyleOptionSpinBox()
    spin.initStyleOption(option)
    hint = spin.style().sizeFromContents(
        QStyle.ContentsType.CT_SpinBox,
        option,
        content,
        spin,
    )
    spin.setFixedWidth(hint.width() + 4)


def fit_combo_width(combo: QComboBox, extra: int = 0) -> None:
    """Size a combo box to its widest entry."""
    metrics = combo.fontMetrics()
    widest = max(
        (metrics.horizontalAdvance(combo.itemText(index))
         for index in range(combo.count())),
        default=0,
    )
    # Padding (6 left, 18 right for the chevron) plus frame and slack.
    combo.setFixedWidth(widest + 48 + extra)


def _paint_glyph(draw_fn, color: QColor, size: int) -> QPixmap:
    """Paint ``draw_fn`` in *color* into a transparent square pixmap."""
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    draw_fn(painter, size)
    painter.end()
    return pixmap


def add_disabled_pixmaps(icon: QIcon, draw_fn, color: QColor,
                         size: int) -> None:
    """Give *icon* a dimmed glyph for the disabled mode, checked or not.

    Both states use the normal color: a disabled button loses its accent
    fill, so the dark checked-state glyph would vanish on the disabled fill.
    """
    dimmed = QColor(color)
    dimmed.setAlphaF(color.alphaF() * DISABLED_ICON_OPACITY)
    pixmap = _paint_glyph(draw_fn, dimmed, size)
    for state in (QIcon.State.Off, QIcon.State.On):
        icon.addPixmap(pixmap, QIcon.Mode.Disabled, state)


def make_icon(draw_fn, color: QColor, size: int = ICON_SIZE) -> QIcon:
    """Create a crisp icon by painting ``draw_fn`` into a pixmap."""
    icon = QIcon(_paint_glyph(draw_fn, color, size))
    add_disabled_pixmaps(icon, draw_fn, color, size)
    return icon


def make_toggle_icon(draw_fn, normal_color: QColor,
                     size: int = ICON_SIZE,
                     checked_color: QColor | None = None) -> QIcon:
    """Create an icon with distinct normal and checked pixmaps.

    *checked_color* is the glyph colour on the accent fill: the theme's
    ``on_accent`` (it differs between themes).
    """
    icon = QIcon()
    for color, state in (
        (normal_color, QIcon.State.Off),
        (checked_color or _CHECKED_ICON_COLOR, QIcon.State.On),
    ):
        pixmap = _paint_glyph(draw_fn, color, size)
        for mode in (
            QIcon.Mode.Normal,
            QIcon.Mode.Active,
            QIcon.Mode.Selected,
        ):
            icon.addPixmap(pixmap, mode, state)
    add_disabled_pixmaps(icon, draw_fn, normal_color, size)
    return icon


def draw_play(painter: QPainter, size: int) -> None:
    margin = int(size * 0.2)
    painter.drawPolygon(QPolygonF([
        QPointF(margin + 2, margin),
        QPointF(size - margin, size / 2),
        QPointF(margin + 2, size - margin),
    ]))


def draw_pause(painter: QPainter, size: int) -> None:
    margin = int(size * 0.22)
    width = int(size * 0.18)
    painter.drawRect(margin, margin, width, size - 2 * margin)
    painter.drawRect(
        size - margin - width,
        margin,
        width,
        size - 2 * margin,
    )


def draw_stop(painter: QPainter, size: int) -> None:
    margin = int(size * 0.22)
    painter.drawRect(margin, margin, size - 2 * margin, size - 2 * margin)


def draw_record(painter: QPainter, size: int) -> None:
    center = size / 2.0
    radius = size * 0.30
    painter.drawEllipse(QPointF(center, center), radius, radius)


def draw_mute(painter: QPainter, size: int) -> None:
    """Draw a speaker with an X."""
    margin = size * 0.18
    body_width = size * 0.16
    body_height = size * 0.28
    # Speaker sits left and the X starts clear of the cone: at 18 px the
    # X's rounded cap used to touch the cone (#186).
    body_x = size * 0.10
    body_y = size / 2.0 - body_height / 2.0
    painter.drawRect(QRectF(body_x, body_y, body_width, body_height))
    cone_x = body_x + body_width
    cone_width = size * 0.20
    painter.drawPolygon(QPolygonF([
        QPointF(cone_x, body_y),
        QPointF(cone_x + cone_width, margin),
        QPointF(cone_x + cone_width, size - margin),
        QPointF(cone_x, body_y + body_height),
    ]))
    pen = QPen(painter.brush().color(), size * 0.09)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    x_start, x_end = size * 0.62, size * 0.88
    painter.drawLine(
        QPointF(x_start, margin * 1.3),
        QPointF(x_end, size - margin * 1.3),
    )
    painter.drawLine(
        QPointF(x_start, size - margin * 1.3),
        QPointF(x_end, margin * 1.3),
    )
    painter.setPen(Qt.PenStyle.NoPen)


def draw_solo(painter: QPainter, size: int) -> None:
    """Draw a headphones icon."""
    margin = size * 0.15
    pen = QPen(painter.brush().color(), size * 0.09)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.drawArc(
        QRectF(margin, margin, size - 2 * margin, size - 2 * margin),
        30 * 16,
        120 * 16,
    )
    painter.setPen(Qt.PenStyle.NoPen)
    cup_width = size * 0.18
    cup_height = size * 0.30
    cup_y = size * 0.52
    painter.drawRoundedRect(
        QRectF(margin, cup_y, cup_width, cup_height), 2, 2,
    )
    painter.drawRoundedRect(
        QRectF(size - margin - cup_width, cup_y, cup_width, cup_height),
        2,
        2,
    )


def draw_power(painter: QPainter, size: int) -> None:
    """Draw a universal power icon."""
    center_x = size / 2.0
    center_y = size / 2.0 + size * 0.08
    radius = size * 0.28
    stroke = max(1.5, size * 0.09)
    pen = QPen(painter.brush().color(), stroke)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.drawArc(
        QRectF(
            center_x - radius,
            center_y - radius,
            2 * radius,
            2 * radius,
        ),
        125 * 16,
        290 * 16,
    )
    line_top = max(stroke * 0.5, center_y - radius - size * 0.06)
    painter.drawLine(
        QPointF(center_x, line_top),
        QPointF(center_x, center_y - size * 0.04),
    )
    painter.setPen(Qt.PenStyle.NoPen)


def draw_trash(painter: QPainter, size: int) -> None:
    """Draw a trash can icon."""
    margin = size * 0.2
    pen = QPen(painter.brush().color(), max(1.0, size * 0.08))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setBrush(Qt.GlobalColor.transparent)
    painter.setPen(pen)
    painter.drawLine(
        QPointF(margin * 0.8, margin * 1.5),
        QPointF(size - margin * 0.8, margin * 1.5),
    )
    painter.drawRect(QRectF(size * 0.4, margin, size * 0.2, margin * 0.5))
    painter.drawRect(
        QRectF(
            margin * 1.2,
            margin * 1.5,
            size - margin * 2.4,
            size - margin * 2.5,
        )
    )
    painter.setPen(Qt.PenStyle.NoPen)


def draw_repeat(painter: QPainter, size: int) -> None:
    """Draw cycle arrows."""
    center = size / 2.0
    margin = size * 0.20
    radius = (size - 2 * margin) / 2.0
    pen = QPen(painter.brush().color(), size * 0.09)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    arc_rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
    painter.drawArc(arc_rect, 20 * 16, 140 * 16)
    painter.drawArc(arc_rect, 200 * 16, 140 * 16)
    painter.setPen(Qt.PenStyle.NoPen)
    arrow = size * 0.12
    angle = math.radians(20)
    x_pos = center + radius * math.cos(angle)
    y_pos = center - radius * math.sin(angle)
    painter.drawPolygon(QPolygonF([
        QPointF(x_pos + arrow, y_pos - arrow * 0.6),
        QPointF(x_pos - arrow * 0.3, y_pos - arrow * 0.8),
        QPointF(x_pos, y_pos + arrow * 0.5),
    ]))
    angle = math.radians(200)
    x_pos = center + radius * math.cos(angle)
    y_pos = center - radius * math.sin(angle)
    painter.drawPolygon(QPolygonF([
        QPointF(x_pos - arrow, y_pos + arrow * 0.6),
        QPointF(x_pos + arrow * 0.3, y_pos + arrow * 0.8),
        QPointF(x_pos, y_pos - arrow * 0.5),
    ]))


def format_time(seconds: float) -> str:
    """Format seconds as mm:ss."""
    minutes = int(seconds) // 60
    remaining = int(seconds) % 60
    return f"{minutes}:{remaining:02d}"


class PitchSpinBox(QSpinBox):
    """Stable-width spinbox with human-readable semitone values."""

    _WIDEST_TEXT = "+7 semi (10/10)"

    def _compute_fixed_hint(self) -> QSize:
        metrics = self.fontMetrics()
        content = QSize(
            metrics.horizontalAdvance(self._WIDEST_TEXT) + 16,
            metrics.height(),
        )
        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        return self.style().sizeFromContents(
            QStyle.ContentsType.CT_SpinBox,
            option,
            content,
            self,
        )

    def sizeHint(self) -> QSize:  # noqa: D401, N802
        return self._compute_fixed_hint()

    def minimumSizeHint(self) -> QSize:  # noqa: D401, N802
        return self._compute_fixed_hint()

    def textFromValue(self, value: int) -> str:  # noqa: D401, N802
        if value == 0:
            return "original"
        sign = "+" if value > 0 else "-"
        return f"{sign}{abs(value)} semi"

    def valueFromText(self, text: str) -> int:  # noqa: D401, N802
        match = re.match(r"[+-]?\d+", text.strip())
        if match is None:
            return 0
        try:
            return int(match.group())
        except ValueError:
            return 0

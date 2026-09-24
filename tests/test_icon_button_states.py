"""Icon buttons must show hover, pressed, and disabled states.

``QPushButton#icon-btn`` sets its own background, and an ID selector
outranks ``QPushButton:hover`` / ``:pressed`` / ``:disabled``. So no
unchecked icon button in the app reacted to the pointer, and a disabled one
(Record while speed or pitch is changed) looked exactly like an enabled one.
"""

import pytest
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QPushButton,
    QStyle,
    QStyleOptionButton,
    QWidget,
)

from src.ui.control_primitives import draw_power, make_toggle_icon
from src.ui.styles import DARK_COLORS, LIGHT_COLORS, get_stylesheet


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _hosted_button(theme):
    """An icon button under the real app sheet, set on a host widget."""
    colors = DARK_COLORS if theme == "dark" else LIGHT_COLORS
    host = QWidget()
    host.setStyleSheet(get_stylesheet(theme))
    layout = QHBoxLayout(host)
    button = QPushButton()
    button.setObjectName("icon-btn")
    button.setCheckable(True)
    button.setFixedSize(36, 36)
    button.setIcon(make_toggle_icon(draw_power, QColor(colors["text"])))
    layout.addWidget(button)
    host.show()
    QApplication.processEvents()
    return host, button, colors


def _fill(button) -> str:
    """Background just inside the border, clear of the icon glyph."""
    return button.grab().toImage().pixelColor(4, 18).name()


def _hover_fill(button) -> str:
    """Paint the button as the style would under the pointer.

    Offscreen Qt never delivers a real hover (a plain QPushButton's working
    :hover rule does not show either), so set State_MouseOver on the style
    option, which is exactly what the stylesheet engine matches :hover on.
    """
    option = QStyleOptionButton()
    button.initStyleOption(option)
    option.state |= QStyle.StateFlag.State_MouseOver
    image = QImage(button.size(), QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    button.style().drawControl(
        QStyle.ControlElement.CE_PushButton, option, painter, button,
    )
    painter.end()
    return image.pixelColor(4, 18).name()


def _icon_pixel(button) -> QColor:
    """A pixel on the power glyph's stem, near the top of the icon."""
    return button.grab().toImage().pixelColor(18, 11)


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_disabled_icon_button_looks_disabled(app, theme):
    host, button, colors = _hosted_button(theme)
    enabled_fill = _fill(button)
    enabled_icon = _icon_pixel(button)

    button.setEnabled(False)
    QApplication.processEvents()

    assert enabled_fill == QColor(colors["surface0"]).name()
    assert _fill(button) == QColor(colors["base"]).name()
    # The glyph dims too, toward the background, not just the fill.
    dimmed = _icon_pixel(button)
    base = QColor(colors["base"])

    def distance(a, b):
        return sum(abs(x - y) for x, y in zip(a.getRgb()[:3], b.getRgb()[:3]))

    assert distance(dimmed, base) < distance(enabled_icon, base)
    host.close()


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_pressed_icon_button_shows_it(app, theme):
    host, button, colors = _hosted_button(theme)

    button.setDown(True)
    QApplication.processEvents()

    assert _fill(button) == QColor(colors["surface2"]).name()
    host.close()


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_hovered_icon_button_shows_it(app, theme):
    host, button, colors = _hosted_button(theme)

    assert _hover_fill(button) == QColor(colors["surface1"]).name()
    host.close()


def test_checked_icon_button_keeps_its_accent(app):
    """The new rules must not override the checked fill."""
    host, button, colors = _hosted_button("dark")

    button.setChecked(True)
    QApplication.processEvents()

    assert _hover_fill(button) == QColor(colors["accent"]).name()
    host.close()

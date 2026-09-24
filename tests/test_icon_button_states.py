"""Icon buttons must show hover, pressed, and disabled states.

``QPushButton#icon-btn`` sets its own background, and an ID selector
outranks ``QPushButton:hover`` / ``:pressed`` / ``:disabled``. So no
unchecked icon button in the app reacted to the pointer, and a disabled one
(Record while speed or pitch is changed) looked exactly like an enabled one.
"""

from importlib import import_module

import pytest
from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QIcon, QImage, QPainter
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
    # The app's size: at Qt's 16px default the glyph is too small to measure.
    button.setIconSize(QSize(24, 24))
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


def _luminance(color: QColor) -> float:
    def channel(value):
        c = value / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(v) for v in color.getRgb()[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: QColor, b: QColor) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _glyph_contrast(button) -> float:
    """WCAG contrast of the glyph against the button's own fill.

    Takes the icon-area pixel that differs most from the fill, so it does
    not depend on where a particular glyph happens to have ink.
    """
    image = button.grab().toImage()
    fill = image.pixelColor(4, 18)
    size = button.iconSize().width()
    x0 = (button.width() - size) // 2
    y0 = (button.height() - size) // 2
    return max(
        _contrast(image.pixelColor(x, y), fill)
        for x in range(x0, x0 + size)
        for y in range(y0, y0 + size)
    )


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_disabled_icon_button_looks_disabled(app, theme):
    host, button, colors = _hosted_button(theme)
    enabled_fill = _fill(button)
    enabled_contrast = _glyph_contrast(button)

    button.setEnabled(False)
    QApplication.processEvents()

    assert enabled_fill == QColor(colors["surface0"]).name()
    assert _fill(button) == QColor(colors["base"]).name()
    # The glyph must recede too. Qt's generated disabled icon blends toward
    # white, which made light dark-theme glyphs brighter when disabled.
    disabled_contrast = _glyph_contrast(button)
    assert disabled_contrast < enabled_contrast
    assert disabled_contrast < 3.0
    host.close()


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_checked_disabled_icon_button_reads_as_disabled(app, theme):
    """Disabled wins over checked: no accent fill, and a receding glyph."""
    host, button, colors = _hosted_button(theme)
    button.setChecked(True)
    button.setEnabled(False)
    QApplication.processEvents()

    assert _fill(button) == QColor(colors["base"]).name()
    assert _glyph_contrast(button) < 3.0
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


@pytest.mark.parametrize("builder", [
    "control_primitives.make_icon",
    "control_primitives.make_toggle_icon",
    "library_panel._make_icon",
    "library_panel._make_toggle_icon",
])
def test_every_icon_builder_dims_its_disabled_glyph(app, builder):
    """The library panel keeps its own icon builders; they must match."""
    module_name, func_name = builder.split(".")
    module = import_module(f"src.ui.{module_name}")
    icon = getattr(module, func_name)(draw_power, QColor("#cdd6f4"))

    def peak_alpha(mode):
        image = icon.pixmap(QSize(24, 24), mode).toImage()
        return max(
            image.pixelColor(x, y).alpha()
            for x in range(image.width())
            for y in range(image.height())
        )

    normal = peak_alpha(QIcon.Mode.Normal)
    disabled = peak_alpha(QIcon.Mode.Disabled)
    assert disabled < normal * 0.5

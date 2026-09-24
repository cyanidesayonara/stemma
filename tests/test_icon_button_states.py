"""Icon buttons must show hover, pressed, and disabled states.

``QPushButton#icon-btn`` sets its own background, and an ID selector
outranks ``QPushButton:hover`` / ``:pressed`` / ``:disabled``. So no
unchecked icon button in the app reacted to the pointer, and a disabled one
(Record while speed or pitch is changed) looked exactly like an enabled one.
"""

from importlib import import_module
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QSize
from PySide6.QtGui import (
    QAccessible,
    QAccessibleActionInterface,
    QColor,
    QIcon,
    QImage,
    QPainter,
)
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QPushButton,
    QStyle,
    QStyleOptionButton,
    QWidget,
)

from src.ui.control_primitives import draw_power, make_toggle_icon
from src.ui.library_panel import REPEAT_ALL, REPEAT_OFF, LibraryPanel
from src.ui.styles import DARK_COLORS, LIGHT_COLORS, get_stylesheet


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _hosted_button(theme):
    """An icon button under the real app sheet, set on a host widget."""
    colors = DARK_COLORS if theme == "dark" else LIGHT_COLORS
    host = QWidget()
    layout = QHBoxLayout(host)
    button = QPushButton()
    button.setObjectName("icon-btn")
    button.setCheckable(True)
    button.setFixedSize(36, 36)
    button.setIcon(make_toggle_icon(draw_power, QColor(colors["text"])))
    # The app's size: at Qt's 16px default the glyph is too small to measure.
    button.setIconSize(QSize(24, 24))
    layout.addWidget(button)
    _show_styled(host, theme)
    return host, button, colors


def _show_styled(host, theme) -> None:
    """Show *host*, then apply the app sheet so every child re-polishes.

    Set before the children were shown, CI occasionally sampled a button
    still unstyled (Qt's default grey).
    """
    host.show()
    QApplication.processEvents()
    host.setStyleSheet(get_stylesheet(theme))
    for widget in host.findChildren(QWidget):
        widget.ensurePolished()
    QApplication.processEvents()


def _fill(button) -> str:
    """Background just inside the border, clear of the icon glyph."""
    return button.grab().toImage().pixelColor(4, 18).name()


def _render_in_state(
    button, state, without=QStyle.StateFlag.State_None,
) -> QImage:
    """Paint the button as the style would with *state* added.

    Offscreen Qt never delivers a real hover (a plain QPushButton's working
    :hover rule does not show either), and focus depends on what else in the
    window can take it, so add the flag to the style option, which is
    exactly what the stylesheet engine matches :hover and :focus on.
    """
    option = QStyleOptionButton()
    button.initStyleOption(option)
    option.state = (option.state | state) & ~without
    image = QImage(button.size(), QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    button.style().drawControl(
        QStyle.ControlElement.CE_PushButton, option, painter, button,
    )
    painter.end()
    return image


def _hover_fill(button) -> str:
    """Fill with the pointer over the button."""
    image = _render_in_state(button, QStyle.StateFlag.State_MouseOver)
    return image.pixelColor(4, 18).name()


def _border(image) -> str:
    """The left border, halfway down, clear of the rounded corners."""
    return image.pixelColor(0, image.height() // 2).name()


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


@pytest.mark.parametrize("checked", [False, True])
def test_focus_is_left_to_the_native_focus_rectangle(app, checked):
    """No stylesheet focus border: the native Windows style draws its own
    focus rectangle on keyboard focus and hides it after a mouse click. A
    :focus border doubled that ring and stayed after clicks."""
    host, button, colors = _hosted_button("dark")
    button.setChecked(checked)
    QApplication.processEvents()

    unfocused = _render_in_state(
        button,
        QStyle.StateFlag.State_None,
        without=QStyle.StateFlag.State_HasFocus,
    )
    focused = _render_in_state(button, QStyle.StateFlag.State_HasFocus)

    assert _border(focused) == _border(unfocused)
    host.close()


def _repeat_panel(theme="dark"):
    colors = DARK_COLORS if theme == "dark" else LIGHT_COLORS
    library = MagicMock()
    library.songs = []
    host = QWidget()
    layout = QHBoxLayout(host)
    panel = LibraryPanel(library)
    panel.apply_theme(theme, colors)
    layout.addWidget(panel)
    _show_styled(host, theme)
    return host, panel, colors


def test_active_repeat_uses_the_shared_checked_styling(app):
    """Active Repeat set its own widget stylesheet, which outranked the app
    sheet and cost it hover, pressed, and focus feedback."""
    host, panel, colors = _repeat_panel()
    button = panel._repeat_btn

    button.click()
    QApplication.processEvents()
    assert panel._repeat_mode == REPEAT_ALL
    assert button.styleSheet() == ""
    assert _fill(button) == QColor(colors["accent"]).name()
    hovered = _render_in_state(button, QStyle.StateFlag.State_MouseOver)
    assert _border(hovered) == QColor(colors["on_accent"]).name()

    button.click()
    button.click()
    QApplication.processEvents()
    assert panel._repeat_mode == REPEAT_OFF
    assert _fill(button) == QColor(colors["surface0"]).name()
    host.close()


def test_repeat_stays_a_button_for_assistive_tech(app):
    """Checkable, Repeat became a checkbox to screen readers, whose Toggle
    action flipped the look without changing the mode. As a button, its
    only action is Press, which cycles the mode like a click, and its
    accessible name says which mode it is in."""
    host, panel, colors = _repeat_panel()
    button = panel._repeat_btn

    assert not button.isCheckable()
    actions = QAccessible.queryAccessibleInterface(button).actionInterface()
    assert "Toggle" not in actions.actionNames()

    actions.doAction(QAccessibleActionInterface.pressAction())
    QTest.qWait(300)  # Press is routed through animateClick.

    assert panel._repeat_mode == REPEAT_ALL
    assert _fill(button) == QColor(colors["accent"]).name()
    assert "all" in button.accessibleName().lower()
    host.close()

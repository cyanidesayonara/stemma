"""Theme stylesheets for stemma.

Provides dark (Catppuccin Mocha) and light (Catppuccin Latte) themes.
The dark theme uses the brand teal (#4fb8b8) as its accent; the light
theme uses a deeper teal of the same hue, because #4fb8b8 is only about
2:1 against a light background (WCAG asks 3:1 for controls and focus).
Applied globally via QApplication.setStyleSheet().
"""

import os

from src.paths import app_root

STEM_COLORS_DARK = {
    "vocals": "#bfa3dc",   # Purple (brand)
    "drums": "#e4ad6e",    # Gold (brand)
    "bass": "#d4849a",     # Rose (brand)
    "guitar": "#5cb85c",   # Green
    "piano": "#5ba3cf",    # Blue
    "other": "#4fb8b8",    # Teal (brand accent)
}

# The brand hues darkened until stem names reach 4.5:1 on the light
# mantle (the lanes use them too, which also keeps them from washing out).
STEM_COLORS_LIGHT = {
    "vocals": "#765d8f",   # Purple
    "drums": "#87612b",    # Gold
    "bass": "#955762",     # Rose
    "guitar": "#347734",   # Green
    "piano": "#346d97",    # Blue
    "other": "#297272",    # Teal
}

STEM_COLORS = STEM_COLORS_DARK

RECORDING_COLOR = "#d4849a"  # Brand rose -- used for recording stem rows
RECORDING_COLOR_LIGHT = "#955762"  # The same rose at 4.5:1 on light


def recording_color(theme: str) -> str:
    """The recording rose, readable on the given theme's background."""
    return RECORDING_COLOR_LIGHT if theme == "light" else RECORDING_COLOR


# Foreground on the dark theme's accent fill: near-black (Catppuccin Mocha
# "crust") on #4fb8b8. Per theme it is colors["on_accent"]; this constant
# is the dark value, for code that builds widgets before a theme is applied.
ON_ACCENT = "#11111b"

DARK_COLORS = {
    "base": "#1e1e2e",
    "mantle": "#181825",
    "surface0": "#313244",
    "surface1": "#45475a",
    "surface2": "#585b70",
    "text": "#cdd6f4",
    # Secondary text that still carries information: 4.5:1 or better.
    "subtext": "#a6adc8",
    "recording": RECORDING_COLOR,
    "accent": "#4fb8b8",
    "on_accent": ON_ACCENT,
    "red": "#f38ba8",
    "item_hover": "#252536",
}

LIGHT_COLORS = {
    "base": "#eff1f5",
    "mantle": "#e6e9ef",
    "surface0": "#ccd0da",
    "surface1": "#bcc0cc",
    "surface2": "#9ca0b0",
    "text": "#4c4f69",
    "subtext": "#5c5f77",
    "recording": RECORDING_COLOR_LIGHT,
    "accent": "#1f7373",
    "on_accent": "#ffffff",
    "red": "#d20f39",
    "item_hover": "#dce0e8",
}


def badge_html(theme: str, label: str, value: str,
               confidence: str = "") -> str:
    """Rich text for a key/chord/tempo readout: label and value in the
    text colour, with a "?" after a low-confidence value. For a
    transposed key ("C major → D major") the mark goes on the detected key,
    which is what the confidence is about.

    Confidence used to colour the value green, amber, or red, and red read
    as an error; the light-theme amber and green were also under 3:1.
    """
    colors = LIGHT_COLORS if theme == "light" else DARK_COLORS
    text = colors["text"]
    if confidence == "low" and value not in ("", "--"):
        detected, arrow, rest = value.partition(" → ")
        value = f"{detected}?{arrow}{rest}"
    value_html = f'<span style="color:{text};">{value}</span>'
    if label:
        return f'<span style="color:{text};">{label} </span>' + value_html
    return value_html


def _chevron_path(theme: str) -> str:
    """Absolute, forward-slashed path of the combo chevron for *theme*."""
    path = os.path.join(app_root(), "assets", "icons", f"chevron_down_{theme}.svg")
    return path.replace("\\", "/")


def _generate_stylesheet(c: dict[str, str], chevron: str = "") -> str:
    """Generate a QSS stylesheet from a color token dict."""
    return f"""
QMainWindow, QDialog {{
    background-color: {c["base"]};
}}

QMenuBar {{
    background-color: {c["mantle"]};
    color: {c["text"]};
    border-bottom: 1px solid {c["surface0"]};
}}

QMenuBar::item:selected {{
    background-color: {c["surface0"]};
}}

QPushButton#theme-toggle {{
    background-color: {c["mantle"]};
    border: 1px solid {c["surface0"]};
    border-radius: 10px;
    padding: 2px 8px;
    min-height: 0;
    min-width: 0;
    font-size: 12pt;
    color: {c["text"]};
}}

QPushButton#theme-toggle:hover {{
    background-color: {c["surface0"]};
}}

QMenu {{
    background-color: {c["base"]};
    color: {c["text"]};
    border: 1px solid {c["surface0"]};
    padding: 4px 0px;
}}

QMenu::item {{
    padding: 6px 24px 6px 24px;
}}

QMenu::item:selected {{
    background-color: {c["surface0"]};
}}

QSplitter::handle {{
    background-color: {c["surface0"]};
    width: 2px;
}}

QWidget {{
    background-color: {c["base"]};
    color: {c["text"]};
    font-family: "Segoe UI", sans-serif;
    font-size: 10pt;
}}

QLabel {{
    color: {c["text"]};
}}

QLabel#title-label {{
    font-size: 12pt;
    font-weight: bold;
    color: {c["text"]};
}}

QLabel#subtle-label {{
    color: {c["subtext"]};
}}

QPushButton {{
    background-color: {c["surface0"]};
    color: {c["text"]};
    border: 1px solid {c["surface1"]};
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 24px;
}}

QPushButton:hover {{
    background-color: {c["surface1"]};
}}

QPushButton:pressed {{
    background-color: {c["surface2"]};
}}

QPushButton:checked {{
    background-color: {c["accent"]};
    color: {c["on_accent"]};
    border: 1px solid {c["accent"]};
}}

QPushButton:checked:hover {{
    background-color: {c["accent"]};
    color: {c["on_accent"]};
    border: 1px solid {c["on_accent"]};
}}

QPushButton:disabled {{
    background-color: {c["base"]};
    color: {c["surface2"]};
    border-color: {c["surface0"]};
}}

QPushButton#icon-btn {{
    background-color: {c["surface0"]};
    border: 1px solid {c["surface1"]};
    border-radius: 4px;
    padding: 2px;
}}

/* The ID selector above outranks QPushButton:hover/:pressed/:disabled, so
   icon buttons restate them. Hover and pressed come before the :checked
   rules so a checked button keeps its accent; disabled comes after them so
   a disabled button always reads as disabled. There is deliberately no
   :focus rule: the native Windows style already draws a focus rectangle on
   keyboard focus and hides it after a mouse click; a :focus border stayed
   after clicks and doubled the native ring. */
QPushButton#icon-btn:hover {{
    background-color: {c["surface1"]};
}}

QPushButton#icon-btn:pressed {{
    background-color: {c["surface2"]};
}}

/* [active="true"] is for multi-state buttons (library Repeat) that must
   not be checkable; it matches :checked in specificity. */
QPushButton#icon-btn:checked,
QPushButton#icon-btn[active="true"] {{
    background-color: {c["accent"]};
    color: {c["on_accent"]};
    border: 1px solid {c["accent"]};
}}

QPushButton#icon-btn:checked:hover,
QPushButton#icon-btn[active="true"]:hover {{
    border: 1px solid {c["on_accent"]};
}}

QPushButton#icon-btn:disabled {{
    background-color: {c["base"]};
    border-color: {c["surface0"]};
}}

QSlider::groove:horizontal {{
    background: {c["surface0"]};
    height: 6px;
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {c["accent"]};
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}}

QSlider::sub-page:horizontal {{
    background: {c["accent"]};
    border-radius: 3px;
}}

QListWidget {{
    background-color: {c["mantle"]};
    border: 1px solid {c["surface0"]};
    border-radius: 4px;
    outline: none;
}}

QListWidget::item {{
    padding: 8px;
    border-bottom: 1px solid {c["surface0"]};
}}

QListWidget::item:selected {{
    background-color: {c["surface0"]};
    color: {c["text"]};
}}

QListWidget::item:hover {{
    background-color: {c["item_hover"]};
}}

QProgressBar {{
    background-color: {c["surface0"]};
    border: none;
    border-radius: 3px;
    height: 8px;
    text-align: center;
    color: transparent;
}}

QProgressBar::chunk {{
    background-color: {c["accent"]};
    border-radius: 3px;
}}

QScrollBar:vertical {{
    background: {c["mantle"]};
    width: 10px;
    border: none;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {c["surface1"]};
    border-radius: 5px;
    min-height: 24px;
    margin: 2px;
}}

QScrollBar::handle:vertical:hover {{
    background: {c["surface2"]};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: {c["mantle"]};
}}

QScrollBar:horizontal {{
    background: {c["mantle"]};
    height: 10px;
    border: none;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {c["surface1"]};
    border-radius: 5px;
    min-width: 24px;
    margin: 2px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {c["surface2"]};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: {c["mantle"]};
}}

QPushButton:focus {{
    border: 1px solid {c["surface2"]};
}}

QSlider:focus {{
    border: 1px solid {c["accent"]};
    border-radius: 3px;
}}

QListWidget:focus {{
    border: 1px solid {c["accent"]};
}}

QLineEdit {{
    background-color: {c["mantle"]};
    border: 1px solid {c["surface0"]};
    border-radius: 4px;
    padding: 4px 8px;
}}

QLineEdit:focus {{
    border: 1px solid {c["accent"]};
}}

QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {c["surface1"]};
    border-radius: 3px;
    background-color: {c["surface0"]};
}}

QCheckBox::indicator:checked {{
    background-color: {c["accent"]};
    border-color: {c["accent"]};
}}

QCheckBox::indicator:hover {{
    border-color: {c["accent"]};
}}

QComboBox {{
    background-color: {c["surface0"]};
    color: {c["text"]};
    border: 1px solid {c["surface1"]};
    border-radius: 4px;
    padding: 4px 18px 4px 6px;
}}

QComboBox:editable {{
    padding: 0px;
}}

QComboBox:focus {{
    border: 1px solid {c["accent"]};
}}

/* A visible chevron: without one, combos read as text fields (#186). */
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 16px;
    border: none;
}}

QComboBox::down-arrow {{
    image: url("{chevron}");
    width: 10px;
    height: 6px;
}}

QComboBox QLineEdit {{
    background-color: transparent;
    color: {c["text"]};
    border: none;
    padding: 4px 4px 4px 6px;
}}

QComboBox QAbstractItemView {{
    background-color: {c["base"]};
    color: {c["text"]};
    border: 1px solid {c["surface0"]};
    selection-background-color: {c["surface0"]};
}}

QToolTip {{
    background-color: {c["surface0"]};
    color: {c["text"]};
    border: 1px solid {c["surface1"]};
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 9pt;
}}

QFrame#card-frame {{
    background-color: {c["mantle"]};
    border: 1px solid {c["surface0"]};
    border-radius: 6px;
}}

/* The global QWidget rule paints the page color, which would tile a darker
   box behind every label, checkbox, and row container inside a card. */
QFrame#card-frame QLabel,
QFrame#card-frame QCheckBox,
QWidget#card-row {{
    background-color: transparent;
}}

QSpinBox {{
    background-color: {c["surface0"]};
    color: {c["text"]};
    border: 1px solid {c["surface1"]};
    border-radius: 4px;
    padding: 2px 4px;
    min-height: 24px;
}}

QSpinBox:focus {{
    border: 1px solid {c["accent"]};
}}

QSpinBox::up-button, QSpinBox::down-button {{
    border: none;
    width: 16px;
}}

QWidget#footer {{
    border-top: 1px solid {c["surface0"]};
}}

/* The transport is anchored below the scrolling practice content, so it
   needs an edge to read as a fixed bar rather than as the last row that
   happened to scroll into view. */
QWidget#transport-bar {{
    border-top: 1px solid {c["surface0"]};
}}

QLabel#copyright {{
    color: {c["subtext"]};
    font-size: 9pt;
    border: none;
}}
"""


DARK_STYLESHEET = _generate_stylesheet(DARK_COLORS, _chevron_path("dark"))
LIGHT_STYLESHEET = _generate_stylesheet(LIGHT_COLORS, _chevron_path("light"))

THEMES = {
    "dark": {"colors": DARK_COLORS, "stylesheet": DARK_STYLESHEET},
    "light": {"colors": LIGHT_COLORS, "stylesheet": LIGHT_STYLESHEET},
}


def get_stylesheet(theme: str) -> str:
    """Return the QSS stylesheet for the given theme name."""
    return THEMES[theme]["stylesheet"]


def get_colors(theme: str) -> dict[str, str]:
    """Return the color token dict for the given theme name."""
    return THEMES[theme]["colors"]


def apply_tooltip_palette(theme: str) -> None:
    """Force QToolTip colors via QPalette.

    Qt's stylesheet-driven ``QToolTip { ... }`` rule is unreliable:
    when a child widget has its own ``setStyleSheet`` (as several do
    in our stem rows), Qt can show tooltips spawned from that widget
    using the *system* palette instead of the app stylesheet -- which
    on Windows Light themes surfaces near-black-on-near-black text.

    Setting the ``ToolTipBase`` / ``ToolTipText`` palette roles
    directly on the QApplication bypasses the stylesheet entirely and
    applies globally regardless of child widget stylesheet scope.
    """
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QColor, QPalette

    app = QApplication.instance()
    if app is None:
        return
    c = get_colors(theme)
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(c["surface0"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(c["text"]))
    app.setPalette(palette)

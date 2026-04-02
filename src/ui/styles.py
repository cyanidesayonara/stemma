"""Theme stylesheets for stemma.

Provides dark (Catppuccin Mocha) and light (Catppuccin Latte) themes.
Both themes share the brand accent teal (#4fb8b8).
Applied globally via QApplication.setStyleSheet().
"""

STEM_COLORS_DARK = {
    "vocals": "#bfa3dc",   # Purple (brand)
    "drums": "#e4ad6e",    # Gold (brand)
    "bass": "#d4849a",     # Rose (brand)
    "guitar": "#5cb85c",   # Green
    "piano": "#5ba3cf",    # Blue
    "other": "#4fb8b8",    # Teal (brand accent)
}

STEM_COLORS_LIGHT = {
    "vocals": "#9878b8",   # Darker purple for light background
    "drums": "#c89040",    # Darker gold for light background
    "bass": "#c0707e",     # Darker rose for light background
    "guitar": "#3d8b3d",   # Darker green
    "piano": "#3d7fb0",    # Darker blue
    "other": "#3da8a8",    # Darker teal
}

STEM_COLORS = STEM_COLORS_DARK

RECORDING_COLOR = "#d4849a"  # Brand rose -- used for recording stem rows

DARK_COLORS = {
    "base": "#1e1e2e",
    "mantle": "#181825",
    "surface0": "#313244",
    "surface1": "#45475a",
    "surface2": "#585b70",
    "text": "#cdd6f4",
    "accent": "#4fb8b8",
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
    "accent": "#4fb8b8",
    "red": "#d20f39",
    "item_hover": "#dce0e8",
}


def _generate_stylesheet(c: dict[str, str]) -> str:
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
    color: {c["surface2"]};
}}

QPushButton {{
    background-color: {c["surface0"]};
    color: {c["text"]};
    border: 1px solid {c["surface1"]};
    border-radius: 4px;
    padding: 6px 14px;
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
    color: {c["base"]};
    border: 1px solid {c["accent"]};
}}

QPushButton:checked:hover {{
    background-color: {c["accent"]};
    color: {c["base"]};
    border: 1px solid {c["text"]};
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
    min-height: 0;
    min-width: 0;
}}

QPushButton#icon-btn:checked {{
    background-color: {c["accent"]};
    border: 1px solid {c["accent"]};
}}

QPushButton#icon-btn:checked:hover {{
    border: 1px solid {c["text"]};
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
    padding: 4px 8px;
}}

QComboBox:focus {{
    border: 1px solid {c["accent"]};
}}

QComboBox::drop-down {{
    border: none;
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
"""


DARK_STYLESHEET = _generate_stylesheet(DARK_COLORS)
LIGHT_STYLESHEET = _generate_stylesheet(LIGHT_COLORS)

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

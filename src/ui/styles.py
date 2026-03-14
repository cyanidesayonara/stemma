"""Dark theme stylesheet for stemma.

Applied globally via QApplication.setStyleSheet().
"""

# Stem color palette used by player_controls.py.
STEM_COLORS = {
    "vocals": "#9b59b6",   # Purple
    "drums": "#e67e22",    # Orange
    "bass": "#3498db",     # Blue
    "guitar": "#e74c3c",   # Red
    "piano": "#2ecc71",    # Green
    "other": "#95a5a6",    # Gray
}

DARK_STYLESHEET = """
QMainWindow, QDialog {
    background-color: #1e1e2e;
}

QMenuBar {
    background-color: #181825;
    color: #cdd6f4;
    border-bottom: 1px solid #313244;
}

QMenuBar::item:selected {
    background-color: #313244;
}

QMenu {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #313244;
}

QMenu::item:selected {
    background-color: #313244;
}

QSplitter::handle {
    background-color: #313244;
    width: 2px;
}

QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", sans-serif;
    font-size: 10pt;
}

QLabel {
    color: #cdd6f4;
}

QLabel#title-label {
    font-size: 15px;
    font-weight: bold;
    color: #cdd6f4;
}

QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 6px 14px;
    min-height: 24px;
}

QPushButton:hover {
    background-color: #45475a;
}

QPushButton:pressed {
    background-color: #585b70;
}

QPushButton:checked {
    background-color: #585b70;
    border: 1px solid #4fb8b8;
}

QPushButton:disabled {
    background-color: #1e1e2e;
    color: #585b70;
    border-color: #313244;
}

QSlider::groove:horizontal {
    background: #313244;
    height: 6px;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #4fb8b8;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}

QSlider::sub-page:horizontal {
    background: #4fb8b8;
    border-radius: 3px;
}

QListWidget {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 4px;
    outline: none;
}

QListWidget::item {
    padding: 8px;
    border-bottom: 1px solid #313244;
}

QListWidget::item:selected {
    background-color: #313244;
    color: #cdd6f4;
}

QListWidget::item:hover {
    background-color: #252536;
}

QProgressBar {
    background-color: #313244;
    border: none;
    border-radius: 3px;
    height: 8px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background-color: #4fb8b8;
    border-radius: 3px;
}

QScrollBar:vertical {
    background: #181825;
    width: 8px;
    border: none;
}

QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 4px;
    min-height: 20px;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QPushButton:focus {
    border: 1px solid #4fb8b8;
}

QSlider:focus {
    border: 1px solid #4fb8b8;
    border-radius: 3px;
}

QListWidget:focus {
    border: 1px solid #4fb8b8;
}

QLineEdit {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 4px;
    padding: 4px 8px;
}

QLineEdit:focus {
    border: 1px solid #4fb8b8;
}

QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
}

QComboBox:focus {
    border: 1px solid #4fb8b8;
}

QComboBox::drop-down {
    border: none;
}

QComboBox QAbstractItemView {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #313244;
    selection-background-color: #313244;
}
"""

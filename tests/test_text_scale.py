"""Tests for Windows "Make text bigger" support (#188)."""

import re
import sys
import types

import pytest
from PySide6.QtCore import QRect, QSize
from PySide6.QtWidgets import QApplication, QComboBox, QSpinBox

import src.ui.styles as styles
import src.ui.text_scale as text_scale
from src.ui.control_primitives import fit_combo_width, fit_spinbox_width
from src.ui.main_window import minimum_window_size


@pytest.fixture(scope="module")
def qapp():
    """One QApplication for the widget-sizing tests."""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def reset_scale():
    """Leave the stylesheet text size at 100 % after each test."""
    yield
    styles.set_text_scale(1.0)


def _fake_winreg(value=None, error=None):
    """A stand-in winreg whose TextScaleFactor is *value*."""
    module = types.SimpleNamespace(HKEY_CURRENT_USER=object())

    class _Key:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def open_key(_root, _path):
        if error is not None:
            raise error
        return _Key()

    def query(_key, name):
        assert name == "TextScaleFactor"
        return value, 4

    module.OpenKey = open_key
    module.QueryValueEx = query
    return module


@pytest.mark.parametrize("raw, expected", [
    (100, 1.0), (150, 1.5), (225, 2.25), (300, 2.25), (50, 1.0),
])
def test_text_scale_factor_reads_registry(monkeypatch, raw, expected):
    monkeypatch.setattr(text_scale.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "winreg", _fake_winreg(raw))
    assert text_scale.text_scale_factor() == expected


def test_unset_value_defaults_to_100_percent(monkeypatch):
    monkeypatch.setattr(text_scale.sys, "platform", "win32")
    monkeypatch.setitem(
        sys.modules, "winreg", _fake_winreg(error=FileNotFoundError()),
    )
    assert text_scale.text_scale_factor() == 1.0


def test_non_windows_defaults_to_100_percent(monkeypatch):
    monkeypatch.setattr(text_scale.sys, "platform", "linux")
    assert text_scale.text_scale_factor() == 1.0


@pytest.mark.parametrize("points, scale, expected", [
    (10, 1.0, "10pt"), (10, 1.5, "15pt"), (9, 1.5, "13.5pt"),
    (12, 2.25, "27pt"), (11, 1.25, "13.8pt"),
])
def test_scaled_pt(points, scale, expected):
    assert text_scale.scaled_pt(points, scale) == expected


def _font_sizes(sheet: str) -> list[float]:
    return [float(v) for v in re.findall(r"font-size:\s*([\d.]+)pt", sheet)]


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_stylesheet_scales_every_font_size(reset_scale, theme):
    base = _font_sizes(styles.get_stylesheet(theme))
    assert base, "the stylesheet sets font sizes"
    styles.set_text_scale(1.5)
    scaled = _font_sizes(styles.get_stylesheet(theme))
    assert scaled == [round(v * 1.5, 1) for v in base]


def test_default_stylesheet_is_unchanged(reset_scale):
    styles.set_text_scale(1.0)
    assert styles.get_stylesheet("dark") == styles.DARK_STYLESHEET
    assert styles.get_stylesheet("light") == styles.LIGHT_STYLESHEET


def test_set_text_scale_clamps(reset_scale):
    styles.set_text_scale(5.0)
    assert styles.get_text_scale() == 2.25
    styles.set_text_scale(0.5)
    assert styles.get_text_scale() == 1.0


def test_scaled_px_follows_text_scale(reset_scale):
    assert styles.scaled_px(70) == 70
    styles.set_text_scale(1.5)
    assert styles.scaled_px(70) == 105


def _fitted_widths(app, scale):
    styles.set_text_scale(scale)
    app.setStyleSheet(styles.get_stylesheet("dark"))
    combo = QComboBox()
    combo.addItems(["0.25x", "0.5x", "0.75x", "1.0x"])
    fit_combo_width(combo)
    spin = QSpinBox()
    spin.setRange(40, 300)
    spin.setSuffix(" BPM")
    fit_spinbox_width(spin)
    return combo.width(), spin.width()


def test_fitted_boxes_grow_with_the_text(qapp, reset_scale):
    """Boxes were measured in the unpolished default font and clipped."""
    try:
        combo_100, spin_100 = _fitted_widths(qapp, 1.0)
        combo_150, spin_150 = _fitted_widths(qapp, 1.5)
    finally:
        qapp.setStyleSheet("")
    assert combo_150 > combo_100
    assert spin_150 > spin_100


def test_fitted_combo_holds_its_widest_entry(qapp, reset_scale):
    try:
        styles.set_text_scale(2.25)
        qapp.setStyleSheet(styles.get_stylesheet("dark"))
        combo = QComboBox()
        combo.addItems(["0.75x"])
        fit_combo_width(combo)
        text = combo.fontMetrics().horizontalAdvance("0.75x")
    finally:
        qapp.setStyleSheet("")
    assert combo.width() > text + 24  # the text plus the chevron


@pytest.mark.parametrize("available, expected", [
    ((1920, 1040), (900, 600)),
    ((1366, 728), (900, 600)),
    # 1920 x 1080 at 200 %: the old 900 x 600 minimum did not fit.
    ((960, 516), (900, 476)),
    # 1366 x 768 at 175 %.
    ((780, 416), (780, 376)),
])
def test_minimum_window_size_fits_the_screen(available, expected):
    area = QRect(0, 0, *available)
    assert minimum_window_size(area) == QSize(*expected)


def test_transport_logo_gives_way_in_a_narrow_row(qapp):
    """The time readout overlapped "Vol:" at 225 % text in a 900 px window."""
    from PySide6.QtWidgets import QWidget

    from src.ui.transport_bar import TransportBar

    # A child, as in the window: a top-level widget is never sized below
    # its own minimum, but a child of a too-small window is.
    host = QWidget()
    bar = TransportBar(host)
    host.resize(1600, 80)
    host.show()
    bar.setGeometry(0, 0, 1600, 60)
    qapp.processEvents()
    with_logo = bar.layout().minimumSize().width()
    assert not bar.logo.isHidden()

    bar.setGeometry(0, 0, with_logo - 40, 60)
    qapp.processEvents()
    assert bar.logo.isHidden()
    assert bar.layout().minimumSize().width() <= with_logo - 40

    bar.setGeometry(0, 0, with_logo + 20, 60)
    qapp.processEvents()
    assert not bar.logo.isHidden()
    host.close()
    host.deleteLater()

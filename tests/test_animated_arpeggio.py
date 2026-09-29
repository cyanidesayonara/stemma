"""Tests for the animated footer arpeggio logo widget."""

import os

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QPointingDevice
from PySide6.QtWidgets import QApplication

import src.ui.animated_arpeggio as arpeggio_module
from src.ui.animated_arpeggio import (
    AnimatedArpeggioWidget,
    _ANIM_END_MS,
    _FADE_MS,
    _GLOW_DUR_MS,
    _LETTERS,
    _NOTE_SPACING_MS,
    _W,
    _H,
    _brightness,
    _glow,
    _load_base_svg,
)
from src.ui.svg_source import local_name, read_svg


def _left_mouse_press(local_x: float, local_y: float) -> QMouseEvent:
    """Build a non-deprecated QMouseEvent (Qt 6 single-point API)."""
    pos = QPointF(local_x, local_y)
    dev = QPointingDevice.primaryPointingDevice()
    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        pos,
        pos,
        pos,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        dev,
    )


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


class TestLoadBaseSvg:
    def test_dark_variant_loads(self):
        svg = _load_base_svg("dark")
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_light_variant_loads(self):
        svg = _load_base_svg("light")
        assert "<svg" in svg

    def test_no_text_elements(self):
        svg = _load_base_svg("dark")
        assert "<text" not in svg

    def test_staff_lines_preserved(self):
        svg = _load_base_svg("dark")
        assert 'y1="20"' in svg
        assert 'y1="80"' in svg

    def test_clef_path_preserved(self):
        svg = _load_base_svg("dark")
        assert "<path" in svg

    def test_reformatted_text_is_still_removed(self, tmp_path, monkeypatch):
        # A text element split across lines (or several on one line) used
        # to survive the line-based filter and draw under the animation.
        icons = tmp_path / "assets" / "icons"
        icons.mkdir(parents=True)
        (icons / "logo_arpeggio_dark.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 8 420 75">'
            '<line x1="12" y1="20" x2="408" y2="20"/><g>\n'
            '  <text x="95"\n        y="80">s</text></g>'
            '<text x="155" y="65">t</text>\n</svg>\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(arpeggio_module, "_ROOT", str(tmp_path))
        svg = _load_base_svg("dark")
        assert "text" not in svg
        assert "<line" in svg

    @pytest.mark.parametrize("theme", ["dark", "light"])
    def test_letters_match_the_svg(self, theme):
        path = os.path.join(
            arpeggio_module._ROOT, "assets", "icons",
            f"logo_arpeggio_{theme}.svg",
        )
        texts = [
            e for e in read_svg(path).iter() if local_name(e) == "text"
        ]
        column = 3 if theme == "dark" else 4
        assert [(e.text, float(e.get("x"))) for e in texts] == [
            (letter[0], float(letter[1])) for letter in _LETTERS
        ]
        assert [e.get("fill").lower() for e in texts] == [
            letter[column].lower() for letter in _LETTERS
        ]


class TestBrightness:
    def test_before_onset(self):
        assert _brightness(0, 100) == 0.0

    def test_at_onset(self):
        assert _brightness(100, 100) == pytest.approx(0.0)

    def test_after_full_fade(self):
        assert _brightness(100 + _FADE_MS + 1, 100) == 1.0

    def test_mid_fade(self):
        val = _brightness(100 + _FADE_MS // 2, 100)
        assert 0.0 < val < 1.0


class TestGlow:
    def test_before_full_appearance(self):
        assert _glow(0, 100) == 0.0

    def test_at_glow_start(self):
        val = _glow(100 + _FADE_MS, 100)
        assert val == pytest.approx(1.0)

    def test_after_glow(self):
        assert _glow(100 + _FADE_MS + _GLOW_DUR_MS + 1, 100) == 0.0

    def test_mid_glow(self):
        val = _glow(100 + _FADE_MS + _GLOW_DUR_MS // 2, 100)
        assert 0.0 < val < 1.0


class TestAnimatedArpeggioConstruction:
    def test_creates_without_error(self, app):
        w = AnimatedArpeggioWidget(theme="dark", play_sound=False)
        assert w.width() == _W
        assert w.height() == _H

    def test_light_theme(self, app):
        w = AnimatedArpeggioWidget(theme="light", play_sound=False)
        assert w._is_dark is False

    def test_base_pixmap_not_null(self, app):
        w = AnimatedArpeggioWidget(theme="dark", play_sound=False)
        assert not w._base_pixmap.isNull()

    def test_cursor_is_pointing_hand(self, app):
        w = AnimatedArpeggioWidget(theme="dark", play_sound=False)
        assert w.cursor().shape() == Qt.CursorShape.PointingHandCursor


class TestPlayIntro:
    def test_starts_timer(self, app):
        w = AnimatedArpeggioWidget(theme="dark", play_sound=False)
        w.play_intro(with_sound=False)
        assert w._timer.isActive()
        assert w._animating is True
        assert w._clock.isValid()
        w._timer.stop()

    def test_restart_resets_animation(self, app):
        w = AnimatedArpeggioWidget(theme="dark", play_sound=False)
        w.play_intro(with_sound=False)
        w._animating = False
        w.play_intro(with_sound=False)
        assert w._animating is True
        w._timer.stop()


class TestThemeSwitching:
    def test_set_theme_updates_flag(self, app):
        w = AnimatedArpeggioWidget(theme="dark", play_sound=False)
        assert w._is_dark is True
        w.set_theme("light")
        assert w._is_dark is False

    def test_set_theme_rebuilds_pixmap(self, app):
        w = AnimatedArpeggioWidget(theme="dark", play_sound=False)
        dark_pix = w._base_pixmap
        w.set_theme("light")
        assert w._base_pixmap is not dark_pix


class TestPaintEvent:
    def test_paint_does_not_crash_static(self, app):
        w = AnimatedArpeggioWidget(theme="dark", play_sound=False)
        w.repaint()

    def test_paint_does_not_crash_animating(self, app):
        w = AnimatedArpeggioWidget(theme="dark", play_sound=False)
        w.play_intro(with_sound=False)
        w.repaint()
        w._timer.stop()

    def test_paint_light_theme(self, app):
        w = AnimatedArpeggioWidget(theme="light", play_sound=False)
        w.repaint()


class TestAnimationConstants:
    def test_letter_count(self):
        assert len(_LETTERS) == 6

    def test_letters_have_correct_fields(self):
        for char, x, y, dark_c, light_c in _LETTERS:
            assert len(char) == 1
            assert isinstance(x, (int, float))
            assert isinstance(y, (int, float))
            assert dark_c.startswith("#")
            assert light_c.startswith("#")

    def test_anim_end_after_last_letter(self):
        last_onset = 5 * _NOTE_SPACING_MS
        assert _ANIM_END_MS > last_onset


class TestClickReplay:
    def test_click_triggers_play_intro(self, app):
        w = AnimatedArpeggioWidget(theme="dark", play_sound=False)
        w.mousePressEvent(_left_mouse_press(10, 10))
        assert w._animating is True
        w._timer.stop()

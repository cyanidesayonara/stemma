"""Tests for the animated main logo widget."""

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QPointingDevice
from PySide6.QtWidgets import QApplication

from src.ui.animated_logo import (
    AnimatedLogoWidget,
    _ANIM_END_MS,
    _BOUNCE_MS,
    _DAMPEN_START_MS,
    _FADE_MS,
    _NOTE_SPACING_MS,
    _NOTES,
    _UNDULATE_START_MS,
    _W,
    _H,
    _bounce_y,
    _load_base_svg,
    _note_alpha,
)


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

    def test_no_ellipses_in_base(self):
        svg = _load_base_svg("dark")
        assert "<ellipse" not in svg

    def test_no_wave_paths_in_base(self):
        svg = _load_base_svg("dark")
        assert 'path d="M105,' not in svg

    def test_staff_lines_preserved(self):
        svg = _load_base_svg("dark")
        assert 'y1="48"' in svg
        assert 'y1="108"' in svg

    def test_stemma_text_preserved(self):
        svg = _load_base_svg("dark")
        assert "translate(76.7,23)" in svg


class TestNoteAlpha:
    def test_before_onset(self):
        assert _note_alpha(0, 100) == 0.0

    def test_at_onset(self):
        assert _note_alpha(100, 100) == pytest.approx(0.0)

    def test_after_full_fade(self):
        assert _note_alpha(100 + _FADE_MS + 1, 100) == 1.0

    def test_mid_fade(self):
        alpha = _note_alpha(100 + _FADE_MS // 2, 100)
        assert 0.0 < alpha < 1.0


class TestBounceY:
    def test_before_onset(self):
        assert _bounce_y(0, 100) == 0.0

    def test_at_onset(self):
        assert _bounce_y(100, 100) == pytest.approx(-4.0)

    def test_after_bounce(self):
        assert _bounce_y(100 + _BOUNCE_MS + 1, 100) == 0.0

    def test_mid_bounce_negative(self):
        val = _bounce_y(100 + _BOUNCE_MS // 2, 100)
        assert -4.0 < val < 0.0


class TestAnimatedLogoConstruction:
    def test_creates_without_error(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        assert w.width() == _W
        assert w.height() == _H

    def test_light_theme(self, app):
        w = AnimatedLogoWidget(theme="light", play_sound=False)
        assert w._is_dark is False

    def test_base_pixmap_not_null(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        assert not w._base_pixmap.isNull()

    def test_cursor_is_pointing_hand(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        assert w.cursor().shape() == Qt.CursorShape.PointingHandCursor

    def test_initial_static_t_shows_final_frame(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        assert w._static_t > 0


class TestPlayIntro:
    def test_starts_timer(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        w.play_intro(with_sound=False)
        assert w._timer.isActive()
        assert w._animating is True
        assert w._clock.isValid()
        w._timer.stop()

    def test_sets_static_t(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        assert w._static_t == _ANIM_END_MS
        w.play_intro(with_sound=False)
        assert w._static_t == _ANIM_END_MS
        w._timer.stop()

    def test_restart_resets_clock(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        w.play_intro(with_sound=False)
        first_start = w._clock.elapsed()
        import time
        time.sleep(0.05)
        w.play_intro(with_sound=False)
        assert w._clock.elapsed() < first_start + 100
        w._timer.stop()


class TestThemeSwitching:
    def test_set_theme_updates_colors(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        assert w._is_dark is True
        w.set_theme("light")
        assert w._is_dark is False
        assert w._colors[0] == "#3da8a8"

    def test_set_theme_rebuilds_pixmap(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        dark_pix = w._base_pixmap
        w.set_theme("light")
        assert w._base_pixmap is not dark_pix


class TestPaintEvent:
    def test_paint_does_not_crash_static(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        w.repaint()

    def test_paint_does_not_crash_animating(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        w.play_intro(with_sound=False)
        w.repaint()
        w._timer.stop()

    def test_paint_light_theme(self, app):
        w = AnimatedLogoWidget(theme="light", play_sound=False)
        w.repaint()


class TestAnimationConstants:
    def test_note_count(self):
        assert len(_NOTES) == 4

    def test_chord_style_fast_spacing(self):
        assert _NOTE_SPACING_MS <= 50

    def test_anim_end_after_dampen(self):
        assert _ANIM_END_MS > _DAMPEN_START_MS

    def test_undulate_before_dampen(self):
        assert _UNDULATE_START_MS < _DAMPEN_START_MS


class TestClickReplay:
    def test_click_triggers_play_intro(self, app):
        w = AnimatedLogoWidget(theme="dark", play_sound=False)
        w.mousePressEvent(_left_mouse_press(10, 10))
        assert w._animating is True
        w._timer.stop()

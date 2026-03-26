"""Tests for the animated splash screen."""

from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QWidget

from src.app_settings import read_startup_play_sound
from src.ui.splash_screen import (
    SplashScreen,
    _ANIM_END_MS,
    _CLEF_DELAY_MS,
    _FADE_MS,
    _LETTERS,
    _MIN_DISPLAY_MS,
    _NOTE_SPACING_MS,
    _make_base_svg,
)


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


@pytest.fixture
def settings_ini(tmp_path):
    return QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)


class TestMakeBaseSvg:
    def test_contains_staff_lines(self):
        svg = _make_base_svg("#aaa", "#bbb")
        for y in (20, 35, 50, 65, 80):
            assert f'y1="{y}"' in svg

    def test_contains_clef_fill(self):
        svg = _make_base_svg("#aaa", "#bbb")
        assert 'fill="#bbb"' in svg

    def test_valid_svg_root(self):
        svg = _make_base_svg("#000", "#fff")
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")


class TestSplashScreenConstruction:
    def test_creates_without_error(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        assert splash.isVisible() is False

    def test_light_theme_sets_colors(self, app):
        splash = SplashScreen(theme="light", play_sound=False)
        assert splash._is_dark is False
        assert splash._bg_color.name() == "#eff1f5"

    def test_dark_theme_sets_colors(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        assert splash._is_dark is True
        assert splash._bg_color.name() == "#1e1e2e"

    def test_base_pixmap_not_null(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        assert not splash._base_pixmap.isNull()


class TestLetterAlpha:
    def test_before_onset_returns_zero(self):
        assert SplashScreen._letter_alpha(0, 0) == 0.0

    def test_at_onset_returns_zero(self):
        onset = _CLEF_DELAY_MS
        assert SplashScreen._letter_alpha(0, onset) == pytest.approx(0.0, abs=0.01)

    def test_after_fade_returns_one(self):
        onset = _CLEF_DELAY_MS + _FADE_MS + 50
        assert SplashScreen._letter_alpha(0, onset) == 1.0

    def test_mid_fade_returns_partial(self):
        onset = _CLEF_DELAY_MS + _FADE_MS // 2
        alpha = SplashScreen._letter_alpha(0, onset)
        assert 0.0 < alpha < 1.0

    def test_later_letter_has_later_onset(self):
        assert SplashScreen._letter_alpha(5, _CLEF_DELAY_MS) == 0.0
        full_onset = _CLEF_DELAY_MS + 5 * _NOTE_SPACING_MS + _FADE_MS + 50
        assert SplashScreen._letter_alpha(5, full_onset) == 1.0


class TestSplashScreenStart:
    def test_start_makes_visible(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        splash.start()
        assert splash.isVisible()
        splash.close()

    def test_start_begins_timer(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        splash.start()
        assert splash._timer.isActive()
        assert splash._clock.isValid()
        splash.close()

    @patch("src.ui.splash_screen._HAS_WINSOUND", False)
    def test_no_sound_when_winsound_unavailable(self, app):
        splash = SplashScreen(
            theme="dark", play_sound=True, audio_path="fake.wav"
        )
        splash.start()
        splash.close()


class TestSplashScreenFinish:
    def test_finish_shows_main_window(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        splash.start()
        splash._clock.restart()

        main_win = QWidget()
        splash._clock = MagicMock()
        splash._clock.isValid.return_value = True
        splash._clock.elapsed.return_value = _MIN_DISPLAY_MS + 100

        splash.finish(main_win)
        app.processEvents()
        app.processEvents()
        assert splash._main_window is main_win

    def test_finish_when_not_visible_shows_window_immediately(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        main_win = QWidget()
        splash.finish(main_win)
        assert main_win.isVisible()
        main_win.close()

    def test_finish_defers_if_early(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        splash.start()

        splash._clock = MagicMock()
        splash._clock.isValid.return_value = True
        splash._clock.elapsed.return_value = 100

        main_win = QWidget()
        splash.finish(main_win)
        assert splash.isVisible()
        splash.close()

    def test_double_finish_is_safe(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        splash.start()

        splash._clock = MagicMock()
        splash._clock.isValid.return_value = True
        splash._clock.elapsed.return_value = _MIN_DISPLAY_MS + 100

        main_win = QWidget()
        splash.finish(main_win)
        splash.finish(main_win)
        assert splash._finishing is True

    def test_on_fade_done_shows_window_and_closes(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        main_win = QWidget()
        splash._main_window = main_win
        splash._on_fade_done()
        assert main_win.isVisible()
        main_win.close()

    def test_begin_fade_out_ignored_if_already_fading(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        splash.start()
        splash._begin_fade_out()
        first_anim = splash._fade_anim
        splash._begin_fade_out()
        assert splash._fade_anim is first_anim
        splash.close()


class TestAnimationConstants:
    def test_letter_count_matches(self):
        assert len(_LETTERS) == 6

    def test_anim_end_formula(self):
        expected = _CLEF_DELAY_MS + 5 * _NOTE_SPACING_MS + _FADE_MS
        assert _ANIM_END_MS == expected

    def test_min_display_at_least_anim_end(self):
        assert _MIN_DISPLAY_MS >= _ANIM_END_MS

    def test_letters_have_correct_fields(self):
        for char, x, y, dark_c, light_c in _LETTERS:
            assert len(char) == 1
            assert isinstance(x, (int, float))
            assert isinstance(y, (int, float))
            assert dark_c.startswith("#")
            assert light_c.startswith("#")


class TestStartupPlaySoundSetting:
    def test_default_true(self, settings_ini):
        assert read_startup_play_sound(settings_ini) is True

    def test_set_false(self, settings_ini):
        settings_ini.setValue("startup/play_sound", False)
        assert read_startup_play_sound(settings_ini) is False

    def test_set_true_explicit(self, settings_ini):
        settings_ini.setValue("startup/play_sound", True)
        assert read_startup_play_sound(settings_ini) is True


class TestPaintEvent:
    def test_paint_does_not_crash(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        splash.start()
        splash.repaint()
        splash.close()

    def test_paint_after_animation_end(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        splash.start()
        splash._clock = MagicMock()
        splash._clock.isValid.return_value = True
        splash._clock.elapsed.return_value = _ANIM_END_MS + 500
        splash.repaint()
        splash.close()

    def test_paint_light_theme(self, app):
        splash = SplashScreen(theme="light", play_sound=False)
        splash.start()
        splash.repaint()
        splash.close()

"""Tests for the animated splash screen."""

from unittest.mock import MagicMock, mock_open, patch

import pytest
from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import QApplication, QWidget

from src.app_settings import read_startup_play_sound
from src.ui.splash_screen import (
    SplashScreen,
    _ANIM_END_MS,
    _CLEF_DELAY_MS,
    _FADE_MS,
    _LETTERS,
    _MIN_ANIM_FRAMES,
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


@pytest.fixture(autouse=True)
def _headless_safe_show(monkeypatch):
    """Prevent native window creation during splash tests.

    Calling ``show()`` on a QWidget on a headless Windows CI runner can
    trigger a synchronous WM_PAINT that fatally faults without a display
    driver (access violation).  ``WA_DontShowOnScreen`` is Qt's designated
    attribute for this: it skips native window creation while leaving
    ``isVisible()``/``setVisible()`` semantics intact.

    Rather than patch every test, we wrap ``QWidget.show`` so the
    attribute is applied immediately before any show() call -- covering
    both the SplashScreen and any QWidget instances that ``finish()``
    may try to show.
    """
    original_show = QWidget.show

    def safe_show(self):
        self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        original_show(self)

    monkeypatch.setattr(QWidget, "show", safe_show)


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
    # processEvents triggers a paint which can access-violation on headless
    # CI runners (no display driver).  Mock it out so we test the Python
    # logic without touching Qt's rendering pipeline.
    @patch("src.ui.splash_screen.QApplication.processEvents")
    def test_start_makes_visible(self, _pe, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        splash.start()
        assert splash.isVisible()
        splash.close()

    @patch("src.ui.splash_screen.QApplication.processEvents")
    def test_start_begins_timer(self, _pe, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        splash.start()
        assert splash._timer.isActive()
        assert splash._clock.isValid()
        splash.close()

    @patch("src.ui.splash_screen.QApplication.processEvents")
    @patch("src.ui.splash_screen._HAS_WINSOUND", False)
    def test_no_sound_when_winsound_unavailable(self, _pe, app):
        splash = SplashScreen(
            theme="dark", play_sound=True, audio_path="fake.wav"
        )
        splash.start()
        splash.close()


class TestSplashScreenFinish:
    def test_finish_shows_main_window(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        splash.start()

        main_win = QWidget()
        splash._sound_start_ms = 0
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

        splash._sound_start_ms = 0
        splash._clock = MagicMock()
        splash._clock.isValid.return_value = True
        splash._clock.elapsed.return_value = 100

        main_win = QWidget()
        splash.finish(main_win)
        assert splash.isVisible()
        splash.close()

    def test_finish_resyncs_when_animation_never_painted(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        splash.start()

        splash._sound_start_ms = 600
        splash._anim_paint_count = 1
        splash._clock = MagicMock()
        splash._clock.isValid.return_value = True
        splash._clock.elapsed.return_value = 9000

        main_win = QWidget()
        splash.finish(main_win)
        assert splash.isVisible()
        assert splash._sound_start_ms == 9000
        splash.close()

    def test_finish_no_resync_when_animation_played(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        splash.start()

        splash._sound_start_ms = 600
        splash._anim_paint_count = _MIN_ANIM_FRAMES + 1
        splash._clock = MagicMock()
        splash._clock.isValid.return_value = True
        splash._clock.elapsed.return_value = 9000

        main_win = QWidget()
        splash.finish(main_win)
        assert splash._sound_start_ms == 600
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


class TestSoundSync:
    def test_sound_not_started_on_first_frame(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        splash.start()
        assert splash._frame_count == 1
        assert splash._sound_started is False
        splash.close()

    def test_sound_started_on_second_frame(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        splash.start()
        splash.repaint()
        assert splash._frame_count == 2
        assert splash._sound_started is True
        splash.close()

    @patch("src.ui.splash_screen.os.path.isfile", return_value=True)
    @patch("src.ui.splash_screen.winsound.PlaySound")
    def test_sound_deferred_when_frame2_is_late(self, mock_play, _isfile, app):
        splash = SplashScreen(
            theme="dark", play_sound=True, audio_path="fake.wav"
        )
        splash.start()
        splash._clock = MagicMock()
        splash._clock.isValid.return_value = True
        splash._clock.elapsed.return_value = 600
        splash.repaint()
        mock_play.assert_not_called()
        assert splash._sound_played_on_frame2 is False
        splash.close()

    @patch("src.ui.splash_screen.os.path.isfile", return_value=True)
    @patch("builtins.open", mock_open(read_data=b"RIFF...."))
    @patch("src.ui.splash_screen.winsound.PlaySound")
    def test_sound_plays_when_frame2_is_prompt(self, mock_play, _isfile, app):
        splash = SplashScreen(
            theme="dark", play_sound=True, audio_path="fake.wav"
        )
        splash.start()
        splash._clock = MagicMock()
        splash._clock.isValid.return_value = True
        splash._clock.elapsed.return_value = 50
        splash.repaint()
        mock_play.assert_called_once()
        assert splash._sound_played_on_frame2 is True
        splash.close()

    def test_sound_start_offset_recorded(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        splash.start()
        splash._clock = MagicMock()
        splash._clock.isValid.return_value = True
        splash._clock.elapsed.return_value = 5000
        splash.repaint()
        assert splash._sound_start_ms == 5000
        splash.close()

    def test_anim_paint_count_increments_after_sound(self, app):
        splash = SplashScreen(theme="dark", play_sound=False)
        splash.start()
        assert splash._anim_paint_count == 0
        splash.repaint()
        assert splash._anim_paint_count == 1
        splash.repaint()
        assert splash._anim_paint_count == 2
        splash.close()


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

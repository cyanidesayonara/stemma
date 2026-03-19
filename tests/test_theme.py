"""Tests for theme switching functionality."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from src.ui.styles import (
    DARK_COLORS,
    DARK_STYLESHEET,
    LIGHT_COLORS,
    LIGHT_STYLESHEET,
    STEM_COLORS,
    get_colors,
    get_stylesheet,
)
from src.ui.main_window import MainWindow
from src.ui.waveform_widget import WaveformWidget
from src.ui.player_controls import PlayerControls


@pytest.fixture(scope="module")
def app():
    """Ensure a QApplication exists for widget tests."""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


class TestThemeColors:
    """Tests for theme color dictionaries and stylesheet generation."""

    def test_dark_colors_has_required_keys(self):
        """Dark color dict contains all required token keys."""
        required = {"base", "mantle", "surface0", "surface1", "surface2",
                     "text", "accent", "red", "item_hover"}
        assert required.issubset(DARK_COLORS.keys())

    def test_light_colors_has_required_keys(self):
        """Light color dict contains all required token keys."""
        required = {"base", "mantle", "surface0", "surface1", "surface2",
                     "text", "accent", "red", "item_hover"}
        assert required.issubset(LIGHT_COLORS.keys())

    def test_accent_color_shared(self):
        """Both themes share the brand teal accent."""
        assert DARK_COLORS["accent"] == LIGHT_COLORS["accent"]
        assert DARK_COLORS["accent"] == "#4fb8b8"

    def test_dark_and_light_differ(self):
        """Dark and light themes have different base/text colors."""
        assert DARK_COLORS["base"] != LIGHT_COLORS["base"]
        assert DARK_COLORS["text"] != LIGHT_COLORS["text"]

    def test_get_stylesheet_dark(self):
        """get_stylesheet('dark') returns the dark stylesheet."""
        ss = get_stylesheet("dark")
        assert ss == DARK_STYLESHEET
        assert DARK_COLORS["base"] in ss

    def test_get_stylesheet_light(self):
        """get_stylesheet('light') returns the light stylesheet."""
        ss = get_stylesheet("light")
        assert ss == LIGHT_STYLESHEET
        assert LIGHT_COLORS["base"] in ss

    def test_get_colors_dark(self):
        """get_colors('dark') returns the dark color dict."""
        assert get_colors("dark") is DARK_COLORS

    def test_get_colors_light(self):
        """get_colors('light') returns the light color dict."""
        assert get_colors("light") is LIGHT_COLORS

    def test_get_stylesheet_invalid_raises(self):
        """get_stylesheet with an unknown theme raises KeyError."""
        with pytest.raises(KeyError):
            get_stylesheet("neon")

    def test_stem_colors_unchanged(self):
        """Stem colors are theme-independent."""
        assert "vocals" in STEM_COLORS
        assert len(STEM_COLORS) == 6

    def test_stylesheets_contain_all_widget_rules(self):
        """Generated stylesheets include rules for all major widget types."""
        for ss in (DARK_STYLESHEET, LIGHT_STYLESHEET):
            assert "QMainWindow" in ss
            assert "QPushButton" in ss
            assert "QSlider" in ss
            assert "QListWidget" in ss
            assert "QProgressBar" in ss
            assert "QScrollBar" in ss
            assert "QLineEdit" in ss
            assert "QComboBox" in ss
            assert "QLabel#title-label" in ss
            assert "QLabel#subtle-label" in ss
            assert "QPushButton#theme-toggle" in ss

    def test_title_label_uses_pt_sizing(self):
        """Title label font-size uses pt, not px."""
        for ss in (DARK_STYLESHEET, LIGHT_STYLESHEET):
            assert "font-size: 12pt" in ss


class TestWaveformWidgetTheme:
    """Tests for WaveformWidget theme color switching."""

    def test_default_colors_are_dark(self, app):
        """WaveformWidget defaults to dark theme colors."""
        widget = WaveformWidget()
        assert widget._bg_color == QColor(DARK_COLORS["base"])
        assert widget._waveform_color == QColor(DARK_COLORS["accent"])
        assert widget._cursor_color == QColor(DARK_COLORS["text"])

    def test_set_theme_colors_updates(self, app):
        """set_theme_colors applies new colors."""
        widget = WaveformWidget()
        widget.set_theme_colors(LIGHT_COLORS)
        assert widget._bg_color == QColor(LIGHT_COLORS["base"])
        assert widget._cursor_color == QColor(LIGHT_COLORS["text"])

    def test_set_theme_colors_invalidates_cache(self, app):
        """Changing theme invalidates the waveform rect cache."""
        widget = WaveformWidget()
        widget._cached_size = (300, 80)

        widget.set_theme_colors(LIGHT_COLORS)
        assert widget._cached_size == (0, 0)

    def test_paint_no_crash_light_theme(self, app):
        """paintEvent works with light theme colors."""
        widget = WaveformWidget()
        widget.set_theme_colors(LIGHT_COLORS)
        widget.resize(200, 80)
        peaks = np.array([0.1, 0.5, 0.3, 0.8], dtype=np.float32)
        widget.set_peaks(peaks)
        widget.set_loop_markers(0.2, 0.8)
        widget.repaint()


class TestPlayerControlsTheme:
    """Tests for PlayerControls.apply_theme()."""

    def _make_player_mock(self):
        player = MagicMock()
        player.stems = {}
        player.muted_stems = set()
        player.soloed_stems = set()
        player.volumes = {}
        player.total_seconds = 0.0
        player.current_seconds = 0.0
        player.loop_a = None
        player.loop_b = None
        player.is_playing = False
        player.has_stems = False
        return player

    def test_apply_theme_changes_internal_state(self, app):
        """apply_theme updates the internal theme name."""
        player = self._make_player_mock()
        controls = PlayerControls(player)
        assert controls._theme == "dark"

        controls.apply_theme("light", LIGHT_COLORS)
        assert controls._theme == "light"

    def test_apply_theme_updates_icons(self, app):
        """apply_theme rebuilds transport icons."""
        player = self._make_player_mock()
        controls = PlayerControls(player)

        old_play = controls._play_icon
        controls.apply_theme("light", LIGHT_COLORS)
        assert controls._play_icon is not old_play

    def test_apply_theme_updates_waveform(self, app):
        """apply_theme delegates to waveform widget."""
        player = self._make_player_mock()
        controls = PlayerControls(player)

        with patch.object(controls._waveform, "set_theme_colors") as mock:
            controls.apply_theme("light", LIGHT_COLORS)
            mock.assert_called_once_with(LIGHT_COLORS)

    def test_apply_theme_back_to_dark(self, app):
        """Switching back to dark restores dark colors."""
        player = self._make_player_mock()
        controls = PlayerControls(player)
        controls.apply_theme("light", LIGHT_COLORS)
        controls.apply_theme("dark", DARK_COLORS)
        assert controls._theme == "dark"
        assert controls._waveform._bg_color == QColor(DARK_COLORS["base"])


class TestThemeToggleButton:
    """Tests for the theme toggle button in the menu bar."""

    def _make_window(self, theme="dark"):
        from PySide6.QtCore import QSettings
        settings = QSettings("stemma", "stemma")
        settings.setValue("theme", theme)

        player = MagicMock()
        player.stems = {}
        player.muted_stems = set()
        player.soloed_stems = set()
        player.volumes = {}
        player.total_seconds = 0.0
        player.current_seconds = 0.0
        player.loop_a = None
        player.loop_b = None
        player.is_playing = False
        player.has_stems = False

        library = MagicMock()
        library.songs = []
        model_mgr = MagicMock()
        return MainWindow(library, player, model_mgr)

    def test_toggle_button_exists(self, app):
        """Menu bar has a corner theme toggle button."""
        win = self._make_window()
        assert win._theme_btn is not None
        assert win._theme_btn.objectName() == "theme-toggle"

    def test_toggle_button_shows_sun_in_dark_mode(self, app):
        """In dark mode the button shows a sun (switch to light)."""
        win = self._make_window()
        assert win._theme == "dark"
        assert "\u2600" in win._theme_btn.text()

    def test_toggle_button_shows_moon_in_light_mode(self, app):
        """In light mode the button shows a crescent moon (switch to dark)."""
        win = self._make_window()
        win._theme_action.setChecked(True)
        assert win._theme == "light"
        assert "\u263D" in win._theme_btn.text()

    def test_toggle_button_syncs_with_menu_action(self, app):
        """Clicking the toggle button toggles the menu action."""
        win = self._make_window()
        assert not win._theme_action.isChecked()
        win._theme_btn.click()
        assert win._theme_action.isChecked()
        assert win._theme == "light"

    def test_menu_action_syncs_with_toggle_button(self, app):
        """Toggling the menu action updates the button text."""
        win = self._make_window()
        win._theme_action.setChecked(True)
        assert "\u263D" in win._theme_btn.text()
        win._theme_action.setChecked(False)
        assert "\u2600" in win._theme_btn.text()

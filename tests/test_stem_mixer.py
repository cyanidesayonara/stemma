"""Tests for stem mixer rows and their wiring into the waveform."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from src.ui.player_controls import PlayerControls
from src.ui.stem_mixer import RecordingStemRow, StemRow
from src.ui.styles import DARK_COLORS, get_stylesheet


@pytest.fixture(scope="module")
def app():
    """Ensure a QApplication exists for widget tests."""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


class TestMixChangedWiring:
    """Mute and solo refresh lane opacity without recomputing peaks."""

    def _make_player_mock(self):
        """Create a mock MultiTrackPlayer with stem data."""
        player = MagicMock()
        player.stems = {
            "vocals": np.full((100, 2), 0.5, dtype=np.float32),
            "drums": np.full((100, 2), 0.3, dtype=np.float32),
        }
        player.muted_stems = set()
        player.soloed_stems = set()
        player.volumes = {}
        player.total_seconds = 2.0
        player.current_seconds = 0.0
        player.loop_a = None
        player.loop_b = None
        player.is_playing = False
        player.has_stems = True
        return player

    def test_mute_refreshes_lane_mix_without_recompute(self, app):
        """Muting a stem refreshes stack opacities without recomputing peaks."""
        player = self._make_player_mock()
        controls = PlayerControls(player)
        controls.set_stem_names(["vocals", "drums"])
        controls._cached_stem_peaks = {
            "vocals": np.array([0.1, 0.5], dtype=np.float32),
            "drums": np.array([0.2, 0.4], dtype=np.float32),
        }

        with patch.object(controls, "_recompute_peaks") as mock_recompute, patch.object(
            controls, "_refresh_waveform_lane_mix",
        ) as mock_refresh:
            controls._stem_rows["vocals"]._on_mute(True)
            mock_refresh.assert_called_once()
            mock_recompute.assert_not_called()

    def test_solo_refreshes_lane_mix_without_recompute(self, app):
        """Soloing a stem refreshes stack opacities without recomputing peaks."""
        player = self._make_player_mock()
        controls = PlayerControls(player)
        controls.set_stem_names(["vocals", "drums"])
        controls._cached_stem_peaks = {
            "vocals": np.array([0.1, 0.5], dtype=np.float32),
            "drums": np.array([0.2, 0.4], dtype=np.float32),
        }

        with patch.object(controls, "_recompute_peaks") as mock_recompute, patch.object(
            controls, "_refresh_waveform_lane_mix",
        ) as mock_refresh:
            controls._stem_rows["drums"]._on_solo(True)
            mock_refresh.assert_called_once()
            mock_recompute.assert_not_called()


class TestStemRowLayout:
    """Tests for simplified stem mixer rows."""

    def test_stem_row_has_no_mini_waveform(self, app):
        player = MagicMock()
        player.muted_stems = set()
        player.soloed_stems = set()
        player.volumes = {}
        row = StemRow("vocals", player)
        assert not hasattr(row, "_mini_waveform")
        assert row._label.text() == "Vocals"
        assert row._mute_btn is not None
        assert row._solo_btn is not None

    def test_stem_row_packs_controls_left(self, app):
        """A trailing stretch keeps the row compact.

        Every control in the row is fixed width, so without one the layout
        distributes the slack between them: at 1366px that put roughly 300px
        of dead space between a stem's name and its own mute button. The mini
        waveform absorbed it until #149 removed it.
        """
        player = MagicMock()
        player.muted_stems = set()
        player.soloed_stems = set()
        player.volumes = {}
        row = StemRow("vocals", player)
        layout = row.layout()

        last = layout.itemAt(layout.count() - 1)
        assert last.widget() is None
        assert last.spacerItem() is not None

    def test_recording_row_controls_come_before_the_stretch(self, app):
        """Recording controls must land in the packed group, not off to the
        right of the stretch that packs it."""
        player = MagicMock()
        player.muted_stems = set()
        player.soloed_stems = set()
        player.volumes = {}
        row = RecordingStemRow("take-1", "Take 1", player)
        layout = row.layout()

        assert layout.itemAt(layout.count() - 1).spacerItem() is not None
        packed = [
            layout.itemAt(index).widget()
            for index in range(layout.count() - 1)
        ]
        # They sit in the take slot every row reserves (so rows align).
        assert row._take_slot in packed
        assert row._nudge_spin.parent() is row._take_slot
        assert row._delete_btn.parent() is row._take_slot

    @pytest.mark.parametrize("width", [560, 700, 900])
    def test_take_rows_line_up_with_stem_rows(self, app, width):
        """Sliders stretch now; take rows' extra controls used to make their
        sliders shorter, so the preset combos zig-zagged down the mixer."""
        from PySide6.QtWidgets import QApplication

        player = MagicMock()
        player.muted_stems = set()
        player.soloed_stems = set()
        player.volumes = {}
        rows = [StemRow("drums", player),
                RecordingStemRow("take-1", "Take 1", player)]
        for row in rows:
            row.resize(width, 36)
            row.show()
        QApplication.processEvents()

        sliders = {row._volume_slider.width() for row in rows}
        combos = {row._vol_combo.x() for row in rows}
        assert len(sliders) == 1 and len(combos) == 1

    @pytest.mark.parametrize("button", ["_mute_btn", "_solo_btn"])
    def test_checked_toggle_shows_the_accent_fill(self, app, button):
        """A muted or soloed stem must look muted or soloed.

        The row set a selector-less ``background: transparent`` stylesheet,
        which cascades to every child and outranks the application sheet, so
        a checked mute/solo button lost its accent fill. Its checked icon is
        drawn dark for that fill, so in the dark theme the button went almost
        blank exactly when it was switched on.
        """
        player = MagicMock()
        player.muted_stems = set()
        player.soloed_stems = set()
        player.volumes = {}
        # Stand in for the QApplication-wide sheet without touching the app.
        host = QWidget()
        layout = QVBoxLayout(host)
        row = StemRow("vocals", player)
        layout.addWidget(row)
        host.resize(420, 40)
        host.show()
        QApplication.processEvents()
        # Applied after the row is parented and shown, so every child is
        # re-polished against it. Set before, CI occasionally grabbed the
        # button still unstyled (Qt's default grey #dbdbdb).
        host.setStyleSheet(get_stylesheet("dark"))

        toggle = getattr(row, button)
        toggle.setChecked(True)
        toggle.ensurePolished()
        QApplication.processEvents()

        image = toggle.grab().toImage()
        # Just inside the border, clear of the icon glyph in the middle.
        assert image.pixelColor(4, 14).name() == QColor(
            DARK_COLORS["accent"]
        ).name()

        host.close()
        host.deleteLater()

    def test_stem_row_snapshot(self, app):
        from tests.widget_visual import (
            assert_widget_snapshot,
            deterministic_render_state,
        )

        player = MagicMock()
        player.muted_stems = set()
        player.soloed_stems = set()
        player.volumes = {}
        # The row sizes its take slot from a spinbox measured when it is
        # built, so build it under the pinned stylesheet and font too.
        with deterministic_render_state():
            row = StemRow("vocals", player)
        assert_widget_snapshot(row, "stem_row_simplified", width=420, height=36)


def test_preset_combo_index_follows_the_value(app):
    from PySide6.QtWidgets import QComboBox

    from src.ui.control_primitives import (
        add_volume_presets,
        make_display_combo,
        show_preset_value,
    )

    combo = QComboBox()
    make_display_combo(combo)
    add_volume_presets(combo)
    # Highest first, so Down (the next item) lowers the volume.
    assert combo.itemData(0) == 200 and combo.itemData(combo.count() - 1) == 0
    assert combo.currentData() == 100
    show_preset_value(combo, 100)
    assert combo.currentData() == 100 and combo.currentText() == "100%"
    show_preset_value(combo, 67)  # between presets: nearest is selected
    assert combo.currentData() == 60 and combo.currentText() == "67%"


@pytest.mark.parametrize("size", [28, 30, 36])
def test_fixed_buttons_stay_square_under_the_stylesheet(app, size):
    """QPushButton's min-height rule made every icon button 30 px tall."""
    from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

    from src.ui.control_primitives import fix_button_size
    from src.ui.styles import get_stylesheet

    previous = app.styleSheet()
    try:
        app.setStyleSheet(get_stylesheet("dark"))
        host = QWidget()
        layout = QHBoxLayout(host)
        button = QPushButton()
        button.setObjectName("icon-btn")
        fix_button_size(button, size, size)
        layout.addWidget(button)
        host.show()
        app.processEvents()
        assert button.size().width() == button.size().height() == size
        app.setStyleSheet(get_stylesheet("light"))  # theme switch re-polishes
        app.processEvents()
        assert button.size().width() == button.size().height() == size
        host.close()
        host.deleteLater()
    finally:
        app.setStyleSheet(previous)


def test_metronome_combo_follows_its_slider(app):
    """Only the edit text followed the slider, so Down jumped from 100%."""
    from src.ui.practice_rack import PracticeRack
    from src.ui.song_info_bar import SongInfoBar

    rack = PracticeRack(SongInfoBar())
    rack._metronome_volume_slider.setValue(40)
    assert rack._metronome_volume_combo.currentData() == 40
    assert rack._metronome_volume_combo.currentText() == "40%"


def test_repeat_stays_square_after_it_turns_on(app, tmp_path):
    """Its active-state repolish restored the 30 px stylesheet minimum."""
    from src.library import SongLibrary
    from src.ui.library_panel import LibraryPanel
    from src.ui.styles import get_stylesheet

    previous = app.styleSheet()
    try:
        app.setStyleSheet(get_stylesheet("dark"))
        panel = LibraryPanel(SongLibrary(str(tmp_path)))
        panel.show()
        app.processEvents()
        button = panel._repeat_btn
        for _ in range(3):  # off -> all -> one -> off
            button.click()
            app.processEvents()
            assert button.width() == button.height() == 28
        panel.close()
        panel.deleteLater()
    finally:
        app.setStyleSheet(previous)


def test_named_button_sizes_have_stylesheet_rules():
    from src.ui import control_primitives as cp
    from src.ui.styles import ICON_BUTTON_SIZES

    for size in (cp.TRANSPORT_BUTTON, cp.ROW_BUTTON, cp.COMPACT_BUTTON):
        assert size in ICON_BUTTON_SIZES

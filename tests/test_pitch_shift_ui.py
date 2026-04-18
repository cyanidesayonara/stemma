"""UI-level tests for pitch-shift behavior in ``PlayerControls``.

Covers the work done to make rapid spinbox scrolling safe and
discoverable:

  - A 200ms debounce coalesces rapid ``valueChanged`` emissions into a
    single ``player.set_pitch`` call.
  - Scrolling the spinbox cancels any in-flight render immediately so
    we stop wasting CPU on a stale target.
  - The pitch spinbox stays enabled during a render (no more frozen UI).
  - ``stretch_progress`` updates the pitch spinbox suffix
    (``" st (2/4)"``) so progress is visually attached to the control.
  - ``stretch_finished`` restores the idle suffix.
  - Speed-only renders fall back to the floating status label because
    a QComboBox cannot carry inline suffix text.
"""

from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication

from src.player import MultiTrackPlayer
from src.ui.player_controls import PlayerControls


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def player():
    return MultiTrackPlayer()


@pytest.fixture
def controls(qapp, player):
    ctrl = PlayerControls(player)
    yield ctrl
    ctrl._cleanup_peak_thread()


# -----------------------------------------------------------------------
# Debounce: rapid spinbox scroll coalesces into a single set_pitch
# -----------------------------------------------------------------------

class TestPitchDebounce:
    """``QSpinBox.valueChanged`` only drives a render after the timer fires."""

    def test_single_change_does_not_call_set_pitch_immediately(
        self, controls, player,
    ):
        """valueChanged must not short-circuit into an immediate render."""
        with patch.object(player, "set_pitch") as mock_set_pitch:
            controls._on_pitch_changed(2)
            mock_set_pitch.assert_not_called()

    def test_timer_starts_on_change(self, controls):
        """The debounce timer should be running after a change."""
        controls._on_pitch_changed(2)
        assert controls._pitch_debounce.isActive()

    def test_pending_value_stored(self, controls):
        """The pending pitch is captured until the timer fires."""
        controls._on_pitch_changed(3)
        assert controls._pending_pitch == 3

    def test_rapid_changes_coalesce(self, controls, player):
        """Scrolling 0 -> 1 -> 2 -> 3 -> 4 yields one set_pitch(4) call."""
        with patch.object(player, "set_pitch") as mock_set_pitch:
            for v in (1, 2, 3, 4):
                controls._on_pitch_changed(v)
            # Flush manually as if the timer had fired.
            controls._flush_pending_pitch()
            mock_set_pitch.assert_called_once_with(4)

    def test_flush_with_no_pending_is_noop(self, controls, player):
        """Firing the timer with no pending value does nothing."""
        controls._pending_pitch = None
        with patch.object(player, "set_pitch") as mock_set_pitch:
            controls._flush_pending_pitch()
            mock_set_pitch.assert_not_called()

    def test_flush_clears_pending(self, controls, player):
        """After flushing, the pending field is reset so the next
        scroll cycle starts clean."""
        controls._on_pitch_changed(5)
        with patch.object(player, "set_pitch"):
            controls._flush_pending_pitch()
        assert controls._pending_pitch is None

    def test_change_cancels_running_render(self, controls, player):
        """Scrolling the spinbox must cancel any in-flight worker
        immediately -- we don't want to keep burning CPU on a stale
        pitch while the user is still scrubbing."""
        with patch.object(player, "cancel_stretch") as mock_cancel:
            controls._on_pitch_changed(2)
            mock_cancel.assert_called_once()


# -----------------------------------------------------------------------
# Status indicator driven by stretch_started / stretch_progress / stretch_finished
# -----------------------------------------------------------------------

class TestStretchStatusIndicator:
    """The render lifecycle paints progress onto the active control."""

    def test_started_keeps_spinbox_enabled(self, controls, player):
        """The spinbox MUST stay interactive so the user can cancel
        a pitch scrub by changing the target again."""
        player._pitch_semitones = 2
        controls._on_stretch_started()
        assert controls._pitch_spin.isEnabled()

    def test_started_keeps_speed_combo_enabled(self, controls, player):
        player._playback_speed = 0.75
        controls._on_stretch_started()
        assert controls._speed_combo.isEnabled()

    def test_pitch_render_updates_spinbox_suffix(self, controls, player):
        player._pitch_semitones = 2
        controls._on_stretch_started()
        # Before any progress ticks, we show a pending state.
        assert controls._pitch_spin.suffix() != " st"

    def test_pitch_progress_appears_in_spinbox_suffix(
        self, controls, player,
    ):
        player._pitch_semitones = 2
        controls._on_stretch_progress(2, 4)
        suffix = controls._pitch_spin.suffix()
        assert "(2/4)" in suffix
        assert "st" in suffix

    def test_pitch_progress_does_not_duplicate_in_floating_label(
        self, controls, player,
    ):
        """When pitch is the active transform, the spinbox suffix is the
        indicator -- the floating label stays empty to avoid duplication."""
        player._pitch_semitones = 2
        controls._on_stretch_progress(2, 4)
        assert controls._speed_status.text() == ""

    def test_speed_only_progress_goes_to_floating_label(
        self, controls, player,
    ):
        """Combos can't carry suffixes, so speed-only renders fall back
        to the floating label near the speed combo."""
        player._pitch_semitones = 0
        player._playback_speed = 0.75
        controls._on_stretch_progress(2, 4)
        text = controls._speed_status.text()
        assert "Time-stretching" in text
        assert "(2/4)" in text
        # And the spinbox suffix stays at its idle state.
        assert controls._pitch_spin.suffix() == " st"

    def test_finished_restores_spinbox_suffix(self, controls, player):
        player._pitch_semitones = 2
        controls._on_stretch_progress(2, 4)
        controls._on_stretch_finished()
        assert controls._pitch_spin.suffix() == " st"

    def test_finished_clears_floating_label(self, controls, player):
        player._playback_speed = 0.75
        controls._on_stretch_progress(1, 4)
        controls._on_stretch_finished()
        assert controls._speed_status.text() == ""


# -----------------------------------------------------------------------
# Label verb selection based on active transforms (helper function)
# -----------------------------------------------------------------------

class TestRenderStatusLabel:
    """``_render_status_label`` composes status text for the floating
    label. Kept for the speed-only case; the pitch case uses the spinbox
    suffix directly."""

    def test_pitch_only(self, controls, player):
        player._pitch_semitones = 3
        player._playback_speed = 1.0
        assert "Transposing" in controls._render_status_label(0, 0)

    def test_speed_only(self, controls, player):
        player._pitch_semitones = 0
        player._playback_speed = 0.75
        assert "Time-stretching" in controls._render_status_label(0, 0)

    def test_both(self, controls, player):
        player._pitch_semitones = 3
        player._playback_speed = 0.5
        label = controls._render_status_label(0, 0)
        assert "Transposing and time-stretching" in label

    def test_identity_falls_back_to_rendering(self, controls, player):
        """Returning to identity (fast path) rarely triggers the worker,
        but the label must still be sensible if it does."""
        player._pitch_semitones = 0
        player._playback_speed = 1.0
        label = controls._render_status_label(0, 0)
        assert "Rendering" in label

    def test_progress_numbers_appear_when_total_positive(
        self, controls, player,
    ):
        player._pitch_semitones = 3
        assert "(2/4)" in controls._render_status_label(2, 4)

    def test_progress_numbers_omitted_at_total_zero(self, controls, player):
        """Before any progress ticks arrive, we show the bare verb."""
        player._pitch_semitones = 3
        label = controls._render_status_label(0, 0)
        assert "(" not in label


# -----------------------------------------------------------------------
# Record button guard includes pitch (regression: was speed-only)
# -----------------------------------------------------------------------

class TestRecordButtonPitchGuard:
    """Record button must be disabled (and auto-unarm) whenever pitch ≠ 0,
    mirroring the player-level guard in arm_recording()."""

    def test_record_button_disabled_at_nonzero_pitch(self, controls, player):
        """Button must be disabled when pitch is non-zero."""
        player._stems = {"vocals": None}  # make has_stems True
        player._pitch_semitones = 2
        player._playback_speed = 1.0
        controls.update_record_button_state()
        assert not controls._record_btn.isEnabled()

    def test_record_button_enabled_at_identity(self, controls, player):
        """Button must be enabled at speed=1.0 AND pitch=0."""
        player._stems = {"vocals": None}
        player._pitch_semitones = 0
        player._playback_speed = 1.0
        controls.update_record_button_state()
        assert controls._record_btn.isEnabled()

    def test_record_button_tooltip_mentions_pitch(self, controls, player):
        """Tooltip must explain why recording is disabled when pitch != 0."""
        player._stems = {"vocals": None}
        player._pitch_semitones = 3
        player._playback_speed = 1.0
        controls.update_record_button_state()
        assert "pitch" in controls._record_btn.toolTip().lower()

    def test_record_button_unarms_on_pitch_change(self, controls, player):
        """If the button was checked (armed) and pitch changes, it must
        uncheck automatically so the UI stays consistent."""
        player._stems = {"vocals": None}
        player._pitch_semitones = 0
        player._playback_speed = 1.0
        controls.update_record_button_state()

        # Arm it manually (bypass player so we can isolate the UI logic).
        controls._record_btn.blockSignals(True)
        controls._record_btn.setChecked(True)
        controls._record_btn.blockSignals(False)

        # Now pitch changes.
        player._pitch_semitones = 1
        with patch.object(player, "arm_recording") as mock_arm:
            controls.update_record_button_state()
        assert not controls._record_btn.isChecked()
        mock_arm.assert_called_once_with(False)


# -----------------------------------------------------------------------
# Debounce state cleared when loading a new song (regression)
# -----------------------------------------------------------------------

class TestDebounceResetOnSongLoad:
    """Pending pitch scroll must not carry over to the next loaded song."""

    def test_pending_pitch_cleared_on_set_stem_names(self, controls):
        """set_stem_names must discard any pending pitch value."""
        controls._on_pitch_changed(5)
        assert controls._pending_pitch == 5
        controls.set_stem_names([])
        assert controls._pending_pitch is None

    def test_debounce_timer_stopped_on_set_stem_names(self, controls):
        """set_stem_names must stop the debounce timer."""
        controls._on_pitch_changed(3)
        assert controls._pitch_debounce.isActive()
        controls.set_stem_names([])
        assert not controls._pitch_debounce.isActive()

    def test_no_spurious_set_pitch_after_song_load(self, controls, player):
        """Timer firing after set_stem_names must not call set_pitch."""
        controls._on_pitch_changed(4)
        controls.set_stem_names([])
        with patch.object(player, "set_pitch") as mock_set_pitch:
            # Manually flush as if the timer fired.
            controls._flush_pending_pitch()
            mock_set_pitch.assert_not_called()

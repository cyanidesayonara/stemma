"""UI-level tests for pitch-shift behavior in ``PlayerControls``.

Covers the work done to make rapid spinbox scrolling safe and
discoverable:

  - A 200ms debounce coalesces rapid ``valueChanged`` emissions into a
    single ``player.set_pitch`` call.
  - Scrolling the spinbox cancels any in-flight render immediately so
    we stop wasting CPU on a stale target.
  - The pitch spinbox stays enabled during a render (no more frozen UI).
  - ``stretch_progress`` updates the pitch spinbox with a processing
    suffix (e.g. ``"+2 semitones (processing 2/4)"``) so progress is
    visually attached to the control that spawned the render.
  - ``stretch_finished`` clears the suffix.
  - Speed-only renders fall back to the floating status label because
    a QComboBox cannot carry inline suffix text.
  - The spinbox itself renders human-readable text via
    ``PitchSpinBox.textFromValue`` -- "original" at 0, "+N semitone(s)"
    otherwise.
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
        # Before any progress ticks, we show a pending state (non-empty).
        assert controls._pitch_spin.suffix() != ""

    def test_pitch_progress_appears_in_spinbox_suffix(
        self, controls, player,
    ):
        player._pitch_semitones = 2
        controls._on_stretch_progress(2, 4)
        suffix = controls._pitch_spin.suffix()
        # Compact indicator: just "(current/total)" -- the control
        # being greyed-out is itself the "this is processing" cue.
        assert "2/4" in suffix

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
        to the floating label next to the speed combo.  The compact
        format matches the spinbox suffix so both indicators read the
        same way."""
        player._pitch_semitones = 0
        player._playback_speed = 0.75
        controls._on_stretch_progress(2, 4)
        text = controls._speed_status.text()
        assert "2/4" in text
        # And the spinbox suffix stays empty (its main text already
        # reads "original" when pitch is 0).
        assert controls._pitch_spin.suffix() == ""

    def test_finished_clears_spinbox_suffix(self, controls, player):
        player._pitch_semitones = 2
        controls._on_stretch_progress(2, 4)
        controls._on_stretch_finished()
        assert controls._pitch_spin.suffix() == ""

    def test_finished_clears_floating_label(self, controls, player):
        player._playback_speed = 0.75
        controls._on_stretch_progress(1, 4)
        controls._on_stretch_finished()
        assert controls._speed_status.text() == ""


# -----------------------------------------------------------------------
# PitchSpinBox human-readable text
# -----------------------------------------------------------------------

class TestPitchSpinBoxText:
    """``textFromValue`` produces human-readable labels rather than a
    bare integer + unit suffix.  This is what the user sees in the UI.

    The text is kept intentionally compact ("+2 semi" rather than
    "+2 semitones") because the progress suffix is appended during
    renders and the combined string has to fit the spinbox width
    without clipping.  "semi" is short for "semitone" and is
    unambiguous in the context of a "Pitch:" label.
    """

    def test_zero_reads_as_original(self, controls):
        controls._pitch_spin.setValue(0)
        assert "original" in controls._pitch_spin.text()

    def test_positive_value_shows_plus_sign(self, controls):
        controls._pitch_spin.setValue(1)
        text = controls._pitch_spin.text()
        assert "+1 semi" in text

    def test_positive_multi_semitone(self, controls):
        controls._pitch_spin.setValue(2)
        assert "+2 semi" in controls._pitch_spin.text()

    def test_negative_value_shows_minus_sign(self, controls):
        controls._pitch_spin.setValue(-1)
        text = controls._pitch_spin.text()
        assert "-1 semi" in text

    def test_negative_multi_semitone(self, controls):
        controls._pitch_spin.setValue(-3)
        assert "-3 semi" in controls._pitch_spin.text()

    def test_idle_spinbox_has_no_processing_suffix(self, controls):
        """When the spinbox is idle we only show the core label --
        the processing tail is added only during a render."""
        controls._pitch_spin.setValue(2)
        text = controls._pitch_spin.text()
        # No "(N/M)" fragment when idle.
        assert "/" not in text

    def test_processing_suffix_appended_during_render(
        self, controls, player,
    ):
        player._pitch_semitones = 2
        controls._pitch_spin.setValue(2)
        controls._on_stretch_progress(1, 4)
        text = controls._pitch_spin.text()
        assert "+2 semi" in text
        # Compact progress format: just "(current/total)", no word.
        assert "(1/4)" in text

    def test_size_hint_fits_widest_processing_text(self, controls):
        """sizeHint must be wide enough for the worst-case text so
        the processing suffix never clips the ``semi`` or the counter.

        Sizing is now fixed at construction (not dynamic) for
        layout-stability reasons -- see PitchSpinBox docstring.
        """
        from PySide6.QtGui import QFontMetrics
        spin = controls._pitch_spin
        fm = QFontMetrics(spin.font())
        # Longest text the spinbox can ever show at ±7 semitones
        # with a progress counter capped at two digits per stem.
        widest_text_w = fm.horizontalAdvance("+7 semi (10/10)")
        assert spin.sizeHint().width() >= widest_text_w

    def test_size_hint_is_stable_across_values(self, controls):
        """Width must not jitter as the value / suffix changes --
        the layout was previously thrashing on every render start."""
        spin = controls._pitch_spin
        spin.setValue(0)
        w_at_zero = spin.sizeHint().width()
        spin.setValue(7)
        w_at_seven = spin.sizeHint().width()
        spin.setSuffix(" (3/10)")
        w_processing = spin.sizeHint().width()
        assert w_at_zero == w_at_seven == w_processing


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


# -----------------------------------------------------------------------
# Speed debounce — mirrors the pitch debounce
# -----------------------------------------------------------------------

class TestSpeedDebounce:
    """Rapid speed combo changes coalesce into one render just like pitch.

    Before this, Shift+Up/Down cycling through presets would spawn a new
    worker per step, feeding the same disk-fill bug that motivated the
    render-serialization fix on the player side.
    """

    def test_speed_change_defers_set_speed(self, controls, player):
        """Selecting a preset does not synchronously call set_speed."""
        with patch.object(player, "set_speed") as mock_set_speed:
            # Find the index for 0.75 and select it.
            idx = controls._speed_combo.findData(0.75)
            assert idx >= 0, "0.75 preset should exist"
            controls._speed_combo.setCurrentIndex(idx)
            mock_set_speed.assert_not_called()

    def test_speed_debounce_timer_starts(self, controls):
        """Speed change should arm the 100ms debounce timer."""
        idx = controls._speed_combo.findData(0.75)
        controls._speed_combo.setCurrentIndex(idx)
        assert controls._speed_debounce.isActive()
        assert controls._pending_speed == 0.75

    def test_speed_change_cancels_in_flight(self, controls, player):
        """A fresh change cancels any running render for CPU relief."""
        with patch.object(player, "cancel_stretch") as mock_cancel:
            idx = controls._speed_combo.findData(0.5)
            controls._speed_combo.setCurrentIndex(idx)
            mock_cancel.assert_called_once()

    def test_speed_rapid_changes_coalesce(self, controls, player):
        """Cycling through 1.0 -> 0.9 -> 0.75 -> 0.5 yields one set_speed(0.5)."""
        with patch.object(player, "set_speed") as mock_set_speed:
            for target in (0.9, 0.75, 0.5):
                idx = controls._speed_combo.findData(target)
                if idx < 0:
                    continue
                controls._speed_combo.setCurrentIndex(idx)
            controls._flush_pending_speed()
            # Final call is whatever the last preset was.
            mock_set_speed.assert_called_once()
            (called_speed,), _ = mock_set_speed.call_args
            assert called_speed == controls._speed_combo.currentData()

    def test_speed_flush_clears_pending(self, controls, player):
        """After flush, _pending_speed resets for the next cycle."""
        idx = controls._speed_combo.findData(0.75)
        controls._speed_combo.setCurrentIndex(idx)
        with patch.object(player, "set_speed"):
            controls._flush_pending_speed()
        assert controls._pending_speed is None

    def test_speed_flush_no_pending_is_noop(self, controls, player):
        """Timer fire with no pending change does nothing."""
        controls._pending_speed = None
        with patch.object(player, "set_speed") as mock_set_speed:
            controls._flush_pending_speed()
            mock_set_speed.assert_not_called()

    def test_load_stems_resets_speed_debounce(self, controls):
        """Switching songs drops any queued speed change."""
        idx = controls._speed_combo.findData(0.75)
        controls._speed_combo.setCurrentIndex(idx)
        assert controls._speed_debounce.isActive()

        controls.set_stem_names([])

        assert not controls._speed_debounce.isActive()
        assert controls._pending_speed is None


# -----------------------------------------------------------------------
# Master volume slider — transport-row indicator
# -----------------------------------------------------------------------

class TestMasterVolumeSlider:
    """The slider mirrors the player's master volume and is the single
    entry point for shortcut-driven volume changes."""

    def test_default_value_is_full(self, controls):
        assert controls._master_volume_slider.value() == 100
        assert controls._master_volume_label.text() == "100%"

    def test_slider_drag_calls_player(self, controls, player):
        """Moving the slider propagates into the player."""
        with patch.object(player, "set_master_volume") as mock_set:
            controls._master_volume_slider.setValue(75)
            mock_set.assert_called_once_with(0.75)

    def test_slider_drag_updates_label(self, controls):
        controls._master_volume_slider.setValue(125)
        assert controls._master_volume_label.text() == "125%"

    def test_set_master_volume_updates_slider(self, controls):
        """External callers (shortcuts, session restore) use set_master_volume."""
        controls.set_master_volume(0.5)
        assert controls._master_volume_slider.value() == 50
        assert controls._master_volume_label.text() == "50%"

    def test_set_master_volume_clamps(self, controls):
        """Out-of-range values are clamped to the 0-200% range."""
        controls.set_master_volume(5.0)
        assert controls._master_volume_slider.value() == 200

        controls.set_master_volume(-1.0)
        assert controls._master_volume_slider.value() == 0

    def test_set_master_volume_blocks_signal_recursion(self, controls, player):
        """set_master_volume must not recurse through the slider signal
        (which would call set_master_volume again via
        _on_master_volume_slider_changed -> player.set_master_volume)."""
        calls: list[float] = []
        with patch.object(
            player, "set_master_volume", side_effect=calls.append,
        ):
            controls.set_master_volume(0.6)
            # Exactly one player call -- not the slider's valueChanged
            # bouncing back through the handler.
            assert calls == [0.6]

    def test_set_master_volume_propagates_to_player(self, controls, player):
        with patch.object(player, "set_master_volume") as mock_set:
            controls.set_master_volume(1.5)
            mock_set.assert_called_once_with(1.5)

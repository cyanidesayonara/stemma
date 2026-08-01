"""Player transport controls and per-stem mute/solo mixer.

PlayerControls remains the composition facade over TransportBar,
StemMixer, PracticeRack, and SongInfoBar. The transport bar hosts a
~280px stacked stem waveform view (``WaveformStackWidget``); peak jobs
route stem lanes here and per-stem mini waveforms were removed from the
mixer. Further practice-cockpit visual recomposition is deferred to later
v3.0 slices; this module preserves the shipped layout and public
integration points.
"""

from concurrent.futures import Future, ThreadPoolExecutor
import time

import numpy as np

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.beat_detector import DetectionResult, DetectionWorker, transpose_key
from src.metronome import tap_tempo
from src.player import SPEED_PRESETS, MultiTrackPlayer
from src.qt_signal_utils import safe_disconnect
from src.ui.animated_arpeggio import AnimatedArpeggioWidget
from src.ui.animated_logo import AnimatedLogoWidget
from src.ui.control_primitives import format_time as _format_time
from src.ui.practice_rack import PracticeRack
from src.ui.song_info_bar import SongInfoBar
from src.ui.stem_mixer import RecordingStemRow, StemMixer, StemRow
from src.ui.styles import (
    CONFIDENCE_COLORS,
    DARK_COLORS,
    LIGHT_COLORS,
    RECORDING_COLOR,
    STEM_COLORS_DARK,
    STEM_COLORS_LIGHT,
)
from src.ui.transport_bar import TransportBar
from src.waveform import compute_stem_peaks

_PEAK_DEBOUNCE_MS = 80


def _compute_peaks_bg(stems, stem_bins=2000):
    """Compute per-stem peaks on a background thread.

    Mute/solo/volume are not applied here: the stacked lanes show each stem
    at full scale and express mix state as paint opacity, so peaks stay valid
    across mix changes and only need recomputing when the audio itself does.

    ``stem_bins`` matches the resolution the old single composite waveform
    used. Bin count is nearly free: the cost is the O(frames) pass over the
    audio, not the reduction into bins.
    """
    return {
        name: compute_stem_peaks(data, num_bins=stem_bins)
        for name, data in stems.items()
    }


_peak_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="peak")


def _get_peak_pool() -> ThreadPoolExecutor:
    """Return the module's peak-computation pool, recreating it if needed.

    ``closeEvent`` calls ``shutdown_peak_pool`` to let the interpreter
    exit promptly instead of blocking in atexit on an in-flight peak
    computation.  In test runs several ``MainWindow`` instances are
    constructed and closed sequentially; once the shared pool is shut
    down, later tests still expect it to accept submits.  Rather than
    plumb a per-window pool through every call site, we detect the
    shutdown state and lazily spin up a fresh pool.
    """
    global _peak_pool
    if getattr(_peak_pool, "_shutdown", False):
        _peak_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="peak",
        )
    return _peak_pool


def shutdown_peak_pool() -> None:
    """Shut down the module-level peak computation pool.

    Called from the main window's ``closeEvent`` so the interpreter
    doesn't block in ``atexit`` waiting for an in-flight waveform peak
    computation to finish.
    """
    _peak_pool.shutdown(wait=True, cancel_futures=True)


class PlayerControls(QWidget):
    """Transport controls and stem mixer panel."""

    def __init__(self, player: MultiTrackPlayer,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._player = player
        self._stem_rows: dict[str, StemRow] = {}
        self._recording_rows: dict[str, RecordingStemRow] = {}
        self._theme = "dark"

        self._peaks_timer = QTimer(self)
        self._peaks_timer.setSingleShot(True)
        self._peaks_timer.setInterval(_PEAK_DEBOUNCE_MS)
        self._peaks_timer.timeout.connect(self._start_peak_computation)

        self._peak_future: Future | None = None
        self._peak_future_generation: int | None = None
        self._peak_generation = 0
        self._peak_refresh_pending = False
        self._cached_stem_peaks: dict[str, np.ndarray] | None = None
        self._peak_poll_timer = QTimer(self)
        self._peak_poll_timer.setInterval(16)  # ~60fps poll
        self._peak_poll_timer.timeout.connect(self._poll_peak_future)

        self._detection_worker: DetectionWorker | None = None
        # Keep Python refs to old workers until their OS threads finish;
        # prevents "QThread: Destroyed while thread is still running".
        self._orphaned_workers: list[DetectionWorker] = []
        self._model_manager = None  # set via set_model_manager()
        self._beat_model_path: str | None = None
        self._beat_model_downloader = None
        self._pending_detect_args: tuple | None = None
        self._pending_detect_generation: int | None = None
        self._detection_generation = 0
        self._closing = False
        self._key_conf: str = ""
        self._bpm_conf: str = ""
        self._detected_key_raw: str = ""
        self._detected_bpm_raw: str = ""
        # Chord display polling timer (~4 Hz).
        self._chord_timer = QTimer(self)
        self._chord_timer.setInterval(250)
        self._chord_timer.timeout.connect(self._update_chord_label)

        self._setup_ui()
        self._connect_signals()

    def _cleanup_peak_thread(self) -> None:
        """Wait for any pending peak computation before destruction."""
        self._peak_generation += 1
        self._peak_refresh_pending = False
        self._peaks_timer.stop()
        self._peak_poll_timer.stop()
        if self._peak_future is not None and not self._peak_future.done():
            self._peak_future.result(timeout=2)
        self._peak_future = None
        self._peak_future_generation = None
        if self._detection_worker is not None:
            safe_disconnect(self._detection_worker.completed)
            safe_disconnect(self._detection_worker.error)
            safe_disconnect(self._detection_worker.finished)
            self._detection_worker.wait()
            self._detection_worker = None

    def shutdown(self) -> None:
        """Drain all background workers before the app tears down.

        Called from MainWindow.closeEvent. Detection runs on every song
        load and takes seconds (ONNX beat tracking over the full mix);
        closing the app shortly after selecting a song used to leave a
        running parentless QThread to be reaped at interpreter exit,
        crashing with 'QThread: Destroyed while thread is still running'.
        """
        self._closing = True
        self._detection_generation += 1
        self._pending_detect_args = None
        self._pending_detect_generation = None
        downloader = self._beat_model_downloader
        self._beat_model_downloader = None
        if downloader is not None:
            safe_disconnect(downloader.download_complete)
            safe_disconnect(downloader.error)
            downloader.cancel()
            if downloader.isRunning():
                downloader.wait()

        self._pitch_debounce.stop()
        self._speed_debounce.stop()
        self._cleanup_peak_thread()
        # Join any orphaned detection workers still finishing up.
        for worker in list(self._orphaned_workers):
            worker.wait()
        self._orphaned_workers.clear()

    @property
    def transport_bar(self) -> TransportBar:
        """Return the core transport component."""
        return self._transport_bar

    @property
    def stem_mixer(self) -> StemMixer:
        """Return the stem and recording mixer component."""
        return self._stem_mixer

    @property
    def practice_rack(self) -> PracticeRack:
        """Return the practice-control component."""
        return self._practice_rack

    @property
    def song_info_bar(self) -> SongInfoBar:
        """Return the song detection readout component."""
        return self._song_info_bar

    def _setup_component_ui(self) -> None:
        """Compose the shipped layout from cohesive child widgets."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        self._empty_widget = QWidget()
        empty_layout = QVBoxLayout(self._empty_widget)
        empty_layout.addStretch(1)
        self._empty_logo = AnimatedLogoWidget(self._theme)
        empty_layout.addWidget(
            self._empty_logo,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        self._hint_label = QLabel("Drop an audio file or use File > Import")
        self._hint_label.setObjectName("subtle-label")
        self._hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(
            self._hint_label,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        empty_layout.addStretch(1)
        layout.addWidget(self._empty_widget, 1)

        self._controls_widget = QWidget()
        controls_layout = QVBoxLayout(self._controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        self._song_info_bar = SongInfoBar(self)
        self._practice_rack = PracticeRack(self._song_info_bar, self)
        self._transport_bar = TransportBar(
            self._practice_rack.count_in_controls,
            self,
        )
        self._stem_mixer = StemMixer(self._player, self)

        controls_layout.addWidget(self._transport_bar)
        controls_layout.addWidget(self._practice_rack)
        controls_layout.addWidget(self._stem_mixer)
        controls_layout.addStretch()
        self._controls_widget.setVisible(False)
        layout.addWidget(self._controls_widget, 1)

        self._footer_widget = QWidget()
        self._footer_widget.setObjectName("footer")
        self._footer_widget.setFixedHeight(44)
        footer_layout = QHBoxLayout(self._footer_widget)
        footer_layout.setContentsMargins(0, 5, 0, 2)
        self._copyright_label = QLabel("© 2026 stemma")
        self._copyright_label.setObjectName("copyright")
        self._copyright_label.setFixedHeight(36)
        self._copyright_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        footer_layout.addWidget(self._copyright_label)
        footer_layout.addStretch()
        self._arpeggio_label = AnimatedArpeggioWidget(self._theme)
        footer_layout.addWidget(self._arpeggio_label)
        layout.addWidget(self._footer_widget)

        self._bind_component_aliases()
        self._connect_component_signals()

        self._pitch_debounce = QTimer(self)
        self._pitch_debounce.setSingleShot(True)
        self._pitch_debounce.setInterval(200)
        self._pitch_debounce.timeout.connect(self._flush_pending_pitch)
        self._pending_pitch: int | None = None

        self._speed_debounce = QTimer(self)
        self._speed_debounce.setSingleShot(True)
        self._speed_debounce.setInterval(100)
        self._speed_debounce.timeout.connect(self._flush_pending_speed)
        self._pending_speed: float | None = None

        self._trainer_enabled = False
        self._trainer_start_speed = 0.75
        self._tap_times: list[float] = []

    def _bind_component_aliases(self) -> None:
        """Retain private names used by existing callers during extraction."""
        transport = self._transport_bar
        self._play_btn = transport.play_button
        self._stop_btn = transport.stop_button
        self._record_btn = transport.record_button
        self._time_label = transport.time_label
        self._master_vol_label_prefix = transport.master_volume_prefix
        self._master_volume_slider = transport.master_volume_slider
        self._master_volume_label = transport.master_volume_label
        self._waveform_frame = transport.waveform_frame
        self._waveform = transport.waveform
        self._play_icon = transport.play_icon
        self._pause_icon = transport.pause_icon
        self._stop_icon = transport.stop_icon
        self._record_icon = transport.record_icon

        practice = self._practice_rack
        self._count_in_label = practice._count_in_label
        self._ci_label = practice._count_in_prefix
        self._count_in_toggle = practice._count_in_toggle
        self._count_in_beats_spin = practice._count_in_beats_spin
        self._count_in_repeats_cb = practice._count_in_repeats
        self._loop_a_btn = practice._loop_a_button
        self._loop_b_btn = practice._loop_b_button
        self._loop_toggle_btn = practice._loop_toggle_button
        self._loop_clear_btn = practice._loop_clear_button
        self._loop_label = practice._loop_label
        self._speed_label = practice._speed_label
        self._speed_combo = practice._speed_combo
        self._speed_status = practice._speed_status
        self._pitch_label = practice._pitch_label
        self._pitch_spin = practice._pitch_spin
        self._trainer_check = practice._trainer_check
        self._trainer_start_combo = practice._trainer_start_combo
        self._trainer_status = practice._trainer_status
        self._metro_label = practice._metronome_label
        self._metronome_toggle = practice._metronome_toggle
        self._bpm_spin = practice._bpm_spin
        self._tap_btn = practice._tap_button
        self._beat_sync_btn = practice._beat_sync_button
        self._beat_nudge_spin = practice._beat_nudge_spin
        self._metronome_vol_slider = practice._metronome_volume_slider
        self._metronome_vol_combo = practice._metronome_volume_combo

        song_info = self._song_info_bar
        self._key_label = song_info.key_label
        self._chord_label = song_info.chord_label
        self._detected_bpm_label = song_info.detected_bpm_label

        mixer = self._stem_mixer
        self._stem_rows = mixer.stem_rows
        self._recording_rows = mixer.recording_rows
        self._mixer_label = mixer._mixer_label
        self._stems_frame = mixer._stems_frame
        self._stem_container = mixer._stem_container
        self._recordings_label = mixer._recordings_label
        self._recordings_frame = mixer._recordings_frame
        self._recordings_container = mixer._recordings_container

    def _connect_component_signals(self) -> None:
        """Route narrow component intents through facade coordination."""
        transport = self._transport_bar
        transport.play_pause_requested.connect(self._on_play_pause)
        transport.stop_requested.connect(self._on_stop)
        transport.record_toggled.connect(self._on_record_toggled)
        transport.master_volume_changed.connect(
            self._on_master_volume_requested
        )
        transport.seek_requested.connect(self._on_waveform_seek)

        practice = self._practice_rack
        practice.loop_a_requested.connect(self.set_loop_a)
        practice.loop_b_requested.connect(self.set_loop_b)
        practice.loop_toggled.connect(self._on_loop_toggled)
        practice.loop_clear_requested.connect(self._on_clear_loop)
        practice.speed_changed.connect(
            lambda _speed: self._on_speed_changed(
                self._speed_combo.currentIndex()
            )
        )
        practice.pitch_changed.connect(self._on_pitch_changed)
        practice.trainer_toggled.connect(self._on_trainer_toggled)
        practice.trainer_start_changed.connect(
            lambda _speed: self._on_trainer_start_changed(
                self._trainer_start_combo.currentIndex()
            )
        )
        practice.metronome_toggled.connect(self._on_metronome_toggled)
        practice.bpm_changed.connect(self._on_bpm_changed)
        practice.tap_requested.connect(self._on_tap)
        practice.beat_sync_toggled.connect(self._on_beat_sync_toggled)
        practice.beat_nudge_changed.connect(self._on_beat_nudge_changed)
        practice.metronome_volume_changed.connect(
            self._on_metronome_volume_requested
        )
        practice.count_in_toggled.connect(self._on_count_in_toggled)
        practice.count_in_beats_changed.connect(
            self._on_count_in_beats_changed
        )
        practice.count_in_repeats_toggled.connect(
            self._on_count_in_repeats_toggled
        )

        self._song_info_bar.key_redetect_requested.connect(
            self._request_key_redetection
        )
        self._song_info_bar.bpm_redetect_requested.connect(
            self._request_bpm_redetection
        )
        self._stem_mixer.mix_changed.connect(self._on_mixer_mix_changed)

    def _request_key_redetection(self) -> None:
        if self._player.stems:
            self._redetect_key_only()

    def _request_bpm_redetection(self) -> None:
        if self._player.stems:
            self._redetect_bpm_only()

    def _on_master_volume_requested(self, volume: float) -> None:
        self._player.set_master_volume(volume)

    def _on_metronome_volume_requested(self, volume: float) -> None:
        self._player.set_metronome_volume(volume)

    def _setup_ui(self) -> None:
        self._setup_component_ui()

    def apply_theme(self, theme: str, colors: dict[str, str]) -> None:
        """Switch all theme-dependent visuals to *theme*."""
        self._theme = theme
        self._transport_bar.apply_theme(colors, self._player.is_playing)
        self._play_icon = self._transport_bar.play_icon
        self._pause_icon = self._transport_bar.pause_icon
        self._stop_icon = self._transport_bar.stop_icon
        self._practice_rack.apply_theme(colors)

        frame = int(
            self._player.current_seconds * self._player.sample_rate
        )
        chord = (
            self._player.chord_at(frame)
            if self._player.chord_sequence
            else None
        )
        effective = None
        pitch = int(self._player.pitch_semitones)
        if self._detected_key_raw and pitch:
            effective = transpose_key(self._detected_key_raw, pitch)
        self._song_info_bar.apply_theme(
            theme,
            effective_key=effective,
            pitch=pitch,
            chord=chord,
            has_chords=bool(self._player.chord_sequence),
        )

        self._empty_logo.set_theme(theme)
        self._arpeggio_label.set_theme(theme)
        self._stem_mixer.apply_theme(theme)
        if self._cached_stem_peaks is not None:
            self._apply_stem_lanes_to_waveform(self._cached_stem_peaks)

    def play_intro_animation(self, with_sound: bool = False) -> None:
        """Trigger the main logo's intro animation (notes + waves)."""
        self._empty_logo.play_intro(with_sound=with_sound)

    def _connect_signals(self) -> None:
        self._player.position_changed.connect(self._on_position_changed)
        self._player.state_changed.connect(self._on_state_changed)
        self._player.play_finished.connect(self._on_play_finished)
        self._player.speed_changed.connect(self._on_speed_applied)
        self._player.pitch_changed.connect(self._on_pitch_applied)
        self._player.stretch_started.connect(self._on_stretch_started)
        self._player.stretch_progress.connect(self._on_stretch_progress)
        self._player.stretch_finished.connect(self._on_stretch_finished)
        self._player.loop_wrapped.connect(self._on_loop_wrapped)

    def set_stem_names(self, stem_names: list[str]) -> None:
        """Populate the stem mixer with rows for each stem."""
        self._invalidate_detection()
        self._cached_stem_peaks = None
        has_stems = bool(stem_names)
        self._empty_widget.setVisible(not has_stems)
        self._controls_widget.setVisible(has_stems)
        self._stem_mixer.set_stem_names(stem_names)

        self._speed_combo.blockSignals(True)
        self._speed_combo.setCurrentText("1.0x")
        self._speed_combo.blockSignals(False)
        self._speed_status.setText("")

        # Kill any in-flight debounce from the previous song so a pending
        # scroll doesn't fire set_pitch / set_speed against the freshly
        # loaded stems.
        self._pitch_debounce.stop()
        self._pending_pitch = None
        self._speed_debounce.stop()
        self._pending_speed = None

        self._pitch_spin.blockSignals(True)
        self._pitch_spin.setValue(0)
        self._pitch_spin.blockSignals(False)

        # Reset the trainer for the new song (speed was reset to 1.0x
        # above); leave the chosen start-speed preset as the user set it.
        self._trainer_check.blockSignals(True)
        self._trainer_check.setChecked(False)
        self._trainer_check.blockSignals(False)
        self._trainer_enabled = False
        self._update_trainer_status()

        self._record_btn.blockSignals(True)
        self._record_btn.setChecked(False)
        self._record_btn.blockSignals(False)
        self.update_record_button_state()

        # Auto-detect if no beat grid has been loaded yet.
        # Old sessions are handled by the det_ver gate in main_window:
        # if det_ver < 4, beat_times are not restored, so this fires.
        if has_stems and not self._player.beat_times:
            self.start_detection()

        if stem_names:
            self._waveform.set_loading(True)
            self._do_recompute_peaks()
        else:
            self._do_recompute_peaks()

    def clear_song(self) -> None:
        """Return to the empty logo state."""
        self._detach_detection_worker()
        self.set_stem_names([])
        self._hint_label.setText("Drop an audio file or use File > Import")
        self._cached_stem_peaks = None
        self._waveform.set_loading(False)
        self._waveform.set_stem_lanes([], muted=set(), soloed=set())
        self._waveform.set_position(0.0)
        self._time_label.setText("0:00 / 0:00")
        self._key_label.setText("")
        self._key_label.setStyleSheet("")
        self._key_conf = ""
        self._key_label.setToolTip(
            "Detected musical key (double-click to re-detect)"
        )
        self._chord_label.setText("")
        self._chord_label.setStyleSheet("")
        self._chord_label.setToolTip("Detected chord (suggestion)")
        self._chord_timer.stop()
        self._detected_key_raw = ""
        self._detected_bpm_raw = ""
        self._detected_bpm_label.setText("")
        self._detected_bpm_label.setStyleSheet("")
        self._bpm_conf = ""
        self._detected_bpm_label.setToolTip(
            "Detected tempo — suggestion only (double-click to re-detect)"
        )
        self._beat_sync_btn.blockSignals(True)
        self._beat_sync_btn.setChecked(False)
        self._beat_sync_btn.setEnabled(False)
        self._beat_sync_btn.blockSignals(False)

        self._beat_nudge_spin.blockSignals(True)
        self._beat_nudge_spin.setValue(0)
        self._beat_nudge_spin.blockSignals(False)

    def show_loading(self, title: str) -> None:
        """Show an intentional empty-state message while stems load."""
        self.clear_song()
        self._hint_label.setText(f"Loading {title}...")

    def restore_stem_state(
        self,
        muted: set[str],
        soloed: set[str],
        volumes: dict[str, float],
    ) -> None:
        """Restore per-stem mute/solo/volume state from a saved session.

        Setting the UI widgets triggers the connected player methods, so
        this also updates the player state.
        """
        self._stem_mixer.restore_stem_state(muted, soloed, volumes)
        self._do_recompute_peaks()

    def restore_loop_state(
        self,
        loop_a: float | None,
        loop_b: float | None,
        looping: bool,
    ) -> None:
        """Restore A-B loop state from a saved session."""
        if loop_a is not None:
            self._player.set_loop_a(loop_a)
        if loop_b is not None:
            self._player.set_loop_b(loop_b)
        self._loop_toggle_btn.setChecked(looping)
        self._update_loop_label()
        self._update_waveform_loop_markers()

    def toggle_stem_mute(self, stem_name: str) -> None:
        """Toggle the mute state of a stem and update the UI button."""
        row = self._stem_rows.get(stem_name)
        if row is not None:
            is_muted = stem_name in self._player.muted_stems
            row.set_muted(not is_muted)

    # -- Transport slots --

    def _on_play_pause(self) -> None:
        if self._player.is_playing:
            self._player.pause()
        else:
            self._player.play()

    def _on_stop(self) -> None:
        self._player.stop()

    def _on_master_volume_slider_changed(self, value: int) -> None:
        """Slider moved -- mirror the new value into the player and label."""
        self._player.set_master_volume(value / 100.0)
        self._master_volume_label.setText(f"{value}%")

    def set_master_volume(self, volume: float) -> None:
        """Set master volume from any entry point (shortcut, session load).

        Keeps the slider, percent label, and player in sync so callers
        don't need to update each surface separately.
        """
        value = max(0, min(200, int(round(float(volume) * 100))))
        if self._master_volume_slider.value() != value:
            self._master_volume_slider.blockSignals(True)
            self._master_volume_slider.setValue(value)
            self._master_volume_slider.blockSignals(False)
        self._master_volume_label.setText(f"{value}%")
        self._player.set_master_volume(value / 100.0)

    def _on_waveform_seek(self, seconds: float) -> None:
        self._player.seek(seconds)

    def _on_position_changed(self, pos_s: float) -> None:
        total = self._player.total_seconds
        self._time_label.setText(
            f"{_format_time(pos_s)} / {_format_time(total)}"
        )
        if total > 0:
            self._waveform.set_position(pos_s / total)
        self.update_count_in_display()

        # Update BPM spinbox with instantaneous BPM when beat-synced.
        if self._beat_sync_btn.isChecked():
            frame = int(pos_s * self._player.sample_rate)
            ibpm = self._player.instantaneous_bpm_at(frame)
            if ibpm > 0:
                self._bpm_spin.blockSignals(True)
                self._bpm_spin.setValue(max(20, min(300, round(ibpm))))
                self._bpm_spin.blockSignals(False)

    def _update_chord_label(self) -> None:
        """Poll the player for the current chord and update the label."""
        if not self._player.is_playing:
            return
        frame = int(self._player.current_seconds * self._player.sample_rate)
        chord = self._player.chord_at(frame)
        self._chord_label.setText(
            self._badge_html("Chord:", chord if chord else "--")
        )

    def _on_state_changed(self, playing: bool) -> None:
        self._play_btn.setIcon(self._pause_icon if playing else self._play_icon)
        self._play_btn.setAccessibleName("Pause" if playing else "Play")
        if not playing:
            self._count_in_label.setText("")
            self._chord_timer.stop()
            # Show placeholder instead of the last detected chord.
            if self._player.chord_sequence:
                self._chord_label.setText(self._badge_html("Chord:", "--"))
            if not self._player.recording_armed:
                self._record_btn.blockSignals(True)
                self._record_btn.setChecked(False)
                self._record_btn.blockSignals(False)
        elif self._player.chord_sequence:
            self._chord_timer.start()

    def _on_play_finished(self) -> None:
        self._play_btn.setIcon(self._play_icon)
        self._play_btn.setAccessibleName("Play")
        self._chord_timer.stop()
        if self._player.chord_sequence:
            self._chord_label.setText(self._badge_html("Chord:", "--"))
        total = self._player.total_seconds
        if total > 0:
            self._waveform.set_position(
                self._player.current_seconds / total
            )
        else:
            self._waveform.set_position(0.0)

    def _build_stem_lanes(
        self, stem_peaks: dict[str, np.ndarray],
    ) -> list[tuple[str, np.ndarray, str]]:
        palette = (
            STEM_COLORS_DARK if self._theme == "dark" else STEM_COLORS_LIGHT
        )
        lanes: list[tuple[str, np.ndarray, str]] = []
        for name in self._stem_mixer.stem_names():
            peaks = stem_peaks.get(name)
            if peaks is None:
                continue
            if name in self._stem_mixer.recording_rows:
                color = RECORDING_COLOR
            else:
                color = palette.get(name, "#95a5a6")
            lanes.append((name, peaks, color))
        return lanes

    def _apply_stem_lanes_to_waveform(
        self, stem_peaks: dict[str, np.ndarray],
    ) -> None:
        try:
            _ = self._waveform
        except RuntimeError:
            return
        self._waveform.set_stem_lanes(
            self._build_stem_lanes(stem_peaks),
            muted=self._player.muted_stems,
            soloed=self._player.soloed_stems,
        )

    def _refresh_waveform_lane_mix(self) -> None:
        try:
            _ = self._waveform
        except RuntimeError:
            return
        if self._cached_stem_peaks is None:
            return
        self._waveform.update_lane_mix(
            muted=self._player.muted_stems,
            soloed=self._player.soloed_stems,
        )

    def _on_mixer_mix_changed(self) -> None:
        if self._cached_stem_peaks is not None:
            self._refresh_waveform_lane_mix()
            return
        self._recompute_peaks()

    def _recompute_peaks(self) -> None:
        """Schedule a debounced waveform peak recomputation.

        Rapid calls (e.g. dragging a volume slider) are batched so that
        only the final state triggers the expensive numpy computation.
        """
        self._peak_generation += 1
        self._peaks_timer.start()

    def _do_recompute_peaks(self) -> None:
        """Invalidate peak work and dispatch an immediate recomputation."""
        self._peak_generation += 1
        self._start_peak_computation()

    def _start_peak_computation(self) -> None:
        """Dispatch peak computation for the current generation."""
        self._peaks_timer.stop()

        if self._peak_future is not None and not self._peak_future.done():
            self._peak_refresh_pending = True
            return

        stems = self._player.stems
        if not stems:
            self._peak_refresh_pending = False
            self._cached_stem_peaks = None
            try:
                self._waveform.set_stem_lanes([], muted=set(), soloed=set())
            except RuntimeError:
                pass
            return

        self._peak_refresh_pending = False
        self._peak_future_generation = self._peak_generation
        self._peak_future = _get_peak_pool().submit(
            _compute_peaks_bg,
            stems=stems,
        )
        self._peak_poll_timer.start()

    def _poll_peak_future(self) -> None:
        """Check if the background peak computation has finished."""
        if self._peak_future is None or not self._peak_future.done():
            return
        self._peak_poll_timer.stop()
        future = self._peak_future
        future_generation = self._peak_future_generation
        self._peak_future = None
        self._peak_future_generation = None
        stale = future_generation != self._peak_generation
        failed = future.cancelled() or future.exception() is not None
        if not stale and not failed:
            try:
                stem_peaks = future.result()
            except Exception:
                failed = True
            else:
                self._on_peaks_computed(stem_peaks)

        refresh = stale or self._peak_refresh_pending
        self._peak_refresh_pending = False
        if refresh:
            self._start_peak_computation()

    def _on_peaks_computed(self, stem_peaks: dict) -> None:
        """Apply peak results from the background thread."""
        self._cached_stem_peaks = stem_peaks
        self._apply_stem_lanes_to_waveform(stem_peaks)
        try:
            _ = self._waveform
        except RuntimeError:
            return  # Widget was destroyed
        self._waveform.set_total_seconds(self._player.total_seconds)

    def _update_waveform_loop_markers(self) -> None:
        """Update loop marker positions on the waveform widget."""
        total = self._player.total_seconds
        a = self._player.loop_a
        b = self._player.loop_b
        if total > 0:
            a_ratio = a / total if a is not None else None
            b_ratio = b / total if b is not None else None
        else:
            a_ratio = None
            b_ratio = None
        self._waveform.set_loop_markers(a_ratio, b_ratio)

    # -- A-B loop slots --

    def set_loop_a(self) -> None:
        """Set loop A to the current playback position."""
        self._player.set_loop_a(self._player.current_seconds)
        self._update_loop_label()
        self._update_waveform_loop_markers()
        self._maybe_redetect_for_loop()
        self._update_trainer_status()

    def set_loop_b(self) -> None:
        """Set loop B to the current playback position."""
        self._player.set_loop_b(self._player.current_seconds)
        self._update_loop_label()
        self._update_waveform_loop_markers()
        self._maybe_redetect_for_loop()
        self._update_trainer_status()

    def _maybe_redetect_for_loop(self) -> None:
        """Re-run detection for the A-B region when both points are set."""
        a = self._player.loop_a
        b = self._player.loop_b
        if a is not None and b is not None and b > a:
            self.start_detection(start_sec=a, end_sec=b)

    def _on_loop_toggled(self, checked: bool) -> None:
        """Enable or disable A-B looping."""
        self._player.set_looping(checked)

    def _on_clear_loop(self) -> None:
        """Clear loop points and disable looping."""
        self._player.clear_loop()
        self._loop_toggle_btn.setChecked(False)
        self._update_loop_label()
        self._update_waveform_loop_markers()
        self._update_trainer_status()
        # Re-detect for the full song after clearing A-B region.
        if self._player.stems:
            self.start_detection()

    def toggle_looping(self) -> None:
        """Toggle the loop button state (e.g. from keyboard shortcut)."""
        self._loop_toggle_btn.setChecked(not self._loop_toggle_btn.isChecked())

    # -- Loop Trainer -----------------------------------------------------

    @property
    def trainer_enabled(self) -> bool:
        """Whether the loop trainer is currently on."""
        return self._trainer_enabled

    @property
    def trainer_start_speed(self) -> float:
        """The trainer's configured start-speed preset."""
        return self._trainer_start_speed

    def restore_trainer_state(
        self, enabled: bool, start_speed: float,
    ) -> None:
        """Restore trainer settings from a saved session.

        Sets the start-speed combo and enabled checkbox without letting
        the toggle immediately re-drop playback speed -- session restore
        drives speed separately.
        """
        idx = self._trainer_start_combo.findText(f"{start_speed}x")
        if idx >= 0:
            self._trainer_start_combo.blockSignals(True)
            self._trainer_start_combo.setCurrentIndex(idx)
            self._trainer_start_combo.blockSignals(False)
            self._trainer_start_speed = float(start_speed)
        self._trainer_check.blockSignals(True)
        self._trainer_check.setChecked(bool(enabled))
        self._trainer_check.blockSignals(False)
        self._trainer_enabled = bool(enabled)
        self._update_trainer_status()

    def _loop_region_valid(self) -> bool:
        """True when a usable A-B region (B > A) is set."""
        a = self._player.loop_a
        b = self._player.loop_b
        return a is not None and b is not None and b > a

    def _next_speed_up(self, speed: float) -> float | None:
        """Smallest speed preset greater than *speed*, capped at 1.0x.

        Returns None once the ramp has reached 1.0x (nothing above it
        that the trainer should step to).
        """
        for preset in SPEED_PRESETS:  # ascending
            if preset > speed + 1e-6 and preset <= 1.0 + 1e-6:
                return preset
        return None

    def _set_speed_preset(self, speed: float) -> None:
        """Drive the speed combo to *speed* (fires the normal render)."""
        idx = self._speed_combo.findText(f"{speed}x")
        if idx >= 0:
            self._speed_combo.setCurrentIndex(idx)

    def _on_trainer_toggled(self, checked: bool) -> None:
        """Enable/disable the trainer.

        Enabling with a valid loop region drops playback to the start
        speed; the ramp then advances one preset per loop repeat.
        """
        self._trainer_enabled = checked
        if checked and self._loop_region_valid():
            if self._player.speed != self._trainer_start_speed:
                self._set_speed_preset(self._trainer_start_speed)
        self._update_trainer_status()

    def _on_trainer_start_changed(self, _index: int) -> None:
        speed = self._trainer_start_combo.currentData()
        if speed is None:
            return
        self._trainer_start_speed = float(speed)
        # If the ramp is already above the new start, re-arm from it.
        if (self._trainer_enabled and self._loop_region_valid()
                and self._player.speed > self._trainer_start_speed):
            self._set_speed_preset(self._trainer_start_speed)
        self._update_trainer_status()

    def _on_loop_wrapped(self) -> None:
        """A-B loop repeated: step the trainer ramp up one preset."""
        if not self._trainer_enabled:
            return
        nxt = self._next_speed_up(self._player.speed)
        if nxt is not None:
            self._set_speed_preset(nxt)
        self._update_trainer_status()

    def _update_trainer_status(self) -> None:
        """Refresh the trainer readout next to the start combo."""
        if not self._trainer_enabled:
            self._trainer_status.setText("")
            return
        if not self._loop_region_valid():
            self._trainer_status.setText("(set an A-B loop)")
            return
        cur = self._player.speed
        if cur >= 1.0:
            self._trainer_status.setText("at 1.0x")
        else:
            self._trainer_status.setText(f"now {cur:g}x")

    def _update_loop_label(self) -> None:
        """Update the loop info label with current A/B points."""
        a = self._player.loop_a
        b = self._player.loop_b
        parts = []
        if a is not None:
            parts.append(f"A: {_format_time(a)}")
        if b is not None:
            parts.append(f"B: {_format_time(b)}")
        self._loop_label.setText("  ".join(parts))

    # -- Speed control slots --

    def _on_speed_changed(self, index: int) -> None:
        """User selected a speed preset from the combo box.

        Debounced so burst input (Shift+Up/Down cycling) coalesces into
        a single render; any in-flight render is cancelled immediately
        to free CPU while the user is still choosing.
        """
        speed = self._speed_combo.currentData()
        if speed is None:
            return
        self._pending_speed = float(speed)
        self._speed_debounce.start()
        self._player.cancel_stretch()

    def _flush_pending_speed(self) -> None:
        """Apply the latest speed value after the debounce window expires."""
        if self._pending_speed is None:
            return
        speed = self._pending_speed
        self._pending_speed = None
        self._player.set_speed(speed)

    def _on_speed_applied(self, speed: float) -> None:
        """Player finished stretching; update UI."""
        self._speed_combo.blockSignals(True)
        label = f"{speed}x"
        idx = self._speed_combo.findText(label)
        if idx >= 0:
            self._speed_combo.setCurrentIndex(idx)
        self._speed_combo.blockSignals(False)
        self._do_recompute_peaks()
        self.update_record_button_state()
        self._update_trainer_status()

    def cycle_speed(self, direction: int) -> None:
        """Cycle to the next/previous speed preset.

        Args:
            direction: +1 for faster, -1 for slower.
        """
        idx = self._speed_combo.currentIndex() + direction
        idx = max(0, min(idx, self._speed_combo.count() - 1))
        self._speed_combo.setCurrentIndex(idx)

    # -- Pitch control slots --

    def _on_pitch_changed(self, semitones: int) -> None:
        """User adjusted the pitch spinbox.

        We don't call ``player.set_pitch`` immediately; a 200ms debounce
        timer coalesces rapid scroll/arrow input into a single render.
        The descriptive status text is set by ``_on_stretch_started``
        once the worker actually spawns, not while the user is still
        adjusting the value.

        Any already-running render is cancelled right away so we stop
        wasting CPU on a stale target -- the new render will spawn when
        the debounce timer fires.
        """
        self._pending_pitch = int(semitones)
        self._pitch_debounce.start()
        # Kill the stale render immediately; the next one is queued.
        self._player.cancel_stretch()

    def _flush_pending_pitch(self) -> None:
        """Apply the latest pitch value after the debounce window expires."""
        if self._pending_pitch is None:
            return
        pitch = self._pending_pitch
        self._pending_pitch = None
        self._player.set_pitch(pitch)

    def _on_pitch_applied(self, semitones: int) -> None:
        """Player finished pitch-shifting; update UI."""
        self._pitch_spin.blockSignals(True)
        self._pitch_spin.setValue(int(semitones))
        self._pitch_spin.blockSignals(False)
        # Refresh peaks (buffers may have new lengths after a combined render)
        self._do_recompute_peaks()
        # Refresh the detected-key label to show the transposed key.
        self._refresh_key_label()
        self.update_record_button_state()

    def bump_pitch(self, direction: int) -> None:
        """Nudge the pitch spinbox by one semitone.

        Args:
            direction: +1 for up, -1 for down. Clamped by the spinbox range.
        """
        self._pitch_spin.setValue(self._pitch_spin.value() + direction)

    # -- Stretch worker progress indicator --
    #
    # Progress is shown *inside* (or immediately beside) the control the
    # user is manipulating.  The pitch spinbox carries a "(processing
    # 2/4)" suffix; the speed combo is followed by a small label with
    # the same suffix text (combos can't carry inline text of their own).
    # When both knobs are active, only the pitch suffix is shown -- the
    # single worker renders both transforms in one pass, and duplicating
    # the indicator confuses the eye.  The control stays enabled
    # throughout -- any new input cancels the in-flight render and queues
    # a fresh one via the debounce timer.

    def _on_stretch_started(self) -> None:
        """Begin showing render progress on the active control.

        The spinbox stays enabled so the user can keep scrubbing -- the
        player cancels the in-flight worker as soon as a new target is
        committed (see ``_flush_pending_pitch``).
        """
        self._update_stretch_indicator(0, 0)

    def _on_stretch_progress(self, current: int, total: int) -> None:
        """Update the live render indicator with per-stem progress."""
        self._update_stretch_indicator(current, total)

    def _on_stretch_finished(self) -> None:
        """Clear the render indicator and restore the idle display."""
        self._pitch_spin.setSuffix("")
        self._speed_status.setText("")

    def _update_stretch_indicator(self, current: int, total: int) -> None:
        """Paint render progress onto the pitch spinbox / speed label.

        The pitch spinbox's primary text ("+2 semi") is produced by
        :class:`PitchSpinBox.textFromValue`; this method only manages
        the trailing progress suffix (e.g. ``" (2/4)"``).  Speed
        progress uses a small floating label next to the speed combo,
        since QComboBox can't carry inline suffix text.

        The suffix format is deliberately minimal -- the spinbox/label
        is already tight, and the bare ``(N/M)`` form is still read
        as progress-out-of-total in context (the control is frozen
        grey while it's showing).  Earlier versions used
        ``(processing N/M)`` but that pushed the spinbox ~80 px wider.
        """
        pitch_on = self._player.pitch_semitones != 0
        speed_on = self._player.speed != 1.0

        if pitch_on:
            if total > 0:
                self._pitch_spin.setSuffix(f" ({current}/{total})")
            else:
                self._pitch_spin.setSuffix(" \u2026")
        else:
            self._pitch_spin.setSuffix("")

        if speed_on and not pitch_on:
            # Match the pitch spinbox suffix format verbatim so both
            # renders look visually identical.  Sits right after the
            # combo so the user sees "Speed: [1.5x] (2/4)".
            if total > 0:
                self._speed_status.setText(f"({current}/{total})")
            else:
                self._speed_status.setText("\u2026")
        else:
            # When pitch is active, the spinbox suffix already carries
            # the indicator; don't duplicate it in a floating label.
            self._speed_status.setText("")

    def _render_status_label(self, current: int, total: int) -> str:
        """Compose the floating-label status text (speed-only renders).

        Retained for tests and for callers that want a single-string
        status. For the pitch case the spinbox suffix is authoritative.
        """
        pitch_on = self._player.pitch_semitones != 0
        speed_on = self._player.speed != 1.0
        if pitch_on and speed_on:
            verb = "Transposing and time-stretching"
        elif pitch_on:
            verb = "Transposing"
        elif speed_on:
            verb = "Time-stretching"
        else:
            # Transition back to identity (fast path emits no progress).
            verb = "Rendering"
        if total > 0:
            return f"{verb} stems ({current}/{total})\u2026"
        return f"{verb} stems\u2026"

    # -- Metronome handlers --

    def _on_bpm_changed(self, value: int) -> None:
        """User changed the BPM spinbox."""
        self._player.set_metronome_bpm(float(value))

    def _on_tap(self) -> None:
        """Record a tap timestamp and update BPM."""
        now = time.monotonic()
        # Discard stale taps (> 2 seconds since last tap).
        if self._tap_times and (now - self._tap_times[-1]) > 2.0:
            self._tap_times.clear()
        self._tap_times.append(now)
        bpm = tap_tempo(self._tap_times)
        if bpm > 0:
            clamped = max(20, min(300, round(bpm)))
            self._bpm_spin.setValue(clamped)

    # -- Detection handlers --------------------------------------------------

    def eventFilter(self, obj, event):  # noqa: N802
        """Handle double-click on detection labels to re-detect."""
        if event.type() == QEvent.Type.MouseButtonDblClick:
            if obj is self._key_label and self._player.stems:
                self._redetect_key_only()
                return True
            if obj is self._detected_bpm_label and self._player.stems:
                self._redetect_bpm_only()
                return True
        return super().eventFilter(obj, event)

    def set_model_manager(self, manager) -> None:
        """Store the ModelManager and derive the beat model path."""
        self._model_manager = manager
        self._beat_model_path = manager.beat_model_path()

    def start_detection(
        self,
        start_sec: float | None = None,
        end_sec: float | None = None,
        *,
        _model_ready: bool = False,
    ) -> None:
        """Start background BPM/key detection.

        Called automatically when stems load and when A-B loop points
        change.  Results are shown as suggestions only — the metronome
        BPM spinbox is *not* modified.

        If the beat_this ONNX model has not been downloaded yet, it is
        fetched first and detection resumes on completion.
        """
        if self._closing or not self._player.stems:
            return
        self._detection_generation += 1
        generation = self._detection_generation
        self._pending_detect_args = None
        self._pending_detect_generation = None
        self._detach_detection_worker()

        # Ensure beat_this model is available.
        if (not _model_ready
                and self._model_manager
                and not self._model_manager.is_beat_model_downloaded()):
            self._pending_detect_args = (start_sec, end_sec)
            self._pending_detect_generation = generation
            self._start_beat_model_download()
            return

        self._run_detection(
            start_sec, end_sec, generation=generation,
        )

    def _invalidate_detection(self) -> None:
        """Invalidate pending and active detection work for old stems."""
        self._detection_generation += 1
        self._pending_detect_args = None
        self._pending_detect_generation = None
        self._detach_detection_worker()

    def _detach_detection_worker(self) -> None:
        """Disconnect and orphan the current detection worker, if any."""
        if self._detection_worker is None:
            return
        old = self._detection_worker
        safe_disconnect(old.completed)
        safe_disconnect(old.error)
        safe_disconnect(old.finished)
        if old.isRunning():
            self._orphaned_workers.append(old)
            old.finished.connect(
                self._on_orphaned_detection_finished,
                Qt.ConnectionType.QueuedConnection,
            )
            if not old.isRunning():
                self._reap_orphaned_detection_worker(old)
        else:
            old.setParent(None)
            old.deleteLater()
        self._detection_worker = None

    def _on_orphaned_detection_finished(self) -> None:
        worker = self.sender()
        self._reap_orphaned_detection_worker(worker)

    def _reap_orphaned_detection_worker(self, worker) -> None:
        """Release a stopped detection worker retained across replacement."""
        if worker in self._orphaned_workers:
            self._orphaned_workers.remove(worker)
            worker.setParent(None)
            worker.deleteLater()

    def _start_beat_model_download(self) -> None:
        """Download beat_this.onnx, then resume detection."""
        if self._beat_model_downloader is not None:
            return  # already downloading
        dim = LIGHT_COLORS if self._theme == "light" else DARK_COLORS
        dim_style = (
            f"background: {dim['surface0']}; "
            f"border: 1px solid {dim['surface1']}; "
            f"border-radius: 4px; "
            f"padding: 1px 6px; color: {dim['text']};"
        )
        self._detected_bpm_label.setStyleSheet(dim_style)
        self._detected_bpm_label.setText("downloading model...")
        self._key_label.setStyleSheet(dim_style)
        self._key_label.setText("downloading model...")

        dl = self._model_manager.download_beat_model()
        dl.download_complete.connect(self._on_beat_model_ready)
        dl.error.connect(self._on_beat_model_error)
        self._beat_model_downloader = dl
        dl.start()

    def _on_beat_model_ready(self, path: str) -> None:
        downloader = self.sender()
        if self._closing or downloader is not self._beat_model_downloader:
            return
        self._beat_model_downloader = None
        self._beat_model_path = path
        args = self._pending_detect_args or (None, None)
        generation = self._pending_detect_generation
        self._pending_detect_args = None
        self._pending_detect_generation = None
        if (
            generation != self._detection_generation
            or not self._player.stems
        ):
            return
        self.start_detection(*args, _model_ready=True)

    def _on_beat_model_error(self, msg: str) -> None:
        downloader = self.sender()
        if self._closing or downloader is not self._beat_model_downloader:
            return
        self._beat_model_downloader = None
        # Fall through without the model — librosa fallback.
        self._beat_model_path = None
        args = self._pending_detect_args or (None, None)
        generation = self._pending_detect_generation
        self._pending_detect_args = None
        self._pending_detect_generation = None
        if (
            generation != self._detection_generation
            or not self._player.stems
        ):
            return
        self.start_detection(*args, _model_ready=True)

    def _run_detection(
        self,
        start_sec: float | None = None,
        end_sec: float | None = None,
        *,
        generation: int | None = None,
    ) -> None:
        """Launch the DetectionWorker with the current model path."""
        if generation is None:
            generation = self._detection_generation
        if (
            self._closing
            or generation != self._detection_generation
            or not self._player.stems
        ):
            return
        dim = LIGHT_COLORS if self._theme == "light" else DARK_COLORS
        dim_style = (
            f"background: {dim['surface0']}; "
            f"border: 1px solid {dim['surface1']}; "
            f"border-radius: 4px; "
            f"padding: 1px 6px; color: {dim['text']};"
        )
        self._detected_bpm_label.setStyleSheet(dim_style)
        self._detected_bpm_label.setText("detecting...")
        self._key_label.setStyleSheet(dim_style)
        self._key_label.setText("detecting...")

        worker = DetectionWorker(
            stems=dict(self._player.stems),
            sample_rate=self._player.sample_rate,
            model_path=self._beat_model_path,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        worker.completed.connect(self._on_detect_completed)
        worker.error.connect(self._on_detect_error)
        worker.finished.connect(self._on_detect_finished)
        worker._detection_generation = generation
        self._detection_worker = worker
        worker.start()

    def _conf_color(self, level: str) -> str:
        """Return the themed colour string for a confidence level."""
        return CONFIDENCE_COLORS[self._theme].get(level, "")

    def _badge_style(self) -> str:
        """Return the CSS stylesheet for a detection badge label."""
        colors = LIGHT_COLORS if self._theme == "light" else DARK_COLORS
        return (
            f"background: {colors['surface0']}; "
            f"color: {colors['text']}; "
            f"border: 1px solid {colors['surface1']}; "
            f"border-radius: 4px; "
            f"padding: 1px 6px; "
            f"margin: 0px 1px;"
        )

    def _badge_html(self, label: str, value: str, color: str = "") -> str:
        """Build rich-text HTML for a detection badge: white label, coloured value."""
        colors = LIGHT_COLORS if self._theme == "light" else DARK_COLORS
        text_c = colors["text"]
        val_c = color or text_c
        if label:
            return (
                f'<span style="color:{text_c};">{label} </span>'
                f'<span style="color:{val_c};">{value}</span>'
            )
        return f'<span style="color:{val_c};">{value}</span>'

    def _refresh_key_label(self) -> None:
        """Re-render the key badge, showing ``detected → effective`` when pitch != 0."""
        if not self._detected_key_raw:
            return
        pitch = self._player.pitch_semitones
        key_c = self._conf_color(self._key_conf) if self._key_conf else ""
        self._key_label.setStyleSheet(self._badge_style())
        if pitch == 0:
            self._key_label.setText(
                self._badge_html("Key:", self._detected_key_raw, key_c)
            )
            self._key_label.setToolTip(
                f"Detected key: {self._detected_key_raw}\n"
                f"Confidence: {self._key_conf}\n"
                f"Double-click to re-detect"
            )
            return
        effective = transpose_key(self._detected_key_raw, pitch)
        shown = f"{self._detected_key_raw} \u2192 {effective}"
        self._key_label.setText(self._badge_html("Key:", shown, key_c))
        self._key_label.setToolTip(
            f"Detected key: {self._detected_key_raw}\n"
            f"Transposed by {pitch:+d} st: {effective}\n"
            f"Confidence: {self._key_conf}\n"
            f"Double-click to re-detect"
        )

    def _update_sync_btn_state(self, has_beats: bool) -> None:
        """Update beat-sync button enabled state."""
        self._beat_sync_btn.setEnabled(has_beats)
        if not has_beats and self._beat_sync_btn.isChecked():
            self._beat_sync_btn.setChecked(False)

    def restore_beat_times(self, beat_times: list[float], downbeat_times: list[float]) -> None:
        """Restore beat times from a saved session and update UI."""
        self._player.set_beat_times(beat_times, downbeat_times)
        self._update_sync_btn_state(len(beat_times) >= 2)

    def _on_detect_completed(self, result: DetectionResult) -> None:
        if not self._is_active_detection_sender():
            return
        # Store beat grid on the player.
        self._player.set_beat_times(result.beat_times, result.downbeat_times)

        # Enable/disable sync button based on whether beats were found.
        has_beats = len(result.beat_times) >= 2
        self._update_sync_btn_state(has_beats)

        badge = self._badge_style()

        # Update detected BPM label (suggestion only — does NOT set spinbox).
        if result.bpm > 0:
            bpm_rounded = round(result.bpm)
            self._bpm_conf = result.bpm_confidence
            self._detected_bpm_raw = f"~{bpm_rounded} BPM"
            bpm_c = self._conf_color(result.bpm_confidence)
            self._detected_bpm_label.setStyleSheet(badge)
            self._detected_bpm_label.setText(
                self._badge_html("Tempo:", self._detected_bpm_raw, bpm_c)
            )
            self._detected_bpm_label.setToolTip(
                f"Detected tempo: {result.bpm:.1f} BPM\n"
                f"Confidence: {result.bpm_confidence}\n"
                f"Double-click to re-detect"
            )
        else:
            self._detected_bpm_label.setText("")
            self._detected_bpm_label.setStyleSheet("")
            self._bpm_conf = ""
            self._detected_bpm_raw = ""

        # Update key label.
        if result.key:
            self._key_conf = result.key_confidence
            self._detected_key_raw = result.key
            self._refresh_key_label()
        else:
            self._key_label.setText("")
            self._key_label.setStyleSheet("")
            self._key_conf = ""
            self._detected_key_raw = ""

        # Store chord sequence and start polling timer.
        if result.chord_sequence:
            self._player.set_chord_sequence(result.chord_sequence)
            self._chord_label.setStyleSheet(badge)
            self._chord_label.setText(self._badge_html("Chord:", "--"))
            if self._player.is_playing:
                self._chord_timer.start()
        else:
            self._player.set_chord_sequence([])
            self._chord_label.setText("")
            self._chord_label.setStyleSheet("")
            self._chord_timer.stop()

    def _on_detect_error(self, msg: str) -> None:
        if not self._is_active_detection_sender():
            return
        for lbl in (self._key_label, self._detected_bpm_label,
                     self._chord_label):
            lbl.setText("")
            lbl.setStyleSheet("")
        self._detected_key_raw = ""
        self._detected_bpm_raw = ""
        self._chord_timer.stop()

    def _on_detect_finished(self) -> None:
        if self._is_active_detection_sender():
            self._detection_worker = None

    def _is_active_detection_sender(self) -> bool:
        worker = self.sender()
        return (
            worker is self._detection_worker
            and getattr(worker, "_detection_generation", None)
            == self._detection_generation
        )

    def _redetect_key_only(self) -> None:
        """Re-run detection but only update the key label."""
        if self._detection_worker is not None:
            return  # Already running.
        self._detection_generation += 1
        generation = self._detection_generation
        dim = LIGHT_COLORS if self._theme == "light" else DARK_COLORS
        self._key_label.setStyleSheet(
            f"background: {dim['surface0']}; "
            f"border: 1px solid {dim['surface1']}; "
            f"border-radius: 4px; "
            f"padding: 1px 6px; color: {dim['text']};"
        )
        self._key_label.setText("detecting...")

        worker = DetectionWorker(
            stems=dict(self._player.stems),
            sample_rate=self._player.sample_rate,
            model_path=self._beat_model_path,
        )
        worker.completed.connect(self._on_key_only_completed)
        worker.error.connect(self._on_key_only_error)
        worker.finished.connect(self._on_detect_finished)
        worker._detection_generation = generation
        self._detection_worker = worker
        worker.start()

    def _on_key_only_error(self, msg: str) -> None:
        """Clear the key badge if re-detection fails (was stuck on
        'detecting...' because no error handler was connected)."""
        if not self._is_active_detection_sender():
            return
        self._key_label.setText("")
        self._key_label.setStyleSheet("")

    def _on_key_only_completed(self, result: DetectionResult) -> None:
        """Update only the key label from a re-detection."""
        if not self._is_active_detection_sender():
            return
        if result.key:
            self._key_conf = result.key_confidence
            self._detected_key_raw = result.key
            self._refresh_key_label()
        else:
            self._key_label.setText("")
            self._key_label.setStyleSheet("")
            self._key_conf = ""
            self._detected_key_raw = ""

    def _redetect_bpm_only(self) -> None:
        """Re-run detection but only update the BPM label."""
        if self._detection_worker is not None:
            return  # Already running.
        self._detection_generation += 1
        generation = self._detection_generation
        dim = LIGHT_COLORS if self._theme == "light" else DARK_COLORS
        self._detected_bpm_label.setStyleSheet(
            f"background: {dim['surface0']}; "
            f"border: 1px solid {dim['surface1']}; "
            f"border-radius: 4px; "
            f"padding: 1px 6px; color: {dim['text']};"
        )
        self._detected_bpm_label.setText("detecting...")

        worker = DetectionWorker(
            stems=dict(self._player.stems),
            sample_rate=self._player.sample_rate,
            model_path=self._beat_model_path,
        )
        worker.completed.connect(self._on_bpm_only_completed)
        worker.error.connect(self._on_bpm_only_error)
        worker.finished.connect(self._on_detect_finished)
        worker._detection_generation = generation
        self._detection_worker = worker
        worker.start()

    def _on_bpm_only_error(self, msg: str) -> None:
        """Clear the tempo badge if re-detection fails (was stuck on
        'detecting...' because no error handler was connected)."""
        if not self._is_active_detection_sender():
            return
        self._detected_bpm_label.setText("")
        self._detected_bpm_label.setStyleSheet("")

    def _on_bpm_only_completed(self, result: DetectionResult) -> None:
        """Update only the BPM label from a re-detection."""
        if not self._is_active_detection_sender():
            return
        self._player.set_beat_times(result.beat_times, result.downbeat_times)
        self._update_sync_btn_state(len(result.beat_times) >= 2)
        if result.bpm > 0:
            bpm_rounded = round(result.bpm)
            self._bpm_conf = result.bpm_confidence
            self._detected_bpm_raw = f"~{bpm_rounded} BPM"
            bpm_c = self._conf_color(result.bpm_confidence)
            self._detected_bpm_label.setStyleSheet(self._badge_style())
            self._detected_bpm_label.setText(
                self._badge_html("Tempo:", self._detected_bpm_raw, bpm_c)
            )
            self._detected_bpm_label.setToolTip(
                f"Detected tempo: {result.bpm:.1f} BPM\n"
                f"Confidence: {result.bpm_confidence}\n"
                f"Double-click to re-detect"
            )
        else:
            self._detected_bpm_label.setText("")
            self._detected_bpm_label.setStyleSheet("")
            self._bpm_conf = ""
            self._detected_bpm_raw = ""

    @property
    def detected_key(self) -> str:
        """Return the raw detected key (e.g. "A minor"), or empty string."""
        return self._detected_key_raw

    @property
    def detected_bpm_text(self) -> str:
        """Return the raw detected BPM text (e.g. '~120 BPM'), or empty."""
        return self._detected_bpm_raw

    @property
    def key_confidence(self) -> str:
        """Return the last key confidence level, or empty string."""
        return self._key_conf

    @property
    def bpm_confidence(self) -> str:
        """Return the last BPM confidence level, or empty string."""
        return self._bpm_conf

    def restore_chord_sequence(
        self, chords: list[tuple[float, str]],
    ) -> None:
        """Restore a previously detected chord sequence from QSettings."""
        self._player.set_chord_sequence(chords)
        if chords:
            # Always apply badge style so the outline is visible even
            # before playback starts (fixes missing badge on app launch).
            self._chord_label.setStyleSheet(self._badge_style())
            self._chord_label.setText(self._badge_html("Chord:", "--"))
            if self._player.is_playing:
                self._chord_timer.start()

    def set_detected_key(self, key: str, confidence: str = "") -> None:
        """Restore a previously detected key label with colour and tooltip."""
        self._detected_key_raw = key
        if key:
            self._key_conf = confidence
            self._refresh_key_label()
        else:
            self._key_label.setText("")
            self._key_label.setStyleSheet("")
            self._key_conf = ""
            self._key_label.setToolTip(
                "Detected musical key (double-click to re-detect)"
            )

    def set_detected_bpm_text(
        self, text: str, confidence: str = "",
    ) -> None:
        """Restore a previously detected BPM suggestion label with colour and tooltip."""
        if text:
            # Strip old "Detected tempo: " prefix from legacy sessions.
            for prefix in ("Detected tempo: ~", "Detected tempo: "):
                if text.startswith(prefix):
                    text = text[len(prefix):]
                    break
            if not text.startswith("~"):
                text = f"~{text}"
            self._detected_bpm_raw = text
            self._bpm_conf = confidence
            c = self._conf_color(confidence) if confidence else ""
            self._detected_bpm_label.setStyleSheet(self._badge_style())
            self._detected_bpm_label.setText(
                self._badge_html("Tempo:", text, c)
            )
            parts = [f"Detected tempo: {text}"]
            if confidence:
                parts.append(f"Confidence: {confidence}")
            parts.append("Double-click to re-detect")
            self._detected_bpm_label.setToolTip("\n".join(parts))
        else:
            self._detected_bpm_label.setText("")
            self._detected_bpm_label.setStyleSheet("")
            self._bpm_conf = ""
            self._detected_bpm_raw = ""
            self._detected_bpm_label.setToolTip(
                "Detected tempo — suggestion only (double-click to re-detect)"
            )

    def _on_metronome_toggled(self, checked: bool) -> None:
        """User toggled the metronome on/off."""
        self._player.set_metronome_enabled(checked)

    def _on_beat_sync_toggled(self, checked: bool) -> None:
        """User toggled beat-sync mode for the metronome."""
        self._player.set_beat_sync_enabled(checked)
        if checked:
            # Make BPM spinbox read-only and show instantaneous BPM.
            self._bpm_spin.setReadOnly(True)
            self._bpm_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
            self._bpm_spin.setToolTip(
                "Metronome synced to detected beats (showing live BPM)"
            )
            self._tap_btn.setEnabled(False)
        else:
            self._bpm_spin.setReadOnly(False)
            self._bpm_spin.setButtonSymbols(
                QSpinBox.ButtonSymbols.UpDownArrows
            )
            self._bpm_spin.setToolTip("Metronome tempo")
            self._tap_btn.setEnabled(True)

    def _on_beat_nudge_changed(self, value: int) -> None:
        self._player.set_beat_sync_nudge_ms(float(value))

    @property
    def beat_sync_nudge_ms(self) -> float:
        """Return the user-selected sync nudge offset in ms."""
        return float(self._beat_nudge_spin.value())

    def set_beat_sync_nudge(self, offset_ms: float) -> None:
        """Restore the beat sync nudge offset from saved session."""
        self._beat_nudge_spin.setValue(int(offset_ms))

    @property
    def beat_sync_enabled(self) -> bool:
        """Return whether beat-sync mode is active."""
        return self._beat_sync_btn.isChecked()

    def set_beat_sync(self, enabled: bool) -> None:
        """Restore beat-sync state from a saved session."""
        self._beat_sync_btn.setChecked(enabled)

    def _on_metronome_vol_changed(self, value: int) -> None:
        """User moved the metronome volume slider."""
        self._player.set_metronome_volume(value / 100.0)
        self._metronome_vol_combo.blockSignals(True)
        self._metronome_vol_combo.setEditText(f"{value}%")
        self._metronome_vol_combo.blockSignals(False)

    def _on_metronome_vol_combo(self, index: int) -> None:
        """User selected a metronome volume preset."""
        value = self._metronome_vol_combo.itemData(index)
        if value is not None:
            self._metronome_vol_slider.setValue(value)

    def toggle_metronome(self) -> None:
        """Toggle metronome on/off (for keyboard shortcut)."""
        self._metronome_toggle.setChecked(
            not self._metronome_toggle.isChecked()
        )

    def restore_metronome_state(
        self, bpm: int, enabled: bool, volume: float
    ) -> None:
        """Restore metronome UI state from saved session."""
        self._bpm_spin.blockSignals(True)
        self._bpm_spin.setValue(bpm)
        self._bpm_spin.blockSignals(False)
        self._player.set_metronome_bpm(float(bpm))

        self._metronome_vol_slider.blockSignals(True)
        self._metronome_vol_slider.setValue(round(volume * 100))
        self._metronome_vol_slider.blockSignals(False)
        val_pct = round(volume * 100)
        text = f"{val_pct}%"
        idx = self._metronome_vol_combo.findText(text)
        if idx >= 0:
            self._metronome_vol_combo.blockSignals(True)
            self._metronome_vol_combo.setCurrentIndex(idx)
            self._metronome_vol_combo.blockSignals(False)
        self._player.set_metronome_volume(volume)

        self._metronome_toggle.blockSignals(True)
        self._metronome_toggle.setChecked(enabled)
        self._metronome_toggle.blockSignals(False)
        self._player.set_metronome_enabled(enabled)

    # -- Count-in handlers --

    def _on_count_in_toggled(self, checked: bool) -> None:
        """User toggled the count-in on/off."""
        self._player.set_count_in_enabled(checked)

    def _on_count_in_beats_changed(self, value: int) -> None:
        """User changed the count-in beat count."""
        self._player.set_count_in_beats(value)

    def _on_count_in_repeats_toggled(self, checked: bool) -> None:
        """User toggled count-in on loop repeats."""
        self._player.set_count_in_on_repeats(checked)

    def toggle_count_in(self) -> None:
        """Toggle count-in on/off (for keyboard shortcut)."""
        self._count_in_toggle.setChecked(
            not self._count_in_toggle.isChecked()
        )

    def update_count_in_display(self) -> None:
        """Update the count-in beat indicator from current player state."""
        if self._player.counting_in:
            beat = self._player.count_in_current_beat
            total = self._player.count_in_beats
            self._count_in_label.setText(f"{beat}/{total}")
        else:
            self._count_in_label.setText("")

    def restore_count_in_state(
        self, enabled: bool, beats: int, on_repeats: bool
    ) -> None:
        """Restore count-in UI state from a saved session."""
        self._count_in_beats_spin.blockSignals(True)
        self._count_in_beats_spin.setValue(beats)
        self._count_in_beats_spin.blockSignals(False)
        self._player.set_count_in_beats(beats)

        self._count_in_repeats_cb.blockSignals(True)
        self._count_in_repeats_cb.setChecked(on_repeats)
        self._count_in_repeats_cb.blockSignals(False)
        self._player.set_count_in_on_repeats(on_repeats)

        self._count_in_toggle.blockSignals(True)
        self._count_in_toggle.setChecked(enabled)
        self._count_in_toggle.blockSignals(False)
        self._player.set_count_in_enabled(enabled)

    # -- Recording handlers --

    def _on_record_toggled(self, checked: bool) -> None:
        """User toggled the record arm button."""
        self._player.arm_recording(checked)
        if checked and not self._player.recording_armed:
            self._record_btn.blockSignals(True)
            self._record_btn.setChecked(False)
            self._record_btn.blockSignals(False)

    def toggle_record(self) -> None:
        """Toggle recording arm (for keyboard shortcut).

        setChecked works even on a disabled button, so the shortcut
        must respect the disabled state itself -- otherwise R bypasses
        the take-limit and speed/pitch guards that disable the button.
        """
        if not self._record_btn.isEnabled():
            return
        self._record_btn.setChecked(not self._record_btn.isChecked())

    def add_recording_row(
        self, stem_name: str, display_name: str
    ) -> RecordingStemRow:
        """Add a recording take row to the recordings section."""
        return self._stem_mixer.add_recording_row(stem_name, display_name)

    def remove_recording_row(self, stem_name: str) -> None:
        """Remove a recording take row by stem name."""
        self._stem_mixer.remove_recording_row(stem_name)

    def clear_recording_rows(self) -> None:
        """Remove all recording rows."""
        self._stem_mixer.clear_recording_rows()

    @property
    def recording_count(self) -> int:
        """Return the number of recording take rows currently shown."""
        return self._stem_mixer.recording_count

    @property
    def max_recordings_reached(self) -> bool:
        """Return True if the maximum number of recording takes is reached."""
        return self._stem_mixer.max_recordings_reached

    def update_record_button_state(self) -> None:
        """Sync Record button enabled state with current speed and pitch."""
        at_identity = (
            self._player.speed == 1.0
            and self._player.pitch_semitones == 0
        )
        self._record_btn.setEnabled(at_identity and self._player.has_stems)
        if not at_identity:
            reasons = []
            if self._player.speed != 1.0:
                reasons.append("1.0x speed")
            if self._player.pitch_semitones != 0:
                reasons.append("0 st pitch")
            self._record_btn.setToolTip(
                "Recording requires " + " and ".join(reasons)
            )
            if self._record_btn.isChecked():
                self._record_btn.blockSignals(True)
                self._record_btn.setChecked(False)
                self._record_btn.blockSignals(False)
                self._player.arm_recording(False)
        else:
            self._record_btn.setToolTip("Arm recording (R)")

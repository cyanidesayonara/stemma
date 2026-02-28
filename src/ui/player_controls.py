"""Player transport controls and per-stem mute/solo mixer.

Transport: Play/Pause, Stop, waveform display, time display.
Per-stem row: label, Mute button, Solo button, volume slider.
Color-coded stems. Full implementation in ticket #9.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.player import MultiTrackPlayer
from src.ui.styles import STEM_COLORS
from src.ui.waveform_widget import WaveformWidget
from src.waveform import compute_peaks


def _format_time(seconds: float) -> str:
    """Format seconds as mm:ss."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


class StemRow(QWidget):
    """A single stem row with label, mute, and solo buttons."""

    mix_changed = Signal()

    def __init__(self, stem_name: str, player: MultiTrackPlayer,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stem_name = stem_name
        self._player = player

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        color = STEM_COLORS.get(stem_name, "#95a5a6")

        self._label = QLabel(stem_name.capitalize())
        self._label.setFixedWidth(70)
        self._label.setStyleSheet(f"color: {color}; font-weight: bold;")
        layout.addWidget(self._label)

        self._mute_btn = QPushButton("M")
        self._mute_btn.setCheckable(True)
        self._mute_btn.setFixedSize(32, 28)
        self._mute_btn.setStyleSheet("padding: 2px;")
        self._mute_btn.setToolTip(f"Mute {stem_name}")
        self._mute_btn.toggled.connect(self._on_mute)
        layout.addWidget(self._mute_btn)

        self._solo_btn = QPushButton("S")
        self._solo_btn.setCheckable(True)
        self._solo_btn.setFixedSize(32, 28)
        self._solo_btn.setStyleSheet("padding: 2px;")
        self._solo_btn.setToolTip(f"Solo {stem_name}")
        self._solo_btn.toggled.connect(self._on_solo)
        layout.addWidget(self._solo_btn)

        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(100)
        self._volume_slider.setFixedWidth(120)
        self._volume_slider.setToolTip(f"{stem_name} volume")
        self._volume_slider.valueChanged.connect(self._on_volume)
        layout.addWidget(self._volume_slider)

        self._vol_label = QLabel("100%")
        self._vol_label.setFixedWidth(40)
        self._vol_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._vol_label)

        layout.addStretch()

    def _on_mute(self, checked: bool) -> None:
        self._player.set_mute(self._stem_name, checked)
        self.mix_changed.emit()

    def _on_solo(self, checked: bool) -> None:
        self._player.set_solo(self._stem_name, checked)
        self.mix_changed.emit()

    def _on_volume(self, value: int) -> None:
        gain = value / 100.0
        self._player.set_volume(self._stem_name, gain)
        self._vol_label.setText(f"{value}%")
        self.mix_changed.emit()

    def set_muted(self, muted: bool) -> None:
        """Programmatically set the mute button state (e.g. from keyboard shortcut)."""
        self._mute_btn.setChecked(muted)


class PlayerControls(QWidget):
    """Transport controls and stem mixer panel."""

    def __init__(self, player: MultiTrackPlayer,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._player = player
        self._stem_rows: dict[str, StemRow] = {}

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # -- Transport bar --
        transport = QHBoxLayout()

        self._play_btn = QPushButton("Play")
        self._play_btn.setFixedWidth(70)
        self._play_btn.clicked.connect(self._on_play_pause)
        transport.addWidget(self._play_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedWidth(60)
        self._stop_btn.clicked.connect(self._on_stop)
        transport.addWidget(self._stop_btn)

        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setFixedWidth(100)
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        transport.addWidget(self._time_label)

        transport.addStretch()
        layout.addLayout(transport)

        # -- Waveform display (replaces seek slider) --
        self._waveform = WaveformWidget()
        self._waveform.seek_requested.connect(self._on_waveform_seek)
        layout.addWidget(self._waveform)

        # -- A-B loop controls --
        loop_bar = QHBoxLayout()

        self._loop_a_btn = QPushButton("Set A")
        self._loop_a_btn.setFixedWidth(60)
        self._loop_a_btn.setToolTip("Set loop start point (A)")
        self._loop_a_btn.clicked.connect(self._on_set_loop_a)
        loop_bar.addWidget(self._loop_a_btn)

        self._loop_b_btn = QPushButton("Set B")
        self._loop_b_btn.setFixedWidth(60)
        self._loop_b_btn.setToolTip("Set loop end point (B)")
        self._loop_b_btn.clicked.connect(self._on_set_loop_b)
        loop_bar.addWidget(self._loop_b_btn)

        self._loop_toggle_btn = QPushButton("Loop")
        self._loop_toggle_btn.setCheckable(True)
        self._loop_toggle_btn.setFixedWidth(60)
        self._loop_toggle_btn.setToolTip("Toggle A-B loop (L)")
        self._loop_toggle_btn.toggled.connect(self._on_loop_toggled)
        loop_bar.addWidget(self._loop_toggle_btn)

        self._loop_clear_btn = QPushButton("Clear")
        self._loop_clear_btn.setFixedWidth(60)
        self._loop_clear_btn.setToolTip("Clear loop points")
        self._loop_clear_btn.clicked.connect(self._on_clear_loop)
        loop_bar.addWidget(self._loop_clear_btn)

        self._loop_label = QLabel("")
        self._loop_label.setStyleSheet("color: #585b70;")
        loop_bar.addWidget(self._loop_label)

        loop_bar.addStretch()
        layout.addLayout(loop_bar)

        # -- Stem mixer --
        self._mixer_label = QLabel("Stems")
        self._mixer_label.setObjectName("title-label")
        layout.addWidget(self._mixer_label)

        self._stem_container = QVBoxLayout()
        layout.addLayout(self._stem_container)

        # Placeholder when no stems are loaded.
        self._empty_label = QLabel("Import a song to get started.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #585b70; padding: 40px;")
        layout.addWidget(self._empty_label)

        layout.addStretch()

    def _connect_signals(self) -> None:
        self._player.position_changed.connect(self._on_position_changed)
        self._player.state_changed.connect(self._on_state_changed)
        self._player.play_finished.connect(self._on_play_finished)

    def set_stem_names(self, stem_names: list[str]) -> None:
        """Populate the stem mixer with rows for each stem."""
        # Clear existing rows.
        for row in self._stem_rows.values():
            row.setParent(None)
            row.deleteLater()
        self._stem_rows.clear()

        self._empty_label.setVisible(not stem_names)

        for name in stem_names:
            row = StemRow(name, self._player)
            row.mix_changed.connect(self._recompute_peaks)
            self._stem_container.addWidget(row)
            self._stem_rows[name] = row

        self._recompute_peaks()

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

    def _on_waveform_seek(self, seconds: float) -> None:
        self._player.seek(seconds)

    def _on_position_changed(self, pos_s: float) -> None:
        total = self._player.total_seconds
        self._time_label.setText(
            f"{_format_time(pos_s)} / {_format_time(total)}"
        )
        if total > 0:
            self._waveform.set_position(pos_s / total)

    def _on_state_changed(self, playing: bool) -> None:
        self._play_btn.setText("Pause" if playing else "Play")

    def _on_play_finished(self) -> None:
        self._play_btn.setText("Play")
        self._waveform.set_position(0.0)

    def _recompute_peaks(self) -> None:
        """Recompute waveform peaks from current stem/mix state."""
        stems = self._player.stems
        if not stems:
            return
        peaks = compute_peaks(
            stems=stems,
            muted=self._player.muted_stems,
            soloed=self._player.soloed_stems,
            volumes=self._player.volumes,
            num_bins=2000,
        )
        self._waveform.set_peaks(peaks)
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

    def _on_set_loop_a(self) -> None:
        """Set loop A to the current playback position."""
        self._player.set_loop_a(self._player.current_seconds)
        self._update_loop_label()
        self._update_waveform_loop_markers()

    def _on_set_loop_b(self) -> None:
        """Set loop B to the current playback position."""
        self._player.set_loop_b(self._player.current_seconds)
        self._update_loop_label()
        self._update_waveform_loop_markers()

    def _on_loop_toggled(self, checked: bool) -> None:
        """Enable or disable A-B looping."""
        self._player.set_looping(checked)

    def _on_clear_loop(self) -> None:
        """Clear loop points and disable looping."""
        self._player.clear_loop()
        self._loop_toggle_btn.setChecked(False)
        self._update_loop_label()
        self._update_waveform_loop_markers()

    def toggle_looping(self) -> None:
        """Toggle the loop button state (e.g. from keyboard shortcut)."""
        self._loop_toggle_btn.setChecked(not self._loop_toggle_btn.isChecked())

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

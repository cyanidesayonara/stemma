"""Player transport controls and per-stem mute/solo mixer.

Transport: Play/Pause, Stop, waveform display, time display.
Per-stem row: label, Mute button, Solo button, volume slider.
Color-coded stems. Full implementation in ticket #9.
"""

import os

import numpy as np

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.player import SPEED_PRESETS, MultiTrackPlayer
from src.ui.styles import STEM_COLORS
from src.ui.waveform_widget import WaveformWidget
from src.waveform import compute_peaks

_PEAK_DEBOUNCE_MS = 80
_ICON_SIZE = 24
_ICON_COLOR = QColor("#cdd6f4")


def _make_icon(draw_fn) -> QIcon:
    """Create a crisp QIcon by painting with *draw_fn(painter, size)*."""
    pixmap = QPixmap(QSize(_ICON_SIZE, _ICON_SIZE))
    pixmap.fill(Qt.GlobalColor.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_ICON_COLOR)
    draw_fn(p, _ICON_SIZE)
    p.end()
    return QIcon(pixmap)


def _draw_play(p: QPainter, s: int) -> None:
    m = int(s * 0.2)
    from PySide6.QtGui import QPolygonF
    from PySide6.QtCore import QPointF
    p.drawPolygon(QPolygonF([
        QPointF(m + 2, m), QPointF(s - m, s / 2), QPointF(m + 2, s - m),
    ]))


def _draw_pause(p: QPainter, s: int) -> None:
    m = int(s * 0.22)
    w = int(s * 0.18)
    p.drawRect(m, m, w, s - 2 * m)
    p.drawRect(s - m - w, m, w, s - 2 * m)


def _draw_stop(p: QPainter, s: int) -> None:
    m = int(s * 0.22)
    p.drawRect(m, m, s - 2 * m, s - 2 * m)


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
        self._mute_btn.setAccessibleName(f"Mute {stem_name}")
        self._mute_btn.toggled.connect(self._on_mute)
        layout.addWidget(self._mute_btn)

        self._solo_btn = QPushButton("S")
        self._solo_btn.setCheckable(True)
        self._solo_btn.setFixedSize(32, 28)
        self._solo_btn.setStyleSheet("padding: 2px;")
        self._solo_btn.setToolTip(f"Solo {stem_name}")
        self._solo_btn.setAccessibleName(f"Solo {stem_name}")
        self._solo_btn.toggled.connect(self._on_solo)
        layout.addWidget(self._solo_btn)

        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 200)
        self._volume_slider.setValue(100)
        self._volume_slider.setFixedWidth(120)
        self._volume_slider.setToolTip(f"{stem_name} volume (0–200%, double-click to reset)")
        self._volume_slider.valueChanged.connect(self._on_volume)
        self._volume_slider.mouseDoubleClickEvent = lambda _: self._volume_slider.setValue(100)
        layout.addWidget(self._volume_slider)

        self._vol_label = QLabel("100%")
        self._vol_label.setFixedWidth(45)
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

        self._peaks_timer = QTimer(self)
        self._peaks_timer.setSingleShot(True)
        self._peaks_timer.setInterval(_PEAK_DEBOUNCE_MS)
        self._peaks_timer.timeout.connect(self._do_recompute_peaks)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # ── Empty state (shown when no song is loaded) ──
        self._empty_widget = QWidget()
        empty_layout = QVBoxLayout(self._empty_widget)
        empty_layout.addStretch(1)  # Top spacer

        self._empty_logo = QLabel()
        logo_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "assets", "icons", "logo_main_dark.svg",
        )
        renderer = QSvgRenderer(logo_path)
        if renderer.isValid():
            image = QImage(600, 370, QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(0)
            painter = QPainter(image)
            renderer.render(painter)
            painter.end()
            pixmap = QPixmap.fromImage(image).scaled(
                300, 185, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._empty_logo.setPixmap(pixmap)
        self._empty_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(self._empty_logo)

        hint = QLabel("Drop an audio file or use File > Import")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #585b70; padding: 10px;")
        empty_layout.addWidget(hint)

        empty_layout.addStretch(1)  # Bottom spacer (centers content vertically)

        layout.addWidget(self._empty_widget, 1)

        # ── Player controls (hidden until a song is loaded) ──
        self._controls_widget = QWidget()
        controls_layout = QVBoxLayout(self._controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        # -- Transport bar --
        transport = QHBoxLayout()

        self._play_icon = _make_icon(_draw_play)
        self._pause_icon = _make_icon(_draw_pause)
        self._stop_icon = _make_icon(_draw_stop)

        self._play_btn = QPushButton()
        self._play_btn.setIcon(self._play_icon)
        self._play_btn.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        self._play_btn.setFixedSize(36, 36)
        self._play_btn.setToolTip("Play / Pause (Space)")
        self._play_btn.setAccessibleName("Play")
        self._play_btn.clicked.connect(self._on_play_pause)
        transport.addWidget(self._play_btn)

        self._stop_btn = QPushButton()
        self._stop_btn.setIcon(self._stop_icon)
        self._stop_btn.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        self._stop_btn.setFixedSize(36, 36)
        self._stop_btn.setToolTip("Stop (S)")
        self._stop_btn.setAccessibleName("Stop")
        self._stop_btn.clicked.connect(self._on_stop)
        transport.addWidget(self._stop_btn)

        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setFixedWidth(100)
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        transport.addWidget(self._time_label)

        transport.addStretch()
        controls_layout.addLayout(transport)

        # -- Waveform display --
        self._waveform = WaveformWidget()
        self._waveform.seek_requested.connect(self._on_waveform_seek)
        controls_layout.addWidget(self._waveform)

        # -- A-B loop controls --
        loop_bar = QHBoxLayout()

        self._loop_a_btn = QPushButton("Set A")
        self._loop_a_btn.setFixedWidth(60)
        self._loop_a_btn.setToolTip("Set loop start point (A)")
        self._loop_a_btn.clicked.connect(self.set_loop_a)
        loop_bar.addWidget(self._loop_a_btn)

        self._loop_b_btn = QPushButton("Set B")
        self._loop_b_btn.setFixedWidth(60)
        self._loop_b_btn.setToolTip("Set loop end point (B)")
        self._loop_b_btn.clicked.connect(self.set_loop_b)
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
        controls_layout.addLayout(loop_bar)

        # -- Speed control --
        speed_bar = QHBoxLayout()

        speed_bar.addWidget(QLabel("Speed:"))

        self._speed_combo = QComboBox()
        for preset in SPEED_PRESETS:
            self._speed_combo.addItem(f"{preset}x", preset)
        self._speed_combo.setCurrentText("1.0x")
        self._speed_combo.setFixedWidth(80)
        self._speed_combo.setToolTip("Playback speed ([ / ])")
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        speed_bar.addWidget(self._speed_combo)

        self._speed_status = QLabel("")
        self._speed_status.setStyleSheet("color: #585b70;")
        speed_bar.addWidget(self._speed_status)

        speed_bar.addStretch()
        controls_layout.addLayout(speed_bar)

        # -- Stem mixer --
        self._mixer_label = QLabel("Stems")
        self._mixer_label.setObjectName("title-label")
        controls_layout.addWidget(self._mixer_label)

        self._stem_container = QVBoxLayout()
        controls_layout.addLayout(self._stem_container)

        controls_layout.addStretch()

        self._controls_widget.setVisible(False)
        layout.addWidget(self._controls_widget, 1)

        # ── Footer bar (pinned to bottom, always visible) ──
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)

        copyright_label = QLabel("\u00A9 2026 stemma")
        copyright_label.setStyleSheet("color: #45475a; font-size: 10pt;")
        footer.addWidget(copyright_label, alignment=Qt.AlignmentFlag.AlignVCenter)

        footer.addStretch()

        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        arpeggio_path = os.path.join(root, "assets", "icons", "logo_arpeggio_dark.svg")
        renderer2 = QSvgRenderer(arpeggio_path)
        if renderer2.isValid():
            img2 = QImage(840, 240, QImage.Format.Format_ARGB32_Premultiplied)
            img2.fill(0)
            p2 = QPainter(img2)
            renderer2.render(p2)
            p2.end()
            arpeggio_pixmap = QPixmap.fromImage(img2).scaled(
                300, 60, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            arpeggio_label = QLabel()
            arpeggio_label.setPixmap(arpeggio_pixmap)
            footer.addWidget(arpeggio_label, alignment=Qt.AlignmentFlag.AlignVCenter)

        layout.addLayout(footer)

    def _connect_signals(self) -> None:
        self._player.position_changed.connect(self._on_position_changed)
        self._player.state_changed.connect(self._on_state_changed)
        self._player.play_finished.connect(self._on_play_finished)
        self._player.speed_changed.connect(self._on_speed_applied)

    def set_stem_names(self, stem_names: list[str]) -> None:
        """Populate the stem mixer with rows for each stem."""
        # Clear existing rows.
        for row in self._stem_rows.values():
            row.setParent(None)
            row.deleteLater()
        self._stem_rows.clear()

        has_stems = bool(stem_names)
        self._empty_widget.setVisible(not has_stems)
        self._controls_widget.setVisible(has_stems)

        for name in stem_names:
            row = StemRow(name, self._player)
            row.mix_changed.connect(self._recompute_peaks)
            self._stem_container.addWidget(row)
            self._stem_rows[name] = row

        # Reset speed combo to 1.0x (load_stems resets the player speed).
        self._speed_combo.blockSignals(True)
        self._speed_combo.setCurrentText("1.0x")
        self._speed_combo.blockSignals(False)
        self._speed_status.setText("")

        if stem_names:
            self._do_recompute_peaks()

    def clear_song(self) -> None:
        """Return to the empty logo state."""
        self.set_stem_names([])
        self._waveform.set_peaks([])
        self._waveform.set_position(0.0)
        self._time_label.setText("0:00 / 0:00")

        self._do_recompute_peaks()
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
        self._play_btn.setIcon(self._pause_icon if playing else self._play_icon)
        self._play_btn.setAccessibleName("Pause" if playing else "Play")

    def _on_play_finished(self) -> None:
        self._play_btn.setIcon(self._play_icon)
        self._play_btn.setAccessibleName("Play")
        # Show cursor at current position (which may be loop-A, not 0).
        total = self._player.total_seconds
        if total > 0:
            self._waveform.set_position(
                self._player.current_seconds / total
            )
        else:
            self._waveform.set_position(0.0)

    def _recompute_peaks(self) -> None:
        """Schedule a debounced waveform peak recomputation.

        Rapid calls (e.g. dragging a volume slider) are batched so that
        only the final state triggers the expensive numpy computation.
        """
        self._peaks_timer.start()

    def _do_recompute_peaks(self) -> None:
        """Perform the actual waveform peak recomputation."""
        self._peaks_timer.stop()  # Cancel any pending debounced call.
        stems = self._player.stems
        if not stems:
            self._waveform.set_peaks(np.zeros(1, dtype=np.float32))
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

    def set_loop_a(self) -> None:
        """Set loop A to the current playback position."""
        self._player.set_loop_a(self._player.current_seconds)
        self._update_loop_label()
        self._update_waveform_loop_markers()

    def set_loop_b(self) -> None:
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

    # -- Speed control slots --

    def _on_speed_changed(self, index: int) -> None:
        """User selected a speed preset from the combo box."""
        speed = self._speed_combo.currentData()
        if speed is None:
            return
        self._speed_status.setText("Stretching..." if speed != 1.0 else "")
        self._player.set_speed(speed)

    def _on_speed_applied(self, speed: float) -> None:
        """Player finished stretching; update UI."""
        self._speed_status.setText("")
        # Update combo without re-triggering _on_speed_changed.
        self._speed_combo.blockSignals(True)
        label = f"{speed}x"
        idx = self._speed_combo.findText(label)
        if idx >= 0:
            self._speed_combo.setCurrentIndex(idx)
        self._speed_combo.blockSignals(False)
        self._do_recompute_peaks()

    def cycle_speed(self, direction: int) -> None:
        """Cycle to the next/previous speed preset.

        Args:
            direction: +1 for faster, -1 for slower.
        """
        idx = self._speed_combo.currentIndex() + direction
        idx = max(0, min(idx, self._speed_combo.count() - 1))
        self._speed_combo.setCurrentIndex(idx)

"""Player transport controls and per-stem mute/solo mixer.

Transport: Play/Pause, Stop, waveform display, time display.
Per-stem row: label, Mute button, Solo button, volume slider.
Color-coded stems. Full implementation in ticket #9.
"""

import time

import numpy as np

from PySide6.QtCore import QPointF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.metronome import tap_tempo
from src.player import SPEED_PRESETS, MultiTrackPlayer
from src.ui.animated_arpeggio import AnimatedArpeggioWidget
from src.ui.animated_logo import AnimatedLogoWidget
from src.ui.styles import (
    DARK_COLORS,
    RECORDING_COLOR,
    STEM_COLORS_DARK,
    STEM_COLORS_LIGHT,
)
from src.ui.waveform_widget import MiniWaveformWidget, WaveformWidget
from src.waveform import compute_peaks, compute_stem_peaks

_PEAK_DEBOUNCE_MS = 80
_ICON_SIZE = 24
_MAX_RECORDING_TAKES = 2
_MINI_WAVEFORM_WIDTH = 250


def _make_icon(draw_fn, color: QColor) -> QIcon:
    """Create a crisp QIcon by painting with *draw_fn(painter, size)*."""
    pixmap = QPixmap(QSize(_ICON_SIZE, _ICON_SIZE))
    pixmap.fill(Qt.GlobalColor.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    draw_fn(p, _ICON_SIZE)
    p.end()
    return QIcon(pixmap)


def _draw_play(p: QPainter, s: int) -> None:
    m = int(s * 0.2)
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


def _draw_record(p: QPainter, s: int) -> None:
    cx = s / 2.0
    r = s * 0.30
    p.drawEllipse(QPointF(cx, cx), r, r)


def _format_time(seconds: float) -> str:
    """Format seconds as mm:ss."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


class StemRow(QWidget):
    """A single stem row with label, mute, and solo buttons."""

    mix_changed = Signal()

    def __init__(self, stem_name: str, player: MultiTrackPlayer,
                 theme: str = "dark",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stem_name = stem_name
        self._player = player
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setStyleSheet("background: transparent;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        palette = STEM_COLORS_DARK if theme == "dark" else STEM_COLORS_LIGHT
        color = palette.get(stem_name, "#95a5a6")

        self._label = QLabel(stem_name.capitalize())
        self._label.setFixedWidth(70)
        self._label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self._label.setStyleSheet(f"color: {color}; font-weight: bold;")
        layout.addWidget(self._label)

        self._mini_waveform = MiniWaveformWidget(color, player)
        self._mini_waveform.seek_requested.connect(self._on_mini_seek)
        layout.addWidget(self._mini_waveform, 1)

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
        self._volume_slider.setToolTip(f"{stem_name} volume (0-200%, double-click to reset)")
        self._volume_slider.setAccessibleName(f"{stem_name} volume")
        self._volume_slider.valueChanged.connect(self._on_volume)
        self._volume_slider.mouseDoubleClickEvent = lambda _: self._volume_slider.setValue(100)
        layout.addWidget(self._volume_slider)

        self._vol_label = QLabel("100%")
        self._vol_label.setFixedWidth(45)
        self._vol_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(self._vol_label)

    def _on_mini_seek(self, seconds: float) -> None:
        self._player.seek(seconds)

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

    def set_soloed(self, soloed: bool) -> None:
        """Programmatically set the solo button state."""
        self._solo_btn.setChecked(soloed)

    def set_volume_slider(self, value: int) -> None:
        """Programmatically set the volume slider (0-200)."""
        self._volume_slider.setValue(value)

    def set_mini_peaks(self, peaks: "np.ndarray") -> None:
        """Update the mini waveform with new peak data."""
        self._mini_waveform.set_peaks(peaks)

    def apply_stem_theme(self, theme: str) -> None:
        """Update stem label and waveform color for the given theme."""
        palette = STEM_COLORS_DARK if theme == "dark" else STEM_COLORS_LIGHT
        color = palette.get(self._stem_name, "#95a5a6")
        self._label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self._mini_waveform._color = QColor(color)
        self._mini_waveform.update()


class RecordingStemRow(StemRow):
    """A stem row for a recording take, with delete and nudge controls."""

    delete_requested = Signal(str)

    def __init__(self, stem_name: str, display_name: str,
                 player: MultiTrackPlayer,
                 theme: str = "dark",
                 parent: QWidget | None = None) -> None:
        super().__init__(stem_name, player, theme, parent)

        self._label.setText(display_name)
        self._label.setStyleSheet(
            f"color: {RECORDING_COLOR}; font-weight: bold;"
        )
        self._mini_waveform._color = QColor(RECORDING_COLOR)

        lay = self.layout()
        insert_pos = lay.count()

        self._nudge_label = QLabel("Nudge:")
        self._nudge_label.setFixedWidth(46)
        lay.insertWidget(insert_pos, self._nudge_label)
        insert_pos += 1

        self._nudge_spin = QSpinBox()
        self._nudge_spin.setRange(-200, 200)
        self._nudge_spin.setValue(0)
        self._nudge_spin.setSuffix(" ms")
        self._nudge_spin.setFixedWidth(90)
        self._nudge_spin.setToolTip(
            f"Nudge {display_name} alignment (-200 to +200 ms)"
        )
        self._nudge_spin.setAccessibleName(f"Nudge {display_name}")
        self._nudge_spin.valueChanged.connect(self._on_nudge_changed)
        lay.insertWidget(insert_pos, self._nudge_spin)
        insert_pos += 1

        self._delete_btn = QPushButton("X")
        self._delete_btn.setFixedSize(28, 28)
        self._delete_btn.setStyleSheet("padding: 2px;")
        self._delete_btn.setToolTip(f"Delete {display_name}")
        self._delete_btn.setAccessibleName(f"Delete {display_name}")
        self._delete_btn.clicked.connect(
            lambda: self.delete_requested.emit(self._stem_name)
        )
        lay.insertWidget(insert_pos, self._delete_btn)

    def _on_nudge_changed(self, value: int) -> None:
        self._player.nudge_stem(self._stem_name, float(value))
        self.mix_changed.emit()


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
        self._peaks_timer.timeout.connect(self._do_recompute_peaks)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        colors = DARK_COLORS

        # -- Empty state (shown when no song is loaded) --
        self._empty_widget = QWidget()
        empty_layout = QVBoxLayout(self._empty_widget)
        empty_layout.addStretch(1)

        self._empty_logo = AnimatedLogoWidget(self._theme)
        empty_layout.addWidget(self._empty_logo, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._hint_label = QLabel("Drop an audio file or use File > Import")
        self._hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint_label.setStyleSheet(
            f"color: {colors['surface2']}; margin-top: 0px;"
        )
        empty_layout.addWidget(self._hint_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        empty_layout.addStretch(1)

        layout.addWidget(self._empty_widget, 1)

        # -- Player controls (hidden until a song is loaded) --
        self._controls_widget = QWidget()
        controls_layout = QVBoxLayout(self._controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        # -- Transport bar --
        transport = QHBoxLayout()

        icon_color = QColor(colors["text"])
        self._play_icon = _make_icon(_draw_play, icon_color)
        self._pause_icon = _make_icon(_draw_pause, icon_color)
        self._stop_icon = _make_icon(_draw_stop, icon_color)

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

        self._record_icon = _make_icon(
            _draw_record, QColor(RECORDING_COLOR)
        )
        self._record_btn = QPushButton()
        self._record_btn.setIcon(self._record_icon)
        self._record_btn.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        self._record_btn.setFixedSize(36, 36)
        self._record_btn.setCheckable(True)
        self._record_btn.setToolTip("Arm recording (R)")
        self._record_btn.setAccessibleName("Record")
        self._record_btn.toggled.connect(self._on_record_toggled)
        transport.addWidget(self._record_btn)

        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setFixedWidth(100)
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        transport.addWidget(self._time_label)

        transport.addStretch()
        controls_layout.addLayout(transport)

        # -- Waveform display --
        self._waveform_frame = QFrame()
        self._waveform_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._waveform_frame.setStyleSheet(
            f"QFrame {{ background-color: {colors['mantle']}; "
            f"border: 1px solid {colors['surface0']}; "
            f"border-radius: 6px; }}"
        )
        wf_layout = QVBoxLayout(self._waveform_frame)
        wf_layout.setContentsMargins(4, 4, 4, 4)

        self._waveform = WaveformWidget()
        self._waveform.seek_requested.connect(self._on_waveform_seek)
        wf_layout.addWidget(self._waveform)

        controls_layout.addWidget(self._waveform_frame)

        # -- Loop + Speed bar (merged) --
        loop_speed_bar = QHBoxLayout()

        self._loop_a_btn = QPushButton("Set A")
        self._loop_a_btn.setFixedWidth(60)
        self._loop_a_btn.setToolTip("Set loop start point (A)")
        self._loop_a_btn.setAccessibleName("Set loop A")
        self._loop_a_btn.clicked.connect(self.set_loop_a)
        loop_speed_bar.addWidget(self._loop_a_btn)

        self._loop_b_btn = QPushButton("Set B")
        self._loop_b_btn.setFixedWidth(60)
        self._loop_b_btn.setToolTip("Set loop end point (B)")
        self._loop_b_btn.setAccessibleName("Set loop B")
        self._loop_b_btn.clicked.connect(self.set_loop_b)
        loop_speed_bar.addWidget(self._loop_b_btn)

        self._loop_toggle_btn = QPushButton("Loop")
        self._loop_toggle_btn.setCheckable(True)
        self._loop_toggle_btn.setFixedWidth(60)
        self._loop_toggle_btn.setToolTip("Toggle A-B loop (L)")
        self._loop_toggle_btn.setAccessibleName("Toggle loop")
        self._loop_toggle_btn.toggled.connect(self._on_loop_toggled)
        loop_speed_bar.addWidget(self._loop_toggle_btn)

        self._loop_clear_btn = QPushButton("Clear")
        self._loop_clear_btn.setFixedWidth(60)
        self._loop_clear_btn.setToolTip("Clear loop points")
        self._loop_clear_btn.setAccessibleName("Clear loop")
        self._loop_clear_btn.clicked.connect(self._on_clear_loop)
        loop_speed_bar.addWidget(self._loop_clear_btn)

        self._loop_label = QLabel("")
        self._loop_label.setStyleSheet(f"color: {colors['surface2']};")
        loop_speed_bar.addWidget(self._loop_label)

        loop_speed_bar.addStretch()

        self._speed_label = QLabel("Speed:")
        loop_speed_bar.addWidget(self._speed_label)

        self._speed_combo = QComboBox()
        for preset in SPEED_PRESETS:
            self._speed_combo.addItem(f"{preset}x", preset)
        self._speed_combo.setCurrentText("1.0x")
        self._speed_combo.setFixedWidth(80)
        self._speed_combo.setToolTip("Playback speed ([ / ])")
        self._speed_combo.setAccessibleName("Playback speed")
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        loop_speed_bar.addWidget(self._speed_combo)

        self._speed_status = QLabel("")
        self._speed_status.setStyleSheet(f"color: {colors['surface2']};")
        loop_speed_bar.addWidget(self._speed_status)

        controls_layout.addLayout(loop_speed_bar)

        # -- Metronome + Count-in bar (merged) --
        metro_ci_bar = QHBoxLayout()

        self._metro_label = QLabel("Metronome:")
        metro_ci_bar.addWidget(self._metro_label)

        self._metronome_toggle = QPushButton()
        self._metronome_toggle.setCheckable(True)
        self._metronome_toggle.setFixedSize(36, 36)
        self._metronome_toggle.setToolTip("Toggle metronome (M)")
        self._metronome_toggle.setAccessibleName("Toggle metronome")
        self._metronome_toggle.setText("M")
        self._metronome_toggle.toggled.connect(self._on_metronome_toggled)
        metro_ci_bar.addWidget(self._metronome_toggle)

        self._bpm_spin = QSpinBox()
        self._bpm_spin.setRange(20, 300)
        self._bpm_spin.setValue(120)
        self._bpm_spin.setSuffix(" BPM")
        self._bpm_spin.setFixedWidth(110)
        self._bpm_spin.setToolTip("Metronome tempo")
        self._bpm_spin.setAccessibleName("Metronome BPM")
        self._bpm_spin.valueChanged.connect(self._on_bpm_changed)
        metro_ci_bar.addWidget(self._bpm_spin)

        self._tap_times: list[float] = []
        self._tap_btn = QPushButton("Tap")
        self._tap_btn.setFixedWidth(60)
        self._tap_btn.setToolTip("Tap to set tempo")
        self._tap_btn.setAccessibleName("Tap tempo")
        self._tap_btn.clicked.connect(self._on_tap)
        metro_ci_bar.addWidget(self._tap_btn)

        self._metronome_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._metronome_vol_slider.setRange(0, 200)
        self._metronome_vol_slider.setValue(50)
        self._metronome_vol_slider.setFixedWidth(70)
        self._metronome_vol_slider.setToolTip("Metronome volume")
        self._metronome_vol_slider.setAccessibleName("Metronome volume")
        self._metronome_vol_slider.valueChanged.connect(
            self._on_metronome_vol_changed
        )
        metro_ci_bar.addWidget(self._metronome_vol_slider)

        self._metronome_vol_label = QLabel("50%")
        self._metronome_vol_label.setFixedWidth(36)
        self._metronome_vol_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        metro_ci_bar.addWidget(self._metronome_vol_label)

        metro_ci_bar.addStretch()

        self._ci_label = QLabel("Count-in:")
        metro_ci_bar.addWidget(self._ci_label)

        self._count_in_toggle = QPushButton()
        self._count_in_toggle.setCheckable(True)
        self._count_in_toggle.setFixedSize(36, 36)
        self._count_in_toggle.setToolTip("Toggle count-in before playback (C)")
        self._count_in_toggle.setAccessibleName("Toggle count-in")
        self._count_in_toggle.setText("CI")
        self._count_in_toggle.toggled.connect(self._on_count_in_toggled)
        metro_ci_bar.addWidget(self._count_in_toggle)

        self._count_in_beats_spin = QSpinBox()
        self._count_in_beats_spin.setRange(1, 8)
        self._count_in_beats_spin.setValue(4)
        self._count_in_beats_spin.setSuffix(" beats")
        self._count_in_beats_spin.setFixedWidth(90)
        self._count_in_beats_spin.setToolTip("Number of count-in beats")
        self._count_in_beats_spin.setAccessibleName("Count-in beats")
        self._count_in_beats_spin.valueChanged.connect(
            self._on_count_in_beats_changed
        )
        metro_ci_bar.addWidget(self._count_in_beats_spin)

        self._count_in_repeats_cb = QCheckBox("Repeats")
        self._count_in_repeats_cb.setToolTip(
            "Also count in before each A-B loop repeat"
        )
        self._count_in_repeats_cb.setAccessibleName(
            "Count-in on loop repeats"
        )
        self._count_in_repeats_cb.toggled.connect(
            self._on_count_in_repeats_toggled
        )
        metro_ci_bar.addWidget(self._count_in_repeats_cb)

        self._count_in_label = QLabel("")
        self._count_in_label.setFixedWidth(40)
        self._count_in_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        metro_ci_bar.addWidget(self._count_in_label)

        controls_layout.addLayout(metro_ci_bar)

        # -- Stem mixer --
        self._mixer_label = QLabel("Stems")
        self._mixer_label.setObjectName("title-label")
        controls_layout.addWidget(self._mixer_label)

        self._stems_frame = QFrame()
        self._stems_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._stems_frame.setStyleSheet(
            f"QFrame {{ background-color: {colors['mantle']}; "
            f"border: 1px solid {colors['surface0']}; "
            f"border-radius: 6px; }}"
        )
        self._stem_container = QVBoxLayout(self._stems_frame)
        self._stem_container.setContentsMargins(6, 4, 6, 4)
        self._stem_container.setSpacing(2)
        controls_layout.addWidget(self._stems_frame)

        self._recordings_label = QLabel("Recordings")
        self._recordings_label.setObjectName("title-label")
        self._recordings_label.setVisible(False)
        controls_layout.addWidget(self._recordings_label)

        self._recordings_frame = QFrame()
        self._recordings_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._recordings_frame.setStyleSheet(
            f"QFrame {{ background-color: {colors['mantle']}; "
            f"border: 1px solid {colors['surface0']}; "
            f"border-radius: 6px; }}"
        )
        self._recordings_frame.setVisible(False)
        self._recordings_container = QVBoxLayout(self._recordings_frame)
        self._recordings_container.setContentsMargins(6, 4, 6, 4)
        self._recordings_container.setSpacing(2)
        controls_layout.addWidget(self._recordings_frame)

        controls_layout.addStretch()

        self._controls_widget.setVisible(False)
        layout.addWidget(self._controls_widget, 1)

        # -- Footer bar --
        self._footer_widget = QWidget()
        self._footer_widget.setFixedHeight(44)
        self._footer_widget.setStyleSheet(
            f"border-top: 1px solid {colors['surface0']};"
        )
        footer_layout = QHBoxLayout(self._footer_widget)
        footer_layout.setContentsMargins(0, 5, 0, 2)

        self._copyright_label = QLabel("\u00A9 2026 stemma")
        self._copyright_label.setFixedHeight(36)
        self._copyright_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self._copyright_label.setStyleSheet(
            f"color: {colors['surface1']}; font-size: 9pt; border: none;"
        )
        footer_layout.addWidget(self._copyright_label)

        footer_layout.addStretch()

        self._arpeggio_label = AnimatedArpeggioWidget(self._theme)
        footer_layout.addWidget(self._arpeggio_label)

        layout.addWidget(self._footer_widget)

    def apply_theme(self, theme: str, colors: dict[str, str]) -> None:
        """Switch all theme-dependent visuals to *theme*."""
        self._theme = theme
        icon_color = QColor(colors["text"])

        self._play_icon = _make_icon(_draw_play, icon_color)
        self._pause_icon = _make_icon(_draw_pause, icon_color)
        self._stop_icon = _make_icon(_draw_stop, icon_color)

        if self._player.is_playing:
            self._play_btn.setIcon(self._pause_icon)
        else:
            self._play_btn.setIcon(self._play_icon)
        self._stop_btn.setIcon(self._stop_icon)

        # Inline-styled labels
        self._hint_label.setStyleSheet(
            f"color: {colors['surface2']}; margin-top: 0px;"
        )
        self._loop_label.setStyleSheet(f"color: {colors['surface2']};")
        self._speed_status.setStyleSheet(f"color: {colors['surface2']};")
        self._footer_widget.setStyleSheet(
            f"border-top: 1px solid {colors['surface0']};"
        )
        self._copyright_label.setStyleSheet(
            f"color: {colors['surface1']}; font-size: 9pt; border: none;"
        )

        # Animated logos
        self._empty_logo.set_theme(theme)
        self._arpeggio_label.set_theme(theme)

        # Waveform frame and colors
        frame_style = (
            f"QFrame {{ background-color: {colors['mantle']}; "
            f"border: 1px solid {colors['surface0']}; "
            f"border-radius: 6px; }}"
        )
        self._waveform_frame.setStyleSheet(frame_style)
        self._stems_frame.setStyleSheet(frame_style)
        self._recordings_frame.setStyleSheet(frame_style)
        self._waveform.set_theme_colors(colors)

        for row in self._stem_rows.values():
            row.apply_stem_theme(theme)
        for row in self._recording_rows.values():
            row.apply_stem_theme(theme)

    def play_intro_animation(self, with_sound: bool = False) -> None:
        """Trigger the main logo's intro animation (notes + waves)."""
        self._empty_logo.play_intro(with_sound=with_sound)

    def _connect_signals(self) -> None:
        self._player.position_changed.connect(self._on_position_changed)
        self._player.state_changed.connect(self._on_state_changed)
        self._player.play_finished.connect(self._on_play_finished)
        self._player.speed_changed.connect(self._on_speed_applied)

    def set_stem_names(self, stem_names: list[str]) -> None:
        """Populate the stem mixer with rows for each stem."""
        for row in self._stem_rows.values():
            row.setParent(None)
            row.deleteLater()
        self._stem_rows.clear()
        self.clear_recording_rows()

        has_stems = bool(stem_names)
        self._empty_widget.setVisible(not has_stems)
        self._controls_widget.setVisible(has_stems)

        for name in stem_names:
            row = StemRow(name, self._player, self._theme)
            row.mix_changed.connect(self._recompute_peaks)
            self._stem_container.addWidget(row)
            self._stem_rows[name] = row

        self._speed_combo.blockSignals(True)
        self._speed_combo.setCurrentText("1.0x")
        self._speed_combo.blockSignals(False)
        self._speed_status.setText("")

        self._record_btn.blockSignals(True)
        self._record_btn.setChecked(False)
        self._record_btn.blockSignals(False)
        self.update_record_button_state()

        if stem_names:
            self._do_recompute_peaks()

    def clear_song(self) -> None:
        """Return to the empty logo state."""
        self.set_stem_names([])
        self._waveform.set_peaks(np.zeros(1, dtype=np.float32))
        self._waveform.set_position(0.0)
        self._time_label.setText("0:00 / 0:00")

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
        for name, row in self._stem_rows.items():
            row.set_muted(name in muted)
            row.set_soloed(name in soloed)
            vol = volumes.get(name, 1.0)
            row.set_volume_slider(round(vol * 100))
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

    def _on_state_changed(self, playing: bool) -> None:
        self._play_btn.setIcon(self._pause_icon if playing else self._play_icon)
        self._play_btn.setAccessibleName("Pause" if playing else "Play")
        if not playing:
            self._count_in_label.setText("")
            if not self._player.recording_armed:
                self._record_btn.blockSignals(True)
                self._record_btn.setChecked(False)
                self._record_btn.blockSignals(False)

    def _on_play_finished(self) -> None:
        self._play_btn.setIcon(self._play_icon)
        self._play_btn.setAccessibleName("Play")
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
        self._peaks_timer.stop()
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

        for name, row in self._stem_rows.items():
            if name in stems:
                stem_peaks = compute_stem_peaks(stems[name], num_bins=200)
                row.set_mini_peaks(stem_peaks)
        for name, row in self._recording_rows.items():
            if name in stems:
                stem_peaks = compute_stem_peaks(stems[name], num_bins=200)
                row.set_mini_peaks(stem_peaks)

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
        self._speed_combo.blockSignals(True)
        label = f"{speed}x"
        idx = self._speed_combo.findText(label)
        if idx >= 0:
            self._speed_combo.setCurrentIndex(idx)
        self._speed_combo.blockSignals(False)
        self._do_recompute_peaks()
        self.update_record_button_state()

    def cycle_speed(self, direction: int) -> None:
        """Cycle to the next/previous speed preset.

        Args:
            direction: +1 for faster, -1 for slower.
        """
        idx = self._speed_combo.currentIndex() + direction
        idx = max(0, min(idx, self._speed_combo.count() - 1))
        self._speed_combo.setCurrentIndex(idx)

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

    def _on_metronome_toggled(self, checked: bool) -> None:
        """User toggled the metronome on/off."""
        self._player.set_metronome_enabled(checked)

    def _on_metronome_vol_changed(self, value: int) -> None:
        """User moved the metronome volume slider."""
        self._metronome_vol_label.setText(f"{value}%")
        self._player.set_metronome_volume(value / 100.0)

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
        self._metronome_vol_label.setText(f"{round(volume * 100)}%")
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
        """Toggle recording arm (for keyboard shortcut)."""
        self._record_btn.setChecked(not self._record_btn.isChecked())

    def add_recording_row(
        self, stem_name: str, display_name: str
    ) -> RecordingStemRow:
        """Add a recording take row to the recordings section."""
        row = RecordingStemRow(
            stem_name, display_name, self._player, self._theme
        )
        row.mix_changed.connect(self._recompute_peaks)
        self._recordings_container.addWidget(row)
        self._recording_rows[stem_name] = row
        self._recordings_label.setVisible(True)
        self._recordings_frame.setVisible(True)
        return row

    def remove_recording_row(self, stem_name: str) -> None:
        """Remove a recording take row by stem name."""
        row = self._recording_rows.pop(stem_name, None)
        if row is not None:
            row.setParent(None)
            row.deleteLater()
        if not self._recording_rows:
            self._recordings_label.setVisible(False)
            self._recordings_frame.setVisible(False)

    def clear_recording_rows(self) -> None:
        """Remove all recording rows."""
        for row in self._recording_rows.values():
            row.setParent(None)
            row.deleteLater()
        self._recording_rows.clear()
        self._recordings_label.setVisible(False)
        self._recordings_frame.setVisible(False)

    @property
    def recording_count(self) -> int:
        """Return the number of recording take rows currently shown."""
        return len(self._recording_rows)

    @property
    def max_recordings_reached(self) -> bool:
        """Return True if the maximum number of recording takes is reached."""
        return len(self._recording_rows) >= _MAX_RECORDING_TAKES

    def update_record_button_state(self) -> None:
        """Sync Record button enabled state with current speed."""
        at_1x = self._player.speed == 1.0
        self._record_btn.setEnabled(at_1x and self._player.has_stems)
        if not at_1x:
            self._record_btn.setToolTip(
                "Recording requires 1.0x speed"
            )
            if self._record_btn.isChecked():
                self._record_btn.blockSignals(True)
                self._record_btn.setChecked(False)
                self._record_btn.blockSignals(False)
                self._player.arm_recording(False)
        else:
            self._record_btn.setToolTip("Arm recording (R)")

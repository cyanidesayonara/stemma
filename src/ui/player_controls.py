"""Player transport controls and per-stem mute/solo mixer.

Transport: Play/Pause, Stop, waveform display, time display.
Per-stem row: label, Mute button, Solo button, volume slider.
Color-coded stems. Full implementation in ticket #9.
"""

import math
import time

import numpy as np

from concurrent.futures import Future, ThreadPoolExecutor

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
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

from src.beat_detector import DetectionResult, DetectionWorker
from src.metronome import tap_tempo
from src.player import SPEED_PRESETS, MultiTrackPlayer
from src.ui.animated_arpeggio import AnimatedArpeggioWidget
from src.ui.animated_logo import AnimatedLogoWidget
from src.ui.styles import (
    DARK_COLORS,
    LIGHT_COLORS,
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


def _make_display_combo(parent_combo: QComboBox) -> None:
    """Make an editable combo act as a read-only display that opens on click.

    Sets the combo editable with a read-only line edit, and installs a mouse
    handler so clicking anywhere on the combo opens the dropdown popup.
    """
    parent_combo.setEditable(True)
    le = parent_combo.lineEdit()
    le.setReadOnly(True)
    le.installEventFilter(parent_combo)
    # Forward mouse presses on the line edit to the combo popup.
    original_mouse = le.mousePressEvent
    def _open_on_click(event):  # noqa: ANN001
        if event.button() == Qt.MouseButton.LeftButton:
            parent_combo.showPopup()
        else:
            original_mouse(event)
    le.mousePressEvent = _open_on_click


def _compute_peaks_bg(stems, muted, soloed, volumes, num_bins=2000,
                      mini_bins=200):
    """Compute peaks on a background thread. Returns (main_peaks, stem_peaks)."""
    main_peaks = compute_peaks(
        stems=stems,
        muted=muted,
        soloed=soloed,
        volumes=volumes,
        num_bins=num_bins,
    )
    stem_peaks = {}
    for name, data in stems.items():
        stem_peaks[name] = compute_stem_peaks(data, num_bins=mini_bins)
    return main_peaks, stem_peaks


_peak_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="peak")


def _make_icon(draw_fn, color: QColor, size: int = _ICON_SIZE) -> QIcon:
    """Create a crisp QIcon by painting with *draw_fn(painter, size)*."""
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.GlobalColor.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    draw_fn(p, size)
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


def _draw_mute(p: QPainter, s: int) -> None:
    """Speaker with X — mute icon."""
    m = s * 0.18
    # Speaker body (small rectangle)
    bw = s * 0.16
    bh = s * 0.28
    bx = m
    by = s / 2.0 - bh / 2.0
    p.drawRect(QRectF(bx, by, bw, bh))
    # Speaker cone (triangle)
    cx = bx + bw
    cw = s * 0.20
    p.drawPolygon(QPolygonF([
        QPointF(cx, by), QPointF(cx + cw, m),
        QPointF(cx + cw, s - m), QPointF(cx, by + bh),
    ]))
    # X slash
    pen = QPen(p.brush().color(), s * 0.09)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    x0 = s * 0.58
    p.drawLine(QPointF(x0, m * 1.3), QPointF(s - m, s - m * 1.3))
    p.drawLine(QPointF(x0, s - m * 1.3), QPointF(s - m, m * 1.3))
    p.setPen(Qt.PenStyle.NoPen)


def _draw_solo(p: QPainter, s: int) -> None:
    """Headphones icon — solo."""
    cx = s / 2.0
    m = s * 0.15
    # Arc for headband
    pen = QPen(p.brush().color(), s * 0.09)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    arc_rect = QRectF(m, m, s - 2 * m, s - 2 * m)
    p.drawArc(arc_rect, 30 * 16, 120 * 16)
    p.setPen(Qt.PenStyle.NoPen)
    # Ear cups (two small rounded rects)
    cw = s * 0.18
    ch = s * 0.30
    cy = s * 0.52
    p.drawRoundedRect(QRectF(m, cy, cw, ch), 2, 2)
    p.drawRoundedRect(QRectF(s - m - cw, cy, cw, ch), 2, 2)


def _draw_power(p: QPainter, s: int) -> None:
    """Universal power/on-off icon — circle with line at top."""
    cx = s / 2.0
    cy = s / 2.0 + s * 0.08  # push ring down to make room for line above
    r = s * 0.28
    stroke = max(1.5, s * 0.09)
    pen = QPen(p.brush().color(), stroke)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    arc_rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
    # Gap centered at 90° (12 o'clock in Qt). Arc from 125° CCW 290° to 55°.
    p.drawArc(arc_rect, 125 * 16, 290 * 16)
    # Vertical line through the top gap
    line_top = max(stroke * 0.5, cy - r - s * 0.06)
    line_bot = cy - s * 0.04
    p.drawLine(QPointF(cx, line_top), QPointF(cx, line_bot))
    p.setPen(Qt.PenStyle.NoPen)


def _draw_trash(p: QPainter, s: int) -> None:
    """Trash can icon for deleting items."""
    m = s * 0.2
    pen = QPen(p.brush().color(), max(1.0, s * 0.08))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    # The painter must be set to NoBrush for stroke-only rects.
    p.setBrush(Qt.GlobalColor.transparent)
    p.setPen(pen)
    # Lid line
    p.drawLine(QPointF(m * 0.8, m * 1.5), QPointF(s - m * 0.8, m * 1.5))
    # Handle top
    p.drawRect(QRectF(s * 0.4, m, s * 0.2, m * 0.5))
    # Body
    p.drawRect(QRectF(m * 1.2, m * 1.5, s - m * 2.4, s - m * 2.5))
    p.setPen(Qt.PenStyle.NoPen)


def _draw_repeat(p: QPainter, s: int) -> None:
    """Cycle/repeat arrows icon."""
    cx = s / 2.0
    m = s * 0.20
    r = (s - 2 * m) / 2.0
    pen = QPen(p.brush().color(), s * 0.09)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    # Draw two arcs (top-right and bottom-left)
    arc_rect = QRectF(m, m, s - 2 * m, s - 2 * m)
    p.drawArc(arc_rect, 20 * 16, 140 * 16)   # top arc
    p.drawArc(arc_rect, 200 * 16, 140 * 16)  # bottom arc
    p.setPen(Qt.PenStyle.NoPen)
    # Arrowhead on top arc (right end)
    a1 = math.radians(20)
    ax1 = cx + r * math.cos(a1)
    ay1 = cx - r * math.sin(a1)
    ah = s * 0.12
    p.drawPolygon(QPolygonF([
        QPointF(ax1 + ah, ay1 - ah * 0.6),
        QPointF(ax1 - ah * 0.3, ay1 - ah * 0.8),
        QPointF(ax1, ay1 + ah * 0.5),
    ]))
    # Arrowhead on bottom arc (left end)
    a2 = math.radians(200)
    ax2 = cx + r * math.cos(a2)
    ay2 = cx - r * math.sin(a2)
    p.drawPolygon(QPolygonF([
        QPointF(ax2 - ah, ay2 + ah * 0.6),
        QPointF(ax2 + ah * 0.3, ay2 + ah * 0.8),
        QPointF(ax2, ay2 - ah * 0.5),
    ]))


_STEM_ICON_SIZE = 18

_CHECKED_ICON_COLOR = QColor("#ffffff")


def _make_toggle_icon(draw_fn, normal_color: QColor,
                      size: int = _ICON_SIZE) -> QIcon:
    """Create an icon with distinct normal (theme text) and checked (white) pixmaps."""
    icon = QIcon()
    for color, state in [
        (normal_color, QIcon.State.Off),
        (_CHECKED_ICON_COLOR, QIcon.State.On),
    ]:
        pixmap = QPixmap(QSize(size, size))
        pixmap.fill(Qt.GlobalColor.transparent)
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        draw_fn(p, size)
        p.end()
        # Ensure icon is visible in all interaction modes
        for mode in [QIcon.Mode.Normal, QIcon.Mode.Active, QIcon.Mode.Selected]:
            icon.addPixmap(pixmap, mode, state)
    return icon


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

        colors = DARK_COLORS if theme == "dark" else LIGHT_COLORS
        text_c = QColor(colors["text"])
        display = stem_name.capitalize()

        self._mute_btn = QPushButton()
        self._mute_btn.setObjectName("icon-btn")
        self._mute_btn.setCheckable(True)
        self._mute_btn.setFixedSize(28, 28)
        self._mute_btn.setIcon(
            _make_toggle_icon(_draw_mute, text_c, _STEM_ICON_SIZE))
        self._mute_btn.setIconSize(QSize(_STEM_ICON_SIZE, _STEM_ICON_SIZE))
        self._mute_btn.setToolTip(f"Mute {display}")
        self._mute_btn.setAccessibleName(f"Mute {display}")
        self._mute_btn.toggled.connect(self._on_mute)
        layout.addWidget(self._mute_btn)

        self._solo_btn = QPushButton()
        self._solo_btn.setObjectName("icon-btn")
        self._solo_btn.setCheckable(True)
        self._solo_btn.setFixedSize(28, 28)
        self._solo_btn.setIcon(
            _make_toggle_icon(_draw_solo, text_c, _STEM_ICON_SIZE))
        self._solo_btn.setIconSize(QSize(_STEM_ICON_SIZE, _STEM_ICON_SIZE))
        self._solo_btn.setToolTip(f"Solo {display}")
        self._solo_btn.setAccessibleName(f"Solo {display}")
        self._solo_btn.toggled.connect(self._on_solo)
        layout.addWidget(self._solo_btn)

        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 200)
        self._volume_slider.setValue(100)
        self._volume_slider.setFixedWidth(120)
        self._volume_slider.setToolTip(f"{display} volume (0-200%, double-click to reset)")
        self._volume_slider.setAccessibleName(f"{display} volume")
        self._volume_slider.valueChanged.connect(self._on_volume)
        self._volume_slider.mouseDoubleClickEvent = lambda _: self._volume_slider.setValue(100)
        layout.addWidget(self._volume_slider)

        self._vol_combo = QComboBox()
        _make_display_combo(self._vol_combo)
        _VOLUME_PRESETS = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        for v in _VOLUME_PRESETS:
            self._vol_combo.addItem(f"{v}%", v)
        self._vol_combo.setCurrentText("100%")
        self._vol_combo.setFixedSize(62, 28)
        self._vol_combo.setToolTip(f"{display} volume")
        self._vol_combo.setAccessibleName(f"{display} volume preset")
        self._vol_combo.activated.connect(self._on_vol_combo)
        layout.addWidget(self._vol_combo)

    def _on_mini_seek(self, seconds: float) -> None:
        self._player.seek(seconds)

    def _on_mute(self, checked: bool) -> None:
        self._player.set_mute(self._stem_name, checked)
        self._mute_btn.clearFocus()
        self.mix_changed.emit()

    def _on_solo(self, checked: bool) -> None:
        self._player.set_solo(self._stem_name, checked)
        self._solo_btn.clearFocus()
        self.mix_changed.emit()

    def _on_volume(self, value: int) -> None:
        gain = value / 100.0
        self._player.set_volume(self._stem_name, gain)
        # Sync combo display without re-firing
        self._vol_combo.blockSignals(True)
        self._vol_combo.setEditText(f"{value}%")
        self._vol_combo.blockSignals(False)
        self.mix_changed.emit()

    def _on_vol_combo(self, index: int) -> None:
        value = self._vol_combo.itemData(index)
        if value is not None:
            self._volume_slider.setValue(value)

    def set_muted(self, muted: bool) -> None:
        """Programmatically set the mute button state (e.g. from keyboard shortcut)."""
        self._mute_btn.setChecked(muted)

    def set_soloed(self, soloed: bool) -> None:
        """Programmatically set the solo button state."""
        self._solo_btn.setChecked(soloed)

    def set_volume_slider(self, value: int) -> None:
        """Programmatically set the volume slider (0-200)."""
        self._volume_slider.setValue(value)
        self._vol_combo.setEditText(f"{value}%")

    def set_mini_peaks(self, peaks: "np.ndarray") -> None:
        """Update the mini waveform with new peak data."""
        self._mini_waveform.set_peaks(peaks)

    def apply_stem_theme(self, theme: str) -> None:
        """Update stem label and waveform color for the given theme."""
        palette = STEM_COLORS_DARK if theme == "dark" else STEM_COLORS_LIGHT
        color = palette.get(self._stem_name, "#95a5a6")
        self._label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self._mini_waveform.set_color(QColor(color))
        self._mini_waveform.update()
        tc = DARK_COLORS if theme == "dark" else LIGHT_COLORS
        text_c = QColor(tc["text"])
        self._mute_btn.setIcon(
            _make_toggle_icon(_draw_mute, text_c, _STEM_ICON_SIZE))
        self._solo_btn.setIcon(
            _make_toggle_icon(_draw_solo, text_c, _STEM_ICON_SIZE))


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
        self._mini_waveform.set_color(QColor(RECORDING_COLOR))

        lay = self.layout()
        insert_pos = lay.count()

        self._nudge_spin = QSpinBox()
        self._nudge_spin.setRange(-200, 200)
        self._nudge_spin.setValue(0)
        self._nudge_spin.setSuffix(" ms")
        self._nudge_spin.setFixedWidth(104)
        self._nudge_spin.setToolTip(
            f"Nudge {display_name} alignment (-200 to +200 ms)"
        )
        self._nudge_spin.setAccessibleName(f"Nudge {display_name}")
        self._nudge_spin.valueChanged.connect(self._on_nudge_changed)
        lay.insertWidget(insert_pos, self._nudge_spin)
        insert_pos += 1

        self._delete_btn = QPushButton()
        self._delete_btn.setObjectName("icon-btn")
        self._delete_btn.setFixedSize(28, 28)
        tc = DARK_COLORS if theme == "dark" else LIGHT_COLORS
        text_c = QColor(tc["text"])
        self._delete_btn.setIcon(_make_icon(_draw_trash, text_c, _STEM_ICON_SIZE))
        self._delete_btn.setIconSize(QSize(_STEM_ICON_SIZE, _STEM_ICON_SIZE))
        self._delete_btn.setToolTip(f"Delete {display_name}")
        self._delete_btn.setAccessibleName(f"Delete {display_name}")
        self._delete_btn.clicked.connect(
            lambda: self.delete_requested.emit(self._stem_name)
        )
        lay.insertWidget(insert_pos, self._delete_btn)

    def _on_nudge_changed(self, value: int) -> None:
        self._player.nudge_stem(self._stem_name, float(value))
        self.mix_changed.emit()

    def set_nudge(self, value: int) -> None:
        """Programmatically set the nudge spinbox."""
        self._nudge_spin.setValue(value)


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

        self._peak_future: Future | None = None
        self._peak_poll_timer = QTimer(self)
        self._peak_poll_timer.setInterval(16)  # ~60fps poll
        self._peak_poll_timer.timeout.connect(self._poll_peak_future)

        self._detection_worker: DetectionWorker | None = None
        self._beat_model_path: str | None = None
        self._key_conf: str = ""
        self._bpm_conf: str = ""

        self._setup_ui()
        self._connect_signals()

    def _cleanup_peak_thread(self) -> None:
        """Wait for any pending peak computation before destruction."""
        self._peaks_timer.stop()
        self._peak_poll_timer.stop()
        if self._peak_future is not None and not self._peak_future.done():
            self._peak_future.result(timeout=2)
        self._peak_future = None
        if self._detection_worker is not None:
            self._detection_worker.wait(2000)
            self._detection_worker = None

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

        # -- Count-in controls (right side of transport bar) --
        self._count_in_label = QLabel("")
        self._count_in_label.setFixedWidth(32)
        self._count_in_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        transport.addWidget(self._count_in_label)

        self._ci_label = QLabel("Count-in:")
        transport.addWidget(self._ci_label)

        self._count_in_toggle = QPushButton()
        self._count_in_toggle.setObjectName("icon-btn")
        self._count_in_toggle.setCheckable(True)
        self._count_in_toggle.setFixedSize(36, 36)
        self._count_in_toggle.setIcon(
            _make_toggle_icon(_draw_power, icon_color))
        self._count_in_toggle.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        self._count_in_toggle.setToolTip("Toggle count-in before playback (C)")
        self._count_in_toggle.setAccessibleName("Toggle count-in")
        self._count_in_toggle.toggled.connect(self._on_count_in_toggled)
        transport.addWidget(self._count_in_toggle)

        self._count_in_beats_spin = QSpinBox()
        self._count_in_beats_spin.setRange(1, 8)
        self._count_in_beats_spin.setValue(4)
        self._count_in_beats_spin.setSuffix(" beats")
        self._count_in_beats_spin.setFixedWidth(96)
        self._count_in_beats_spin.setToolTip("Number of count-in beats")
        self._count_in_beats_spin.setAccessibleName("Count-in beats")
        self._count_in_beats_spin.valueChanged.connect(
            self._on_count_in_beats_changed
        )
        transport.addWidget(self._count_in_beats_spin)

        self._count_in_repeats_cb = QPushButton()
        self._count_in_repeats_cb.setObjectName("icon-btn")
        self._count_in_repeats_cb.setCheckable(True)
        self._count_in_repeats_cb.setFixedSize(36, 36)
        self._count_in_repeats_cb.setIcon(
            _make_toggle_icon(_draw_repeat, icon_color))
        self._count_in_repeats_cb.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        self._count_in_repeats_cb.setToolTip(
            "Also count in before each A-B loop repeat"
        )
        self._count_in_repeats_cb.setAccessibleName(
            "Count-in on loop repeats"
        )
        self._count_in_repeats_cb.toggled.connect(
            self._on_count_in_repeats_toggled
        )
        transport.addWidget(self._count_in_repeats_cb)

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
        self._loop_a_btn.setFixedWidth(50)
        self._loop_a_btn.setToolTip("Set loop start point (A)")
        self._loop_a_btn.setAccessibleName("Set loop A")
        self._loop_a_btn.clicked.connect(self.set_loop_a)
        loop_speed_bar.addWidget(self._loop_a_btn)

        self._loop_b_btn = QPushButton("Set B")
        self._loop_b_btn.setFixedWidth(50)
        self._loop_b_btn.setToolTip("Set loop end point (B)")
        self._loop_b_btn.setAccessibleName("Set loop B")
        self._loop_b_btn.clicked.connect(self.set_loop_b)
        loop_speed_bar.addWidget(self._loop_b_btn)

        self._loop_toggle_btn = QPushButton("Loop")
        self._loop_toggle_btn.setCheckable(True)
        self._loop_toggle_btn.setFixedWidth(48)
        self._loop_toggle_btn.setToolTip("Toggle A-B loop (L)")
        self._loop_toggle_btn.setAccessibleName("Toggle loop")
        self._loop_toggle_btn.toggled.connect(self._on_loop_toggled)
        loop_speed_bar.addWidget(self._loop_toggle_btn)

        self._loop_clear_btn = QPushButton("Clear")
        self._loop_clear_btn.setFixedWidth(48)
        self._loop_clear_btn.setToolTip("Clear loop points")
        self._loop_clear_btn.setAccessibleName("Clear loop")
        self._loop_clear_btn.clicked.connect(self._on_clear_loop)
        loop_speed_bar.addWidget(self._loop_clear_btn)

        self._loop_label = QLabel("")
        self._loop_label.setStyleSheet(f"color: {colors['surface2']};")
        loop_speed_bar.addWidget(self._loop_label)

        self._key_label = QLabel("")
        self._key_label.setToolTip("Detected musical key (double-click to re-detect)")
        self._key_label.setAccessibleName("Detected key")
        self._key_label.setStyleSheet(f"color: {colors['surface2']};")
        self._key_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._key_label.installEventFilter(self)
        loop_speed_bar.addWidget(self._key_label)

        self._speed_status = QLabel("")
        self._speed_status.setStyleSheet(f"color: {colors['surface2']};")
        loop_speed_bar.addWidget(self._speed_status)

        loop_speed_bar.addStretch()

        self._speed_label = QLabel("Speed:")
        loop_speed_bar.addWidget(self._speed_label)

        self._speed_combo = QComboBox()
        for preset in SPEED_PRESETS:
            self._speed_combo.addItem(f"{preset}x", preset)
        self._speed_combo.setCurrentText("1.0x")
        self._speed_combo.setFixedWidth(66)
        self._speed_combo.setToolTip("Playback speed ([ / ])")
        self._speed_combo.setAccessibleName("Playback speed")
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        loop_speed_bar.addWidget(self._speed_combo)

        controls_layout.addLayout(loop_speed_bar)

        # -- Metronome bar --
        metro_ci_bar = QHBoxLayout()

        self._metro_label = QLabel("Metronome:")
        metro_ci_bar.addWidget(self._metro_label)

        self._metronome_toggle = QPushButton()
        self._metronome_toggle.setObjectName("icon-btn")
        self._metronome_toggle.setCheckable(True)
        self._metronome_toggle.setFixedSize(36, 36)
        self._metronome_toggle.setIcon(
            _make_toggle_icon(_draw_power, icon_color))
        self._metronome_toggle.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        self._metronome_toggle.setToolTip("Toggle metronome (M)")
        self._metronome_toggle.setAccessibleName("Toggle metronome")
        self._metronome_toggle.toggled.connect(self._on_metronome_toggled)
        metro_ci_bar.addWidget(self._metronome_toggle)

        self._bpm_spin = QSpinBox()
        self._bpm_spin.setRange(20, 300)
        self._bpm_spin.setValue(120)
        self._bpm_spin.setSuffix(" BPM")
        self._bpm_spin.setFixedWidth(105)
        self._bpm_spin.setToolTip("Metronome tempo")
        self._bpm_spin.setAccessibleName("Metronome BPM")
        self._bpm_spin.valueChanged.connect(self._on_bpm_changed)
        metro_ci_bar.addWidget(self._bpm_spin)

        self._tap_times: list[float] = []
        self._tap_btn = QPushButton("Tap")
        self._tap_btn.setFixedWidth(46)
        self._tap_btn.setToolTip("Tap to set tempo")
        self._tap_btn.setAccessibleName("Tap tempo")
        self._tap_btn.clicked.connect(self._on_tap)
        metro_ci_bar.addWidget(self._tap_btn)

        self._beat_sync_btn = QPushButton("Sync")
        self._beat_sync_btn.setCheckable(True)
        self._beat_sync_btn.setFixedWidth(50)
        self._beat_sync_btn.setToolTip(
            "Sync metronome to detected beats (click on actual beat positions)"
        )
        self._beat_sync_btn.setAccessibleName("Sync to track")
        self._beat_sync_btn.setEnabled(False)
        self._beat_sync_btn.toggled.connect(self._on_beat_sync_toggled)
        metro_ci_bar.addWidget(self._beat_sync_btn)

        self._beat_nudge_spin = QSpinBox()
        self._beat_nudge_spin.setRange(-500, 500)
        self._beat_nudge_spin.setValue(0)
        self._beat_nudge_spin.setSuffix(" ms")
        self._beat_nudge_spin.setFixedWidth(104)
        self._beat_nudge_spin.setToolTip("Metronome nudge (shift metronome clicking)")
        self._beat_nudge_spin.setAccessibleName("Sync Nudge")
        self._beat_nudge_spin.valueChanged.connect(self._on_beat_nudge_changed)
        metro_ci_bar.addWidget(self._beat_nudge_spin)

        self._metronome_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._metronome_vol_slider.setRange(0, 200)
        self._metronome_vol_slider.setValue(100)
        self._metronome_vol_slider.setFixedWidth(70)
        self._metronome_vol_slider.setToolTip(
            "Metronome volume (0-200%, double-click to reset)"
        )
        self._metronome_vol_slider.setAccessibleName("Metronome volume")
        self._metronome_vol_slider.valueChanged.connect(
            self._on_metronome_vol_changed
        )
        self._metronome_vol_slider.mouseDoubleClickEvent = (
            lambda _: self._metronome_vol_slider.setValue(100)
        )
        metro_ci_bar.addWidget(self._metronome_vol_slider)

        self._metronome_vol_combo = QComboBox()
        _make_display_combo(self._metronome_vol_combo)
        _MET_VOL_PRESETS = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        for v in _MET_VOL_PRESETS:
            self._metronome_vol_combo.addItem(f"{v}%", v)
        self._metronome_vol_combo.setCurrentText("100%")
        self._metronome_vol_combo.setFixedWidth(62)
        self._metronome_vol_combo.setToolTip("Metronome volume")
        self._metronome_vol_combo.setAccessibleName("Metronome volume preset")
        self._metronome_vol_combo.activated.connect(
            self._on_metronome_vol_combo
        )
        metro_ci_bar.addWidget(self._metronome_vol_combo)

        self._detected_bpm_label = QLabel("")
        self._detected_bpm_label.setToolTip(
            "Detected tempo — suggestion only (double-click to re-detect)"
        )
        self._detected_bpm_label.setAccessibleName("Detected BPM")
        self._detected_bpm_label.setStyleSheet(f"color: {colors['surface2']};")
        self._detected_bpm_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._detected_bpm_label.installEventFilter(self)
        metro_ci_bar.addWidget(self._detected_bpm_label)

        metro_ci_bar.addStretch()
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
        base_c = QColor(colors["base"])

        self._play_icon = _make_icon(_draw_play, icon_color)
        self._pause_icon = _make_icon(_draw_pause, icon_color)
        self._stop_icon = _make_icon(_draw_stop, icon_color)

        if self._player.is_playing:
            self._play_btn.setIcon(self._pause_icon)
        else:
            self._play_btn.setIcon(self._play_icon)
        self._stop_btn.setIcon(self._stop_icon)

        # Toggle button icons
        self._metronome_toggle.setIcon(
            _make_toggle_icon(_draw_power, icon_color))
        self._count_in_toggle.setIcon(
            _make_toggle_icon(_draw_power, icon_color))
        self._count_in_repeats_cb.setIcon(
            _make_toggle_icon(_draw_repeat, icon_color))

        # Inline-styled labels
        self._hint_label.setStyleSheet(
            f"color: {colors['surface2']}; margin-top: 0px;"
        )
        self._loop_label.setStyleSheet(f"color: {colors['surface2']};")
        self._speed_status.setStyleSheet(f"color: {colors['surface2']};")
        if not self._key_label.text():
            self._key_label.setStyleSheet(f"color: {colors['surface2']};")
        else:
            c = self._conf_color(self._key_conf)
            if c:
                self._key_label.setStyleSheet(f"color: {c};")

        if not self._detected_bpm_label.text():
            self._detected_bpm_label.setStyleSheet(
                f"color: {colors['surface2']};"
            )
        else:
            c = self._conf_color(self._bpm_conf)
            if c:
                self._detected_bpm_label.setStyleSheet(f"color: {c};")
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
        # Preserve mute/solo state from previous song
        saved_muted = set(self._player.muted_stems)
        saved_soloed = set(self._player.soloed_stems)

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

        # Auto-detect if no beat grid is loaded.
        if has_stems and not self._player.beat_times:
            self.start_detection()

        if stem_names:
            self._waveform.set_loading(True)
            # Restore mute/solo from previous song if stems match
            for name, row in self._stem_rows.items():
                if name in saved_muted:
                    row.set_muted(True)
                if name in saved_soloed:
                    row.set_soloed(True)
            self._do_recompute_peaks()

    def clear_song(self) -> None:
        """Return to the empty logo state."""
        self.set_stem_names([])
        self._waveform.set_loading(False)
        self._waveform.set_peaks(np.zeros(1, dtype=np.float32))
        self._waveform.set_position(0.0)
        self._time_label.setText("0:00 / 0:00")
        self._key_label.setText("")
        self._key_conf = ""
        self._key_label.setToolTip(
            "Detected musical key (double-click to re-detect)"
        )
        self._detected_bpm_label.setText("")
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

        # Update BPM spinbox with instantaneous BPM when beat-synced.
        if self._beat_sync_btn.isChecked():
            frame = int(pos_s * self._player.sample_rate)
            ibpm = self._player.instantaneous_bpm_at(frame)
            if ibpm > 0:
                self._bpm_spin.blockSignals(True)
                self._bpm_spin.setValue(max(20, min(300, round(ibpm))))
                self._bpm_spin.blockSignals(False)

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
        """Dispatch waveform peak computation to a background thread."""
        self._peaks_timer.stop()
        stems = self._player.stems
        if not stems:
            self._waveform.set_peaks(np.zeros(1, dtype=np.float32))
            return

        # If a previous computation is still running, let the debounce
        # timer re-fire after the current one finishes.
        if self._peak_future is not None and not self._peak_future.done():
            self._peaks_timer.start()
            return

        self._peak_future = _peak_pool.submit(
            _compute_peaks_bg,
            stems=stems,
            muted=self._player.muted_stems,
            soloed=self._player.soloed_stems,
            volumes=self._player.volumes,
        )
        self._peak_poll_timer.start()

    def _poll_peak_future(self) -> None:
        """Check if the background peak computation has finished."""
        if self._peak_future is None or not self._peak_future.done():
            return
        self._peak_poll_timer.stop()
        future = self._peak_future
        self._peak_future = None
        if future.cancelled() or future.exception():
            return
        try:
            main_peaks, stem_peaks = future.result()
        except Exception:
            return
        self._on_peaks_computed(main_peaks, stem_peaks)

    def _on_peaks_computed(self, main_peaks: np.ndarray,
                           stem_peaks: dict) -> None:
        """Apply peak results from the background thread."""
        try:
            _ = self._waveform
        except RuntimeError:
            return  # Widget was destroyed
        self._waveform.set_peaks(main_peaks)
        self._waveform.set_total_seconds(self._player.total_seconds)

        for name, row in self._stem_rows.items():
            if name in stem_peaks:
                row.set_mini_peaks(stem_peaks[name])
        for name, row in self._recording_rows.items():
            if name in stem_peaks:
                row.set_mini_peaks(stem_peaks[name])

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

    def set_loop_b(self) -> None:
        """Set loop B to the current playback position."""
        self._player.set_loop_b(self._player.current_seconds)
        self._update_loop_label()
        self._update_waveform_loop_markers()
        self._maybe_redetect_for_loop()

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
        # Re-detect for the full song after clearing A-B region.
        if self._player.stems:
            self.start_detection()

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

    def set_beat_model_path(self, path: str) -> None:
        """Set the path to the beat_this ONNX model file."""
        self._beat_model_path = path

    def start_detection(
        self,
        start_sec: float | None = None,
        end_sec: float | None = None,
    ) -> None:
        """Start background BPM/key detection.

        Called automatically when stems load and when A-B loop points
        change.  Results are shown as suggestions only — the metronome
        BPM spinbox is *not* modified.
        """
        if not self._player.stems:
            return
        if self._detection_worker is not None:
            self._detection_worker.wait(2000)
            self._detection_worker = None

        dim = LIGHT_COLORS if self._theme == "light" else DARK_COLORS
        self._detected_bpm_label.setStyleSheet(
            f"color: {dim['surface2']};"
        )
        self._detected_bpm_label.setText("detecting...")
        self._key_label.setStyleSheet(f"color: {dim['surface2']};")
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
        self._detection_worker = worker
        worker.start()

    # Confidence label colours — Catppuccin Mocha semantic palette.
    _CONF_COLORS_DARK = {
        "high": "#a6e3a1", "medium": "#f9e2af", "low": "#f38ba8",
    }
    _CONF_COLORS_LIGHT = {
        "high": "#40a02b", "medium": "#df8e1d", "low": "#d20f39",
    }

    def _conf_color(self, level: str) -> str:
        """Return the themed colour string for a confidence level."""
        if self._theme == "light":
            return self._CONF_COLORS_LIGHT.get(level, "")
        return self._CONF_COLORS_DARK.get(level, "")

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
        # Store beat grid on the player.
        self._player.set_beat_times(result.beat_times, result.downbeat_times)

        # Enable/disable sync button based on whether beats were found.
        has_beats = len(result.beat_times) >= 2
        self._update_sync_btn_state(has_beats)

        # Update detected BPM label (suggestion only — does NOT set spinbox).
        if result.bpm > 0:
            bpm_rounded = round(result.bpm)
            self._bpm_conf = result.bpm_confidence
            bpm_c = self._conf_color(result.bpm_confidence)
            if bpm_c:
                self._detected_bpm_label.setStyleSheet(f"color: {bpm_c};")
            self._detected_bpm_label.setText(f"Detected tempo: ~{bpm_rounded} BPM")
            self._detected_bpm_label.setToolTip(
                f"Detected tempo: {result.bpm:.1f} BPM\n"
                f"Confidence: {result.bpm_confidence}\n"
                f"Double-click to re-detect"
            )
        else:
            self._detected_bpm_label.setText("")
            self._bpm_conf = ""

        # Update key label.
        if result.key:
            self._key_conf = result.key_confidence
            key_c = self._conf_color(result.key_confidence)
            if key_c:
                self._key_label.setStyleSheet(f"color: {key_c};")
            self._key_label.setText(f"Detected key: {result.key}")
            self._key_label.setToolTip(
                f"Detected key: {result.key}\n"
                f"Confidence: {result.key_confidence}\n"
                f"Double-click to re-detect"
            )
        else:
            self._key_label.setText("")
            self._key_conf = ""

    def _on_detect_error(self, msg: str) -> None:
        self._key_label.setText("")
        self._detected_bpm_label.setText("")

    def _on_detect_finished(self) -> None:
        self._detection_worker = None

    def _redetect_key_only(self) -> None:
        """Re-run detection but only update the key label."""
        if self._detection_worker is not None:
            return  # Already running.
        dim = LIGHT_COLORS if self._theme == "light" else DARK_COLORS
        self._key_label.setStyleSheet(f"color: {dim['surface2']};")
        self._key_label.setText("detecting...")

        worker = DetectionWorker(
            stems=dict(self._player.stems),
            sample_rate=self._player.sample_rate,
            model_path=self._beat_model_path,
        )
        worker.completed.connect(self._on_key_only_completed)
        worker.finished.connect(self._on_detect_finished)
        self._detection_worker = worker
        worker.start()

    def _on_key_only_completed(self, result: DetectionResult) -> None:
        """Update only the key label from a re-detection."""
        if result.key:
            self._key_conf = result.key_confidence
            key_c = self._conf_color(result.key_confidence)
            if key_c:
                self._key_label.setStyleSheet(f"color: {key_c};")
            self._key_label.setText(f"Detected key: {result.key}")
            self._key_label.setToolTip(
                f"Detected key: {result.key}\n"
                f"Confidence: {result.key_confidence}\n"
                f"Double-click to re-detect"
            )
        else:
            self._key_label.setText("")
            self._key_conf = ""

    def _redetect_bpm_only(self) -> None:
        """Re-run detection but only update the BPM label."""
        if self._detection_worker is not None:
            return  # Already running.
        dim = LIGHT_COLORS if self._theme == "light" else DARK_COLORS
        self._detected_bpm_label.setStyleSheet(
            f"color: {dim['surface2']};"
        )
        self._detected_bpm_label.setText("detecting...")

        worker = DetectionWorker(
            stems=dict(self._player.stems),
            sample_rate=self._player.sample_rate,
            model_path=self._beat_model_path,
        )
        worker.completed.connect(self._on_bpm_only_completed)
        worker.finished.connect(self._on_detect_finished)
        self._detection_worker = worker
        worker.start()

    def _on_bpm_only_completed(self, result: DetectionResult) -> None:
        """Update only the BPM label from a re-detection."""
        self._player.set_beat_times(result.beat_times, result.downbeat_times)
        self._update_sync_btn_state(len(result.beat_times) >= 2)
        if result.bpm > 0:
            bpm_rounded = round(result.bpm)
            self._bpm_conf = result.bpm_confidence
            bpm_c = self._conf_color(result.bpm_confidence)
            if bpm_c:
                self._detected_bpm_label.setStyleSheet(f"color: {bpm_c};")
            self._detected_bpm_label.setText(f"Detected tempo: ~{bpm_rounded} BPM")
            self._detected_bpm_label.setToolTip(
                f"Detected tempo: {result.bpm:.1f} BPM\n"
                f"Confidence: {result.bpm_confidence}\n"
                f"Double-click to re-detect"
            )
        else:
            self._detected_bpm_label.setText("")
            self._bpm_conf = ""

    @property
    def detected_key(self) -> str:
        """Return the raw detected key (e.g. "A minor"), or empty string."""
        text = self._key_label.text()
        prefix = "Detected key: "
        if text.startswith(prefix):
            return text[len(prefix):]
        return ""

    @property
    def detected_bpm_text(self) -> str:
        """Return the currently displayed detected BPM text, or empty."""
        return self._detected_bpm_label.text()

    @property
    def key_confidence(self) -> str:
        """Return the last key confidence level, or empty string."""
        return self._key_conf

    @property
    def bpm_confidence(self) -> str:
        """Return the last BPM confidence level, or empty string."""
        return self._bpm_conf

    def set_detected_key(self, key: str, confidence: str = "") -> None:
        """Restore a previously detected key label with colour and tooltip."""
        if key:
            self._key_label.setText(f"Detected key: {key}")
            self._key_conf = confidence
            c = self._conf_color(confidence) if confidence else ""
            if c:
                self._key_label.setStyleSheet(f"color: {c};")
            parts = [f"Detected key: {key}"]
            if confidence:
                parts.append(f"Confidence: {confidence}")
            parts.append("Double-click to re-detect")
            self._key_label.setToolTip("\n".join(parts))
        else:
            self._key_label.setText("")
            self._key_conf = ""
            self._key_label.setToolTip(
                "Detected musical key (double-click to re-detect)"
            )

    def set_detected_bpm_text(
        self, text: str, confidence: str = "",
    ) -> None:
        """Restore a previously detected BPM suggestion label with colour and tooltip."""
        if text:
            # Ensure prefix is present (older sessions may lack it).
            if not text.startswith("Detected tempo:"):
                text = f"Detected tempo: {text}"
            self._detected_bpm_label.setText(text)
            self._bpm_conf = confidence
            c = self._conf_color(confidence) if confidence else ""
            if c:
                self._detected_bpm_label.setStyleSheet(f"color: {c};")
            parts = [f"Detected tempo: {text}"]
            if confidence:
                parts.append(f"Confidence: {confidence}")
            parts.append("Double-click to re-detect")
            self._detected_bpm_label.setToolTip("\n".join(parts))
        else:
            self._detected_bpm_label.setText("")
            self._bpm_conf = ""
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

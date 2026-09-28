"""Core playback transport controls.

The waveform lives in WaveformPanel rather than here: the transport stays
anchored to the bottom of the window while the waveform scrolls with the rest
of the practice content.
"""

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.ui.control_primitives import (
    add_volume_presets,
    fix_button_size,
    COMPACT_BUTTON,
    ICON_SIZE,
    TRANSPORT_BUTTON,
    draw_pause,
    draw_play,
    draw_record,
    draw_stop,
    make_display_combo,
    make_icon,
    make_toggle_icon,
    show_preset_value,
)
from src.ui.styles import DARK_COLORS


def _record_icon(colors: dict[str, str]):
    """Record glyph: rose, and on-accent while armed (the checked fill
    is the accent, where rose was about 1:1)."""
    return make_toggle_icon(
        draw_record, QColor(colors["recording"]),
        checked_color=QColor(colors["on_accent"]),
    )


class TransportBar(QWidget):
    """Playback, recording, and master-volume controls."""

    play_pause_requested = Signal()
    stop_requested = Signal()
    record_toggled = Signal(bool)
    master_volume_changed = Signal(float)

    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("transport-bar")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)

        transport = QHBoxLayout()
        icon_color = QColor("#cdd6f4")
        self._play_icon = make_icon(draw_play, icon_color)
        self._pause_icon = make_icon(draw_pause, icon_color)
        self._stop_icon = make_icon(draw_stop, icon_color)

        self._play_button = QPushButton()
        self._play_button.setObjectName("icon-btn")
        self._play_button.setIcon(self._play_icon)
        self._play_button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        fix_button_size(self._play_button, TRANSPORT_BUTTON, TRANSPORT_BUTTON)
        self._play_button.setToolTip("Play / Pause (Space)")
        self._play_button.setAccessibleName("Play")
        self._play_button.clicked.connect(self.play_pause_requested.emit)
        transport.addWidget(self._play_button)

        self._stop_button = QPushButton()
        self._stop_button.setObjectName("icon-btn")
        self._stop_button.setIcon(self._stop_icon)
        self._stop_button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        fix_button_size(self._stop_button, TRANSPORT_BUTTON, TRANSPORT_BUTTON)
        self._stop_button.setToolTip("Stop (S)")
        self._stop_button.setAccessibleName("Stop")
        self._stop_button.clicked.connect(self.stop_requested.emit)
        transport.addWidget(self._stop_button)

        self._record_icon = _record_icon(DARK_COLORS)
        self._record_button = QPushButton()
        self._record_button.setObjectName("icon-btn")
        self._record_button.setIcon(self._record_icon)
        self._record_button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        fix_button_size(self._record_button, TRANSPORT_BUTTON, TRANSPORT_BUTTON)
        self._record_button.setCheckable(True)
        self._record_button.setToolTip("Arm recording (R)")
        self._record_button.setAccessibleName("Record")
        self._record_button.toggled.connect(self.record_toggled.emit)
        transport.addWidget(self._record_button)

        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setFixedWidth(100)
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        transport.addWidget(self._time_label)

        self._master_volume_prefix = QLabel("Vol:")
        transport.addWidget(self._master_volume_prefix)

        self._master_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._master_volume_slider.setRange(0, 200)
        self._master_volume_slider.setValue(100)
        self._master_volume_slider.setFixedWidth(90)
        self._master_volume_slider.setToolTip(
            "Master volume (Up / Down)"
        )
        self._master_volume_slider.setAccessibleName("Master volume")
        self._master_volume_slider.valueChanged.connect(
            self._on_master_volume_changed
        )
        transport.addWidget(self._master_volume_slider)

        # The same preset readout as each stem row, where it used to be a
        # plain grey label styled unlike every other volume (#186).
        self._master_volume_combo = QComboBox()
        make_display_combo(self._master_volume_combo)
        add_volume_presets(self._master_volume_combo)
        self._master_volume_combo.setFixedSize(70, COMPACT_BUTTON)
        self._master_volume_combo.setToolTip("Master volume")
        self._master_volume_combo.setAccessibleName("Master volume preset")
        self._master_volume_combo.activated.connect(self._on_master_preset)
        transport.addWidget(self._master_volume_combo)

        transport.addStretch()
        layout.addLayout(transport)

    @property
    def play_button(self) -> QPushButton:
        return self._play_button

    @property
    def stop_button(self) -> QPushButton:
        return self._stop_button

    @property
    def record_button(self) -> QPushButton:
        return self._record_button

    @property
    def time_label(self) -> QLabel:
        return self._time_label

    @property
    def master_volume_prefix(self) -> QLabel:
        return self._master_volume_prefix

    @property
    def master_volume_slider(self) -> QSlider:
        return self._master_volume_slider

    @property
    def master_volume_combo(self) -> QComboBox:
        return self._master_volume_combo

    @property
    def play_icon(self):
        return self._play_icon

    @property
    def pause_icon(self):
        return self._pause_icon

    @property
    def stop_icon(self):
        return self._stop_icon

    @property
    def record_icon(self):
        return self._record_icon

    def _on_master_volume_changed(self, value: int) -> None:
        show_preset_value(self._master_volume_combo, value)
        self.master_volume_changed.emit(value / 100.0)

    def _on_master_preset(self, index: int) -> None:
        value = self._master_volume_combo.itemData(index)
        if value is not None:
            self._master_volume_slider.setValue(value)

    def set_master_volume_display(self, volume: float) -> float:
        """Update the slider and label, returning the clamped gain."""
        value = max(0, min(200, int(round(float(volume) * 100))))
        if self._master_volume_slider.value() != value:
            self._master_volume_slider.blockSignals(True)
            self._master_volume_slider.setValue(value)
            self._master_volume_slider.blockSignals(False)
        show_preset_value(self._master_volume_combo, value)
        return value / 100.0

    def apply_theme(self, colors: dict[str, str], playing: bool) -> None:
        """Rebuild theme-dependent transport icons."""
        icon_color = QColor(colors["text"])
        self._record_icon = _record_icon(colors)
        self._record_button.setIcon(self._record_icon)
        self._play_icon = make_icon(draw_play, icon_color)
        self._pause_icon = make_icon(draw_pause, icon_color)
        self._stop_icon = make_icon(draw_stop, icon_color)
        self._play_button.setIcon(
            self._pause_icon if playing else self._play_icon
        )
        self._stop_button.setIcon(self._stop_icon)

    def set_playing(self, playing: bool) -> None:
        """Reflect the current player state in the play button."""
        self._play_button.setIcon(
            self._pause_icon if playing else self._play_icon
        )
        self._play_button.setAccessibleName("Pause" if playing else "Play")

"""Loop, trainer, speed, pitch, metronome, and count-in controls."""

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.player import (
    PITCH_MAX_SEMITONES,
    PITCH_MIN_SEMITONES,
    SPEED_PRESETS,
)
from src.ui.control_primitives import (
    ICON_SIZE,
    PitchSpinBox,
    draw_power,
    draw_repeat,
    fit_combo_width,
    fit_spinbox_width,
    make_display_combo,
    make_toggle_icon,
)
from src.ui.song_info_bar import SongInfoBar
from src.ui.styles import DARK_COLORS


class PracticeRack(QWidget):
    """Practice-oriented controls with narrow user-intent signals."""

    loop_a_requested = Signal()
    loop_b_requested = Signal()
    loop_toggled = Signal(bool)
    loop_clear_requested = Signal()
    speed_changed = Signal(float)
    pitch_changed = Signal(int)
    trainer_toggled = Signal(bool)
    trainer_start_changed = Signal(float)
    metronome_toggled = Signal(bool)
    bpm_changed = Signal(int)
    tap_requested = Signal()
    beat_sync_toggled = Signal(bool)
    beat_nudge_changed = Signal(int)
    metronome_volume_changed = Signal(float)
    count_in_toggled = Signal(bool)
    count_in_beats_changed = Signal(int)
    count_in_repeats_toggled = Signal(bool)

    def __init__(
        self,
        song_info_bar: SongInfoBar,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        icon_color = QColor(DARK_COLORS["text"])

        self._count_in_controls = QWidget(self)
        count_in = QHBoxLayout(self._count_in_controls)
        count_in.setContentsMargins(0, 0, 0, 0)

        self._count_in_label = QLabel("")
        self._count_in_label.setFixedWidth(32)
        self._count_in_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        count_in.addWidget(self._count_in_label)

        self._count_in_prefix = QLabel("Count-in:")
        count_in.addWidget(self._count_in_prefix)

        self._count_in_toggle = QPushButton()
        self._count_in_toggle.setObjectName("icon-btn")
        self._count_in_toggle.setCheckable(True)
        self._count_in_toggle.setFixedSize(36, 36)
        self._count_in_toggle.setIcon(
            make_toggle_icon(draw_power, icon_color)
        )
        self._count_in_toggle.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self._count_in_toggle.setToolTip(
            "Toggle count-in before playback (C)"
        )
        self._count_in_toggle.setAccessibleName("Toggle count-in")
        self._count_in_toggle.toggled.connect(
            self.count_in_toggled.emit
        )
        count_in.addWidget(self._count_in_toggle)

        self._count_in_beats_spin = QSpinBox()
        self._count_in_beats_spin.setRange(1, 8)
        self._count_in_beats_spin.setValue(4)
        self._count_in_beats_spin.setSuffix(" beats")
        fit_spinbox_width(self._count_in_beats_spin)
        self._count_in_beats_spin.setToolTip("Number of count-in beats")
        self._count_in_beats_spin.setAccessibleName("Count-in beats")
        self._count_in_beats_spin.valueChanged.connect(
            self.count_in_beats_changed.emit
        )
        count_in.addWidget(self._count_in_beats_spin)

        self._count_in_repeats = QPushButton()
        self._count_in_repeats.setObjectName("icon-btn")
        self._count_in_repeats.setCheckable(True)
        self._count_in_repeats.setFixedSize(36, 36)
        self._count_in_repeats.setIcon(
            make_toggle_icon(draw_repeat, icon_color)
        )
        self._count_in_repeats.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self._count_in_repeats.setToolTip(
            "Also count in before each A-B loop repeat"
        )
        self._count_in_repeats.setAccessibleName(
            "Count-in on loop repeats"
        )
        self._count_in_repeats.toggled.connect(
            self.count_in_repeats_toggled.emit
        )
        count_in.addWidget(self._count_in_repeats)

        loop_speed = QHBoxLayout()
        self._loop_a_button = QPushButton("Set A")
        self._loop_a_button.setToolTip("Set loop start point (A)")
        self._loop_a_button.setAccessibleName("Set loop A")
        self._loop_a_button.clicked.connect(self.loop_a_requested.emit)
        loop_speed.addWidget(self._loop_a_button)

        self._loop_b_button = QPushButton("Set B")
        self._loop_b_button.setToolTip("Set loop end point (B)")
        self._loop_b_button.setAccessibleName("Set loop B")
        self._loop_b_button.clicked.connect(self.loop_b_requested.emit)
        loop_speed.addWidget(self._loop_b_button)

        self._loop_toggle_button = QPushButton("Loop")
        self._loop_toggle_button.setCheckable(True)
        self._loop_toggle_button.setToolTip("Toggle A-B loop (L)")
        self._loop_toggle_button.setAccessibleName("Toggle loop")
        self._loop_toggle_button.toggled.connect(self.loop_toggled.emit)
        loop_speed.addWidget(self._loop_toggle_button)

        self._loop_clear_button = QPushButton("Clear")
        self._loop_clear_button.setToolTip("Clear loop points")
        self._loop_clear_button.setAccessibleName("Clear loop")
        self._loop_clear_button.clicked.connect(
            self.loop_clear_requested.emit
        )
        loop_speed.addWidget(self._loop_clear_button)

        self._loop_label = QLabel("")
        self._loop_label.setObjectName("subtle-label")
        loop_speed.addWidget(self._loop_label)

        loop_speed.addWidget(song_info_bar)
        loop_speed.addStretch()

        self._speed_label = QLabel("Speed:")
        loop_speed.addWidget(self._speed_label)

        self._speed_combo = QComboBox()
        for preset in SPEED_PRESETS:
            self._speed_combo.addItem(f"{preset}x", preset)
        self._speed_combo.setCurrentText("1.0x")
        fit_combo_width(self._speed_combo)
        self._speed_combo.setToolTip("Playback speed ([ / ])")
        self._speed_combo.setAccessibleName("Playback speed")
        self._speed_combo.currentIndexChanged.connect(
            self._emit_speed_changed
        )
        loop_speed.addWidget(self._speed_combo)

        self._speed_status = QLabel("")
        self._speed_status.setObjectName("subtle-label")
        loop_speed.addWidget(self._speed_status)

        self._pitch_label = QLabel("Pitch:")
        loop_speed.addWidget(self._pitch_label)

        self._pitch_spin = PitchSpinBox()
        self._pitch_spin.setRange(
            PITCH_MIN_SEMITONES,
            PITCH_MAX_SEMITONES,
        )
        self._pitch_spin.setValue(0)
        self._pitch_spin.setToolTip(
            "Transpose in semitones (Shift+Left / Shift+Right)"
        )
        self._pitch_spin.setAccessibleName("Pitch semitones")
        self._pitch_spin.valueChanged.connect(self.pitch_changed.emit)
        loop_speed.addWidget(self._pitch_spin)
        layout.addLayout(loop_speed)

        trainer = QHBoxLayout()
        self._trainer_check = QCheckBox("Loop Trainer")
        self._trainer_check.setToolTip(
            "Step speed up one preset each loop repeat, from the start "
            "speed up to 1.0x. Requires an A-B loop."
        )
        self._trainer_check.setAccessibleName("Loop trainer")
        self._trainer_check.toggled.connect(self.trainer_toggled.emit)
        trainer.addWidget(self._trainer_check)
        trainer.addWidget(QLabel("from"))

        self._trainer_start_combo = QComboBox()
        for preset in SPEED_PRESETS:
            if preset < 1.0:
                self._trainer_start_combo.addItem(f"{preset}x", preset)
        self._trainer_start_combo.setCurrentText("0.75x")
        fit_combo_width(self._trainer_start_combo)
        self._trainer_start_combo.setToolTip("Trainer start speed")
        self._trainer_start_combo.setAccessibleName("Trainer start speed")
        self._trainer_start_combo.currentIndexChanged.connect(
            self._emit_trainer_start_changed
        )
        trainer.addWidget(self._trainer_start_combo)
        trainer.addWidget(QLabel("→ 1.0x"))

        self._trainer_status = QLabel("")
        self._trainer_status.setObjectName("subtle-label")
        trainer.addWidget(self._trainer_status)
        trainer.addStretch()
        layout.addLayout(trainer)

        metronome = QHBoxLayout()
        self._metronome_label = QLabel("Metronome:")
        metronome.addWidget(self._metronome_label)

        self._metronome_toggle = QPushButton()
        self._metronome_toggle.setObjectName("icon-btn")
        self._metronome_toggle.setCheckable(True)
        self._metronome_toggle.setFixedSize(36, 36)
        self._metronome_toggle.setIcon(
            make_toggle_icon(draw_power, icon_color)
        )
        self._metronome_toggle.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self._metronome_toggle.setToolTip("Toggle metronome (M)")
        self._metronome_toggle.setAccessibleName("Toggle metronome")
        self._metronome_toggle.toggled.connect(
            self.metronome_toggled.emit
        )
        metronome.addWidget(self._metronome_toggle)

        self._bpm_spin = QSpinBox()
        self._bpm_spin.setRange(20, 300)
        self._bpm_spin.setValue(120)
        self._bpm_spin.setSuffix(" BPM")
        fit_spinbox_width(self._bpm_spin)
        self._bpm_spin.setToolTip("Metronome tempo")
        self._bpm_spin.setAccessibleName("Metronome BPM")
        self._bpm_spin.valueChanged.connect(self.bpm_changed.emit)
        metronome.addWidget(self._bpm_spin)

        self._tap_button = QPushButton("Tap")
        self._tap_button.setToolTip("Tap to set tempo")
        self._tap_button.setAccessibleName("Tap tempo")
        self._tap_button.clicked.connect(self.tap_requested.emit)
        metronome.addWidget(self._tap_button)

        self._beat_sync_button = QPushButton("Sync")
        self._beat_sync_button.setCheckable(True)
        self._beat_sync_button.setToolTip(
            "Sync metronome to detected beats "
            "(click on actual beat positions)"
        )
        self._beat_sync_button.setAccessibleName("Sync to track")
        self._beat_sync_button.setEnabled(False)
        self._beat_sync_button.toggled.connect(
            self.beat_sync_toggled.emit
        )
        metronome.addWidget(self._beat_sync_button)

        self._beat_nudge_spin = QSpinBox()
        self._beat_nudge_spin.setRange(-500, 500)
        self._beat_nudge_spin.setValue(0)
        self._beat_nudge_spin.setSuffix(" ms")
        fit_spinbox_width(self._beat_nudge_spin, sample="-500 ms")
        self._beat_nudge_spin.setToolTip(
            "Metronome nudge (shift metronome clicking)"
        )
        self._beat_nudge_spin.setAccessibleName("Sync Nudge")
        self._beat_nudge_spin.valueChanged.connect(
            self.beat_nudge_changed.emit
        )
        metronome.addWidget(self._beat_nudge_spin)

        self._metronome_volume_slider = QSlider(
            Qt.Orientation.Horizontal
        )
        self._metronome_volume_slider.setRange(0, 200)
        self._metronome_volume_slider.setValue(100)
        self._metronome_volume_slider.setFixedWidth(70)
        self._metronome_volume_slider.setToolTip(
            "Metronome volume (0-200%, double-click to reset)"
        )
        self._metronome_volume_slider.setAccessibleName(
            "Metronome volume"
        )
        self._metronome_volume_slider.valueChanged.connect(
            self._on_metronome_volume_changed
        )
        self._metronome_volume_slider.mouseDoubleClickEvent = (
            lambda _: self._metronome_volume_slider.setValue(100)
        )
        metronome.addWidget(self._metronome_volume_slider)

        self._metronome_volume_combo = QComboBox()
        make_display_combo(self._metronome_volume_combo)
        for value in range(0, 201, 20):
            self._metronome_volume_combo.addItem(f"{value}%", value)
        self._metronome_volume_combo.setCurrentText("100%")
        self._metronome_volume_combo.setFixedWidth(62)
        self._metronome_volume_combo.setToolTip("Metronome volume")
        self._metronome_volume_combo.setAccessibleName(
            "Metronome volume preset"
        )
        self._metronome_volume_combo.activated.connect(
            self._on_metronome_volume_preset
        )
        metronome.addWidget(self._metronome_volume_combo)

        metronome.addWidget(song_info_bar.detected_bpm_label)
        metronome.addStretch()
        layout.addLayout(metronome)

    @property
    def count_in_controls(self) -> QWidget:
        return self._count_in_controls

    @property
    def speed_combo(self) -> QComboBox:
        return self._speed_combo

    @property
    def pitch_spin(self) -> PitchSpinBox:
        return self._pitch_spin

    def _emit_speed_changed(self, _index: int) -> None:
        speed = self._speed_combo.currentData()
        if speed is not None:
            self.speed_changed.emit(float(speed))

    def _emit_trainer_start_changed(self, _index: int) -> None:
        speed = self._trainer_start_combo.currentData()
        if speed is not None:
            self.trainer_start_changed.emit(float(speed))

    def _on_metronome_volume_changed(self, value: int) -> None:
        self._metronome_volume_combo.blockSignals(True)
        self._metronome_volume_combo.setEditText(f"{value}%")
        self._metronome_volume_combo.blockSignals(False)
        self.metronome_volume_changed.emit(value / 100.0)

    def _on_metronome_volume_preset(self, index: int) -> None:
        value = self._metronome_volume_combo.itemData(index)
        if value is not None:
            self._metronome_volume_slider.setValue(value)

    def apply_theme(self, colors: dict[str, str]) -> None:
        """Rebuild all theme-sensitive toggle icons."""
        icon_color = QColor(colors["text"])
        self._metronome_toggle.setIcon(
            make_toggle_icon(draw_power, icon_color)
        )
        self._count_in_toggle.setIcon(
            make_toggle_icon(draw_power, icon_color)
        )
        self._count_in_repeats.setIcon(
            make_toggle_icon(draw_repeat, icon_color)
        )

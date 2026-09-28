"""Loop, trainer, speed, pitch, metronome, and count-in controls."""

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
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
    add_volume_presets,
    fix_button_size,
    ROW_BUTTON,
    ROW_ICON_SIZE,
    PitchSpinBox,
    draw_power,
    draw_repeat,
    fit_combo_width,
    fit_spinbox_width,
    make_display_combo,
    make_toggle_icon,
    show_preset_value,
)
from src.ui.song_info_bar import SongInfoBar
from src.ui.styles import DARK_COLORS


def _make_card(
    title: str, status: QLabel | None = None,
) -> tuple[QWidget, QVBoxLayout]:
    """Build a titled card and return it with the layout for its contents.

    Matches the label-above-frame idiom StemMixer uses for Stems and
    Recordings, so the practice controls read as part of the same interface.
    *status*, if given, sits beside the title: the title row has room to
    spare, where a status inside the card widened it and wrapped the cards.
    """
    container = QWidget()
    outer = QVBoxLayout(container)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(2)

    label = QLabel(title)
    label.setObjectName("title-label")
    if status is None:
        outer.addWidget(label)
    else:
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title_row.addWidget(label)
        title_row.addWidget(status)
        title_row.addStretch()
        outer.addLayout(title_row)

    frame = QFrame()
    frame.setObjectName("card-frame")
    frame.setFrameShape(QFrame.Shape.StyledPanel)
    # Preferred still stretches each card to its grid row, so side-by-side
    # cards share one bottom edge. Expanding also pulled a tall window's spare
    # height into the cards, which belongs to the waveform instead.
    frame.setSizePolicy(
        QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
    )
    body = QVBoxLayout(frame)
    body.setContentsMargins(8, 6, 8, 6)
    body.setSpacing(4)
    # A card shorter than its row neighbors keeps its rows packed at the top
    # instead of spreading them apart.
    body.setAlignment(Qt.AlignmentFlag.AlignTop)
    # The frame takes all of a short card's extra height, so titles and
    # frame tops stay aligned across a row instead of the title growing.
    outer.addWidget(frame, 1)

    return container, body


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
        self._count_in_controls.setObjectName("card-row")
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
        fix_button_size(self._count_in_toggle, ROW_BUTTON, ROW_BUTTON)
        self._count_in_toggle.setIcon(
            make_toggle_icon(draw_power, icon_color, ROW_ICON_SIZE)
        )
        self._count_in_toggle.setIconSize(QSize(ROW_ICON_SIZE, ROW_ICON_SIZE))
        self._count_in_toggle.setToolTip(
            "Toggle count-in before playback (C)"
        )
        self._count_in_toggle.setAccessibleName("Count-in")
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
        fix_button_size(self._count_in_repeats, ROW_BUTTON, ROW_BUTTON)
        self._count_in_repeats.setIcon(
            make_toggle_icon(draw_repeat, icon_color, ROW_ICON_SIZE)
        )
        self._count_in_repeats.setIconSize(
            QSize(ROW_ICON_SIZE, ROW_ICON_SIZE)
        )
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
        # Without this the spare card width is distributed between the label
        # and its controls, stranding them at opposite ends of the row.
        count_in.addStretch()

        # Key, chord, and tempo read as one strip under the waveform instead
        # of being split across two unrelated control rows.
        layout.addWidget(song_info_bar)

        self._cards_grid = QGridLayout()
        self._cards_grid.setSpacing(8)
        self._cards_grid.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._cards_grid)

        # The Loop Trainer's progress ("now 0.75x"). Inside the card it
        # widened the Loop card enough to wrap every card at 1366 px, and
        # the waveform lanes shrank to slivers (#211 review).
        self._trainer_status = QLabel("")
        self._trainer_status.setObjectName("subtle-label")
        loop_card, loop_body = _make_card(
            "Loop and Trainer", status=self._trainer_status,
        )
        self._loop_card = loop_card

        loop_row = QHBoxLayout()
        self._loop_a_button = QPushButton("Set A")
        self._loop_a_button.setToolTip("Set loop start point (A)")
        self._loop_a_button.setAccessibleName("Set loop A")
        self._loop_a_button.clicked.connect(self.loop_a_requested.emit)
        loop_row.addWidget(self._loop_a_button)

        self._loop_b_button = QPushButton("Set B")
        self._loop_b_button.setToolTip("Set loop end point (B)")
        self._loop_b_button.setAccessibleName("Set loop B")
        self._loop_b_button.clicked.connect(self.loop_b_requested.emit)
        loop_row.addWidget(self._loop_b_button)

        self._loop_toggle_button = QPushButton("Loop")
        self._loop_toggle_button.setCheckable(True)
        self._loop_toggle_button.setToolTip("Toggle A-B loop (L)")
        self._loop_toggle_button.setAccessibleName("A-B loop")
        self._loop_toggle_button.toggled.connect(self.loop_toggled.emit)
        loop_row.addWidget(self._loop_toggle_button)

        self._loop_clear_button = QPushButton("Clear")
        self._loop_clear_button.setToolTip("Clear loop points")
        self._loop_clear_button.setAccessibleName("Clear loop")
        self._loop_clear_button.clicked.connect(
            self.loop_clear_requested.emit
        )
        loop_row.addWidget(self._loop_clear_button)

        # Loop points are drawn as tags on the waveform markers; the label
        # stays for the facade's text API but is not shown (it widened the
        # card and wrapped the rack at 1366 px when a loop was set).
        self._loop_label = QLabel("", self)
        self._loop_label.setObjectName("subtle-label")
        self._loop_label.setVisible(False)

        loop_row.addStretch()
        loop_body.addLayout(loop_row)

        speed_card, speed_body = _make_card("Speed and Pitch")
        self._speed_card = speed_card

        speed_row = QHBoxLayout()
        self._speed_label = QLabel("Speed:")
        speed_row.addWidget(self._speed_label)

        self._speed_combo = QComboBox()
        for preset in SPEED_PRESETS:
            self._speed_combo.addItem(f"{preset}x", preset)
        self._speed_combo.setCurrentText("1.0x")
        fit_combo_width(self._speed_combo)
        self._speed_combo.setToolTip("Playback speed (Shift+Up / Shift+Down)")
        self._speed_combo.setAccessibleName("Playback speed")
        self._speed_combo.currentIndexChanged.connect(
            self._emit_speed_changed
        )
        speed_row.addWidget(self._speed_combo)

        self._speed_status = QLabel("")
        self._speed_status.setObjectName("subtle-label")
        speed_row.addWidget(self._speed_status)
        speed_row.addStretch()
        speed_body.addLayout(speed_row)

        pitch_row = QHBoxLayout()
        self._pitch_label = QLabel("Pitch:")
        pitch_row.addWidget(self._pitch_label)

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
        pitch_row.addWidget(self._pitch_spin)
        pitch_row.addStretch()
        speed_body.addLayout(pitch_row)

        trainer = QHBoxLayout()
        self._trainer_check = QCheckBox("Loop Trainer")
        self._trainer_check.setToolTip(
            "Step speed up one preset each loop repeat, from the start "
            "speed up to 1.0x. Requires an A-B loop."
        )
        self._trainer_check.setAccessibleName("Loop Trainer")
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
        trainer.addStretch()
        loop_body.addLayout(trainer)

        metronome_card, metronome_body = _make_card("Metronome and Count-in")
        self._metronome_card = metronome_card

        metronome = QHBoxLayout()
        self._metronome_label = QLabel("Metronome:")
        metronome.addWidget(self._metronome_label)

        self._metronome_toggle = QPushButton()
        self._metronome_toggle.setObjectName("icon-btn")
        self._metronome_toggle.setCheckable(True)
        fix_button_size(self._metronome_toggle, ROW_BUTTON, ROW_BUTTON)
        self._metronome_toggle.setIcon(
            make_toggle_icon(draw_power, icon_color, ROW_ICON_SIZE)
        )
        self._metronome_toggle.setIconSize(
            QSize(ROW_ICON_SIZE, ROW_ICON_SIZE)
        )
        self._metronome_toggle.setToolTip("Toggle metronome (M)")
        self._metronome_toggle.setAccessibleName("Metronome")
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
            "Sync the metronome to the beats detected in the song"
        )
        self._beat_sync_button.setAccessibleName("Sync metronome to song")
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
            "Shift the metronome clicks earlier or later"
        )
        self._beat_nudge_spin.setAccessibleName("Metronome nudge")
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
        add_volume_presets(self._metronome_volume_combo)
        self._metronome_volume_combo.setFixedWidth(70)  # room for the chevron
        self._metronome_volume_combo.setToolTip("Metronome volume")
        self._metronome_volume_combo.setAccessibleName(
            "Metronome volume preset"
        )
        self._metronome_volume_combo.activated.connect(
            self._on_metronome_volume_preset
        )
        metronome.addWidget(self._metronome_volume_combo)

        metronome.addStretch()
        metronome_body.addLayout(metronome)
        # Count-in belongs next to the tempo it counts, not alone in the
        # transport corner.
        metronome_body.addWidget(self._count_in_controls)

        # The rack's width comes from the space it is given, never from its
        # cards. Otherwise three cards side by side hold the rack at their
        # combined width, so it can never be narrowed enough to wrap them:
        # when a scrollbar appears, or loop points widen the loop card, the
        # rack overflows its scroll area and is clipped instead. The explicit
        # minimum set in _reflow_cards keeps the wrapped layout whole.
        # Vertically the rack never grows past its content: once the waveform
        # reaches its cap, spare height opens above the mixer instead of
        # stretching the cards.
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Maximum
        )
        self._cards_wide: bool | None = None
        self._reflow_cards()

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Reflow the cards when the available width changes."""
        super().resizeEvent(event)
        self._reflow_cards()

    def event(self, event) -> bool:
        """Reflow when card content changes size at a constant width."""
        handled = super().event(event)
        if event.type() == QEvent.Type.LayoutRequest:
            self._reflow_cards()
        return handled

    def _required_card_width(self) -> int:
        """Width needed to stand all three cards side by side."""
        cards = (self._loop_card, self._speed_card, self._metronome_card)
        spacing = self._cards_grid.horizontalSpacing() * (len(cards) - 1)
        return sum(c.minimumSizeHint().width() for c in cards) + spacing

    def _wrapped_card_width(self) -> int:
        """Width needed with the metronome card wrapped to its own row."""
        spacing = self._cards_grid.horizontalSpacing()
        top = (
            self._loop_card.minimumSizeHint().width()
            + spacing
            + self._speed_card.minimumSizeHint().width()
        )
        return max(top, self._metronome_card.minimumSizeHint().width())

    def _reflow_cards(self) -> None:
        """Stand the cards in one row, or wrap to two when width is short.

        Side by side the three cards need roughly 1140px. The window's
        minimum is 900px wide, which leaves the rack far less than that, and
        without wrapping the metronome card's labels and buttons are clipped
        to fragments. Wrapped, the widest row is the metronome card alone,
        which fits comfortably at the minimum window size.
        """
        floor = self._wrapped_card_width()
        if self.minimumWidth() != floor:
            self.setMinimumWidth(floor)

        wide = self.width() >= self._required_card_width()
        if wide == self._cards_wide:
            return
        self._cards_wide = wide

        grid = self._cards_grid
        for card in (self._loop_card, self._speed_card, self._metronome_card):
            grid.removeWidget(card)

        grid.addWidget(self._loop_card, 0, 0)
        grid.addWidget(self._speed_card, 0, 1)
        if wide:
            grid.addWidget(self._metronome_card, 0, 2)
        else:
            grid.addWidget(self._metronome_card, 1, 0, 1, 2)
        grid.setColumnStretch(0, 4)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 5 if wide else 0)

    @property
    def cards_side_by_side(self) -> bool:
        """Whether the cards currently stand in a single row."""
        return bool(self._cards_wide)

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
        show_preset_value(self._metronome_volume_combo, value)
        self.metronome_volume_changed.emit(value / 100.0)

    def _on_metronome_volume_preset(self, index: int) -> None:
        value = self._metronome_volume_combo.itemData(index)
        if value is not None:
            self._metronome_volume_slider.setValue(value)

    def apply_theme(self, colors: dict[str, str]) -> None:
        """Rebuild all theme-sensitive toggle icons."""
        icon_color = QColor(colors["text"])
        on_accent = QColor(colors["on_accent"])
        self._metronome_toggle.setIcon(make_toggle_icon(
            draw_power, icon_color, ROW_ICON_SIZE, checked_color=on_accent,
        ))
        self._count_in_toggle.setIcon(make_toggle_icon(
            draw_power, icon_color, ROW_ICON_SIZE, checked_color=on_accent,
        ))
        self._count_in_repeats.setIcon(make_toggle_icon(
            draw_repeat, icon_color, ROW_ICON_SIZE, checked_color=on_accent,
        ))

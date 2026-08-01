"""Stem and recording-row presentation for PlayerControls."""

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor
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

from src.player import MultiTrackPlayer
from src.ui.control_primitives import (
    STEM_ICON_SIZE,
    draw_mute,
    draw_solo,
    draw_trash,
    fit_spinbox_width,
    make_display_combo,
    make_icon,
    make_toggle_icon,
)
from src.ui.styles import (
    DARK_COLORS,
    LIGHT_COLORS,
    RECORDING_COLOR,
    STEM_COLORS_DARK,
    STEM_COLORS_LIGHT,
)

MAX_RECORDING_TAKES = 2


class StemRow(QWidget):
    """A single stem row with mute, solo, and volume controls."""

    mix_changed = Signal()

    def __init__(
        self,
        stem_name: str,
        player: MultiTrackPlayer,
        theme: str = "dark",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._stem_name = stem_name
        self._player = player
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setStyleSheet("background: transparent;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        palette = (
            STEM_COLORS_DARK if theme == "dark" else STEM_COLORS_LIGHT
        )
        color = palette.get(stem_name, "#95a5a6")

        self._label = QLabel(stem_name.capitalize())
        self._label.setFixedWidth(70)
        self._label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self._label.setStyleSheet(
            f"color: {color}; font-weight: bold;"
        )
        layout.addWidget(self._label)

        colors = DARK_COLORS if theme == "dark" else LIGHT_COLORS
        icon_color = QColor(colors["text"])
        display = stem_name.capitalize()

        self._mute_btn = QPushButton()
        self._mute_btn.setObjectName("icon-btn")
        self._mute_btn.setCheckable(True)
        self._mute_btn.setFixedSize(28, 28)
        self._mute_btn.setIcon(
            make_toggle_icon(draw_mute, icon_color, STEM_ICON_SIZE)
        )
        self._mute_btn.setIconSize(QSize(STEM_ICON_SIZE, STEM_ICON_SIZE))
        self._mute_btn.setToolTip(f"Mute {display}")
        self._mute_btn.setAccessibleName(f"Mute {display}")
        self._mute_btn.toggled.connect(self._on_mute)
        layout.addWidget(self._mute_btn)

        self._solo_btn = QPushButton()
        self._solo_btn.setObjectName("icon-btn")
        self._solo_btn.setCheckable(True)
        self._solo_btn.setFixedSize(28, 28)
        self._solo_btn.setIcon(
            make_toggle_icon(draw_solo, icon_color, STEM_ICON_SIZE)
        )
        self._solo_btn.setIconSize(QSize(STEM_ICON_SIZE, STEM_ICON_SIZE))
        self._solo_btn.setToolTip(f"Solo {display}")
        self._solo_btn.setAccessibleName(f"Solo {display}")
        self._solo_btn.toggled.connect(self._on_solo)
        layout.addWidget(self._solo_btn)

        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 200)
        self._volume_slider.setValue(100)
        self._volume_slider.setFixedWidth(120)
        self._volume_slider.setToolTip(
            f"{display} volume (0-200%, double-click to reset)"
        )
        self._volume_slider.setAccessibleName(f"{display} volume")
        self._volume_slider.valueChanged.connect(self._on_volume)
        self._volume_slider.mouseDoubleClickEvent = (
            lambda _: self._volume_slider.setValue(100)
        )
        layout.addWidget(self._volume_slider)

        self._vol_combo = QComboBox()
        make_display_combo(self._vol_combo)
        for value in range(0, 201, 20):
            self._vol_combo.addItem(f"{value}%", value)
        self._vol_combo.setCurrentText("100%")
        self._vol_combo.setFixedSize(62, 28)
        self._vol_combo.setToolTip(f"{display} volume")
        self._vol_combo.setAccessibleName(f"{display} volume preset")
        self._vol_combo.activated.connect(self._on_vol_combo)
        layout.addWidget(self._vol_combo)

    def _on_mute(self, checked: bool) -> None:
        self._player.set_mute(self._stem_name, checked)
        self._mute_btn.clearFocus()
        self.mix_changed.emit()

    def _on_solo(self, checked: bool) -> None:
        self._player.set_solo(self._stem_name, checked)
        self._solo_btn.clearFocus()
        self.mix_changed.emit()

    def _on_volume(self, value: int) -> None:
        self._player.set_volume(self._stem_name, value / 100.0)
        self._vol_combo.blockSignals(True)
        self._vol_combo.setEditText(f"{value}%")
        self._vol_combo.blockSignals(False)
        self.mix_changed.emit()

    def _on_vol_combo(self, index: int) -> None:
        value = self._vol_combo.itemData(index)
        if value is not None:
            self._volume_slider.setValue(value)

    def set_muted(self, muted: bool) -> None:
        self._mute_btn.setChecked(muted)

    def set_soloed(self, soloed: bool) -> None:
        self._solo_btn.setChecked(soloed)

    def set_volume_slider(self, value: int) -> None:
        self._volume_slider.setValue(value)
        self._vol_combo.setEditText(f"{value}%")

    def apply_stem_theme(self, theme: str) -> None:
        palette = (
            STEM_COLORS_DARK if theme == "dark" else STEM_COLORS_LIGHT
        )
        color = palette.get(self._stem_name, "#95a5a6")
        self._label.setStyleSheet(
            f"color: {color}; font-weight: bold;"
        )
        colors = DARK_COLORS if theme == "dark" else LIGHT_COLORS
        icon_color = QColor(colors["text"])
        self._mute_btn.setIcon(
            make_toggle_icon(draw_mute, icon_color, STEM_ICON_SIZE)
        )
        self._solo_btn.setIcon(
            make_toggle_icon(draw_solo, icon_color, STEM_ICON_SIZE)
        )


class RecordingStemRow(StemRow):
    """A recording take row with nudge and delete controls."""

    delete_requested = Signal(str)

    def __init__(
        self,
        stem_name: str,
        display_name: str,
        player: MultiTrackPlayer,
        theme: str = "dark",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(stem_name, player, theme, parent)
        self._label.setText(display_name)
        self._label.setStyleSheet(
            f"color: {RECORDING_COLOR}; font-weight: bold;"
        )

        layout = self.layout()
        insert_position = layout.count()
        self._nudge_spin = QSpinBox()
        self._nudge_spin.setRange(-200, 200)
        self._nudge_spin.setValue(0)
        self._nudge_spin.setSuffix(" ms")
        fit_spinbox_width(self._nudge_spin, sample="-200 ms")
        self._nudge_spin.setToolTip(
            f"Nudge {display_name} alignment (-200 to +200 ms)"
        )
        self._nudge_spin.setAccessibleName(f"Nudge {display_name}")
        self._nudge_spin.valueChanged.connect(self._on_nudge_changed)
        layout.insertWidget(insert_position, self._nudge_spin)
        insert_position += 1

        colors = DARK_COLORS if theme == "dark" else LIGHT_COLORS
        self._delete_btn = QPushButton()
        self._delete_btn.setObjectName("icon-btn")
        self._delete_btn.setFixedSize(28, 28)
        self._delete_btn.setIcon(
            make_icon(
                draw_trash,
                QColor(colors["text"]),
                STEM_ICON_SIZE,
            )
        )
        self._delete_btn.setIconSize(
            QSize(STEM_ICON_SIZE, STEM_ICON_SIZE)
        )
        self._delete_btn.setToolTip(f"Delete {display_name}")
        self._delete_btn.setAccessibleName(f"Delete {display_name}")
        self._delete_btn.clicked.connect(
            lambda: self.delete_requested.emit(self._stem_name)
        )
        layout.insertWidget(insert_position, self._delete_btn)

    def _on_nudge_changed(self, value: int) -> None:
        self._player.nudge_stem(self._stem_name, float(value))
        self.mix_changed.emit()

    def set_nudge(self, value: int) -> None:
        self._nudge_spin.setValue(value)


class StemMixer(QWidget):
    """Own stem and recording rows plus their visible section frames."""

    mix_changed = Signal()

    def __init__(
        self,
        player: MultiTrackPlayer,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._player = player
        self._theme = "dark"
        self._stem_rows: dict[str, StemRow] = {}
        self._recording_rows: dict[str, RecordingStemRow] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._mixer_label = QLabel("Stems")
        self._mixer_label.setObjectName("title-label")
        layout.addWidget(self._mixer_label)

        self._stems_frame = QFrame()
        self._stems_frame.setObjectName("card-frame")
        self._stems_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._stem_container = QVBoxLayout(self._stems_frame)
        self._stem_container.setContentsMargins(6, 4, 6, 4)
        self._stem_container.setSpacing(2)
        layout.addWidget(self._stems_frame)

        self._recordings_label = QLabel("Recordings")
        self._recordings_label.setObjectName("title-label")
        self._recordings_label.setVisible(False)
        layout.addWidget(self._recordings_label)

        self._recordings_frame = QFrame()
        self._recordings_frame.setObjectName("card-frame")
        self._recordings_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._recordings_frame.setVisible(False)
        self._recordings_container = QVBoxLayout(
            self._recordings_frame
        )
        self._recordings_container.setContentsMargins(6, 4, 6, 4)
        self._recordings_container.setSpacing(2)
        layout.addWidget(self._recordings_frame)

    @property
    def stem_rows(self) -> dict[str, StemRow]:
        return self._stem_rows

    @property
    def recording_rows(self) -> dict[str, RecordingStemRow]:
        return self._recording_rows

    @property
    def recording_count(self) -> int:
        return len(self._recording_rows)

    @property
    def max_recordings_reached(self) -> bool:
        return len(self._recording_rows) >= MAX_RECORDING_TAKES

    def stem_names(self) -> list[str]:
        """Return stem row names in display order (source stems, then recordings)."""
        return list(self._stem_rows) + list(self._recording_rows)

    def set_stem_names(self, stem_names: list[str]) -> None:
        """Replace source rows while preserving matching player state."""
        saved_muted = set(self._player.muted_stems)
        saved_soloed = set(self._player.soloed_stems)
        for row in self._stem_rows.values():
            row.setParent(None)
            row.deleteLater()
        self._stem_rows.clear()
        self.clear_recording_rows()

        for name in stem_names:
            row = StemRow(name, self._player, self._theme)
            row.mix_changed.connect(self.mix_changed.emit)
            self._stem_container.addWidget(row)
            self._stem_rows[name] = row
            if name in saved_muted:
                row.set_muted(True)
            if name in saved_soloed:
                row.set_soloed(True)

    def restore_stem_state(
        self,
        muted: set[str],
        soloed: set[str],
        volumes: dict[str, float],
    ) -> None:
        for name, row in self._stem_rows.items():
            row.set_muted(name in muted)
            row.set_soloed(name in soloed)
            row.set_volume_slider(round(volumes.get(name, 1.0) * 100))

    def add_recording_row(
        self,
        stem_name: str,
        display_name: str,
    ) -> RecordingStemRow:
        row = RecordingStemRow(
            stem_name,
            display_name,
            self._player,
            self._theme,
        )
        row.mix_changed.connect(self.mix_changed.emit)
        self._recordings_container.addWidget(row)
        self._recording_rows[stem_name] = row
        self._recordings_label.setVisible(True)
        self._recordings_frame.setVisible(True)
        return row

    def remove_recording_row(self, stem_name: str) -> None:
        row = self._recording_rows.pop(stem_name, None)
        if row is not None:
            row.setParent(None)
            row.deleteLater()
        if not self._recording_rows:
            self._recordings_label.setVisible(False)
            self._recordings_frame.setVisible(False)

    def clear_recording_rows(self) -> None:
        for row in self._recording_rows.values():
            row.setParent(None)
            row.deleteLater()
        self._recording_rows.clear()
        self._recordings_label.setVisible(False)
        self._recordings_frame.setVisible(False)

    def apply_theme(self, theme: str) -> None:
        self._theme = theme
        for row in self._stem_rows.values():
            row.apply_stem_theme(theme)
        for row in self._recording_rows.values():
            row.apply_stem_theme(theme)

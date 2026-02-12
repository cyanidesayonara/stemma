"""Player transport controls and per-stem mute/solo mixer.

Transport: Play/Pause, Stop, seek slider, time display.
Per-stem row: label, Mute button, Solo button.
Color-coded stems. Full implementation in ticket #9.
"""

from PySide6.QtCore import Qt
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


def _format_time(seconds: float) -> str:
    """Format seconds as mm:ss."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


class StemRow(QWidget):
    """A single stem row with label, mute, and solo buttons."""

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
        self._mute_btn.setToolTip(f"Mute {stem_name}")
        self._mute_btn.toggled.connect(self._on_mute)
        layout.addWidget(self._mute_btn)

        self._solo_btn = QPushButton("S")
        self._solo_btn.setCheckable(True)
        self._solo_btn.setFixedSize(32, 28)
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

    def _on_solo(self, checked: bool) -> None:
        self._player.set_solo(self._stem_name, checked)

    def _on_volume(self, value: int) -> None:
        gain = value / 100.0
        self._player.set_volume(self._stem_name, gain)
        self._vol_label.setText(f"{value}%")


class PlayerControls(QWidget):
    """Transport controls and stem mixer panel."""

    def __init__(self, player: MultiTrackPlayer,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._player = player
        self._stem_rows: dict[str, StemRow] = {}
        self._seeking = False

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

        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 1000)
        self._seek_slider.sliderPressed.connect(self._on_seek_start)
        self._seek_slider.sliderReleased.connect(self._on_seek_end)
        transport.addWidget(self._seek_slider)

        layout.addLayout(transport)

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
            self._stem_container.addWidget(row)
            self._stem_rows[name] = row

    # -- Transport slots --

    def _on_play_pause(self) -> None:
        if self._player.is_playing:
            self._player.pause()
        else:
            self._player.play()

    def _on_stop(self) -> None:
        self._player.stop()

    def _on_seek_start(self) -> None:
        self._seeking = True

    def _on_seek_end(self) -> None:
        self._seeking = False
        total = self._player.total_seconds
        if total > 0:
            ratio = self._seek_slider.value() / 1000.0
            self._player.seek(ratio * total)

    def _on_position_changed(self, pos_s: float) -> None:
        total = self._player.total_seconds
        self._time_label.setText(
            f"{_format_time(pos_s)} / {_format_time(total)}"
        )
        if not self._seeking and total > 0:
            self._seek_slider.setValue(int(pos_s / total * 1000))

    def _on_state_changed(self, playing: bool) -> None:
        self._play_btn.setText("Pause" if playing else "Play")

    def _on_play_finished(self) -> None:
        self._play_btn.setText("Play")
        self._seek_slider.setValue(0)

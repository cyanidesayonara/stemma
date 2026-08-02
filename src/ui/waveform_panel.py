"""Framed stacked-stem waveform presentation.

Split out of TransportBar so the waveform can scroll with the rest of the
practice content while the transport stays anchored to the bottom of the
window.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QVBoxLayout, QWidget

from src.ui.waveform_stack_widget import WaveformStackWidget


class WaveformPanel(QWidget):
    """The stem-lane waveform inside its card frame."""

    seek_requested = Signal(float)  # Emits seconds

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._frame = QFrame()
        self._frame.setObjectName("card-frame")
        self._frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame_layout = QVBoxLayout(self._frame)
        frame_layout.setContentsMargins(4, 4, 4, 4)

        self._waveform = WaveformStackWidget()
        self._waveform.seek_requested.connect(self.seek_requested.emit)
        frame_layout.addWidget(self._waveform)
        layout.addWidget(self._frame)

    @property
    def frame(self) -> QFrame:
        return self._frame

    @property
    def waveform(self) -> WaveformStackWidget:
        return self._waveform

    def apply_theme(self, colors: dict[str, str]) -> None:
        """Repaint the lanes for a new theme."""
        self._waveform.set_theme_colors(colors)

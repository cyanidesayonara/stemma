"""Tempo, key, and live-chord presentation for PlayerControls."""

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from src.ui.styles import CONFIDENCE_COLORS, DARK_COLORS, LIGHT_COLORS


class SongInfoBar(QWidget):
    """Own detection state and the tempo, key, and chord readouts."""

    key_redetect_requested = Signal()
    bpm_redetect_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._theme = "dark"
        self._key_confidence = ""
        self._bpm_confidence = ""
        self._detected_key = ""
        self._detected_bpm = ""

        self._key_label = QLabel("")
        self._key_label.setTextFormat(Qt.TextFormat.RichText)
        self._key_label.setToolTip(
            "Detected musical key (double-click to re-detect)"
        )
        self._key_label.setAccessibleName("Detected key")
        self._key_label.setObjectName("subtle-label")
        self._key_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._key_label.installEventFilter(self)
        layout.addWidget(self._key_label)

        self._chord_label = QLabel("")
        self._chord_label.setTextFormat(Qt.TextFormat.RichText)
        self._chord_label.setToolTip("Detected chord (suggestion)")
        self._chord_label.setAccessibleName("Detected chord")
        layout.addWidget(self._chord_label)

        self._detected_bpm_label = QLabel("")
        self._detected_bpm_label.setTextFormat(Qt.TextFormat.RichText)
        self._detected_bpm_label.setToolTip(
            "Detected tempo — suggestion only (double-click to re-detect)"
        )
        self._detected_bpm_label.setAccessibleName("Detected BPM")
        self._detected_bpm_label.setObjectName("subtle-label")
        self._detected_bpm_label.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        self._detected_bpm_label.installEventFilter(self)
        # Tempo belongs with key and chord as one readout. It used to be
        # re-parented into the metronome row, which rendered "detecting..."
        # twice in different places while detection ran.
        layout.addWidget(self._detected_bpm_label)
        layout.addStretch()

    @property
    def key_label(self) -> QLabel:
        return self._key_label

    @property
    def chord_label(self) -> QLabel:
        return self._chord_label

    @property
    def detected_bpm_label(self) -> QLabel:
        return self._detected_bpm_label

    @property
    def detected_key(self) -> str:
        return self._detected_key

    @property
    def detected_bpm_text(self) -> str:
        return self._detected_bpm

    @property
    def key_confidence(self) -> str:
        return self._key_confidence

    @property
    def bpm_confidence(self) -> str:
        return self._bpm_confidence

    def eventFilter(self, watched, event):  # noqa: N802
        """Turn readout double-clicks into narrow re-detection intents."""
        if event.type() == QEvent.Type.MouseButtonDblClick:
            if watched is self._key_label:
                self.key_redetect_requested.emit()
                return True
            if watched is self._detected_bpm_label:
                self.bpm_redetect_requested.emit()
                return True
        return super().eventFilter(watched, event)

    def confidence_color(self, level: str) -> str:
        return CONFIDENCE_COLORS[self._theme].get(level, "")

    def badge_style(self) -> str:
        colors = LIGHT_COLORS if self._theme == "light" else DARK_COLORS
        return (
            f"background: {colors['surface0']}; "
            f"color: {colors['text']}; "
            f"border: 1px solid {colors['surface1']}; "
            f"border-radius: 4px; "
            f"padding: 1px 6px; "
            f"margin: 0px 1px;"
        )

    def badge_html(
        self,
        label: str,
        value: str,
        color: str = "",
    ) -> str:
        colors = LIGHT_COLORS if self._theme == "light" else DARK_COLORS
        text_color = colors["text"]
        value_color = color or text_color
        if label:
            return (
                f'<span style="color:{text_color};">{label} </span>'
                f'<span style="color:{value_color};">{value}</span>'
            )
        return f'<span style="color:{value_color};">{value}</span>'

    def set_key(
        self,
        key: str,
        confidence: str = "",
        *,
        effective_key: str | None = None,
        pitch: int = 0,
    ) -> None:
        """Store and render detected key state."""
        self._detected_key = key
        self._key_confidence = confidence if key else ""
        if not key:
            self._key_label.setText("")
            self._key_label.setStyleSheet("")
            self._key_label.setToolTip(
                "Detected musical key (double-click to re-detect)"
            )
            return

        color = self.confidence_color(confidence) if confidence else ""
        shown = key if effective_key is None else f"{key} → {effective_key}"
        self._key_label.setStyleSheet(self.badge_style())
        self._key_label.setText(self.badge_html("Key:", shown, color))
        tooltip = [f"Detected key: {key}"]
        if effective_key is not None:
            tooltip.append(
                f"Transposed by {pitch:+d} st: {effective_key}"
            )
        tooltip.extend([
            f"Confidence: {confidence}",
            "Double-click to re-detect",
        ])
        self._key_label.setToolTip("\n".join(tooltip))

    def set_bpm(self, text: str, confidence: str = "") -> None:
        """Store and render detected tempo state."""
        self._detected_bpm = text
        self._bpm_confidence = confidence if text else ""
        if not text:
            self._detected_bpm_label.setText("")
            self._detected_bpm_label.setStyleSheet("")
            self._detected_bpm_label.setToolTip(
                "Detected tempo — suggestion only (double-click to re-detect)"
            )
            return

        color = self.confidence_color(confidence) if confidence else ""
        self._detected_bpm_label.setStyleSheet(self.badge_style())
        self._detected_bpm_label.setText(
            self.badge_html("Tempo:", text, color)
        )
        tooltip = [f"Detected tempo: {text}"]
        if confidence:
            tooltip.append(f"Confidence: {confidence}")
        tooltip.append("Double-click to re-detect")
        self._detected_bpm_label.setToolTip("\n".join(tooltip))

    def set_chord(self, chord: str) -> None:
        """Render the current chord or the stopped/silent placeholder."""
        self._chord_label.setStyleSheet(self.badge_style())
        self._chord_label.setText(
            self.badge_html("Chord:", chord if chord else "--")
        )

    def clear_chord(self) -> None:
        self._chord_label.setText("")
        self._chord_label.setStyleSheet("")
        self._chord_label.setToolTip("Detected chord (suggestion)")

    def show_detection_status(self, text: str) -> None:
        """Display a neutral status in both asynchronous readouts."""
        colors = LIGHT_COLORS if self._theme == "light" else DARK_COLORS
        style = (
            f"background: {colors['surface0']}; "
            f"border: 1px solid {colors['surface1']}; "
            f"border-radius: 4px; "
            f"padding: 1px 6px; color: {colors['text']};"
        )
        for label in (self._key_label, self._detected_bpm_label):
            label.setStyleSheet(style)
            label.setText(text)

    def apply_theme(
        self,
        theme: str,
        *,
        effective_key: str | None = None,
        pitch: int = 0,
        chord: str | None = None,
        has_chords: bool = False,
    ) -> None:
        """Regenerate rich-text colors without changing stored state."""
        self._theme = theme
        if self._detected_key:
            self.set_key(
                self._detected_key,
                self._key_confidence,
                effective_key=effective_key,
                pitch=pitch,
            )
        if self._detected_bpm:
            self.set_bpm(self._detected_bpm, self._bpm_confidence)
        if has_chords:
            self.set_chord(chord or "--")

    def clear(self) -> None:
        """Reset all song-specific readouts and persisted presentation."""
        self.set_key("")
        self.set_bpm("")
        self.clear_chord()

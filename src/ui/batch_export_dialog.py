"""Choices for exporting several songs at once."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class BatchExportDialog(QDialog):
    """Asks which stems to export, how, and in which format."""

    def __init__(
        self,
        song_count: int,
        stem_names: Sequence[str],
        default_format: str = "wav",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Songs")
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Export {song_count} songs. Stems to include:"))
        self._stem_checks: dict[str, QCheckBox] = {}
        for name in stem_names:
            check = QCheckBox(name.capitalize())
            check.setChecked(True)
            check.toggled.connect(self._update_ok)
            layout.addWidget(check)
            self._stem_checks[name] = check

        form = QFormLayout()
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._layout_combo = QComboBox()
        self._layout_combo.addItem("One mix per song", False)
        self._layout_combo.addItem("Separate stem files", True)
        form.addRow("&Export as:", self._layout_combo)
        self._format_combo = QComboBox()
        self._format_combo.addItem("WAV", ".wav")
        self._format_combo.addItem("MP3", ".mp3")
        self._format_combo.setCurrentIndex(1 if default_format == "mp3" else 0)
        form.addRow("&Format:", self._format_combo)
        layout.addLayout(form)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok.setText("Choose Folder...")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

    def _update_ok(self) -> None:
        self._ok.setEnabled(bool(self.stems))

    @property
    def stems(self) -> list[str]:
        """Names of the checked stems."""
        return [n for n, c in self._stem_checks.items() if c.isChecked()]

    @property
    def separate_files(self) -> bool:
        """True to write one file per stem instead of a mix."""
        return bool(self._layout_combo.currentData())

    @property
    def extension(self) -> str:
        """File extension of the chosen format, with the dot."""
        return str(self._format_combo.currentData())

"""What the empty player shows while an imported song is being separated.

A long song can take several minutes to separate on the CPU (4 and 6
stems). The player used to sit empty for all of it, with progress only in
the library row (#159). This view names the song and shows the stage, a
progress bar, and an estimate of the time left, and says the app stays
usable meanwhile.
"""

from __future__ import annotations

import math
import time

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

# Both workers report 0-15 % for loading, resampling, and model setup,
# then map the separation itself to 15-90 %; the estimate uses only that
# part, whose pace is steady. After 90 % come post-processing and saving,
# which are short and send no reports of their own.
_ESTIMATE_FROM_PERCENT = 15
_SEPARATION_END_PERCENT = 90
_TAIL_S = 5.0
_MIN_ELAPSED_S = 4.0
_MIN_PROGRESS = 2.0
_SMOOTHING = 0.3


def stage_for(percent: int) -> str:
    """Plain stage name for a worker progress *percent*.

    The workers' own messages ("Processing segment 57/150...", "Using
    DirectML (GPU) for MDX separation.") are not shown: the percentage and
    the time left already say how far along the separation is.
    """
    if percent < 5:
        return "Loading audio…"
    if percent < _ESTIMATE_FROM_PERCENT:
        return "Preparing the model…"
    if percent < _SEPARATION_END_PERCENT:
        return "Separating stems…"
    return "Saving stems…"


class SeparationEta:
    """Estimate the seconds left from (time, percent) progress reports."""

    def __init__(self) -> None:
        self._start: tuple[float, float] | None = None
        self._rate: float | None = None  # Seconds per percent.

    def update(self, percent: float, now: float) -> float | None:
        """Add a report; return the seconds left, or None while unsure."""
        if percent >= _SEPARATION_END_PERCENT:
            return 0.0
        if percent < _ESTIMATE_FROM_PERCENT:
            return None
        if self._start is None:
            self._start = (now, percent)
            return None
        t0, p0 = self._start
        elapsed = now - t0
        done = percent - p0
        if elapsed >= _MIN_ELAPSED_S and done >= _MIN_PROGRESS:
            rate = elapsed / done
            if self._rate is None:
                self._rate = rate
            else:
                # Blend towards the latest rate so the estimate doesn't jump.
                self._rate += _SMOOTHING * (rate - self._rate)
        if self._rate is None:
            return None
        left = (_SEPARATION_END_PERCENT - percent) * self._rate
        return max(0.0, left) + _TAIL_S


def format_eta(seconds: float | None) -> str:
    """Readable time left: "About 3 min left", "Almost done"."""
    if seconds is None:
        return "Estimating time left…"
    if seconds < 10:
        return "Almost done"
    if seconds < 60:
        return f"About {int(math.ceil(seconds / 10.0) * 10)} s left"
    return f"About {int(math.ceil(seconds / 60.0))} min left"


class _ElidedLabel(QLabel):
    """One-line label that elides its text to the width it is given.

    Its minimum width is zero, so a long title never widens the window;
    the full text is in the tooltip.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = ""

    def set_full_text(self, text: str) -> None:
        """Show *text*, elided to fit, with all of it in the tooltip."""
        self._full_text = text
        self.setToolTip(text)
        self.updateGeometry()
        self._elide()

    def full_text(self) -> str:
        return self._full_text

    def sizeHint(self) -> QSize:
        margins = self.contentsMargins()
        width = (self.fontMetrics().horizontalAdvance(self._full_text)
                 + margins.left() + margins.right() + 2)
        return QSize(width, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        shown = self.fontMetrics().elidedText(
            self._full_text,
            Qt.TextElideMode.ElideRight,
            max(0, self.contentsRect().width()),
        )
        if shown != self.text():
            self.setText(shown)


class SeparationView(QWidget):
    """Song, stage, progress, and time left for the running separation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._eta = SeparationEta()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(6)
        center = Qt.AlignmentFlag.AlignHCenter

        self._title = _ElidedLabel()
        self._title.setObjectName("title-label")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title, alignment=center)

        self._artist = _ElidedLabel()
        self._artist.setObjectName("subtle-label")
        self._artist.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._artist, alignment=center)

        self._stage = QLabel("")
        self._stage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._stage, alignment=center)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setFixedWidth(360)
        self._bar.setAccessibleName("Separation progress")
        layout.addWidget(self._bar, alignment=center)

        self._time_left = QLabel("")
        self._time_left.setObjectName("subtle-label")
        layout.addWidget(self._time_left, alignment=center)

        self._queued = QLabel("")
        self._queued.setObjectName("subtle-label")
        self._queued.setVisible(False)
        layout.addWidget(self._queued, alignment=center)

        # One line: a centred, wrapped label in this layout is not given
        # its second line's height.
        self._note = QLabel(
            "Keep using stemma meanwhile: the song opens here when it's ready."
        )
        self._note.setObjectName("subtle-label")
        self._note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._note, alignment=center)

    def start(self, title: str, artist: str) -> None:
        """Show *title* by *artist* at the start of its separation."""
        self._eta = SeparationEta()
        self._title.set_full_text(title)
        self._artist.set_full_text(artist)
        self._artist.setVisible(bool(artist))
        self._stage.setText("Starting…")
        self._bar.setValue(0)
        self._time_left.setText(f"0% · {format_eta(None)}")

    def update_progress(self, percent: int, now: float | None = None) -> None:
        """Show a worker progress report of *percent*."""
        now = time.monotonic() if now is None else now
        percent = max(0, min(100, int(percent)))
        self._bar.setValue(percent)
        self._stage.setText(stage_for(percent))
        left = format_eta(self._eta.update(percent, now))
        self._time_left.setText(f"{percent}% · {left}")

    def set_queued(self, count: int) -> None:
        """Say how many more imports wait behind this one."""
        self._queued.setVisible(count > 0)
        songs = "song waits" if count == 1 else "songs wait"
        self._queued.setText(f"{count} more {songs} to be separated")

    @property
    def title(self) -> str:
        return self._title.full_text()

    @property
    def stage(self) -> str:
        return self._stage.text()

    @property
    def time_left(self) -> str:
        return self._time_left.text()

    @property
    def percent(self) -> int:
        return self._bar.value()

    @property
    def queued_text(self) -> str:
        """The queue line, or "" while it is hidden."""
        return "" if self._queued.isHidden() else self._queued.text()

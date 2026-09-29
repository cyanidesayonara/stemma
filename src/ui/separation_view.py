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

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

# The workers report 0-15 % for loading, resampling, and model setup,
# then the separation itself; the estimate uses only the separation part,
# whose pace is steady.
_ESTIMATE_FROM_PERCENT = 15
_MIN_ELAPSED_S = 4.0
_MIN_PROGRESS = 2.0
_SMOOTHING = 0.3


class SeparationEta:
    """Estimate the seconds left from (time, percent) progress reports."""

    def __init__(self) -> None:
        self._start: tuple[float, float] | None = None
        self._estimate: float | None = None

    def update(self, percent: float, now: float) -> float | None:
        """Add a report; return the seconds left, or None while unsure."""
        if percent < _ESTIMATE_FROM_PERCENT:
            return None
        if self._start is None:
            self._start = (now, percent)
            return None
        t0, p0 = self._start
        elapsed = now - t0
        done = percent - p0
        if elapsed < _MIN_ELAPSED_S or done < _MIN_PROGRESS:
            return self._estimate
        remaining = (100.0 - percent) * elapsed / done
        if self._estimate is None:
            self._estimate = remaining
        else:
            # Blend towards the latest rate so the estimate doesn't jump.
            self._estimate += _SMOOTHING * (remaining - self._estimate)
        return max(0.0, self._estimate)


def format_eta(seconds: float | None) -> str:
    """Readable time left: "About 3 min left", "Almost done"."""
    if seconds is None:
        return "Estimating time left…"
    if seconds < 10:
        return "Almost done"
    if seconds < 60:
        return f"About {int(math.ceil(seconds / 10.0) * 10)} s left"
    return f"About {int(math.ceil(seconds / 60.0))} min left"


class SeparationView(QWidget):
    """Song, stage, progress, and time left for the running separation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._eta = SeparationEta()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(6)
        center = Qt.AlignmentFlag.AlignHCenter

        self._title = QLabel("")
        self._title.setObjectName("title-label")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title, alignment=center)

        self._artist = QLabel("")
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
        self._title.setText(title)
        self._artist.setText(artist)
        self._artist.setVisible(bool(artist))
        self._stage.setText("Starting…")
        self._bar.setValue(0)
        self._time_left.setText(f"0% · {format_eta(None)}")

    def update_progress(
        self, percent: int, message: str, now: float | None = None,
    ) -> None:
        """Show a worker progress report (*percent*, stage *message*)."""
        now = time.monotonic() if now is None else now
        percent = max(0, min(100, int(percent)))
        self._bar.setValue(percent)
        if message:
            self._stage.setText(message)
        left = format_eta(self._eta.update(percent, now))
        self._time_left.setText(f"{percent}% · {left}")

    def set_queued(self, count: int) -> None:
        """Say how many more imports wait behind this one."""
        self._queued.setVisible(count > 0)
        songs = "song waits" if count == 1 else "songs wait"
        self._queued.setText(f"{count} more {songs} to be separated")

    @property
    def title(self) -> str:
        return self._title.text()

    @property
    def stage(self) -> str:
        return self._stage.text()

    @property
    def time_left(self) -> str:
        return self._time_left.text()

    @property
    def percent(self) -> int:
        return self._bar.value()

"""Export several songs' stems to a folder in the background."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
import re

from PySide6.QtCore import QThread, Signal

from src.exporter import StemExporter
from src.import_messages import describe_error

_UNSAFE_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# Windows reserves these names whatever extension follows.
_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"{port}{n}" for port in ("com", "lpt") for n in range(1, 10)}
)
_MAX_NAME_LENGTH = 120


def safe_file_name(name: str) -> str:
    """Return *name* as a legal Windows file name without extension."""
    cleaned = _UNSAFE_CHARS_RE.sub("_", name)[:_MAX_NAME_LENGTH].strip(" .")
    if not cleaned:
        return "Untitled"
    if cleaned.split(".")[0].strip().lower() in _RESERVED_NAMES:
        return f"_{cleaned}"
    return cleaned


def free_path(folder: str, name: str, ext: str = "") -> str:
    """Return a path in *folder* that does not exist yet.

    Adds ``(2)``, ``(3)``... so a batch never overwrites earlier work.
    """
    path = os.path.join(folder, name + ext)
    number = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{name} ({number}){ext}")
        number += 1
    return path


@dataclass(frozen=True)
class BatchExportItem:
    """One song to export: its display name and stem files."""

    name: str
    stem_paths: dict[str, str]


class BatchExportWorker(QThread):
    """Exports songs one after another; one failure does not stop the rest.

    Signals:
        progress(int, str): songs finished so far, and the one starting.
        completed(int, list): songs exported, and failure messages.
    """

    progress = Signal(int, str)
    completed = Signal(int, list)

    def __init__(
        self,
        items: Sequence[BatchExportItem],
        folder: str,
        stems: Sequence[str],
        *,
        ext: str = ".wav",
        separate_files: bool = False,
        mp3_bitrate: int = 320,
    ) -> None:
        super().__init__()
        self._items = list(items)
        self._folder = folder
        self._stems = list(stems)
        self._ext = ext
        self._separate_files = separate_files
        self._mp3_bitrate = mp3_bitrate

    def run(self) -> None:
        exported = 0
        failures: list[str] = []
        for number, item in enumerate(self._items):
            if self.isInterruptionRequested():
                break
            self.progress.emit(number, item.name)
            try:
                if self._export(item):
                    exported += 1
            except Exception as exc:
                failures.append(
                    f"{item.name}: {describe_error(exc, 'Export failed')}"
                )
        self.completed.emit(exported, failures)

    def _export(self, item: BatchExportItem) -> bool:
        """Export one song; False when cancelled partway through."""
        stems = [
            s for s in self._stems
            if os.path.isfile(item.stem_paths.get(s, ""))
        ]
        if not stems:
            raise ValueError("The song has none of the chosen stems")
        exporter = StemExporter(item.stem_paths)
        name = safe_file_name(item.name)
        if not self._separate_files:
            exporter.export_mix(
                free_path(self._folder, name, self._ext),
                stem_names=stems,
                mp3_bitrate=self._mp3_bitrate,
            )
            return True
        folder = free_path(self._folder, name)
        for stem in stems:
            if self.isInterruptionRequested():
                return False
            # One stem at unity gain; export_mix takes the bitrate.
            exporter.export_mix(
                os.path.join(folder, stem + self._ext),
                stem_names=[stem],
                mp3_bitrate=self._mp3_bitrate,
            )
        return True

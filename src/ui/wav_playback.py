"""Play short WAV files for logo Easter eggs (non-blocking, can overlap).

Implementation is loaded on first use (``importlib``) so the splash screen can
use ``winsound`` before Qt Multimedia initialises.  Playback runs on the
caller's thread (must be the Qt GUI thread).
"""

from __future__ import annotations

import importlib
from pathlib import Path


def play_wav_async(path: str | Path) -> None:
    """Play *path* without blocking; multiple paths may overlap."""

    p = Path(path)
    if not p.is_file():
        return

    impl = importlib.import_module("src.ui._wav_playback_impl")
    impl.play_impl(p)

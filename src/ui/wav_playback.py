"""Play short WAV files for logo Easter eggs (non-blocking, can overlap).

Playback runs on the caller's thread (must be the Qt GUI thread).
"""

from __future__ import annotations

from pathlib import Path


def play_wav_async(path: str | Path) -> None:
    """Play *path* without blocking; multiple paths may overlap."""

    p = Path(path)
    if not p.is_file():
        return

    # Deferred to the first logo click, off the startup path the splash is
    # animating over. A plain import statement, not importlib with a module
    # name string: PyInstaller only bundles modules it sees imported, and
    # the string form left this one out of the 3.0.0 build, so every logo
    # click animated in silence.
    from src.ui import _wav_playback_impl

    _wav_playback_impl.play_impl(p)

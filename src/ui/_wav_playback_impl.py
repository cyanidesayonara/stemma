"""Logo SFX playback: winsound on Windows, Qt Multimedia elsewhere.

On Windows each sound is played with ``winsound`` from memory on a daemon
thread; it works in MSIX / Store builds, where ``QSoundEffect`` sometimes
played nothing, and it never opens a PortAudio stream (``sounddevice``
cannot open a second output stream beside the player on many Windows
setups).

Elsewhere a ``QSoundEffect`` per file (at most two: ``chord.wav`` and
``arpeggio.wav``) plays on the GUI thread. ``QSoundEffect`` may report
``Loading`` briefly after ``setSource``; status is polled until ``Ready``
or ``Error`` (avoids duplicate ``statusChanged`` connections on rapid
clicks).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtWidgets import QApplication

if TYPE_CHECKING:
    from PySide6.QtMultimedia import QSoundEffect

try:
    import winsound

    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False

# One QSoundEffect per resolved path (expected: chord.wav, arpeggio.wav).
_cache: dict[str, QSoundEffect] = {}

_POLL_INTERVAL_MS = 20
_POLL_MAX_ATTEMPTS = 40


def _play_winsound_fallback(path: Path) -> None:
    if not _HAS_WINSOUND:
        return
    try:
        # Use SND_MEMORY to avoid MSIX sandbox path-resolution failures.
        # Play in a daemon thread: Python disallows SND_ASYNC | SND_MEMORY.
        data = path.read_bytes()
        threading.Thread(
            target=winsound.PlaySound,
            args=(data, winsound.SND_MEMORY),
            daemon=True,
        ).start()
    except OSError:
        pass


def _sound_effect_class():
    """Return ``QSoundEffect``, or None when Qt Multimedia is unavailable.

    Imported only on the path that uses it. On Windows playback goes
    through winsound, so Qt Multimedia is never loaded there: importing it
    at module scope cost about 170 ms on the first logo click (the sound
    then lagged the animation), made the whole module unimportable in a
    bundle without the Qt Multimedia backend, and hangs on GitHub Actions
    Windows runners.
    """
    try:
        from PySide6.QtMultimedia import QSoundEffect
    except ImportError:  # pragma: no cover - depends on the bundle
        return None
    return QSoundEffect


def _try_play_effect(eff: QSoundEffect, path: Path) -> None:
    """Play when ``Ready``; on ``Error`` or timeout, use ``winsound``."""
    status = type(eff).Status

    def poll(attempt: int) -> None:
        st = eff.status()
        if st == status.Ready:
            eff.play()
        elif st == status.Error:
            _play_winsound_fallback(path)
        elif attempt < _POLL_MAX_ATTEMPTS:
            nxt = attempt + 1
            QTimer.singleShot(
                _POLL_INTERVAL_MS,
                lambda n=nxt: poll(n),
            )
        else:
            _play_winsound_fallback(path)

    st = eff.status()
    if st == status.Ready:
        eff.play()
    elif st == status.Error:
        _play_winsound_fallback(path)
    else:
        poll(0)


def play_impl(path: Path) -> None:
    path = path.resolve()
    if not path.is_file():
        return

    # Prefer winsound on Windows — it works reliably in MSIX / MS Store
    # builds where QSoundEffect sometimes silently fails to produce audio.
    if _HAS_WINSOUND:
        _play_winsound_fallback(path)
        return

    sound_effect = _sound_effect_class()
    if sound_effect is None:
        # No Qt multimedia in this build and no winsound (non-Windows):
        # nothing can play, but the caller's animation still runs.
        return

    app = QApplication.instance()
    if app is None:
        return

    key = str(path)
    if key not in _cache:
        eff = sound_effect(app)
        eff.setSource(QUrl.fromLocalFile(key))
        _cache[key] = eff
    _try_play_effect(_cache[key], path)

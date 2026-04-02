"""Logo SFX playback via Qt Multimedia (overlapping-friendly).

``sounddevice`` cannot open multiple simultaneous PortAudio output streams on
many Windows setups (MME error when mixing with the main player).  Each
``QSoundEffect`` instance can play independently on the main thread.

Loaded only on first logo click so the splash screen can use ``winsound``
first without initialising Qt Multimedia.

Caches at most two ``QSoundEffect`` instances (``chord.wav`` and
``arpeggio.wav``).  If Qt fails to load or play, falls back to
``winsound`` on Windows when available.

``QSoundEffect`` may report ``Loading`` briefly after ``setSource``; we poll
status on the GUI thread until ``Ready`` or ``Error`` (avoids duplicate
``statusChanged`` connections on rapid clicks).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtWidgets import QApplication

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
        winsound.PlaySound(
            str(path),
            winsound.SND_ASYNC | winsound.SND_FILENAME,
        )
    except OSError:
        pass


def _try_play_effect(eff: QSoundEffect, path: Path) -> None:
    """Play when ``Ready``; on ``Error`` or timeout, use ``winsound``."""

    def poll(attempt: int) -> None:
        st = eff.status()
        if st == QSoundEffect.Status.Ready:
            eff.play()
        elif st == QSoundEffect.Status.Error:
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
    if st == QSoundEffect.Status.Ready:
        eff.play()
    elif st == QSoundEffect.Status.Error:
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

    app = QApplication.instance()
    if app is None:
        return

    key = str(path)
    if key not in _cache:
        eff = QSoundEffect(app)
        eff.setSource(QUrl.fromLocalFile(key))
        _cache[key] = eff
    _try_play_effect(_cache[key], path)

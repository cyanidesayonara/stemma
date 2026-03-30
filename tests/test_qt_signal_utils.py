"""Tests for src.qt_signal_utils."""

from PySide6.QtCore import QObject, Signal

from src.qt_signal_utils import safe_disconnect


class _Emitter(QObject):
    sig = Signal()


def test_safe_disconnect_with_connection():
    o = _Emitter()
    o.sig.connect(lambda: None)
    safe_disconnect(o.sig)


def test_safe_disconnect_no_connections():
    o = _Emitter()
    safe_disconnect(o.sig)

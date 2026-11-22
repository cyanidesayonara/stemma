"""Tests for the Windows Show animations preference helper (#188)."""

from unittest.mock import MagicMock

import src.ui.motion as motion


def _fake_windll(ok: int, value: int, *, side_effect=None):
    def _spi(_action, _param, out_ptr, _winini):
        if side_effect is not None:
            raise side_effect
        out_ptr._obj.value = value
        return ok

    windll = MagicMock()
    windll.user32.SystemParametersInfoW.side_effect = _spi
    return windll


def test_win32_animations_on_returns_true(monkeypatch):
    monkeypatch.setattr(motion.sys, "platform", "win32")
    monkeypatch.setattr(motion.ctypes, "windll", _fake_windll(1, 1), raising=False)
    assert motion.animations_enabled() is True


def test_win32_animations_off_returns_false(monkeypatch):
    monkeypatch.setattr(motion.sys, "platform", "win32")
    monkeypatch.setattr(motion.ctypes, "windll", _fake_windll(1, 0), raising=False)
    assert motion.animations_enabled() is False


def test_win32_api_failure_defaults_true(monkeypatch):
    monkeypatch.setattr(motion.sys, "platform", "win32")
    monkeypatch.setattr(motion.ctypes, "windll", _fake_windll(0, 0), raising=False)
    assert motion.animations_enabled() is True


def test_win32_oserror_defaults_true(monkeypatch):
    monkeypatch.setattr(motion.sys, "platform", "win32")
    monkeypatch.setattr(
        motion.ctypes,
        "windll",
        _fake_windll(1, 0, side_effect=OSError("unavailable")),
        raising=False,
    )
    assert motion.animations_enabled() is True


def test_non_windows_defaults_true(monkeypatch):
    monkeypatch.setattr(motion.sys, "platform", "linux")
    assert motion.animations_enabled() is True
